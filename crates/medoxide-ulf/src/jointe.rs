//! Estimation jointe B0 + image (méthode C de fmi, `compare_methods.estimate_joint`) : démarrage à chaud
//! (`utils_cpr.prefit`) puis boucle jointe ; critères et résultats : docs/ULF.md (P4 bis, critère 4).
//!
//! Les pas sont exposés un par un ([`pas_prefit_champ`], [`pas_prefit_image`], [`pas_joint`]) pour être comparés à
//! PyTorch ; la boucle complète les enchaîne.
//!
//! **Adam.** L'Adam de Burn 0.22.1 garde β₁, β₂ et ε en `f32` (corrections de biais comprises) : 1 − 0,9f32 diffère de
//! 0,1 de 2,4e-7 en relatif, ce qui suffit à manquer 1e-10 en `f64`. [`Adam`] reprend donc `torch.optim.Adam`
//! (`_single_tensor_adam`) avec des scalaires `f64`, et prend un lr par appel, ce qui couvre les groupes de paramètres et
//! la décroissance cosinus.

use std::collections::HashMap;

use burn::module::{Param, ParamId};
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::{DType, Gradients};
use ndarray::Array3;

use crate::cpr::Reel;
use crate::inr::{moyenne, ChampB0, Encodage, ImageInr, InrHachage};
use crate::pertes::{depuis_tenseur, evaluer, vers_tenseur, Donnees, Termes};

/// Réglages de `compare_methods` (P1).
#[derive(Clone, Debug)]
pub struct Reglages {
    /// Itérations du démarrage à chaud, pour le champ puis pour l'image (`prefit`, 300).
    pub n_prefit: usize,
    /// lr du démarrage à chaud (1e-2, même lr pour tous les paramètres).
    pub lr_prefit: f64,
    /// Itérations de la boucle jointe (`N_ITER_JOINT`, 200).
    pub n_iter: usize,
    /// lr des réseaux (`LR`, 1e-3) et des deux scalaires du champ (`SCALAR_LR`, 1e-2).
    pub lr: f64,
    pub lr_scalaires: f64,
    /// lr final / lr initial de la décroissance cosinus (`LR_MIN_RATIO`, 1e-2).
    pub ratio_lr_min: f64,
    /// Poids des termes : fidélité, image, b0 (`W_JOINT`), lissage (`W_SMOOTH_JOINT`), support (`W_SUPPORT`),
    /// scalaires (`W_SCALAR_PENALTY`).
    pub poids: Poids,
}

#[derive(Clone, Copy, Debug)]
pub struct Poids {
    pub fidelite: f64,
    pub image: f64,
    pub b0: f64,
    pub lissage: f64,
    pub support: f64,
    pub scalaires: f64,
}

impl Default for Reglages {
    fn default() -> Self {
        Self {
            n_prefit: 300,
            lr_prefit: 1e-2,
            n_iter: 200,
            lr: 1e-3,
            lr_scalaires: 1e-2,
            ratio_lr_min: 1e-2,
            poids: Poids {
                fidelite: 1.0,
                image: 1e-2,
                b0: 1e-2,
                lissage: 1e-2 * 1e-2,
                support: 1.0,
                scalaires: 1e-5,
            },
        }
    }
}

impl Reglages {
    /// Facteur du lr à l'itération `it` de la boucle jointe (`compare_methods.cosine_schedule`) : de 1 à
    /// `ratio_lr_min`.
    pub fn facteur_lr(&self, it: usize) -> f64 {
        let r = self.ratio_lr_min;
        r + (1.0 - r) * 0.5 * (1.0 + (std::f64::consts::PI * it as f64 / self.n_iter as f64).cos())
    }
}

/// Moments d'un paramètre (aplatis) et nombre de pas faits.
struct Etat {
    m: Tensor<1>,
    v: Tensor<1>,
    pas: i32,
}

/// Adam de `torch.optim.Adam` (β = (0,9 ; 0,999), sans décroissance des poids), scalaires en `f64`.
pub struct Adam {
    beta1: f64,
    beta2: f64,
    eps: f64,
    etats: HashMap<ParamId, Etat>,
}

impl Default for Adam {
    fn default() -> Self {
        Self::nouveau(1e-8)
    }
}

impl Adam {
    pub fn nouveau(eps: f64) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps,
            etats: HashMap::new(),
        }
    }

    /// Un pas sur un paramètre, de lr `lr` ; un paramètre sans gradient est laissé tel quel (comme PyTorch).
    pub fn pas<const D: usize>(&mut self, p: Param<Tensor<D>>, grads: &Gradients, lr: f64) -> Param<Tensor<D>> {
        let Some(g) = p.val().grad(grads) else {
            return p;
        };
        let id = p.id;
        let (beta1, beta2, eps) = (self.beta1, self.beta2, self.eps);
        let forme = g.dims();
        let n: usize = forme.iter().product();
        let g = g.reshape([n]);
        let etat = self.etats.entry(id).or_insert_with(|| Etat {
            m: g.zeros_like(),
            v: g.zeros_like(),
            pas: 0,
        });
        etat.pas += 1;
        // exp_avg.lerp_(grad, 1 − β₁) ; exp_avg_sq.mul_(β₂).addcmul_(grad, grad, value = 1 − β₂)
        etat.m = etat.m.clone() + (g.clone() - etat.m.clone()) * (1.0 - beta1);
        etat.v = etat.v.clone() * beta2 + g.clone() * (1.0 - beta2) * g;
        let bc1 = 1.0 - beta1.powi(etat.pas);
        let bc2 = 1.0 - beta2.powi(etat.pas);
        let denom = (etat.v.clone().sqrt() / bc2.sqrt()) + eps;
        let delta = (etat.m.clone() / denom * (lr / bc1)).reshape(forme);
        p.map(|t| Tensor::from_inner(t.inner() - delta).require_grad())
    }

    fn pas_lineaire(&mut self, c: Linear, grads: &Gradients, lr: f64) -> Linear {
        Linear {
            weight: self.pas(c.weight, grads, lr),
            bias: c.bias.map(|b| self.pas(b, grads, lr)),
        }
    }

    /// Un pas sur un réseau : tables de lr `lr_tables`, couches de lr `lr_couches`.
    pub fn pas_reseau(&mut self, r: InrHachage, grads: &Gradients, lr_tables: f64, lr_couches: f64) -> InrHachage {
        InrHachage {
            tables: r.tables.into_iter().map(|t| self.pas(t, grads, lr_tables)).collect(),
            couches: r
                .couches
                .into_iter()
                .map(|c| self.pas_lineaire(c, grads, lr_couches))
                .collect(),
        }
    }

    /// Un pas sur le champ : réseau de lr `lr`, scalaires de lr `lr_scalaires`.
    pub fn pas_champ(&mut self, c: ChampB0, grads: &Gradients, lr: f64, lr_scalaires: f64) -> ChampB0 {
        ChampB0 {
            reseau: self.pas_reseau(c.reseau, grads, lr, lr),
            s: self.pas(c.s, grads, lr_scalaires),
            o: self.pas(c.o, grads, lr_scalaires),
            b0_min_hz: c.b0_min_hz,
            b0_max_hz: c.b0_max_hz,
        }
    }
}

/// Moyenne quadratique d'une cible (`target.abs().square().mean().sqrt().clamp(min=1e-12)`).
fn rms<'a>(valeurs: impl Iterator<Item = &'a f64>) -> f64 {
    let (s, n) = valeurs.fold((0.0, 0usize), |(s, n), v| (s + v * v, n + 1));
    (s / n as f64).sqrt().max(1e-12)
}

/// Cible du démarrage à chaud du champ : b0_init aplati.
pub struct CibleChamp {
    valeurs: Tensor<1>,
    inv_echelle: f64,
}

impl CibleChamp {
    pub fn nouvelle(b0_init: &Array3<f64>, device: &Device, dtype: DType) -> Self {
        let n = b0_init.len();
        Self {
            valeurs: vers_tenseur(b0_init, device, dtype).reshape([n]),
            inv_echelle: 1.0 / rms(b0_init.iter()),
        }
    }
}

/// Un pas du démarrage à chaud du champ, perte moyenne(((b0 − cible) / rms(cible))²) ; rend le champ et la perte
/// (évaluée avant le pas).
pub fn pas_prefit_champ(
    champ: ChampB0,
    encodage: &Encodage,
    cible: &CibleChamp,
    adam: &mut Adam,
    lr: f64,
) -> (ChampB0, f64) {
    let e = (champ.b0(encodage) - cible.valeurs.clone()) * cible.inv_echelle;
    let perte = moyenne(e.powi_scalar(2));
    let grads = perte.backward();
    let valeur = scalaire(perte);
    (adam.pas_champ(champ, &grads, lr, lr), valeur)
}

/// Cible du démarrage à chaud de l'image (`averaged_corrected`, phase nulle) : moyenne des magnitudes de reg corrigée
/// avec +b0_init et de inv corrigée avec −b0_init, à l'échelle des données normalisées.
pub struct CibleImage {
    pub magnitude: Array3<f64>,
    valeurs: Tensor<1>,
    inv_echelle: f64,
}

impl CibleImage {
    pub fn nouvelle<T: Reel>(d: &Donnees<T>, b0_init: &Array3<f64>, device: &Device, dtype: DType) -> Self {
        let b0 = b0_init.mapv(|v| T::from_f64(v).unwrap());
        let moins_b0 = b0.mapv(|v| -v);
        let c_reg = d.modele.adjoint(&d.k_reg, &b0);
        let c_inv = d.modele.adjoint(&d.k_inv, &moins_b0);
        let magnitude = ndarray::Zip::from(&c_reg)
            .and(&c_inv)
            .map_collect(|a, b| (a.norm().to_f64().unwrap() + b.norm().to_f64().unwrap()) / 2.0);
        let n = magnitude.len();
        Self {
            valeurs: vers_tenseur(&magnitude, device, dtype).reshape([n]),
            inv_echelle: 1.0 / rms(magnitude.iter()),
            magnitude,
        }
    }
}

/// Un pas du démarrage à chaud de l'image, perte moyenne((|u − cible| / rms(cible))²) ; rend l'image et la perte.
pub fn pas_prefit_image(
    image: ImageInr,
    encodage: &Encodage,
    cible: &CibleImage,
    adam: &mut Adam,
    lr: f64,
) -> (ImageInr, f64) {
    let (re, im) = image.image(encodage);
    let (er, ei) = ((re - cible.valeurs.clone()) * cible.inv_echelle, im * cible.inv_echelle);
    let perte = moyenne(er.powi_scalar(2) + ei.powi_scalar(2));
    let grads = perte.backward();
    let valeur = scalaire(perte);
    (
        ImageInr {
            reseau: adam.pas_reseau(image.reseau, &grads, lr, lr),
        },
        valeur,
    )
}

fn scalaire(t: Tensor<1>) -> f64 {
    t.try_into_data_dtype(DType::F64).unwrap().try_to_vec::<f64>().unwrap()[0]
}

/// Valeurs non pondérées des termes et perte totale pondérée, évaluées avant le pas.
#[derive(Clone, Debug)]
pub struct Bilan {
    pub fidelite: f64,
    pub image: f64,
    pub b0: f64,
    pub lissage: f64,
    pub support: f64,
    pub scalaires: f64,
    pub totale: f64,
    /// Durée (s) du modèle CPR et de son gradient (sur le dispositif depuis P4 ter), attente de la file du dispositif
    /// comprise.
    pub secondes_operateur: f64,
}

/// Un pas de la boucle jointe, de lr `facteur` × lr de base.
///
/// Deux `backward` (coupure du graphe, docs/ULF.md) : le premier part de la perte, avec u et b0 détachés des réseaux et
/// placés dans des feuilles ; [`crate::pertes::Coupure::gradients`] y ajoute les contributions du modèle CPR ; le
/// second part de Σ (u, b0) · gradient (détaché) + pénalité des scalaires et descend dans les réseaux.
#[allow(clippy::too_many_arguments)]
pub fn pas_joint<T: Reel>(
    d: &Donnees<T>,
    champ: ChampB0,
    image: ImageInr,
    enc_champ: &Encodage,
    enc_image: &Encodage,
    adam: &mut Adam,
    reglages: &Reglages,
    facteur: f64,
) -> (ChampB0, ImageInr, Bilan) {
    let dims = d.dims;
    let (re, im) = image.image(enc_image);
    let b0 = champ.b0(enc_champ);
    let feuille = |x: &Tensor<1>| x.clone().detach().reshape(dims).require_grad();
    let (f_re, f_im, f_b0) = (feuille(&re), feuille(&im), feuille(&b0));
    let debut = std::time::Instant::now();
    let (termes, coupure) = evaluer(d, &f_re, &f_im, &f_b0);
    let mut secondes_operateur = debut.elapsed().as_secs_f64();
    let w = reglages.poids;
    let Termes {
        fidelite,
        image: t_image,
        b0: t_b0,
        lissage,
        support,
    } = termes;
    let perte = fidelite.clone() * w.fidelite
        + t_image.clone() * w.image
        + t_b0.clone() * w.b0
        + lissage.clone() * w.lissage
        + support.clone() * w.support;
    let grads = perte.backward();
    let debut = std::time::Instant::now();
    let (gu_op, gb0_op) = coupure.gradients(d, &grads);
    secondes_operateur += debut.elapsed().as_secs_f64();
    let direct = |x: &Tensor<3>| x.grad(&grads).map(Tensor::from_inner);
    let plus = |a: Option<Tensor<3>>, b: Tensor<3>| a.map_or_else(|| b.clone(), |a| a + b.clone());
    let n = re.dims()[0];
    let g_re = plus(direct(&f_re), gu_op.re).reshape([n]).detach();
    let g_im = plus(direct(&f_im), gu_op.im).reshape([n]).detach();
    let g_b0 = plus(direct(&f_b0), gb0_op).reshape([n]).detach();
    let penalite = champ.penalite_scalaires();
    let second = (re * g_re).sum() + (im * g_im).sum() + (b0 * g_b0).sum() + penalite.clone() * w.scalaires;
    let grads2 = second.backward();
    let bilan = Bilan {
        fidelite: scalaire(fidelite),
        image: scalaire(t_image),
        b0: scalaire(t_b0),
        lissage: scalaire(lissage),
        support: scalaire(support),
        scalaires: scalaire(penalite.clone()),
        totale: scalaire(perte) + w.scalaires * scalaire(penalite),
        secondes_operateur,
    };
    let (lr, lr_s) = (reglages.lr * facteur, reglages.lr_scalaires * facteur);
    let image = ImageInr {
        reseau: adam.pas_reseau(image.reseau, &grads2, lr, lr),
    };
    let champ = adam.pas_champ(champ, &grads2, lr, lr_s);
    (champ, image, bilan)
}

/// Résultat de [`estimer`] : champ (Hz) et image (à l'échelle des données brutes), sur la grille.
pub struct Estimation {
    pub b0: Array3<f64>,
    pub u_re: Array3<f64>,
    pub u_im: Array3<f64>,
    pub champ: ChampB0,
    pub image: ImageInr,
    pub durees: Durees,
}

/// Durées (s) des phases de [`estimer`].
#[derive(Clone, Debug, Default)]
pub struct Durees {
    pub demarrage_champ: f64,
    pub demarrage_image: f64,
    pub jointe: f64,
    /// Part de `jointe` passée dans le modèle CPR (voir [`Bilan::secondes_operateur`]).
    pub operateur: f64,
}

/// Méthode C complète : démarrage à chaud du champ sur b0_init, de l'image sur la moyenne des magnitudes corrigées,
/// puis `n_iter` pas joints à lr décroissant. `suivi` reçoit (itération, bilan) à chaque pas joint.
#[allow(clippy::too_many_arguments)]
pub fn estimer<T: Reel>(
    d: &Donnees<T>,
    champ: ChampB0,
    image: ImageInr,
    enc_champ: &Encodage,
    enc_image: &Encodage,
    b0_init: &Array3<f64>,
    reglages: &Reglages,
    mut suivi: impl FnMut(usize, &Bilan),
) -> Estimation {
    let t = champ.reseau.tables[0].val();
    let (device, dtype) = (t.device(), t.dtype());
    let mut durees = Durees::default();
    let debut = std::time::Instant::now();
    let cible = CibleChamp::nouvelle(b0_init, &device, dtype);
    let mut champ = champ;
    let mut adam = Adam::default();
    for _ in 0..reglages.n_prefit {
        champ = pas_prefit_champ(champ, enc_champ, &cible, &mut adam, reglages.lr_prefit).0;
    }
    let _ = scalaire(champ.s.val());
    durees.demarrage_champ = debut.elapsed().as_secs_f64();
    let debut = std::time::Instant::now();
    let cible = CibleImage::nouvelle(d, b0_init, &device, dtype);
    let mut image = image;
    let mut adam = Adam::default();
    for _ in 0..reglages.n_prefit {
        image = pas_prefit_image(image, enc_image, &cible, &mut adam, reglages.lr_prefit).0;
    }
    let _ = scalaire(image.reseau.couches[0].bias.as_ref().unwrap().val().slice(0..1));
    durees.demarrage_image = debut.elapsed().as_secs_f64();
    let debut = std::time::Instant::now();
    let mut adam = Adam::default();
    for it in 0..reglages.n_iter {
        let (c, i, bilan) = pas_joint(
            d,
            champ,
            image,
            enc_champ,
            enc_image,
            &mut adam,
            reglages,
            reglages.facteur_lr(it),
        );
        (champ, image) = (c, i);
        durees.operateur += bilan.secondes_operateur;
        suivi(it, &bilan);
    }
    durees.jointe = debut.elapsed().as_secs_f64();
    let dims = d.dims;
    let b0 = depuis_tenseur::<f64>(champ.b0(enc_champ).detach().reshape(dims));
    let (re, im) = image.image(enc_image);
    let vers = |x: Tensor<1>| depuis_tenseur::<f64>(x.detach().reshape(dims)).mapv(|v| v * d.echelle);
    Estimation {
        b0,
        u_re: vers(re),
        u_im: vers(im),
        champ,
        image,
        durees,
    }
}
