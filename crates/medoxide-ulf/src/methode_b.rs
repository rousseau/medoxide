//! Méthode B de fmi (`compare_methods.estimate_inr`) : champ INR seul, ajusté pour que reg corrigée avec +b0 et inv
//! corrigée avec −b0 se ressemblent (LNCC), avec une pénalité de flexion ; il fournit le `b0_init` de l'estimation
//! jointe. Critères et résultats : docs/ULF.md (P5).
//!
//! Un pas se fait comme celui de la jointe ([`crate::jointe::pas_joint`]) : coupure du graphe autour du modèle CPR, deux
//! `backward`, [`Adam`] de `torch.optim.Adam` à deux lr (réseau ; scalaires s et o), décroissance cosinus.

use burn::prelude::*;
use burn::tensor::DType;
use ndarray::Array3;

use crate::cpr::Reel;
use crate::inr::{ChampB0, Encodage};
use crate::jointe::Adam;
use crate::pertes::{depuis_tenseur, evaluer_b0, Donnees};

/// Réglages de `compare_methods` pour la méthode B.
#[derive(Clone, Debug)]
pub struct ReglagesB {
    /// Itérations (`N_ITER_INR`, 400).
    pub n_iter: usize,
    /// lr du réseau (`LR`, 1e-3) et des scalaires (`SCALAR_LR`, 1e-2).
    pub lr: f64,
    pub lr_scalaires: f64,
    /// lr final / lr initial de la décroissance cosinus (`LR_MIN_RATIO`, 1e-2).
    pub ratio_lr_min: f64,
    /// Poids du lissage (`W_SMOOTH`, 1e-2) et de la pénalité des scalaires (`W_SCALAR_PENALTY`, 1e-5) ; le terme b0 a
    /// le poids 1.
    pub poids_lissage: f64,
    pub poids_scalaires: f64,
}

impl Default for ReglagesB {
    fn default() -> Self {
        Self {
            n_iter: 400,
            lr: 1e-3,
            lr_scalaires: 1e-2,
            ratio_lr_min: 1e-2,
            poids_lissage: 1e-2,
            poids_scalaires: 1e-5,
        }
    }
}

impl ReglagesB {
    /// Facteur du lr à l'itération `it` (`compare_methods.cosine_schedule` sur `n_iter`).
    pub fn facteur_lr(&self, it: usize) -> f64 {
        let r = self.ratio_lr_min;
        r + (1.0 - r) * 0.5 * (1.0 + (std::f64::consts::PI * it as f64 / self.n_iter as f64).cos())
    }
}

/// Termes non pondérés et perte totale pondérée, évalués avant le pas.
#[derive(Clone, Debug)]
pub struct BilanB {
    pub b0: f64,
    pub lissage: f64,
    pub scalaires: f64,
    pub totale: f64,
}

fn scalaire(t: Tensor<1>) -> f64 {
    t.try_into_data_dtype(DType::F64).unwrap().try_to_vec::<f64>().unwrap()[0]
}

/// Perte de la méthode B en b0 (tenseur de forme `dims`, sans les scalaires) et son gradient par rapport à b0 :
/// (terme b0, lissage, gradient). Sert au critère 2 et à [`pas_b`].
pub fn perte_et_gradient<T: Reel>(d: &Donnees<T>, b0: &Tensor<3>, reglages: &ReglagesB) -> (f64, f64, Tensor<3>) {
    let feuille = b0.clone().detach().require_grad();
    let (terme_b0, lissage, coupure) = evaluer_b0(d, &feuille);
    let perte = terme_b0.clone() + lissage.clone() * reglages.poids_lissage;
    let grads = perte.backward();
    let g_op = coupure.gradient(d, &grads);
    let g = feuille
        .grad(&grads)
        .map_or_else(|| g_op.clone(), |g| Tensor::from_inner(g) + g_op.clone());
    (scalaire(terme_b0), scalaire(lissage), g.detach())
}

/// Un pas de la méthode B, de lr `facteur` × lr de base.
pub fn pas_b<T: Reel>(
    d: &Donnees<T>,
    champ: ChampB0,
    encodage: &Encodage,
    adam: &mut Adam,
    reglages: &ReglagesB,
    facteur: f64,
) -> (ChampB0, BilanB) {
    let b0 = champ.b0(encodage);
    let n = b0.dims()[0];
    let (t_b0, lissage, g) = perte_et_gradient(d, &b0.clone().reshape(d.dims), reglages);
    let penalite = champ.penalite_scalaires();
    let second = (b0 * g.reshape([n])).sum() + penalite.clone() * reglages.poids_scalaires;
    let grads = second.backward();
    let scalaires = scalaire(penalite);
    let bilan = BilanB {
        b0: t_b0,
        lissage,
        scalaires,
        totale: t_b0 + reglages.poids_lissage * lissage + reglages.poids_scalaires * scalaires,
    };
    let champ = adam.pas_champ(champ, &grads, reglages.lr * facteur, reglages.lr_scalaires * facteur);
    (champ, bilan)
}

/// Méthode B complète : `n_iter` pas à lr décroissant ; rend le champ final (Hz, sur la grille) et le champ appris.
/// `suivi` reçoit (itération, bilan) à chaque pas.
pub fn estimer_b<T: Reel>(
    d: &Donnees<T>,
    champ: ChampB0,
    encodage: &Encodage,
    reglages: &ReglagesB,
    mut suivi: impl FnMut(usize, &BilanB),
) -> (Array3<f64>, ChampB0) {
    let mut champ = champ;
    let mut adam = Adam::default();
    for it in 0..reglages.n_iter {
        let (c, bilan) = pas_b(d, champ, encodage, &mut adam, reglages, reglages.facteur_lr(it));
        champ = c;
        suivi(it, &bilan);
    }
    let b0 = depuis_tenseur::<f64>(champ.b0(encodage).detach().reshape(d.dims));
    (b0, champ)
}
