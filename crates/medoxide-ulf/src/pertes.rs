//! Pertes de l'estimation jointe B0 + image (portage de `utils_cpr.joint_losses` avec les réglages de
//! `compare_methods.estimate_joint`, de `bending_penalty` et de la pénalité hors support), et coupure du graphe autour du
//! modèle CPR ; critères et résultats : docs/ULF.md (P4 bis, critère 3).
//!
//! **Coupure du graphe.** Le modèle CPR tourne sur CPU ([`ModeleCpr`], `rustfft`), hors de l'autodiff de Burn. [`evaluer`]
//! lit u et b0, calcule les images du modèle et les place dans des feuilles de l'autodiff ; les pertes partent de ces
//! feuilles et de u, b0. Après `backward`, [`Coupure::gradients`] ramène les gradients des feuilles sur u et b0 par les
//! gradients écrits du modèle ([`ModeleCpr::gradient_direct`], [`ModeleCpr::gradient_adjoint_b0`]) ; ils s'ajoutent aux
//! gradients que Burn a calculés directement pour u et b0.
//!
//! Termes, non pondérés (u à l'échelle des données normalisées, k divisé par `echelle`) :
//! - `fidelite` = moyenne((|F⁻¹k_reg| − |F⁻¹A(+b0)u|)²) + même terme pour inv avec −b0 ;
//! - `image` = −[LNCC(|u|, |Aᴴ(+b0)k_reg|) + LNCC(|u|, |Aᴴ(−b0)k_inv|)], images corrigées détachées ;
//! - `b0` = −LNCC(|Aᴴ(+b0)k_reg|, |Aᴴ(−b0)k_inv|) ;
//! - `lissage` = Σ_axes moyenne((∂²b0 / h²)²) (différences finies d'ordre 2) ;
//! - `support` = moyenne(|u|²) hors du masque de support.

use burn::prelude::*;
use burn::tensor::{DType, Gradients, TensorData};
use ndarray::{Array3, Zip};
use rustfft::num_complex::{Complex, Complex64};

use crate::cpr::{ModeleCpr, Reel};
use crate::inr::{diffuser, moyenne};

/// Fenêtre gaussienne de la LNCC (mm), `compare_methods.LNCC_SIGMA_MM`.
pub const SIGMA_LNCC_MM: f64 = 4.0;

/// Noyau gaussien normalisé de `utils_cpr.gaussian_blur_3d` : rayon max(1, arrondi(3σ)) (arrondi au pair le plus proche,
/// comme `round` en Python).
pub fn noyau_gaussien(sigma: f64) -> Vec<f64> {
    let rayon = ((3.0 * sigma).round_ties_even() as i64).max(1);
    let k: Vec<f64> = (-rayon..=rayon)
        .map(|t| (-0.5 * (t as f64 / sigma).powi(2)).exp())
        .collect();
    let somme: f64 = k.iter().sum();
    k.into_iter().map(|v| v / somme).collect()
}

/// Matrice (n, n) du flou 1D à bords nuls (corrélation par le noyau, symétrique) : y = M x le long d'un axe.
fn matrice_flou(n: usize, noyau: &[f64]) -> Vec<f64> {
    let r = (noyau.len() / 2) as i64;
    let mut m = vec![0.0; n * n];
    for i in 0..n as i64 {
        for j in (i - r).max(0)..=(i + r).min(n as i64 - 1) {
            m[i as usize * n + j as usize] = noyau[(j - i + r) as usize];
        }
    }
    m
}

/// Flou gaussien séparable à bords nuls sur l'hôte (`gaussian_blur_3d`), σ en voxels par axe (0 : pas de flou).
pub fn flou_hote(x: &Array3<f64>, sigma: [f64; 3]) -> Array3<f64> {
    let mut y = x.clone();
    for (axe, &s) in sigma.iter().enumerate() {
        if s <= 0.0 {
            continue;
        }
        let noyau = noyau_gaussien(s);
        let r = (noyau.len() / 2) as i64;
        let n = y.shape()[axe] as i64;
        let entree = y.clone();
        for ((i0, i1, i2), v) in y.indexed_iter_mut() {
            let i = [i0, i1, i2];
            let mut s = 0.0;
            for (k, w) in noyau.iter().enumerate() {
                let j = i[axe] as i64 + k as i64 - r;
                if (0..n).contains(&j) {
                    let mut p = i;
                    p[axe] = j as usize;
                    s += w * entree[p];
                }
            }
            *v = s;
        }
    }
    y
}

/// Masque du support de l'objet (`utils_cpr.object_support_mask`) : maximum des magnitudes non corrigées de reg et inv,
/// flou de 1,5 voxel, seuil à 0,1 · maximum, dilaté de (6, 3, 3) voxels (boîte, comme `max_pool3d`). Calculé en `f64`.
pub fn masque_support(img_reg: &Array3<f64>, img_inv: &Array3<f64>) -> Array3<bool> {
    let img = Zip::from(img_reg).and(img_inv).map_collect(|a, b| a.max(*b));
    let img = flou_hote(&img, [1.5; 3]);
    let max = img.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut masque = img.mapv(|v| v > 0.1 * max);
    for (axe, r) in [6i64, 3, 3].into_iter().enumerate() {
        let n = masque.shape()[axe] as i64;
        let entree = masque.clone();
        for ((i0, i1, i2), v) in masque.indexed_iter_mut() {
            let i = [i0, i1, i2];
            *v = (-r..=r).any(|d| {
                let j = i[axe] as i64 + d;
                (0..n).contains(&j) && {
                    let mut p = i;
                    p[axe] = j as usize;
                    entree[p]
                }
            });
        }
    }
    masque
}

fn dtype_de<T: Reel>() -> DType {
    if std::mem::size_of::<T>() == 8 {
        DType::F64
    } else {
        DType::F32
    }
}

/// Tableau réel de l'hôte → tenseur (forme identique) dans le type `dtype`.
pub fn vers_tenseur<T: Reel>(a: &Array3<T>, device: &Device, dtype: DType) -> Tensor<3> {
    let (n0, n1, n2) = a.dim();
    let v: Vec<f64> = a.iter().map(|x| x.to_f64().unwrap()).collect();
    Tensor::from_data(TensorData::new(v, [n0, n1, n2]), (device, dtype))
}

/// Tenseur → tableau réel de l'hôte.
pub fn depuis_tenseur<T: Reel>(t: Tensor<3>) -> Array3<T> {
    let [n0, n1, n2] = t.dims();
    let v = t.try_into_data_dtype(DType::F64).unwrap().try_to_vec::<f64>().unwrap();
    Array3::from_shape_vec((n0, n1, n2), v.into_iter().map(|x| T::from_f64(x).unwrap()).collect()).unwrap()
}

fn complexe<T: Reel>(re: Array3<T>, im: Array3<T>) -> Array3<Complex<T>> {
    Zip::from(&re).and(&im).map_collect(|r, i| Complex::new(*r, *i))
}

fn module(re: Tensor<3>, im: Tensor<3>) -> Tensor<3> {
    (re.clone() * re + im.clone() * im).sqrt()
}

fn aplatir(x: Tensor<3>) -> Tensor<1> {
    let n: usize = x.dims().iter().product();
    x.reshape([n])
}

/// Données d'une paire reg / inv, préparées une fois : modèle, k-space normalisés, magnitudes observées, masque.
pub struct Donnees<T: Reel> {
    pub modele: ModeleCpr<T>,
    pub dims: [usize; 3],
    /// Facteur de normalisation (`data_scale`) : maximum des |F⁻¹ k| de reg et inv.
    pub echelle: f64,
    pub k_reg: Array3<Complex<T>>,
    pub k_inv: Array3<Complex<T>>,
    pub espacement_mm: [f64; 3],
    pub masque: Array3<bool>,
    device: Device,
    dtype: DType,
    obs_reg: Tensor<3>,
    obs_inv: Tensor<3>,
    hors_support: Tensor<3>,
    n_hors_support: usize,
    flou: [Tensor<2>; 3],
}

impl<T: Reel> Donnees<T> {
    /// Prépare la paire (k-space bruts en complex128, temps des échantillons de lecture, espacement en mm).
    pub fn nouveau(
        k_reg: &Array3<Complex64>,
        k_inv: &Array3<Complex64>,
        temps: &[f64],
        espacement_mm: [f64; 3],
        device: &Device,
    ) -> Self {
        let dims: [usize; 3] = k_reg.dim().into();
        let dtype = dtype_de::<T>();
        let modele64 = ModeleCpr::<f64>::new(dims, temps);
        let zero64 = Array3::<f64>::zeros(dims);
        let (img_reg, img_inv) = (
            modele64.adjoint(k_reg, &zero64).mapv(|z| z.norm()),
            modele64.adjoint(k_inv, &zero64).mapv(|z| z.norm()),
        );
        let max = |a: &Array3<f64>| a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let echelle = max(&img_reg).max(max(&img_inv));
        let masque = masque_support(&img_reg, &img_inv);
        let temps_t: Vec<T> = temps.iter().map(|&t| T::from_f64(t).unwrap()).collect();
        let modele = ModeleCpr::<T>::new(dims, &temps_t);
        let norm = |k: &Array3<Complex64>| {
            k.mapv(|z| {
                Complex::new(
                    T::from_f64(z.re / echelle).unwrap(),
                    T::from_f64(z.im / echelle).unwrap(),
                )
            })
        };
        let (k_reg, k_inv) = (norm(k_reg), norm(k_inv));
        let zero = Array3::<T>::zeros(dims);
        let obs = |k: &Array3<Complex<T>>| vers_tenseur(&modele.adjoint(k, &zero).mapv(|z| z.norm()), device, dtype);
        let (obs_reg, obs_inv) = (obs(&k_reg), obs(&k_inv));
        let hors = masque.mapv(|m| if m { T::zero() } else { T::one() });
        let n_hors_support = masque.iter().filter(|m| !**m).count();
        let sigma = espacement_mm.map(|h| SIGMA_LNCC_MM / h);
        let flou = std::array::from_fn(|a| {
            let n = dims[a];
            Tensor::from_data(
                TensorData::new(matrice_flou(n, &noyau_gaussien(sigma[a])), [n, n]),
                (device, dtype),
            )
        });
        Self {
            modele,
            dims,
            echelle,
            k_reg,
            k_inv,
            espacement_mm,
            masque,
            device: device.clone(),
            dtype,
            obs_reg,
            obs_inv,
            hors_support: vers_tenseur(&hors, device, dtype),
            n_hors_support,
            flou,
        }
    }

    /// Flou gaussien de la LNCC (σ = 4 mm par axe), séparable, bords nuls, par produits matriciels.
    fn flou(&self, x: Tensor<3>) -> Tensor<3> {
        let mut x = x;
        for (axe, m) in self.flou.iter().enumerate() {
            let x2 = x.swap_dims(axe, 2);
            let [p, q, r] = x2.dims();
            x = x2
                .reshape([p * q, r])
                .matmul(m.clone())
                .reshape([p, q, r])
                .swap_dims(axe, 2);
        }
        x
    }

    /// Variance non biaisée (comme `torch.var`) de tout le volume, forme (1).
    fn variance(&self, x: Tensor<3>) -> Tensor<1> {
        let x = aplatir(x);
        let n = x.dims()[0];
        let centre = x.clone() - diffuser(moyenne(x), n);
        moyenne(centre.powi_scalar(2)) * (n as f64 / (n - 1) as f64)
    }

    /// NCC locale moyenne (`local_normalized_cross_correlation`, eps = 1e-6), forme (1).
    pub fn lncc(&self, a: Tensor<3>, b: Tensor<3>) -> Tensor<1> {
        let (mu1, mu2) = (self.flou(a.clone()), self.flou(b.clone()));
        let var1 = self.flou(a.clone() * a.clone()) - mu1.clone() * mu1.clone();
        let var2 = self.flou(b.clone() * b.clone()) - mu2.clone() * mu2.clone();
        let cov = self.flou(a.clone() * b.clone()) - mu1 * mu2;
        let plancher2 = (self.variance(a) * self.variance(b)) * 1e-12; // (eps · sqrt(var a · var b))²
        let n: usize = self.dims.iter().product();
        let den = aplatir(var1.clamp_min(0.0) * var2.clamp_min(0.0)) + diffuser(plancher2, n);
        moyenne(aplatir(cov) / den.sqrt())
    }
}

/// Les cinq termes non pondérés, chacun de forme (1).
pub struct Termes {
    pub fidelite: Tensor<1>,
    pub image: Tensor<1>,
    pub b0: Tensor<1>,
    pub lissage: Tensor<1>,
    pub support: Tensor<1>,
}

/// Feuilles de l'autodiff placées à la sortie du modèle CPR, et ce qu'il faut pour en ramener les gradients.
pub struct Coupure<T: Reel> {
    u: Array3<Complex<T>>,
    b0: Array3<T>,
    /// F⁻¹A(+b0)u, F⁻¹A(−b0)u, Aᴴ(+b0)k_reg, Aᴴ(−b0)k_inv : (réel, imaginaire).
    feuilles: [(Tensor<3>, Tensor<3>); 4],
    atk_reg: Array3<Complex<T>>,
    atk_inv: Array3<Complex<T>>,
}

/// Évalue les termes en (u, b0) (tenseurs de forme `dims`, dans le type de `donnees`) et la coupure du graphe.
pub fn evaluer<T: Reel>(d: &Donnees<T>, u_re: &Tensor<3>, u_im: &Tensor<3>, b0: &Tensor<3>) -> (Termes, Coupure<T>) {
    let u = complexe(depuis_tenseur::<T>(u_re.clone()), depuis_tenseur::<T>(u_im.clone()));
    let b0h = depuis_tenseur::<T>(b0.clone());
    let moins_b0 = b0h.mapv(|v| -v);
    let zero = Array3::<T>::zeros(d.dims);
    let m = &d.modele;
    let pred_reg = m.adjoint(&m.direct(&u, &b0h), &zero);
    let pred_inv = m.adjoint(&m.direct(&u, &moins_b0), &zero);
    let (c_reg, atk_reg) = m.adjoint_et_pondere(&d.k_reg, &b0h);
    let (c_inv, atk_inv) = m.adjoint_et_pondere(&d.k_inv, &moins_b0);
    let feuille = |z: &Array3<Complex<T>>| {
        (
            vers_tenseur(&z.mapv(|c| c.re), &d.device, d.dtype).require_grad(),
            vers_tenseur(&z.mapv(|c| c.im), &d.device, d.dtype).require_grad(),
        )
    };
    let feuilles = [feuille(&pred_reg), feuille(&pred_inv), feuille(&c_reg), feuille(&c_inv)];
    let abs = |f: &(Tensor<3>, Tensor<3>)| module(f.0.clone(), f.1.clone());
    let l2 = |obs: &Tensor<3>, f: &(Tensor<3>, Tensor<3>)| moyenne(aplatir(obs.clone() - abs(f)).powi_scalar(2));
    let fidelite = l2(&d.obs_reg, &feuilles[0]) + l2(&d.obs_inv, &feuilles[1]);
    let (abs_c_reg, abs_c_inv) = (abs(&feuilles[2]), abs(&feuilles[3]));
    let abs_u = module(u_re.clone(), u_im.clone());
    let image = (d.lncc(abs_u.clone(), abs_c_reg.clone().detach()) + d.lncc(abs_u, abs_c_inv.clone().detach())).neg();
    let terme_b0 = d.lncc(abs_c_reg, abs_c_inv).neg();
    let lissage = (0..3)
        .map(|axe| {
            let n = d.dims[axe];
            let tranche = |a: usize, b: usize| b0.clone().narrow(axe, a, b - a);
            let diff2 = tranche(2, n) - tranche(1, n - 1) * 2.0 + tranche(0, n - 2);
            moyenne(aplatir(diff2 * (1.0 / d.espacement_mm[axe].powi(2))).powi_scalar(2))
        })
        .reduce(|a, b| a + b)
        .unwrap();
    let u2 = u_re.clone() * u_re.clone() + u_im.clone() * u_im.clone();
    let n: usize = d.dims.iter().product();
    let support = moyenne(aplatir(u2 * d.hors_support.clone())) * (n as f64 / d.n_hors_support as f64);
    let termes = Termes {
        fidelite,
        image,
        b0: terme_b0,
        lissage,
        support,
    };
    (
        termes,
        Coupure {
            u,
            b0: b0h,
            feuilles,
            atk_reg,
            atk_inv,
        },
    )
}

impl<T: Reel> Coupure<T> {
    /// Contributions du modèle CPR aux gradients de la perte par rapport à u (complexe, ∂L/∂Re + i ∂L/∂Im) et à b0,
    /// à ajouter aux gradients que Burn donne directement pour u et b0.
    pub fn gradients(&self, d: &Donnees<T>, grads: &Gradients) -> (Array3<Complex<T>>, Array3<T>) {
        let lire = |f: &(Tensor<3>, Tensor<3>)| {
            let g = |t: &Tensor<3>| t.grad(grads).map_or_else(|| Array3::zeros(d.dims), depuis_tenseur::<T>);
            complexe(g(&f.0), g(&f.1))
        };
        let [g_pred_reg, g_pred_inv, g_c_reg, g_c_inv] = self.feuilles.each_ref().map(lire);
        let m = &d.modele;
        let zero = Array3::<T>::zeros(d.dims);
        let moins_b0 = self.b0.mapv(|v| -v);
        // pred = F⁻¹ y avec F unitaire : ∂L/∂y = F ∂L/∂pred, et F = A(0).
        let (gu_reg, gb_reg) = m.gradient_direct(&self.u, &self.b0, &m.direct(&g_pred_reg, &zero));
        let (gu_inv, gb_inv) = m.gradient_direct(&self.u, &moins_b0, &m.direct(&g_pred_inv, &zero));
        let gc_reg = m.gradient_adjoint_b0(&self.atk_reg, &g_c_reg);
        let gc_inv = m.gradient_adjoint_b0(&self.atk_inv, &g_c_inv);
        // les termes calculés avec −b0 contribuent avec le signe opposé
        let gb0 = &gb_reg - &gb_inv + &gc_reg - &gc_inv;
        (&gu_reg + &gu_inv, gb0)
    }
}
