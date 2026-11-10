//! **Gradient analytique** du coût de recalage d'une coupe (F4.0.6, 0.6a) : la même chaîne que [`super::slice_ncc`] et
//! [`super::temporal_penalty`], dérivée à la main, en `f64`, sur des tableaux ordinaires (aucun tenseur Burn, aucune différentiation
//! automatique).
//!
//! La perte est celle de [`super::register_slice_with_prior`] : `L(p) = −NCC(a, s(p)) + λ · moyenne ‖x'(p) − Q‖²`, avec
//! `p = (φ, t)`, `ω = φ / échelle`, `x' = c + R(ω)(x − c) + t`, `v = A⁻¹ x'`, `s = trilinéaire(v)`. Gradient :
//!
//! - `∂L/∂t = Σᵢ gᵢ` et `∂L/∂φₖ = (1/échelle) Σᵢ gᵢ · (∂R/∂ωₖ) dᵢ`, où `dᵢ = xᵢ − c` et `gᵢ = ∂L/∂x'ᵢ` ;
//! - `gᵢ = −(∂NCC/∂sᵢ) A⁻ᵀ ∇ₓs(vᵢ) + (2λ/P)(x'ᵢ − Qᵢ)` ;
//! - `∂NCC/∂bᵢ = wᵢ / D · [(aᵢ − ā) − (cov V_a / D²)(bᵢ − b̄)]`, `D = √(V_a V_b + ε)` (les termes de dérivation des moyennes pesées
//!   s'annulent) ;
//! - `∂R/∂ωₖ = 2ωₖα K + a Eₖ + 2ωₖβ K² + b (Eₖ K + K Eₖ)`, avec la **même série de Taylor sous θ² = 1e-2** que
//!   [`super::rotation_matrix`] (`α = da/dθ²`, `β = db/dθ²`).
//!
//! **Conventions reproduites à l'identique de l'autodiff**, là où la dérivée n'est pas unique : cellule `[floor(c), floor(c) + 1]` aux
//! coordonnées entières, indices bornés au dernier voxel (les deux coins sont alors le même voxel : dérivée nulle sur cet axe), coordonnée
//! ramenée dans `[0, n − 1]` (`clamp`) avant le calcul de la fraction, pixels hors grille ou de poids nul sans contribution.
//!
//! L'autodiff de Burn est l'oracle de la précision (voir les tests) ; l'optimiseur est Adam, réécrit à l'identique de celui de Burn
//! (`burn-optim` 0.22 : `β₁ = 0,9`, `β₂ = 0,999`, `ε = 1e-5` en `f32`, correction de biais fusionnée dans le pas).

use nalgebra::{Matrix3, Matrix4, Vector3};

use super::{
    departs_deterministes, RegistrationConfig, RegistrationResult, RobustConfig, RobustResult, NCC_EPS,
    ROTATION_SERIES_THETA2,
};
use crate::Volume;

/// Un volume de référence lu sur place (aucune copie) : voxels dans l'ordre d'un `Array3` standard (`index = (i · ny + j) · nz + k`),
/// dimensions et inverse de l'affine (monde → indices de voxel).
#[derive(Debug, Clone, Copy)]
pub struct VolumeHote<'a> {
    voxels: &'a [f32],
    dims: [usize; 3],
    inverse: Matrix4<f64>,
}

impl<'a> VolumeHote<'a> {
    /// Lit `volume` sans le copier.
    ///
    /// # Panics
    /// Si les voxels ne sont pas contigus en ordre standard (cas impossible pour un [`Volume`] construit par [`Volume::new`]).
    pub fn new(volume: &'a Volume) -> VolumeHote<'a> {
        let (nx, ny, nz) = volume.dim();
        VolumeHote {
            voxels: volume
                .data
                .as_slice()
                .expect("les voxels d'un Volume sont contigus, en ordre standard"),
            dims: [nx, ny, nz],
            inverse: volume.inverse,
        }
    }
}

/// Une coupe à recaler : ses pixels dans le monde (à la pose d'en-tête), le pivot, l'échelle de rotation, ses intensités et ses poids
/// (masque, 0 ou 1).
#[derive(Debug, Clone, Copy)]
pub struct CoupeHote<'a> {
    pub points: &'a [Vector3<f64>],
    pub pivot: Vector3<f64>,
    pub echelle_mm: f64,
    pub intensites: &'a [f64],
    pub masque: &'a [f64],
}

/// A priori temporel (voir [`super::TemporalPrior`]) : positions cibles de chaque pixel et poids λ (1/mm²).
#[derive(Debug, Clone, Copy)]
pub struct PrioriHote<'a> {
    pub cibles: &'a [Vector3<f64>],
    pub poids: f64,
}

/// Valeur de la NCC et du coût complet, et gradient du coût complet par rapport aux 6 paramètres `(φx, φy, φz, tx, ty, tz)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoutEtGradient {
    pub ncc: f64,
    /// `−NCC + pénalité` (la perte minimisée) ; `−NCC` sans a priori.
    pub perte: f64,
    pub penalite: f64,
    /// Gradient de la perte.
    pub gradient: [f64; 6],
}

/// Les coefficients de la formule de Rodrigues et leurs dérivées en `θ²` : `(a, b, α, β)` avec `a = sin θ / θ`,
/// `b = (1 − cos θ) / θ²`, `α = da/dθ²`, `β = db/dθ²`. Série de Taylor sous [`ROTATION_SERIES_THETA2`] (même seuil que la matrice
/// de rotation de l'autodiff), formules exactes au-dessus.
fn coefficients_rodrigues(th2: f64) -> (f64, f64, f64, f64) {
    if th2 < ROTATION_SERIES_THETA2 {
        let th4 = th2 * th2;
        (
            1.0 - th2 / 6.0 + th4 / 120.0,
            0.5 - th2 / 24.0 + th4 / 720.0,
            -1.0 / 6.0 + th2 / 60.0,
            -1.0 / 24.0 + th2 / 360.0,
        )
    } else {
        let th = th2.sqrt();
        let (s, c) = th.sin_cos();
        (
            s / th,
            (1.0 - c) / th2,
            (th * c - s) / (2.0 * th * th2),
            (th * s - 2.0 * (1.0 - c)) / (2.0 * th2 * th2),
        )
    }
}

/// Matrice antisymétrique `[v]×` du produit vectoriel (`[v]× x = v × x`).
fn croix(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// `R(ω)` et les trois dérivées `∂R/∂ω_k` (la dérivée de `R d` par rapport à `ω_k` est `(∂R/∂ω_k) d`).
fn rotation_et_derivees(omega: Vector3<f64>) -> (Matrix3<f64>, [Matrix3<f64>; 3]) {
    let (a, b, alpha, beta) = coefficients_rodrigues(omega.norm_squared());
    let k = croix(omega);
    let k2 = k * k;
    let r = Matrix3::identity() + k * a + k2 * b;
    let derivee = |axe: usize| {
        let e = croix(Vector3::ith(axe, 1.0));
        k * (2.0 * omega[axe] * alpha) + e * a + k2 * (2.0 * omega[axe] * beta) + (e * k + k * e) * b
    };
    (r, [derivee(0), derivee(1), derivee(2)])
}

/// Interpolation trilinéaire en `v` (indices de voxel continus) : `(valeur, dérivée par rapport à v, dans la grille ?)`, avec les
/// conventions de [`super::trilinear_sample`] (voir l'en-tête du module).
fn echantillon(volume: &VolumeHote, v: Vector3<f64>) -> (f64, Vector3<f64>, bool) {
    let mut dedans = true;
    let mut i0 = [0usize; 3];
    let mut i1 = [0usize; 3];
    let mut t = [0.0f64; 3];
    let mut dt = [0.0f64; 3];
    for a in 0..3 {
        let n = volume.dims[a];
        let max = (n - 1) as f64;
        dedans &= v[a] >= 0.0 && v[a] <= max;
        let c = v[a].clamp(0.0, max);
        let plancher = c.floor();
        t[a] = c - plancher;
        // la dérivée de `clamp` est 1 dans [0, n−1] bornes comprises (convention de Burn, vérifiée par le test aux bornes)
        dt[a] = 1.0;
        i0[a] = (plancher as usize).min(n - 1);
        i1[a] = (i0[a] + 1).min(n - 1);
    }
    let pas = [volume.dims[1] * volume.dims[2], volume.dims[2], 1];
    let voxel = |ix: usize, iy: usize, iz: usize| f64::from(volume.voxels[ix * pas[0] + iy * pas[1] + iz * pas[2]]);
    let mut valeur = 0.0;
    let mut gradient = Vector3::zeros();
    for (bx, ix) in [i0[0], i1[0]].into_iter().enumerate() {
        let wx = if bx == 1 { t[0] } else { 1.0 - t[0] };
        let sx = if bx == 1 { 1.0 } else { -1.0 };
        for (by, iy) in [i0[1], i1[1]].into_iter().enumerate() {
            let wy = if by == 1 { t[1] } else { 1.0 - t[1] };
            let sy = if by == 1 { 1.0 } else { -1.0 };
            for (bz, iz) in [i0[2], i1[2]].into_iter().enumerate() {
                let wz = if bz == 1 { t[2] } else { 1.0 - t[2] };
                let sz = if bz == 1 { 1.0 } else { -1.0 };
                let x = voxel(ix, iy, iz);
                valeur += wx * wy * wz * x;
                gradient.x += sx * dt[0] * wy * wz * x;
                gradient.y += wx * sy * dt[1] * wz * x;
                gradient.z += wx * wy * sz * dt[2] * x;
            }
        }
    }
    (valeur, gradient, dedans)
}

/// `(ω, R, ∂R/∂ω_k)` d'une pose : `ω = φ / échelle`.
fn rotation_de_la_pose(params: &[f64; 6], echelle_mm: f64) -> (Matrix3<f64>, [Matrix3<f64>; 3]) {
    rotation_et_derivees(Vector3::new(params[0], params[1], params[2]) / echelle_mm)
}

/// Gradient de la perte par rapport aux paramètres, à partir des dérivées `g_i = ∂L/∂x'_i` de chaque pixel.
fn gradient_des_parametres(coupe: &CoupeHote, derivees_r: &[Matrix3<f64>; 3], g: &[Vector3<f64>]) -> [f64; 6] {
    let mut sortie = [0.0; 6];
    for (gi, x) in g.iter().zip(coupe.points) {
        let d = x - coupe.pivot;
        for k in 0..3 {
            sortie[k] += gi.dot(&(derivees_r[k] * d)) / coupe.echelle_mm;
        }
        sortie[3] += gi.x;
        sortie[4] += gi.y;
        sortie[5] += gi.z;
    }
    sortie
}

/// Positions déplacées `x' = c + R d + t` de tous les pixels de la coupe.
fn deplaces(coupe: &CoupeHote, r: &Matrix3<f64>, params: &[f64; 6]) -> Vec<Vector3<f64>> {
    let t = Vector3::new(params[3], params[4], params[5]);
    coupe
        .points
        .iter()
        .map(|x| coupe.pivot + r * (x - coupe.pivot) + t)
        .collect()
}

/// Pénalité temporelle `λ · moyenne ‖x' − Q‖²` et sa dérivée par rapport à chaque `x'`.
fn penalite(deplaces: &[Vector3<f64>], prior: &PrioriHote) -> (f64, Vec<Vector3<f64>>) {
    let p = deplaces.len() as f64;
    let facteur = prior.poids / p;
    let mut valeur = 0.0;
    let derivees = deplaces
        .iter()
        .zip(prior.cibles)
        .map(|(x, q)| {
            let e = x - q;
            valeur += e.norm_squared();
            e * (2.0 * facteur)
        })
        .collect();
    (valeur * facteur, derivees)
}

/// NCC pondérée (formule de [`super::ncc`], `f64`) et dérivée par rapport à chaque valeur échantillonnée `b_i`.
fn ncc_et_derivees(a: &[f64], b: &[f64], w: &[f64]) -> (f64, Vec<f64>) {
    let somme_w = w.iter().sum::<f64>().max(1e-12);
    let moyenne = |x: &[f64]| x.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / somme_w;
    let (ma, mb) = (moyenne(a), moyenne(b));
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for ((&ai, &bi), &wi) in a.iter().zip(b).zip(w) {
        let (da, db) = (ai - ma, bi - mb);
        cov += wi * da * db;
        va += wi * da * da;
        vb += wi * db * db;
    }
    let d2 = va * vb + NCC_EPS;
    let d = d2.sqrt();
    let facteur = cov * va / d2;
    let derivees = a
        .iter()
        .zip(b)
        .zip(w)
        .map(|((&ai, &bi), &wi)| wi / d * ((ai - ma) - facteur * (bi - mb)))
        .collect();
    (cov / d, derivees)
}

/// **Valeur et gradient analytiques de la NCC seule** (`−` est pour la perte : voir [`cout_et_gradient`]). Le gradient rendu est celui de
/// la NCC (à maximiser).
pub fn ncc_et_gradient(volume: &VolumeHote, coupe: &CoupeHote, params: &[f64; 6]) -> (f64, [f64; 6]) {
    let c = cout_et_gradient(volume, coupe, params, None);
    (c.ncc, c.gradient.map(|g| -g))
}

/// **Valeur et gradient analytiques de la pénalité seule** `λ · moyenne ‖x' − Q‖²`.
pub fn penalite_et_gradient(coupe: &CoupeHote, prior: &PrioriHote, params: &[f64; 6]) -> (f64, [f64; 6]) {
    let (r, dr) = rotation_de_la_pose(params, coupe.echelle_mm);
    let x2 = deplaces(coupe, &r, params);
    let (valeur, g) = penalite(&x2, prior);
    (valeur, gradient_des_parametres(coupe, &dr, &g))
}

/// **Perte `−NCC + pénalité`** et son gradient, en une passe (la NCC et la pénalité partagent la rotation et ses dérivées).
///
/// Un a priori de poids nul ou négatif est ignoré, comme dans [`super::register_slice_with_prior`].
pub fn cout_et_gradient(
    volume: &VolumeHote,
    coupe: &CoupeHote,
    params: &[f64; 6],
    prior: Option<&PrioriHote>,
) -> CoutEtGradient {
    let p = coupe.points.len();
    let (r, dr) = rotation_de_la_pose(params, coupe.echelle_mm);
    let x2 = deplaces(coupe, &r, params);
    let mut valeurs = vec![0.0; p];
    let mut poids = vec![0.0; p];
    let mut grad_x = vec![Vector3::zeros(); p]; // ∇ₓ s(vᵢ) = A⁻ᵀ ∇ᵥ s
    let lineaire = volume.inverse.fixed_view::<3, 3>(0, 0).into_owned();
    for i in 0..p {
        let v = lineaire * x2[i] + volume.inverse.fixed_view::<3, 1>(0, 3).into_owned();
        let (valeur, dv, dedans) = echantillon(volume, v);
        valeurs[i] = valeur;
        poids[i] = if dedans { coupe.masque[i] } else { 0.0 };
        grad_x[i] = lineaire.transpose() * dv;
    }
    let (ncc, d_ncc) = ncc_et_derivees(coupe.intensites, &valeurs, &poids);
    // g_i = ∂(−NCC)/∂x'_i
    let mut g: Vec<Vector3<f64>> = (0..p).map(|i| grad_x[i] * (-d_ncc[i])).collect();
    let mut perte = -ncc;
    let mut valeur_penalite = 0.0;
    if let Some(a) = prior.filter(|a| a.poids > 0.0) {
        let (pen, dpen) = penalite(&x2, a);
        valeur_penalite = pen;
        perte += pen;
        for (gi, di) in g.iter_mut().zip(dpen) {
            *gi += di;
        }
    }
    CoutEtGradient {
        ncc,
        perte,
        penalite: valeur_penalite,
        gradient: gradient_des_parametres(coupe, &dr, &g),
    }
}

/// Adam de Burn (`burn-optim` 0.22), réécrit : état `(t, m, v)` ; pas `lr · m · c / (√v + ε √(1 − β₂ᵗ))`, `c = √(1 − β₂ᵗ) / (1 − β₁ᵗ)`.
struct Adam {
    t: i32,
    m: [f64; 6],
    v: [f64; 6],
}

impl Adam {
    // constantes de Burn : f32, converties sans arrondi supplémentaire
    const BETA1: f64 = 0.9_f32 as f64;
    const BETA2: f64 = 0.999_f32 as f64;
    const EPSILON: f64 = 1e-5_f32 as f64;

    fn new() -> Adam {
        Adam {
            t: 0,
            m: [0.0; 6],
            v: [0.0; 6],
        }
    }

    fn pas(&mut self, params: &mut [f64; 6], gradient: &[f64; 6], lr: f64) {
        self.t += 1;
        let (b1, b2) = (Self::BETA1, Self::BETA2);
        let correction2 = (1.0 - b2.powi(self.t)).sqrt();
        let facteur = correction2 / (1.0 - b1.powi(self.t));
        for k in 0..6 {
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * gradient[k];
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * gradient[k] * gradient[k];
            let delta = self.m[k] * facteur / (self.v[k].sqrt() + Self::EPSILON * correction2);
            params[k] -= lr * delta;
        }
    }
}

/// **Recale une coupe** comme [`super::register_slice_with_prior`], avec le gradient analytique à la place de l'autodiff : même perte, même
/// Adam (pas initial, décroissance, nombre d'itérations de `config`), même historique de NCC (valeur avant chaque mise à jour, puis à la pose finale).
pub fn register_slice_with_prior_analytique(
    volume: &VolumeHote,
    coupe: &CoupeHote,
    config: RegistrationConfig,
    prior: Option<&PrioriHote>,
) -> RegistrationResult {
    let mut params = config.initial;
    let mut adam = Adam::new();
    let mut historique = Vec::with_capacity(config.iterations + 1);
    let mut pas = config.learning_rate;
    for _ in 0..config.iterations {
        let c = cout_et_gradient(volume, coupe, &params, prior);
        historique.push(c.ncc);
        adam.pas(&mut params, &c.gradient, pas);
        pas *= config.lr_decay;
    }
    historique.push(cout_et_gradient(volume, coupe, &params, None).ncc);
    RegistrationResult {
        params,
        ncc_history: historique,
    }
}

/// **Recalage robuste** d'une coupe, comme [`super::register_slice_robust_with_prior`] : premier recalage depuis `config.base.initial` ; si sa
/// NCC finale est sous `config.suspect_ncc`, relances depuis `config.extra_starts` poses déterministes, et on garde l'essai de **perte finale** la
/// plus basse (`NCC − pénalité` la plus haute).
pub fn register_slice_robust_with_prior_analytique(
    volume: &VolumeHote,
    coupe: &CoupeHote,
    config: RobustConfig,
    prior: Option<&PrioriHote>,
) -> RobustResult {
    let finale = |r: &RegistrationResult| *r.ncc_history.last().unwrap();
    let objectif = |r: &RegistrationResult| {
        let penalite = match prior {
            Some(a) if a.poids > 0.0 => penalite_et_gradient(coupe, a, &r.params).0,
            _ => 0.0,
        };
        finale(r) - penalite
    };
    let premier = register_slice_with_prior_analytique(volume, coupe, config.base, prior);
    let first_ncc = finale(&premier);
    let mut meilleur = RobustResult {
        params: premier.params,
        ncc: first_ncc,
        first_ncc,
        runs: 1,
    };
    if first_ncc >= config.suspect_ncc {
        return meilleur;
    }
    let mut meilleur_objectif = objectif(&premier);
    for depart in departs_deterministes(config.extra_starts, config.start_amplitude_mm, config.seed) {
        let essai = register_slice_with_prior_analytique(
            volume,
            coupe,
            RegistrationConfig {
                initial: depart,
                ..config.base
            },
            prior,
        );
        meilleur.runs += 1;
        if objectif(&essai) > meilleur_objectif {
            meilleur_objectif = objectif(&essai);
            meilleur.ncc = finale(&essai);
            meilleur.params = essai.params;
        }
    }
    meilleur
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{rotation_scale_mm, slice_ncc, temporal_penalty, TemporalPrior, VolumeTensors};
    use burn::prelude::*;
    use ndarray::Array3;

    /// xorshift64 déterministe, valeurs dans [0, 1[.
    struct Alea(u64);
    impl Alea {
        fn suivant(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn entre(&mut self, a: f64, b: f64) -> f64 {
            a + (b - a) * self.suivant()
        }
    }

    fn f32_exact(x: f64) -> f64 {
        f64::from(x as f32)
    }

    /// Trois grilles : cube régulier, grille anisotrope (0,8 / 0,8 / 3 mm), grille d'affine tournée avec décalage.
    fn affines() -> [(&'static str, Matrix4<f64>); 3] {
        let tournee = nalgebra::Rotation3::from_euler_angles(0.3, -0.2, 0.5);
        let mut t = Matrix4::identity();
        t.fixed_view_mut::<3, 3>(0, 0).copy_from(&(tournee.into_inner() * 0.9));
        t[(0, 3)] = 4.5;
        t[(1, 3)] = -7.25;
        t[(2, 3)] = 12.125;
        [
            ("cube", Matrix4::identity()),
            (
                "anisotrope",
                Matrix4::new_nonuniform_scaling(&Vector3::new(0.8, 0.8, 3.0)),
            ),
            ("tournée", t),
        ]
    }

    /// Volume lisse (produits de sinus) plus un bruit, d'affine donnée.
    fn volume(affine: Matrix4<f64>, alea: &mut Alea) -> Volume {
        let data = Array3::from_shape_fn((24, 22, 20), |(i, j, k)| {
            let (x, y, z) = (i as f64, j as f64, k as f64);
            let lisse = (0.35 * x).sin() * (0.3 * y).cos() + 0.7 * (0.25 * z + 0.1 * x).sin();
            (lisse + 0.25 * alea.suivant()) as f32
        });
        Volume::new(data, affine).unwrap()
    }

    /// Une coupe de 12 × 12 pixels, hors des nœuds de la grille (positions fractionnaires), et sa lecture d'une pose connue plus du bruit.
    struct Scene {
        points: Vec<Vector3<f64>>,
        pivot: Vector3<f64>,
        echelle: f64,
        intensites: Vec<f64>,
        masque: Vec<f64>,
        cibles: Vec<Vector3<f64>>,
    }

    fn scene(volume: &Volume, alea: &mut Alea) -> Scene {
        let mut points = Vec::new();
        for a in 0..12 {
            for b in 0..12 {
                let (a, b) = (a as f64, b as f64);
                let v = Vector3::new(6.13 + 0.91 * a, 5.27 + 0.87 * b, 9.31 + 0.013 * a - 0.007 * b);
                let m = volume.affine * v.push(1.0);
                points.push(Vector3::new(f32_exact(m.x), f32_exact(m.y), f32_exact(m.z)));
            }
        }
        let pivot = points.iter().sum::<Vector3<f64>>() / points.len() as f64;
        let pivot = Vector3::new(f32_exact(pivot.x), f32_exact(pivot.y), f32_exact(pivot.z));
        let echelle = f32_exact(rotation_scale_mm(&points, &pivot).unwrap());
        let vrai = [0.4, -0.3, 0.5, 0.6, -0.5, 0.4];
        let intensites = points
            .iter()
            .map(|p| {
                let m = crate::diff::pose_to_matrix(&vrai, &pivot, echelle) * p.push(1.0);
                let lu = volume.sample(&m.xyz()).unwrap_or(0.0);
                f32_exact(f64::from(lu) * 1.3 + 0.2 + alea.entre(-0.1, 0.1))
            })
            .collect();
        let cibles = points
            .iter()
            .map(|p| {
                Vector3::new(
                    f32_exact(p.x + alea.entre(-1.0, 1.0)),
                    f32_exact(p.y + alea.entre(-1.0, 1.0)),
                    f32_exact(p.z + alea.entre(-1.0, 1.0)),
                )
            })
            .collect();
        let n = points.len();
        Scene {
            points,
            pivot,
            echelle,
            intensites,
            masque: vec![1.0; n],
            cibles,
        }
    }

    impl Scene {
        fn coupe(&self) -> CoupeHote<'_> {
            CoupeHote {
                points: &self.points,
                pivot: self.pivot,
                echelle_mm: self.echelle,
                intensites: &self.intensites,
                masque: &self.masque,
            }
        }
    }

    fn pose_aleatoire(alea: &mut Alea) -> [f64; 6] {
        std::array::from_fn(|k| {
            f32_exact(if k < 3 {
                alea.entre(-3.0, 3.0)
            } else {
                alea.entre(-1.2, 1.2)
            })
        })
    }

    /// Référence `f64` indépendante du code testé : NCC et pénalité par une chaîne écrite autrement (`pose_to_matrix` de nalgebra,
    /// `Volume::sample` à la maille `f64`, formule de la NCC sans dérivée).
    fn reference_f64(volume: &Volume, s: &Scene, params: &[f64; 6], poids_prior: f64) -> (f64, f64) {
        let m = crate::diff::pose_to_matrix(params, &s.pivot, s.echelle);
        let mut b = Vec::new();
        let mut w = Vec::new();
        let mut pen = 0.0;
        for (i, p) in s.points.iter().enumerate() {
            let x2 = (m * p.push(1.0)).xyz();
            pen += (x2 - s.cibles[i]).norm_squared();
            let vox = volume.voxel_coordinates(&x2);
            let voisins = volume.trilinear(&vox);
            match voisins {
                Some(v) => {
                    b.push(v.iter().map(|(idx, w)| w * f64::from(volume.data[*idx])).sum::<f64>());
                    w.push(s.masque[i]);
                }
                None => {
                    b.push(0.0);
                    w.push(0.0);
                }
            }
        }
        let sw: f64 = w.iter().sum();
        let ma = s.intensites.iter().zip(&w).map(|(x, w)| x * w).sum::<f64>() / sw;
        let mb = b.iter().zip(&w).map(|(x, w)| x * w).sum::<f64>() / sw;
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
        for i in 0..b.len() {
            let (da, db) = (s.intensites[i] - ma, b[i] - mb);
            cov += w[i] * da * db;
            va += w[i] * da * da;
            vb += w[i] * db * db;
        }
        (
            cov / (va * vb + NCC_EPS).sqrt(),
            pen * poids_prior / s.points.len() as f64,
        )
    }

    /// Oracle : la perte de Burn (`slice_ncc`, `temporal_penalty`) et son gradient par l'autodiff, en `f32`.
    fn autodiff(volume: &Volume, s: &Scene, params: &[f64; 6], poids_prior: f64) -> (f64, f64, [f64; 6]) {
        let device = Device::flex().autodiff();
        let vt = VolumeTensors::new(volume, &device);
        let tenseur =
            |v: &[f64]| Tensor::<1>::from_floats(v.iter().map(|&x| x as f32).collect::<Vec<_>>().as_slice(), &device);
        let pts: Vec<f32> = s
            .points
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let cib: Vec<f32> = s
            .cibles
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let n = s.points.len();
        let points = Tensor::<2>::from_floats(TensorData::new(pts, [n, 3]), &device);
        let prior = TemporalPrior {
            targets: Tensor::<2>::from_floats(TensorData::new(cib, [n, 3]), &device),
            weight: poids_prior,
        };
        let pose = tenseur(params).require_grad();
        let pivot = tenseur(&[s.pivot.x, s.pivot.y, s.pivot.z]);
        let ncc = slice_ncc(
            &vt,
            points.clone(),
            pivot.clone(),
            pose.clone(),
            s.echelle,
            tenseur(&s.intensites),
            tenseur(&s.masque),
        );
        let pen = temporal_penalty(points, pivot, pose.clone(), s.echelle, &prior);
        let valeur_ncc = f64::from(ncc.clone().into_data().try_to_vec::<f32>().unwrap()[0]);
        let valeur_pen = f64::from(pen.clone().into_data().try_to_vec::<f32>().unwrap()[0]);
        let perte = (ncc.neg() + pen).sum();
        let grads = perte.backward();
        let g: Vec<f32> = pose.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();
        (valeur_ncc, valeur_pen, std::array::from_fn(|k| f64::from(g[k])))
    }

    fn norme(v: &[f64; 6]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn ecart_relatif(a: &[f64; 6], b: &[f64; 6]) -> f64 {
        let d: [f64; 6] = std::array::from_fn(|k| a[k] - b[k]);
        norme(&d) / norme(b)
    }

    /// Différences finies centrées en `f64` de la perte de la référence indépendante ; `None` si une coordonnée de voxel est trop près d'un
    /// plan de voxels (la dérivée y saute et les différences finies ne valent rien).
    fn gradient_fd(
        volume: &Volume,
        s: &Scene,
        params: &[f64; 6],
        poids_prior: f64,
        avec_ncc: bool,
        avec_pen: bool,
    ) -> Option<[f64; 6]> {
        let h = 1e-5;
        let perte = |p: &[f64; 6]| {
            let (n, pen) = reference_f64(volume, s, p, poids_prior);
            (if avec_ncc { -n } else { 0.0 }) + if avec_pen { pen } else { 0.0 }
        };
        for &delta in &[0.0, h, -h] {
            let mut p = *params;
            p[0] += delta;
            let m = crate::diff::pose_to_matrix(&p, &s.pivot, s.echelle);
            for x in &s.points {
                let v = volume.voxel_coordinates(&(m * x.push(1.0)).xyz());
                if (0..3).any(|a| (v[a] - v[a].round()).abs() < 2e-3) {
                    return None;
                }
            }
        }
        Some(std::array::from_fn(|k| {
            let (mut plus, mut moins) = (*params, *params);
            plus[k] += h;
            moins[k] -= h;
            (perte(&plus) - perte(&moins)) / (2.0 * h)
        }))
    }

    /// Valeurs et gradients de la NCC seule, de la pénalité seule et de la perte complète : analytique contre autodiff `f32` (oracle de
    /// Burn) et contre des différences finies `f64` indépendantes, sur 100 poses aléatoires et trois grilles.
    ///
    /// Précision demandée (F4.0.6) : valeurs à 1e-6 en relatif ; gradients à 1e-5 en relatif, en norme : l'analytique contre l'autodiff
    /// là où l'autodiff est lui-même à 1e-5 des différences finies, et l'analytique contre les différences finies partout ailleurs.
    #[test]
    fn analytique_contre_autodiff_et_differences_finies() {
        let mut alea = Alea(0x2026_1110_0006_0A01);
        let poids = [0.0, 0.02, 0.3];
        for (nom, affine) in affines() {
            let vol = volume(affine, &mut alea);
            let s = scene(&vol, &mut alea);
            let hote = VolumeHote::new(&vol);
            let (mut n, mut fiables, mut ecarts_fd, mut ecarts_auto, mut pire_valeur) =
                (0usize, 0usize, 0.0_f64, 0.0_f64, 0.0_f64);
            let mut pire_auto_non_fiable = 0.0_f64;
            while n < 100 {
                let params = pose_aleatoire(&mut alea);
                let lambda = poids[n % 3];
                let prior = PrioriHote {
                    cibles: &s.cibles,
                    poids: lambda,
                };
                let Some(g_fd_perte) = gradient_fd(&vol, &s, &params, lambda, true, lambda > 0.0) else {
                    continue;
                };
                n += 1;
                let analytique = cout_et_gradient(&hote, &s.coupe(), &params, (lambda > 0.0).then_some(&prior));
                let (val_ref, pen_ref) = reference_f64(&vol, &s, &params, lambda);
                // valeurs : contre la référence f64 indépendante (1e-9) puis contre l'autodiff f32 (1e-6)
                assert!(
                    (analytique.ncc - val_ref).abs() <= 1e-9 * val_ref.abs().max(1e-3),
                    "{nom} NCC {} contre {val_ref}",
                    analytique.ncc
                );
                assert!(
                    (analytique.penalite - if lambda > 0.0 { pen_ref } else { 0.0 }).abs()
                        <= 1e-9 * pen_ref.abs().max(1e-3),
                    "{nom} pénalité"
                );
                let (a_ncc, a_pen, a_grad) = autodiff(&vol, &s, &params, lambda);
                pire_valeur = pire_valeur.max((analytique.ncc - a_ncc).abs() / a_ncc.abs().max(1e-3));
                // gradients : contre les différences finies f64 (toujours)
                let e_fd = ecart_relatif(&analytique.gradient, &g_fd_perte);
                ecarts_fd = ecarts_fd.max(e_fd);
                assert!(
                    e_fd < 1e-5,
                    "{nom} pose {params:?} λ {lambda} : analytique contre différences finies {e_fd:.2e}"
                );
                // contre l'autodiff, là où il est fiable
                let e_auto_fd = ecart_relatif(&a_grad, &g_fd_perte);
                let e_auto = ecart_relatif(&analytique.gradient, &a_grad);
                if e_auto_fd < 1e-5 {
                    fiables += 1;
                    ecarts_auto = ecarts_auto.max(e_auto);
                    assert!(
                        e_auto < 1e-5,
                        "{nom} pose {params:?} : analytique contre autodiff {e_auto:.2e} (autodiff fiable)"
                    );
                } else {
                    pire_auto_non_fiable = pire_auto_non_fiable.max(e_auto_fd);
                }
                let _ = (a_pen, pen_ref);
            }
            // NCC seule et pénalité seule, sur quelques poses : les trois morceaux du gradient
            for _ in 0..10 {
                let params = pose_aleatoire(&mut alea);
                let prior = PrioriHote {
                    cibles: &s.cibles,
                    poids: 0.2,
                };
                let Some(fd_ncc) = gradient_fd(&vol, &s, &params, 0.2, true, false) else {
                    continue;
                };
                let (_, g_ncc) = ncc_et_gradient(&hote, &s.coupe(), &params);
                let g_ncc_perte = g_ncc.map(|g| -g);
                assert!(ecart_relatif(&g_ncc_perte, &fd_ncc) < 1e-5, "{nom} NCC seule");
                let fd_pen = gradient_fd(&vol, &s, &params, 0.2, false, true).unwrap_or([0.0; 6]);
                let (_, g_pen) = penalite_et_gradient(&s.coupe(), &prior, &params);
                if norme(&fd_pen) > 0.0 {
                    assert!(ecart_relatif(&g_pen, &fd_pen) < 1e-5, "{nom} pénalité seule");
                }
            }
            println!(
                "{nom} : {n} poses ; gradient contre différences finies f64 : pire {ecarts_fd:.2e} ; contre autodiff (fiable sur {fiables}/{n}) : pire {ecarts_auto:.2e} ; \
                 autodiff hors 1e-5 des différences finies : pire {pire_auto_non_fiable:.2e} ; valeur de la NCC contre autodiff : pire {pire_valeur:.2e}"
            );
            // F4.0.6 demandait 1e-6 en relatif contre l'autodiff. Mesuré : l'analytique `f64` est à 1e-9 de la référence `f64`
            // indépendante (assertions ci-dessus), l'autodiff `f32` s'en écarte de quelques 1e-6 (somme de 144 pixels en `f32`) : 1e-6
            // contre l'oracle `f32` n'est donc pas tenable par une implémentation exacte ; on borne à 1e-5 et l'écart mesuré est rapporté.
            assert!(
                pire_valeur < 1e-5,
                "{nom} : valeur de la NCC contre l'autodiff {pire_valeur:.2e}"
            );
            assert!(
                fiables >= 50,
                "{nom} : l'autodiff f32 n'est fiable que sur {fiables} poses sur {n}"
            );
        }
    }

    /// Gradient analytique contre autodiff (conventions) : normes relatives à `max(‖g‖, 1e-6)` ; les deux doivent être finis.
    fn comparer_a_l_autodiff(nom: &str, vol: &Volume, s: &Scene, params: &[f64; 6], lambda: f64) {
        let hote = VolumeHote::new(vol);
        let prior = PrioriHote {
            cibles: &s.cibles,
            poids: lambda,
        };
        let a = cout_et_gradient(&hote, &s.coupe(), params, (lambda > 0.0).then_some(&prior));
        let (_, _, g) = autodiff(vol, s, params, lambda);
        assert!(
            a.gradient.iter().all(|v| v.is_finite()),
            "{nom} : gradient analytique non fini {:?}",
            a.gradient
        );
        assert!(
            g.iter().all(|v| v.is_finite()),
            "{nom} : gradient autodiff non fini {g:?}"
        );
        let ecart = norme(&std::array::from_fn(|k| a.gradient[k] - g[k])) / norme(&g).max(1e-6);
        assert!(
            ecart < 1e-5,
            "{nom} : analytique contre autodiff {ecart:.2e} ({:?} contre {g:?})",
            a.gradient
        );
    }

    /// Cas dégénérés de F4.0.6 : pose nulle (branche série), θ² de part et d'autre de 1e-2, points exactement sur un plan de voxels, sur le
    /// premier et le dernier voxel, coupe constante (NCC = 0, gradient fini), coupe sans pixel de poids non nul, pivot hors du volume.
    #[test]
    fn cas_degeneres() {
        let mut alea = Alea(77);
        for (nom, affine) in affines() {
            let vol = volume(affine, &mut alea);
            let s = scene(&vol, &mut alea);
            // pose nulle
            comparer_a_l_autodiff(&format!("{nom} pose nulle"), &vol, &s, &[0.0; 6], 0.1);
            // θ² juste sous et juste au-dessus du seuil de la série (ω = φ / échelle)
            for th2 in [0.0099_f64, 0.0101] {
                let omega = th2.sqrt() / 3.0_f64.sqrt();
                let phi = f32_exact(omega * s.echelle);
                let p = [phi, phi, phi, 0.3, -0.2, 0.1];
                comparer_a_l_autodiff(&format!("{nom} θ²≈{th2}"), &vol, &s, &p, 0.1);
                // différences finies : seulement si aucun point n'est près d'un plan de voxels (sinon la dérivée y saute)
                if let Some(fd) = gradient_fd(&vol, &s, &p, 0.1, true, true) {
                    let hote = VolumeHote::new(&vol);
                    let prior = PrioriHote {
                        cibles: &s.cibles,
                        poids: 0.1,
                    };
                    let a = cout_et_gradient(&hote, &s.coupe(), &p, Some(&prior));
                    assert!(
                        ecart_relatif(&a.gradient, &fd) < 1e-5,
                        "{nom} θ²≈{th2} contre différences finies"
                    );
                }
            }
            // coupe constante : NCC = 0, gradient fini
            let mut constante = scene(&vol, &mut alea);
            constante.intensites = vec![5.0; constante.points.len()];
            comparer_a_l_autodiff(
                &format!("{nom} coupe constante"),
                &vol,
                &constante,
                &[0.5, -0.4, 0.3, 0.2, 0.1, -0.2],
                0.0,
            );
            let c = cout_et_gradient(
                &VolumeHote::new(&vol),
                &constante.coupe(),
                &[0.5, -0.4, 0.3, 0.2, 0.1, -0.2],
                None,
            );
            assert!(c.ncc.abs() < 1e-6, "{nom} : NCC d'une coupe constante {}", c.ncc);
            // aucun pixel de poids non nul
            let mut vide = scene(&vol, &mut alea);
            vide.masque = vec![0.0; vide.points.len()];
            comparer_a_l_autodiff(
                &format!("{nom} sans poids"),
                &vol,
                &vide,
                &[0.5, -0.4, 0.3, 0.2, 0.1, -0.2],
                0.0,
            );
            // pivot hors du volume (loin) : petites rotations, grands bras de levier
            let mut loin = scene(&vol, &mut alea);
            loin.pivot = Vector3::new(f32_exact(400.0), f32_exact(-300.0), f32_exact(250.0));
            loin.echelle = f32_exact(rotation_scale_mm(&loin.points, &loin.pivot).unwrap());
            // (avec un bras de levier de 500 mm, l'autodiff `f32` perd de la précision dans `x − c` : l'oracle est alors la référence `f64`)
            let p_loin = [0.02, -0.01, 0.015, 0.2, 0.1, -0.1];
            let fd = gradient_fd(&vol, &loin, &p_loin, 0.05, true, true);
            let prior = PrioriHote {
                cibles: &loin.cibles,
                poids: 0.05,
            };
            let a = cout_et_gradient(&VolumeHote::new(&vol), &loin.coupe(), &p_loin, Some(&prior));
            let (_, _, g_auto) = autodiff(&vol, &loin, &p_loin, 0.05);
            assert!(
                g_auto.iter().all(|v| v.is_finite()),
                "{nom} pivot loin : autodiff non fini"
            );
            if let Some(fd) = fd {
                assert!(
                    ecart_relatif(&a.gradient, &fd) < 1e-5,
                    "{nom} pivot loin contre différences finies"
                );
                assert!(ecart_relatif(&g_auto, &fd) < 1e-3, "{nom} pivot loin : autodiff");
            } else {
                assert!(
                    ecart_relatif(&a.gradient, &g_auto) < 1e-3,
                    "{nom} pivot loin contre autodiff"
                );
            }
        }
    }

    /// Points exactement sur un plan de voxels, sur le premier et sur le dernier voxel, et grille d'un seul voxel d'épaisseur : là, la
    /// dérivée n'est pas unique et l'analytique doit suivre la convention de l'autodiff (cube d'affine identité : coordonnée de voxel =
    /// coordonnée monde).
    #[test]
    fn plans_de_voxels_premier_et_dernier_voxel() {
        let mut alea = Alea(4242);
        let vol = volume(Matrix4::identity(), &mut alea);
        // x = 0, 7, 23 (dernier voxel, n = 24) ; y et z fractionnaires. Poids : tous « dedans » (bornes comprises).
        for x in [0.0, 7.0, 23.0] {
            let mut s = scene(&vol, &mut alea);
            for (i, p) in s.points.iter_mut().enumerate() {
                *p = Vector3::new(x, 4.0 + 0.917 * (i % 12) as f64, 5.0 + 1.113 * (i / 12) as f64);
                *p = Vector3::new(f32_exact(p.x), f32_exact(p.y), f32_exact(p.z));
            }
            s.pivot = Vector3::new(f32_exact(x), 9.0, 10.0);
            s.echelle = f32_exact(rotation_scale_mm(&s.points, &s.pivot).unwrap());
            // pose nulle : les points restent exactement sur le plan ; on regarde aussi une translation dans le plan (tx = 0)
            comparer_a_l_autodiff(&format!("plan x={x} pose nulle"), &vol, &s, &[0.0; 6], 0.0);
            comparer_a_l_autodiff(
                &format!("plan x={x} translation dans le plan"),
                &vol,
                &s,
                &[0.0, 0.0, 0.0, 0.0, 0.25, -0.5],
                0.0,
            );
        }
        // y et z sur des plans entiers aussi (coin d'un voxel)
        let mut s = scene(&vol, &mut alea);
        for (i, p) in s.points.iter_mut().enumerate() {
            *p = Vector3::new(3.0 + (i % 12) as f64, 8.0, 7.0);
        }
        s.pivot = Vector3::new(8.0, 8.0, 7.0);
        s.echelle = f32_exact(rotation_scale_mm(&s.points, &s.pivot).unwrap());
        comparer_a_l_autodiff("plan y=8 z=7", &vol, &s, &[0.0; 6], 0.0);
    }
}
