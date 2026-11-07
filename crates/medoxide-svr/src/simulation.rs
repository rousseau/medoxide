//! Simulateur d'acquisition INDÉPENDANT du modèle direct de la reconstruction (docs/ROADMAP.md, F4.0.1).
//!
//! Le banc E2 simule les coupes avec [`crate::Volume::simulate_slice`], le modèle même que la reconstruction inverse : un « crime inverse »
//! qui favorise la méthode partageant ces hypothèses. Ce module réécrit la physique d'acquisition avec d'autres choix : interpolation de la
//! source par B-spline cubique (au lieu de la trilinéaire), intégration d'un voxel de basse résolution sur une grille fine de la source,
//! profil de coupe en sinc fenêtré et résolution dans le plan par un sinc fenêtré (au lieu de noyaux gaussiens), bruit de Rice, biais lisse,
//! échelle par coupe. Il ne modifie aucun élément existant.

use crate::mouvement::Alea;
use crate::recon::SlicePoses;
use crate::{Slice, Stack, SvrError, Volume};
use nalgebra::{Matrix4, Vector3};
use ndarray::{Array2, Array3};
use nifti::{writer::WriterOptions, NiftiHeader};
use rayon::prelude::*;
use std::path::Path;

/// Pôle du préfiltre de la B-spline cubique : `z = √3 − 2`.
const POLE: f64 = -0.267_949_192_431_122_7;

/// Préfiltre récursif (causal puis anticausal) d'un signal 1D : transforme les échantillons en coefficients de B-spline cubique tels que
/// la synthèse `(c[k−1] + 4 c[k] + c[k+1]) / 6` redonne les échantillons (bords en miroir). L'initialisation du filtre causal tronque la
/// somme infinie au rang où `|z|^k < 1e-12` : l'erreur due aux bords est de l'ordre de `|z|^n`, négligeable dès `n ≳ 24`.
fn prefiltre_1d(c: &mut [f64]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    for v in c.iter_mut() {
        *v *= 6.0; // gain (1 − z)(1 − 1/z) = 6
    }
    let horizon = n.min(((1e-12_f64).ln() / POLE.abs().ln()).ceil() as usize);
    let (mut zk, mut somme) = (POLE, c[0]);
    for v in c.iter().take(horizon).skip(1) {
        somme += zk * v;
        zk *= POLE;
    }
    c[0] = somme;
    for k in 1..n {
        c[k] += POLE * c[k - 1];
    }
    c[n - 1] = (POLE / (POLE * POLE - 1.0)) * (c[n - 1] + POLE * c[n - 2]);
    for k in (0..n - 1).rev() {
        c[k] = POLE * (c[k + 1] - c[k]);
    }
}

/// Poids des quatre coefficients voisins (indices `i0 − 1 … i0 + 2`) pour une abscisse fractionnaire `t ∈ [0, 1)` : B-spline cubique.
fn poids_cubiques(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [
        (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// Indice ramené dans `[0, n)` par réflexion (bords en miroir).
fn miroir(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let periode = 2 * (n as i64 - 1);
    let m = i.rem_euclid(periode);
    (if m < n as i64 { m } else { periode - m }) as usize
}

/// Volume interpolable par B-spline cubique : stocke les coefficients (volume préfiltré) et rend la valeur en toute coordonnée de voxel
/// (centres de voxels aux entiers). Hors du volume (au-delà d'un demi-voxel des bords), la valeur est 0 : la source est entourée de fond.
pub(crate) struct BSpline3 {
    coefficients: Array3<f64>,
}

impl BSpline3 {
    /// Calcule les coefficients du volume `donnees` (préfiltre le long de chacun des trois axes).
    pub(crate) fn new(donnees: &Array3<f32>) -> BSpline3 {
        let mut c = donnees.mapv(f64::from);
        for axe in 0..3 {
            for mut ligne in c.lanes_mut(ndarray::Axis(axe)) {
                let mut tampon: Vec<f64> = ligne.iter().copied().collect();
                prefiltre_1d(&mut tampon);
                for (dest, v) in ligne.iter_mut().zip(tampon) {
                    *dest = v;
                }
            }
        }
        BSpline3 { coefficients: c }
    }

    /// Valeur interpolée en la coordonnée de voxel `x` ; 0 si `x` est hors du volume.
    pub(crate) fn echantillon(&self, x: [f64; 3]) -> f64 {
        let dim = self.coefficients.dim();
        let n = [dim.0, dim.1, dim.2];
        let mut base = [0_i64; 3];
        let mut poids = [[0.0_f64; 4]; 3];
        for a in 0..3 {
            if x[a] < -0.5 || x[a] > n[a] as f64 - 0.5 {
                return 0.0;
            }
            let f = x[a].floor();
            base[a] = f as i64;
            poids[a] = poids_cubiques(x[a] - f);
        }
        let mut somme = 0.0;
        for (di, wi) in poids[0].iter().enumerate() {
            let i = miroir(base[0] + di as i64 - 1, n[0]);
            for (dj, wj) in poids[1].iter().enumerate() {
                let j = miroir(base[1] + dj as i64 - 1, n[1]);
                for (dk, wk) in poids[2].iter().enumerate() {
                    let k = miroir(base[2] + dk as i64 - 1, n[2]);
                    somme += wi * wj * wk * self.coefficients[[i, j, k]];
                }
            }
        }
        somme
    }
}

/// Produit `sinc(z / w) · Hann(z ; demi-largeur h)` (fenêtre de Hann : continue, nulle en `±h`, sans coupure brutale du support) : forme des noyaux du simulateur. `sinc(x) = sin(πx) / (πx)`.
fn sinc_fenetre(z: f64, w: f64, h: f64) -> f64 {
    if z.abs() >= h {
        return 0.0;
    }
    let x = z / w;
    let sinc = if x.abs() < 1e-12 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    };
    sinc * (0.5 + 0.5 * (std::f64::consts::PI * z / h).cos())
}

/// Largeur à mi-hauteur du produit [`sinc_fenetre`] : le profil décroît de 1 à 0 sur `[0, w]` (premier zéro du sinc, `w < h`), bissection du point à 1/2.
fn largeur_a_mi_hauteur(w: f64, h: f64) -> f64 {
    let (mut a, mut b) = (0.0, w);
    for _ in 0..100 {
        let m = 0.5 * (a + b);
        if sinc_fenetre(m, w, h) > 0.5 {
            a = m;
        } else {
            b = m;
        }
    }
    a + b
}

/// Profil de coupe : décalages `z` (mm, de `−2·épaisseur` à `+2·épaisseur` par pas `pas`) le long de la normale, et poids de somme 1.
/// Forme : sinc fenêtré (Hann sur ± 2 épaisseurs) dont la largeur est ajustée (bissection) pour que la largeur à mi-hauteur vaille
/// `epaisseur` ; ce n'est PAS la gaussienne du modèle direct de la reconstruction (lobes secondaires négatifs, support fini).
pub(crate) fn profil_de_coupe(epaisseur: f64, pas: f64) -> Vec<(f64, f64)> {
    let h = 2.0 * epaisseur;
    let (mut bas, mut haut) = (0.1 * epaisseur, 1.9 * epaisseur);
    for _ in 0..100 {
        let w = 0.5 * (bas + haut);
        if largeur_a_mi_hauteur(w, h) < epaisseur {
            bas = w;
        } else {
            haut = w;
        }
    }
    let w = 0.5 * (bas + haut);
    let n = (h / pas).floor() as i64;
    let mut profil: Vec<(f64, f64)> = (-n..=n)
        .map(|m| {
            let z = m as f64 * pas;
            (z, sinc_fenetre(z, w, h))
        })
        .collect();
    let somme: f64 = profil.iter().map(|&(_, p)| p).sum();
    for (_, p) in &mut profil {
        *p /= somme;
    }
    profil
}

/// Noyau 1D de résolution dans le plan, sur une grille `sur_echantillonnage` fois plus fine que le pixel de `pixel` mm : `sinc(u / pixel)`
/// (passe-bas idéal de bande 1 / (2·pixel), équivalent en espace réel de la troncature de l'espace de Fourier de l'acquisition) fenêtré
/// par Hann sur ± 3 pixels, de somme 1. Renvoie `6 · sur_echantillonnage + 1` poids, pour les décalages `m · pixel / sur_echantillonnage`.
pub(crate) fn noyau_dans_le_plan(pixel: f64, sur_echantillonnage: usize) -> Vec<f64> {
    let n = 3 * sur_echantillonnage as i64;
    let mut noyau: Vec<f64> = (-n..=n)
        .map(|m| sinc_fenetre(m as f64 * pixel / sur_echantillonnage as f64, pixel, 3.0 * pixel + 1e-9))
        .collect();
    let somme: f64 = noyau.iter().sum();
    for v in &mut noyau {
        *v /= somme;
    }
    noyau
}

/// Tirage gaussien centré réduit (Box–Muller) à partir du générateur `Alea` du banc.
pub(crate) fn gaussienne(alea: &mut crate::mouvement::Alea) -> f64 {
    let (u, v) = (alea.suivant().max(1e-12), alea.suivant());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

/// Bruit de Rice : le module d'un signal complexe `s` dont les parties réelle et imaginaire portent chacune un bruit gaussien d'écart type
/// `sigma`, `√((s + σ n₁)² + (σ n₂)²)`. Biaise vers le haut les faibles intensités (pas un bruit additif gaussien).
pub(crate) fn bruit_de_rice(signal: f64, sigma: f64, alea: &mut crate::mouvement::Alea) -> f64 {
    let a = signal + sigma * gaussienne(alea);
    let b = sigma * gaussienne(alea);
    (a * a + b * b).sqrt()
}

/// Réglages de bruit, de biais et d'échelle du simulateur (tous en %, 0 : désactivé).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReglagesSimulation {
    /// Écart type du bruit de Rice, en % de l'intensité moyenne du tissu de la source.
    pub bruit_pct: f64,
    /// Amplitude du biais multiplicatif lisse (gain global et quatre ondes planes de 100 à 200 mm), fixe par pile dans le repère du scanner.
    pub biais_pct: f64,
    /// Écart type, en %, du logarithme du facteur d'échelle de chaque coupe (log-normal) : effet d'histoire de spin ou de gain d'une coupe.
    pub echelle_coupe_pct: f64,
    /// Graine de tout le tirage aléatoire (bruit, biais, échelles).
    pub graine: u64,
}

/// Mélange de bits (SplitMix64) : une graine par (réglage, pile, coupe), indépendante de l'ordre d'exécution des fils.
fn melange(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Graine non nulle d'un générateur `Alea` pour la coupe `coupe` de la pile `pile` (`etiquette` distingue bruit, biais, échelles).
fn graine_de(graine: u64, etiquette: u64, pile: usize, coupe: usize) -> u64 {
    melange(melange(melange(graine ^ etiquette.wrapping_mul(0x1234_5678_9ABC_DEF1)) ^ pile as u64) ^ coupe as u64) | 1
}

/// Biais multiplicatif lisse d'une pile : `gain · (1 + a · Σ cos(2π x·d / λ + φ))` sur quatre ondes planes de longueur d'onde 100 à 200 mm,
/// avec `a = biais_pct / 200` ; `gain = 1 + biais_pct/100 · u`, `u ∈ [−1, 1]`. Fixe dans le repère du scanner (indépendant du mouvement de la coupe).
pub(crate) struct ChampDeBiais {
    gain: f64,
    amplitude: f64,
    ondes: Vec<(Vector3<f64>, f64, f64)>,
}

impl ChampDeBiais {
    /// Champ de la pile `pile` ; `biais_pct ≤ 0` : champ identiquement égal à 1.
    pub(crate) fn new(biais_pct: f64, graine: u64, pile: usize) -> ChampDeBiais {
        if biais_pct <= 0.0 {
            return ChampDeBiais {
                gain: 1.0,
                amplitude: 0.0,
                ondes: vec![],
            };
        }
        let mut a = Alea(graine_de(graine, 2, pile, 0));
        let mut tire = |bas: f64, haut: f64| bas + (haut - bas) * a.suivant();
        let ondes = (0..4)
            .map(|_| {
                let d = Vector3::new(tire(-1.0, 1.0), tire(-1.0, 1.0), tire(-1.0, 1.0)).normalize();
                (d, tire(100.0, 200.0), tire(0.0, 2.0 * std::f64::consts::PI))
            })
            .collect();
        ChampDeBiais {
            gain: 1.0 + biais_pct / 100.0 * tire(-1.0, 1.0),
            amplitude: biais_pct / 200.0,
            ondes,
        }
    }

    /// Valeur du champ au point `x` du monde (mm).
    pub(crate) fn valeur(&self, x: &Vector3<f64>) -> f64 {
        self.gain
            * (1.0
                + self
                    .ondes
                    .iter()
                    .map(|(d, l, p)| self.amplitude * (2.0 * std::f64::consts::PI * x.dot(d) / l + p).cos())
                    .sum::<f64>())
    }
}

/// Facteurs d'échelle des `n` coupes d'une pile : `exp(σ · g)`, `g` gaussien centré réduit, `σ = echelle_coupe_pct / 100` (tous égaux à 1 si `σ = 0`).
pub(crate) fn echelles_par_coupe(n: usize, r: &ReglagesSimulation, pile: usize) -> Vec<f64> {
    let sigma = r.echelle_coupe_pct / 100.0;
    (0..n)
        .map(|k| {
            if sigma <= 0.0 {
                1.0
            } else {
                (sigma * gaussienne(&mut Alea(graine_de(r.graine, 3, pile, k)))).exp()
            }
        })
        .collect()
}

/// Intensité moyenne du tissu de la source (voxels de plus de 3 % du maximum), référence du niveau de bruit (comme le banc E2).
pub(crate) fn intensite_moyenne_du_tissu(source: &Volume) -> f64 {
    let maximum = source.data().iter().copied().fold(0.0_f32, f32::max);
    let (somme, n) = source
        .data()
        .iter()
        .filter(|&&v| v > 0.03 * maximum)
        .fold((0.0_f64, 0usize), |(s, n), &v| (s + f64::from(v), n + 1));
    somme / n.max(1) as f64
}

/// Sur-échantillonnage dans le plan de la grille fine intermédiaire (deux sous-pixels par pixel, par axe).
const SUR_ECHANTILLONNAGE: usize = 2;

/// Valeurs SANS bruit, biais ni échelle de la coupe `coupe` (à sa pose, mouvement compris) : moyenne de la source (interpolée en B-spline cubique)
/// sur le profil de coupe le long de la normale, puis sur le noyau dans le plan, évalué sur une grille fine de demi-pixels.
fn coupe_propre(
    source: &Volume,
    spline: &BSpline3,
    coupe: &Slice,
    profil: &[(f64, f64)],
    noyau_x: &[f64],
    noyau_y: &[f64],
) -> Array2<f64> {
    let (nx, ny) = coupe.dim();
    let demi = 3 * SUR_ECHANTILLONNAGE;
    let (fx, fy) = (
        SUR_ECHANTILLONNAGE * (nx - 1) + 2 * demi + 1,
        SUR_ECHANTILLONNAGE * (ny - 1) + 2 * demi + 1,
    );
    let normale = coupe.normal();
    let mut fine = Array2::<f64>::zeros((fx, fy));
    for ((a, b), v) in fine.indexed_iter_mut() {
        let centre = coupe.pixel_to_world(
            (a as f64 - demi as f64) / SUR_ECHANTILLONNAGE as f64,
            (b as f64 - demi as f64) / SUR_ECHANTILLONNAGE as f64,
        );
        *v = profil
            .iter()
            .map(|&(z, w)| {
                let x = source.voxel_coordinates(&(centre + z * normale));
                w * spline.echantillon([x.x, x.y, x.z])
            })
            .sum();
    }
    let mut milieu = Array2::<f64>::zeros((nx, fy));
    for ((i, b), v) in milieu.indexed_iter_mut() {
        *v = noyau_x
            .iter()
            .enumerate()
            .map(|(m, w)| w * fine[[SUR_ECHANTILLONNAGE * i + m, b]])
            .sum();
    }
    let mut sortie = Array2::<f64>::zeros((nx, ny));
    for ((i, j), v) in sortie.indexed_iter_mut() {
        *v = noyau_y
            .iter()
            .enumerate()
            .map(|(m, w)| w * milieu[[i, SUR_ECHANTILLONNAGE * j + m]])
            .sum();
    }
    sortie
}

/// **Simule une pile** : géométrie, résolution et épaisseur de `modele` ; mouvement vrai `poses` (pile numéro `pile`) ; biais, échelle de coupe
/// et bruit de Rice selon `r`. Renvoie les valeurs et le masque (pixel dont la valeur sans bruit dépasse 3 % du maximum de la source). Déterministe :
/// les graines dépendent de (réglage, pile, coupe), jamais de l'ordre d'exécution des fils.
pub(crate) fn simuler_pile(
    source: &Volume,
    spline: &BSpline3,
    modele: &Stack,
    pile: usize,
    poses: &SlicePoses,
    r: &ReglagesSimulation,
    intensite_moyenne: f64,
) -> (Array3<f32>, Array3<u8>) {
    let dim = modele.dim();
    let s = modele.spacing();
    let profil = profil_de_coupe(s[2], s[2] / 4.0);
    let (noyau_x, noyau_y) = (
        noyau_dans_le_plan(s[0], SUR_ECHANTILLONNAGE),
        noyau_dans_le_plan(s[1], SUR_ECHANTILLONNAGE),
    );
    let maximum = f64::from(source.data().iter().copied().fold(0.0_f32, f32::max));
    let champ = ChampDeBiais::new(r.biais_pct, r.graine, pile);
    let echelles = echelles_par_coupe(dim.2, r, pile);
    let sigma_bruit = r.bruit_pct / 100.0 * intensite_moyenne;
    let coupes: Vec<(Array2<f32>, Array2<u8>)> = (0..dim.2)
        .into_par_iter()
        .map(|k| {
            let nominale = modele.slice(k).expect("indice de coupe valide");
            let posee = nominale.with_motion(*poses.get(pile, k));
            let propre = coupe_propre(source, spline, &posee, &profil, &noyau_x, &noyau_y);
            let mut alea = Alea(graine_de(r.graine, 1, pile, k));
            let mut valeurs = Array2::<f32>::zeros((dim.0, dim.1));
            let mut masque = Array2::<u8>::zeros((dim.0, dim.1));
            for ((i, j), v) in valeurs.indexed_iter_mut() {
                let propre_ij = propre[[i, j]];
                masque[[i, j]] = u8::from(propre_ij > 0.03 * maximum);
                let biais = champ.valeur(&nominale.pixel_to_world(i as f64, j as f64));
                let signal = propre_ij * biais * echelles[k];
                *v = if sigma_bruit > 0.0 {
                    bruit_de_rice(signal, sigma_bruit, &mut alea)
                } else {
                    signal
                } as f32;
            }
            (valeurs, masque)
        })
        .collect();
    let (mut data, mut masque) = (Array3::<f32>::zeros(dim), Array3::<u8>::zeros(dim));
    for (k, (v, m)) in coupes.into_iter().enumerate() {
        data.index_axis_mut(ndarray::Axis(2), k).assign(&v);
        masque.index_axis_mut(ndarray::Axis(2), k).assign(&m);
    }
    (data, masque)
}

/// En-tête NIfTI d'une pile : affine voxel → monde (sform) et pas des voxels.
fn en_tete(affine: &Matrix4<f64>, pixdim: [f64; 3]) -> NiftiHeader {
    let mut h = NiftiHeader::default();
    h.sform_code = 1;
    h.srow_x = std::array::from_fn(|c| affine[(0, c)] as f32);
    h.srow_y = std::array::from_fn(|c| affine[(1, c)] as f32);
    h.srow_z = std::array::from_fn(|c| affine[(2, c)] as f32);
    h.pixdim = [
        1.0,
        pixdim[0] as f32,
        pixdim[1] as f32,
        pixdim[2] as f32,
        1.0,
        1.0,
        1.0,
        1.0,
    ];
    h
}

/// Écrit une pile (image `f32`) et son masque (`u8`) en NIfTI, avec l'affine donnée ; relit la pile avec son masque attaché.
///
/// # Erreurs
/// Erreur d'écriture ou de relecture, avec le chemin en cause.
pub(crate) fn ecrire_pile(
    image: &Path,
    masque_chemin: &Path,
    data: &Array3<f32>,
    masque: &Array3<u8>,
    affine: &Matrix4<f64>,
    pixdim: [f64; 3],
) -> Result<Stack, SvrError> {
    let h = en_tete(affine, pixdim);
    WriterOptions::new(image)
        .reference_header(&h)
        .write_nifti(data)
        .map_err(|e| SvrError::Write {
            path: image.to_path_buf(),
            message: e.to_string(),
        })?;
    WriterOptions::new(masque_chemin)
        .reference_header(&h)
        .write_nifti(masque)
        .map_err(|e| SvrError::Write {
            path: masque_chemin.to_path_buf(),
            message: e.to_string(),
        })?;
    let mut stack = Stack::read(image)?;
    stack.set_brain_mask(masque_chemin)?;
    Ok(stack)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alea(graine: u64) -> impl FnMut() -> f64 {
        let mut a = crate::mouvement::Alea(graine);
        move || a.suivant()
    }

    /// Aux nœuds (coordonnées entières), l'interpolation redonne les données : propriété du préfiltre (inverse exact du noyau (1, 4, 1)/6).
    #[test]
    fn the_bspline_interpolates_the_samples_at_the_nodes() {
        let mut u = alea(3);
        let donnees = Array3::from_shape_fn((24, 25, 26), |_| u() as f32);
        let spline = BSpline3::new(&donnees);
        for &(i, j, k) in &[(0, 0, 0), (23, 24, 25), (5, 7, 11), (12, 12, 12), (1, 24, 0)] {
            let v = spline.echantillon([i as f64, j as f64, k as f64]);
            assert!(
                (v - f64::from(donnees[[i, j, k]])).abs() < 1e-9,
                "nœud ({i}, {j}, {k}) : {v}"
            );
        }
    }

    /// Un polynôme de degré 3 par axe est reproduit exactement loin des bords (les bords en miroir ne le sont pas : erreur en `|z|^distance`).
    #[test]
    fn the_bspline_reproduces_a_cubic_polynomial_away_from_the_borders() {
        let f = |x: f64, y: f64, z: f64| 0.01 * x * x * x - 0.02 * y * y * z + 0.5 * z + 0.003 * x * y * z + 7.0;
        let donnees = Array3::from_shape_fn((48, 48, 48), |(i, j, k)| f(i as f64, j as f64, k as f64) as f32);
        // Les données sont arrondies en f32 : la tolérance suit (valeurs jusqu'à ≈ 1000, epsilon relatif 6e-8).
        let spline = BSpline3::new(&donnees);
        for &(x, y, z) in &[
            (22.3, 24.7, 25.1),
            (26.5, 21.2, 23.9),
            (24.0, 24.5, 24.25),
            (21.9, 27.7, 22.1),
        ] {
            let v = spline.echantillon([x, y, z]);
            assert!(
                (v - f(x, y, z)).abs() < 5e-3,
                "({x}, {y}, {z}) : {v} contre {}",
                f(x, y, z)
            );
        }
    }

    /// Une constante reste constante, bords compris ; et hors du volume la valeur est nulle.
    #[test]
    fn the_bspline_keeps_a_constant_and_is_zero_outside() {
        let spline = BSpline3::new(&Array3::from_elem((24, 24, 24), 3.5_f32));
        for &x in &[
            [0.0, 0.0, 0.0],
            [23.0, 23.0, 23.0],
            [-0.4, 11.3, 5.5],
            [11.5, 11.5, 23.4],
        ] {
            assert!((spline.echantillon(x) - 3.5).abs() < 1e-9, "{x:?}");
        }
        assert_eq!(spline.echantillon([-0.6, 5.0, 5.0]), 0.0);
        assert_eq!(spline.echantillon([5.0, 5.0, 23.6]), 0.0);
    }

    /// Les quatre poids forment une partition de l'unité et sont positifs.
    #[test]
    fn the_cubic_weights_form_a_partition_of_unity() {
        for n in 0..=20 {
            let t = f64::from(n) / 20.0;
            let w = poids_cubiques(t);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            assert!(w.iter().all(|&v| v >= 0.0));
        }
    }

    /// Largeur à mi-hauteur d'un profil échantillonné, par interpolation linéaire des deux points qui encadrent 1/2 du maximum.
    fn fwhm_echantillonnee(profil: &[(f64, f64)]) -> f64 {
        let max = profil.iter().map(|&(_, p)| p).fold(f64::MIN, f64::max);
        let centre = profil.iter().position(|&(_, p)| p == max).unwrap();
        let croisement = |sens: i64| {
            let mut i = centre as i64;
            while profil[(i + sens) as usize].1 > 0.5 * max {
                i += sens;
            }
            let (z0, p0) = profil[i as usize];
            let (z1, p1) = profil[(i + sens) as usize];
            z0 + (0.5 * max - p0) / (p1 - p0) * (z1 - z0)
        };
        croisement(1) - croisement(-1)
    }

    /// Le profil de coupe est normalisé, symétrique, de largeur à mi-hauteur égale à l'épaisseur, et à lobes négatifs (ce n'est pas une gaussienne).
    #[test]
    fn the_slice_profile_is_normalised_symmetric_with_the_requested_fwhm() {
        for &epaisseur in &[2.0, 3.0, 3.5] {
            let fin = profil_de_coupe(epaisseur, epaisseur / 64.0);
            assert!((fin.iter().map(|&(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
            for i in 0..fin.len() {
                assert!((fin[i].1 - fin[fin.len() - 1 - i].1).abs() < 1e-14, "symétrie");
            }
            let l = fwhm_echantillonnee(&fin);
            assert!(
                (l / epaisseur - 1.0).abs() < 0.01,
                "épaisseur {epaisseur} : largeur {l}"
            );
            let min = fin.iter().map(|&(_, p)| p).fold(f64::MAX, f64::min);
            let max = fin.iter().map(|&(_, p)| p).fold(f64::MIN, f64::max);
            assert!(min < -0.005 * max, "lobes secondaires négatifs attendus : {min}");
        }
    }

    /// Le pas utilisé en production (épaisseur / 4) donne la même variance que le pas fin (épaisseur / 64) à 2 % près.
    #[test]
    fn the_production_step_of_the_slice_profile_is_fine_enough() {
        let variance = |p: &[(f64, f64)]| p.iter().map(|&(z, w)| w * z * z).sum::<f64>();
        let (grossier, fin) = (profil_de_coupe(3.0, 0.75), profil_de_coupe(3.0, 3.0 / 64.0));
        assert!((variance(&grossier) / variance(&fin) - 1.0).abs() < 0.02);
    }

    /// Réponse en fréquence du noyau dans le plan (DFT écrite à part) : gain ≈ 1 en bande passante, ≈ 0 au-delà de la fréquence de Nyquist du pixel.
    #[test]
    fn the_in_plane_kernel_is_a_low_pass_at_the_pixel_nyquist_frequency() {
        let (pixel, os) = (0.8, 2);
        let noyau = noyau_dans_le_plan(pixel, os);
        assert_eq!(noyau.len(), 6 * os + 1);
        assert!((noyau.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let gain = |f: f64| {
            // f en cycles par mm ; décalages m · pixel / os
            let (mut re, mut im) = (0.0, 0.0);
            for (i, w) in noyau.iter().enumerate() {
                let u = (i as f64 - 3.0 * os as f64) * pixel / os as f64;
                re += w * (2.0 * std::f64::consts::PI * f * u).cos();
                im += w * (2.0 * std::f64::consts::PI * f * u).sin();
            }
            (re * re + im * im).sqrt()
        };
        assert!(gain(0.0) > 0.999999);
        assert!(gain(0.15 / pixel) > 0.97, "bande passante : {}", gain(0.15 / pixel));
        assert!(gain(0.85 / pixel) < 0.03, "bande coupée : {}", gain(0.85 / pixel));
    }

    /// Bruit de Rice : sans signal, `E[R] = σ √(π/2)` et `E[R²] = 2σ²` ; avec un signal fort, `E[R²] = s² + 2σ²` (valeurs calculées à la main).
    #[test]
    fn rician_noise_has_the_analytic_moments() {
        let n = 200_000;
        let mut alea = crate::mouvement::Alea(12345);
        let sigma = 2.0;
        let r0: Vec<f64> = (0..n).map(|_| bruit_de_rice(0.0, sigma, &mut alea)).collect();
        let moyenne = r0.iter().sum::<f64>() / n as f64;
        let e2 = r0.iter().map(|r| r * r).sum::<f64>() / n as f64;
        assert!(
            (moyenne / (sigma * (std::f64::consts::PI / 2.0).sqrt()) - 1.0).abs() < 0.01,
            "{moyenne}"
        );
        assert!((e2 / (2.0 * sigma * sigma) - 1.0).abs() < 0.01, "{e2}");
        let s = 10.0 * sigma;
        let e2s = (0..n).map(|_| bruit_de_rice(s, sigma, &mut alea).powi(2)).sum::<f64>() / n as f64;
        assert!((e2s / (s * s + 2.0 * sigma * sigma) - 1.0).abs() < 0.005, "{e2s}");
    }

    /// Même graine, même bruit, bit à bit.
    #[test]
    fn rician_noise_is_deterministic() {
        let tirer = |g| {
            let mut a = crate::mouvement::Alea(g);
            (0..10)
                .map(|_| bruit_de_rice(5.0, 1.0, &mut a).to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(tirer(9), tirer(9));
        assert_ne!(tirer(9), tirer(10));
    }

    /// Dossier temporaire propre à un test (supprimé à la fin).
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new(nom: &str) -> Temp {
            let d = std::env::temp_dir().join(format!("medoxide_simulation_{nom}_{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Temp(d)
        }
        fn fichier(&self, nom: &str) -> std::path::PathBuf {
            self.0.join(nom)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const NUL: ReglagesSimulation = ReglagesSimulation {
        bruit_pct: 0.0,
        biais_pct: 0.0,
        echelle_coupe_pct: 0.0,
        graine: 5,
    };

    /// Source : rampe `100 + 0,5 x + 0,3 y + 0,2 z` (mm, monde) sur 80³ voxels de 1 mm, origine en (−40, −40, −40) ; la pile modèle : 16 × 16 × 8 pixels
    /// de 1,5 × 1,5 × 3 mm, oblique, centrée à l'origine. Renvoie (source, pile, rampe).
    fn rampe_et_pile(t: &Temp) -> (Volume, Stack, impl Fn(&Vector3<f64>) -> f64) {
        let rampe = |x: &Vector3<f64>| 100.0 + 0.5 * x.x + 0.3 * x.y + 0.2 * x.z;
        let mut a = Matrix4::identity();
        a[(0, 3)] = -40.0;
        a[(1, 3)] = -40.0;
        a[(2, 3)] = -40.0;
        let data = Array3::from_shape_fn((80, 80, 80), |(i, j, k)| {
            rampe(&Vector3::new(i as f64 - 40.0, j as f64 - 40.0, k as f64 - 40.0)) as f32
        });
        let source = Volume::new(data, a).unwrap();
        let r = nalgebra::Rotation3::from_euler_angles(0.2, -0.15, 0.1);
        let lineaire = r.matrix() * nalgebra::Matrix3::from_diagonal(&Vector3::new(1.5, 1.5, 3.0));
        let origine = -(lineaire * Vector3::new(7.5, 7.5, 3.5));
        let mut affine = Matrix4::identity();
        affine.fixed_view_mut::<3, 3>(0, 0).copy_from(&lineaire);
        affine.fixed_view_mut::<3, 1>(0, 3).copy_from(&origine);
        let pile = ecrire_pile(
            &t.fichier("modele.nii.gz"),
            &t.fichier("modele_mask.nii.gz"),
            &Array3::zeros((16, 16, 8)),
            &Array3::from_elem((16, 16, 8), 1u8),
            &affine,
            [1.5, 1.5, 3.0],
        )
        .unwrap();
        (source, pile, rampe)
    }

    /// Sans mouvement ni bruit, chaque pixel simulé vaut la rampe au centre du pixel : la B-spline reproduit une fonction linéaire et les noyaux
    /// symétriques de somme 1 la laissent inchangée au centre.
    #[test]
    fn a_simulated_stack_of_a_linear_ramp_gives_the_ramp_at_the_pixel_centres() {
        let t = Temp::new("rampe");
        let (source, pile, rampe) = rampe_et_pile(&t);
        let spline = BSpline3::new(source.data());
        let (data, masque) = simuler_pile(
            &source,
            &spline,
            &pile,
            0,
            &SlicePoses::identity(std::slice::from_ref(&pile)),
            &NUL,
            100.0,
        );
        for &(i, j, k) in &[(0, 0, 0), (7, 8, 3), (15, 15, 7), (3, 12, 5)] {
            let attendu = rampe(&pile.slice(k).unwrap().pixel_to_world(i as f64, j as f64));
            assert!(
                (f64::from(data[[i, j, k]]) - attendu).abs() < 1e-3,
                "({i}, {j}, {k}) : {} contre {attendu}",
                data[[i, j, k]]
            );
        }
        assert!(masque.iter().all(|&m| m == 1));
    }

    /// Avec un mouvement par coupe, la valeur est la rampe à la position MOUVANTE du pixel (`pixel_to_world` de la coupe posée), et le modèle direct
    /// de la reconstruction (`Volume::simulate_slice`) donne la même valeur sur cette fonction linéaire : les deux modèles partagent la convention de pose.
    #[test]
    fn the_motion_convention_agrees_with_the_reconstruction_forward_model() {
        let t = Temp::new("pose");
        let (source, pile, rampe) = rampe_et_pile(&t);
        let spline = BSpline3::new(source.data());
        let mut poses = SlicePoses::identity(std::slice::from_ref(&pile));
        for k in 0..8 {
            poses.set(
                0,
                k,
                crate::mouvement::mouvement_rigide(
                    &Vector3::new(0.03 * k as f64, -0.02, 0.04),
                    &Vector3::new(1.0 - 0.3 * k as f64, 0.5, -0.7),
                    &Vector3::new(0.0, 0.0, 0.0),
                ),
            );
        }
        let (data, _) = simuler_pile(&source, &spline, &pile, 0, &poses, &NUL, 100.0);
        for k in [0, 3, 7] {
            let posee = pile.slice(k).unwrap().with_motion(*poses.get(0, k));
            let direct = source.simulate_slice(&posee, &posee.psf());
            for &(i, j) in &[(2, 3), (8, 8), (13, 5)] {
                let attendu = rampe(&posee.pixel_to_world(i as f64, j as f64));
                assert!(
                    (f64::from(data[[i, j, k]]) - attendu).abs() < 1e-3,
                    "indépendant ({i}, {j}, {k})"
                );
                assert!(
                    (f64::from(direct.values[[i, j]]) - attendu).abs() < 1e-2,
                    "modèle direct ({i}, {j}, {k})"
                );
            }
        }
    }

    /// Source constante : le bruit de Rice du simulateur a le niveau demandé (`bruit_pct` % de l'intensité moyenne) ; deux appels identiques donnent les mêmes bits.
    #[test]
    fn the_simulated_noise_has_the_requested_level_and_is_deterministic() {
        let t = Temp::new("bruit");
        let (source, pile, _) = rampe_et_pile(&t);
        let constante = Volume::new(Array3::from_elem((80, 80, 80), 100.0_f32), *source.affine()).unwrap();
        let spline = BSpline3::new(constante.data());
        let poses = SlicePoses::identity(std::slice::from_ref(&pile));
        let r = ReglagesSimulation { bruit_pct: 5.0, ..NUL };
        let (a, _) = simuler_pile(&constante, &spline, &pile, 0, &poses, &r, 100.0);
        let (b, _) = simuler_pile(&constante, &spline, &pile, 0, &poses, &r, 100.0);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()));
        let n = a.len() as f64;
        let e2 = a.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / n;
        assert!((e2 / (100.0 * 100.0 + 2.0 * 25.0) - 1.0).abs() < 0.01, "E[R²] = {e2}");
        let moy = a.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = a.iter().map(|&v| (f64::from(v) - moy).powi(2)).sum::<f64>() / n;
        assert!((var / 25.0 - 1.0).abs() < 0.12, "variance {var}");
        let (c, _) = simuler_pile(
            &constante,
            &spline,
            &pile,
            0,
            &poses,
            &ReglagesSimulation { graine: 6, ..r },
            100.0,
        );
        assert!(a.iter().zip(c.iter()).any(|(x, y)| x.to_bits() != y.to_bits()));
    }

    /// Source constante, sans bruit : l'échelle de chaque coupe est constante dans la coupe et varie d'une coupe à l'autre ; le biais est borné et lisse.
    #[test]
    fn slice_scales_and_the_bias_field_act_as_declared() {
        let t = Temp::new("echelles");
        let (source, pile, _) = rampe_et_pile(&t);
        let constante = Volume::new(Array3::from_elem((80, 80, 80), 100.0_f32), *source.affine()).unwrap();
        let spline = BSpline3::new(constante.data());
        let poses = SlicePoses::identity(std::slice::from_ref(&pile));
        let r = ReglagesSimulation {
            echelle_coupe_pct: 5.0,
            ..NUL
        };
        let (data, _) = simuler_pile(&constante, &spline, &pile, 0, &poses, &r, 100.0);
        let echelles = echelles_par_coupe(8, &r, 0);
        for k in 0..8 {
            for &(i, j) in &[(0, 0), (8, 8), (15, 3)] {
                assert!((f64::from(data[[i, j, k]]) - 100.0 * echelles[k]).abs() < 1e-3);
            }
        }
        assert!(echelles.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-4));
        let moyenne_log = echelles_par_coupe(20_000, &r, 1).iter().map(|e| e.ln()).sum::<f64>() / 20_000.0;
        assert!(moyenne_log.abs() < 2e-3, "moyenne du log {moyenne_log}");
        assert!(echelles_par_coupe(5, &NUL, 0).iter().all(|&e| e == 1.0));

        let champ = ChampDeBiais::new(10.0, 5, 0);
        let (mut mini, mut maxi) = (f64::MAX, f64::MIN);
        let mut a = Alea(77);
        for _ in 0..2000 {
            let x = Vector3::new(
                100.0 * (a.suivant() - 0.5),
                100.0 * (a.suivant() - 0.5),
                100.0 * (a.suivant() - 0.5),
            );
            let v = champ.valeur(&x);
            mini = mini.min(v);
            maxi = maxi.max(v);
            // lisse : un déplacement de 1 mm change le champ d'au plus 4 ondes × (a · 2π / 100 mm) en valeur relative
            assert!(
                (champ.valeur(&(x + Vector3::new(1.0, 0.0, 0.0))) - v).abs()
                    < 4.0 * 0.05 * 2.0 * std::f64::consts::PI / 100.0 * 1.2
            );
        }
        assert!(mini > 0.7 && maxi < 1.3 && maxi - mini > 0.01, "bornes {mini}, {maxi}");
        assert_eq!(ChampDeBiais::new(0.0, 5, 0).valeur(&Vector3::new(3.0, 4.0, 5.0)), 1.0);
        assert_ne!(
            ChampDeBiais::new(10.0, 5, 0).valeur(&Vector3::zeros()),
            ChampDeBiais::new(10.0, 5, 1).valeur(&Vector3::zeros())
        );
    }

    /// L'écriture puis la relecture d'une pile redonne affine, pas des voxels, valeurs et masque.
    #[test]
    fn a_written_stack_is_read_back_unchanged() {
        let t = Temp::new("ecriture");
        let (_, pile, _) = rampe_et_pile(&t);
        assert_eq!(pile.dim(), (16, 16, 8));
        let s = pile.spacing();
        assert!((s[0] - 1.5).abs() < 1e-5 && (s[1] - 1.5).abs() < 1e-5 && (s[2] - 3.0).abs() < 1e-5);
        assert_eq!(pile.brain_mask().unwrap().voxel_count(), 16 * 16 * 8);
    }

    /// Intensité moyenne du tissu, calculée à la main : 10 voxels à 100 et 20 à 150 (tissu : plus de 3 % du maximum 150, soit 4,5), 70 voxels à 4 (sous le seuil) : (10·100 + 20·150) / 30.
    #[test]
    fn the_mean_tissue_intensity_ignores_the_background() {
        let mut data = Array3::from_elem((10, 10, 1), 4.0_f32);
        for (n, v) in data.iter_mut().enumerate() {
            if n < 10 {
                *v = 100.0;
            } else if n < 30 {
                *v = 150.0;
            }
        }
        let source = Volume::new(data, Matrix4::identity()).unwrap();
        assert!((intensite_moyenne_du_tissu(&source) - (10.0 * 100.0 + 20.0 * 150.0) / 30.0).abs() < 1e-9);
    }
}
