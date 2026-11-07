//! Simulateur d'acquisition INDÉPENDANT du modèle direct de la reconstruction (docs/ROADMAP.md, F4.0.1).
//!
//! Le banc E2 simule les coupes avec [`crate::Volume::simulate_slice`], le modèle même que la reconstruction inverse : un « crime inverse »
//! qui favorise la méthode partageant ces hypothèses. Ce module réécrit la physique d'acquisition avec d'autres choix : interpolation de la
//! source par B-spline cubique (au lieu de la trilinéaire), intégration d'un voxel de basse résolution sur une grille fine de la source,
//! profil de coupe en sinc fenêtré et résolution dans le plan par un sinc fenêtré (au lieu de noyaux gaussiens), bruit de Rice, biais lisse,
//! échelle par coupe. Il ne modifie aucun élément existant.

use ndarray::Array3;

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
}
