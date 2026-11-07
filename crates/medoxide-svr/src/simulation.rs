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
}
