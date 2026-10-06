//! Coût **différentiable** du recalage des coupes (étape 3) : les calculs sont écrits en opérations sur
//! tenseurs Burn, pour que la différentiation automatique en tire les gradients.
//!
//! Burn choisit son backend à l'exécution par un `Device` : tous les tenseurs d'un calcul vivent sur le
//! même `Device`.

use burn::prelude::*;
use nalgebra::Vector3;

/// Résultat de [`trilinear_sample`] : une valeur et un indicateur « dans la grille » par point.
#[derive(Debug, Clone)]
pub struct TrilinearSample {
    /// Valeur interpolée de chaque point, forme `[P]`. Sans signification là où `inside` vaut 0.
    pub values: Tensor<1>,
    /// 1 si le point est dans la zone entre les centres du premier et du dernier voxel de chaque axe, 0 sinon.
    /// Non différentiable (calculé par des comparaisons).
    pub inside: Tensor<1>,
}

/// Interpolation **trilinéaire** d'un volume en `P` points donnés en indices de voxel continus, en opérations
/// sur tenseurs (même définition que [`crate::Volume::sample`], en `f32`).
///
/// - `volume` : voxels aplatis dans l'ordre d'un `Array3` standard, `index = (i · ny + j) · nz + k`, forme `[nx·ny·nz]`.
/// - `dims` : `[nx, ny, nz]`.
/// - `coords` : forme `[P, 3]`, indices continus `(i, j, k)`.
///
/// Un tenseur ne peut pas répondre « dehors » par `None` : les points hors de la grille sont ramenés au bord pour la
/// lecture et signalés par `inside = 0`. La différentiation passe par les **fractions de distance**
/// `t = c − floor(c)` (poids trilinéaires) ; les indices de voxel, entiers, ne sont pas différentiables.
///
/// # Panics
/// Si un axe de `dims` est nul ou si `volume` et `coords` ne sont pas sur le même `Device`.
pub fn trilinear_sample(volume: &Tensor<1>, dims: [usize; 3], coords: Tensor<2>) -> TrilinearSample {
    assert!(dims.iter().all(|&n| n > 0), "chaque axe doit avoir au moins un voxel");
    let p = coords.dims()[0];
    let device = coords.device();

    // Colonnes de `coords` : une coordonnée par axe, forme [P].
    let axe = |a: usize| coords.clone().narrow(1, a, 1).reshape([p]);
    let mut inside = Tensor::<1>::ones([p], &device);
    let mut bas: Vec<Tensor<1, Int>> = Vec::new(); // indice du voxel de départ, par axe
    let mut haut: Vec<Tensor<1, Int>> = Vec::new(); // indice du voxel suivant (même si l'axe n'a qu'un voxel)
    let mut frac: Vec<Tensor<1>> = Vec::new(); // fraction de distance dans [0, 1]
    for a in 0..3 {
        let c = axe(a);
        let max = (dims[a] - 1) as f64;
        // 1 si 0 <= c <= n-1 : produit des indicateurs des axes.
        let dedans = c.clone().greater_equal_elem(0.0).bool_and(c.clone().lower_equal_elem(max)).float();
        inside = inside * dedans;
        // Ramené au bord pour que la lecture reste valide ; les points dehors sont ignorés par `inside`.
        let c = c.clamp(0.0, max);
        let plancher = c.clone().floor();
        frac.push(c - plancher.clone());
        let i0 = plancher.int().clamp(0, dims[a] as i64 - 1);
        haut.push((i0.clone() + 1).clamp(0, dims[a] as i64 - 1));
        bas.push(i0);
    }

    // Les 8 coins du cube : indice aplati, poids = produit des (t ou 1-t) de chaque axe.
    let pas = [(dims[1] * dims[2]) as i64, dims[2] as i64, 1];
    let mut valeurs = Tensor::<1>::zeros([p], &device);
    for coin in 0..8usize {
        let mut indice = Tensor::<1, Int>::zeros([p], &device);
        let mut poids = Tensor::<1>::ones([p], &device);
        for a in 0..3 {
            let haut_a = (coin >> (2 - a)) & 1 == 1; // x est le bit de poids fort, comme l'ordre des boucles de `Volume`
            let (i, w) = if haut_a {
                (haut[a].clone(), frac[a].clone())
            } else {
                (bas[a].clone(), frac[a].clone().neg() + 1.0)
            };
            indice = indice + i * pas[a];
            poids = poids * w;
        }
        valeurs = valeurs + poids * volume.clone().select(0, indice);
    }
    TrilinearSample { values: valeurs, inside }
}

/// Au-dessous de ce `θ²` (rad²), [`rotation_matrix`] utilise la série de Taylor : en `f32`, `1 − cos θ` y perd presque toute
/// sa précision (20 % d'erreur relative à θ = 1e-3), alors que la série à 3 termes y est exacte à ≈ 1e-10.
const ROTATION_SERIES_THETA2: f64 = 1e-2;

/// Matrice de rotation 3×3 d'un **vecteur de rotation** `omega` (axe = direction, angle = norme, en radians), par la
/// formule de Rodrigues : `R = I + a K + b K²`, `K` matrice antisymétrique de `omega`, `a = sin θ / θ`, `b = (1 − cos θ) / θ²`.
///
/// Les deux fractions valent `0/0` en `omega = 0`, point de départ de tout recalage (pose = delta nul) : elles sont
/// remplacées par leur série de Taylor pour les petits angles. Dans une opération `mask_where`, la branche non retenue
/// est quand même dérivée par l'autodiff, et un `NaN` y contaminerait le gradient : la branche exacte reçoit donc un
/// argument « sûr » (1) là où la série est retenue.
///
/// `omega` : forme `[3]`. Rend une matrice de forme `[3, 3]` : `R · x` s'écrit `x.matmul(R.transpose())` pour des lignes `x`.
pub fn rotation_matrix(omega: Tensor<1>) -> Tensor<2> {
    let device = omega.device();
    // K = somme des omega_i · E_i, avec les trois générateurs E_i constants (forme [3, 3, 3]).
    let generateurs = Tensor::<3>::from_floats(
        TensorData::new(
            vec![
                0.0_f32, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, // E_x
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, // E_y
                0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // E_z
            ],
            [3, 3, 3],
        ),
        &device,
    );
    let k: Tensor<2> = (generateurs * omega.clone().reshape([3, 1, 1])).sum_dim(0).reshape([3, 3]);

    let th2 = (omega.clone() * omega).sum(); // forme [1]
    let petit = th2.clone().lower_elem(ROTATION_SERIES_THETA2);
    let th2_sur = th2.clone().mask_where(petit.clone(), Tensor::<1>::ones([1], &device)); // argument sûr de la branche exacte
    let th = th2_sur.clone().sqrt();
    let a_exacte = th.clone().sin() / th.clone();
    let b_exacte = (th.cos().neg() + 1.0) / th2_sur;
    let th4 = th2.clone() * th2.clone();
    let a_serie = th2.clone().neg() / 6.0 + th4.clone() / 120.0 + 1.0;
    let b_serie = th2.neg() / 24.0 + th4 / 720.0 + 0.5;
    let a = a_exacte.mask_where(petit.clone(), a_serie).reshape([1, 1]);
    let b = b_exacte.mask_where(petit, b_serie).reshape([1, 1]);

    let identite = Tensor::<2>::from_floats(TensorData::new(vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3]), &device);
    identite + a * k.clone() + b * k.clone().matmul(k)
}

/// Applique la **pose** `params` à des points du monde : `x' = c + R(ω) (x − c) + t`, rotation autour du pivot `c`.
///
/// - `points` : forme `[P, 3]`, mm ; `pivot` : forme `[3]`, mm (par exemple le pivot P3 de la coupe).
/// - `params` : forme `[6]` = `(φx, φy, φz, tx, ty, tz)`. `t` est en mm ; la rotation est exprimée en **mm équivalents** :
///   `ω = φ / rotation_scale_mm` (voir [`rotation_scale_mm`]), pour que les 6 paramètres aient le même ordre de grandeur de
///   sensibilité (un pas de 1 déplace un point typique d'environ 1 mm, quel que soit le paramètre).
///
/// Pose nulle = identité. Le mouvement est un **delta** autour de la pose d'en-tête de la coupe, composé à chaque pas.
pub fn apply_pose(points: Tensor<2>, pivot: Tensor<1>, params: Tensor<1>, rotation_scale_mm: f64) -> Tensor<2> {
    let phi = params.clone().narrow(0, 0, 3);
    let t = params.narrow(0, 3, 3).reshape([1, 3]);
    let r = rotation_matrix(phi / rotation_scale_mm);
    let pivot = pivot.reshape([1, 3]);
    (points - pivot.clone()).matmul(r.transpose()) + pivot + t
}

/// Échelle de la rotation, en mm : `√(2/3) · r_rms`, où `r_rms` est la distance quadratique moyenne des `points` au `pivot`.
///
/// Pour un axe unité `e`, un point à `d` du pivot se déplace de `‖e × d‖` ; en moyenne sur les trois axes, le carré vaut
/// `(2/3)‖d‖²`. Avec cette échelle, une unité de rotation `φ` déplace donc les points de **1 mm en moyenne quadratique**,
/// comme une unité de translation. `None` si la liste est vide ou si tous les points sont au pivot.
pub fn rotation_scale_mm(points: &[Vector3<f64>], pivot: &Vector3<f64>) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let moyenne = points.iter().map(|p| (p - pivot).norm_squared()).sum::<f64>() / points.len() as f64;
    let echelle = (2.0 / 3.0 * moyenne).sqrt();
    (echelle > 0.0).then_some(echelle)
}

/// Constante ajoutée sous la racine de [`ncc`] : une coupe constante ou sans pixel valide a une variance nulle, donc
/// `0/0` et un gradient `NaN` ; avec elle, la corrélation vaut 0 et le gradient reste fini. Négligeable devant des
/// variances d'intensité normales (≥ 1e-4).
const NCC_EPS: f64 = 1e-8;

/// **Corrélation normalisée** (NCC) pondérée, une par coupe : corrélation de Pearson entre `a` et `b` où chaque pixel
/// compte avec son poids `w`. Les moyennes sont pondérées elles aussi.
///
/// - `a` : intensités des coupes, forme `[S, P]` (S coupes, P pixels).
/// - `b` : intensités du volume échantillonné aux mêmes pixels, même forme.
/// - `w` : poids des pixels (par exemple masque cérébral × `inside` de [`trilinear_sample`]), même forme ; un poids nul
///   retire le pixel du calcul.
///
/// Rend un tenseur de forme `[S]`, de −1 à 1 : 1 si `b = α a + β` avec `α > 0` sur les pixels pesés (invariance aux changements
/// d'échelle et de décalage d'intensité de chaque image). Chaque coupe est indépendante : le gradient d'une coupe ne dépend
/// pas des autres. Valeur 0 (et gradient fini) pour une coupe constante ou sans pixel de poids non nul.
pub fn ncc(a: Tensor<2>, b: Tensor<2>, w: Tensor<2>) -> Tensor<1> {
    let [s, _] = a.dims();
    let somme_w = w.clone().sum_dim(1).clamp_min(1e-12); // [S, 1] ; jamais nul, pour la division
    let moyenne = |x: &Tensor<2>| (w.clone() * x.clone()).sum_dim(1) / somme_w.clone();
    let da = a.clone() - moyenne(&a);
    let db = b.clone() - moyenne(&b);
    let covariance = (w.clone() * da.clone() * db.clone()).sum_dim(1);
    let variance_a = (w.clone() * da.clone() * da).sum_dim(1);
    let variance_b = (w * db.clone() * db).sum_dim(1);
    (covariance / (variance_a * variance_b + NCC_EPS).sqrt()).reshape([s])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Volume;
    use nalgebra::{Matrix4, Vector3, Vector4};
    use ndarray::Array3;

    /// Générateur pseudo-aléatoire (xorshift64) déterministe, valeurs dans [0, 1[.
    struct Alea(u64);
    impl Alea {
        fn suivant(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// CPU déterministe : le backend `flex` de Burn (Rust pur).
    fn device() -> Device {
        Device::flex()
    }

    fn aplati(data: &Array3<f32>) -> Tensor<1> {
        let v: Vec<f32> = data.iter().copied().collect(); // ordre standard : i en premier, k en dernier
        Tensor::<1>::from_floats(v.as_slice(), &device())
    }

    fn points(coords: &[[f64; 3]]) -> Tensor<2> {
        let v: Vec<f32> = coords.iter().flatten().map(|&x| x as f32).collect();
        Tensor::<2>::from_floats(TensorData::new(v, [coords.len(), 3]), &device())
    }

    fn vers_vec(t: Tensor<1>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// Volume aléatoire `nx × ny × nz` d'affine identité.
    fn volume_aleatoire(dims: (usize, usize, usize), alea: &mut Alea) -> Volume {
        let data = Array3::from_shape_fn(dims, |_| alea.suivant() as f32 * 2.0 - 1.0);
        Volume::new(data, Matrix4::identity()).unwrap()
    }

    /// Même valeur et même « dans la grille » que `Volume::sample`, sur des points tirés au hasard, dedans et dehors.
    /// Les points à moins de 1e-3 voxel d'une limite sont écartés : le test de la limite se fait en `f32` ici et en
    /// `f64` (avec tolérance de 1e-9) dans `Volume`.
    #[test]
    fn tensor_sampling_matches_volume_sample() {
        let mut alea = Alea(0x1234_5678_9ABC_DEF1);
        for dims in [(22, 20, 18), (7, 1, 5), (1, 1, 1), (2, 3, 2)] {
            let volume = volume_aleatoire(dims, &mut alea);
            let n = [dims.0, dims.1, dims.2];
            let mut coords = Vec::new();
            while coords.len() < 400 {
                // Un point sur deux est tiré dans la grille [0, n-1] (un axe d'un seul voxel : exactement 0), l'autre dans
                // [-2, n+1] : sans cela, presque aucun point ne serait dedans sur les petites grilles.
                let dedans = coords.len() % 2 == 0;
                let c = [0, 1, 2].map(|a| match (dedans, n[a]) {
                    (true, 1) => 0.0,
                    (true, _) => alea.suivant() * (n[a] - 1) as f64,
                    (false, _) => alea.suivant() * (n[a] as f64 + 3.0) - 2.0,
                });
                // Les points à moins de 1e-3 voxel d'une limite sont écartés (sauf le cas exact d'un axe d'un seul voxel).
                if (0..3).all(|a| n[a] == 1 && c[a] == 0.0 || (c[a] - 0.0).abs() > 1e-3 && (c[a] - (n[a] - 1) as f64).abs() > 1e-3) {
                    coords.push(c);
                }
            }
            let sortie = trilinear_sample(&aplati(volume.data()), [n[0], n[1], n[2]], points(&coords));
            let (valeurs, dedans) = (vers_vec(sortie.values), vers_vec(sortie.inside));
            let (mut n_dedans, mut pire) = (0, 0.0_f64);
            for (m, c) in coords.iter().enumerate() {
                let attendu = volume.sample(&volume_vers_monde(&volume, c));
                assert_eq!(dedans[m] == 1.0, attendu.is_some(), "dims {dims:?}, point {c:?}");
                if let Some(a) = attendu {
                    n_dedans += 1;
                    pire = pire.max((f64::from(valeurs[m]) - f64::from(a)).abs());
                }
            }
            println!("dims {dims:?} : {n_dedans} points dedans sur {} ; écart max {pire:.2e}", coords.len());
            assert!(pire < 1e-5, "dims {dims:?} : {pire:.2e}");
        }
    }

    /// Les centres du premier et du dernier voxel sont dans la grille (bord inclus) et rendent la valeur du voxel.
    #[test]
    fn tensor_sampling_includes_the_boundary_voxels() {
        let mut alea = Alea(7);
        let volume = volume_aleatoire((5, 4, 3), &mut alea);
        let coins = [[0.0, 0.0, 0.0], [4.0, 3.0, 2.0], [0.0, 3.0, 2.0]];
        let sortie = trilinear_sample(&aplati(volume.data()), [5, 4, 3], points(&coins));
        let (valeurs, dedans) = (vers_vec(sortie.values), vers_vec(sortie.inside));
        for (m, c) in coins.iter().enumerate() {
            assert_eq!(dedans[m], 1.0, "{c:?}");
            assert_eq!(valeurs[m], volume.data()[[c[0] as usize, c[1] as usize, c[2] as usize]], "{c:?}");
        }
    }

    /// L'affine du volume est l'identité : monde = indices.
    fn volume_vers_monde(volume: &Volume, c: &[f64; 3]) -> Vector3<f64> {
        (volume.affine() * Vector4::new(c[0], c[1], c[2], 1.0)).xyz()
    }

    /// Une fonction linéaire des indices est reproduite exactement (à `f32` près).
    #[test]
    fn tensor_sampling_is_exact_for_a_linear_function() {
        let dims = (6, 5, 4);
        let data = Array3::from_shape_fn(dims, |(i, j, k)| (1.0 + 2.0 * i as f64 - 3.0 * j as f64 + 0.5 * k as f64) as f32);
        let coords = [[0.0, 0.0, 0.0], [5.0, 4.0, 3.0], [1.5, 0.25, 2.75], [4.9, 3.1, 0.3]];
        let sortie = trilinear_sample(&aplati(&data), [6, 5, 4], points(&coords));
        let valeurs = vers_vec(sortie.values);
        for (v, c) in valeurs.iter().zip(&coords) {
            let attendu = 1.0 + 2.0 * c[0] - 3.0 * c[1] + 0.5 * c[2];
            assert!((f64::from(*v) - attendu).abs() < 1e-4, "{c:?} : {v} contre {attendu}");
        }
        assert!(vers_vec(sortie.inside).iter().all(|&m| m == 1.0));
    }
    // ------------------------------------------------------------------ sous-étape 2 : pose

    use nalgebra::{Matrix3, Rotation3};

    fn tenseur_1d(v: &[f64], device: &Device) -> Tensor<1> {
        Tensor::<1>::from_floats(v.iter().map(|&x| x as f32).collect::<Vec<_>>().as_slice(), device)
    }

    fn matrice_depuis(t: Tensor<2>) -> Matrix3<f64> {
        let v = t.into_data().try_to_vec::<f32>().unwrap();
        Matrix3::from_row_slice(&v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>())
    }

    /// Critère 1 : Rodrigues contre nalgebra (f64) ; orthonormée de déterminant +1.
    #[test]
    fn rodrigues_matches_nalgebra_and_is_a_rotation() {
        let mut alea = Alea(0xABCD_EF01_2345_6789);
        let mut vecteurs: Vec<Vector3<f64>> = vec![
            Vector3::zeros(),
            Vector3::new(1e-9, 0.0, 0.0),
            Vector3::new(0.0, 0.05, 0.0),
            Vector3::new(0.0, 0.0999, 0.0), // juste sous le seuil de la série
            Vector3::new(0.0, 0.1001, 0.0), // juste au-dessus
            Vector3::new(std::f64::consts::PI - 1e-3, 0.0, 0.0),
        ];
        while vecteurs.len() < 300 {
            let axe = Vector3::new(alea.suivant() - 0.5, alea.suivant() - 0.5, alea.suivant() - 0.5).normalize();
            vecteurs.push(axe * (alea.suivant() * 3.1));
        }
        let (mut pire, mut pire_ortho) = (0.0_f64, 0.0_f64);
        for w in &vecteurs {
            let r = matrice_depuis(rotation_matrix(tenseur_1d(&[w.x, w.y, w.z], &device())));
            let attendue = Rotation3::from_scaled_axis(*w).into_inner();
            pire = pire.max((r - attendue).abs().max());
            pire_ortho = pire_ortho.max((r.transpose() * r - Matrix3::identity()).abs().max()).max((r.determinant() - 1.0).abs());
        }
        println!("{} rotations : écart max à nalgebra {pire:.2e} ; orthonormalité/déterminant {pire_ortho:.2e}", vecteurs.len());
        assert!(pire < 1e-5, "{pire:.2e}");
        assert!(pire_ortho < 1e-5, "{pire_ortho:.2e}");
    }

    fn nuage_isotrope(n: usize, alea: &mut Alea) -> Vec<Vector3<f64>> {
        // gaussienne 3D par Box-Muller : isotrope, de rayon quadratique moyen ≈ 30 mm
        let mut gauss = || {
            let (u1, u2) = (alea.suivant().max(1e-12), alea.suivant());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        (0..n).map(|_| Vector3::new(gauss(), gauss(), gauss()) * 17.3 + Vector3::new(10.0, -20.0, 5.0)).collect()
    }

    fn tenseur_points(points: &[Vector3<f64>], device: &Device) -> Tensor<2> {
        let v: Vec<f32> = points.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
        Tensor::<2>::from_floats(TensorData::new(v, [points.len(), 3]), device)
    }

    fn deplacement_rms(avant: &[Vector3<f64>], apres: Tensor<2>) -> f64 {
        let v = apres.into_data().try_to_vec::<f32>().unwrap();
        let somme: f64 = avant
            .iter()
            .enumerate()
            .map(|(i, p)| (Vector3::new(f64::from(v[3 * i]), f64::from(v[3 * i + 1]), f64::from(v[3 * i + 2])) - p).norm_squared())
            .sum();
        (somme / avant.len() as f64).sqrt()
    }

    /// Critères 2 et 3 : pivot fixe, translation, et équivalence en mm de la rotation.
    #[test]
    fn pose_keeps_the_pivot_and_has_unit_mm_sensitivity() {
        let device = device();
        let mut alea = Alea(99);
        let points = nuage_isotrope(4000, &mut alea);
        let pivot = points.iter().sum::<Vector3<f64>>() / points.len() as f64;
        let echelle = rotation_scale_mm(&points, &pivot).unwrap();
        let (tp, tc) = (tenseur_points(&points, &device), tenseur_1d(&[pivot.x, pivot.y, pivot.z], &device));
        // pivot : invariant sans translation, déplacé de t avec
        let seul = tenseur_points(&[pivot], &device);
        let tourne = apply_pose(seul.clone(), tc.clone(), tenseur_1d(&[3.0, -2.0, 5.0, 0.0, 0.0, 0.0], &device), echelle);
        assert!(deplacement_rms(&[pivot], tourne) < 1e-4, "le pivot doit rester fixe");
        let decale = apply_pose(seul, tc.clone(), tenseur_1d(&[3.0, -2.0, 5.0, 1.0, 2.0, -3.0], &device), echelle);
        assert!((deplacement_rms(&[pivot], decale) - (1.0f64 + 4.0 + 9.0).sqrt()).abs() < 1e-4);
        // sensibilité : 1 unité de chaque paramètre déplace de ≈ 1 mm (rms) ; rotation moyennée sur les 3 axes
        let mut rotations = Vec::new();
        for a in 0..6 {
            let mut p = [0.0; 6];
            p[a] = 1.0;
            let d = deplacement_rms(&points, apply_pose(tp.clone(), tc.clone(), tenseur_1d(&p, &device), echelle));
            println!("paramètre {a} = 1 : déplacement quadratique moyen {d:.4} mm");
            if a >= 3 {
                assert!((d - 1.0).abs() < 1e-4, "translation : {d}");
            } else {
                rotations.push(d * d);
            }
        }
        let moyenne = (rotations.iter().sum::<f64>() / 3.0).sqrt();
        println!("rotation, moyenne sur les 3 axes : {moyenne:.4} mm (échelle {echelle:.2} mm)");
        assert!((moyenne - 1.0).abs() < 0.02, "{moyenne}");
    }

    #[test]
    fn rotation_scale_handles_degenerate_clouds() {
        assert!(rotation_scale_mm(&[], &Vector3::zeros()).is_none());
        let p = Vector3::new(1.0, 2.0, 3.0);
        assert!(rotation_scale_mm(&[p, p], &p).is_none(), "tous les points au pivot");
        // deux points à 3 mm du pivot : r_rms = 3, échelle = 3 √(2/3)
        let e = rotation_scale_mm(&[Vector3::new(3.0, 0.0, 0.0), Vector3::new(0.0, -3.0, 0.0)], &Vector3::zeros()).unwrap();
        assert!((e - 3.0 * (2.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    /// Coût de test : somme des carrés des écarts entre la pose appliquée à `x` et les cibles `y`, en f64 pour la référence.
    fn cout_f64(p: &[f64; 6], x: &[Vector3<f64>], y: &[Vector3<f64>], pivot: &Vector3<f64>, echelle: f64) -> f64 {
        let r = Rotation3::from_scaled_axis(Vector3::new(p[0], p[1], p[2]) / echelle);
        let t = Vector3::new(p[3], p[4], p[5]);
        x.iter().zip(y).map(|(x, y)| (pivot + r * (x - pivot) + t - y).norm_squared()).sum()
    }

    /// Critère 4 : le gradient automatique de Burn (f32) égale les différences finies centrées (f64), en ω = 0 comme ailleurs.
    #[test]
    fn autodiff_gradient_matches_finite_differences_including_at_zero() {
        let device = device().autodiff();
        let mut alea = Alea(2024);
        let x = nuage_isotrope(200, &mut alea);
        let pivot = x.iter().sum::<Vector3<f64>>() / x.len() as f64;
        let echelle = rotation_scale_mm(&x, &pivot).unwrap();
        // cibles : x déplacé par une pose connue (3 mm, 2°) puis bruité
        let vraie = Rotation3::from_scaled_axis(Vector3::new(0.02, -0.03, 0.01));
        let y: Vec<Vector3<f64>> = x
            .iter()
            .map(|p| pivot + vraie * (p - pivot) + Vector3::new(1.0, -2.0, 1.5) + Vector3::new(alea.suivant() - 0.5, alea.suivant() - 0.5, alea.suivant() - 0.5) * 0.4)
            .collect();
        let (tx, ty, tc) = (tenseur_points(&x, &device), tenseur_points(&y, &device), tenseur_1d(&[pivot.x, pivot.y, pivot.z], &device));
        for p0 in [[0.0; 6], [1.5, -2.0, 0.7, 0.4, -0.3, 0.9], [20.0, -10.0, 15.0, 0.0, 0.0, 0.0]] {
            let params = tenseur_1d(&p0, &device).require_grad();
            let ecart = apply_pose(tx.clone(), tc.clone(), params.clone(), echelle) - ty.clone();
            let perte = (ecart.clone() * ecart).sum();
            let grads = perte.backward();
            let g_auto: Vec<f64> = params.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap().iter().map(|&v| f64::from(v)).collect();
            assert!(g_auto.iter().all(|v| v.is_finite()), "gradient non fini en {p0:?} : {g_auto:?}");
            let h = 1e-4;
            let g_diff: Vec<f64> = (0..6)
                .map(|a| {
                    let (mut plus, mut moins) = (p0, p0);
                    plus[a] += h;
                    moins[a] -= h;
                    (cout_f64(&plus, &x, &y, &pivot, echelle) - cout_f64(&moins, &x, &y, &pivot, echelle)) / (2.0 * h)
                })
                .collect();
            let norme = g_diff.iter().map(|v| v * v).sum::<f64>().sqrt();
            let erreur = g_auto.iter().zip(&g_diff).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() / norme;
            println!("p0 = {p0:?} : |g| = {norme:.1}, écart relatif autodiff / différences finies {erreur:.2e}");
            assert!(erreur < 1e-3, "{erreur:.2e} en {p0:?}");
        }
    }
    // ------------------------------------------------------------------ sous-étape 3 : NCC

    fn tenseur_2d(lignes: &[&[f64]], device: &Device) -> Tensor<2> {
        let v: Vec<f32> = lignes.iter().flat_map(|l| l.iter().map(|&x| x as f32)).collect();
        Tensor::<2>::from_floats(TensorData::new(v, [lignes.len(), lignes[0].len()]), device)
    }

    fn ncc_de(a: &[&[f64]], b: &[&[f64]], w: &[&[f64]]) -> Vec<f64> {
        let d = device();
        ncc(tenseur_2d(a, &d), tenseur_2d(b, &d), tenseur_2d(w, &d)).into_data().try_to_vec::<f32>().unwrap().iter().map(|&v| f64::from(v)).collect()
    }

    const A8: [f64; 8] = [0.3, 1.2, -0.5, 2.0, 0.9, 1.7, -1.1, 0.4];
    const B8: [f64; 8] = [0.5, 1.0, -0.2, 1.6, 1.4, 1.2, -0.9, 0.1];
    const W8: [f64; 8] = [1.0, 0.5, 0.0, 2.0, 1.0, 0.25, 1.0, 0.0];

    /// Critère 1 : valeur contre numpy (`np.cov(a, b, aweights=w)`) et cas calculable à la main.
    #[test]
    fn ncc_matches_numpy_and_a_hand_computed_case() {
        let n = ncc_de(&[&A8], &[&B8], &[&W8])[0];
        println!("NCC pondérée : {n:.10} (numpy 0.9567003371)");
        assert!((n - 0.9567003370886679).abs() < 1e-6, "{n}");
        let main = ncc_de(&[&[1.0, 2.0, 3.0, 4.0]], &[&[1.0, 3.0, 2.0, 4.0]], &[&[1.0; 4]])[0];
        assert!((main - 0.8).abs() < 1e-6, "{main}");
    }

    /// Critère 2 : propriétés de la NCC.
    #[test]
    fn ncc_has_the_expected_invariances() {
        let w = [&W8[..]];
        let affine = |alpha: f64, beta: f64| B8.map(|v| alpha * v + beta);
        // +1 / -1 pour une relation affine croissante / décroissante
        let a_affine = A8.map(|v| 3.0 * v + 7.0);
        assert!((ncc_de(&[&A8], &[&a_affine], &w)[0] - 1.0).abs() < 1e-5);
        let a_oppose = A8.map(|v| -2.0 * v + 1.0);
        assert!((ncc_de(&[&A8], &[&a_oppose], &w)[0] + 1.0).abs() < 1e-5);
        // invariance aux changements d'échelle et de décalage de chaque image
        let base = ncc_de(&[&A8], &[&B8], &w)[0];
        for (alpha, beta) in [(2.5, 0.0), (0.4, 3.0), (10.0, -5.0)] {
            assert!((ncc_de(&[&A8], &[&affine(alpha, beta)], &w)[0] - base).abs() < 1e-5, "b -> {alpha} b + {beta}");
            assert!((ncc_de(&[&A8.map(|v| alpha * v + beta)], &[&B8], &w)[0] - base).abs() < 1e-5, "a -> {alpha} a + {beta}");
        }
        // un pixel de poids nul n'a aucun effet, quelle que soit sa valeur
        let (mut a_bis, mut b_bis) = (A8, B8);
        a_bis[2] = 1000.0;
        b_bis[7] = -500.0;
        assert!((ncc_de(&[&a_bis], &[&b_bis], &w)[0] - base).abs() < 1e-6);
        // un poids 2 équivaut à un pixel dupliqué
        let a_dup = [0.3, 1.2, 2.0, 2.0, 0.9];
        let b_dup = [0.5, 1.0, 1.6, 1.6, 1.4];
        let a_poids = [0.3, 1.2, 2.0, 0.9];
        let b_poids = [0.5, 1.0, 1.6, 1.4];
        let duplique = ncc_de(&[&a_dup], &[&b_dup], &[&[1.0; 5]])[0];
        let pese = ncc_de(&[&a_poids], &[&b_poids], &[&[1.0, 1.0, 2.0, 1.0]])[0];
        assert!((duplique - pese).abs() < 1e-6, "{duplique} contre {pese}");
    }

    /// Critère 3 : chaque ligne d'un lot égale la même coupe calculée seule.
    #[test]
    fn ncc_rows_are_independent() {
        let a2 = A8.map(|v| v * v - 0.3);
        let b2 = B8.map(|v| 1.0 - v);
        let lot = ncc_de(&[&A8, &a2], &[&B8, &b2], &[&W8, &[1.0; 8]]);
        let seul1 = ncc_de(&[&A8], &[&B8], &[&W8])[0];
        let seul2 = ncc_de(&[&a2], &[&b2], &[&[1.0; 8]])[0];
        assert!((lot[0] - seul1).abs() < 1e-7 && (lot[1] - seul2).abs() < 1e-7, "{lot:?} contre {seul1}, {seul2}");
        assert!((lot[0] - lot[1]).abs() > 0.1, "les deux coupes de test doivent différer");
    }

    /// Critère 4 : coupe constante, masque vide : NCC nulle, valeur et gradient finis.
    #[test]
    fn ncc_is_finite_for_degenerate_slices() {
        let device = device().autodiff();
        let a = tenseur_2d(&[&[5.0; 6], &A8[..6]], &device).require_grad();
        let b = tenseur_2d(&[&B8[..6], &B8[..6]], &device).require_grad();
        let w = tenseur_2d(&[&[1.0; 6], &[0.0; 6]], &device); // ligne 0 : `a` constant ; ligne 1 : masque vide
        let n = ncc(a.clone(), b.clone(), w);
        let valeurs = n.clone().into_data().try_to_vec::<f32>().unwrap();
        assert!(valeurs.iter().all(|&v| v == 0.0), "{valeurs:?}");
        let grads = n.sum().backward();
        for (nom, t) in [("a", &a), ("b", &b)] {
            let g = t.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();
            assert!(g.iter().all(|v| v.is_finite()), "gradient non fini pour {nom} : {g:?}");
        }
    }

    /// Critère 5 : gradient automatique contre différences finies (f64) par rapport à `b`.
    #[test]
    fn ncc_gradient_matches_finite_differences() {
        let ncc_f64 = |a: &[f64], b: &[f64], w: &[f64]| -> f64 {
            let sw: f64 = w.iter().sum();
            let m = |x: &[f64]| x.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / sw;
            let (ma, mb) = (m(a), m(b));
            let cov: f64 = (0..a.len()).map(|i| w[i] * (a[i] - ma) * (b[i] - mb)).sum();
            let va: f64 = (0..a.len()).map(|i| w[i] * (a[i] - ma).powi(2)).sum();
            let vb: f64 = (0..a.len()).map(|i| w[i] * (b[i] - mb).powi(2)).sum();
            cov / (va * vb + NCC_EPS).sqrt()
        };
        let device = device().autodiff();
        let (a, b, w) = (tenseur_2d(&[&A8], &device), tenseur_2d(&[&B8], &device).require_grad(), tenseur_2d(&[&W8], &device));
        let grads = ncc(a, b.clone(), w).sum().backward();
        let g_auto: Vec<f64> = b.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap().iter().map(|&v| f64::from(v)).collect();
        let h = 1e-6;
        let g_diff: Vec<f64> = (0..8)
            .map(|i| {
                let (mut plus, mut moins) = (B8, B8);
                plus[i] += h;
                moins[i] -= h;
                (ncc_f64(&A8, &plus, &W8) - ncc_f64(&A8, &moins, &W8)) / (2.0 * h)
            })
            .collect();
        let norme = g_diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        let erreur = g_auto.iter().zip(&g_diff).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt() / norme;
        println!("|g| = {norme:.3}, écart relatif autodiff / différences finies {erreur:.2e}");
        assert!(erreur < 1e-3, "{erreur:.2e}");
        // un pixel de poids nul a un gradient nul
        assert!(g_auto[2].abs() < 1e-7 && g_auto[7].abs() < 1e-7, "{g_auto:?}");
    }
}
