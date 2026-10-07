//! Coût **différentiable** du recalage des coupes (étape 3) : les calculs sont écrits en opérations sur
//! tenseurs Burn, pour que la différentiation automatique en tire les gradients.
//!
//! Burn choisit son backend à l'exécution par un `Device` : tous les tenseurs d'un calcul vivent sur le
//! même `Device`.

use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use nalgebra::Vector3;

use crate::Volume;

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

/// Un [`Volume`] converti en tenseurs, prêt à être échantillonné par [`slice_ncc`] : voxels aplatis, dimensions, et
/// l'inverse de l'affine (monde → indices de voxel) en `f32`.
#[derive(Debug, Clone)]
pub struct VolumeTensors {
    data: Tensor<1>,
    dims: [usize; 3],
    inverse_linear: Tensor<2>,
    inverse_offset: Tensor<1>,
}

impl VolumeTensors {
    /// Copie les voxels et l'inverse de l'affine du volume sur `device`.
    pub fn new(volume: &Volume, device: &Device) -> VolumeTensors {
        let (nx, ny, nz) = volume.dim();
        let voxels: Vec<f32> = volume.data().iter().copied().collect(); // ordre standard : k varie le plus vite
        let inverse = &volume.inverse;
        let lineaire: Vec<f32> = (0..3).flat_map(|r| (0..3).map(move |c| (r, c))).map(|(r, c)| inverse[(r, c)] as f32).collect();
        let decalage: Vec<f32> = (0..3).map(|r| inverse[(r, 3)] as f32).collect();
        VolumeTensors {
            data: Tensor::<1>::from_floats(voxels.as_slice(), device),
            dims: [nx, ny, nz],
            inverse_linear: Tensor::<2>::from_floats(TensorData::new(lineaire, [3, 3]), device),
            inverse_offset: Tensor::<1>::from_floats(decalage.as_slice(), device),
        }
    }
}

/// **NCC d'une coupe** pour une pose donnée : la chaîne complète du coût de recalage.
///
/// `points` (`[P, 3]`, mm) sont les centres des pixels de la coupe dans le monde, à la pose d'en-tête. La pose `params`
/// (`[6]`, voir [`apply_pose`]) les déplace autour de `pivot` ; le volume est lu en ces points (trilinéaire) et comparé à
/// `intensities` (`[P]`, les pixels de la coupe) par la corrélation normalisée. Le poids d'un pixel est `mask` (`[P]`, 0 ou 1,
/// par exemple le masque cérébral de la coupe) multiplié par « le point est dans le volume ».
///
/// Rend un tenseur de forme `[1]` (la NCC, à maximiser : la perte est son opposé). Le gradient par rapport à `params` existe
/// presque partout : l'interpolation trilinéaire est continue mais affine par morceaux, donc sa dérivée saute quand un point
/// franchit un plan de voxels.
pub fn slice_ncc(
    volume: &VolumeTensors,
    points: Tensor<2>,
    pivot: Tensor<1>,
    params: Tensor<1>,
    rotation_scale_mm: f64,
    intensities: Tensor<1>,
    mask: Tensor<1>,
) -> Tensor<1> {
    let p = points.dims()[0];
    let deplaces = apply_pose(points, pivot, params, rotation_scale_mm);
    // monde -> indices de voxel : x_vox = A⁻¹ x ; pour des lignes de points, x.matmul(Mᵀ) + b
    let voxels = deplaces.matmul(volume.inverse_linear.clone().transpose()) + volume.inverse_offset.clone().reshape([1, 3]);
    let lecture = trilinear_sample(&volume.data, volume.dims, voxels);
    ncc(intensities.reshape([1, p]), lecture.values.reshape([1, p]), (mask * lecture.inside).reshape([1, p]))
}

/// Les 6 paramètres de pose, dans un `Module` : c'est ce que les optimiseurs de Burn savent mettre à jour. Un `Param` est un tenseur
/// qui porte un identifiant, auquel `backward` associe un gradient.
#[derive(Module, Debug)]
struct PoseModule {
    params: Param<Tensor<1>>,
}

/// Réglages de [`register_slice`].
#[derive(Debug, Clone, Copy)]
pub struct RegistrationConfig {
    /// Pas initial d'Adam, en mm (les 6 paramètres sont en mm : un pas de 1 déplace un point typique d'environ 1 mm).
    pub learning_rate: f64,
    /// Nombre d'itérations (une évaluation du coût et un gradient chacune).
    pub iterations: usize,
    /// Facteur multiplicatif du pas à chaque itération (1 : pas constant).
    pub lr_decay: f64,
    /// Pose de départ `(φx, φy, φz, tx, ty, tz)` ; `[0.0; 6]` (la pose d'en-tête) dans le cas courant.
    pub initial: [f64; 6],
}

/// Résultat de [`register_slice`].
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// Pose estimée `(φx, φy, φz, tx, ty, tz)`, voir [`apply_pose`].
    pub params: [f64; 6],
    /// NCC à chaque itération (avant la mise à jour) puis à la pose finale : `iterations + 1` valeurs.
    pub ncc_history: Vec<f64>,
}

/// **Recale une coupe** : cherche la pose (delta autour de la pose d'en-tête, départ en pose nulle) qui maximise la NCC avec
/// le volume, par Adam (Burn) sur le gradient automatique de [`slice_ncc`]. Le départ est `config.initial`.
///
/// Les tenseurs doivent être sur un `Device` avec autodiff (`Device::...().autodiff()`). Un niveau de résolution seulement : la
/// pyramide viendra si la capture mesurée l'exige.
pub fn register_slice(
    volume: &VolumeTensors,
    points: Tensor<2>,
    pivot: Tensor<1>,
    rotation_scale_mm: f64,
    intensities: Tensor<1>,
    mask: Tensor<1>,
    config: RegistrationConfig,
) -> RegistrationResult {
    let device = points.device();
    let depart: Vec<f32> = config.initial.iter().map(|&v| v as f32).collect();
    let mut module = PoseModule { params: Param::from_tensor(Tensor::<1>::from_floats(depart.as_slice(), &device)) };
    let mut optimiseur = AdamConfig::new().init();
    let evaluer = |pose: Tensor<1>| {
        slice_ncc(volume, points.clone(), pivot.clone(), pose, rotation_scale_mm, intensities.clone(), mask.clone())
    };
    let mut historique = Vec::with_capacity(config.iterations + 1);
    let mut pas = config.learning_rate;
    for _ in 0..config.iterations {
        let ncc = evaluer(module.params.val());
        historique.push(f64::from(ncc.clone().into_data().try_to_vec::<f32>().unwrap()[0]));
        let gradients = ncc.neg().sum().backward(); // perte = −NCC
        // `from_grads` emprunte `module` ; `step` le consomme et rend le module mis à jour : deux instructions, dans cet ordre.
        let gradients = GradientsParams::from_grads(gradients, &module);
        module = optimiseur.step(pas, module, gradients);
        pas *= config.lr_decay;
    }
    let finale = module.params.val();
    historique.push(f64::from(evaluer(finale.clone()).into_data().try_to_vec::<f32>().unwrap()[0]));
    let params: Vec<f32> = finale.into_data().try_to_vec::<f32>().unwrap();
    RegistrationResult { params: std::array::from_fn(|i| f64::from(params[i])), ncc_history: historique }
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
    // ------------------------------------------------------------------ sous-étape 4 : coût complet d'une coupe

    use crate::Stack;

    /// L'atlas de Gholipour, **tourné** dans le monde par une rotation arbitraire : son affine d'origine est diagonale, donc
    /// la transposition de l'inverse ne serait pas éprouvée (une mutation l'a montré). Mêmes voxels, affine oblique ; l'atlas est
    /// centré en (0, 0, 0), le volume reste donc autour de l'origine.
    fn atlas() -> Volume {
        let chemin = format!("{}/../../data/atlas/gholipour/STA21.nii.gz", env!("CARGO_MANIFEST_DIR"));
        let original = Volume::from_stack(&Stack::read(std::path::Path::new(&chemin)).expect("atlas absent : voir data/atlas/gholipour"));
        let rotation = Rotation3::from_euler_angles(0.3, -0.2, 0.5).to_homogeneous();
        Volume::new(original.data().clone(), rotation * original.affine()).unwrap()
    }

    /// Une coupe de 56 × 56 pixels de 1 mm, oblique, au milieu de l'atlas : centres de pixels dans le monde.
    fn pixels_de_coupe() -> Vec<Vector3<f64>> {
        let r = Rotation3::from_euler_angles(0.5, -0.4, 0.8);
        (0..56).flat_map(|j| (0..56).map(move |i| (i, j))).map(|(i, j)| r * Vector3::new(i as f64 - 27.5, j as f64 - 27.5, 0.0) + Vector3::new(1.0, -2.0, 3.0)).collect()
    }

    /// Cas de test : atlas, pixels, intensités simulées à la pose vraie (échelle et décalage d'intensité arbitraires), masque, pivot.
    struct Cas {
        volume: Volume,
        points: Vec<Vector3<f64>>,
        intensites: Vec<f64>,
        masque: Vec<f64>,
        pivot: Vector3<f64>,
        echelle: f64,
    }
    const POSE_VRAIE: [f64; 6] = [4.0, -3.0, 2.0, 1.5, -1.0, 0.8];

    /// Valeur du volume en `f64` exact (sans l'arrondi en `f32` de `Volume::sample`), avec les mêmes coefficients trilinéaires
    /// (`Volume::trilinear`, validé contre scipy et SimpleITK par `sample`). Les différences finies ont besoin de cette précision.
    fn echantillon_f64(v: &Volume, x: &Vector3<f64>) -> Option<f64> {
        let voisins = v.trilinear(&v.voxel_coordinates(x))?;
        Some(voisins.iter().map(|(i, w)| w * f64::from(v.data[*i])).sum())
    }

    /// Pose appliquée en `f64` (référence) : `c + R (x − c) + t`.
    fn deplace(p: &[f64; 6], x: &Vector3<f64>, pivot: &Vector3<f64>, echelle: f64) -> Vector3<f64> {
        pivot + Rotation3::from_scaled_axis(Vector3::new(p[0], p[1], p[2]) / echelle) * (x - pivot) + Vector3::new(p[3], p[4], p[5])
    }

    fn cas() -> Cas {
        let volume = atlas();
        let points = pixels_de_coupe();
        // pivot = barycentre des pixels, échelle = √(2/3) r_rms de ces pixels
        let pivot = points.iter().sum::<Vector3<f64>>() / points.len() as f64;
        let echelle = rotation_scale_mm(&points, &pivot).unwrap();
        // coupe « acquise » : le volume lu à la pose vraie (intensités à échelle et décalage arbitraires), masque = tissu
        let (mut intensites, mut masque) = (Vec::new(), Vec::new());
        for x in &points {
            let v = echantillon_f64(&volume, &deplace(&POSE_VRAIE, x, &pivot, echelle));
            intensites.push(3.0 * v.unwrap_or(0.0) + 10.0);
            masque.push(f64::from(u8::from(v.is_some_and(|v| v > 150.0))));
        }
        Cas { volume, points, intensites, masque, pivot, echelle }
    }

    /// Coût de référence en f64 : pose de nalgebra, `Volume::sample` (validé contre scipy et SimpleITK), NCC pondérée.
    fn ncc_reference(c: &Cas, p: &[f64; 6]) -> f64 {
        let (mut sw, mut sa, mut sb) = (0.0, 0.0, 0.0);
        let mut lectures = Vec::new();
        for (i, x) in c.points.iter().enumerate() {
            let b = echantillon_f64(&c.volume, &deplace(p, x, &c.pivot, c.echelle));
            let w = c.masque[i] * f64::from(u8::from(b.is_some()));
            let b = b.unwrap_or(0.0);
            lectures.push((c.intensites[i], b, w));
            sw += w;
            sa += w * c.intensites[i];
            sb += w * b;
        }
        let (ma, mb) = (sa / sw, sb / sw);
        let cov: f64 = lectures.iter().map(|(a, b, w)| w * (a - ma) * (b - mb)).sum();
        let va: f64 = lectures.iter().map(|(a, _, w)| w * (a - ma).powi(2)).sum();
        let vb: f64 = lectures.iter().map(|(_, b, w)| w * (b - mb).powi(2)).sum();
        cov / (va * vb + NCC_EPS).sqrt()
    }

    struct Tenseurs {
        volume: VolumeTensors,
        points: Tensor<2>,
        pivot: Tensor<1>,
        intensites: Tensor<1>,
        masque: Tensor<1>,
    }

    fn en_tenseurs(c: &Cas, device: &Device) -> Tenseurs {
        let v1 = |v: &[f64]| tenseur_1d(v, device);
        Tenseurs {
            volume: VolumeTensors::new(&c.volume, device),
            points: tenseur_points(&c.points, device),
            pivot: v1(&[c.pivot.x, c.pivot.y, c.pivot.z]),
            intensites: v1(&c.intensites),
            masque: v1(&c.masque),
        }
    }

    fn ncc_tenseurs(c: &Cas, t: &Tenseurs, params: Tensor<1>) -> Tensor<1> {
        slice_ncc(&t.volume, t.points.clone(), t.pivot.clone(), params, c.echelle, t.intensites.clone(), t.masque.clone())
    }

    /// L'échantillonneur `f64` de référence rend la même valeur que `Volume::sample` à l'arrondi `f32` près.
    #[test]
    fn f64_reference_sampler_agrees_with_volume_sample() {
        let c = cas();
        let mut pire = 0.0_f64;
        for x in &c.points {
            let y = deplace(&POSE_VRAIE, x, &c.pivot, c.echelle);
            match (c.volume.sample(&y), echantillon_f64(&c.volume, &y)) {
                (Some(a), Some(b)) => pire = pire.max((f64::from(a) - b).abs() / 3485.0),
                (None, None) => {}
                autre => panic!("domaines différents : {autre:?}"),
            }
        }
        assert!(pire < 1e-7, "{pire:.2e}");
    }

    /// Critère 1 : valeur contre la référence f64 indépendante.
    #[test]
    fn slice_ncc_matches_the_f64_reference_on_a_real_volume() {
        let c = cas();
        let device = device();
        let t = en_tenseurs(&c, &device);
        println!("{} pixels, dont {} dans le masque ; échelle de rotation {:.1} mm", c.points.len(), c.masque.iter().sum::<f64>(), c.echelle);
        assert!(c.masque.iter().sum::<f64>() > 1000.0, "le masque doit contenir assez de pixels");
        let mut pire = 0.0_f64;
        for p in [[0.0; 6], POSE_VRAIE, [2.0, 1.0, -3.0, 0.5, 0.5, -1.0], [-6.0, 5.0, 4.0, 3.0, -2.0, 2.5], [0.0, 0.0, 0.0, 45.0, 0.0, 0.0]] {
            let n = f64::from(ncc_tenseurs(&c, &t, tenseur_1d(&p, &device)).into_data().try_to_vec::<f32>().unwrap()[0]);
            let r = ncc_reference(&c, &p);
            println!("pose {p:?} : NCC tenseurs {n:.6}, référence f64 {r:.6}");
            pire = pire.max((n - r).abs());
        }
        assert!(pire < 1e-4, "{pire:.2e}");
    }

    /// Critère 2 : gradient automatique contre différences finies de la référence f64, en trois poses.
    #[test]
    fn slice_ncc_gradient_matches_finite_differences_on_a_real_volume() {
        let c = cas();
        let device = device().autodiff();
        let t = en_tenseurs(&c, &device);
        for p0 in [[0.0; 6], [2.0, 1.0, -3.0, 0.5, 0.5, -1.0], [-6.0, 5.0, 4.0, 3.0, -2.0, 2.5]] {
            let params = tenseur_1d(&p0, &device).require_grad();
            let grads = ncc_tenseurs(&c, &t, params.clone()).sum().backward();
            let g_auto: Vec<f64> = params.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap().iter().map(|&v| f64::from(v)).collect();
            assert!(g_auto.iter().all(|v| v.is_finite()), "{g_auto:?}");
            // Pas 1e-6 : l'interpolation trilinéaire est affine par morceaux, donc l'écart des différences finies décroît
            // comme le pas (sauts de voxel dans la fenêtre : 1,4e-2 à 1e-2, 1,4e-3 à 1e-4, 2e-6 à 1e-6, limite de l'autodiff `f32`).
            let h = 1e-6;
            let g_diff: Vec<f64> = (0..6)
                .map(|a| {
                    let (mut plus, mut moins) = (p0, p0);
                    plus[a] += h;
                    moins[a] -= h;
                    (ncc_reference(&c, &plus) - ncc_reference(&c, &moins)) / (2.0 * h)
                })
                .collect();
            let norme = g_diff.iter().map(|v| v * v).sum::<f64>().sqrt();
            let erreur = g_auto.iter().zip(&g_diff).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() / norme;
            println!("p0 = {p0:?} : |g| = {norme:.4}\n   autodiff {g_auto:.4?}\n   diff. finies {g_diff:.4?}\n   écart relatif {erreur:.2e}");
            assert!(erreur < 1e-3, "{erreur:.2e} en {p0:?}");
        }
    }

    /// Critère 3 : bout en bout. À la pose vraie la NCC vaut 1 et son gradient s'annule ; le coût baisse en s'en éloignant, dans
    /// chacune des 6 directions.
    #[test]
    fn slice_ncc_peaks_at_the_true_pose() {
        let c = cas();
        let device = device().autodiff();
        let t = en_tenseurs(&c, &device);
        let valeur = |p: &[f64; 6]| f64::from(ncc_tenseurs(&c, &t, tenseur_1d(p, &device)).into_data().try_to_vec::<f32>().unwrap()[0]);
        let gradient = |p: &[f64; 6]| -> f64 {
            let params = tenseur_1d(p, &device).require_grad();
            let g = ncc_tenseurs(&c, &t, params.clone()).sum().backward();
            params.grad(&g).unwrap().into_data().try_to_vec::<f32>().unwrap().iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt()
        };
        let a_la_verite = valeur(&POSE_VRAIE);
        let (g_vrai, g_loin) = (gradient(&POSE_VRAIE), gradient(&[0.0; 6]));
        println!("NCC à la pose vraie {a_la_verite:.6} ; |gradient| à la pose vraie {g_vrai:.2e}, à la pose nulle {g_loin:.2e}");
        assert!(a_la_verite >= 0.9999, "{a_la_verite}");
        assert!(g_vrai < 0.01 * g_loin, "{g_vrai:.2e} contre {g_loin:.2e}");
        for a in 0..6 {
            let mut precedent = a_la_verite;
            for pas in 1..=5 {
                for signe in [1.0, -1.0] {
                    let mut p = POSE_VRAIE;
                    p[a] += signe * f64::from(pas);
                    let v = valeur(&p);
                    assert!(v < precedent + 1e-6 || signe < 0.0, "paramètre {a}, pas {}", signe * f64::from(pas));
                    if signe > 0.0 {
                        precedent = v;
                    }
                }
            }
            let (plus5, moins5) = ({ let mut p = POSE_VRAIE; p[a] += 5.0; valeur(&p) }, { let mut p = POSE_VRAIE; p[a] -= 5.0; valeur(&p) });
            println!("paramètre {a} : NCC à ±5 : {plus5:.4} / {moins5:.4}");
            assert!(plus5 < a_la_verite - 0.001 && moins5 < a_la_verite - 0.001);
        }
    }
    // ------------------------------------------------------------------ sous-étape 5 : recalage d'une coupe

    use nifti::{writer::WriterOptions, NiftiHeader};

    const N_PIXELS: usize = 64;
    const PIXEL_MM: f64 = 0.8;
    const EPAISSEUR_MM: f64 = 3.5;

    /// Affine de la coupe d'en-tête : oblique, pixels de 0,8 mm, épaisseur 3,5 mm, centrée près de (1, −2, 3).
    fn affine_entete() -> Matrix4<f64> {
        let r = Rotation3::from_euler_angles(0.5, -0.4, 0.8);
        let lineaire = r.matrix() * Matrix3::from_diagonal(&Vector3::new(PIXEL_MM, PIXEL_MM, EPAISSEUR_MM));
        let milieu = (N_PIXELS as f64 - 1.0) / 2.0;
        let origine = r * Vector3::new(-milieu * PIXEL_MM, -milieu * PIXEL_MM, 0.0) + Vector3::new(1.0, -2.0, 3.0);
        let mut a = Matrix4::identity();
        a.fixed_view_mut::<3, 3>(0, 0).copy_from(&lineaire);
        a.fixed_view_mut::<3, 1>(0, 3).copy_from(&origine);
        a
    }

    /// Écrit un stack d'une seule coupe d'affine donnée (des zéros : seuls l'affine et le pas comptent) et le relit.
    fn stack_une_coupe(affine: &Matrix4<f64>, nom: &str) -> Stack {
        let chemin = std::env::temp_dir().join(format!("medoxide_diff_{nom}_{}.nii.gz", std::process::id()));
        let mut h = NiftiHeader::default();
        h.sform_code = 1;
        h.srow_x = std::array::from_fn(|c| affine[(0, c)] as f32);
        h.srow_y = std::array::from_fn(|c| affine[(1, c)] as f32);
        h.srow_z = std::array::from_fn(|c| affine[(2, c)] as f32);
        h.pixdim = [1.0, PIXEL_MM as f32, PIXEL_MM as f32, EPAISSEUR_MM as f32, 1.0, 1.0, 1.0, 1.0];
        WriterOptions::new(&chemin).reference_header(&h).write_nifti(&Array3::<f32>::zeros((N_PIXELS, N_PIXELS, 1))).unwrap();
        let stack = Stack::read(&chemin).unwrap();
        let _ = std::fs::remove_file(&chemin);
        stack
    }

    /// Mouvement vrai : rotation `omega` (rad) et translation `t` (mm) autour de `centre`, en matrice 4×4.
    fn mouvement_vrai(omega: &Vector3<f64>, t: &Vector3<f64>, centre: &Vector3<f64>) -> Matrix4<f64> {
        let r = Rotation3::from_scaled_axis(*omega);
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&(centre + t - r * centre));
        m
    }

    /// Une expérience : la coupe d'en-tête, sa version « acquise » après un mouvement vrai (simulée par l'opérateur complet de
    /// l'étape 2, PSF orientée selon la normale vraie), et tout ce que le recalage reçoit.
    struct Experience {
        points: Vec<Vector3<f64>>,
        intensites: Vec<f64>,
        masque: Vec<f64>,
        verite: Matrix4<f64>,
    }

    /// `bruit` : écart type du bruit gaussien additif, en fraction de l'intensité moyenne du tissu (0 : aucun) ; l'intensité
    /// est de plus transformée par `0,5 · v + 100` (échelle et décalage propres à la coupe, que la NCC ignore).
    fn experience(volume: &Volume, omega: &Vector3<f64>, t: &Vector3<f64>, bruit: f64, nom: &str) -> Experience {
        let a_h = affine_entete();
        let points: Vec<Vector3<f64>> = (0..N_PIXELS)
            .flat_map(|j| (0..N_PIXELS).map(move |i| (i, j)))
            .map(|(i, j)| (a_h * nalgebra::Vector4::new(i as f64, j as f64, 0.0, 1.0)).xyz())
            .collect();
        let centre = points.iter().sum::<Vector3<f64>>() / points.len() as f64; // le patient bouge autour du centre de la coupe
        let verite = mouvement_vrai(omega, t, &centre);
        let stack = stack_une_coupe(&(verite * a_h), nom);
        let coupe = stack.slice(0).unwrap();
        let sim = volume.simulate_slice(&coupe, &coupe.psf());
        let (mut intensites, mut masque) = (Vec::new(), Vec::new());
        for j in 0..N_PIXELS {
            for i in 0..N_PIXELS {
                let v = f64::from(sim.values[[i, j]]);
                intensites.push(v);
                masque.push(f64::from(u8::from(sim.coverage[[i, j]] >= 0.99 && v > 150.0)));
            }
        }
        let tissu: Vec<f64> = intensites.iter().zip(&masque).filter(|(_, &m)| m > 0.0).map(|(&v, _)| v).collect();
        let moyenne = tissu.iter().sum::<f64>() / tissu.len() as f64;
        let mut alea = Alea(0xBEEF ^ tissu.len() as u64);
        for v in &mut intensites {
            // somme de 12 uniformes - 6 : gaussienne centrée réduite approchée
            let gauss: f64 = (0..12).map(|_| alea.suivant()).sum::<f64>() - 6.0;
            *v = 0.5 * (*v + bruit * moyenne * gauss) + 100.0;
        }
        Experience { points, intensites, masque, verite }
    }

    /// Mouvement tiré comme dans la simulation de pyrecon : chaque angle et chaque translation uniformes dans ±amplitude.
    fn mouvement_aleatoire(alea: &mut Alea, degres: f64, mm: f64) -> (Vector3<f64>, Vector3<f64>) {
        let mut tire = |a: f64| (alea.suivant() * 2.0 - 1.0) * a;
        (Vector3::new(tire(degres), tire(degres), tire(degres)).map(f64::to_radians), Vector3::new(tire(mm), tire(mm), tire(mm)))
    }

    /// Erreur quadratique moyenne (TRE) sur les points masqués entre la pose estimée et le mouvement vrai.
    fn tre(e: &Experience, params: &[f64; 6], pivot: &Vector3<f64>, echelle: f64) -> f64 {
        let (somme, n) = e
            .points
            .iter()
            .zip(&e.masque)
            .filter(|(_, &m)| m > 0.0)
            .fold((0.0, 0), |(s, n), (x, _)| {
                let vrai = (e.verite * x.push(1.0)).xyz();
                (s + (deplace(params, x, pivot, echelle) - vrai).norm_squared(), n + 1)
            });
        (somme / f64::from(n)).sqrt()
    }

    /// Recale une expérience et rend (TRE avant, TRE après, NCC finale), ou `None` si la coupe a moins de 500 pixels de masque
    /// (elle est sortie du tissu : le recalage serait sans objet, et c'est le cas que le pipeline devra écarter et signaler).
    fn recaler(volume: &VolumeTensors, e: &Experience, config: RegistrationConfig, device: &Device) -> Option<(f64, f64, f64)> {
        let masques: Vec<Vector3<f64>> = e.points.iter().zip(&e.masque).filter(|(_, &m)| m > 0.0).map(|(x, _)| *x).collect();
        if masques.len() < 500 {
            return None;
        }
        let pivot = masques.iter().sum::<Vector3<f64>>() / masques.len() as f64;
        let echelle = rotation_scale_mm(&masques, &pivot).unwrap();
        let resultat = register_slice(
            volume,
            tenseur_points(&e.points, device),
            tenseur_1d(&[pivot.x, pivot.y, pivot.z], device),
            echelle,
            tenseur_1d(&e.intensites, device),
            tenseur_1d(&e.masque, device),
            config,
        );
        Some((tre(e, &[0.0; 6], &pivot, echelle), tre(e, &resultat.params, &pivot, echelle), *resultat.ncc_history.last().unwrap()))
    }

    const CONFIG_RECALAGE: RegistrationConfig = RegistrationConfig { learning_rate: 0.3, iterations: 150, lr_decay: 1.0, initial: [0.0; 6] };

    /// Série de poses tirées avec une graine, évaluée par `recaler` : (TRE avant, TRE après, NCC finale) par pose.
    fn serie(volume: &Volume, graine: u64, n: usize, degres: f64, mm: f64, bruit: f64, etiquette: &str) -> Vec<(f64, f64, f64)> {
        let device = device().autodiff();
        let vt = VolumeTensors::new(volume, &device);
        let mut alea = Alea(graine);
        (0..n)
            .filter_map(|k| {
                let (omega, t) = mouvement_aleatoire(&mut alea, degres, mm);
                recaler(&vt, &experience(volume, &omega, &t, bruit, &format!("{etiquette}{k}")), CONFIG_RECALAGE, &device)
            })
            .collect()
    }

    fn resume(res: &[(f64, f64, f64)]) -> (f64, f64, f64) {
        let mut apres: Vec<f64> = res.iter().map(|r| r.1).collect();
        apres.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let succes = apres.iter().filter(|&&e| e <= 0.5).count() as f64 / apres.len() as f64;
        (succes, apres[apres.len() / 2], apres[apres.len() - 1])
    }

    /// Critère de la sous-étape 5 (fixé avant le code) : sur 20 poses jamais vues pendant le réglage, mouvement dans la plage de
    /// pyrecon (±3° par axe, ±3 mm), TRE finale ≤ 0,5 mm pour au moins 95 % des poses. À lancer en `--release --ignored` (la
    /// simulation des coupes est lente en debug : ≈ 2 min) ; `registration_smoke_test_recovers_two_poses` en est la version courte.
    #[test]
    #[ignore = "lent en debug : lancer avec --release --ignored"]
    fn registration_recovers_motions_in_the_pyrecon_range() {
        let volume = atlas();
        let res = serie(&volume, 271828, 20, 3.0, 3.0, 0.0, "eval");
        let avant = res.iter().map(|r| r.0).sum::<f64>() / res.len() as f64;
        let (succes, mediane, pire) = resume(&res);
        println!("20 poses : TRE moyenne avant {avant:.2} mm ; après : médiane {mediane:.2}, pire {pire:.2} mm ; succès (≤ 0,5 mm) {:.0} %", succes * 100.0);
        assert!(succes >= 0.95, "taux de succès {succes}");
        assert_eq!(res.len(), 20, "toutes les coupes de l'évaluation doivent avoir un masque");
        assert!(res.iter().all(|r| r.1 < r.0), "le recalage ne doit jamais dégrader la pose");
    }

    /// Version courte du critère, dans la suite par défaut : les deux premières poses de la série d'évaluation.
    #[test]
    fn registration_smoke_test_recovers_two_poses() {
        let volume = atlas();
        let res = serie(&volume, 271828, 2, 3.0, 3.0, 0.0, "fumee");
        assert_eq!(res.len(), 2);
        for (avant, apres, ncc) in &res {
            println!("TRE {avant:.2} → {apres:.2} mm, NCC finale {ncc:.4}");
            // NCC finale rapportée, sans seuil : la coupe simulée avec PSF n'égale jamais un échantillonnage ponctuel (≈ 0,98).
            assert!(*apres <= 0.5 && apres < avant, "TRE {avant} → {apres} (NCC finale {ncc})");
        }
    }

    /// Exploration (à lancer en `--release --ignored --nocapture`) : taux de succès en fonction de l'amplitude du mouvement, sans
    /// puis avec bruit (10 % de l'intensité moyenne du tissu) ; rien n'est forcé.
    #[test]
    #[ignore = "exploration de la capture (lent en debug)"]
    fn registration_capture_curve() {
        let volume = atlas();
        for bruit in [0.0, 0.1] {
            for amplitude in [3.0, 6.0, 10.0, 15.0, 20.0] {
                let res = serie(&volume, 1618, 30, amplitude, amplitude, bruit, "capture");
                let avant = res.iter().map(|r| r.0).sum::<f64>() / res.len() as f64;
                let (succes, mediane, pire) = resume(&res);
                println!("bruit {bruit} ±{amplitude}°/mm : {} coupes sur 30 avec masque ; TRE avant",res.len());
                println!("   TRE avant {avant:.1} → après médiane {mediane:.2}, pire {pire:.2} ; succès {:.0} %", succes * 100.0);
            }
        }
    }
    // ------------------------------------------------------------------ essai sur de vrais stacks (étape 3d, version courte)

    /// Premier fichier `sub-X/ses-*/anat/*acq-<acq>_run-1_T2w.nii.gz` du jeu de développement, et le masque cérébral associé.
    fn stack_reel(sujet: &str, acq: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let racine = std::path::PathBuf::from(format!("{}/../../data/svr/jeu_reel_tru_haste", env!("CARGO_MANIFEST_DIR")));
        let mut sessions: Vec<_> = std::fs::read_dir(racine.join(sujet)).unwrap().flatten().map(|e| e.path()).collect();
        sessions.sort();
        for ses in sessions {
            let nom = format!("{sujet}_{}_acq-{acq}_run-1", ses.file_name().unwrap().to_string_lossy());
            let image = ses.join("anat").join(format!("{nom}_T2w.nii.gz"));
            if image.exists() {
                let masque = racine.join("derivatives/medx-fetalbet").join(sujet).join(ses.file_name().unwrap()).join("anat").join(format!("{nom}_desc-brain_mask.nii.gz"));
                return (image, masque);
            }
        }
        panic!("stack introuvable : {sujet} {acq}");
    }

    /// Résultat du recalage d'une coupe réelle contre une référence.
    struct CoupeRecalee {
        ncc_avant: f64,
        ncc_apres: f64,
        deplacement_moyen: Vector3<f64>, // moyenne, sur les points du masque, de T(x) − x (mm)
        deplacement_rms: f64,
    }

    fn mediane(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// Recale les coupes retenues d'un stack (masque cérébral requis) contre un volume de référence, une par une.
    fn recaler_stack(mobile: &Stack, retenues: &[usize], reference: &VolumeTensors, device: &Device) -> Vec<CoupeRecalee> {
        let masque = mobile.brain_mask().unwrap().voxels();
        retenues
            .iter()
            .map(|&k| {
                let coupe = mobile.slice(k).unwrap();
                let (nx, ny) = coupe.dim();
                let donnees = coupe.data();
                let (mut points, mut intensites) = (Vec::new(), Vec::new());
                for j in 0..ny {
                    for i in 0..nx {
                        if masque[[i, j, k]] {
                            points.push(coupe.pixel_to_world(i as f64, j as f64));
                            intensites.push(f64::from(donnees[[i, j]]));
                        }
                    }
                }
                let pivot = coupe.brain_pivot().unwrap();
                let echelle = rotation_scale_mm(&points, &pivot).unwrap();
                let resultat = register_slice(
                    reference,
                    tenseur_points(&points, device),
                    tenseur_1d(&[pivot.x, pivot.y, pivot.z], device),
                    echelle,
                    tenseur_1d(&intensites, device),
                    tenseur_1d(&vec![1.0; points.len()], device),
                    CONFIG_RECALAGE,
                );
                let deplacements: Vec<Vector3<f64>> = points.iter().map(|x| deplace(&resultat.params, x, &pivot, echelle) - x).collect();
                CoupeRecalee {
                    ncc_avant: resultat.ncc_history[0],
                    ncc_apres: *resultat.ncc_history.last().unwrap(),
                    deplacement_moyen: deplacements.iter().sum::<Vector3<f64>>() / deplacements.len() as f64,
                    deplacement_rms: (deplacements.iter().map(|d| d.norm_squared()).sum::<f64>() / deplacements.len() as f64).sqrt(),
                }
            })
            .collect()
    }

    /// Essai sur vrais stacks : voir `docs/svr/etude-05-estimation-mouvement.md` §9 pour le protocole et les critères. À lancer
    /// en `--release --ignored --nocapture` (données locales).
    #[test]
    #[ignore = "données locales ; lent en debug"]
    fn registration_on_real_stacks_with_two_independent_references() {
        let device = device().autodiff();
        let sujets = ["sub-S01", "sub-S02", "sub-S03", "sub-S05", "sub-S09", "sub-S10", "sub-S11", "sub-S13", "sub-S14"];
        let (mut toutes_ncc, mut tous_rms) = ([Vec::new(), Vec::new()], Vec::new());
        let (mut somme_diff2, mut somme_v2, mut n_v) = (0.0, 0.0, 0usize);
        let (mut n_coupes, mut n_baisse, mut n_grand) = (0usize, [0usize; 2], 0usize);
        for sujet in sujets {
            let (chemin_b, masque_b) = stack_reel(sujet, "truficor");
            let mut mobile = Stack::read(&chemin_b).unwrap();
            mobile.set_brain_mask(&masque_b).unwrap();
            let aires: Vec<usize> = (0..mobile.dim().2).map(|k| mobile.brain_mask().unwrap().voxels().index_axis(ndarray::Axis(2), k).iter().filter(|&&v| v).count()).collect();
            let max_aire = *aires.iter().max().unwrap();
            let retenues: Vec<usize> = (0..aires.len()).filter(|&k| aires[k] as f64 >= 0.25 * max_aire as f64).collect();
            let refs: Vec<Vec<CoupeRecalee>> = ["trufiax", "trufisag"]
                .iter()
                .map(|acq| {
                    let (chemin, _) = stack_reel(sujet, acq);
                    let volume = Volume::from_stack(&Stack::read(&chemin).unwrap());
                    recaler_stack(&mobile, &retenues, &VolumeTensors::new(&volume, &device), &device)
                })
                .collect();
            // M3 : mouvement relatif entre coupes = déplacement moyen moins sa médiane sur les coupes
            let residus: Vec<Vec<Vector3<f64>>> = refs
                .iter()
                .map(|r| {
                    let med = Vector3::from_fn(|a, _| mediane(r.iter().map(|c| c.deplacement_moyen[a]).collect()));
                    r.iter().map(|c| c.deplacement_moyen - med).collect()
                })
                .collect();
            let offsets: Vec<Vector3<f64>> = refs.iter().map(|r| Vector3::from_fn(|a, _| mediane(r.iter().map(|c| c.deplacement_moyen[a]).collect()))).collect();
            let (mut d2, mut v2) = (0.0, 0.0);
            for k in 0..retenues.len() {
                d2 += (residus[0][k] - residus[1][k]).norm_squared();
                v2 += 0.5 * (residus[0][k].norm_squared() + residus[1][k].norm_squared());
            }
            somme_diff2 += d2;
            somme_v2 += v2;
            n_v += retenues.len();
            for (r, ncc) in refs.iter().zip(toutes_ncc.iter_mut()) {
                ncc.extend(r.iter().map(|c| (c.ncc_avant, c.ncc_apres)));
                tous_rms.extend(r.iter().map(|c| c.deplacement_rms));
            }
            n_coupes += refs[0].len() * 2;
            for (m, r) in refs.iter().enumerate() {
                n_baisse[m] += r.iter().filter(|c| c.ncc_apres <= c.ncc_avant).count();
            }
            n_grand += refs.iter().flatten().filter(|c| c.deplacement_rms > 6.0).count();
            println!(
                "{sujet} : {} coupes ; NCC médiane axial {:.2} → {:.2}, sagittal {:.2} → {:.2} ; décalage global (axial / sagittal) {:.1} / {:.1} mm ; mouvement relatif RMS {:.2} / {:.2} mm, écart entre références {:.2} mm",
                retenues.len(),
                mediane(refs[0].iter().map(|c| c.ncc_avant).collect()),
                mediane(refs[0].iter().map(|c| c.ncc_apres).collect()),
                mediane(refs[1].iter().map(|c| c.ncc_avant).collect()),
                mediane(refs[1].iter().map(|c| c.ncc_apres).collect()),
                offsets[0].norm(),
                offsets[1].norm(),
                (residus[0].iter().map(|v| v.norm_squared()).sum::<f64>() / retenues.len() as f64).sqrt(),
                (residus[1].iter().map(|v| v.norm_squared()).sum::<f64>() / retenues.len() as f64).sqrt(),
                (d2 / retenues.len() as f64).sqrt()
            );
        }
        let rapport = (somme_diff2 / somme_v2).sqrt();
        let mut rms = tous_rms.clone();
        rms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| rms[((rms.len() - 1) as f64 * p) as usize];
        println!("--- {n_v} coupes par référence ({n_coupes} recalages)");
        for (m, nom) in ["axial", "sagittal"].iter().enumerate() {
            let hausse = toutes_ncc[m].iter().filter(|(a, b)| b > a).count() as f64 / toutes_ncc[m].len() as f64;
            println!("M1 référence {nom} : NCC en hausse pour {:.1} % des coupes ({} baisses)", hausse * 100.0, n_baisse[m]);
        }
        println!("M2 amplitude du déplacement estimé (RMS sur le masque) : médiane {:.2}, p90 {:.2}, max {:.2} mm ; {:.1} % des recalages > 6 mm", q(0.5), q(0.9), q(1.0), 100.0 * n_grand as f64 / n_coupes as f64);
        println!("M3 écart entre références / mouvement relatif : {rapport:.2} (critère ≤ 0,5)");
    }
    /// Résultat d'un recalage d'une coupe depuis un départ donné.
    struct Depart {
        ncc: f64,
        deplacement_rms: f64, // RMS, sur le masque, de T(x) − x
        distance_au_premier: f64, // RMS de T(x) − T_premier(x) (le premier départ est la pose nulle)
    }

    /// Recale chaque coupe retenue depuis chacun des `departs` (le premier doit être la pose nulle) ; une `Vec<Depart>` par coupe.
    fn recaler_multi_departs(mobile: &Stack, retenues: &[usize], reference: &VolumeTensors, device: &Device, departs: &[[f64; 6]]) -> Vec<Vec<Depart>> {
        let masque = mobile.brain_mask().unwrap().voxels();
        retenues
            .iter()
            .map(|&k| {
                let coupe = mobile.slice(k).unwrap();
                let (nx, ny) = coupe.dim();
                let donnees = coupe.data();
                let (mut points, mut intensites) = (Vec::new(), Vec::new());
                for j in 0..ny {
                    for i in 0..nx {
                        if masque[[i, j, k]] {
                            points.push(coupe.pixel_to_world(i as f64, j as f64));
                            intensites.push(f64::from(donnees[[i, j]]));
                        }
                    }
                }
                let pivot = coupe.brain_pivot().unwrap();
                let echelle = rotation_scale_mm(&points, &pivot).unwrap();
                let (tp, tc, ti, tm) = (
                    tenseur_points(&points, device),
                    tenseur_1d(&[pivot.x, pivot.y, pivot.z], device),
                    tenseur_1d(&intensites, device),
                    tenseur_1d(&vec![1.0; points.len()], device),
                );
                let mut poses: Vec<[f64; 6]> = Vec::new();
                let mut sorties: Vec<Depart> = Vec::new();
                for depart in departs {
                    let config = RegistrationConfig { initial: *depart, ..CONFIG_RECALAGE };
                    let r = register_slice(reference, tp.clone(), tc.clone(), echelle, ti.clone(), tm.clone(), config);
                    let rms_depuis = |autre: &[f64; 6]| {
                        (points.iter().map(|x| (deplace(&r.params, x, &pivot, echelle) - deplace(autre, x, &pivot, echelle)).norm_squared()).sum::<f64>() / points.len() as f64).sqrt()
                    };
                    sorties.push(Depart {
                        ncc: *r.ncc_history.last().unwrap(),
                        deplacement_rms: rms_depuis(&[0.0; 6]),
                        distance_au_premier: poses.first().map_or(0.0, |p0| rms_depuis(p0)),
                    });
                    poses.push(r.params);
                }
                sorties
            })
            .collect()
    }

    /// Diagnostic A : les recalages à un niveau, partis de la pose d'en-tête, sont-ils bloqués dans un minimum local ? Chaque coupe est
    /// recalée depuis la pose nulle et depuis 6 poses tirées dans ±5 mm ; on compare la NCC finale du départ nul au meilleur départ.
    /// À lancer en `--release --ignored --nocapture` (longue : une dizaine de minutes).
    #[test]
    #[ignore = "données locales ; longue"]
    fn multi_start_diagnostic_on_real_stacks() {
        let device = device().autodiff();
        let mut alea = Alea(8675309);
        let mut departs = vec![[0.0; 6]];
        for _ in 0..6 {
            departs.push(std::array::from_fn(|_| (alea.suivant() * 2.0 - 1.0) * 5.0));
        }
        let sujets = ["sub-S01", "sub-S02", "sub-S03", "sub-S05", "sub-S09", "sub-S10", "sub-S11", "sub-S13", "sub-S14"];
        // (rms_depuis_zero, gain_ncc, distance entre la pose du meilleur départ et celle du départ nul, nombre de départs à moins de 0,005 du meilleur)
        let mut lignes: Vec<(f64, f64, f64, usize)> = Vec::new();
        for sujet in sujets {
            let (chemin_b, masque_b) = stack_reel(sujet, "truficor");
            let mut mobile = Stack::read(&chemin_b).unwrap();
            mobile.set_brain_mask(&masque_b).unwrap();
            let aires: Vec<usize> = (0..mobile.dim().2).map(|k| mobile.brain_mask().unwrap().voxels().index_axis(ndarray::Axis(2), k).iter().filter(|&&v| v).count()).collect();
            let max_aire = *aires.iter().max().unwrap();
            let retenues: Vec<usize> = (0..aires.len()).filter(|&k| aires[k] as f64 >= 0.25 * max_aire as f64).collect();
            for acq in ["trufiax", "trufisag"] {
                let (chemin, _) = stack_reel(sujet, acq);
                let volume = Volume::from_stack(&Stack::read(&chemin).unwrap());
                let resultats = recaler_multi_departs(&mobile, &retenues, &VolumeTensors::new(&volume, &device), &device, &departs);
                let avant = lignes.len();
                for r in &resultats {
                    let meilleur = r.iter().enumerate().max_by(|a, b| a.1.ncc.partial_cmp(&b.1.ncc).unwrap()).unwrap();
                    let proches = r.iter().filter(|d| d.ncc >= meilleur.1.ncc - 0.005).count();
                    lignes.push((r[0].deplacement_rms, meilleur.1.ncc - r[0].ncc, if meilleur.0 == 0 { 0.0 } else { meilleur.1.distance_au_premier }, proches));
                }
                let gains: Vec<f64> = lignes[avant..].iter().map(|l| l.1).collect();
                println!("{sujet} {acq} : {} coupes ; gain de NCC du meilleur départ : médiane {:.3}, max {:.3} ; > 0,02 pour {} coupes", gains.len(), mediane(gains.clone()), gains.iter().cloned().fold(0.0, f64::max), gains.iter().filter(|&&g| g > 0.02).count());
            }
        }
        let n = lignes.len() as f64;
        let part = |seuil: f64| lignes.iter().filter(|l| l.1 > seuil).count() as f64 / n * 100.0;
        println!("--- {} recalages de coupe (9 sujets, 2 références)", lignes.len());
        println!("part des coupes dont le meilleur départ gagne plus de 0,005 / 0,02 / 0,05 de NCC : {:.1} % / {:.1} % / {:.1} %", part(0.005), part(0.02), part(0.05));
        println!("gain de NCC : médiane {:.4}, p90 {:.3}, max {:.3}", mediane(lignes.iter().map(|l| l.1).collect()), { let mut g: Vec<f64> = lignes.iter().map(|l| l.1).collect(); g.sort_by(|a, b| a.partial_cmp(b).unwrap()); g[(g.len() as f64 * 0.9) as usize] }, lignes.iter().map(|l| l.1).fold(0.0, f64::max));
        for (nom, bas, haut) in [("correction ≤ 3 mm", 0.0, 3.0), ("3 à 6 mm", 3.0, 6.0), ("> 6 mm", 6.0, f64::MAX)] {
            let groupe: Vec<&(f64, f64, f64, usize)> = lignes.iter().filter(|l| l.0 > bas && l.0 <= haut).collect();
            if groupe.is_empty() { continue; }
            let part2 = groupe.iter().filter(|l| l.1 > 0.02).count() as f64 / groupe.len() as f64 * 100.0;
            println!("  {nom} (départ nul) : {} coupes, {:.1} % avec un gain > 0,02", groupe.len(), part2);
        }
        let ameliorees: Vec<&(f64, f64, f64, usize)> = lignes.iter().filter(|l| l.1 > 0.02).collect();
        if !ameliorees.is_empty() {
            println!("pour les coupes à gain > 0,02 : le meilleur minimum est à {:.1} mm (médiane) de celui du départ nul ; départs à moins de 0,005 du meilleur : médiane {} sur {}", mediane(ameliorees.iter().map(|l| l.2).collect()), { let mut c: Vec<usize> = ameliorees.iter().map(|l| l.3).collect(); c.sort(); c[c.len() / 2] }, departs.len());
        }
    }
}
