//! Coût **différentiable** du recalage des coupes (étape 3) : les calculs sont écrits en opérations sur
//! tenseurs Burn, pour que la différentiation automatique en tire les gradients.
//!
//! Burn choisit son backend à l'exécution par un `Device` : tous les tenseurs d'un calcul vivent sur le
//! même `Device`.

use burn::prelude::*;

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
}
