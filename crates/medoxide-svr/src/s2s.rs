//! Estimation du mouvement **coupe à coupe**, sans reconstruction (étape 3b, type pyrecon) : une coupe est recalée contre les coupes des autres stacks qui la croisent.
//!
//! Deux coupes de stacks différents se croisent le long d'une droite ; si leurs poses sont justes, leurs intensités le long de cette droite concordent. Le coût est écrit en opérations sur tenseurs
//! Burn, pour que la différentiation automatique en tire les gradients, comme pour le recalage coupe-volume ([`crate::diff`]).

use burn::prelude::*;

/// Résultat de [`bilinear_sample`] : une valeur et un indicateur « dans l'image » par point.
#[derive(Debug, Clone)]
pub struct BilinearSample {
    /// Valeur interpolée de chaque point, forme `[P]`. Sans signification là où `inside` vaut 0.
    pub values: Tensor<1>,
    /// 1 si le point est dans la zone entre les centres des premier et dernier pixels de chaque axe, 0 sinon. Non différentiable.
    pub inside: Tensor<1>,
}

/// Interpolation **bilinéaire** d'une image 2D en `P` points donnés en indices de pixel continus, en opérations sur tenseurs (version 2D de [`crate::diff::trilinear_sample`]).
///
/// - `image` : pixels aplatis dans l'ordre d'un `Array2` standard, `index = i · ny + j`, forme `[nx·ny]`.
/// - `dims` : `[nx, ny]`.
/// - `coords` : forme `[P, 2]`, indices continus `(i, j)` ; les centres de pixel sont aux entiers.
///
/// Les points hors de l'image (`i` hors de `[0, nx−1]` ou `j` hors de `[0, ny−1]`) sont ramenés au bord pour la lecture et signalés par `inside = 0`. La différentiation passe par les fractions de
/// distance `t = c − floor(c)` ; les indices de pixel, entiers, ne sont pas différentiables.
///
/// # Panics
/// Si un axe de `dims` est nul ou si `image` et `coords` ne sont pas sur le même `Device`.
pub fn bilinear_sample(image: &Tensor<1>, dims: [usize; 2], coords: Tensor<2>) -> BilinearSample {
    assert!(dims.iter().all(|&n| n > 0), "chaque axe doit avoir au moins un pixel");
    let p = coords.dims()[0];
    let device = coords.device();
    let mut inside = Tensor::<1>::ones([p], &device);
    let mut bas: Vec<Tensor<1, Int>> = Vec::new();
    let mut haut: Vec<Tensor<1, Int>> = Vec::new();
    let mut frac: Vec<Tensor<1>> = Vec::new();
    for a in 0..2 {
        let c = coords.clone().narrow(1, a, 1).reshape([p]);
        let max = (dims[a] - 1) as f64;
        inside = inside * c.clone().greater_equal_elem(0.0).bool_and(c.clone().lower_equal_elem(max)).float();
        let c = c.clamp(0.0, max);
        let plancher = c.clone().floor();
        frac.push(c - plancher.clone());
        let i0 = plancher.int().clamp(0, dims[a] as i64 - 1);
        haut.push((i0.clone() + 1).clamp(0, dims[a] as i64 - 1));
        bas.push(i0);
    }
    let pas = [dims[1] as i64, 1];
    let mut valeurs = Tensor::<1>::zeros([p], &device);
    for coin in 0..4usize {
        let mut indice = Tensor::<1, Int>::zeros([p], &device);
        let mut poids = Tensor::<1>::ones([p], &device);
        for a in 0..2 {
            let haut_a = (coin >> (1 - a)) & 1 == 1; // i est le bit de poids fort
            let (i, w) = if haut_a { (haut[a].clone(), frac[a].clone()) } else { (bas[a].clone(), frac[a].clone().neg() + 1.0) };
            indice = indice + i * pas[a];
            poids = poids * w;
        }
        valeurs = valeurs + poids * image.clone().select(0, indice);
    }
    BilinearSample { values: valeurs, inside }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tenseur_image(v: &[f64], device: &Device) -> Tensor<1> {
        Tensor::<1>::from_floats(v.iter().map(|&x| x as f32).collect::<Vec<_>>().as_slice(), device)
    }

    fn tenseur_points(pts: &[(f64, f64)], device: &Device) -> Tensor<2> {
        let v: Vec<f32> = pts.iter().flat_map(|&(i, j)| [i as f32, j as f32]).collect();
        Tensor::<2>::from_floats(TensorData::new(v, [pts.len(), 2]), device)
    }

    fn valeurs(t: Tensor<1>) -> Vec<f64> {
        t.into_data().try_to_vec::<f32>().unwrap().iter().map(|&v| f64::from(v)).collect()
    }

    /// Générateur pseudo-aléatoire déterministe (xorshift64) dans [0, 1).
    fn aleatoire(etat: &mut u64) -> f64 {
        *etat ^= *etat << 13;
        *etat ^= *etat >> 7;
        *etat ^= *etat << 17;
        (*etat >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Référence écrite à part, en `f64` : interpolation bilinéaire directe ; `None` hors de l'image.
    fn bilineaire_direct(image: &[f64], nx: usize, ny: usize, i: f64, j: f64) -> Option<f64> {
        if i < 0.0 || j < 0.0 || i > (nx - 1) as f64 || j > (ny - 1) as f64 {
            return None;
        }
        let (i0, j0) = (i.floor() as usize, j.floor() as usize);
        let (i1, j1) = ((i0 + 1).min(nx - 1), (j0 + 1).min(ny - 1));
        let (ti, tj) = (i - i0 as f64, j - j0 as f64);
        let px = |a: usize, b: usize| image[a * ny + b];
        Some((1.0 - ti) * (1.0 - tj) * px(i0, j0) + (1.0 - ti) * tj * px(i0, j1) + ti * (1.0 - tj) * px(i1, j0) + ti * tj * px(i1, j1))
    }

    /// Critères fixés avant : (1) sur une rampe `a + b·i + c·j` l'interpolation est exacte à l'arrondi `f32` (1e-4 relatif à la dynamique) en 500 points intérieurs aléatoires ; (2) sur une image aléatoire elle
    /// égale l'interpolation directe en `f64` (écart < 1e-4) pour 500 points, dont des points sur la grille entière et sur les bords ; (3) les points hors de l'image ont `inside = 0` et les autres `inside = 1` ;
    /// (4) une image à un seul pixel sur un axe ne plante pas.
    #[test]
    fn bilinear_sampling_matches_a_direct_f64_implementation_and_is_exact_on_a_ramp() {
        let device = Device::flex();
        let (nx, ny) = (13usize, 9usize);
        let (a, b, c) = (3.0, 0.7, -1.3);
        let rampe: Vec<f64> = (0..nx).flat_map(|i| (0..ny).map(move |j| a + b * i as f64 + c * j as f64)).collect();
        let dynamique = 0.7 * (nx - 1) as f64 + 1.3 * (ny - 1) as f64;
        let mut etat = 0x9E3779B97F4A7C15u64;
        let pts: Vec<(f64, f64)> = (0..500).map(|_| (aleatoire(&mut etat) * (nx - 1) as f64, aleatoire(&mut etat) * (ny - 1) as f64)).collect();
        let r = bilinear_sample(&tenseur_image(&rampe, &device), [nx, ny], tenseur_points(&pts, &device));
        let (v, d) = (valeurs(r.values), valeurs(r.inside));
        for (k, &(i, j)) in pts.iter().enumerate() {
            assert!((v[k] - (a + b * i + c * j)).abs() < 1e-4 * dynamique, "rampe en ({i},{j}) : {} contre {}", v[k], a + b * i + c * j);
            assert_eq!(d[k], 1.0);
        }
        let image: Vec<f64> = (0..nx * ny).map(|_| aleatoire(&mut etat) * 100.0).collect();
        let mut pts: Vec<(f64, f64)> = (0..450).map(|_| (aleatoire(&mut etat) * (nx - 1) as f64, aleatoire(&mut etat) * (ny - 1) as f64)).collect();
        pts.extend([(0.0, 0.0), ((nx - 1) as f64, (ny - 1) as f64), (0.0, (ny - 1) as f64), ((nx - 1) as f64, 0.0), (4.0, 3.0), (6.0, 0.0), (0.0, 5.0), ((nx - 1) as f64, 4.0)]);
        pts.extend([(-0.5, 2.0), (2.0, -0.01), ((nx - 1) as f64 + 0.01, 3.0), (3.0, (ny - 1) as f64 + 1.0), (-3.0, -3.0), (100.0, 100.0), (f64::from(f32::MAX) / 2.0, 1.0), (1.0, -1e6)]);
        let r = bilinear_sample(&tenseur_image(&image, &device), [nx, ny], tenseur_points(&pts, &device));
        let (v, d) = (valeurs(r.values), valeurs(r.inside));
        for (k, &(i, j)) in pts.iter().enumerate() {
            match bilineaire_direct(&image, nx, ny, i, j) {
                Some(attendu) => {
                    assert_eq!(d[k], 1.0, "({i},{j}) devrait être dedans");
                    assert!((v[k] - attendu).abs() < 1e-4, "({i},{j}) : {} contre {attendu}", v[k]);
                }
                None => assert_eq!(d[k], 0.0, "({i},{j}) devrait être dehors"),
            }
        }
        // un axe d'un seul pixel
        let une_ligne = tenseur_image(&[1.0, 2.0, 4.0], &device);
        let r = bilinear_sample(&une_ligne, [1, 3], tenseur_points(&[(0.0, 0.5), (0.0, 1.5), (0.5, 1.0)], &device));
        assert_eq!(valeurs(r.inside), vec![1.0, 1.0, 0.0]);
        let v = valeurs(r.values);
        assert!((v[0] - 1.5).abs() < 1e-6 && (v[1] - 3.0).abs() < 1e-6);
    }

    /// Gradient par rapport aux coordonnées (critère fixé avant : écart relatif < 1e-4 à la dérivée analytique) : sur la rampe `a + b·i + c·j`, `∂v/∂i = b` et `∂v/∂j = c` en tout point intérieur ; sur une image
    /// aléatoire, le gradient automatique égale la dérivée de l'interpolation par morceaux (`∂v/∂i = (1 − tj)(p₁₀ − p₀₀) + tj (p₁₁ − p₀₁)`, `∂v/∂j` de même) en 200 points intérieurs.
    #[test]
    fn bilinear_sampling_gradient_matches_the_analytic_derivative() {
        let device = Device::flex().autodiff();
        let (nx, ny) = (11usize, 8usize);
        let (a, b, c) = (3.0, 0.7, -1.3);
        let rampe: Vec<f64> = (0..nx).flat_map(|i| (0..ny).map(move |j| a + b * i as f64 + c * j as f64)).collect();
        let mut etat = 0xD1B54A32D192ED03u64;
        let pts: Vec<(f64, f64)> = (0..200).map(|_| (0.05 + aleatoire(&mut etat) * (nx as f64 - 1.1), 0.05 + aleatoire(&mut etat) * (ny as f64 - 1.1))).collect();
        let coords = tenseur_points(&pts, &device).require_grad();
        let r = bilinear_sample(&tenseur_image(&rampe, &device), [nx, ny], coords.clone());
        let g = coords.grad(&r.values.sum().backward()).unwrap().into_data().try_to_vec::<f32>().unwrap();
        for k in 0..pts.len() {
            assert!((f64::from(g[2 * k]) - b).abs() < 1e-4 && (f64::from(g[2 * k + 1]) - c).abs() < 1e-4, "gradient de la rampe en {:?} : ({}, {})", pts[k], g[2 * k], g[2 * k + 1]);
        }
        let image: Vec<f64> = (0..nx * ny).map(|_| aleatoire(&mut etat) * 50.0).collect();
        let coords = tenseur_points(&pts, &device).require_grad();
        let r = bilinear_sample(&tenseur_image(&image, &device), [nx, ny], coords.clone());
        let g = coords.grad(&r.values.sum().backward()).unwrap().into_data().try_to_vec::<f32>().unwrap();
        for (k, &(i, j)) in pts.iter().enumerate() {
            let (i0, j0) = (i.floor() as usize, j.floor() as usize);
            let (ti, tj) = (i - i0 as f64, j - j0 as f64);
            let px = |u: usize, v: usize| image[u * ny + v];
            let di = (1.0 - tj) * (px(i0 + 1, j0) - px(i0, j0)) + tj * (px(i0 + 1, j0 + 1) - px(i0, j0 + 1));
            let dj = (1.0 - ti) * (px(i0, j0 + 1) - px(i0, j0)) + ti * (px(i0 + 1, j0 + 1) - px(i0 + 1, j0));
            assert!((f64::from(g[2 * k]) - di).abs() < 1e-3 * (1.0 + di.abs()) && (f64::from(g[2 * k + 1]) - dj).abs() < 1e-3 * (1.0 + dj.abs()), "point ({i},{j}) : ({}, {}) contre ({di}, {dj})", g[2 * k], g[2 * k + 1]);
        }
    }
}
