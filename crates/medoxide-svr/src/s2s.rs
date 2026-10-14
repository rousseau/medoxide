//! Estimation du mouvement **coupe à coupe**, sans reconstruction (étape 3b, type pyrecon) : une coupe est recalée contre les coupes des autres stacks qui la croisent.
//!
//! Deux coupes de stacks différents se croisent le long d'une droite ; si leurs poses sont justes, leurs intensités le long de cette droite concordent. Le coût est écrit en opérations sur tenseurs
//! Burn, pour que la différentiation automatique en tire les gradients, comme pour le recalage coupe-volume ([`crate::diff`]).

use burn::module::Param;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;
use nalgebra::{Matrix4, Vector3, Vector4};
use ndarray::{Array2, ArrayView2};

use crate::diff::{ncc, rotation_matrix, PoseModule, RegistrationConfig, RegistrationResult};
use crate::align::{preparer_coupe, verifier_entrees, AlignmentReport, SliceReport};
use crate::diff::pose_to_matrix;
use crate::recon::SlicePoses;
use crate::{Stack, SvrError};

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

/// Les images et masques de TOUTES les coupes de tous les stacks, aplatis dans deux tenseurs : un échantillonnage par lot peut alors lire des coupes différentes dans un seul appel.
/// Constant tout au long d'un recalage (seules les poses changent).
#[derive(Debug, Clone)]
pub struct SliceAtlas {
    images: Tensor<1>,
    masks: Tensor<1>,
    /// `entries[stack][coupe] = (décalage dans l'atlas, [nx, ny])`.
    entries: Vec<Vec<(usize, [usize; 2])>>,
}

/// Flou gaussien 2D séparable d'une coupe **dans son masque** (convolution normalisée) : `flou(image · masque) / flou(masque)`, pour que le fond (hors masque) n'entre pas dans les pixels voisins. `sigma_pixels` = 0 : l'image telle quelle.
/// Les bords de la coupe sont traités par répétition du pixel de bord. Un pixel de masque dont le voisinage n'a aucun poids garde sa valeur.
pub(crate) fn blur_slice_in_mask(image: ArrayView2<f32>, masque: ArrayView2<bool>, sigma_pixels: f64) -> Array2<f32> {
    if sigma_pixels <= 0.0 {
        return image.to_owned();
    }
    let rayon = (3.0 * sigma_pixels).ceil() as isize;
    let noyau: Vec<f64> = (-rayon..=rayon).map(|i| (-0.5 * (i as f64 / sigma_pixels).powi(2)).exp()).collect();
    // pas de normalisation du noyau : elle s'annulerait dans le rapport signal / masque
    let mut valeurs = image.mapv(|v| f64::from(v));
    let mut poids = masque.mapv(|m| if m { 1.0_f64 } else { 0.0 });
    valeurs.zip_mut_with(&poids, |v, &w| *v *= w);
    for axe in 0..2 {
        for tableau in [&mut valeurs, &mut poids] {
            for mut ligne in tableau.lanes_mut(ndarray::Axis(axe)) {
                let n = ligne.len() as isize;
                let source: Vec<f64> = ligne.iter().copied().collect();
                for (i, v) in ligne.iter_mut().enumerate() {
                    *v = noyau.iter().enumerate().map(|(t, w)| w * source[(i as isize + t as isize - rayon).clamp(0, n - 1) as usize]).sum();
                }
            }
        }
    }
    let mut sortie = image.to_owned();
    for ((i, j), v) in sortie.indexed_iter_mut() {
        if masque[[i, j]] && poids[[i, j]] > 1e-12 {
            *v = (valeurs[[i, j]] / poids[[i, j]]) as f32;
        }
    }
    sortie
}

impl SliceAtlas {
    /// Construit l'atlas de `stacks` (masques requis) sur `device`.
    ///
    /// # Erreurs
    /// [`SvrError::NoMask`] si un stack n'a pas de masque.
    pub fn new(stacks: &[Stack], device: &Device) -> Result<SliceAtlas, SvrError> {
        SliceAtlas::new_blurred(stacks, device, 0.0)
    }

    /// Comme [`SliceAtlas::new`], mais chaque coupe est floutée dans son masque (écart type `sigma_pixels`, voir [`blur_slice_in_mask`]) : des profils plus lisses élargissent le bassin d'attraction du recalage (gros vers fin).
    ///
    /// # Erreurs
    /// [`SvrError::NoMask`] si un stack n'a pas de masque.
    pub fn new_blurred(stacks: &[Stack], device: &Device, sigma_pixels: f64) -> Result<SliceAtlas, SvrError> {
        let (mut images, mut masks, mut entries) = (Vec::new(), Vec::new(), Vec::new());
        for stack in stacks {
            let masque = stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?.voxels();
            let (nx, ny, nz) = stack.dim();
            let mut e = Vec::with_capacity(nz);
            for k in 0..nz {
                e.push((images.len(), [nx, ny]));
                let lisse = blur_slice_in_mask(stack.data().index_axis(ndarray::Axis(2), k), masque.index_axis(ndarray::Axis(2), k), sigma_pixels);
                for i in 0..nx {
                    for j in 0..ny {
                        images.push(lisse[[i, j]]);
                        masks.push(if masque[[i, j, k]] { 1.0_f32 } else { 0.0 });
                    }
                }
            }
            entries.push(e);
        }
        Ok(SliceAtlas { images: Tensor::<1>::from_floats(images.as_slice(), device), masks: Tensor::<1>::from_floats(masks.as_slice(), device), entries })
    }
}

/// Un indice par point d'échantillonnage dans l'atlas : décalage de la coupe, dimensions de la coupe.
struct AtlasIndex {
    offset: Tensor<1, Int>,
    ny: Tensor<1, Int>,
    max_i: Tensor<1>,
    max_j: Tensor<1>,
    max_i_int: Tensor<1, Int>,
    max_j_int: Tensor<1, Int>,
}

/// Comme [`bilinear_sample`], mais chaque point lit sa propre coupe de l'atlas (décalage et dimensions propres à chaque point).
fn bilinear_sample_atlas(atlas: &Tensor<1>, idx: &AtlasIndex, coords: Tensor<2>) -> BilinearSample {
    let p = coords.dims()[0];
    let device = coords.device();
    let mut inside = Tensor::<1>::ones([p], &device);
    let (mut bas, mut haut, mut frac) = (Vec::new(), Vec::new(), Vec::new());
    for (a, (max, max_int)) in [(&idx.max_i, &idx.max_i_int), (&idx.max_j, &idx.max_j_int)].into_iter().enumerate() {
        let c = coords.clone().narrow(1, a, 1).reshape([p]);
        inside = inside * c.clone().greater_equal_elem(0.0).bool_and(c.clone().lower_equal(max.clone())).float();
        let c = c.clamp_min(0.0).min_pair(max.clone());
        let plancher = c.clone().floor();
        frac.push(c - plancher.clone());
        let i0 = plancher.int().min_pair(max_int.clone());
        haut.push((i0.clone() + 1).min_pair(max_int.clone()));
        bas.push(i0);
    }
    let mut valeurs = Tensor::<1>::zeros([p], &device);
    for coin in 0..4usize {
        let mut poids = Tensor::<1>::ones([p], &device);
        let mut i_choisi = None;
        let mut j_choisi = None;
        for a in 0..2 {
            let haut_a = (coin >> (1 - a)) & 1 == 1;
            let (indice, w) = if haut_a { (haut[a].clone(), frac[a].clone()) } else { (bas[a].clone(), frac[a].clone().neg() + 1.0) };
            poids = poids * w;
            if a == 0 {
                i_choisi = Some(indice);
            } else {
                j_choisi = Some(indice);
            }
        }
        let indice = idx.offset.clone() + i_choisi.unwrap() * idx.ny.clone() + j_choisi.unwrap();
        valeurs = valeurs + poids * atlas.clone().select(0, indice);
    }
    BilinearSample { values: valeurs, inside }
}

fn produit_vectoriel(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    let c = |t: &Tensor<2>, i: usize| t.clone().narrow(1, i, 1);
    Tensor::cat(
        vec![
            c(&a, 1) * c(&b, 2) - c(&a, 2) * c(&b, 1),
            c(&a, 2) * c(&b, 0) - c(&a, 0) * c(&b, 2),
            c(&a, 0) * c(&b, 1) - c(&a, 1) * c(&b, 0),
        ],
        1,
    )
}

/// Réglages de l'échantillonnage des droites d'intersection.
#[derive(Debug, Clone, Copy)]
pub struct IntersectionConfig {
    /// Angle minimal (°) entre les plans de deux coupes pour qu'elles forment une paire (30 : en deçà, la droite d'intersection est mal définie).
    pub min_angle_deg: f64,
    /// Pas d'échantillonnage le long de la droite (mm).
    pub step_mm: f64,
    /// Demi-longueur de la droite échantillonnée autour de son point d'origine (mm).
    pub reach_mm: f64,
    /// Nombre minimal de points communs aux deux coupes et à leurs masques pour qu'une paire compte (20, soit 10 mm à 0,5 mm).
    pub min_points: usize,
    /// Stack d'ancrage : si `Some(a)`, seules les paires dont une des deux coupes est dans le stack `a` sont formées (les paires entre deux autres stacks sont ignorées). Sert à garder des paires
    /// non utilisées par l'estimation pour les évaluer (`None` : toutes les paires).
    pub anchor_stack: Option<usize>,
    /// Passe-haut des profils : écart type (mm) du lissage gaussien (restreint aux pixels de masque) soustrait à chaque profil avant la corrélation, pour ignorer les variations lentes de l'intensité (champ de biais, gain) ; 0 : aucun.
    pub highpass_sigma_mm: f64,
}

impl Default for IntersectionConfig {
    fn default() -> Self {
        IntersectionConfig { min_angle_deg: 30.0, step_mm: 0.5, reach_mm: 150.0, min_points: 20, anchor_stack: None, highpass_sigma_mm: 0.0 }
    }
}

/// Plan d'une coupe à sa pose : matrice pixel → monde, son inverse, centre et normale unitaire (monde).
struct Plan {
    inverse: Matrix4<f64>,
    centre: Vector3<f64>,
    normale: Vector3<f64>,
}

fn plan_de(stack: &Stack, k: usize, pose: &Matrix4<f64>) -> Plan {
    let ma = pose * stack.affine();
    let (nx, ny, _) = stack.dim();
    Plan {
        inverse: ma.try_inverse().expect("l'affine d'une coupe est inversible"),
        centre: (ma * Vector4::new((nx - 1) as f64 / 2.0, (ny - 1) as f64 / 2.0, k as f64, 1.0)).xyz(),
        normale: ma.fixed_view::<3, 1>(0, 2).into_owned().normalize(),
    }
}

/// Matrice `[T, T]` du lissage gaussien le long d'un profil : `K[a][b] = exp(−½ ((tₐ − t_b) / σ)²)` pour les abscisses `t` (mm).
pub(crate) fn gaussian_smoothing_matrix(t: &[f64], sigma_mm: f64, device: &Device) -> Tensor<2> {
    let n = t.len();
    let valeurs: Vec<f32> = (0..n).flat_map(|a| (0..n).map(move |b| (-0.5 * ((t[a] - t[b]) / sigma_mm).powi(2)).exp() as f32)).collect();
    Tensor::<2>::from_floats(TensorData::new(valeurs, [n, n]), device)
}

/// Passe-haut de profils `p` (`[J, T]`) : `p − (p ⊙ w) K / (w K)`, le lissage gaussien (matrice `k`) étant **restreint aux points de poids `w` non nul** (convolution normalisée) pour que le fond n'entre pas dans les points voisins. Un profil constant
/// ou linéaire (loin des bords) donne un résidu nul ; une oscillation bien plus rapide que le lissage est conservée.
pub(crate) fn highpass_profiles(p: Tensor<2>, w: Tensor<2>, k: Tensor<2>) -> Tensor<2> {
    let numerateur = (p.clone() * w.clone()).matmul(k.clone());
    let denominateur = w.matmul(k).clamp_min(1e-6);
    p - numerateur / denominateur
}

/// Le problème de recalage coupe-coupe d'UNE coupe (la coupe mobile) : ses partenaires (les coupes des AUTRES stacks qui la croisent, à leur pose courante, gardées si la droite d'intersection a au moins
/// `min_points` points communs aux deux coupes et à leurs masques), et tout ce qu'il faut pour calculer [`IntersectionProblem::cost`].
pub struct IntersectionProblem {
    n_pairs: usize,
    t_grid: Tensor<1>,
    min_points: f64,
    // coupe mobile
    image: Tensor<1>,
    mask: Tensor<1>,
    dims: [usize; 2],
    normal: Tensor<1>,
    centre: Tensor<1>,
    to_pixel_t: Tensor<2>,
    to_pixel_offset: Tensor<1>,
    // partenaires
    partner_normals: Tensor<2>,
    partner_offsets: Tensor<2>,
    partner_to_pixel_t: Tensor<3>,
    partner_to_pixel_offset: Tensor<2>,
    index: AtlasIndex,
    atlas_images: Tensor<1>,
    atlas_masks: Tensor<1>,
    /// Matrice du lissage gaussien du passe-haut des profils (`None` : pas de passe-haut).
    highpass: Option<Tensor<2>>,
    /// Indices (stack, coupe) des partenaires, dans l'ordre des paires.
    pub partners: Vec<(usize, usize)>,
}

impl IntersectionProblem {
    /// Construit le problème de la coupe `k` du stack `si` à la pose courante `poses`. `None` si elle n'a aucun partenaire (aucune paire valide).
    pub fn build(atlas: &SliceAtlas, stacks: &[Stack], poses: &SlicePoses, si: usize, k: usize, config: &IntersectionConfig, device: &Device) -> Option<IntersectionProblem> {
        let mobile = plan_de(&stacks[si], k, poses.get(si, k));
        let masque_mobile = stacks[si].brain_mask()?.voxels();
        let (nx, ny, _) = stacks[si].dim();
        let sin_min = config.min_angle_deg.to_radians().sin();
        let n_t = (2.0 * config.reach_mm / config.step_mm).round() as usize + 1;
        let t_valeurs: Vec<f64> = (0..n_t).map(|i| -config.reach_mm + i as f64 * config.step_mm).collect();
        let mut partenaires: Vec<(usize, usize, Plan)> = Vec::new();
        for (sj, stack) in stacks.iter().enumerate() {
            if sj == si || config.anchor_stack.is_some_and(|a| si != a && sj != a) {
                continue;
            }
            let masque = stack.brain_mask()?.voxels();
            let (mx, my, mz) = stack.dim();
            for j in 0..mz {
                let autre = plan_de(stack, j, poses.get(sj, j));
                let d = mobile.normale.cross(&autre.normale);
                if d.norm() < sin_min {
                    continue;
                }
                let x0 = ((mobile.normale.dot(&mobile.centre)) * autre.normale.cross(&d) + (autre.normale.dot(&autre.centre)) * d.cross(&mobile.normale)) / d.norm_squared();
                let dh = d / d.norm();
                let mut compte = 0usize;
                for &t in &t_valeurs {
                    let p = (x0 + t * dh).push(1.0);
                    let (a, b) = (mobile.inverse * p, autre.inverse * p);
                    let (ia, ja, ib, jb) = (a.x.round(), a.y.round(), b.x.round(), b.y.round());
                    if a.x >= 0.0 && a.x <= (nx - 1) as f64 && a.y >= 0.0 && a.y <= (ny - 1) as f64 && b.x >= 0.0 && b.x <= (mx - 1) as f64 && b.y >= 0.0 && b.y <= (my - 1) as f64
                        && masque_mobile[[ia as usize, ja as usize, k]]
                        && masque[[ib as usize, jb as usize, j]]
                    {
                        compte += 1;
                    }
                }
                if compte >= config.min_points {
                    partenaires.push((sj, j, autre));
                }
            }
        }
        if partenaires.is_empty() {
            return None;
        }
        let jn = partenaires.len();
        let lisse = |v: &[f64]| Tensor::<1>::from_floats(v.iter().map(|&x| x as f32).collect::<Vec<_>>().as_slice(), device);
        let fk = mobile.inverse.fixed_view::<3, 3>(0, 0).into_owned();
        let mut pn = Vec::new();
        let mut ph = Vec::new();
        let mut po = Vec::new();
        let (mut off, mut nyv, mut mi, mut mj) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (sj, j, plan) in &partenaires {
            pn.extend(plan.normale.iter().copied());
            ph.push(plan.normale.dot(&plan.centre));
            po.extend(plan.inverse.fixed_view::<3, 1>(0, 3).iter().copied());
            let (decalage, [px, py]) = atlas.entries[*sj][*j];
            for _ in 0..n_t {
                off.push(decalage as i64);
                nyv.push(py as i64);
                mi.push((px - 1) as f64);
                mj.push((py - 1) as f64);
            }
        }
        // pour chaque partenaire, Fᵀ en ordre LIGNE (Burn) : (Fᵀ)[r][c] = F[c][r]
        let mut pf_t: Vec<f64> = Vec::new();
        for (_, _, plan) in &partenaires {
            let f = plan.inverse.fixed_view::<3, 3>(0, 0).into_owned();
            for r in 0..3 {
                for c in 0..3 {
                    pf_t.push(f[(c, r)]); // (Fᵀ)[r][c] = F[c][r]
                }
            }
        }
        let int = |v: &[i64]| Tensor::<1, Int>::from_ints(v, device);
        let max_i_int: Vec<i64> = mi.iter().map(|&v| v as i64).collect();
        let max_j_int: Vec<i64> = mj.iter().map(|&v| v as i64).collect();
        Some(IntersectionProblem {
            n_pairs: jn,
            t_grid: lisse(&t_valeurs),
            min_points: config.min_points as f64,
            image: atlas.images.clone().narrow(0, atlas.entries[si][k].0, nx * ny),
            mask: atlas.masks.clone().narrow(0, atlas.entries[si][k].0, nx * ny),
            dims: [nx, ny],
            normal: lisse(mobile.normale.as_slice()),
            centre: lisse(mobile.centre.as_slice()),
            to_pixel_t: Tensor::<2>::from_floats(TensorData::new(fk.iter().map(|&x| x as f32).collect::<Vec<_>>(), [3, 3]), device), // ordre colonne de F = ordre ligne de Fᵀ
            to_pixel_offset: lisse(mobile.inverse.fixed_view::<3, 1>(0, 3).iter().copied().collect::<Vec<f64>>().as_slice()),
            partner_normals: Tensor::<2>::from_floats(TensorData::new(pn.iter().map(|&x| x as f32).collect::<Vec<_>>(), [jn, 3]), device),
            partner_offsets: Tensor::<2>::from_floats(TensorData::new(ph.iter().map(|&x| x as f32).collect::<Vec<_>>(), [jn, 1]), device),
            partner_to_pixel_t: Tensor::<3>::from_floats(TensorData::new(pf_t.iter().map(|&x| x as f32).collect::<Vec<_>>(), [jn, 3, 3]), device),
            partner_to_pixel_offset: Tensor::<2>::from_floats(TensorData::new(po.iter().map(|&x| x as f32).collect::<Vec<_>>(), [jn, 3]), device),
            index: AtlasIndex { offset: int(&off), ny: int(&nyv), max_i: lisse(&mi), max_j: lisse(&mj), max_i_int: int(&max_i_int), max_j_int: int(&max_j_int) },
            atlas_images: atlas.images.clone(),
            atlas_masks: atlas.masks.clone(),
            highpass: (config.highpass_sigma_mm > 0.0).then(|| gaussian_smoothing_matrix(&t_valeurs, config.highpass_sigma_mm, device)),
            partners: partenaires.iter().map(|(s, j, _)| (*s, *j)).collect(),
        })
    }

    /// Nombre de paires (partenaires gardés).
    pub fn n_pairs(&self) -> usize {
        self.n_pairs
    }

    /// **Concordance moyenne** de la coupe mobile avec ses partenaires, pour la correction `params` (`[6]`, comme [`crate::diff::apply_pose`] : rotation en mm équivalents autour de `pivot`, `[3]`) :
    /// la moyenne, sur les paires valides, de la NCC des deux profils d'intensité échantillonnés le long de la droite d'intersection de la coupe corrigée avec chaque partenaire. À MAXIMISER ; forme `[1]`.
    /// Une paire est valide si au moins `min_points` points sont dans les deux coupes et dans leurs masques (calculé sans dérivée). Les positions des points dépendent de `params` : le gradient existe
    /// partout où l'échantillonnage bilinéaire est différentiable.
    pub fn cost(&self, params: Tensor<1>, pivot: Tensor<1>, rotation_scale_mm: f64) -> Tensor<1> {
        let (j, t) = (self.n_pairs, self.t_grid.dims()[0]);
        let phi = params.clone().narrow(0, 0, 3);
        let tr = params.narrow(0, 3, 3).reshape([1, 3]);
        let r = rotation_matrix(phi / rotation_scale_mm);
        let pivot = pivot.reshape([1, 3]);
        // plan de la coupe mobile après correction : normale R n, centre c + R (c_k − c) + t
        let n_mob = self.normal.clone().reshape([1, 3]).matmul(r.clone().transpose());
        let c_mob = pivot.clone() + (self.centre.clone().reshape([1, 3]) - pivot.clone()).matmul(r.clone().transpose()) + tr.clone();
        let h1 = (n_mob.clone() * c_mob).sum().reshape([1, 1]);
        let n_mob_j = n_mob.expand([j, 3]);
        let d = produit_vectoriel(n_mob_j.clone(), self.partner_normals.clone());
        let d2 = (d.clone() * d.clone()).sum_dim(1).clamp_min(1e-6);
        let dh = d.clone() / d2.clone().sqrt();
        let x0 = (produit_vectoriel(self.partner_normals.clone(), d.clone()) * h1 + produit_vectoriel(d, n_mob_j) * self.partner_offsets.clone()) / d2;
        let x = x0.reshape([j, 1, 3]) + dh.reshape([j, 1, 3]) * self.t_grid.clone().reshape([1, t, 1]);
        // coordonnées de pixel dans la coupe mobile : D⁻¹(x) = c + (x − c − t) R, puis l'inverse de sa pose de départ
        let y = pivot.clone() + (x.clone().reshape([j * t, 3]) - pivot - tr).matmul(r);
        let u = y.matmul(self.to_pixel_t.clone()) + self.to_pixel_offset.clone().reshape([1, 3]);
        let v = (x.matmul(self.partner_to_pixel_t.clone()) + self.partner_to_pixel_offset.clone().reshape([j, 1, 3])).reshape([j * t, 3]);
        let (u2, v2) = (u.narrow(1, 0, 2), v.narrow(1, 0, 2));
        let mobile = bilinear_sample(&self.image, self.dims, u2.clone());
        let masque_mobile = bilinear_sample(&self.mask, self.dims, u2).values.greater_elem(0.5).float();
        let fixe = bilinear_sample_atlas(&self.atlas_images, &self.index, v2.clone());
        let masque_fixe = bilinear_sample_atlas(&self.atlas_masks, &self.index, v2).values.greater_elem(0.5).float();
        let w = (mobile.inside * fixe.inside * masque_mobile * masque_fixe).reshape([j, t]);
        let (profil_mobile, profil_fixe) = (mobile.values.reshape([j, t]), fixe.values.reshape([j, t]));
        let (profil_mobile, profil_fixe) = match &self.highpass {
            Some(k) => (highpass_profiles(profil_mobile, w.clone(), k.clone()), highpass_profiles(profil_fixe, w.clone(), k.clone())),
            None => (profil_mobile, profil_fixe),
        };
        let ncc_paires = ncc(profil_mobile, profil_fixe, w.clone());
        let valides = w.sum_dim(1).reshape([j]).greater_equal_elem(self.min_points).float();
        (ncc_paires * valides.clone()).sum() / valides.sum().clamp_min(1.0)
    }
}

/// **Recale une coupe par concordance aux intersections** : cherche la correction `(φx, φy, φz, tx, ty, tz)` (mm équivalents autour de `pivot`, départ `config.initial`) qui maximise
/// [`IntersectionProblem::cost`], par Adam (Burn) sur le gradient automatique, comme [`crate::diff::register_slice`] mais sans volume. Le problème fixe la pose courante de la coupe
/// et de ses partenaires : la nouvelle pose de la coupe est `D(params) · M`, avec `M` sa pose au moment de [`IntersectionProblem::build`].
///
/// `ncc_history` du résultat contient la concordance moyenne à chaque itération (avant la mise à jour) puis à la correction finale. Les tenseurs doivent être sur un `Device` avec autodiff.
pub fn register_slice_s2s(problem: &IntersectionProblem, pivot: Tensor<1>, rotation_scale_mm: f64, config: RegistrationConfig) -> RegistrationResult {
    let device = pivot.device();
    let depart: Vec<f32> = config.initial.iter().map(|&v| v as f32).collect();
    let mut module = PoseModule { params: Param::from_tensor(Tensor::<1>::from_floats(depart.as_slice(), &device)) };
    let mut optimiseur = AdamConfig::new().init();
    let valeur = |t: &Tensor<1>| f64::from(t.clone().into_data().try_to_vec::<f32>().unwrap()[0]);
    let mut historique = Vec::with_capacity(config.iterations + 1);
    let mut pas = config.learning_rate;
    for _ in 0..config.iterations {
        let concordance = problem.cost(module.params.val(), pivot.clone(), rotation_scale_mm);
        historique.push(valeur(&concordance));
        let gradients = GradientsParams::from_grads(concordance.neg().sum().backward(), &module); // perte = −concordance
        module = optimiseur.step(pas, module, gradients);
        pas *= config.lr_decay;
    }
    let finale = module.params.val();
    historique.push(valeur(&problem.cost(finale.clone(), pivot, rotation_scale_mm)));
    let params: Vec<f32> = finale.into_data().try_to_vec::<f32>().unwrap();
    RegistrationResult { params: std::array::from_fn(|i| f64::from(params[i])), ncc_history: historique }
}

/// Réglages de [`register_slices_s2s`].
#[derive(Debug, Clone, Copy)]
pub struct S2sConfig {
    /// Nombre de balayages : chaque balayage recale toutes les coupes une fois, dans un ordre aléatoire.
    pub sweeps: usize,
    /// Réglages du recalage de chaque coupe.
    pub registration: RegistrationConfig,
    /// Réglages des droites d'intersection.
    pub intersection: IntersectionConfig,
    /// Graine de l'ordre des coupes (déterministe).
    pub seed: u64,
    /// Gros vers fin : écart type (en pixels) du flou des coupes pour les `coarse_sweeps` premiers balayages, qui précèdent les `sweeps` balayages sur les images nettes (0 : pas de passe grossière).
    pub coarse_sigma_pixels: f64,
    /// Nombre de balayages avec les images floutées (avec `coarse_sigma_pixels > 0`), à chaque niveau.
    pub coarse_sweeps: usize,
    /// Nombre de niveaux grossiers : le niveau `l` (0 = le plus grossier) floute avec `coarse_sigma_pixels / 2^l` ; 1 : un seul niveau.
    pub coarse_levels: usize,
    /// Alignement par blocs, avant les balayages par coupe : nombre de niveaux hiérarchiques (0 : aucun). Le niveau `l` partage une transformation rigide entre les coupes de chacun des `2^l` blocs consécutifs de chaque stack (niveau 0 : le stack entier,
    /// le premier stack étant l'ancre).
    pub block_levels: usize,
    /// Nombre de balayages (tous les blocs de tous les stacks) à chaque niveau.
    pub block_sweeps: usize,
    /// Recherche exhaustive avant l'optimisation d'un bloc : demi-étendue (mm) de la grille de translations, puis de rotations exprimées en mm équivalents (0 : pas de recherche) ; pas `block_search_step_mm`.
    pub block_search_mm: f64,
    /// Pas (mm) de la grille de la recherche exhaustive.
    pub block_search_step_mm: f64,
}

impl Default for S2sConfig {
    fn default() -> Self {
        S2sConfig { sweeps: 3, registration: RegistrationConfig { learning_rate: 0.3, iterations: 120, lr_decay: 0.97, initial: [0.0; 6] }, intersection: IntersectionConfig::default(), seed: 1, coarse_sigma_pixels: 0.0, coarse_sweeps: 2, coarse_levels: 1, block_levels: 0, block_sweeps: 2, block_search_mm: 12.0, block_search_step_mm: 4.0 }
    }
}

/// Permutation déterministe de `0..n` (Fisher–Yates, générateur xorshift64) : l'ordre des coupes d'un balayage.
fn ordre_aleatoire(n: usize, graine: u64) -> Vec<usize> {
    let mut etat = graine.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut ordre: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        etat ^= etat << 13;
        etat ^= etat >> 7;
        etat ^= etat << 17;
        ordre.swap(i, (etat % (i as u64 + 1)) as usize);
    }
    ordre
}

/// Concordance moyenne d'un bloc : la moyenne, sur les coupes du bloc, de la concordance de chacune (`IntersectionProblem::cost`) pour la MÊME transformation rigide `params` autour du MÊME `pivot`.
fn block_cost(problems: &[IntersectionProblem], params: Tensor<1>, pivot: Tensor<1>, rotation_scale_mm: f64) -> Tensor<1> {
    let n = problems.len() as f64;
    let mut somme = problems[0].cost(params.clone(), pivot.clone(), rotation_scale_mm);
    for p in &problems[1..] {
        somme = somme + p.cost(params.clone(), pivot.clone(), rotation_scale_mm);
    }
    somme / n
}

/// Points d'une grille régulière de `[-demi, demi]` de pas `pas` (le zéro est toujours dedans, au milieu).
pub(crate) fn grille_1d(demi: f64, pas: f64) -> Vec<f64> {
    let n = (demi / pas).floor() as i64;
    (-n..=n).map(|i| i as f64 * pas).collect()
}

/// Recherche exhaustive des 6 paramètres `[rotations (mm équivalents), translations (mm)]` maximisant `evaluer`, sur la grille `grille` (de [`grille_1d`]) : d'abord la meilleure translation (rotation nulle), puis la meilleure rotation autour de CETTE translation ;
/// `zero` est la valeur de `evaluer` au paramètre nul, que la recherche ne rend qu'à égalité ou en dessous d'un candidat strictement meilleur. Rend le meilleur point (le paramètre nul si aucun candidat ne fait mieux).
pub(crate) fn recherche_exhaustive(evaluer: &dyn Fn([f64; 6]) -> f64, grille: &[f64], zero: f64) -> [f64; 6] {
    let (mut meilleur, mut valeur_max) = ([0.0; 6], zero);
    for &x in grille {
        for &y in grille {
            for &z in grille {
                let p = [0.0, 0.0, 0.0, x, y, z];
                let v = evaluer(p);
                if v > valeur_max {
                    (meilleur, valeur_max) = (p, v);
                }
            }
        }
    }
    let translation = [meilleur[3], meilleur[4], meilleur[5]];
    for &a in grille {
        for &b in grille {
            for &c in grille {
                let p = [a, b, c, translation[0], translation[1], translation[2]];
                let v = evaluer(p);
                if v > valeur_max {
                    (meilleur, valeur_max) = (p, v);
                }
            }
        }
    }
    meilleur
}

/// **Recalage rigide commun d'un bloc de coupes** : un seul jeu de 6 paramètres (autour de `pivot`, rotations en mm équivalents) pour toutes les coupes de `problems`. Avec `search_mm > 0`, une recherche exhaustive précède
/// l'optimisation : d'abord la meilleure translation de la grille `[-search_mm, search_mm]³` (pas `step_mm`, rotation nulle), puis la meilleure rotation de la grille (mêmes bornes, en mm équivalents) autour de cette translation ; Adam part de ce point.
/// Rend les paramètres et l'historique de la concordance (première valeur : au zéro ; dernière : finale).
pub(crate) fn register_block_s2s(problems: &[IntersectionProblem], pivot: Tensor<1>, rotation_scale_mm: f64, config: RegistrationConfig, search_mm: f64, step_mm: f64) -> RegistrationResult {
    let device = pivot.device();
    let valeur = |t: Tensor<1>| f64::from(t.into_data().try_to_vec::<f32>().unwrap()[0]);
    let evaluer = |p: [f64; 6]| valeur(block_cost(problems, Tensor::<1>::from_floats(p.map(|v| v as f32), &device), pivot.clone(), rotation_scale_mm));
    let zero = evaluer([0.0; 6]);
    let mut depart = config.initial;
    if search_mm > 0.0 && step_mm > 0.0 {
        depart = recherche_exhaustive(&evaluer, &grille_1d(search_mm, step_mm), zero);
    }
    let module_depart: Vec<f32> = depart.iter().map(|&v| v as f32).collect();
    let mut module = PoseModule { params: Param::from_tensor(Tensor::<1>::from_floats(module_depart.as_slice(), &device)) };
    let mut optimiseur = AdamConfig::new().init();
    let mut pas = config.learning_rate;
    for _ in 0..config.iterations {
        let concordance = block_cost(problems, module.params.val(), pivot.clone(), rotation_scale_mm);
        let gradients = GradientsParams::from_grads(concordance.neg().sum().backward(), &module);
        module = optimiseur.step(pas, module, gradients);
        pas *= config.lr_decay;
    }
    let finale = module.params.val();
    let apres = valeur(block_cost(problems, finale.clone(), pivot.clone(), rotation_scale_mm));
    let params: Vec<f32> = finale.into_data().try_to_vec::<f32>().unwrap();
    RegistrationResult { params: std::array::from_fn(|i| f64::from(params[i])), ncc_history: vec![zero, apres] }
}

/// Partition des `n` coupes d'un stack en `2^niveau` blocs consécutifs de tailles presque égales (indices de coupes) ; niveau 0 : un seul bloc.
pub(crate) fn block_partition(n: usize, niveau: usize) -> Vec<std::ops::Range<usize>> {
    let blocs = (1usize << niveau).min(n.max(1));
    (0..blocs).map(|b| (b * n / blocs)..((b + 1) * n / blocs)).collect()
}

/// **Alignement par blocs** : pour chaque niveau hiérarchique (stack entier, puis 2, 4… blocs par stack), [`align_blocks_level`].
fn align_blocks(stacks: &[Stack], poses: &mut SlicePoses, atlas: &SliceAtlas, config: &S2sConfig, device: &Device) {
    for niveau in 0..config.block_levels {
        align_blocks_level(stacks, poses, atlas, config, niveau, device);
    }
}

/// Un niveau de l'alignement par blocs : `config.block_sweeps` balayages ; chaque bloc de coupes consécutives (les coupes recalables : assez de pixels de masque et au moins une paire ; au moins 3 pour que le bloc soit recalé) reçoit une
/// transformation rigide COMMUNE ([`register_block_s2s`]), acceptée seulement si la concordance du bloc ne baisse pas, et composée à la pose de toutes ses coupes. Au niveau 0, le premier stack ne bouge pas (le repère est libre). Les stacks sont traités dans l'ordre
/// (Gauss-Seidel : un bloc voit les poses déjà corrigées des autres stacks).
pub(crate) fn align_blocks_level(stacks: &[Stack], poses: &mut SlicePoses, atlas: &SliceAtlas, config: &S2sConfig, niveau: usize, device: &Device) {
    for _ in 0..config.block_sweeps {
        for si in 0..stacks.len() {
            if niveau == 0 && si == 0 {
                continue;
            }
            for bloc in block_partition(stacks[si].dim().2, niveau) {
                let mut problemes = Vec::new();
                let (mut pivots, mut echelles, mut ks) = (Vec::new(), Vec::new(), Vec::new());
                for k in bloc {
                    let courante = *poses.get(si, k);
                    let Some(preparee) = preparer_coupe(&stacks[si], k, courante) else { continue };
                    let Some(probleme) = IntersectionProblem::build(atlas, stacks, poses, si, k, &config.intersection, device) else { continue };
                    problemes.push(probleme);
                    pivots.push(preparee.pivot);
                    echelles.push(preparee.echelle);
                    ks.push(k);
                }
                if problemes.len() < 3 {
                    continue;
                }
                let pivot = pivots.iter().fold(Vector3::zeros(), |a, p| a + p) / pivots.len() as f64;
                let echelle = (echelles.iter().map(|e| e * e).sum::<f64>() / echelles.len() as f64).sqrt();
                let t_pivot = Tensor::<1>::from_floats([pivot.x as f32, pivot.y as f32, pivot.z as f32], device);
                let resultat = register_block_s2s(&problemes, t_pivot, echelle, config.registration, config.block_search_mm, config.block_search_step_mm);
                if resultat.ncc_history[1] >= resultat.ncc_history[0] {
                    let delta = pose_to_matrix(&resultat.params, &pivot, echelle);
                    for k in ks {
                        let nouvelle = delta * poses.get(si, k);
                        poses.set(si, k, nouvelle);
                    }
                }
            }
        }
    }
}

/// Calendrier des balayages de [`register_slices_s2s`] : une liste de `(écart type du flou en pixels, nombre de balayages, net)`. D'abord les niveaux grossiers (`coarse_levels` niveaux, le flou `coarse_sigma_pixels` divisé par 2 à chaque niveau,
/// `coarse_sweeps` balayages chacun ; `coarse_levels` = 0 compte pour 1), seulement si `coarse_sigma_pixels > 0` et `coarse_sweeps > 0` ; puis `sweeps` balayages sur les images nettes (`net = true`, les seuls à alimenter le rapport).
pub(crate) fn sweep_schedule(config: &S2sConfig) -> Vec<(f64, usize, bool)> {
    let mut niveaux = Vec::new();
    if config.coarse_sigma_pixels > 0.0 && config.coarse_sweeps > 0 {
        for l in 0..config.coarse_levels.max(1) {
            niveaux.push((config.coarse_sigma_pixels / 2f64.powi(l as i32), config.coarse_sweeps, false));
        }
    }
    niveaux.push((0.0, config.sweeps, true));
    niveaux
}

/// **Estimation du mouvement coupe-coupe, sans reconstruction** : `config.sweeps` balayages sur toutes les coupes de tous les stacks, dans un ordre aléatoire (graine `config.seed`, ordre différent à chaque
/// balayage). Pour chaque coupe : le problème d'intersection est construit aux poses COURANTES (les partenaires déjà recalés comptent), la coupe est recalée par [`register_slice_s2s`] depuis la pose nulle
/// (6 paramètres autour de son pivot P3), et sa pose devient `D · M` aussitôt (Gauss–Seidel : les coupes suivantes voient la pose mise à jour). La mise à jour n'est **acceptée que si la concordance ne baisse
/// pas**. Une coupe sans partenaire, ou de moins de [`crate::align::MIN_MASK_PIXELS`] pixels de masque, garde sa pose.
///
/// Dans le rapport (un par coupe, dernier balayage), `ncc_first` et `ncc_final` sont des CONCORDANCES (avant et après le recalage de ce balayage), `runs` le nombre de balayages où la coupe a été recalée,
/// `correction_rms_mm` le déplacement quadratique moyen, sur ses pixels de masque, entre la pose reçue et la pose finale. Rien n'ancre le repère : la solution n'est définie qu'à un mouvement rigide commun
/// près (qui ne change pas la qualité).
///
/// `device` doit avoir l'autodiff activé.
///
/// # Erreurs
/// [`SvrError::NoMask`] si un stack n'a pas de masque ; [`SvrError::PoseMismatch`] si `poses` n'a pas la forme de `stacks`.
pub fn register_slices_s2s(stacks: &[Stack], poses: &SlicePoses, config: &S2sConfig, device: &Device) -> Result<AlignmentReport, SvrError> {
    verifier_entrees(stacks, poses)?;
    let coupes: Vec<(usize, usize)> = stacks.iter().enumerate().flat_map(|(si, s)| (0..s.dim().2).map(move |k| (si, k))).collect();
    let mut courantes = poses.clone();
    let mut rapports: Vec<SliceReport> =
        coupes.iter().map(|&(stack, slice)| SliceReport { stack, slice, registered: false, ncc_first: 0.0, ncc_final: 0.0, runs: 0, correction_rms_mm: 0.0 }).collect();
    if config.block_levels > 0 {
        let atlas = SliceAtlas::new(stacks, device)?;
        align_blocks(stacks, &mut courantes, &atlas, config, device);
    }
    let niveaux = sweep_schedule(config);
    let mut decalage_graine = 0u64;
    for (sigma, balayages, net) in niveaux {
        let atlas = SliceAtlas::new_blurred(stacks, device, sigma)?;
        for balayage in 0..balayages {
            for &n in &ordre_aleatoire(coupes.len(), config.seed + decalage_graine + balayage as u64) {
                let (si, k) = coupes[n];
                let courante = *courantes.get(si, k);
                let Some(preparee) = preparer_coupe(&stacks[si], k, courante) else { continue };
                let Some(probleme) = IntersectionProblem::build(&atlas, stacks, &courantes, si, k, &config.intersection, device) else { continue };
                let pivot = Tensor::<1>::from_floats([preparee.pivot.x as f32, preparee.pivot.y as f32, preparee.pivot.z as f32], device);
                let resultat = register_slice_s2s(&probleme, pivot, preparee.echelle, config.registration);
                let (avant, apres) = (resultat.ncc_history[0], *resultat.ncc_history.last().unwrap());
                if net {
                    let rapport = &mut rapports[n];
                    rapport.registered = true;
                    rapport.runs += 1;
                    rapport.ncc_first = avant;
                    rapport.ncc_final = avant;
                    if apres >= avant {
                        rapport.ncc_final = apres;
                    }
                }
                if apres >= avant {
                    courantes.set(si, k, pose_to_matrix(&resultat.params, &preparee.pivot, preparee.echelle) * courante);
                }
            }
        }
        decalage_graine += balayages as u64;
    }
    for (n, &(si, k)) in coupes.iter().enumerate() {
        if let Some(depart) = preparer_coupe(&stacks[si], k, *poses.get(si, k)) {
            let relatif = courantes.get(si, k) * poses.get(si, k).try_inverse().expect("une pose est inversible");
            rapports[n].correction_rms_mm = (depart.points.iter().map(|x| ((relatif * x.push(1.0)).xyz() - x).norm_squared()).sum::<f64>() / depart.points.len() as f64).sqrt();
        }
    }
    Ok(AlignmentReport { poses: courantes, slices: rapports })
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
