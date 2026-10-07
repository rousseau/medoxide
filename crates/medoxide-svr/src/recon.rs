//! Reconstruction sur grille (étape 4a) : la grille du volume à estimer et son initialisation par **adjoint normalisé**.
//!
//! Le volume reconstruit est une grille isotrope **alignée sur les axes du monde** (RAS+), qui ne dépend d'aucun stack. Son
//! étendue est la boîte englobante de tous les voxels de masque cérébral de tous les stacks, plus une marge.

use ndarray::{Array3, ArrayView2, Axis};
use rayon::prelude::*;
use nalgebra::{Matrix4, Vector3};

use crate::{Stack, SvrError, Volume};

/// Mouvement rigide de **chaque coupe de chaque stack** (matrice 4 × 4 dans le repère monde, composée après l'affine d'en-tête : voir
/// [`crate::Slice::with_motion`]). L'identité partout au départ ; c'est ce que le recalage fait évoluer à chaque cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct SlicePoses {
    motions: Vec<Vec<Matrix4<f64>>>,
}

impl SlicePoses {
    /// Des poses identité pour toutes les coupes de `stacks`.
    pub fn identity(stacks: &[Stack]) -> SlicePoses {
        SlicePoses { motions: stacks.iter().map(|s| vec![Matrix4::identity(); s.dim().2]).collect() }
    }

    /// Mouvement de la coupe `k` du stack `s`.
    ///
    /// # Panics
    /// Si `s` ou `k` est hors du jeu de poses.
    pub fn get(&self, s: usize, k: usize) -> &Matrix4<f64> {
        &self.motions[s][k]
    }

    /// Fixe le mouvement de la coupe `k` du stack `s`.
    ///
    /// # Panics
    /// Si `s` ou `k` est hors du jeu de poses.
    pub fn set(&mut self, s: usize, k: usize, motion: Matrix4<f64>) {
        self.motions[s][k] = motion;
    }

    /// `true` si ce jeu de poses a exactement la forme de `stacks` (même nombre de stacks, même nombre de coupes par stack).
    pub fn matches(&self, stacks: &[Stack]) -> bool {
        self.motions.len() == stacks.len() && self.motions.iter().zip(stacks).all(|(m, s)| m.len() == s.dim().2)
    }
}

/// Support minimal (somme des poids rétroprojetés) au-dessous duquel un voxel est considéré comme non couvert par les données.
/// Chaque pixel de masque répartit une masse totale de 1 sur les voxels : le support compte, en pixels, les données qui touchent
/// le voxel. 1e-3 écarte les voxels que seule la queue d'une PSF atteint.
pub const SUPPORT_MIN: f32 = 1e-3;

/// Géométrie d'une grille de reconstruction : affine voxel → monde (axes du monde, voxels isotropes) et dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct GridSpec {
    /// Affine voxel → monde : `diag(résolution)` et translation = position monde du centre du voxel `(0, 0, 0)`.
    pub affine: Matrix4<f64>,
    /// Nombre de voxels par axe.
    pub dims: (usize, usize, usize),
    /// Taille du voxel isotrope, en mm.
    pub resolution_mm: f64,
}

impl GridSpec {
    /// Un [`Volume`] nul de cette géométrie.
    pub fn zeros(&self) -> Volume {
        Volume::new(Array3::<f32>::zeros(self.dims), self.affine).expect("l'affine d'une grille est diagonale, de pas > 0 : inversible")
    }
}

/// **Grille de reconstruction** : boîte englobante, élargie de `margin_mm` de chaque côté, des positions monde de tous les voxels
/// de masque (nettoyés, voir [`crate::BrainMask`]) de tous les `stacks`, en voxels isotropes de `resolution_mm` alignés sur les
/// axes du monde. Le coin minimal de la boîte est le centre du voxel `(0, 0, 0)` ; `n = ⌈(haut − bas) / résolution⌉ + 1` par axe.
///
/// # Erreurs
/// [`SvrError::NoStacks`] ; [`SvrError::InvalidGrid`] (résolution ≤ 0 ou marge < 0, non finies) ;
/// [`SvrError::NoMask`] si un stack n'a pas de masque attaché.
pub fn reconstruction_grid(stacks: &[Stack], resolution_mm: f64, margin_mm: f64) -> Result<GridSpec, SvrError> {
    if stacks.is_empty() {
        return Err(SvrError::NoStacks);
    }
    // `!(x > 0.0)` est vrai aussi pour NaN.
    if !(resolution_mm > 0.0) || !resolution_mm.is_finite() || !(margin_mm >= 0.0) || !margin_mm.is_finite() {
        return Err(SvrError::InvalidGrid);
    }
    let mut bas = Vector3::from_element(f64::INFINITY);
    let mut haut = Vector3::from_element(f64::NEG_INFINITY);
    for stack in stacks {
        let masque = stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?;
        let a = stack.affine();
        for ((i, j, k), &dedans) in masque.voxels().indexed_iter() {
            if dedans {
                let monde = (a * nalgebra::Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
                bas = bas.inf(&monde);
                haut = haut.sup(&monde);
            }
        }
    }
    let marge = Vector3::from_element(margin_mm);
    let (bas, haut) = (bas - marge, haut + marge);
    let n = |a: usize| ((haut[a] - bas[a]) / resolution_mm).ceil() as usize + 1;
    let mut affine = Matrix4::identity();
    for a in 0..3 {
        affine[(a, a)] = resolution_mm;
        affine[(a, 3)] = bas[a];
    }
    Ok(GridSpec { affine, dims: (n(0), n(1), n(2)), resolution_mm })
}

/// Toutes les coupes de tous les stacks, **avec leur mouvement** (`poses`), chacune avec son masque 2D (vue sur le masque du stack). Les stacks
/// doivent avoir un masque et `poses` doit avoir leur forme.
fn coupes_avec_masque<'a>(stacks: &'a [Stack], poses: &'a SlicePoses) -> Vec<(ArrayView2<'a, bool>, crate::Slice<'a>)> {
    stacks
        .iter()
        .enumerate()
        .flat_map(|(s, stack)| {
            let masque = stack.brain_mask().expect("le masque a été vérifié").voxels();
            stack.slices().map(move |coupe| (masque.index_axis(Axis(2), coupe.index()), coupe.with_motion(*poses.get(s, coupe.index()))))
        })
        .collect()
}

/// Résultat de [`normalized_adjoint`].
#[derive(Debug, Clone)]
pub struct NormalizedAdjoint {
    /// Volume initial `x₀` : moyenne pondérée des pixels qui touchent chaque voxel ; 0 là où le support est inférieur à [`SUPPORT_MIN`].
    pub image: Array3<f32>,
    /// Support : somme des poids rétroprojetés (compte, en pixels de masque, des données qui touchent le voxel).
    pub support: Array3<f32>,
}

/// **Adjoint normalisé** : `x₀ = Σ Aₖᵀ (mₖ ⊙ yₖ) / Σ Aₖᵀ mₖ`, où `yₖ` sont les pixels de la coupe `k`, `mₖ` son masque cérébral (0 ou 1)
/// et `Aₖᵀ` la transposée **exacte** de l'opérateur d'acquisition ([`Volume::back_project`]). C'est, en chaque voxel, la moyenne des
/// pixels de masque qui le touchent, pondérée par les coefficients de l'opérateur.
///
/// Toutes les coupes de tous les `stacks` (masque requis) sont utilisées, avec leur [`crate::Slice::psf`]. Les intensités sont prises
/// telles quelles : leur mise à l'échelle entre stacks relève de l'étape 4c.
///
/// # Erreurs
/// [`SvrError::NoMask`] si un stack n'a pas de masque attaché.
pub fn normalized_adjoint(grid: &GridSpec, stacks: &[Stack]) -> Result<NormalizedAdjoint, SvrError> {
    normalized_adjoint_with_poses(grid, stacks, &SlicePoses::identity(stacks))
}

/// Comme [`normalized_adjoint`], avec le mouvement `poses` de chaque coupe (la PSF tourne avec elle).
///
/// # Erreurs
/// [`SvrError::NoMask`] ; [`SvrError::PoseMismatch`] si `poses` n'a pas la forme de `stacks`.
pub fn normalized_adjoint_with_poses(grid: &GridSpec, stacks: &[Stack], poses: &SlicePoses) -> Result<NormalizedAdjoint, SvrError> {
    if !poses.matches(stacks) {
        return Err(SvrError::PoseMismatch);
    }
    let geometrie = grid.zeros();
    for stack in stacks {
        stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?;
    }
    let coupes = coupes_avec_masque(stacks, poses);
    // Les coupes sont indépendantes : chaque tâche accumule dans ses propres volumes (écrire dans un volume partagé serait une course aux
    // données), puis on additionne les volumes.
    let (numerateur, denominateur) = coupes
        .par_iter()
        .fold(
            || (Array3::<f64>::zeros(grid.dims), Array3::<f64>::zeros(grid.dims)),
            |(mut num, mut den), (m, coupe)| {
                // numérateur et dénominateur en une seule passe, sur les pixels du masque : y n'est lu que là
                geometrie.back_project_with_weight(coupe, &coupe.psf(), coupe.data(), *m, &mut num, &mut den);
                (num, den)
            },
        )
        .reduce_with(|(mut n1, mut d1), (n2, d2)| {
            n1 += &n2;
            d1 += &d2;
            (n1, d1)
        })
        .unwrap_or_else(|| (Array3::<f64>::zeros(grid.dims), Array3::<f64>::zeros(grid.dims)));
    let mut image = Array3::<f32>::zeros(grid.dims);
    let mut support = Array3::<f32>::zeros(grid.dims);
    for ((idx, &d), s) in denominateur.indexed_iter().zip(support.iter_mut()) {
        *s = d as f32;
        if *s >= SUPPORT_MIN {
            image[idx] = (numerateur[idx] / d) as f32;
        }
    }
    Ok(NormalizedAdjoint { image, support })
}

/// Évaluation de l'objectif : valeur de chaque terme et gradient.
#[derive(Debug, Clone)]
pub struct Evaluation {
    /// Attache aux données `½ Σ ‖mₖ ⊙ (Aₖ x − yₖ)‖²`.
    pub data: f64,
    /// Régularisation `(α/2) ‖G x‖²`.
    pub regularization: f64,
    /// Gradient de la somme, sur la grille entière (nul hors du domaine).
    pub gradient: Array3<f64>,
}

impl Evaluation {
    /// Valeur de l'objectif : attache aux données plus régularisation.
    pub fn value(&self) -> f64 {
        self.data + self.regularization
    }
}

/// Le problème de reconstruction régularisée : `f(x) = ½ Σₖ ‖mₖ ⊙ (Aₖ x − yₖ)‖² + (α/2) ‖G x‖²`.
///
/// - `Aₖ` est l'opérateur d'acquisition de la coupe `k` ([`Volume::simulate_slice`]), `yₖ` ses pixels, `mₖ` son masque cérébral (0 ou 1).
/// - `G` est le gradient discret (différences avant sur chaque axe, divisées par le pas de la grille) ; `α` n'est **pas** normalisé par
///   un nombre de pixels ou de voxels, pour que les optimiseurs se comparent sur un objectif sans mise à l'échelle cachée.
/// - Les **inconnues** sont les voxels du *domaine* (support de l'adjoint normalisé ≥ [`SUPPORT_MIN`]). Les autres sont fixés à 0 ;
///   la régularisation ne lie que des paires de voxels voisins du domaine.
///
/// Le gradient est analytique : `Σₖ Aₖᵀ mₖ (Aₖ x − yₖ) + α GᵀG x`, avec l'adjoint exact [`Volume::back_project`].
pub struct ReconstructionProblem<'a> {
    grid: GridSpec,
    stacks: &'a [Stack],
    alpha: f64,
    poses: SlicePoses,
    domain: Array3<bool>,
    initial: Array3<f32>,
    support: Array3<f32>,
}

impl<'a> ReconstructionProblem<'a> {
    /// Construit le problème : calcule l'adjoint normalisé, dont le support définit le domaine et l'image le point de départ.
    ///
    /// # Erreurs
    /// [`SvrError::InvalidRegularization`] si `alpha` est négatif ou non fini ; [`SvrError::NoMask`] si un stack n'a pas de masque.
    pub fn new(grid: &GridSpec, stacks: &'a [Stack], alpha: f64) -> Result<ReconstructionProblem<'a>, SvrError> {
        ReconstructionProblem::with_poses(grid, stacks, &SlicePoses::identity(stacks), alpha)
    }

    /// Comme [`ReconstructionProblem::new`], avec le mouvement `poses` de chaque coupe. Le domaine et le point de départ en dépendent ; la grille,
    /// elle, est celle qu'on lui donne (celle des poses d'en-tête, marge comprise : un mouvement plus grand que la marge ferait sortir de la grille
    /// des voxels de masque).
    ///
    /// # Erreurs
    /// Celles de [`ReconstructionProblem::new`], plus [`SvrError::PoseMismatch`] si `poses` n'a pas la forme de `stacks`.
    pub fn with_poses(grid: &GridSpec, stacks: &'a [Stack], poses: &SlicePoses, alpha: f64) -> Result<ReconstructionProblem<'a>, SvrError> {
        // `!(a >= 0.0)` est vrai aussi pour NaN.
        if !(alpha >= 0.0) || !alpha.is_finite() {
            return Err(SvrError::InvalidRegularization);
        }
        let init = normalized_adjoint_with_poses(grid, stacks, poses)?;
        let domain = init.support.mapv(|s| s >= SUPPORT_MIN);
        Ok(ReconstructionProblem { grid: grid.clone(), stacks, alpha, poses: poses.clone(), domain, initial: init.image, support: init.support })
    }

    /// Voxels inconnus du problème.
    pub fn domain(&self) -> &Array3<bool> {
        &self.domain
    }

    /// Support de l'adjoint normalisé : somme des poids rétroprojetés en chaque voxel (voir [`NormalizedAdjoint::support`]).
    pub fn support(&self) -> &Array3<f32> {
        &self.support
    }

    /// Poids `α` de la régularisation.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Pas `h` de la grille, en mm.
    pub fn resolution_mm(&self) -> f64 {
        self.grid.resolution_mm
    }

    /// Point de départ : l'adjoint normalisé (nul hors du domaine).
    pub fn initial_guess(&self) -> Array3<f64> {
        self.initial.mapv(f64::from)
    }

    /// Valeur de l'objectif et gradient en `x` (grille entière ; les voxels hors du domaine sont ignorés).
    pub fn evaluate(&self, x: &Array3<f64>) -> Evaluation {
        self.evaluate_with(x, true)
    }

    /// Comme [`ReconstructionProblem::evaluate`] ; avec `use_data = false`, les pixels `yₖ` sont remplacés par 0, ce qui donne le
    /// gradient de la partie quadratique seule : `H x` (voir [`ReconstructionProblem::normal_operator`]).
    fn evaluate_with(&self, x: &Array3<f64>, use_data: bool) -> Evaluation {
        assert_eq!(x.dim(), self.grid.dims, "x doit avoir la forme de la grille");
        // x restreint au domaine, converti en f32 pour l'opérateur (qui accumule en f64)
        let mut restreint = Array3::<f32>::zeros(self.grid.dims);
        for (r, (&v, &d)) in restreint.iter_mut().zip(x.iter().zip(self.domain.iter())) {
            if d {
                *r = v as f32;
            }
        }
        let volume = Volume::new(restreint, self.grid.affine).expect("l'affine d'une grille est inversible");
        // Une passe normale par coupe, en parallèle ; chaque tâche accumule le gradient dans son propre volume, puis on additionne.
        let dims = self.grid.dims;
        let (mut gradient, data) = coupes_avec_masque(self.stacks, &self.poses)
            .par_iter()
            .fold(
                || (Array3::<f64>::zeros(dims), 0.0),
                |(mut g, d), (m, coupe)| {
                    let pixels = use_data.then(|| coupe.data());
                    let terme = volume.normal_pass_masked(coupe, &coupe.psf(), pixels, *m, &mut g);
                    (g, d + terme)
                },
            )
            .reduce_with(|(mut g1, d1), (g2, d2)| {
                g1 += &g2;
                (g1, d1 + d2)
            })
            .unwrap_or_else(|| (Array3::<f64>::zeros(dims), 0.0));
        let regularization = regularization(x, &self.domain, self.grid.resolution_mm, self.alpha, &mut gradient);
        for (g, &d) in gradient.iter_mut().zip(self.domain.iter()) {
            if !d {
                *g = 0.0;
            }
        }
        Evaluation { data, regularization, gradient }
    }
}

/// Résultat de [`ReconstructionProblem::conjugate_gradient`].
#[derive(Debug, Clone)]
pub struct CgResult {
    /// Solution (grille entière, nulle hors du domaine).
    pub x: Array3<f64>,
    /// Nombre d'itérations effectuées.
    pub iterations: usize,
    /// Objectif suivi par récurrence : `iterations + 1` valeurs, de la valeur au départ à la valeur finale.
    pub objective_history: Vec<f64>,
    /// Résidu relatif `‖b − H x‖ / ‖b‖` suivi par récurrence : `iterations + 1` valeurs.
    pub residual_history: Vec<f64>,
    /// `true` si la tolérance a été atteinte avant `max_iterations`.
    pub converged: bool,
}

/// Produit scalaire de deux tableaux de même forme.
fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

impl<'a> ReconstructionProblem<'a> {
    /// **Produit `H p`** de la partie quadratique `H = Σ AₖᵀMₖAₖ + α GᵀG` avec `p` : le gradient de l'objectif privé de ses données (`y = 0`)
    /// évalué en `p`. Symétrique et défini positif sur le domaine.
    pub fn normal_operator(&self, p: &Array3<f64>) -> Array3<f64> {
        self.evaluate_with(p, false).gradient
    }

    /// Second membre `b = Σ AₖᵀMₖyₖ` des équations normales `H x = b` : l'opposé du gradient de l'objectif en 0.
    pub fn right_hand_side(&self) -> Array3<f64> {
        let zero = Array3::<f64>::zeros(self.grid.dims);
        -self.evaluate_with(&zero, true).gradient
    }

    /// **Gradient conjugué** sur les équations normales `H x = b`, à partir de `x0`, jusqu'à ce que `‖b − H x‖ / ‖b‖ ≤ tolerance` ou
    /// `max_iterations` itérations (une passe `H p` chacune : une simulation et une rétroprojection de toutes les coupes).
    ///
    /// Le long d'une itération, l'objectif diminue exactement de `½ · pas · ‖r‖²` : il est suivi sans passe supplémentaire (voir
    /// [`CgResult::objective_history`]). `x0` hors du domaine est ignoré.
    pub fn conjugate_gradient(&self, x0: &Array3<f64>, max_iterations: usize, tolerance: f64) -> CgResult {
        self.conjugate_gradient_observed(x0, max_iterations, tolerance, |_, _| {})
    }

    /// Comme [`ReconstructionProblem::conjugate_gradient`], en appelant `observateur(k, x)` avec l'itéré `x` après chaque itération `k`
    /// (`k = 0` : le départ restreint au domaine). Sert à enregistrer des instantanés pour comparer les optimiseurs.
    pub fn conjugate_gradient_observed(&self, x0: &Array3<f64>, max_iterations: usize, tolerance: f64, mut observateur: impl FnMut(usize, &Array3<f64>)) -> CgResult {
        let b = self.right_hand_side();
        let norme_b = dot(&b, &b).sqrt();
        let mut x = x0.clone();
        for (v, &d) in x.iter_mut().zip(self.domain.iter()) {
            if !d {
                *v = 0.0;
            }
        }
        observateur(0, &x);
        let depart = self.evaluate(&x);
        let mut objectif = depart.value();
        // r = b − H x = −gradient (le gradient de f en x vaut H x − b)
        let mut r = -depart.gradient;
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let mut historique_f = vec![objectif];
        let mut historique_r = vec![rs.sqrt() / norme_b];
        let mut converge = historique_r[0] <= tolerance;
        let mut iterations = 0;
        while !converge && iterations < max_iterations {
            let hp = self.normal_operator(&p);
            let pas = rs / dot(&p, &hp);
            x.scaled_add(pas, &p);
            r.scaled_add(-pas, &hp);
            objectif -= 0.5 * pas * rs;
            let rs_nouveau = dot(&r, &r);
            p = &r + &(&p * (rs_nouveau / rs));
            rs = rs_nouveau;
            iterations += 1;
            observateur(iterations, &x);
            historique_f.push(objectif);
            historique_r.push(rs.sqrt() / norme_b);
            converge = *historique_r.last().unwrap() <= tolerance;
        }
        CgResult { x, iterations, objective_history: historique_f, residual_history: historique_r, converged: converge }
    }
}

/// `(α/2) ‖G x‖²` pour le gradient discret `G` (différences avant divisées par `h`, paires de voxels tous deux dans `domaine`) ;
/// ajoute son gradient `α GᵀG x` à `gradient`.
fn regularization(x: &Array3<f64>, domaine: &Array3<bool>, h: f64, alpha: f64, gradient: &mut Array3<f64>) -> f64 {
    let (nx, ny, nz) = x.dim();
    let mut valeur = 0.0;
    for axe in 0..3 {
        let pas = [usize::from(axe == 0), usize::from(axe == 1), usize::from(axe == 2)];
        for i in 0..nx.saturating_sub(pas[0]) {
            for j in 0..ny.saturating_sub(pas[1]) {
                for k in 0..nz.saturating_sub(pas[2]) {
                    let (a, b) = ([i, j, k], [i + pas[0], j + pas[1], k + pas[2]]);
                    if domaine[a] && domaine[b] {
                        let d = x[b] - x[a];
                        valeur += 0.5 * alpha * d * d / (h * h);
                        gradient[b] += alpha * d / (h * h);
                        gradient[a] -= alpha * d / (h * h);
                    }
                }
            }
        }
    }
    valeur
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector4};
    use nifti::{writer::WriterOptions, NiftiHeader};
    use std::path::{Path, PathBuf};

    fn racine() -> String {
        format!("{}/../..", env!("CARGO_MANIFEST_DIR"))
    }

    /// Dossier temporaire propre à un test (nom + processus), supprimé en fin de test.
    struct Temp(PathBuf);
    impl Temp {
        fn new(nom: &str) -> Temp {
            let d = std::env::temp_dir().join(format!("medoxide_recon_{nom}_{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Temp(d)
        }
        fn fichier(&self, nom: &str) -> PathBuf {
            self.0.join(nom)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn en_tete(affine: &Matrix4<f64>, pixdim: [f32; 3]) -> NiftiHeader {
        let mut h = NiftiHeader::default();
        h.sform_code = 1;
        h.srow_x = std::array::from_fn(|c| affine[(0, c)] as f32);
        h.srow_y = std::array::from_fn(|c| affine[(1, c)] as f32);
        h.srow_z = std::array::from_fn(|c| affine[(2, c)] as f32);
        h.pixdim = [1.0, pixdim[0], pixdim[1], pixdim[2], 1.0, 1.0, 1.0, 1.0];
        h
    }

    /// Écrit un stack (image et masque) d'affine donnée et le relit, masque attaché.
    fn stack_avec_masque(t: &Temp, nom: &str, data: &Array3<f32>, masque: &Array3<u8>, affine: &Matrix4<f64>, pixdim: [f32; 3]) -> Stack {
        let h = en_tete(affine, pixdim);
        let (image, m) = (t.fichier(&format!("{nom}.nii.gz")), t.fichier(&format!("{nom}_mask.nii.gz")));
        WriterOptions::new(&image).reference_header(&h).write_nifti(data).unwrap();
        WriterOptions::new(&m).reference_header(&h).write_nifti(masque).unwrap();
        let mut stack = Stack::read(&image).unwrap();
        stack.set_brain_mask(&m).unwrap();
        stack
    }

    /// Affine `R · diag(échelle)` + origine telle que l'indice `centre` tombe en `cible`.
    fn affine_centree(r: &Rotation3<f64>, echelle: [f64; 3], centre: [f64; 3], cible: [f64; 3]) -> Matrix4<f64> {
        let lineaire = r.matrix() * nalgebra::Matrix3::from_diagonal(&Vector3::from(echelle));
        let origine = Vector3::from(cible) - lineaire * Vector3::from(centre);
        let mut a = Matrix4::identity();
        a.fixed_view_mut::<3, 3>(0, 0).copy_from(&lineaire);
        a.fixed_view_mut::<3, 1>(0, 3).copy_from(&origine);
        a
    }

    /// Critère 1 : grille identique à la référence indépendante (nibabel et scipy) sur les trois stacks TRUFI de 3 sujets réels,
    /// pour 3 couples résolution / marge.
    #[test]
    fn grid_matches_the_python_reference_on_real_stacks() {
        let tsv = std::fs::read_to_string(format!("{}/data/reference/grid/grilles.tsv", racine())).expect("références absentes : lancer scripts/make_reference_grid.py");
        let mut n_lignes = 0;
        for ligne in tsv.lines() {
            let c: Vec<&str> = ligne.split('\t').collect();
            let (resolution, marge): (f64, f64) = (c[0].parse().unwrap(), c[1].parse().unwrap());
            let bas: [f64; 3] = [c[3].parse().unwrap(), c[4].parse().unwrap(), c[5].parse().unwrap()];
            let n: [usize; 3] = [c[6].parse().unwrap(), c[7].parse().unwrap(), c[8].parse().unwrap()];
            let stacks: Vec<Stack> = c[9]
                .split(',')
                .map(|chemin| {
                    let mut s = Stack::read(Path::new(&format!("{}/{chemin}", racine()))).unwrap();
                    let (dossier, nom) = chemin.rsplit_once("/anat/").unwrap();
                    let (sujet_ses, _) = (dossier.strip_prefix("data/svr/jeu_reel_tru_haste/").unwrap(), ());
                    let masque = format!("{}/data/svr/jeu_reel_tru_haste/derivatives/medx-fetalbet/{sujet_ses}/anat/{}", racine(), nom.replace("_T2w.nii.gz", "_desc-brain_mask.nii.gz"));
                    s.set_brain_mask(Path::new(&masque)).unwrap();
                    s
                })
                .collect();
            let g = reconstruction_grid(&stacks, resolution, marge).unwrap();
            assert_eq!((g.dims.0, g.dims.1, g.dims.2), (n[0], n[1], n[2]), "{} résolution {resolution} marge {marge}", c[2]);
            for a in 0..3 {
                assert!((g.affine[(a, 3)] - bas[a]).abs() < 1e-6, "{} axe {a} : {} contre {}", c[2], g.affine[(a, 3)], bas[a]);
                assert_eq!(g.affine[(a, a)], resolution);
            }
            n_lignes += 1;
        }
        assert_eq!(n_lignes, 9);
    }

    /// Géométrie calculée à la main : trois voxels de masque connexes (diagonale) d'un stack d'affine diagonale.
    #[test]
    fn grid_covers_the_mask_with_the_margin() {
        let t = Temp::new("grille");
        let affine = Matrix4::new(1.0, 0.0, 0.0, 10.0, 0.0, 2.0, 0.0, 20.0, 0.0, 0.0, 3.0, 30.0, 0.0, 0.0, 0.0, 1.0);
        let mut masque = Array3::<u8>::zeros((6, 5, 4));
        for k in 0..3 {
            masque[[k, k, k]] = 1; // positions monde (10, 20, 30), (11, 22, 33), (12, 24, 36)
        }
        let stack = stack_avec_masque(&t, "s", &Array3::from_elem((6, 5, 4), 1.0), &masque, &affine, [1.0, 2.0, 3.0]);
        let g = reconstruction_grid(&[stack], 1.0, 2.0).unwrap();
        // bas = (8, 18, 28), haut = (14, 26, 38) -> n = (6+1, 8+1, 10+1)
        assert_eq!((g.dims.0, g.dims.1, g.dims.2), (7, 9, 11));
        assert_eq!([g.affine[(0, 3)], g.affine[(1, 3)], g.affine[(2, 3)]], [8.0, 18.0, 28.0]);
        assert!(g.affine.fixed_view::<3, 3>(0, 0).determinant() > 0.0);
        let v = g.zeros();
        assert_eq!(v.dim(), g.dims);
        // le voxel du coin haut contient (14, 26, 38) à l'arrondi près : le dernier centre est à bas + (n-1)·res >= haut
        assert!(g.affine[(0, 3)] + 6.0 >= 14.0 && g.affine[(2, 3)] + 10.0 >= 38.0);
    }

    #[test]
    fn grid_rejects_invalid_inputs() {
        let t = Temp::new("grille_erreurs");
        let affine = Matrix4::identity();
        let masque = Array3::<u8>::from_elem((3, 3, 3), 1);
        let avec = stack_avec_masque(&t, "a", &Array3::from_elem((3, 3, 3), 1.0), &masque, &affine, [1.0, 1.0, 1.0]);
        assert!(matches!(reconstruction_grid(&[], 1.0, 0.0), Err(SvrError::NoStacks)));
        for (res, marge) in [(0.0, 1.0), (-1.0, 1.0), (f64::NAN, 1.0), (f64::INFINITY, 1.0), (1.0, -0.1), (1.0, f64::NAN)] {
            assert!(matches!(reconstruction_grid(std::slice::from_ref(&avec), res, marge), Err(SvrError::InvalidGrid)), "{res} / {marge}");
        }
        let sans = Stack::read(&t.fichier("a.nii.gz")).unwrap();
        assert!(matches!(reconstruction_grid(&[sans], 1.0, 0.0), Err(SvrError::NoMask(_))));
    }

    /// Critère 2 : si toutes les coupes viennent d'un volume constant, l'adjoint normalisé redonne la constante partout où le support
    /// est positif, même avec un masque partiel (il entre dans le numérateur et le dénominateur).
    #[test]
    fn normalized_adjoint_of_constant_slices_is_constant() {
        let t = Temp::new("constante");
        let mut alea = 0x9E37_79B9_7F4A_7C15_u64;
        let mut suivant = move || {
            alea ^= alea << 13;
            alea ^= alea >> 7;
            alea ^= alea << 17;
            (alea >> 11) as f64 / (1u64 << 53) as f64
        };
        let axial = affine_centree(&Rotation3::from_euler_angles(0.1, -0.15, 0.2), [1.0, 1.0, 2.0], [11.5, 11.5, 1.5], [0.0, 0.0, 0.0]);
        let coronal = affine_centree(&Rotation3::from_euler_angles(std::f64::consts::FRAC_PI_2 + 0.1, 0.05, -0.1), [1.0, 1.0, 2.0], [11.5, 11.5, 1.5], [1.0, 0.5, -0.5]);
        let stacks: Vec<Stack> = [("ax", axial), ("cor", coronal)]
            .iter()
            .map(|(nom, a)| {
                let masque = Array3::from_shape_fn((24, 24, 4), |_| u8::from(suivant() < 0.6)); // masque partiel : 60 % des pixels
                stack_avec_masque(&t, nom, &Array3::from_elem((24, 24, 4), 5.0), &masque, a, [1.0, 1.0, 2.0])
            })
            .collect();
        // BrainMask::read garde la plus grande composante connexe : le masque effectif est un sous-ensemble du tirage
        let grille = reconstruction_grid(&stacks, 1.0, 3.0).unwrap();
        let r = normalized_adjoint(&grille, &stacks).unwrap();
        let couverts = r.support.iter().filter(|&&s| s >= SUPPORT_MIN).count();
        let pire = r.image.iter().zip(r.support.iter()).filter(|(_, &s)| s >= SUPPORT_MIN).map(|(&v, _)| (v - 5.0).abs()).fold(0.0_f32, f32::max);
        println!("{couverts} voxels couverts sur {} ; écart max à la constante : {pire:.1e}", r.image.len());
        assert!(couverts > 1000, "trop peu de voxels couverts : {couverts}");
        assert!(pire < 1e-4, "{pire}");
        // hors support : 0
        assert!(r.image.iter().zip(r.support.iter()).filter(|(_, &s)| s < SUPPORT_MIN).all(|(&v, _)| v == 0.0));
    }
    /// Un atlas de Gholipour (`STAxx`, tous centrés en 0) et trois stacks (axial, coronal, sagittal ; pixels 0,8 mm, épaisseur 3,5 mm, légèrement
    /// obliques) simulés par l'opérateur de l'étape 2, sans mouvement ni bruit. Les stacks sont mis en **cache** dans `data/atlas/sim/<atlas>/` (hors de Git) : la
    /// simulation (≈ 1 min en release) n'est refaite que si les fichiers manquent. Le masque d'un pixel est « couvert et tissu (> 150) ».
    fn stacks_atlas_sans_bruit(nom_atlas: &str) -> (Volume, Vec<Stack>) {
        let chemin = format!("{}/data/atlas/gholipour/{nom_atlas}.nii.gz", racine());
        let atlas = Volume::from_stack(&Stack::read(Path::new(&chemin)).expect("atlas absent : voir data/atlas/gholipour"));
        let dossier = PathBuf::from(format!("{}/data/atlas/sim/{nom_atlas}", racine()));
        std::fs::create_dir_all(&dossier).unwrap();
        let demi_tour = std::f64::consts::FRAC_PI_2;
        let specs = [
            ("axial", Rotation3::from_euler_angles(0.05, -0.04, 0.03)),
            ("coronal", Rotation3::from_euler_angles(demi_tour + 0.04, 0.03, -0.05)),
            ("sagittal", Rotation3::from_euler_angles(0.03, demi_tour - 0.04, 0.05)),
        ];
        let dims = (188, 188, 40);
        let stacks = specs
            .iter()
            .map(|(nom, r)| {
                let (image, masque_chemin) = (dossier.join(format!("{nom}.nii.gz")), dossier.join(format!("{nom}_mask.nii.gz")));
                if !image.exists() || !masque_chemin.exists() {
                    let affine = affine_centree(r, [0.8, 0.8, 3.5], [93.5, 93.5, 19.5], [0.0, 0.0, 0.0]);
                    let t = Temp::new(&format!("atlas_{nom}"));
                    let modele = stack_avec_masque(&t, "modele", &Array3::zeros(dims), &Array3::from_elem(dims, 1u8), &affine, [0.8, 0.8, 3.5]);
                    let (mut data, mut masque) = (Array3::<f32>::zeros(dims), Array3::<u8>::zeros(dims));
                    for coupe in modele.slices() {
                        let sim = atlas.simulate_slice(&coupe, &coupe.psf());
                        for ((i, j), &v) in sim.values.indexed_iter() {
                            data[[i, j, coupe.index()]] = v;
                            masque[[i, j, coupe.index()]] = u8::from(sim.coverage[[i, j]] >= 0.99 && v > 150.0);
                        }
                    }
                    let h = en_tete(&affine, [0.8, 0.8, 3.5]);
                    WriterOptions::new(&image).reference_header(&h).write_nifti(&data).unwrap();
                    WriterOptions::new(&masque_chemin).reference_header(&h).write_nifti(&masque).unwrap();
                }
                let mut stack = Stack::read(&image).unwrap();
                stack.set_brain_mask(&masque_chemin).unwrap();
                stack
            })
            .collect();
        (atlas, stacks)
    }

    /// Comme [`stacks_atlas_sans_bruit`], avec un bruit gaussien additif sur tous les pixels des stacks, d'écart type `bruit_pct` % de
    /// l'intensité moyenne des pixels de masque (0 : aucun bruit). Les stacks bruités sont mis en cache dans `data/atlas/sim/<atlas>_b<pct>/`
    /// (graine fixe par stack : reproductible), le masque est celui du stack sans bruit.
    fn stacks_atlas_simules(nom_atlas: &str, bruit_pct: u32) -> (Volume, Vec<Stack>) {
        let (atlas, propres) = stacks_atlas_sans_bruit(nom_atlas);
        if bruit_pct == 0 {
            return (atlas, propres);
        }
        let dossier = PathBuf::from(format!("{}/data/atlas/sim/{nom_atlas}_b{bruit_pct}", racine()));
        std::fs::create_dir_all(&dossier).unwrap();
        let stacks = propres
            .iter()
            .enumerate()
            .map(|(n, propre)| {
                let image = dossier.join(format!("stack{n}.nii.gz"));
                let masque_chemin = dossier.join(format!("stack{n}_mask.nii.gz"));
                if !image.exists() || !masque_chemin.exists() {
                    let masque = propre.brain_mask().unwrap().voxels();
                    let (somme, n_m) = propre.data().iter().zip(masque.iter()).filter(|(_, &m)| m).fold((0.0_f64, 0usize), |(s, n), (&v, _)| (s + f64::from(v), n + 1));
                    let sigma = f64::from(bruit_pct) / 100.0 * somme / n_m as f64;
                    let mut alea = Alea(0xC0FFEE ^ (n as u64 + 1) * 7919);
                    let bruite = propre.data().mapv(|v| {
                        let gauss: f64 = (0..12).map(|_| alea.suivant()).sum::<f64>() - 6.0; // somme de 12 uniformes - 6 : gaussienne centrée réduite approchée
                        (f64::from(v) + sigma * gauss) as f32
                    });
                    let h = en_tete(propre.affine(), [propre.spacing()[0] as f32, propre.spacing()[1] as f32, propre.spacing()[2] as f32]);
                    WriterOptions::new(&image).reference_header(&h).write_nifti(&bruite).unwrap();
                    WriterOptions::new(&masque_chemin).reference_header(&h).write_nifti(&masque.mapv(u8::from)).unwrap();
                }
                let mut stack = Stack::read(&image).unwrap();
                stack.set_brain_mask(&masque_chemin).unwrap();
                stack
            })
            .collect();
        (atlas, stacks)
    }

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let (va, vb): (f64, f64) = (a.iter().map(|x| (x - ma).powi(2)).sum(), b.iter().map(|y| (y - mb).powi(2)).sum());
        cov / (va * vb).sqrt()
    }

    /// PSNR (dB) de `a` par rapport à la référence `b`, pour une dynamique `plage`.
    fn psnr(a: &[f64], b: &[f64], plage: f64) -> f64 {
        let eqm = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
        20.0 * (plage / eqm.sqrt()).log10()
    }

    /// Critère 3 (descriptif, ligne de base pour la suite) : trois stacks (axial, coronal, sagittal ; pixels 0,8 mm, épaisseur 3,5 mm,
    /// légèrement obliques) simulés par l'opérateur de l'étape 2 depuis l'atlas de Gholipour, sans mouvement ni bruit ; qualité de
    /// l'adjoint normalisé par rapport à l'atlas, comparée à chacun des stacks pris seul (trilinéaire). À lancer en
    /// `--release --ignored --nocapture` (≈ 1 min).
    #[test]
    #[ignore = "données locales ; lent en debug"]
    fn normalized_adjoint_on_stacks_simulated_from_the_atlas() {
        let (atlas, stacks) = stacks_atlas_simules("STA21", 0);
        let specs = ["axial", "coronal", "sagittal"];
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let debut = std::time::Instant::now();
        let r = normalized_adjoint(&grille, &stacks).unwrap();
        println!("grille {:?} ({} voxels) ; adjoint normalisé en {:.1} s", grille.dims, r.image.len(), debut.elapsed().as_secs_f64());
        let singles: Vec<Volume> = stacks.iter().map(Volume::from_stack).collect();
        // voxels de comparaison : support suffisant, tissu de l'atlas, et lisibles dans les trois stacks pour les lignes de base
        let (mut x0, mut verite) = (Vec::new(), Vec::new());
        let mut bases: Vec<Vec<f64>> = vec![Vec::new(); 3];
        for ((i, j, k), &s) in r.support.indexed_iter() {
            if s < 0.5 {
                continue;
            }
            let monde = (grille.affine * Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
            let a = match atlas.sample(&monde) {
                Some(a) if a > 150.0 => f64::from(a),
                _ => continue,
            };
            let valeurs: Vec<Option<f32>> = singles.iter().map(|v| v.sample(&monde)).collect();
            if valeurs.iter().any(Option::is_none) {
                continue;
            }
            x0.push(f64::from(r.image[[i, j, k]]));
            verite.push(a);
            for (b, v) in bases.iter_mut().zip(valeurs) {
                b.push(f64::from(v.unwrap()));
            }
        }
        let plage = {
            let mut v = verite.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[(v.len() as f64 * 0.99) as usize]
        };
        println!("{} voxels de comparaison ; dynamique (p99 de l'atlas) {plage:.0}", verite.len());
        println!("adjoint normalisé : NCC {:.4}, PSNR {:.2} dB", correlation(&x0, &verite), psnr(&x0, &verite, plage));
        for (nom, b) in specs.iter().zip(&bases) {
            println!("stack {nom:9} seul (trilinéaire) : NCC {:.4}, PSNR {:.2} dB", correlation(b, &verite), psnr(b, &verite, plage));
        }
        assert!(verite.len() > 100_000, "trop peu de voxels de comparaison");
    }
    // ------------------------------------------------------------------ sous-étape 2 : objectif (valeur, gradient)

    use nalgebra::{DMatrix, DVector};

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

    /// Deux petits stacks (axial et coronal, 6 × 6 × 2 pixels de 3 mm, épaisseur 4 mm, obliques), intensités et masques (≈ 70 %)
    /// aléatoires, et leur grille de 5 mm avec une marge de 12 mm, assez large pour que des voxels soient **hors du domaine** (sans quoi les vérifications
    /// « hors du domaine » seraient vides) : un problème assez petit pour construire sa matrice dense.
    fn petit_probleme(t: &Temp) -> (GridSpec, Vec<Stack>) {
        let mut alea = Alea(0xDEAD_BEEF_1234_5678);
        let specs = [
            ("ax", Rotation3::from_euler_angles(0.1, -0.08, 0.15), [0.0, 0.0, 0.0]),
            ("cor", Rotation3::from_euler_angles(std::f64::consts::FRAC_PI_2 + 0.1, 0.05, -0.08), [1.0, 0.5, -0.5]),
        ];
        let stacks: Vec<Stack> = specs
            .iter()
            .map(|(nom, r, centre)| {
                let affine = affine_centree(r, [3.0, 3.0, 4.0], [2.5, 2.5, 0.5], *centre);
                let data = Array3::from_shape_fn((6, 6, 2), |_| (alea.suivant() * 10.0) as f32);
                let masque = Array3::from_shape_fn((6, 6, 2), |_| u8::from(alea.suivant() < 0.7));
                stack_avec_masque(t, nom, &data, &masque, &affine, [3.0, 3.0, 4.0])
            })
            .collect();
        let grille = reconstruction_grid(&stacks, 5.0, 12.0).unwrap();
        (grille, stacks)
    }


    /// Le problème sous forme dense : voxels du domaine, matrice `A` (lignes : pixels de toutes les coupes de tous les stacks ; colonnes :
    /// voxels du domaine, obtenues en appliquant l'opérateur à chaque voxel unité), pixels `y`, masque `m`, paires de voisins du domaine.
    struct Dense {
        voxels: Vec<[usize; 3]>,
        a: DMatrix<f64>,
        y: Vec<f64>,
        m: Vec<f64>,
        paires: Vec<(usize, usize)>,
    }

    fn systeme_dense(probleme: &ReconstructionProblem, grille: &GridSpec, stacks: &[Stack]) -> Dense {
        let voxels: Vec<[usize; 3]> = probleme.domain().indexed_iter().filter(|(_, &d)| d).map(|((i, j, k), _)| [i, j, k]).collect();
        let n = voxels.len();
        let (mut y, mut m) = (Vec::new(), Vec::new());
        for stack in stacks {
            let masque = stack.brain_mask().unwrap().voxels();
            for coupe in stack.slices() {
                for i in 0..coupe.dim().0 {
                    for j in 0..coupe.dim().1 {
                        y.push(f64::from(coupe.data()[[i, j]]));
                        m.push(f64::from(u8::from(masque[[i, j, coupe.index()]])));
                    }
                }
            }
        }
        let mut a = DMatrix::<f64>::zeros(y.len(), n);
        for (col, v) in voxels.iter().enumerate() {
            let mut unite = Array3::<f32>::zeros(grille.dims);
            unite[*v] = 1.0;
            let volume = Volume::new(unite, grille.affine).unwrap();
            let mut r = 0;
            for (si, stack) in stacks.iter().enumerate() {
                for coupe in stack.slices() {
                    let coupe = coupe.with_motion(*probleme.poses.get(si, coupe.index()));
                    let sim = volume.simulate_slice(&coupe, &coupe.psf());
                    for i in 0..coupe.dim().0 {
                        for j in 0..coupe.dim().1 {
                            a[(r, col)] = f64::from(sim.values[[i, j]]);
                            r += 1;
                        }
                    }
                }
            }
        }
        let colonne: std::collections::HashMap<[usize; 3], usize> = voxels.iter().enumerate().map(|(c, v)| (*v, c)).collect();
        let mut paires = Vec::new();
        for (c, v) in voxels.iter().enumerate() {
            for axe in 0..3 {
                let mut w = *v;
                w[axe] += 1;
                if let Some(&c2) = colonne.get(&w) {
                    paires.push((c, c2));
                }
            }
        }
        Dense { voxels, a, y, m, paires }
    }

    /// Critère 1 : valeur et gradient égalent l'algèbre dense. La matrice `A` est construite en appliquant l'opérateur à chaque voxel
    /// unité du domaine ; `½‖M(Ax − y)‖² + (α/2)‖Gx‖²` et `AᵀM(Ax − y) + α GᵀGx` sont ensuite calculés en matrices denses.
    #[test]
    fn objective_matches_dense_linear_algebra() {
        verifier_objectif_dense(false);
    }

    /// Même équivalence avec des poses non triviales : rotations jusqu'à ±3° et translations jusqu'à ±2 mm, différentes pour chaque coupe.
    #[test]
    fn objective_with_poses_matches_dense_linear_algebra() {
        verifier_objectif_dense(true);
    }

    /// Mouvement rigide `M = T(c + t) · R(ω) · T(−c)` : rotation du vecteur `omega` (rad) autour du point `c`, puis translation `t`.
    fn mouvement_rigide(omega: &Vector3<f64>, t: &Vector3<f64>, c: &Vector3<f64>) -> Matrix4<f64> {
        let r = Rotation3::from_scaled_axis(*omega);
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&(c + t - r * c));
        m
    }

    /// Poses aléatoires par coupe (autour du centre de la coupe) : rotation et translation tirées uniformément dans ± `degres` et ± `mm`.
    fn poses_aleatoires(stacks: &[Stack], degres: f64, mm: f64, graine: u64) -> SlicePoses {
        let mut alea = Alea(graine);
        let mut poses = SlicePoses::identity(stacks);
        let mut tire = |a: f64| (alea.suivant() * 2.0 - 1.0) * a;
        for (si, stack) in stacks.iter().enumerate() {
            for coupe in stack.slices() {
                let omega = Vector3::new(tire(degres), tire(degres), tire(degres)).map(f64::to_radians);
                let t = Vector3::new(tire(mm), tire(mm), tire(mm));
                poses.set(si, coupe.index(), mouvement_rigide(&omega, &t, &coupe.geometric_center()));
            }
        }
        poses
    }

    fn verifier_objectif_dense(avec_poses: bool) {
        let t = Temp::new(if avec_poses { "dense_poses" } else { "dense" });
        let (grille, stacks) = petit_probleme(&t);
        let alpha = 0.7;
        let poses = if avec_poses { poses_aleatoires(&stacks, 3.0, 2.0, 99) } else { SlicePoses::identity(&stacks) };
        let probleme = ReconstructionProblem::with_poses(&grille, &stacks, &poses, alpha).unwrap();
        let dense = systeme_dense(&probleme, &grille, &stacks);
        let (voxels, a, y, m, paires) = (dense.voxels, dense.a, dense.y, dense.m, dense.paires);
        let n = voxels.len();
        let hors = probleme.domain().iter().filter(|&&d| !d).count();
        println!("grille {:?}, {n} voxels dans le domaine, {hors} hors du domaine", grille.dims);
        assert!(n > 40 && n < 600, "taille du problème dense : {n}");
        assert!(hors > 100, "il faut des voxels hors du domaine pour éprouver leur exclusion : {hors}");
        let h = grille.resolution_mm;
        let mut alea = Alea(77);
        let x_d = DVector::from_fn(n, |_, _| alea.suivant() * 10.0);
        // x complet : valeurs du domaine, 7,0 ailleurs (doit être ignoré)
        let mut x_plein = Array3::<f64>::from_elem(grille.dims, 7.0);
        for (c, v) in voxels.iter().enumerate() {
            x_plein[*v] = x_d[c];
        }
        let masque_ligne = DVector::from_vec(m.clone());
        let r = (&a * &x_d - DVector::from_vec(y)).component_mul(&masque_ligne);
        let data = 0.5 * r.norm_squared();
        let reg: f64 = paires.iter().map(|&(c1, c2)| 0.5 * alpha * (x_d[c2] - x_d[c1]).powi(2) / (h * h)).sum();
        let mut g = a.transpose() * &r;
        for &(c1, c2) in &paires {
            let d = alpha * (x_d[c2] - x_d[c1]) / (h * h);
            g[c2] += d;
            g[c1] -= d;
        }
        let e = probleme.evaluate(&x_plein);
        let relatif = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        println!("attache {:.6} (dense {data:.6}) ; régularisation {:.6} (dense {reg:.6})", e.data, e.regularization);
        assert!(relatif(e.data, data) < 1e-5, "{} contre {data}", e.data);
        assert!(relatif(e.regularization, reg) < 1e-5, "{} contre {reg}", e.regularization);
        let norme = g.amax();
        let pire = voxels.iter().enumerate().map(|(c, v)| (e.gradient[*v] - g[c]).abs()).fold(0.0, f64::max) / norme;
        println!("gradient : écart max relatif {pire:.2e} (|g|max = {norme:.2})");
        assert!(pire < 1e-5, "{pire:.2e}");
        // hors du domaine : gradient nul ; changer x hors du domaine ne change rien
        assert!(e.gradient.indexed_iter().filter(|(idx, _)| !probleme.domain()[*idx]).all(|(_, &v)| v == 0.0));
        let mut autre = x_plein.clone();
        for (idx, v) in autre.indexed_iter_mut() {
            if !probleme.domain()[idx] {
                *v = -123.0;
            }
        }
        let e2 = probleme.evaluate(&autre);
        assert_eq!(e2.value(), e.value());
        assert_eq!(e2.gradient, e.gradient);
        assert!(e.data > 0.0 && e.regularization > 0.0);
    }

    /// Critère 2 : une rampe linéaire `x = c · i` le long d'un axe sur un domaine complet.
    #[test]
    fn regularization_of_a_linear_ramp() {
        let (nx, ny, nz) = (5, 3, 2);
        let domaine = Array3::from_elem((nx, ny, nz), true);
        let (c, h, alpha) = (1.5, 2.0, 0.4);
        let x = Array3::from_shape_fn((nx, ny, nz), |(i, _, _)| c * i as f64);
        let mut g = Array3::<f64>::zeros((nx, ny, nz));
        let valeur = regularization(&x, &domaine, h, alpha, &mut g);
        // G = (différence de valeurs entre voisins) / h : la rampe a une pente c/h par mm. Paires selon x : (nx-1)·ny·nz, chacune de valeur
        // ½ α (c/h)² ; les autres axes : 0.
        let attendue = 0.5 * alpha * (c / h).powi(2) * ((nx - 1) * ny * nz) as f64;
        assert!((valeur - attendue).abs() < 1e-12, "{valeur} contre {attendue}");
        // gradient α GᵀG x : bord bas −α c/h², bord haut +α c/h², intérieur 0 (Laplacien d'une fonction linéaire)
        for ((i, _, _), &v) in g.indexed_iter() {
            let theorique = if i == 0 {
                -alpha * c / (h * h)
            } else if i == nx - 1 {
                alpha * c / (h * h)
            } else {
                0.0
            };
            assert!((v - theorique).abs() < 1e-12, "i = {i} : {v} contre {theorique}");
        }
        // domaine troué : une paire dont un voxel est hors du domaine ne compte pas
        let mut troue = domaine.clone();
        troue[[2, 1, 0]] = false;
        let mut g2 = Array3::<f64>::zeros((nx, ny, nz));
        let v2 = regularization(&x, &troue, h, alpha, &mut g2);
        // le voxel (2,1,0) a 2 paires selon x (de valeur ½ α (c/h)² chacune) ; ses paires selon y et z ont une différence nulle
        let retirees_x = 2.0 * 0.5 * alpha * (c / h).powi(2);
        assert!((v2 - (attendue - retirees_x)).abs() < 1e-12, "{v2}");
    }

    #[test]
    fn problem_rejects_invalid_alpha() {
        let t = Temp::new("alpha");
        let (grille, stacks) = petit_probleme(&t);
        for alpha in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(ReconstructionProblem::new(&grille, &stacks, alpha), Err(SvrError::InvalidRegularization)), "{alpha}");
        }
        assert!(ReconstructionProblem::new(&grille, &stacks, 0.0).is_ok());
    }
    // ------------------------------------------------------------------ sous-étape 3 : gradient conjugué

    /// Critères 1 à 4 : le gradient conjugué égale la solution directe des équations normales denses `(AᵀMA + α GᵀG) x = AᵀM y` ;
    /// l'objectif décroît ; la valeur suivie égale `evaluate` ; le résidu final est petit ; `H` est symétrique et égale sa version dense.
    #[test]
    fn conjugate_gradient_matches_the_dense_solution() {
        let t = Temp::new("cg");
        let (grille, stacks) = petit_probleme(&t);
        let alpha = 0.7;
        let probleme = ReconstructionProblem::new(&grille, &stacks, alpha).unwrap();
        let dense = systeme_dense(&probleme, &grille, &stacks);
        let (voxels, n) = (&dense.voxels, dense.voxels.len());
        let h = grille.resolution_mm;
        // H_d = Aᵀ diag(m) A + α GᵀG et b_d = Aᵀ diag(m) y
        let masque = DVector::from_vec(dense.m.clone());
        let a_masquee = DMatrix::from_fn(dense.a.nrows(), n, |r, c| dense.a[(r, c)] * masque[r]);
        let mut h_d = dense.a.transpose() * &a_masquee;
        for &(c1, c2) in &dense.paires {
            let w = alpha / (h * h);
            h_d[(c1, c1)] += w;
            h_d[(c2, c2)] += w;
            h_d[(c1, c2)] -= w;
            h_d[(c2, c1)] -= w;
        }
        let b_d = dense.a.transpose() * DVector::from_vec(dense.y.iter().zip(&dense.m).map(|(y, m)| y * m).collect());
        let x_dense = h_d.clone().lu().solve(&b_d).expect("H est inversible");
        let en_plein = |v: &DVector<f64>| {
            let mut x = Array3::<f64>::zeros(grille.dims);
            for (c, idx) in voxels.iter().enumerate() {
                x[*idx] = v[c];
            }
            x
        };
        let vers_dense = |x: &Array3<f64>| DVector::from_fn(n, |c, _| x[voxels[c]]);

        // H p dense contre normal_operator, et symétrie pᵀHq = qᵀHp
        let mut alea = Alea(5);
        let (p, q) = (DVector::from_fn(n, |_, _| alea.suivant() - 0.5), DVector::from_fn(n, |_, _| alea.suivant() - 0.5));
        let hp = vers_dense(&probleme.normal_operator(&en_plein(&p)));
        let hq = vers_dense(&probleme.normal_operator(&en_plein(&q)));
        let ecart_h = (&hp - &h_d * &p).norm() / (&h_d * &p).norm();
        let (pq, qp) = (q.dot(&hp), p.dot(&hq));
        println!("H p : écart au produit dense {ecart_h:.2e} ; symétrie : qᵀHp = {pq:.6}, pᵀHq = {qp:.6}");
        assert!(ecart_h < 1e-5, "{ecart_h:.2e}");
        assert!((pq - qp).abs() / pq.abs() < 1e-6, "{pq} contre {qp}");
        // second membre
        let ecart_b = (vers_dense(&probleme.right_hand_side()) - &b_d).norm() / b_d.norm();
        assert!(ecart_b < 1e-5, "{ecart_b:.2e}");

        // gradient conjugué depuis l'adjoint normalisé
        let x0 = probleme.initial_guess();
        let r = probleme.conjugate_gradient(&x0, 2 * n, 1e-9);
        let ecart = (vers_dense(&r.x) - &x_dense).norm() / x_dense.norm();
        let residu_reel = {
            let e = probleme.evaluate(&r.x);
            dot(&e.gradient, &e.gradient).sqrt() / b_d.norm()
        };
        let f_reel = probleme.evaluate(&r.x).value();
        println!(
            "gradient conjugué : {} itérations (convergé : {}), résidu suivi {:.1e}, résidu réel {residu_reel:.1e}, écart à la solution dense {ecart:.2e} ; objectif {:.4} → {:.4} (recalculé {f_reel:.4})",
            r.iterations,
            r.converged,
            r.residual_history.last().unwrap(),
            r.objective_history[0],
            r.objective_history.last().unwrap()
        );
        assert!(r.converged && r.iterations > 5);
        assert!(ecart < 1e-4, "écart à la solution dense {ecart:.2e}");
        assert!(residu_reel <= 1e-3, "résidu réel {residu_reel:.2e}");
        assert!(r.objective_history.windows(2).all(|w| w[1] <= w[0] + 1e-9 * w[0].abs()), "l'objectif doit décroître");
        assert!((r.objective_history.last().unwrap() - f_reel).abs() / f_reel.abs() < 1e-6, "{} contre {f_reel}", r.objective_history.last().unwrap());
        // nul hors du domaine
        assert!(r.x.indexed_iter().filter(|(idx, _)| !probleme.domain()[*idx]).all(|(_, &v)| v == 0.0));
    }

    /// Le suivi de l'objectif par récurrence égale l'objectif recalculé à chaque itération (0 à 4), et décroît strictement.
    #[test]
    fn conjugate_gradient_objective_tracking_matches_evaluation() {
        let t = Temp::new("cg_suivi");
        let (grille, stacks) = petit_probleme(&t);
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.7).unwrap();
        let x0 = probleme.initial_guess();
        let suivi = probleme.conjugate_gradient(&x0, 4, 0.0).objective_history;
        assert_eq!(suivi.len(), 5);
        for k in 0..=4 {
            let r = probleme.conjugate_gradient(&x0, k, 0.0);
            let reel = probleme.evaluate(&r.x).value();
            assert!((suivi[k] - reel).abs() / reel.abs() < 1e-6, "itération {k} : suivi {} contre recalculé {reel}", suivi[k]);
            assert_eq!(r.iterations, k);
        }
        assert!(suivi.windows(2).all(|w| w[1] < w[0]), "{suivi:?}");
    }

    /// Zéro itération : le départ restreint au domaine, avec un historique d'une valeur.
    #[test]
    fn conjugate_gradient_with_zero_iterations_returns_the_restricted_start() {
        let t = Temp::new("cg_zero");
        let (grille, stacks) = petit_probleme(&t);
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.7).unwrap();
        let depart = Array3::<f64>::from_elem(grille.dims, 3.0);
        let r = probleme.conjugate_gradient(&depart, 0, 1e-12);
        assert_eq!((r.iterations, r.objective_history.len(), r.residual_history.len()), (0, 1, 1));
        for (idx, &v) in r.x.indexed_iter() {
            assert_eq!(v, if probleme.domain()[idx] { 3.0 } else { 0.0 });
        }
    }
    /// Banc d'essai de la reconstruction sur l'atlas (3 stacks, 120 coupes) : temps de l'adjoint normalisé, d'une évaluation (valeur et
    /// gradient) et d'un produit `H p`. À lancer en `--release --ignored --nocapture`.
    #[test]
    #[ignore = "mesure de temps sur données locales"]
    fn bench_reconstruction_on_the_atlas() {
        let (_atlas, stacks) = stacks_atlas_simules("STA21", 0);
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let pixels: usize = stacks.iter().map(|s| s.brain_mask().unwrap().voxels().iter().filter(|&&v| v).count()).sum();
        println!("{} coupes, {pixels} pixels de masque sur {} ; grille {:?}", stacks.iter().map(|s| s.dim().2).sum::<usize>(), stacks.iter().map(|s| s.data().len()).sum::<usize>(), grille.dims);
        let t = std::time::Instant::now();
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.1).unwrap();
        println!("construction (adjoint normalisé compris) : {:.2} s", t.elapsed().as_secs_f64());
        let x0 = probleme.initial_guess();
        let t = std::time::Instant::now();
        let e = probleme.evaluate(&x0);
        println!("evaluate : {:.2} s (valeur {:.6e})", t.elapsed().as_secs_f64(), e.value());
        let t = std::time::Instant::now();
        let hp = probleme.normal_operator(&x0);
        println!("H p : {:.2} s (|Hp|² {:.6e})", t.elapsed().as_secs_f64(), dot(&hp, &hp));
    }
    // ------------------------------------------------------------------ non-régression de la performance

    /// Référence **séquentielle et non optimisée** de [`ReconstructionProblem::evaluate_with`] : tous les pixels de chaque coupe sont
    /// simulés, puis rétroprojetés en deux passes (`simulate_slice` suivi de `back_project`), sans `rayon` ni masque dans l'opérateur.
    fn evaluation_de_reference(probleme: &ReconstructionProblem, x: &Array3<f64>, use_data: bool) -> (f64, f64, Array3<f64>) {
        let mut restreint = Array3::<f32>::zeros(probleme.grid.dims);
        for (r, (&v, &d)) in restreint.iter_mut().zip(x.iter().zip(probleme.domain.iter())) {
            if d {
                *r = v as f32;
            }
        }
        let volume = Volume::new(restreint, probleme.grid.affine).unwrap();
        let mut gradient = Array3::<f64>::zeros(probleme.grid.dims);
        let mut data = 0.0;
        for stack in probleme.stacks {
            let masque = stack.brain_mask().unwrap();
            for coupe in stack.slices() {
                let m = masque.voxels().index_axis(Axis(2), coupe.index());
                let sim = volume.simulate_slice(&coupe, &coupe.psf());
                let mut residu = ndarray::Array2::<f32>::zeros(coupe.dim());
                for ((i, j), r) in residu.indexed_iter_mut() {
                    if m[[i, j]] {
                        *r = sim.values[[i, j]] - if use_data { coupe.data()[[i, j]] } else { 0.0 };
                        data += 0.5 * f64::from(*r) * f64::from(*r);
                    }
                }
                volume.back_project(&coupe, &coupe.psf(), residu.view(), &mut gradient);
            }
        }
        let reg = regularization(x, &probleme.domain, probleme.grid.resolution_mm, probleme.alpha, &mut gradient);
        for (g, &d) in gradient.iter_mut().zip(probleme.domain.iter()) {
            if !d {
                *g = 0.0;
            }
        }
        (data, reg, gradient)
    }

    /// Critère 1 de la performance : la version optimisée (pixels du masque, passe fusionnée, `rayon`) égale la référence séquentielle non
    /// optimisée, pour l'objectif complet et pour `H p` (sans données), à 1e-9 relatif près (seul l'ordre des sommes en `f64` change).
    #[test]
    fn optimized_evaluation_matches_the_sequential_reference() {
        let t = Temp::new("perf_eval");
        let (grille, stacks) = petit_probleme(&t);
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.7).unwrap();
        let mut alea = Alea(31);
        let x = Array3::from_shape_fn(grille.dims, |_| alea.suivant() * 10.0);
        for use_data in [true, false] {
            let (data_ref, reg_ref, grad_ref) = evaluation_de_reference(&probleme, &x, use_data);
            let e = probleme.evaluate_with(&x, use_data);
            let norme = grad_ref.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let ecart_grad = e.gradient.iter().zip(grad_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / norme;
            println!("use_data {use_data} : valeur {:.9} (réf {data_ref:.9}) ; gradient : écart max relatif {ecart_grad:.1e}", e.data);
            assert!((e.data - data_ref).abs() / data_ref.abs() < 1e-9, "{} contre {data_ref}", e.data);
            assert!((e.regularization - reg_ref).abs() / reg_ref.abs() < 1e-12);
            assert!(ecart_grad < 1e-9, "{ecart_grad:.2e}");
        }
    }

    /// L'adjoint normalisé fusionné et parallèle égale la version à deux passes non masquées, séquentielle.
    #[test]
    fn optimized_normalized_adjoint_matches_the_two_pass_reference() {
        let t = Temp::new("perf_adjoint");
        let (grille, stacks) = petit_probleme(&t);
        let geometrie = grille.zeros();
        let (mut num, mut den) = (Array3::<f64>::zeros(grille.dims), Array3::<f64>::zeros(grille.dims));
        for stack in &stacks {
            let masque = stack.brain_mask().unwrap();
            for coupe in stack.slices() {
                let m = masque.voxels().index_axis(Axis(2), coupe.index()).mapv(|v| f32::from(u8::from(v)));
                let y = &coupe.data() * &m;
                let psf = coupe.psf();
                geometrie.back_project(&coupe, &psf, y.view(), &mut num);
                geometrie.back_project(&coupe, &psf, m.view(), &mut den);
            }
        }
        let r = normalized_adjoint(&grille, &stacks).unwrap();
        let mut pire = 0.0_f64;
        for (idx, &s) in r.support.indexed_iter() {
            assert!((f64::from(s) - den[idx]).abs() <= 1e-6 * den[idx].abs().max(1e-12), "support {s} contre {}", den[idx]);
            if den[idx] as f32 >= SUPPORT_MIN {
                pire = pire.max((f64::from(r.image[idx]) - num[idx] / den[idx]).abs() / (num[idx] / den[idx]).abs().max(1e-9));
            } else {
                assert_eq!(r.image[idx], 0.0);
            }
        }
        println!("adjoint normalisé : écart relatif max {pire:.1e}");
        assert!(pire < 1e-6, "{pire:.2e}");
    }
    // ------------------------------------------------------------------ évaluation de la reconstruction sur l'atlas

    /// Jeu de comparaison : les voxels de la grille où le support est suffisant, qui sont du tissu dans l'atlas (> 150) et lisibles dans les
    /// trois stacks pris seuls (lignes de base trilinéaires).
    struct Comparaison {
        voxels: Vec<[usize; 3]>,
        verite: Vec<f64>,
        bases: Vec<Vec<f64>>,
        plage: f64,
        atlas_grille: Array3<f32>,
        dans: Array3<bool>,
    }

    fn construire_comparaison(grille: &GridSpec, support: &Array3<f32>, atlas: &Volume, stacks: &[Stack]) -> Comparaison {
        let singles: Vec<Volume> = stacks.iter().map(Volume::from_stack).collect();
        let mut c = Comparaison { voxels: Vec::new(), verite: Vec::new(), bases: vec![Vec::new(); singles.len()], plage: 0.0, atlas_grille: Array3::zeros(grille.dims), dans: Array3::from_elem(grille.dims, false) };
        for ((i, j, k), &s) in support.indexed_iter() {
            let monde = (grille.affine * Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
            if let Some(a) = atlas.sample(&monde) {
                c.atlas_grille[[i, j, k]] = a;
            }
            // Le support d'un voxel est la masse de pixels qu'il reçoit : il varie comme son volume. Seuil de 0,5 à 0,8 mm, mis à l'échelle du volume
            // du voxel pour les autres résolutions (sinon, à 0,5 mm, aucun voxel ne le passerait).
            if s < 0.5 * (grille.resolution_mm / 0.8).powi(3) as f32 {
                continue;
            }
            let a = match atlas.sample(&monde) {
                Some(a) if a > 150.0 => f64::from(a),
                _ => continue,
            };
            let valeurs: Vec<Option<f32>> = singles.iter().map(|v| v.sample(&monde)).collect();
            if valeurs.iter().any(Option::is_none) {
                continue;
            }
            c.voxels.push([i, j, k]);
            c.verite.push(a);
            c.dans[[i, j, k]] = true;
            for (b, v) in c.bases.iter_mut().zip(valeurs) {
                b.push(f64::from(v.unwrap()));
            }
        }
        let mut v = c.verite.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        c.plage = v[(v.len() as f64 * 0.99) as usize];
        c
    }

    /// Norme du gradient (différences centrées, en intensité par voxel) de `a` en `(i, j, k)`.
    fn norme_gradient(a: &Array3<f32>, i: usize, j: usize, k: usize) -> f64 {
        let d = |p: [usize; 3], q: [usize; 3]| f64::from(a[p] - a[q]) / 2.0;
        (d([i + 1, j, k], [i - 1, j, k]).powi(2) + d([i, j + 1, k], [i, j - 1, k]).powi(2) + d([i, j, k + 1], [i, j, k - 1]).powi(2)).sqrt()
    }

    /// (NCC, PSNR en dB, netteté) d'une image sur la grille par rapport à l'atlas, sur le jeu de comparaison. Netteté = gradient moyen de
    /// l'image / gradient moyen de l'atlas, sur les voxels de comparaison dont les 6 voisins en font aussi partie (1 : aussi nette que l'atlas).
    fn qualite(c: &Comparaison, image: &Array3<f64>) -> (f64, f64, f64) {
        let valeurs: Vec<f64> = c.voxels.iter().map(|v| image[*v]).collect();
        let image32 = image.mapv(|v| v as f32);
        let (mut g_image, mut g_atlas, mut n) = (0.0, 0.0, 0usize);
        for &[i, j, k] in &c.voxels {
            let (nx, ny, nz) = c.dans.dim();
            if i == 0 || j == 0 || k == 0 || i + 1 >= nx || j + 1 >= ny || k + 1 >= nz {
                continue;
            }
            let voisins = [[i + 1, j, k], [i - 1, j, k], [i, j + 1, k], [i, j - 1, k], [i, j, k + 1], [i, j, k - 1]];
            if voisins.iter().all(|v| c.dans[*v]) {
                g_image += norme_gradient(&image32, i, j, k);
                g_atlas += norme_gradient(&c.atlas_grille, i, j, k);
                n += 1;
            }
        }
        assert!(n > 1000);
        (correlation(&valeurs, &c.verite), psnr(&valeurs, &c.verite, c.plage), g_image / g_atlas)
    }

    /// Reconstruit par gradient conjugué (depuis l'adjoint normalisé) ; rend l'image, le nombre d'itérations, le résidu relatif final et la durée.
    fn reconstruire(grille: &GridSpec, stacks: &[Stack], alpha: f64) -> (Array3<f64>, usize, f64, f64) {
        let debut = std::time::Instant::now();
        let probleme = ReconstructionProblem::new(grille, stacks, alpha).unwrap();
        let r = probleme.conjugate_gradient(&probleme.initial_guess(), 300, 1e-4);
        (r.x, r.iterations, *r.residual_history.last().unwrap(), debut.elapsed().as_secs_f64())
    }

    /// Poids de régularisation essayés (log-espacés). La première grille (0,01 à 10, données sans bruit) avait son optimum au bord : étendue vers le bas.
    const ALPHAS_REGLAGE: [f64; 9] = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0];

    fn ligne_qualite(nom: &str, c: &Comparaison, image: &Array3<f64>, iterations: usize, residu: f64, duree: f64) -> (f64, f64, f64) {
        let q = qualite(c, image);
        println!("{nom:12} : NCC {:.4}, PSNR {:6.2} dB, netteté {:.3} ({iterations} itérations, résidu {residu:.1e}, {duree:.1} s)", q.0, q.1, q.2);
        q
    }

    /// RÉGLAGE de α (étape 4a) : sur l'atlas STA21 (le plus petit cerveau), avec un bruit de 5 %. Règle fixée avant : on retient le α qui
    /// maximise le PSNR, à condition qu'il ne soit pas au bord de la grille. Sans bruit, le meilleur α est le plus petit essayé (la
    /// régularisation ne peut qu'abîmer) : régler α sur des données sans bruit n'aurait pas de sens.
    /// À lancer en `--release --ignored --nocapture`.
    #[test]
    #[ignore = "données locales ; longue"]
    fn tune_alpha_on_sta21() {
        let (atlas, stacks) = stacks_atlas_simules("STA21", 5);
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let init = normalized_adjoint(&grille, &stacks).unwrap();
        let c = construire_comparaison(&grille, &init.support, &atlas, &stacks);
        println!("STA21 : {} voxels de comparaison, dynamique {:.0}", c.voxels.len(), c.plage);
        ligne_qualite("init", &c, &init.image.mapv(f64::from), 0, 0.0, 0.0);
        for (nom, b) in ["axial", "coronal", "sagittal"].iter().zip(&c.bases) {
            println!("stack {nom:8} : NCC {:.4}, PSNR {:6.2} dB", correlation(b, &c.verite), psnr(b, &c.verite, c.plage));
        }
        let mut meilleur = (f64::MIN, 0.0);
        for alpha in ALPHAS_REGLAGE {
            let (x, it, res, t) = reconstruire(&grille, &stacks, alpha);
            let q = ligne_qualite(&format!("α = {alpha}"), &c, &x, it, res, t);
            if q.1 > meilleur.0 {
                meilleur = (q.1, alpha);
            }
        }
        println!("α retenu (PSNR maximal) : {} ({:.2} dB){}", meilleur.1, meilleur.0, if meilleur.1 == ALPHAS_REGLAGE[0] || meilleur.1 == ALPHAS_REGLAGE[ALPHAS_REGLAGE.len() - 1] { " : AU BORD DE LA GRILLE, optimum non encadré" } else { "" });
    }
    /// α retenu par `tune_alpha_on_sta21` (bruit 5 %, grille de 0,0003 à 3) : PSNR maximal à 0,01 (28,93 dB ; 28,76 dB à 0,03), optimum encadré.
    const ALPHA_CHOISI: f64 = 0.01;

    /// ÉVALUATION (étape 4a) sur l'atlas STA31, jamais vu pendant le réglage de α : trois stacks simulés avec 5 % de bruit, reconstruction par
    /// gradient conjugué avec `ALPHA_CHOISI`. Critères (fixés à l'étude 06) : NCC ≥ 0,95 ; au moins 2 dB de PSNR de plus que l'adjoint normalisé ;
    /// meilleur que le meilleur stack seul en NCC et en PSNR ; résolution en moins de 5 minutes. Sensibilité à α : descriptive.
    /// À lancer en `--release --ignored --nocapture`.
    #[test]
    #[ignore = "données locales ; longue"]
    fn evaluate_reconstruction_on_sta31() {
        let (atlas, stacks) = stacks_atlas_simules("STA31", 5);
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let init = normalized_adjoint(&grille, &stacks).unwrap();
        let c = construire_comparaison(&grille, &init.support, &atlas, &stacks);
        println!("STA31 : grille {:?}, {} voxels de comparaison, dynamique {:.0}", grille.dims, c.voxels.len(), c.plage);
        let q_init = ligne_qualite("init", &c, &init.image.mapv(f64::from), 0, 0.0, 0.0);
        let mut meilleur_stack = (f64::MIN, f64::MIN);
        for (nom, b) in ["axial", "coronal", "sagittal"].iter().zip(&c.bases) {
            let (n, p) = (correlation(b, &c.verite), psnr(b, &c.verite, c.plage));
            println!("stack {nom:8} : NCC {n:.4}, PSNR {p:6.2} dB");
            meilleur_stack = (meilleur_stack.0.max(n), meilleur_stack.1.max(p));
        }
        for alpha in ALPHAS_REGLAGE {
            let (x, it, res, t) = reconstruire(&grille, &stacks, alpha);
            ligne_qualite(&format!("α = {alpha}"), &c, &x, it, res, t);
        }
        let (x, it, res, t) = reconstruire(&grille, &stacks, ALPHA_CHOISI);
        println!("--- évaluation avec α = {ALPHA_CHOISI}");
        let q = ligne_qualite("CG", &c, &x, it, res, t);
        println!("critères : NCC {:.4} ≥ 0,95 ; gain sur l'init {:.2} dB ≥ 2 ; meilleur stack seul NCC {:.4} / PSNR {:.2} dB ; durée {t:.0} s", q.0, q.1 - q_init.1, meilleur_stack.0, meilleur_stack.1);
        assert!(q.0 >= 0.95, "NCC {}", q.0);
        assert!(q.1 - q_init.1 >= 2.0, "gain {}", q.1 - q_init.1);
        assert!(q.0 > meilleur_stack.0 && q.1 > meilleur_stack.1, "doit battre le meilleur stack seul");
        assert!(t < 300.0, "durée {t}");
    }
    /// Copie d'un stack dont l'affine est composée d'un mouvement rigide `T` : rotation de `angle_deg` autour de `axe` (par l'origine, qui est
    /// le centre de l'atlas), puis translation `decalage` (mm). Les voxels et le masque sont inchangés : la géométrie annoncée est fausse.
    fn stack_perturbe(t: &Temp, nom: &str, stack: &Stack, angle_deg: f64, axe: Vector3<f64>, decalage: Vector3<f64>) -> Stack {
        let r = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axe), angle_deg.to_radians());
        let mut mouvement = Matrix4::identity();
        mouvement.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
        mouvement.fixed_view_mut::<3, 1>(0, 3).copy_from(&decalage);
        let masque = stack.brain_mask().unwrap().voxels().mapv(u8::from);
        let sp = stack.spacing();
        stack_avec_masque(t, nom, stack.data(), &masque, &(mouvement * stack.affine()), [sp[0] as f32, sp[1] as f32, sp[2] as f32])
    }

    /// Sensibilité (descriptive, rien n'est forcé) sur STA31 avec `ALPHA_CHOISI` : résolution de la grille (1,0, 0,8 et 0,5 mm), puis stacks
    /// à pose perturbée (rotation et translation d'un ou de tous les stacks). Pour la perturbation, la reconstruction utilise la géométrie
    /// (fausse) des stacks perturbés, et la qualité est mesurée sur le jeu de comparaison de la géométrie exacte.
    /// À lancer en `--release --ignored --nocapture`.
    #[test]
    #[ignore = "données locales ; longue"]
    fn reconstruction_sensitivity_on_sta31() {
        let (atlas, stacks) = stacks_atlas_simules("STA31", 5);
        for resolution in [1.0, 0.8, 0.5] {
            let grille = reconstruction_grid(&stacks, resolution, 10.0).unwrap();
            let init = normalized_adjoint(&grille, &stacks).unwrap();
            let c = construire_comparaison(&grille, &init.support, &atlas, &stacks);
            let (x, it, res, t) = reconstruire(&grille, &stacks, ALPHA_CHOISI);
            println!("résolution {resolution} mm : grille {:?} ({:.1} M voxels), {} voxels de comparaison", grille.dims, (grille.dims.0 * grille.dims.1 * grille.dims.2) as f64 / 1e6, c.voxels.len());
            ligne_qualite(&format!("  CG {resolution} mm"), &c, &x, it, res, t);
            let meilleur = c.bases.iter().map(|b| psnr(b, &c.verite, c.plage)).fold(f64::MIN, f64::max);
            println!("  meilleur stack seul : PSNR {meilleur:.2} dB ; init : PSNR {:.2} dB", qualite(&c, &init.image.mapv(f64::from)).1);
        }
        // poses perturbées, à 0,8 mm
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let init = normalized_adjoint(&grille, &stacks).unwrap();
        let c = construire_comparaison(&grille, &init.support, &atlas, &stacks);
        let t = Temp::new("perturbations");
        let axe = Vector3::new(1.0, 1.0, 0.5);
        let dec = Vector3::new(1.0, -1.0, 0.5).normalize();
        let cas: [(&str, f64, f64, Vec<usize>); 5] = [
            ("aucune perturbation", 0.0, 0.0, vec![]),
            ("coronal ±1°/1 mm", 1.0, 1.0, vec![1]),
            ("coronal 3°/3 mm", 3.0, 3.0, vec![1]),
            ("tous 1°/1 mm", 1.0, 1.0, vec![0, 1, 2]),
            ("tous 3°/3 mm", 3.0, 3.0, vec![0, 1, 2]),
        ];
        for (nom, angle, mm, cibles) in cas {
            let perturbes: Vec<Stack> = stacks
                .iter()
                .enumerate()
                .map(|(n, s)| {
                    if cibles.contains(&n) {
                        // un sens différent par stack, pour que les erreurs ne se compensent pas
                        let signe = if n % 2 == 0 { 1.0 } else { -1.0 };
                        stack_perturbe(&t, &format!("{nom}_{n}").replace([' ', '/', '°', '±'], "_"), s, signe * angle, axe, signe * mm * dec)
                    } else {
                        stack_perturbe(&t, &format!("{nom}_{n}_id").replace([' ', '/', '°', '±'], "_"), s, 0.0, axe, Vector3::zeros())
                    }
                })
                .collect();
            let (x, it, res, d) = reconstruire(&grille, &perturbes, ALPHA_CHOISI);
            ligne_qualite(nom, &c, &x, it, res, d);
        }
    }
    // ------------------------------------------------------------------ optimiseurs (comparaison, gradient analytique)

    use crate::optimizers;

    /// Chaque optimiseur diminue l'objectif sur le petit problème, et son suivi de l'objectif égale l'objectif recalculé sur les instantanés de
    /// l'itéré (le gradient conjugué et la plus forte pente suivent `f` par récurrence, les autres le mesurent à chaque passe).
    #[test]
    fn every_optimizer_decreases_the_objective_and_its_trace_is_consistent() {
        let t = Temp::new("optimiseurs");
        let (grille, stacks) = petit_probleme(&t);
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.7).unwrap();
        let x0 = probleme.initial_guess();
        let f0 = probleme.evaluate(&x0).value();
        let budget = 12;
        let points = [4, 8, 12];
        let device = burn::tensor::Device::flex().autodiff();
        let traces: Vec<(&str, optimizers::Trace)> = vec![
            ("gradient conjugué", optimizers::conjugate_gradient(&probleme, &x0, budget, &points)),
            ("plus forte pente", optimizers::steepest_descent(&probleme, &x0, budget, &points)),
            ("Barzilai-Borwein", optimizers::barzilai_borwein(&probleme, &x0, budget, &points)),
            ("Jacobi", optimizers::jacobi(&probleme, &x0, 1.0, budget, &points)),
            ("Adam (Burn)", optimizers::adam(&probleme, &x0, 0.3, budget, &points, &device)),
            ("L-BFGS (Burn)", optimizers::lbfgs(&probleme, &x0, 1.0, 10, false, 1.0, budget, &points, &device)),
        ];
        for (nom, tr) in &traces {
            let f_fin = *tr.objective.last().unwrap();
            println!("{nom:20} : {} passes, objectif {f0:.3} → {f_fin:.3}", tr.passes);
            assert_eq!(tr.passes, budget, "{nom}");
            assert!(f_fin < f0, "{nom} : {f_fin} contre {f0}");
            assert_eq!(tr.snapshots.len(), 3, "{nom}");
            for (passes, x) in &tr.snapshots {
                // la valeur de l'itéré disponible après `passes` passes : objective[passes - 2] pour le gradient conjugué (2 passes de départ),
                // objective[passes - 1] pour les autres
                let indice = if *nom == "gradient conjugué" { passes - 2 } else { passes - 1 };
                let reel = probleme.evaluate(x).value();
                let suivi = tr.objective[indice];
                assert!((suivi - reel).abs() / reel.abs() < 1e-5, "{nom}, {passes} passes : suivi {suivi} contre recalculé {reel}");
            }
        }
        // monotones : gradient conjugué et plus forte pente à pas exact
        for nom in ["gradient conjugué", "plus forte pente"] {
            let tr = &traces.iter().find(|(n, _)| *n == nom).unwrap().1;
            assert!(tr.objective.windows(2).all(|w| w[1] <= w[0] + 1e-9 * w[0].abs()), "{nom} doit être monotone");
        }
        // à budget égal, le gradient conjugué n'est pas pire que la plus forte pente (hypothèse de la théorie, vérifiée ici)
        let f = |nom: &str| *traces.iter().find(|(n, _)| *n == nom).unwrap().1.objective.last().unwrap();
        assert!(f("gradient conjugué") <= f("plus forte pente") * (1.0 + 1e-9) + 1e-9);
    }

    /// Les conversions tableau ↔ tenseur de Burn sont fidèles à l'arrondi `f32` près et respectent l'ordre des axes.
    #[test]
    fn array_tensor_conversion_round_trips() {
        let a = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| (100 * i + 10 * j + k) as f64 + 0.25);
        let device = burn::tensor::Device::flex().autodiff();
        // un optimiseur sans itération rend le départ converti puis reconverti : on l'observe par `adam` avec 1 passe
        let t = Temp::new("conversion");
        let (grille, stacks) = petit_probleme(&t);
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.7).unwrap();
        let x0 = Array3::from_shape_fn(grille.dims, |(i, j, k)| (i * 100 + j * 10 + k) as f64 * 0.01);
        let tr = optimizers::adam(&probleme, &x0, 0.1, 1, &[1], &device);
        for (idx, &v) in tr.snapshots[0].1.indexed_iter() {
            let attendu = if probleme.domain()[idx] { x0[idx] } else { 0.0 };
            assert!((v - attendu).abs() <= 1e-6 * attendu.abs().max(1.0), "{idx:?} : {v} contre {attendu}");
        }
        let _ = a;
    }
    /// COMPARAISON des optimiseurs (étape 4a) sur STA21 (5 % de bruit), `α = 0,01`, depuis l'adjoint normalisé, à budgets égaux de passes
    /// (10, 25, 50, 100). Pour chaque optimiseur, sous-optimalité relative `(f − f*) / (f₀ − f*)` (`f*` : le minimum, par gradient conjugué
    /// poussé à 200 itérations) et qualité (NCC, PSNR) contre l'atlas. Les optimiseurs à paramètre sont essayés avec plusieurs valeurs.
    /// À lancer en `--release --ignored --nocapture`.
    #[test]
    #[ignore = "données locales ; longue"]
    fn compare_optimizers_on_sta21() {
        let (atlas, stacks) = stacks_atlas_simules("STA21", 5);
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.01).unwrap();
        let c = construire_comparaison(&grille, probleme.support(), &atlas, &stacks);
        let x0 = probleme.initial_guess();
        let f0 = probleme.evaluate(&x0).value();
        let reference = optimizers::conjugate_gradient(&probleme, &x0, 202, &[]);
        let f_etoile = probleme.evaluate(&reference.x).value();
        let q_etoile = qualite(&c, &reference.x);
        println!("f₀ = {f0:.6e}, f* = {f_etoile:.6e} (CG, 200 itérations), qualité du minimum : NCC {:.4}, PSNR {:.2} dB", q_etoile.0, q_etoile.1);
        let points = [10, 25, 50, 100];
        let device = burn::tensor::Device::flex().autodiff();
        let essais: Vec<(String, Box<dyn Fn() -> (optimizers::Trace, usize)>)> = vec![
            ("gradient conjugué".into(), Box::new(|| (optimizers::conjugate_gradient(&probleme, &x0, 100, &points), 2))),
            ("plus forte pente".into(), Box::new(|| (optimizers::steepest_descent(&probleme, &x0, 100, &points), 1))),
            ("Barzilai-Borwein".into(), Box::new(|| (optimizers::barzilai_borwein(&probleme, &x0, 100, &points), 1))),
            ("Jacobi ω = 0,5".into(), Box::new(|| (optimizers::jacobi(&probleme, &x0, 0.5, 100, &points), 1))),
            ("Jacobi ω = 1".into(), Box::new(|| (optimizers::jacobi(&probleme, &x0, 1.0, 100, &points), 1))),
            ("Jacobi ω = 1,5".into(), Box::new(|| (optimizers::jacobi(&probleme, &x0, 1.5, 100, &points), 1))),
            ("Adam lr = 2".into(), Box::new(|| (optimizers::adam(&probleme, &x0, 2.0, 100, &points, &device), 1))),
            ("Adam lr = 10".into(), Box::new(|| (optimizers::adam(&probleme, &x0, 10.0, 100, &points, &device), 1))),
            ("Adam lr = 50".into(), Box::new(|| (optimizers::adam(&probleme, &x0, 50.0, 100, &points, &device), 1))),
            ("L-BFGS (pas fixe)".into(), Box::new(|| (optimizers::lbfgs(&probleme, &x0, 1.0, 10, false, 1.0, 100, &points, &device), 1))),
            ("L-BFGS (Wolfe)".into(), Box::new(|| (optimizers::lbfgs(&probleme, &x0, 1.0, 10, true, 1.0, 100, &points, &device), 1))),
        ];
        println!("{:20} | {:^27} | {:^45} | durée", "optimiseur", "sous-optimalité à 10/25/50/100", "PSNR (dB) à 10/25/50/100");
        for (nom, lancer) in &essais {
            let debut = std::time::Instant::now();
            let (tr, decalage) = lancer();
            let duree = debut.elapsed().as_secs_f64();
            let mut sous = Vec::new();
            let mut psnrs = Vec::new();
            for &k in &points {
                let indice = (k - decalage).min(tr.objective.len() - 1);
                sous.push(((tr.objective[indice] - f_etoile) / (f0 - f_etoile)).max(1e-12));
                let image = &tr.snapshots.iter().find(|(p, _)| *p == k).map(|(_, x)| x.clone());
                psnrs.push(image.as_ref().map_or(f64::NAN, |x| qualite(&c, x).1));
            }
            println!(
                "{nom:20} | {:7.1e} {:7.1e} {:7.1e} {:7.1e} | {:6.2} {:6.2} {:6.2} {:6.2} (NCC final {:.4}) | {duree:.0} s, {} passes",
                sous[0], sous[1], sous[2], sous[3], psnrs[0], psnrs[1], psnrs[2], psnrs[3], qualite(&c, &tr.x).0, tr.passes
            );
        }
    }
    /// Sondes sur les anomalies de `compare_optimizers_on_sta21` : (1) Adam avec des `lr` plus grands (l'optimum était au bord de la grille) ;
    /// (2) L-BFGS de Burn mis à l'échelle (valeur et gradient divisés par f₀) avec et sans Wolfe. Même protocole et même tableau.
    #[test]
    #[ignore = "données locales ; longue"]
    fn probe_optimizer_anomalies_on_sta21() {
        let (atlas, stacks) = stacks_atlas_simules("STA21", 5);
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.01).unwrap();
        let c = construire_comparaison(&grille, probleme.support(), &atlas, &stacks);
        let x0 = probleme.initial_guess();
        let f0 = probleme.evaluate(&x0).value();
        let f_etoile = probleme.evaluate(&optimizers::conjugate_gradient(&probleme, &x0, 202, &[]).x).value();
        let points = [10, 25, 50, 100];
        let device = burn::tensor::Device::flex().autodiff();
        let essais: Vec<(String, Box<dyn Fn() -> optimizers::Trace>)> = vec![
            ("Adam lr = 100".into(), Box::new(|| optimizers::adam(&probleme, &x0, 100.0, 100, &points, &device))),
            ("Adam lr = 200".into(), Box::new(|| optimizers::adam(&probleme, &x0, 200.0, 100, &points, &device))),
            ("Adam lr = 400".into(), Box::new(|| optimizers::adam(&probleme, &x0, 400.0, 100, &points, &device))),
            ("L-BFGS échelle f₀".into(), Box::new(|| optimizers::lbfgs(&probleme, &x0, 1.0, 10, false, f0, 100, &points, &device))),
            ("L-BFGS Wolfe échelle f₀".into(), Box::new(|| optimizers::lbfgs(&probleme, &x0, 1.0, 10, true, f0, 100, &points, &device))),
        ];
        println!("{:24} | {:^27} | {:^28}", "optimiseur", "sous-optimalité à 10/25/50/100", "PSNR (dB) à 10/25/50/100");
        for (nom, lancer) in &essais {
            let tr = lancer();
            let mut sous = Vec::new();
            let mut psnrs = Vec::new();
            for &k in &points {
                sous.push(((tr.objective[(k - 1).min(tr.objective.len() - 1)] - f_etoile) / (f0 - f_etoile)).max(1e-12));
                psnrs.push(tr.snapshots.iter().find(|(p, _)| *p == k).map_or(f64::NAN, |(_, x)| qualite(&c, x).1));
            }
            println!("{nom:24} | {:7.1e} {:7.1e} {:7.1e} {:7.1e} | {:6.2} {:6.2} {:6.2} {:6.2} ({} passes)", sous[0], sous[1], sous[2], sous[3], psnrs[0], psnrs[1], psnrs[2], psnrs[3], tr.passes);
        }
    }
    /// Sonde de l'explication de l'échec de L-BFGS de Burn (Wolfe, ou mise à l'échelle) : le premier pas vaut `min(1/‖g‖₁, 1) · lr` ; si ce pas est
    /// trop petit pour modifier `x` en `f32`, l'historique n'est jamais mis à jour. Avec `lr = c · ‖g₀‖₁` le premier pas vaut `c` : on essaie `c = 0,1`
    /// avec et sans Wolfe, sans mise à l'échelle de l'objectif.
    #[test]
    #[ignore = "données locales ; longue"]
    fn probe_lbfgs_initial_step_on_sta21() {
        let (atlas, stacks) = stacks_atlas_simules("STA21", 5);
        let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
        let probleme = ReconstructionProblem::new(&grille, &stacks, 0.01).unwrap();
        let c = construire_comparaison(&grille, probleme.support(), &atlas, &stacks);
        let x0 = probleme.initial_guess();
        let e0 = probleme.evaluate(&x0);
        let (f0, g1) = (e0.value(), e0.gradient.iter().map(|v| v.abs()).sum::<f64>());
        let f_etoile = probleme.evaluate(&optimizers::conjugate_gradient(&probleme, &x0, 202, &[]).x).value();
        println!("‖g₀‖₁ = {g1:.3e}, |g₀|max = {:.3e}", e0.gradient.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
        let points = [10, 25, 50, 100];
        let device = burn::tensor::Device::flex().autodiff();
        for (nom, wolfe, facteur) in [("sans Wolfe, premier pas 0,1", false, 0.1), ("Wolfe, premier pas 0,1", true, 0.1), ("Wolfe, premier pas 1", true, 1.0)] {
            let tr = optimizers::lbfgs(&probleme, &x0, facteur * g1, 10, wolfe, 1.0, 100, &points, &device);
            let sous: Vec<f64> = points.iter().map(|&k| ((tr.objective[(k - 1).min(tr.objective.len() - 1)] - f_etoile) / (f0 - f_etoile)).max(1e-12)).collect();
            let psnrs: Vec<f64> = points.iter().map(|&k| tr.snapshots.iter().find(|(p, _)| *p == k).map_or(f64::NAN, |(_, x)| qualite(&c, x).1)).collect();
            println!("{nom:30} | {:7.1e} {:7.1e} {:7.1e} {:7.1e} | {:6.2} {:6.2} {:6.2} {:6.2} ({} passes)", sous[0], sous[1], sous[2], sous[3], psnrs[0], psnrs[1], psnrs[2], psnrs[3], tr.passes);
        }
    }
    /// Si toutes les coupes d'un stack ont le même mouvement `M`, le problème est identique à celui d'un stack dont l'affine est `M · A` (même
    /// données, même masque), sur la même grille : valeur de chaque terme et gradient.
    #[test]
    fn whole_stack_motion_equals_a_stack_with_the_moved_affine() {
        let t = Temp::new("pose_stack");
        let (grille, stacks) = petit_probleme(&t);
        let mouvements = [
            mouvement_rigide(&Vector3::new(0.05, -0.03, 0.04), &Vector3::new(1.2, -0.8, 0.5), &Vector3::new(2.0, 1.0, -1.0)),
            mouvement_rigide(&Vector3::new(-0.04, 0.06, -0.02), &Vector3::new(-0.9, 1.1, 0.3), &Vector3::new(-1.0, 2.0, 0.5)),
        ];
        let mut poses = SlicePoses::identity(&stacks);
        let mut deplaces = Vec::new();
        for (si, stack) in stacks.iter().enumerate() {
            for k in 0..stack.dim().2 {
                poses.set(si, k, mouvements[si]);
            }
            let masque = stack.brain_mask().unwrap().voxels().mapv(u8::from);
            let sp = stack.spacing();
            deplaces.push(stack_avec_masque(&t, &format!("deplace{si}"), stack.data(), &masque, &(mouvements[si] * stack.affine()), [sp[0] as f32, sp[1] as f32, sp[2] as f32]));
        }
        let (a, b) = (ReconstructionProblem::with_poses(&grille, &stacks, &poses, 0.7).unwrap(), ReconstructionProblem::new(&grille, &deplaces, 0.7).unwrap());
        assert_eq!(a.domain().iter().filter(|&&d| d).count(), b.domain().iter().filter(|&&d| d).count(), "même domaine");
        let mut alea = Alea(4242);
        let x = Array3::from_shape_fn(grille.dims, |_| alea.suivant() * 10.0);
        let (ea, eb) = (a.evaluate(&x), b.evaluate(&x));
        let relatif = |p: f64, q: f64| (p - q).abs() / q.abs().max(1e-12);
        println!("attache {:.6} / {:.6} ; régularisation {:.6} / {:.6}", ea.data, eb.data, ea.regularization, eb.regularization);
        assert!(relatif(ea.data, eb.data) < 1e-5 && relatif(ea.regularization, eb.regularization) < 1e-9);
        let norme = eb.gradient.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let pire = ea.gradient.iter().zip(eb.gradient.iter()).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max) / norme;
        println!("gradient : écart max relatif {pire:.1e}");
        assert!(pire < 1e-5, "{pire:.2e}");
        // et ce n'est pas trivial : sans les poses, le problème est bien différent
        let sans = ReconstructionProblem::new(&grille, &stacks, 0.7).unwrap().evaluate(&x);
        assert!(relatif(sans.data, eb.data) > 1e-3, "les poses doivent changer le problème : {} contre {}", sans.data, eb.data);
    }

    #[test]
    fn poses_must_have_the_shape_of_the_stacks() {
        let t = Temp::new("poses_forme");
        let (grille, stacks) = petit_probleme(&t);
        let autre = SlicePoses::identity(&stacks[..1]);
        assert!(matches!(ReconstructionProblem::with_poses(&grille, &stacks, &autre, 0.7), Err(SvrError::PoseMismatch)));
        assert!(matches!(normalized_adjoint_with_poses(&grille, &stacks, &autre), Err(SvrError::PoseMismatch)));
        assert!(SlicePoses::identity(&stacks).matches(&stacks));
        assert!(!autre.matches(&stacks));
    }
    /// Un atlas et trois stacks (axial, coronal, sagittal) simulés avec un **mouvement rigide propre à chaque coupe**, et ce mouvement vrai.
    /// Pour chaque coupe : le mouvement est tiré uniformément dans ± `amplitude` (degrés par axe et mm par axe) autour du centre de la
    /// coupe, la coupe est simulée par l'opérateur **à sa position réelle** (PSF orientée avec elle), puis on ajoute `bruit_pct` % de bruit gaussien.
    /// Les stacks gardent leur affine d'en-tête : le mouvement est la quantité inconnue que le recalage devra retrouver. Mis en cache dans
    /// `data/atlas/sim/<atlas>_mvt<amplitude>_b<bruit>/` (les poses vraies dans `poses.tsv`).
    fn stacks_atlas_avec_mouvement(nom_atlas: &str, amplitude: u32, bruit_pct: u32) -> (Volume, Vec<Stack>, SlicePoses) {
        let (atlas, modeles) = stacks_atlas_sans_bruit(nom_atlas);
        let dossier = PathBuf::from(format!("{}/data/atlas/sim/{nom_atlas}_mvt{amplitude}_b{bruit_pct}", racine()));
        std::fs::create_dir_all(&dossier).unwrap();
        let fichier_poses = dossier.join("poses.tsv");
        if !fichier_poses.exists() {
            let vraies = poses_aleatoires(&modeles, f64::from(amplitude), f64::from(amplitude), 1000 + u64::from(amplitude));
            let mut lignes = Vec::new();
            for (n, modele) in modeles.iter().enumerate() {
                let dims = modele.dim();
                let (mut data, mut masque) = (Array3::<f32>::zeros(dims), Array3::<u8>::zeros(dims));
                for coupe in modele.slices() {
                    let posee = coupe.with_motion(*vraies.get(n, coupe.index()));
                    let sim = atlas.simulate_slice(&posee, &posee.psf());
                    for ((i, j), &v) in sim.values.indexed_iter() {
                        data[[i, j, coupe.index()]] = v;
                        masque[[i, j, coupe.index()]] = u8::from(sim.coverage[[i, j]] >= 0.99 && v > 150.0);
                    }
                    let m = vraies.get(n, coupe.index());
                    lignes.push(format!("{n}\t{}\t{}", coupe.index(), m.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>().join("\t")));
                }
                if bruit_pct > 0 {
                    let (somme, n_m) = data.iter().zip(masque.iter()).filter(|(_, &m)| m > 0).fold((0.0_f64, 0usize), |(s, n), (&v, _)| (s + f64::from(v), n + 1));
                    let sigma = f64::from(bruit_pct) / 100.0 * somme / n_m as f64;
                    let mut alea = Alea(0xC0FFEE ^ (n as u64 + 1) * 7919 ^ u64::from(amplitude));
                    data.mapv_inplace(|v| {
                        let gauss: f64 = (0..12).map(|_| alea.suivant()).sum::<f64>() - 6.0;
                        (f64::from(v) + sigma * gauss) as f32
                    });
                }
                let sp = modele.spacing();
                let h = en_tete(modele.affine(), [sp[0] as f32, sp[1] as f32, sp[2] as f32]);
                WriterOptions::new(dossier.join(format!("stack{n}.nii.gz"))).reference_header(&h).write_nifti(&data).unwrap();
                WriterOptions::new(dossier.join(format!("stack{n}_mask.nii.gz"))).reference_header(&h).write_nifti(&masque).unwrap();
            }
            std::fs::write(&fichier_poses, lignes.join("\n")).unwrap();
        }
        let stacks: Vec<Stack> = (0..modeles.len())
            .map(|n| {
                let mut stack = Stack::read(&dossier.join(format!("stack{n}.nii.gz"))).unwrap();
                stack.set_brain_mask(&dossier.join(format!("stack{n}_mask.nii.gz"))).unwrap();
                stack
            })
            .collect();
        let mut vraies = SlicePoses::identity(&stacks);
        for ligne in std::fs::read_to_string(&fichier_poses).unwrap().lines() {
            let c: Vec<&str> = ligne.split('\t').collect();
            let valeurs: Vec<f64> = c[2..].iter().map(|v| v.parse().unwrap()).collect();
            vraies.set(c[0].parse().unwrap(), c[1].parse().unwrap(), Matrix4::from_iterator(valeurs.iter().copied()));
        }
        (atlas, stacks, vraies)
    }

    /// BORNES de l'évaluation de la boucle recalage / reconstruction (étude 06 §15) sur STA31 avec un mouvement propre à chaque coupe : la
    /// reconstruction avec les **vraies poses** (borne haute) et **sans correction**, c'est-à-dire avec les poses d'en-tête (borne basse). Le
    /// déplacement quadratique moyen (sur les pixels de masque) entre poses d'en-tête et vraies poses donne l'erreur de départ du recalage.
    /// À lancer en `--release --ignored --nocapture`.
    #[test]
    #[ignore = "données locales ; longue"]
    fn bounds_with_true_poses_and_without_correction_on_sta31() {
        for amplitude in [2u32, 4, 6] {
            let (atlas, stacks, vraies) = stacks_atlas_avec_mouvement("STA31", amplitude, 5);
            let grille = reconstruction_grid(&stacks, 0.8, 10.0).unwrap();
            let identite = SlicePoses::identity(&stacks);
            let (haute, basse) = (ReconstructionProblem::with_poses(&grille, &stacks, &vraies, 0.01).unwrap(), ReconstructionProblem::with_poses(&grille, &stacks, &identite, 0.01).unwrap());
            let init_vraies = normalized_adjoint_with_poses(&grille, &stacks, &vraies).unwrap();
            let c = construire_comparaison(&grille, &init_vraies.support, &atlas, &stacks);
            // erreur de départ : déplacement entre pose d'en-tête et vraie pose, sur les pixels de masque
            let (mut somme, mut n) = (0.0, 0usize);
            for (si, stack) in stacks.iter().enumerate() {
                let masque = stack.brain_mask().unwrap().voxels();
                for coupe in stack.slices() {
                    let m = vraies.get(si, coupe.index());
                    for ((i, j), &dedans) in masque.index_axis(Axis(2), coupe.index()).indexed_iter() {
                        if dedans {
                            let x = coupe.pixel_to_world(i as f64, j as f64);
                            somme += ((m * x.push(1.0)).xyz() - x).norm_squared();
                            n += 1;
                        }
                    }
                }
            }
            println!("--- mouvement ±{amplitude}°/±{amplitude} mm : erreur de départ (déplacement quadratique moyen) {:.2} mm", (somme / n as f64).sqrt());
            for (nom, probleme) in [("vraies poses (borne haute)", &haute), ("sans correction (borne basse)", &basse)] {
                let debut = std::time::Instant::now();
                let r = probleme.conjugate_gradient(&probleme.initial_guess(), 300, 1e-4);
                ligne_qualite(nom, &c, &r.x, r.iterations, *r.residual_history.last().unwrap(), debut.elapsed().as_secs_f64());
            }
        }
    }
}
