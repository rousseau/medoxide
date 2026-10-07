//! Reconstruction sur grille (étape 4a) : la grille du volume à estimer et son initialisation par **adjoint normalisé**.
//!
//! Le volume reconstruit est une grille isotrope **alignée sur les axes du monde** (RAS+), qui ne dépend d'aucun stack. Son
//! étendue est la boîte englobante de tous les voxels de masque cérébral de tous les stacks, plus une marge.

use ndarray::{Array3, ArrayView2, Axis};
use rayon::prelude::*;
use nalgebra::{Matrix4, Vector3};

use crate::{Stack, SvrError, Volume};

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

/// Toutes les coupes de tous les stacks, chacune avec son masque 2D (vue sur le masque du stack). Les stacks doivent avoir un masque.
fn coupes_avec_masque<'a>(stacks: &'a [Stack]) -> Vec<(ArrayView2<'a, bool>, crate::Slice<'a>)> {
    stacks
        .iter()
        .flat_map(|stack| {
            let masque = stack.brain_mask().expect("le masque a été vérifié").voxels();
            stack.slices().map(move |coupe| (masque.index_axis(Axis(2), coupe.index()), coupe))
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
    let geometrie = grid.zeros();
    for stack in stacks {
        stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?;
    }
    let coupes = coupes_avec_masque(stacks);
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
    domain: Array3<bool>,
    initial: Array3<f32>,
}

impl<'a> ReconstructionProblem<'a> {
    /// Construit le problème : calcule l'adjoint normalisé, dont le support définit le domaine et l'image le point de départ.
    ///
    /// # Erreurs
    /// [`SvrError::InvalidRegularization`] si `alpha` est négatif ou non fini ; [`SvrError::NoMask`] si un stack n'a pas de masque.
    pub fn new(grid: &GridSpec, stacks: &'a [Stack], alpha: f64) -> Result<ReconstructionProblem<'a>, SvrError> {
        // `!(a >= 0.0)` est vrai aussi pour NaN.
        if !(alpha >= 0.0) || !alpha.is_finite() {
            return Err(SvrError::InvalidRegularization);
        }
        let init = normalized_adjoint(grid, stacks)?;
        let domain = init.support.mapv(|s| s >= SUPPORT_MIN);
        Ok(ReconstructionProblem { grid: grid.clone(), stacks, alpha, domain, initial: init.image })
    }

    /// Voxels inconnus du problème.
    pub fn domain(&self) -> &Array3<bool> {
        &self.domain
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
        let (mut gradient, data) = coupes_avec_masque(self.stacks)
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
        let b = self.right_hand_side();
        let norme_b = dot(&b, &b).sqrt();
        let mut x = x0.clone();
        for (v, &d) in x.iter_mut().zip(self.domain.iter()) {
            if !d {
                *v = 0.0;
            }
        }
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
    /// L'atlas de Gholipour et trois stacks (axial, coronal, sagittal ; pixels 0,8 mm, épaisseur 3,5 mm, légèrement obliques) simulés
    /// par l'opérateur de l'étape 2, sans mouvement ni bruit. Les stacks sont mis en **cache** dans `data/atlas/sim/` (hors de Git) : la
    /// simulation (≈ 1 min en release) n'est refaite que si les fichiers manquent. Le masque d'un pixel est « couvert et tissu (> 150) ».
    fn stacks_atlas_simules() -> (Volume, Vec<Stack>) {
        let chemin = format!("{}/data/atlas/gholipour/STA21.nii.gz", racine());
        let atlas = Volume::from_stack(&Stack::read(Path::new(&chemin)).expect("atlas absent : voir data/atlas/gholipour"));
        let dossier = PathBuf::from(format!("{}/data/atlas/sim", racine()));
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
        let (atlas, stacks) = stacks_atlas_simules();
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
            for stack in stacks {
                for coupe in stack.slices() {
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
        let t = Temp::new("dense");
        let (grille, stacks) = petit_probleme(&t);
        let alpha = 0.7;
        let probleme = ReconstructionProblem::new(&grille, &stacks, alpha).unwrap();
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
        let (_atlas, stacks) = stacks_atlas_simules();
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
}
