//! Recalage de **toutes les coupes** contre un volume de référence (boucle recalage / reconstruction) : chaque coupe est recalée depuis sa pose
//! courante, le delta estimé est composé à gauche (`D · M`) et un rapport par coupe est rendu.

use burn::prelude::Device;
use nalgebra::{Matrix4, Vector3};
use ndarray::{Array3, ArrayView2, Axis};

use crate::diff::{pose_to_matrix, register_slice_robust, rotation_scale_mm, RobustConfig, VolumeTensors};
use crate::recon::{adjoint_sums, normalized_adjoint_with_poses, GridSpec, ReconstructionProblem, SlicePoses, SUPPORT_MIN};
use crate::{Stack, SvrError, Volume};

/// Nombre minimal de pixels de masque pour recaler une coupe. En deçà, la NCC est trop peu informée (et la mise à l'échelle de la rotation, mal
/// définie) : la coupe garde sa pose et le rapport la signale.
pub const MIN_MASK_PIXELS: usize = 500;

/// Rapport du recalage d'une coupe.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceReport {
    /// Indice du stack.
    pub stack: usize,
    /// Indice de la coupe dans son stack.
    pub slice: usize,
    /// `false` si la coupe n'a pas été recalée (masque de moins de [`MIN_MASK_PIXELS`] pixels) : sa pose est inchangée.
    pub registered: bool,
    /// NCC finale du premier recalage (celui qui part de la pose courante) ; 0 si la coupe n'a pas été recalée.
    pub ncc_first: f64,
    /// NCC finale de la pose retenue (la meilleure, après les relances éventuelles) ; 0 si la coupe n'a pas été recalée.
    pub ncc_final: f64,
    /// Nombre de recalages effectués (1 si la coupe n'était pas suspecte) ; 0 si la coupe n'a pas été recalée.
    pub runs: usize,
    /// Amplitude de la correction : déplacement quadratique moyen, sur les pixels du masque, entre l'ancienne et la nouvelle pose (mm).
    pub correction_rms_mm: f64,
}

/// Résultat de [`register_slices`].
#[derive(Debug, Clone)]
pub struct AlignmentReport {
    /// Nouvelles poses de toutes les coupes.
    pub poses: SlicePoses,
    /// Un rapport par coupe, dans l'ordre des stacks puis des coupes.
    pub slices: Vec<SliceReport>,
}

/// **Recale chaque coupe** de chaque stack contre `reference`, depuis sa pose courante dans `poses`, par [`register_slice_robust`] sur les pixels du masque
/// cérébral. Pour une coupe à la pose `M` : les positions des pixels et le pivot P3 sont ceux de la coupe **déjà déplacée** par `M` ; le recalage
/// estime un delta `D` depuis la pose nulle (6 paramètres en mm équivalents autour du pivot) ; la nouvelle pose est `D · M`.
///
/// `device` doit avoir l'autodiff activé. Les coupes de moins de [`MIN_MASK_PIXELS`] pixels de masque gardent leur pose et sont signalées
/// (`registered = false`).
///
/// # Erreurs
/// [`SvrError::NoMask`] si un stack n'a pas de masque ; [`SvrError::PoseMismatch`] si `poses` n'a pas la forme de `stacks`.
pub fn register_slices(stacks: &[Stack], poses: &SlicePoses, reference: &VolumeTensors, config: RobustConfig, device: &Device) -> Result<AlignmentReport, SvrError> {
    verifier_entrees(stacks, poses)?;
    let mut nouvelles = poses.clone();
    let mut rapports = Vec::new();
    for (si, stack) in stacks.iter().enumerate() {
        for coupe in stack.slices() {
            let k = coupe.index();
            let courante = *poses.get(si, k);
            match preparer_coupe(stack, k, courante) {
                None => rapports.push(SliceReport { stack: si, slice: k, registered: false, ncc_first: 0.0, ncc_final: 0.0, runs: 0, correction_rms_mm: 0.0 }),
                Some(preparee) => {
                    let (delta, rapport) = recaler_preparee(&preparee, si, k, reference, config, device);
                    nouvelles.set(si, k, delta * courante);
                    rapports.push(rapport);
                }
            }
        }
    }
    Ok(AlignmentReport { poses: nouvelles, slices: rapports })
}

/// Vérifie que `poses` a la forme de `stacks` et que chaque stack a un masque.
fn verifier_entrees(stacks: &[Stack], poses: &SlicePoses) -> Result<(), SvrError> {
    if !poses.matches(stacks) {
        return Err(SvrError::PoseMismatch);
    }
    for stack in stacks {
        stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?;
    }
    Ok(())
}

/// Ce qu'il faut pour recaler une coupe : positions monde (à sa pose courante) et intensités de ses pixels de masque, pivot P3 et échelle de rotation.
struct CoupePreparee {
    points: Vec<Vector3<f64>>,
    intensites: Vec<f64>,
    pivot: Vector3<f64>,
    echelle: f64,
}

/// Prépare la coupe `k` d'un stack à la pose `courante` ; `None` si elle a moins de [`MIN_MASK_PIXELS`] pixels de masque ou si son pivot ou son échelle sont indéfinis.
fn preparer_coupe(stack: &Stack, k: usize, courante: Matrix4<f64>) -> Option<CoupePreparee> {
    let masque = stack.brain_mask().expect("vérifié").voxels();
    let coupe = stack.slices().nth(k).expect("coupe existante");
    let posee = coupe.with_motion(courante);
    let (nx, ny) = posee.dim();
    let donnees = posee.data();
    let (mut points, mut intensites) = (Vec::new(), Vec::new());
    for j in 0..ny {
        for i in 0..nx {
            if masque[[i, j, k]] {
                points.push(posee.pixel_to_world(i as f64, j as f64));
                intensites.push(f64::from(donnees[[i, j]]));
            }
        }
    }
    let pivot = posee.brain_pivot()?;
    let echelle = rotation_scale_mm(&points, &pivot)?;
    (points.len() >= MIN_MASK_PIXELS).then_some(CoupePreparee { points, intensites, pivot, echelle })
}

/// Recale une coupe préparée contre `reference` ; rend le delta (à composer à gauche de la pose courante) et le rapport.
fn recaler_preparee(p: &CoupePreparee, si: usize, k: usize, reference: &VolumeTensors, config: RobustConfig, device: &Device) -> (Matrix4<f64>, SliceReport) {
    let resultat = register_slice_robust(
        reference,
        vecteurs_vers_tenseur(&p.points, device),
        tenseur_1d(&[p.pivot.x, p.pivot.y, p.pivot.z], device),
        p.echelle,
        tenseur_1d(&p.intensites, device),
        tenseur_1d(&vec![1.0; p.points.len()], device),
        config,
    );
    let delta: Matrix4<f64> = pose_to_matrix(&resultat.params, &p.pivot, p.echelle);
    let correction = (p.points.iter().map(|x| ((delta * x.push(1.0)).xyz() - x).norm_squared()).sum::<f64>() / p.points.len() as f64).sqrt();
    (delta, SliceReport { stack: si, slice: k, registered: true, ncc_first: resultat.first_ncc, ncc_final: resultat.ncc, runs: resultat.runs, correction_rms_mm: correction })
}

/// **Recalage de chaque coupe contre une référence qui ne la contient pas** (« leave-one-slice-out ») : la référence de la coupe `k` est l'adjoint normalisé de TOUTES les autres
/// coupes de TOUS les stacks, `(N − nₖ) / (D − dₖ)`, où `N = Σ Aₖᵀ(mₖ yₖ)` et `D = Σ Aₖᵀ mₖ` sont les sommes de l'adjoint normalisé et `nₖ`, `dₖ` la contribution de la coupe `k`
/// (retirée exactement, sans refaire la somme). Elle est riche (les deux autres stacks et les autres coupes du sien) et sans auto-biais (la coupe n'y est plus). Là où le support
/// restant `D − dₖ` est inférieur à [`SUPPORT_MIN`], la référence vaut 0, comme [`normalized_adjoint_with_poses`].
///
/// Même sortie et mêmes conventions que [`register_slices`] (delta composé à gauche, coupes de moins de [`MIN_MASK_PIXELS`] pixels ignorées).
///
/// # Erreurs
/// Celles de [`register_slices`].
pub fn register_slices_leave_slice_out(grid: &GridSpec, stacks: &[Stack], poses: &SlicePoses, config: RobustConfig, device: &Device) -> Result<AlignmentReport, SvrError> {
    verifier_entrees(stacks, poses)?;
    let (numerateur, denominateur) = adjoint_sums(grid, stacks, poses)?;
    let geometrie = grid.zeros();
    let mut nouvelles = poses.clone();
    let mut rapports = Vec::new();
    for (si, stack) in stacks.iter().enumerate() {
        let masque = stack.brain_mask().expect("vérifié").voxels();
        for coupe in stack.slices() {
            let k = coupe.index();
            let courante = *poses.get(si, k);
            let Some(preparee) = preparer_coupe(stack, k, courante) else {
                rapports.push(SliceReport { stack: si, slice: k, registered: false, ncc_first: 0.0, ncc_final: 0.0, runs: 0, correction_rms_mm: 0.0 });
                continue;
            };
            let (num_k, den_k) = contribution_de_la_coupe(&geometrie, &coupe.with_motion(courante), masque.index_axis(Axis(2), k));
            let image = reference_sans_la_coupe(&numerateur, &denominateur, &num_k, &den_k);
            let reference = Volume::new(image, grid.affine)?;
            let (delta, rapport) = recaler_preparee(&preparee, si, k, &VolumeTensors::new(&reference, device), config, device);
            nouvelles.set(si, k, delta * courante);
            rapports.push(rapport);
        }
    }
    Ok(AlignmentReport { poses: nouvelles, slices: rapports })
}

/// L'adjoint normalisé de TOUTES les coupes (celle qu'on recale comprise) comme volume : la référence « complète » à laquelle comparer [`register_slices_leave_slice_out`].
pub(crate) fn normalized_reference_volume(grid: &GridSpec, stacks: &[Stack], poses: &SlicePoses) -> Result<Volume, SvrError> {
    Volume::new(normalized_adjoint_with_poses(grid, stacks, poses)?.image, grid.affine)
}

/// Contribution `(nₖ, dₖ) = (Aₖᵀ(mₖ yₖ), Aₖᵀ mₖ)` d'une coupe (déjà à sa pose) à l'adjoint normalisé, sur la grille de `geometrie`.
pub(crate) fn contribution_de_la_coupe(geometrie: &Volume, coupe: &crate::Slice, masque: ArrayView2<bool>) -> (Array3<f64>, Array3<f64>) {
    let dims = geometrie.dim();
    let (mut num, mut den) = (Array3::<f64>::zeros(dims), Array3::<f64>::zeros(dims));
    geometrie.back_project_with_weight(coupe, &coupe.psf(), coupe.data(), masque, &mut num, &mut den);
    (num, den)
}

/// `(N − nₖ) / (D − dₖ)` en `f32`, 0 là où `D − dₖ < SUPPORT_MIN`.
pub(crate) fn reference_sans_la_coupe(n: &Array3<f64>, d: &Array3<f64>, n_k: &Array3<f64>, d_k: &Array3<f64>) -> Array3<f32> {
    let mut image = Array3::<f32>::zeros(n.dim());
    for (idx, v) in image.indexed_iter_mut() {
        let reste = d[idx] - d_k[idx];
        if reste >= f64::from(SUPPORT_MIN) {
            *v = ((n[idx] - n_k[idx]) / reste) as f32;
        }
    }
    image
}

/// Nombre maximal de pixels de masque utilisés pour recaler un stack entier (sous-échantillonnage régulier au-delà) : le coût de la NCC croît avec ce nombre.
const MAX_STACK_POINTS: usize = 100_000;

/// Rapport du recalage d'un stack entier ([`register_stacks`]) : dernière visite de ce stack.
#[derive(Debug, Clone, PartialEq)]
pub struct StackReport {
    /// Indice du stack.
    pub stack: usize,
    /// `false` si le stack n'a jamais été recalé (le stack d'ancrage, ou trop peu de pixels de masque).
    pub registered: bool,
    /// NCC finale du premier recalage (celui qui part de la pose courante).
    pub ncc_first: f64,
    /// NCC finale de la pose retenue (après les relances éventuelles).
    pub ncc_final: f64,
    /// Nombre de recalages effectués (1 si le stack n'était pas suspect).
    pub runs: usize,
    /// Amplitude de la dernière correction : déplacement quadratique moyen des pixels du masque du stack (mm).
    pub correction_rms_mm: f64,
}

/// Résultat de [`register_stacks`].
#[derive(Debug, Clone)]
pub struct StackAlignment {
    /// Poses après alignement : toutes les coupes d'un stack ont reçu la même correction, composée à gauche de leur pose d'origine.
    pub poses: SlicePoses,
    /// Un rapport par stack (le stack d'ancrage, d'indice 0, n'est pas recalé).
    pub stacks: Vec<StackReport>,
}

/// **Recalage de stacks entiers** : estime, pour chaque stack, UN mouvement rigide partagé par toutes ses coupes (6 paramètres au lieu de 6 par coupe), de façon à
/// aligner les stacks les uns sur les autres. Il retire la partie *cohérente* du mouvement (le décalage entre stacks) avant le recalage par coupe de la boucle.
///
/// Le stack 0 est l'**ancre** : il ne bouge pas et définit le repère. Chaque autre stack est recalé (même NCC et même recalage robuste que par coupe, sur tous ses
/// pixels de masque, sous-échantillonnés à [`MAX_STACK_POINTS`] ; pivot = barycentre de ces pixels) contre la référence `adjoint normalisé des autres stacks`
/// ([`normalized_adjoint_with_poses`], sur la grille `grid`, sans gradient conjugué). D'abord une passe d'ancrage (le stack `s` contre les stacks `0..s` déjà alignés),
/// puis `sweeps` balayages où chaque stack non ancre est recalé contre **tous** les autres. La correction `D` d'un stack est composée à gauche : pose `D · M` pour chaque
/// coupe de pose `M`.
///
/// Avec un seul stack, les poses sont rendues inchangées.
///
/// # Erreurs
/// [`SvrError::NoMask`], [`SvrError::PoseMismatch`] (comme [`register_slices`]) ; celles de [`normalized_adjoint_with_poses`].
pub fn register_stacks(
    grid: &GridSpec,
    stacks: &[Stack],
    poses: &SlicePoses,
    sweeps: usize,
    config: RobustConfig,
    device: &Device,
) -> Result<StackAlignment, SvrError> {
    if !poses.matches(stacks) {
        return Err(SvrError::PoseMismatch);
    }
    for stack in stacks {
        stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?;
    }
    let n = stacks.len();
    let mut nouvelles = poses.clone();
    let mut rapports: Vec<StackReport> = (0..n).map(|stack| StackReport { stack, registered: false, ncc_first: 0.0, ncc_final: 0.0, runs: 0, correction_rms_mm: 0.0 }).collect();
    let mut visites: Vec<(usize, Vec<usize>)> = (1..n).map(|s| (s, (0..s).collect())).collect();
    for _ in 0..sweeps {
        visites.extend((1..n).map(|s| (s, (0..n).filter(|&t| t != s).collect())));
    }
    for (s, references) in visites {
        let sous_stacks: Vec<Stack> = references.iter().map(|&t| stacks[t].clone()).collect();
        let image = normalized_adjoint_with_poses(grid, &sous_stacks, &nouvelles.select(&references))?.image;
        let reference = Volume::new(image, grid.affine)?;
        if let Some(rapport) = recaler_un_stack(&stacks[s], s, &mut nouvelles, &reference, config, device) {
            rapports[s] = rapport;
        }
    }
    Ok(StackAlignment { poses: nouvelles, stacks: rapports })
}

/// Recale le stack `s` contre `reference` et compose la correction à gauche des poses de toutes ses coupes. `None` si le stack a trop peu de pixels de masque.
fn recaler_un_stack(stack: &Stack, s: usize, poses: &mut SlicePoses, reference: &Volume, config: RobustConfig, device: &Device) -> Option<StackReport> {
    let masque = stack.brain_mask().expect("vérifié").voxels();
    let (mut points, mut intensites) = (Vec::new(), Vec::new());
    for coupe in stack.slices() {
        let k = coupe.index();
        let posee = coupe.with_motion(*poses.get(s, k));
        let donnees = posee.data();
        let (nx, ny) = posee.dim();
        for j in 0..ny {
            for i in 0..nx {
                if masque[[i, j, k]] {
                    points.push(posee.pixel_to_world(i as f64, j as f64));
                    intensites.push(f64::from(donnees[[i, j]]));
                }
            }
        }
    }
    if points.len() < MIN_MASK_PIXELS {
        return None;
    }
    let pas = points.len().div_ceil(MAX_STACK_POINTS);
    let (points, intensites): (Vec<Vector3<f64>>, Vec<f64>) = points.into_iter().zip(intensites).step_by(pas).unzip();
    let pivot = points.iter().sum::<Vector3<f64>>() / points.len() as f64;
    let echelle = rotation_scale_mm(&points, &pivot)?;
    let resultat = register_slice_robust(
        &VolumeTensors::new(reference, device),
        vecteurs_vers_tenseur(&points, device),
        tenseur_1d(&[pivot.x, pivot.y, pivot.z], device),
        echelle,
        tenseur_1d(&intensites, device),
        tenseur_1d(&vec![1.0; points.len()], device),
        config,
    );
    let delta: Matrix4<f64> = pose_to_matrix(&resultat.params, &pivot, echelle);
    for k in 0..stack.dim().2 {
        let courante = *poses.get(s, k);
        poses.set(s, k, delta * courante);
    }
    let correction = (points.iter().map(|x| ((delta * x.push(1.0)).xyz() - x).norm_squared()).sum::<f64>() / points.len() as f64).sqrt();
    Some(StackReport { stack: s, registered: true, ncc_first: resultat.first_ncc, ncc_final: resultat.ncc, runs: resultat.runs, correction_rms_mm: correction })
}

/// **Recalage du gros vers le fin** : comme [`register_slices`] contre `reference`, mais en deux passes si `sigma_voxels > 0` : d'abord contre la référence floutée
/// de `sigma_voxels` voxels ([`Volume::blurred`]), puis contre la référence nette depuis les poses du premier passage (composées à gauche, comme toujours). Une
/// référence floue a une NCC plus lisse, donc un bassin d'attraction plus large : une coupe très mal placée y descend vers le bon optimum, que le passage net affine.
/// Avec `sigma_voxels = 0`, un seul passage net (identique à [`register_slices`]).
///
/// Le rapport est celui du **dernier** passage (NCC, relances, `correction_rms_mm` : la correction de ce passage seulement, pas la correction totale).
///
/// # Erreurs
/// Celles de [`register_slices`] ; [`SvrError::InvalidBlur`] si `sigma_voxels` est négatif ou non fini.
pub fn register_slices_coarse_to_fine(
    stacks: &[Stack],
    poses: &SlicePoses,
    reference: &Volume,
    sigma_voxels: f64,
    config: RobustConfig,
    device: &Device,
) -> Result<AlignmentReport, SvrError> {
    if sigma_voxels == 0.0 {
        return register_slices(stacks, poses, &VolumeTensors::new(reference, device), config, device);
    }
    let floue = reference.blurred(sigma_voxels)?;
    let premier = register_slices(stacks, poses, &VolumeTensors::new(&floue, device), config, device)?;
    register_slices(stacks, &premier.poses, &VolumeTensors::new(reference, device), config, device)
}

/// Réglages de la boucle recalage / reconstruction ([`reconstruct_with_motion_correction`]).
#[derive(Debug, Clone, Copy)]
pub struct LoopConfig {
    /// Nombre de cycles « recaler puis reconstruire » après la première reconstruction.
    pub cycles: usize,
    /// Poids `α` de la régularisation de la reconstruction (voir [`ReconstructionProblem`]).
    pub alpha: f64,
    /// Itérations maximales du gradient conjugué à chaque reconstruction.
    pub cg_max_iterations: usize,
    /// Tolérance relative du gradient conjugué.
    pub cg_tolerance: f64,
    /// Réglages du recalage de chaque coupe.
    pub robust: RobustConfig,
    /// Nombre de premiers cycles où chaque stack est recalé contre une reconstruction faite **sans lui** (avec les poses courantes des autres
    /// stacks), pour que ses coupes n'aient pas imprimé leur propre erreur dans la référence ; les cycles suivants recalent contre la reconstruction
    /// complète. Coûte une reconstruction par stack et par cycle concerné ; exige au moins 2 stacks si non nul.
    pub leave_one_stack_out_cycles: usize,
    /// Écart type (en voxels de la grille) du flou du premier passage de recalage : si non nul, chaque recalage se fait en deux passes, d'abord contre la référence
    /// **floutée** (bassin d'attraction plus large, voir [`Volume::blurred`]), puis contre la référence nette depuis les poses du premier ([`register_slices_coarse_to_fine`]).
    /// 0 : un seul passage net. Mesuré sur une référence parfaite : 2 voxels (1,6 mm) corrigent les échecs à ±6° sans rien dégrader, 4 voxels créent des échecs rares.
    pub coarse_sigma_voxels: f64,
    /// Nombre de premiers cycles où chaque coupe est recalée contre une référence qui ne la contient pas, l'adjoint normalisé de toutes les AUTRES coupes
    /// ([`register_slices_leave_slice_out`]) ; prioritaire sur `leave_one_stack_out_cycles`. 0 : désactivé. Voir l'étude 06 §30 : option expérimentale.
    pub leave_one_slice_out_cycles: usize,
}

/// Résultat de la boucle.
#[derive(Debug, Clone)]
pub struct LoopResult {
    /// Poses finales (celles de la dernière reconstruction).
    pub poses: SlicePoses,
    /// Dernière reconstruction, sur la grille.
    pub volume: Array3<f64>,
    /// Rapport du recalage de chaque cycle (`config.cycles` rapports).
    pub cycles: Vec<AlignmentReport>,
    /// Itérations du gradient conjugué de chaque reconstruction (`config.cycles + 1` valeurs).
    pub cg_iterations: Vec<usize>,
}

/// **Boucle recalage / reconstruction** : reconstruit le volume avec les poses `initial`, puis répète `config.cycles` fois « recaler toutes les
/// coupes contre le volume courant ([`register_slices`]), reconstruire avec les nouvelles poses ». Chaque reconstruction repart (démarrage à chaud)
/// de la précédente, restreinte au nouveau domaine.
///
/// `observateur(c, x, poses)` est appelé après chaque reconstruction (`c = 0` : poses de départ ; `c = config.cycles` : reconstruction finale) ;
/// il sert à mesurer la qualité à chaque cycle sans que la boucle connaisse la vérité.
///
/// `device` doit avoir l'autodiff activé.
///
/// # Erreurs
/// Celles de [`ReconstructionProblem::with_poses`] et de [`register_slices`].
pub fn reconstruct_with_motion_correction(
    grid: &GridSpec,
    stacks: &[Stack],
    initial: &SlicePoses,
    config: LoopConfig,
    device: &Device,
    mut observateur: impl FnMut(usize, &Array3<f64>, &SlicePoses),
) -> Result<LoopResult, SvrError> {
    let mut poses = initial.clone();
    let mut x: Option<Array3<f64>> = None;
    let mut hors_stack: Vec<Option<Array3<f64>>> = vec![None; stacks.len()]; // démarrage à chaud de chaque reconstruction « sans le stack s »
    let (mut rapports, mut iterations) = (Vec::new(), Vec::new());
    for c in 0..=config.cycles {
        let probleme = ReconstructionProblem::with_poses(grid, stacks, &poses, config.alpha)?;
        let depart = x.take().unwrap_or_else(|| probleme.initial_guess());
        let resultat = probleme.conjugate_gradient(&depart, config.cg_max_iterations, config.cg_tolerance);
        iterations.push(resultat.iterations);
        observateur(c, &resultat.x, &poses);
        if c < config.cycles {
            let rapport = if c < config.leave_one_slice_out_cycles {
                register_slices_leave_slice_out(grid, stacks, &poses, config.robust, device)?
            } else if c < config.leave_one_stack_out_cycles {
                recaler_sans_soi_meme(grid, stacks, &poses, config, device, &mut hors_stack)?
            } else {
                let reference = Volume::new(resultat.x.mapv(|v| v as f32), grid.affine)?;
                register_slices_coarse_to_fine(stacks, &poses, &reference, config.coarse_sigma_voxels, config.robust, device)?
            };
            poses = rapport.poses.clone();
            rapports.push(rapport);
        }
        x = Some(resultat.x);
    }
    Ok(LoopResult { poses, volume: x.expect("au moins une reconstruction"), cycles: rapports, cg_iterations: iterations })
}

/// Un recalage de tous les stacks où chaque stack `s` est recalé contre la reconstruction faite avec **les autres** stacks (poses courantes).
/// `departs[s]` mémorise la dernière reconstruction « sans `s` » pour repartir à chaud.
fn recaler_sans_soi_meme(
    grid: &GridSpec,
    stacks: &[Stack],
    poses: &SlicePoses,
    config: LoopConfig,
    device: &Device,
    departs: &mut [Option<Array3<f64>>],
) -> Result<AlignmentReport, SvrError> {
    let mut nouvelles = poses.clone();
    let mut rapports = Vec::new();
    for s in 0..stacks.len() {
        let autres: Vec<usize> = (0..stacks.len()).filter(|&t| t != s).collect();
        let sous_stacks: Vec<Stack> = autres.iter().map(|&t| stacks[t].clone()).collect();
        let sous_poses = poses.select(&autres);
        let probleme = ReconstructionProblem::with_poses(grid, &sous_stacks, &sous_poses, config.alpha)?;
        let depart = departs[s].take().unwrap_or_else(|| probleme.initial_guess());
        let x = probleme.conjugate_gradient(&depart, config.cg_max_iterations, config.cg_tolerance).x;
        let reference = Volume::new(x.mapv(|v| v as f32), grid.affine)?;
        let rapport = register_slices_coarse_to_fine(std::slice::from_ref(&stacks[s]), &poses.select(&[s]), &reference, config.coarse_sigma_voxels, config.robust, device)?;
        for r in rapport.slices {
            nouvelles.set(s, r.slice, *rapport.poses.get(0, r.slice));
            rapports.push(SliceReport { stack: s, ..r });
        }
        departs[s] = Some(x);
    }
    Ok(AlignmentReport { poses: nouvelles, slices: rapports })
}

fn tenseur_1d(v: &[f64], device: &Device) -> burn::prelude::Tensor<1> {
    burn::prelude::Tensor::<1>::from_floats(v.iter().map(|&x| x as f32).collect::<Vec<_>>().as_slice(), device)
}

fn vecteurs_vers_tenseur(points: &[Vector3<f64>], device: &Device) -> burn::prelude::Tensor<2> {
    let v: Vec<f32> = points.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
    burn::prelude::Tensor::<2>::from_floats(burn::prelude::TensorData::new(v, [points.len(), 3]), device)
}
