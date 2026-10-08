//! Recalage de **toutes les coupes** contre un volume de référence (boucle recalage / reconstruction) : chaque coupe est recalée depuis sa pose
//! courante, le delta estimé est composé à gauche (`D · M`) et un rapport par coupe est rendu.

use burn::prelude::Device;
use nalgebra::{Matrix4, Vector3};
use ndarray::Array3;

use crate::diff::{pose_to_matrix, register_slice_robust, rotation_scale_mm, RobustConfig, VolumeTensors};
use crate::recon::{GridSpec, ReconstructionProblem, SlicePoses};
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
    if !poses.matches(stacks) {
        return Err(SvrError::PoseMismatch);
    }
    for stack in stacks {
        stack.brain_mask().ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?;
    }
    let mut nouvelles = poses.clone();
    let mut rapports = Vec::new();
    for (si, stack) in stacks.iter().enumerate() {
        let masque = stack.brain_mask().expect("vérifié").voxels();
        for coupe in stack.slices() {
            let k = coupe.index();
            let courante = *poses.get(si, k);
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
            let pivot = posee.brain_pivot();
            let echelle = pivot.and_then(|c| rotation_scale_mm(&points, &c));
            let (Some(pivot), Some(echelle), true) = (pivot, echelle, points.len() >= MIN_MASK_PIXELS) else {
                rapports.push(SliceReport { stack: si, slice: k, registered: false, ncc_first: 0.0, ncc_final: 0.0, runs: 0, correction_rms_mm: 0.0 });
                continue;
            };
            let resultat = register_slice_robust(
                reference,
                vecteurs_vers_tenseur(&points, device),
                tenseur_1d(&[pivot.x, pivot.y, pivot.z], device),
                echelle,
                tenseur_1d(&intensites, device),
                tenseur_1d(&vec![1.0; points.len()], device),
                config,
            );
            let delta: Matrix4<f64> = pose_to_matrix(&resultat.params, &pivot, echelle);
            let correction = (points.iter().map(|x| ((delta * x.push(1.0)).xyz() - x).norm_squared()).sum::<f64>() / points.len() as f64).sqrt();
            nouvelles.set(si, k, delta * courante);
            rapports.push(SliceReport { stack: si, slice: k, registered: true, ncc_first: resultat.first_ncc, ncc_final: resultat.ncc, runs: resultat.runs, correction_rms_mm: correction });
        }
    }
    Ok(AlignmentReport { poses: nouvelles, slices: rapports })
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
            let rapport = if c < config.leave_one_stack_out_cycles {
                recaler_sans_soi_meme(grid, stacks, &poses, config, device, &mut hors_stack)?
            } else {
                let reference = Volume::new(resultat.x.mapv(|v| v as f32), grid.affine)?;
                register_slices(stacks, &poses, &VolumeTensors::new(&reference, device), config.robust, device)?
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
        let rapport = register_slices(std::slice::from_ref(&stacks[s]), &poses.select(&[s]), &VolumeTensors::new(&reference, device), config.robust, device)?;
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
