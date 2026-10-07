//! Recalage de **toutes les coupes** contre un volume de référence (boucle recalage / reconstruction) : chaque coupe est recalée depuis sa pose
//! courante, le delta estimé est composé à gauche (`D · M`) et un rapport par coupe est rendu.

use burn::prelude::Device;
use nalgebra::{Matrix4, Vector3};

use crate::diff::{pose_to_matrix, register_slice_robust, rotation_scale_mm, RobustConfig, VolumeTensors};
use crate::recon::SlicePoses;
use crate::{Stack, SvrError};

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

fn tenseur_1d(v: &[f64], device: &Device) -> burn::prelude::Tensor<1> {
    burn::prelude::Tensor::<1>::from_floats(v.iter().map(|&x| x as f32).collect::<Vec<_>>().as_slice(), device)
}

fn vecteurs_vers_tenseur(points: &[Vector3<f64>], device: &Device) -> burn::prelude::Tensor<2> {
    let v: Vec<f32> = points.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
    burn::prelude::Tensor::<2>::from_floats(burn::prelude::TensorData::new(v, [points.len(), 3]), device)
}
