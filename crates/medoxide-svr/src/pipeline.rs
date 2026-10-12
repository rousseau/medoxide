//! Chaîne complète de reconstruction : de stacks avec masques à un volume et des poses par coupe.
//!
//! Ce module rassemble les réglages (jusqu'ici dispersés dans des variables d'environnement
//! `MEDOXIDE_*` lues par des tests) en une seule structure, [`ReconstructionConfig`], que la ligne de
//! commande `medx svr` remplit.

use burn::prelude::Device;
use ndarray::Array3;

use crate::align::{reconstruct_with_motion_correction, LoopConfig};
use crate::diff::RobustConfig;
use crate::recon::{reconstruction_grid, GridSpec, ReconstructionProblem, SlicePoses};
use crate::s2s::{register_slices_s2s, S2sConfig};
use crate::{Stack, SvrError};

/// Comment le mouvement de chaque coupe est estimé.
///
/// Les deux méthodes produisent les mêmes poses (une transformation rigide par coupe) ; la reconstruction
/// sur grille les utilise ensuite de la même façon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Motion {
    /// Coupe-coupe : chaque coupe est recalée contre les coupes des autres stacks qui la croisent, **sans volume**
    /// ([`crate::s2s`]). Rapide (30 à 90 s pour trois stacks) ; à qualité de reconstruction mesurée comparable.
    SliceToSlice,
    /// Coupe-volume : chaque coupe est recalée contre un volume reconstruit qui ne la contient pas, puis le volume est
    /// reconstruit avec les nouvelles poses, en boucle ([`crate::align::reconstruct_with_motion_correction`]).
    /// Lent (12 à 21 min pour trois stacks).
    SliceToVolume,
}

/// Réglages de la chaîne de reconstruction. [`Default`] donne la configuration retenue après les mesures sur données réelles.
#[derive(Debug, Clone)]
pub struct ReconstructionConfig {
    /// Méthode d'estimation du mouvement.
    pub motion: Motion,
    /// Côté des voxels isotropes de la grille de reconstruction, en mm.
    pub resolution_mm: f64,
    /// Marge ajoutée autour de la boîte des masques cérébraux pour définir la grille, en mm.
    pub margin_mm: f64,
    /// Poids `α` de la régularisation de la reconstruction (voir [`crate::recon::ReconstructionProblem`]).
    pub alpha: f64,
    /// Itérations maximales du gradient conjugué à chaque reconstruction.
    pub cg_max_iterations: usize,
    /// Tolérance relative du gradient conjugué.
    pub cg_tolerance: f64,
    /// Coupe-coupe : nombre de balayages (chaque balayage recale toutes les coupes une fois).
    pub s2s_sweeps: usize,
    /// Coupe-volume : nombre de cycles « recaler puis reconstruire ».
    pub s2v_cycles: usize,
    /// Coupe-volume : nombre de poses de départ supplémentaires pour une coupe mal recalée.
    pub s2v_extra_starts: usize,
    /// Coupe-volume : amplitude (mm) de ces poses de départ.
    pub s2v_start_amplitude_mm: f64,
    /// Coupe-volume : seuil de corrélation finale en dessous duquel une coupe est relancée depuis d'autres poses.
    pub s2v_suspect_ncc: f64,
    /// Coupe-volume : chaque coupe est recalée contre une reconstruction qui **ne la contient pas** (`true`, la meilleure option mesurée sur données réelles),
    /// ou contre une reconstruction faite sans son stack entier (`false`, l'ancienne boucle).
    pub s2v_leave_slice_out: bool,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        ReconstructionConfig {
            motion: Motion::SliceToSlice,
            resolution_mm: 0.8,
            margin_mm: 10.0,
            alpha: 0.01,
            cg_max_iterations: 300,
            cg_tolerance: 1e-4,
            s2s_sweeps: 3,
            s2v_cycles: 6,
            s2v_extra_starts: 6,
            s2v_start_amplitude_mm: 5.0,
            s2v_suspect_ncc: 0.9,
            s2v_leave_slice_out: true,
        }
    }
}

/// Résultat de [`reconstruct`].
#[derive(Debug, Clone)]
pub struct ReconstructionOutput {
    /// Géométrie de la grille du volume.
    pub grid: GridSpec,
    /// Volume reconstruit, sur `grid`.
    pub volume: Array3<f64>,
    /// Mouvement estimé de chaque coupe (composé après l'affine d'en-tête).
    pub poses: SlicePoses,
}

/// **Reconstruit un volume** à partir de stacks (avec masque cérébral) : définit la grille, estime le mouvement de chaque coupe selon
/// [`ReconstructionConfig::motion`], puis reconstruit le volume avec ces poses (gradient conjugué, Tikhonov d'ordre 1).
///
/// `device` doit avoir l'autodiff activé.
///
/// # Erreurs
/// [`SvrError::NotEnoughStacks`] si moins de deux stacks ; [`SvrError::NoMask`] si un stack n'a pas de masque ; celles de
/// [`reconstruction_grid`] et de [`ReconstructionProblem::with_poses`] (réglages invalides).
pub fn reconstruct(stacks: &[Stack], config: &ReconstructionConfig, device: &Device) -> Result<ReconstructionOutput, SvrError> {
    if stacks.len() < 2 {
        return Err(SvrError::NotEnoughStacks);
    }
    let grid = reconstruction_grid(stacks, config.resolution_mm, config.margin_mm)?;
    let header_poses = SlicePoses::identity(stacks);
    match config.motion {
        Motion::SliceToSlice => {
            let s2s = S2sConfig { sweeps: config.s2s_sweeps, ..Default::default() };
            let poses = register_slices_s2s(stacks, &header_poses, &s2s, device)?.poses;
            let volume = reconstruct_with_poses(&grid, stacks, &poses, config)?;
            Ok(ReconstructionOutput { grid, volume, poses })
        }
        Motion::SliceToVolume => {
            let loop_config = loop_config(config);
            let result = reconstruct_with_motion_correction(&grid, stacks, &header_poses, loop_config, device, |_, _, _| {})?;
            Ok(ReconstructionOutput { grid, volume: result.volume, poses: result.poses })
        }
    }
}

/// Réglages de la boucle coupe-volume correspondant à `config` ; les options expérimentales non retenues (flou grossier, rejet des poses aberrantes) sont désactivées.
pub(crate) fn loop_config(config: &ReconstructionConfig) -> LoopConfig {
    let cycles = config.s2v_cycles;
    LoopConfig {
        cycles,
        alpha: config.alpha,
        cg_max_iterations: config.cg_max_iterations,
        cg_tolerance: config.cg_tolerance,
        robust: RobustConfig { extra_starts: config.s2v_extra_starts, start_amplitude_mm: config.s2v_start_amplitude_mm, ..RobustConfig::new(config.s2v_suspect_ncc) },
        leave_one_stack_out_cycles: if config.s2v_leave_slice_out { 0 } else { cycles },
        coarse_sigma_voxels: 0.0,
        leave_one_slice_out_cycles: if config.s2v_leave_slice_out { cycles } else { 0 },
        leave_slice_out_window: 0,
        pose_outlier_window: 0,
        pose_outlier_max_mm: 5.0,
        pose_outlier_max_deg: 10.0,
    }
}

/// Volume reconstruit sur `grid` avec des poses données : gradient conjugué (réglages `alpha`, `cg_max_iterations` et `cg_tolerance` de `config`) depuis l'adjoint normalisé.
pub(crate) fn reconstruct_with_poses(grid: &GridSpec, stacks: &[Stack], poses: &SlicePoses, config: &ReconstructionConfig) -> Result<Array3<f64>, SvrError> {
    let problem = ReconstructionProblem::with_poses(grid, stacks, poses, config.alpha)?;
    Ok(problem.conjugate_gradient(&problem.initial_guess(), config.cg_max_iterations, config.cg_tolerance).x)
}
