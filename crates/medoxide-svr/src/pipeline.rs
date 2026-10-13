//! Chaîne complète de reconstruction : de stacks avec masques à un volume et des poses par coupe.
//!
//! Ce module rassemble les réglages (jusqu'ici dispersés dans des variables d'environnement
//! `MEDOXIDE_*` lues par des tests) en une seule structure, [`ReconstructionConfig`], que la ligne de
//! commande `medx svr` remplit.

use std::path::Path;

use burn::prelude::Device;
use nifti::{writer::WriterOptions, NiftiHeader};
use ndarray::Array3;

use crate::align::{reconstruct_with_motion_correction, LoopConfig};
use crate::diff::RobustConfig;
use crate::recon::{reconstruction_grid, GridSpec, ReconstructionProblem, SlicePoses};
use crate::s2s::{register_slices_s2s, S2sConfig};
use crate::{Stack, SvrError};

/// Où se font les calculs de tenseurs (recalage et reconstruction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// GPU, par `wgpu` (Metal sur Mac) ; à demander explicitement, il est aujourd'hui plus lent (voir [`default_device_kind`]).
    Wgpu,
    /// CPU de la machine, par le backend `flex` de Burn (Rust pur, déterministe).
    Cpu,
}

/// Device par défaut d'un algorithme de mouvement : le CPU de la machine pour les deux, le choix étant fait **par algorithme** (le `match` oblige à décider pour tout nouvel algorithme).
///
/// Mesures (étude 06, §41) : le recalage coupe par coupe lance un grand nombre de petits calculs, que le GPU (`wgpu`) traite plus lentement que le CPU (coupe-coupe : 118 s contre 18 s sur des
/// stacks simulés, 556 s contre 150 s sur un sujet réel, avec des poses de même qualité). Coupe-volume, sujet réel à trois stacks : 151 s sur CPU ; sur GPU 349 s avec le recalage par lots (`s2v_batched`), après avoir dépassé 90 min coupe par coupe : le CPU reste le plus rapide.
pub fn default_device_kind(motion: Motion) -> DeviceKind {
    match motion {
        Motion::SliceToSlice => DeviceKind::Cpu,
        Motion::SliceToVolume => DeviceKind::Cpu,
    }
}

/// Le device de calcul demandé, avec la différentiation automatique activée (nécessaire au recalage).
pub fn autodiff_device(kind: DeviceKind) -> Device {
    match kind {
        DeviceKind::Wgpu => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice).autodiff(),
        DeviceKind::Cpu => Device::flex().autodiff(),
    }
}

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
    /// Coupe-volume : recaler toutes les coupes d'un stack par lots (voir [`crate::diff::register_slices_robust_batch`]). Même calcul ; bien plus rapide sur GPU, sans gain sur CPU.
    pub s2v_batched: bool,
    /// Coupe-volume : ne pas reconstruire aux cycles intermédiaires (avec `s2v_leave_slice_out`, ces volumes ne servent pas au recalage) ; seule la reconstruction finale est faite. Sans effet sur les poses.
    pub s2v_skip_unused_reconstructions: bool,
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
            s2v_batched: false,
            s2v_skip_unused_reconstructions: true,
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

impl ReconstructionOutput {
    /// Écrit le volume au format NIfTI (`float32`), dans le repère monde de la grille (sform) ; le fichier est compressé si `path` finit par `.gz`.
    ///
    /// # Erreurs
    /// [`SvrError::Write`] si le fichier ne peut pas être écrit (dossier absent, droits).
    pub fn write_volume(&self, path: &Path) -> Result<(), SvrError> {
        let mut header = NiftiHeader::default();
        header.sform_code = 1;
        header.srow_x = std::array::from_fn(|c| self.grid.affine[(0, c)] as f32);
        header.srow_y = std::array::from_fn(|c| self.grid.affine[(1, c)] as f32);
        header.srow_z = std::array::from_fn(|c| self.grid.affine[(2, c)] as f32);
        let r = self.grid.resolution_mm as f32;
        header.pixdim = [1.0, r, r, r, 1.0, 1.0, 1.0, 1.0];
        WriterOptions::new(path)
            .reference_header(&header)
            .write_nifti(&self.volume.mapv(|v| v as f32))
            .map_err(|e| SvrError::Write { path: path.to_path_buf(), message: e.to_string() })
    }

    /// Écrit le mouvement de chaque coupe : un fichier texte, une ligne par coupe, `stack \t coupe \t` puis les 16 valeurs de la matrice 4 × 4 **en ordre colonne**
    /// (lisible en Python par `reshape(4, 4, order="F")`). `stacks` donne les stacks de `poses`, dans le même ordre ; l'indice de stack écrit est celui de cette liste.
    /// Les nombres sont écrits avec leur représentation décimale la plus courte qui se relit exactement.
    ///
    /// # Erreurs
    /// [`SvrError::PoseMismatch`] si `stacks` n'a pas la forme des poses ; [`SvrError::Write`] si le fichier ne peut pas être écrit.
    pub fn write_poses(&self, stacks: &[Stack], path: &Path) -> Result<(), SvrError> {
        if !self.poses.matches(stacks) {
            return Err(SvrError::PoseMismatch);
        }
        let mut lines = Vec::new();
        for (s, stack) in stacks.iter().enumerate() {
            for k in 0..stack.dim().2 {
                let values: Vec<String> = self.poses.get(s, k).iter().map(|v| format!("{v:?}")).collect();
                lines.push(format!("{s}\t{k}\t{}", values.join("\t")));
            }
        }
        std::fs::write(path, lines.join("\n")).map_err(|e| SvrError::Write { path: path.to_path_buf(), message: e.to_string() })
    }
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
        skip_unused_reconstructions: config.s2v_skip_unused_reconstructions && config.s2v_leave_slice_out,
        cycles,
        alpha: config.alpha,
        cg_max_iterations: config.cg_max_iterations,
        cg_tolerance: config.cg_tolerance,
        robust: RobustConfig { extra_starts: config.s2v_extra_starts, start_amplitude_mm: config.s2v_start_amplitude_mm, batched: config.s2v_batched, ..RobustConfig::new(config.s2v_suspect_ncc) },
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
