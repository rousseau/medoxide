//! Chaîne complète de reconstruction : de stacks avec masques à un volume et des poses par coupe.
//!
//! Ce module rassemble les réglages (jusqu'ici dispersés dans des variables d'environnement
//! `MEDOXIDE_*` lues par des tests) en une seule structure, [`ReconstructionConfig`], que la ligne de
//! commande `medx svr` remplit.

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
        }
    }
}
