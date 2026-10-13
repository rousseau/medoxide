//! `medx` : point d'entrée CLI unique de medoxide.
//!
//! Chaque module (mask, recon, register, ...) devient une sous-commande,
//! sur le modèle `git <sous-commande>` / `cargo <sous-commande>`.
//! Sous-commandes : `fetalbet`, nommée d'après l'algorithme (Fetal-BET, extraction cérébrale en
//! IRM fœtale), et `svr` (reconstruction coupe-vers-volume, en développement).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use medoxide_svr::pipeline::{DeviceKind, Motion, ReconstructionConfig};

#[derive(Parser)]
#[command(name = "medx", version, about = "Boîte à outils medoxide pour l'imagerie médicale")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Estimation du mouvement, telle qu'écrite sur la ligne de commande (`--motion`).
#[derive(Clone, Copy, ValueEnum)]
enum MotionArg {
    /// Coupe-coupe : sans volume intermédiaire, rapide
    S2s,
    /// Coupe-volume : coupes recalées contre un volume reconstruit, en boucle, lent
    S2v,
}

impl From<MotionArg> for Motion {
    fn from(m: MotionArg) -> Motion {
        match m {
            MotionArg::S2s => Motion::SliceToSlice,
            MotionArg::S2v => Motion::SliceToVolume,
        }
    }
}

/// Où se font les calculs, tel qu'écrit sur la ligne de commande (`--device`).
#[derive(Clone, Copy, ValueEnum)]
enum DeviceArg {
    /// GPU (wgpu : Metal sur Mac), plus lent aujourd'hui
    Wgpu,
    /// CPU de la machine (backend flex, déterministe)
    Cpu,
}

impl From<DeviceArg> for DeviceKind {
    fn from(d: DeviceArg) -> DeviceKind {
        match d {
            DeviceArg::Wgpu => DeviceKind::Wgpu,
            DeviceArg::Cpu => DeviceKind::Cpu,
        }
    }
}

#[derive(Subcommand)]
enum SvrAction {
    /// Reconstruit un volume 3D à partir de stacks 2D (IRM fœtale) : estime le mouvement de chaque coupe, puis reconstruit
    Recon {
        /// Deux stacks NIfTI au moins (`.nii` ou `.nii.gz`), de plans différents
        #[arg(long, num_args = 2.., required = true)]
        input: Vec<PathBuf>,
        /// Masques cérébraux (par exemple produits par `medx fetalbet`), un par stack et dans le même ordre
        #[arg(long, num_args = 2.., required = true)]
        mask: Vec<PathBuf>,
        /// Volume NIfTI reconstruit (compressé si le nom finit par `.gz`)
        #[arg(long)]
        output: PathBuf,
        /// Écrit aussi le mouvement estimé de chaque coupe (texte : stack, coupe, matrice 4 x 4 en ordre colonne)
        #[arg(long)]
        poses: Option<PathBuf>,
        /// Estimation du mouvement
        #[arg(long, value_enum, default_value_t = MotionArg::S2s)]
        motion: MotionArg,
        /// Où se font les calculs. Par défaut : le CPU de la machine, choix fait par algorithme (le GPU est aujourd'hui plus lent)
        #[arg(long, value_enum)]
        device: Option<DeviceArg>,
        /// Côté des voxels du volume reconstruit, en mm
        #[arg(long, default_value_t = ReconstructionConfig::default().resolution_mm)]
        resolution_mm: f64,
        /// Poids de la régularisation (lissage)
        #[arg(long, default_value_t = ReconstructionConfig::default().alpha)]
        alpha: f64,
        /// Nombre minimal de stacks qui doivent atteindre un voxel pour qu'il soit reconstruit (1 : tous les voxels touchés ; 2 : supprime les « languettes » vues par un seul stack)
        #[arg(long, default_value_t = ReconstructionConfig::default().min_stack_coverage)]
        min_stack_coverage: usize,
        /// Reconstruction préservant les contours (Huber) : seuil relatif κ (0 : lissage de Tikhonov seul) ; un contour est une différence entre voxels voisins supérieure à κ fois la différence médiane
        #[arg(long, default_value_t = ReconstructionConfig::default().edge_kappa)]
        edge_kappa: f64,
        /// Avec `--motion s2s` : nombre de balayages de toutes les coupes
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_sweeps)]
        s2s_sweeps: usize,
        /// Avec `--motion s2s` : écart type (en pixels) du flou des coupes pour des balayages préliminaires du gros vers le fin, qui élargissent la portée de capture (0 : aucun)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_coarse_sigma_pixels)]
        s2s_coarse_sigma: f64,
        /// Avec `--s2s-coarse-sigma` : nombre de balayages sur les coupes floutées
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_coarse_sweeps)]
        s2s_coarse_sweeps: usize,
        /// Avec `--s2s-coarse-sigma` : nombre de niveaux grossiers (le flou est divisé par 2 à chaque niveau)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_coarse_levels)]
        s2s_coarse_levels: usize,
        /// Avec `--motion s2v` : nombre de cycles « recaler puis reconstruire »
        #[arg(long, default_value_t = ReconstructionConfig::default().s2v_cycles)]
        s2v_cycles: usize,
    },
    /// Affiche la géométrie de stacks NIfTI : dimensions, espacement, normale des coupes, boîte dans le monde
    Info {
        /// Un ou plusieurs stacks NIfTI (`.nii` ou `.nii.gz`)
        #[arg(long, num_args = 1.., required = true)]
        input: Vec<PathBuf>,
        /// Masques cérébraux, un par stack et dans le même ordre : permettent de repérer les stacks qui
        /// ne partagent pas un repère (groupes) d'après les barycentres des masques
        #[arg(long, num_args = 1..)]
        mask: Vec<PathBuf>,
        /// Écart maximal entre barycentres de masques, en mm, pour que deux stacks soient dans le même groupe
        #[arg(long, default_value_t = medoxide_svr::DEFAULT_GROUP_GAP_MM)]
        group_gap_mm: f64,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Extraction du masque cérébral en IRM fœtale (portage de Fetal-BET)
    Fetalbet {
        /// Volume NIfTI d'entrée
        #[arg(long)]
        input: PathBuf,
        /// Masque NIfTI de sortie
        #[arg(long)]
        output: PathBuf,
        /// Poids du modèle (fichier .bpk). Priorité : cette option, puis la
        /// variable d'environnement MEDOXIDE_MODEL, puis le cache
        /// (~/.cache/medoxide/attunet.bpk), où ils sont téléchargés depuis Hugging
        /// Face au premier lancement
        #[arg(long, env = "MEDOXIDE_MODEL")]
        model: Option<PathBuf>,
    },
    /// Reconstruction coupe-vers-volume en IRM fœtale (en développement)
    Svr {
        #[command(subcommand)]
        action: SvrAction,
    },
}

/// Reconstruit un volume depuis des stacks : lit les stacks et leurs masques, estime le mouvement, reconstruit, écrit le volume (et les poses si demandé).
fn svr_recon(entrees: &[PathBuf], masques: &[PathBuf], sortie: &std::path::Path, poses: Option<&std::path::Path>, device: Option<DeviceArg>, config: &ReconstructionConfig) -> ExitCode {
    if masques.len() != entrees.len() {
        eprintln!("medx svr recon : {} masque(s) pour {} stack(s) : il en faut un par stack", masques.len(), entrees.len());
        return ExitCode::FAILURE;
    }
    let debut = std::time::Instant::now();
    let resultat = (|| -> Result<(), medoxide_svr::SvrError> {
        let mut stacks = medoxide_svr::read_stacks(entrees)?;
        for (s, m) in stacks.iter_mut().zip(masques) {
            s.set_brain_mask(m)?;
        }
        let n_coupes: usize = stacks.iter().map(|s| s.dim().2).sum();
        let kind = device.map_or_else(|| medoxide_svr::pipeline::default_device_kind(config.motion), DeviceKind::from);
        // sur GPU, le recalage coupe-volume se fait par lots (bien plus rapide) ; sur CPU, le lot n'apporte rien
        let config = &ReconstructionConfig { s2v_batched: kind == DeviceKind::Wgpu, ..config.clone() };
        eprintln!("{} stacks, {n_coupes} coupes ; mouvement {:?}, calculs {}", stacks.len(), config.motion, match kind { DeviceKind::Wgpu => "GPU (wgpu)", DeviceKind::Cpu => "CPU (flex)" });
        let sortie_reconstruction = medoxide_svr::pipeline::reconstruct(&stacks, config, &medoxide_svr::pipeline::autodiff_device(kind))?;
        sortie_reconstruction.write_volume(sortie)?;
        if let Some(p) = poses {
            sortie_reconstruction.write_poses(&stacks, p)?;
        }
        Ok(())
    })();
    match resultat {
        Ok(()) => {
            println!("Volume écrit dans {sortie:?} en {:.0} s", debut.elapsed().as_secs_f64());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("medx svr recon : {e}");
            ExitCode::FAILURE
        }
    }
}

/// Affiche la géométrie de chaque stack, puis la boîte de l'ensemble (repère monde RAS+, mm).
/// Avec des masques cérébraux, repère aussi les groupes de stacks qui partagent un repère.
fn svr_info(chemins: &[PathBuf], masques: &[PathBuf], ecart_mm: f64) -> ExitCode {
    if !masques.is_empty() && masques.len() != chemins.len() {
        eprintln!("medx svr info : {} masque(s) pour {} stack(s) : il en faut un par stack", masques.len(), chemins.len());
        return ExitCode::FAILURE;
    }
    let mut stacks = match medoxide_svr::read_stacks(chemins) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("medx svr info : {e}");
            return ExitCode::FAILURE;
        }
    };
    for (s, m) in stacks.iter_mut().zip(masques) {
        if let Err(e) = s.set_brain_mask(m) {
            eprintln!("medx svr info : {e}");
            return ExitCode::FAILURE;
        }
    }
    let mut ensemble: Option<medoxide_svr::BoundingBox> = None;
    for (n, s) in stacks.iter().enumerate() {
        let (nx, ny, nz) = s.dim();
        let sp = s.spacing();
        let normale = s.slice(0).map(|c| c.normal()).unwrap_or_default();
        let gauche = s.affine().fixed_view::<3, 3>(0, 0).determinant() < 0.0;
        let b = s.world_bounding_box();
        println!("stack {}/{} : {}", n + 1, stacks.len(), s.path().display());
        println!("  dimensions   {nx} x {ny} x {nz} voxels ({nz} coupes)");
        println!("  espacement   {:.3} x {:.3} x {:.3} mm", sp[0], sp[1], sp[2]);
        println!("  normale      ({:+.3}, {:+.3}, {:+.3})  sens des coupes croissantes", normale.x, normale.y, normale.z);
        println!("  repère       {}", if gauche { "main gauche (déterminant < 0)" } else { "main droite (déterminant > 0)" });
        println!("  boîte monde  {}", format_boite(&b));
        ensemble = Some(match ensemble {
            Some(e) => e.union(&b),
            None => b,
        });
    }
    if let Some(e) = ensemble {
        println!("ensemble de {} stack(s) : {}", stacks.len(), format_boite(&e));
    }
    if !masques.is_empty() {
        match medoxide_svr::group_stacks(&stacks, ecart_mm) {
            Ok(g) => {
                println!("\ngroupes de stacks (écart entre barycentres de masques <= {ecart_mm} mm, de proche en proche) :");
                for (n, groupe) in g.groups.iter().enumerate() {
                    let noms: Vec<String> = groupe.iter().map(|i| (i + 1).to_string()).collect();
                    println!("  groupe {} : stacks {}", n + 1, noms.join(", "));
                }
                println!("distances entre barycentres (mm), stacks numérotés comme ci-dessus :");
                for ligne in &g.distances {
                    println!("  {}", ligne.iter().map(|d| format!("{d:7.1}")).collect::<Vec<_>>().join(" "));
                }
            }
            Err(e) => {
                eprintln!("medx svr info : {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

/// Texte d'une boîte : intervalle en x, y, z (mm).
fn format_boite(b: &medoxide_svr::BoundingBox) -> String {
    format!(
        "x [{:.1}, {:.1}]  y [{:.1}, {:.1}]  z [{:.1}, {:.1}] mm",
        b.min.x, b.max.x, b.min.y, b.max.y, b.min.z, b.max.z
    )
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Command::Svr { action: SvrAction::Info { input, mask, group_gap_mm } } => svr_info(&input, &mask, group_gap_mm),
        Command::Svr { action: SvrAction::Recon { input, mask, output, poses, motion, device, resolution_mm, alpha, min_stack_coverage, edge_kappa, s2s_sweeps, s2s_coarse_sigma, s2s_coarse_sweeps, s2s_coarse_levels, s2v_cycles } } => {
            let config = ReconstructionConfig { motion: motion.into(), resolution_mm, alpha, min_stack_coverage, edge_kappa, s2s_sweeps, s2s_coarse_sigma_pixels: s2s_coarse_sigma, s2s_coarse_sweeps, s2s_coarse_levels, s2v_cycles, ..Default::default() };
            svr_recon(&input, &mask, &output, poses.as_deref(), device, &config)
        }
        Command::Fetalbet { input, output, model } => {
            // `model.as_deref()` donne un `Option<&Path>` sans consommer l'`Option` ;
            // `None` : `segment` télécharge les poids par défaut au besoin.
            match medoxide_fetalbet::segment(&input, &output, model.as_deref()) {
                Ok(()) => {
                    println!("Masque écrit dans {output:?}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("medx fetalbet : {e}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::Cli;

    /// Les définitions d'arguments sont cohérentes (pas de doublon, valeurs par défaut valides) : clap le vérifie pour toutes les sous-commandes.
    #[test]
    fn the_command_line_definition_is_consistent() {
        Cli::command().debug_assert();
    }
}
