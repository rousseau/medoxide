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

impl From<DeviceArg> for medoxide_bounti::segmentation::Calcul {
    fn from(d: DeviceArg) -> Self {
        match d {
            DeviceArg::Wgpu => Self::Gpu,
            DeviceArg::Cpu => Self::Cpu,
        }
    }
}

impl From<DeviceArg> for DeviceKind {
    fn from(d: DeviceArg) -> DeviceKind {
        match d {
            DeviceArg::Wgpu => DeviceKind::Wgpu,
            DeviceArg::Cpu => DeviceKind::Cpu,
        }
    }
}

// `Recon` porte beaucoup d'options et `Info` très peu : l'énumération est construite une seule fois par appel du programme, boxer une variante n'apporterait rien.
#[allow(clippy::large_enum_variant)]
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
        /// Écrit aussi un indicateur de qualité par coupe (texte : stack, coupe, recalée, score avant, score après, nombre de recalages, correction) : le score est la concordance aux intersections avec `s2s`
        #[arg(long)]
        quality: Option<PathBuf>,
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
        /// Avec `--motion s2s` : passe-haut des profils d'intensité (écart type en mm du lissage soustrait) pour ignorer le champ de biais et le gain entre stacks (0 : aucun)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_highpass_mm)]
        s2s_highpass_mm: f64,
        /// Avec `--motion s2s` : alignement par blocs avant les coupes (niveaux hiérarchiques : le stack entier, puis 2, 4… blocs de coupes consécutives par stack ; 0 : aucun)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_block_levels)]
        s2s_block_levels: usize,
        /// Avec `--s2s-block-levels` : balayages par niveau
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_block_sweeps)]
        s2s_block_sweeps: usize,
        /// Avec `--s2s-block-levels` : demi-étendue en mm de la recherche exhaustive de chaque bloc (0 : aucune recherche)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_block_search_mm)]
        s2s_block_search_mm: f64,
        /// Avec `--motion s2s` : alternances « estimer le champ de biais de chaque stack, corriger, recaler » (0 : aucune correction de biais)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_bias_rounds)]
        s2s_bias_rounds: usize,
        /// Avec `--motion s2s` : poids λ (1/mm²) de la régularisation temporelle : pénalité sur l'écart entre le mouvement d'une coupe et celui de ses voisines directes du stack (0 : aucune)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_temporal_weight)]
        s2s_temporal_weight: f64,
        /// Avec `--motion s2s` : échelle c (mm) de la pénalité robuste de la régularisation temporelle (0 : quadratique)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_temporal_robust_mm)]
        s2s_temporal_robust_mm: f64,
        /// Avec `--motion s2s` : échelle robuste de départ (mm) ; > 0 : l'échelle décroît de cette valeur à `--s2s-temporal-robust-mm` au fil des balayages (0 : constante)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_temporal_robust_start_mm)]
        s2s_temporal_robust_start_mm: f64,
        /// Avec `--motion s2s` : exposant de la pondération du poids temporel par l'information de la coupe (0 : poids constant)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_temporal_info_exponent)]
        s2s_temporal_info_exponent: f64,
        /// Avec `--motion s2s` : cycles coupe-volume de raffinement après le coupe-coupe (0 : aucun)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2v_refine_cycles)]
        s2v_refine_cycles: usize,
        /// Avec `--motion s2s` : fraction des coupes recalées de plus faible concordance écartées de la reconstruction (0 : aucune)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_exclude_fraction)]
        s2s_exclude_fraction: f64,
        /// Avec `--s2v-refine-cycles` : résolution (mm) de la reconstruction des cycles de raffinement
        #[arg(long, default_value_t = ReconstructionConfig::default().s2v_refine_resolution_mm)]
        s2v_refine_resolution_mm: f64,
        /// Avec `--s2s-bias-rounds` : réduction relative minimale de l'incohérence d'intensité entre stacks pour appliquer la correction estimée (0 : toujours)
        #[arg(long, default_value_t = ReconstructionConfig::default().s2s_bias_min_reduction)]
        s2s_bias_min_reduction: f64,
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
    /// Segmentation du cerveau fœtal en 19 tissus (portage de BOUNTI) d'une reconstruction 3D T2
    Bounti {
        /// Reconstruction 3D (NIfTI)
        #[arg(long)]
        input: PathBuf,
        /// Étiquettes de sortie (NIfTI) ; la table `.tsv` est écrite à côté. Hors BIDS
        #[arg(long, conflicts_with = "bids", required_unless_present = "bids")]
        output: Option<PathBuf>,
        /// Masque du cerveau (NIfTI), en plus des étiquettes. Hors BIDS
        #[arg(long, conflicts_with = "bids")]
        brain_mask: Option<PathBuf>,
        /// Racine d'un jeu de données BIDS : écrit dans `<sortie>/derivatives/medx-bounti/sub-X/ses-Y/anat/`
        /// (`_desc-bounti_dseg.nii.gz`, `_dseg.tsv`, `_desc-bounti_mask.nii.gz`)
        #[arg(long, requires = "participant")]
        bids: Option<PathBuf>,
        /// Identifiant du participant, sans préfixe `sub-` (avec --bids)
        #[arg(long, conflicts_with = "output")]
        participant: Option<String>,
        /// Identifiant de la session, sans préfixe `ses-` (avec --bids)
        #[arg(long, conflicts_with = "output")]
        session: Option<String>,
        /// Dossier racine de sortie à la place de la racine BIDS (avec --bids)
        #[arg(long, conflicts_with = "output")]
        output_dir: Option<PathBuf>,
        /// Dossier des poids convertis. Priorité : cette option, puis la variable d'environnement
        /// MEDOXIDE_BOUNTI_WEIGHTS, puis le cache (~/.cache/medoxide/bounti). Poids sur https://huggingface.co/rousseau/medoxide-bounti (GPL v3)
        #[arg(long)]
        weights: Option<PathBuf>,
        /// GPU (wgpu) ou CPU (beaucoup plus lent)
        #[arg(long, value_enum, default_value_t = DeviceArg::Wgpu)]
        device: DeviceArg,
    },
    /// Segmentation du cerveau fœtal en 43 étiquettes (portage de Multi-BOUNTI) d'une reconstruction 3D T2 à 0,5 mm
    Multibounti {
        /// Reconstruction 3D à 0,5 mm (NIfTI)
        #[arg(long)]
        input: PathBuf,
        /// Étiquettes de sortie (NIfTI)
        #[arg(long)]
        output: PathBuf,
        /// Dossier des poids convertis (`bet/`, `lr/`, `multi/`, `gabarit/` : voir docs/MULTIBOUNTI.md ; licence GPL v3, non inclus dans le dépôt)
        #[arg(long, env = "MEDOXIDE_MULTIBOUNTI_WEIGHTS")]
        weights: PathBuf,
        /// GPU (wgpu) ou CPU (beaucoup plus lent)
        #[arg(long, value_enum, default_value_t = DeviceArg::Wgpu)]
        device: DeviceArg,
    },
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
fn svr_recon(
    entrees: &[PathBuf],
    masques: &[PathBuf],
    sortie: &std::path::Path,
    poses: Option<&std::path::Path>,
    qualite: Option<&std::path::Path>,
    device: Option<DeviceArg>,
    config: &ReconstructionConfig,
) -> ExitCode {
    if masques.len() != entrees.len() {
        eprintln!(
            "medx svr recon : {} masque(s) pour {} stack(s) : il en faut un par stack",
            masques.len(),
            entrees.len()
        );
        return ExitCode::FAILURE;
    }
    let debut = std::time::Instant::now();
    let resultat = (|| -> Result<(), medoxide_svr::SvrError> {
        let mut stacks = medoxide_svr::read_stacks(entrees)?;
        for (s, m) in stacks.iter_mut().zip(masques) {
            s.set_brain_mask(m)?;
        }
        let n_coupes: usize = stacks.iter().map(|s| s.dim().2).sum();
        let kind = device.map_or_else(
            || medoxide_svr::pipeline::default_device_kind(config.motion),
            DeviceKind::from,
        );
        // sur GPU, le recalage coupe-volume se fait par lots (bien plus rapide) ; sur CPU, le lot n'apporte rien
        let config = &ReconstructionConfig {
            s2v_batched: kind == DeviceKind::Wgpu,
            ..config.clone()
        };
        eprintln!(
            "{} stacks, {n_coupes} coupes ; mouvement {:?}, calculs {}",
            stacks.len(),
            config.motion,
            match kind {
                DeviceKind::Wgpu => "GPU (wgpu)",
                DeviceKind::Cpu => "CPU (flex)",
            }
        );
        let sortie_reconstruction =
            medoxide_svr::pipeline::reconstruct(&stacks, config, &medoxide_svr::pipeline::autodiff_device(kind))?;
        sortie_reconstruction.write_volume(sortie)?;
        if let Some(p) = poses {
            sortie_reconstruction.write_poses(&stacks, p)?;
        }
        if let Some(q) = qualite {
            sortie_reconstruction.write_quality(q)?;
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
        eprintln!(
            "medx svr info : {} masque(s) pour {} stack(s) : il en faut un par stack",
            masques.len(),
            chemins.len()
        );
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
        println!(
            "  normale      ({:+.3}, {:+.3}, {:+.3})  sens des coupes croissantes",
            normale.x, normale.y, normale.z
        );
        println!(
            "  repère       {}",
            if gauche {
                "main gauche (déterminant < 0)"
            } else {
                "main droite (déterminant > 0)"
            }
        );
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
                println!(
                    "\ngroupes de stacks (écart entre barycentres de masques <= {ecart_mm} mm, de proche en proche) :"
                );
                for (n, groupe) in g.groups.iter().enumerate() {
                    let noms: Vec<String> = groupe.iter().map(|i| (i + 1).to_string()).collect();
                    println!("  groupe {} : stacks {}", n + 1, noms.join(", "));
                }
                println!("distances entre barycentres (mm), stacks numérotés comme ci-dessus :");
                for ligne in &g.distances {
                    println!(
                        "  {}",
                        ligne.iter().map(|d| format!("{d:7.1}")).collect::<Vec<_>>().join(" ")
                    );
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

/// Segmente une reconstruction en 19 tissus et écrit les résultats : les chemins de sortie et les poids sont vérifiés avant tout calcul.
fn bounti(
    entree: &std::path::Path,
    fichiers: Result<medoxide_bounti::sortie::Fichiers, medoxide_bounti::sortie::ErreurSortie>,
    poids: Option<&std::path::Path>,
    calcul: medoxide_bounti::segmentation::Calcul,
) -> ExitCode {
    use medoxide_bounti::{geometrie::Volume, poids::dossier_des_poids, segmentation, sortie};
    let fichiers = match fichiers {
        Ok(f) => f,
        Err(e) => {
            eprintln!("medx bounti : {e}");
            return ExitCode::FAILURE;
        }
    };
    let dossier = match dossier_des_poids(poids) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("medx bounti : {e}");
            return ExitCode::FAILURE;
        }
    };
    let volume = match Volume::lire(entree) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("medx bounti : {} : {e}", entree.display());
            return ExitCode::FAILURE;
        }
    };
    let appareil = calcul.appareil();
    let reseaux = segmentation::Reseaux::charger(&dossier, &appareil);
    let resultat = match segmentation::segmenter(&volume, &reseaux, &appareil) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("medx bounti : {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = sortie::ecrire(&fichiers, &resultat, &volume.affine) {
        eprintln!("medx bounti : {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "Étiquettes écrites dans {:?}, table dans {:?}",
        fichiers.etiquettes, fichiers.table
    );
    if let Some(masque) = &fichiers.masque {
        println!("Masque du cerveau écrit dans {masque:?}");
    }
    ExitCode::SUCCESS
}

fn multibounti(
    entree: &std::path::Path,
    sortie: &std::path::Path,
    poids: &std::path::Path,
    calcul: medoxide_bounti::segmentation::Calcul,
) -> ExitCode {
    use medoxide_bounti::{geometrie::Volume, segmentation_multi, sortie as ecriture};
    let manquants = segmentation_multi::fichiers_manquants(poids);
    if !manquants.is_empty() {
        eprintln!(
            "medx multibounti : poids manquants dans {} : {}",
            poids.display(),
            manquants.join(", ")
        );
        return ExitCode::FAILURE;
    }
    let volume = match Volume::lire(entree) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("medx multibounti : {} : {e}", entree.display());
            return ExitCode::FAILURE;
        }
    };
    let appareil = calcul.appareil();
    let reseaux = segmentation_multi::ReseauxMulti::charger(poids, &appareil);
    let resultat = match segmentation_multi::segmenter(&volume, &reseaux, &appareil) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("medx multibounti : {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = ecriture::ecrire_volume(sortie, &resultat.etiquettes, &volume.affine) {
        eprintln!("medx multibounti : {e}");
        return ExitCode::FAILURE;
    }
    println!("Étiquettes écrites dans {sortie:?}");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Command::Bounti {
            input,
            output,
            brain_mask,
            bids,
            participant,
            session,
            output_dir,
            weights,
            device,
        } => {
            use medoxide_bounti::sortie;
            let fichiers = match (&bids, &output) {
                (Some(racine), _) => sortie::fichiers_bids(
                    output_dir.as_deref().unwrap_or(racine),
                    participant.as_deref().unwrap_or_default(),
                    session.as_deref(),
                ),
                (None, Some(sortie_etiquettes)) => {
                    Ok(sortie::fichiers_libres(sortie_etiquettes, brain_mask.as_deref()))
                }
                // clap l'interdit (`required_unless_present`) ; cas traité sans paniquer
                (None, None) => Err(sortie::ErreurSortie::Identifiant(
                    "--output ou --bids requis".to_string(),
                )),
            };
            bounti(&input, fichiers, weights.as_deref(), device.into())
        }
        Command::Multibounti {
            input,
            output,
            weights,
            device,
        } => multibounti(&input, &output, &weights, device.into()),
        Command::Svr {
            action:
                SvrAction::Info {
                    input,
                    mask,
                    group_gap_mm,
                },
        } => svr_info(&input, &mask, group_gap_mm),
        Command::Svr {
            action:
                SvrAction::Recon {
                    input,
                    mask,
                    output,
                    quality,
                    poses,
                    motion,
                    device,
                    resolution_mm,
                    alpha,
                    min_stack_coverage,
                    edge_kappa,
                    s2s_sweeps,
                    s2s_coarse_sigma,
                    s2s_coarse_sweeps,
                    s2s_coarse_levels,
                    s2s_highpass_mm,
                    s2s_block_levels,
                    s2s_block_sweeps,
                    s2s_block_search_mm,
                    s2s_bias_rounds,
                    s2s_bias_min_reduction,
                    s2s_temporal_weight,
                    s2s_temporal_robust_mm,
                    s2s_temporal_robust_start_mm,
                    s2s_temporal_info_exponent,
                    s2v_refine_cycles,
                    s2v_refine_resolution_mm,
                    s2s_exclude_fraction,
                    s2v_cycles,
                },
        } => {
            let config = ReconstructionConfig {
                motion: motion.into(),
                resolution_mm,
                alpha,
                min_stack_coverage,
                edge_kappa,
                s2s_sweeps,
                s2s_coarse_sigma_pixels: s2s_coarse_sigma,
                s2s_coarse_sweeps,
                s2s_coarse_levels,
                s2s_highpass_mm,
                s2s_block_levels,
                s2s_block_sweeps,
                s2s_block_search_mm,
                s2s_bias_rounds,
                s2s_bias_min_reduction,
                s2s_temporal_weight,
                s2s_temporal_robust_mm,
                s2s_temporal_robust_start_mm,
                s2s_temporal_info_exponent,
                s2v_refine_cycles,
                s2v_refine_resolution_mm,
                s2s_exclude_fraction,
                s2v_cycles,
                ..Default::default()
            };
            svr_recon(
                &input,
                &mask,
                &output,
                poses.as_deref(),
                quality.as_deref(),
                device,
                &config,
            )
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
    use clap::{CommandFactory, Parser};

    use super::Cli;

    /// Les définitions d'arguments sont cohérentes (pas de doublon, valeurs par défaut valides) : clap le vérifie pour toutes les sous-commandes.
    #[test]
    fn the_command_line_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    /// Arguments de `medx bounti` : hors BIDS `--output` est requis ; en BIDS `--participant` est requis ; `--output` et `--brain-mask` sont exclusifs de `--bids` ; `--participant`, `--session` et `--output-dir` demandent `--bids`.
    #[test]
    fn the_bounti_arguments_are_exclusive_or_required_as_documented() {
        let essai = |args: &[&str]| Cli::try_parse_from(std::iter::once("medx").chain(args.iter().copied())).is_ok();
        assert!(essai(&["bounti", "--input", "a", "--output", "b"]));
        assert!(essai(&[
            "bounti",
            "--input",
            "a",
            "--output",
            "b",
            "--brain-mask",
            "m",
            "--weights",
            "w",
            "--device",
            "cpu"
        ]));
        assert!(essai(&["bounti", "--input", "a", "--bids", "r", "--participant", "01"]));
        assert!(essai(&[
            "bounti",
            "--input",
            "a",
            "--bids",
            "r",
            "--participant",
            "01",
            "--session",
            "1",
            "--output-dir",
            "o"
        ]));
        assert!(!essai(&["bounti", "--input", "a"]), "ni --output ni --bids");
        assert!(!essai(&["bounti", "--output", "b"]), "sans --input");
        assert!(
            !essai(&["bounti", "--input", "a", "--bids", "r"]),
            "--bids sans --participant"
        );
        assert!(
            !essai(&[
                "bounti",
                "--input",
                "a",
                "--bids",
                "r",
                "--participant",
                "01",
                "--output",
                "b"
            ]),
            "--output avec --bids"
        );
        assert!(
            !essai(&[
                "bounti",
                "--input",
                "a",
                "--bids",
                "r",
                "--participant",
                "01",
                "--brain-mask",
                "m"
            ]),
            "--brain-mask avec --bids"
        );
        assert!(
            !essai(&["bounti", "--input", "a", "--output", "b", "--participant", "01"]),
            "--participant sans --bids"
        );
        assert!(
            !essai(&["bounti", "--input", "a", "--output", "b", "--session", "1"]),
            "--session sans --bids"
        );
        assert!(
            !essai(&["bounti", "--input", "a", "--output", "b", "--output-dir", "o"]),
            "--output-dir sans --bids"
        );
    }

    /// Arguments de `medx multibounti` : `--input`, `--output` et `--weights` (ou la variable `MEDOXIDE_MULTIBOUNTI_WEIGHTS`, non lue ici) sont requis ; `--device` accepte `cpu`.
    #[test]
    fn the_multibounti_arguments_are_required_as_documented() {
        let essai = |args: &[&str]| Cli::try_parse_from(std::iter::once("medx").chain(args.iter().copied())).is_ok();
        assert!(essai(&[
            "multibounti",
            "--input",
            "a",
            "--output",
            "b",
            "--weights",
            "w"
        ]));
        assert!(essai(&[
            "multibounti",
            "--input",
            "a",
            "--output",
            "b",
            "--weights",
            "w",
            "--device",
            "cpu"
        ]));
        assert!(
            !essai(&["multibounti", "--input", "a", "--output", "b"])
                || std::env::var_os("MEDOXIDE_MULTIBOUNTI_WEIGHTS").is_some()
        );
        assert!(!essai(&["multibounti", "--output", "b", "--weights", "w"]));
        assert!(!essai(&["multibounti", "--input", "a", "--weights", "w"]));
    }
}
