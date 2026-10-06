//! `medx` : point d'entrée CLI unique de medoxide.
//!
//! Chaque module (mask, recon, register, ...) devient une sous-commande,
//! sur le modèle `git <sous-commande>` / `cargo <sous-commande>`.
//! Sous-commandes : `fetalbet`, nommée d'après l'algorithme (Fetal-BET, extraction cérébrale en
//! IRM fœtale), et `svr` (reconstruction coupe-vers-volume, en développement).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "medx", version, about = "Boîte à outils medoxide pour l'imagerie médicale")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum SvrAction {
    /// Affiche la géométrie de stacks NIfTI : dimensions, espacement, normale des coupes, boîte dans le monde
    Info {
        /// Un ou plusieurs stacks NIfTI (`.nii` ou `.nii.gz`)
        #[arg(long, num_args = 1.., required = true)]
        input: Vec<PathBuf>,
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

/// Affiche la géométrie de chaque stack, puis la boîte de l'ensemble (repère monde RAS+, mm).
fn svr_info(chemins: &[PathBuf]) -> ExitCode {
    let stacks = match medoxide_svr::read_stacks(chemins) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("medx svr info : {e}");
            return ExitCode::FAILURE;
        }
    };
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
        Command::Svr { action: SvrAction::Info { input } } => svr_info(&input),
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
