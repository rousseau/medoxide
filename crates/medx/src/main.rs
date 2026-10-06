//! `medx` : point d'entrée CLI unique de medoxide.
//!
//! Chaque module (mask, recon, register, ...) devient une sous-commande,
//! sur le modèle `git <sous-commande>` / `cargo <sous-commande>`.
//! Aujourd'hui une seule sous-commande existe : `fetalbet`, nommée d'après
//! l'algorithme (Fetal-BET, extraction cérébrale en IRM fœtale).

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
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
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
