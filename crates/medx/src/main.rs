//! `medx` : point d'entrée CLI unique de medoxide.
//!
//! Chaque module (mask, recon, register, ...) devient une sous-commande,
//! sur le modèle `git <sous-commande>` / `cargo <sous-commande>`.
//! Aujourd'hui une seule sous-commande existe : `mask`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "medx", version, about = "Boîte à outils medoxide pour l'imagerie médicale")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Extraction du masque cérébral fœtal (portage de Fetal-BET)
    Mask {
        /// Volume NIfTI d'entrée
        #[arg(long)]
        input: PathBuf,
        /// Masque NIfTI de sortie
        #[arg(long)]
        output: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Mask { input, output } => match medoxide_mask::segment(&input, &output) {
            Ok(()) => println!("Masque écrit dans {output:?}"),
            Err(e) => eprintln!("medx mask : {e}"),
        },
    }
}
