//! Retient le commit du code compilé (variable `MEDX_COMMIT`, lue par `option_env!`), pour le déclarer dans les `dataset_description.json` écrits par `medx`.
//! Sans `git` ou hors d'un dépôt (archive des sources), la variable n'est pas définie et le commit est omis.

use std::path::PathBuf;
use std::process::Command;

/// Sortie de `git <arguments>` (texte sans espaces de bordure), si la commande réussit.
fn git(arguments: &[&str]) -> Option<String> {
    let sortie = Command::new("git").args(arguments).output().ok()?;
    let texte = String::from_utf8(sortie.stdout).ok()?;
    let texte = texte.trim().to_string();
    (sortie.status.success() && !texte.is_empty()).then_some(texte)
}

fn main() {
    if let Some(commit) = git(&["rev-parse", "--short=12", "HEAD"]) {
        println!("cargo:rustc-env=MEDX_COMMIT={commit}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    // Relancé à chaque nouveau commit : `HEAD` ne change que de branche ; un commit déplace le fichier de la branche (ou `packed-refs`).
    if let Some(dossier_git) = git(&["rev-parse", "--absolute-git-dir"]).map(PathBuf::from) {
        println!("cargo:rerun-if-changed={}", dossier_git.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", dossier_git.join("packed-refs").display());
        if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
            // en worktree, les branches sont dans le dossier commun
            let commun =
                git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]).map_or(dossier_git, PathBuf::from);
            println!("cargo:rerun-if-changed={}", commun.join(reference).display());
        }
    }
}
