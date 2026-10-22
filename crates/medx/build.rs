//! Retient le commit du code compilé (variable `MEDX_COMMIT`, lue par `option_env!`), pour le déclarer dans les `dataset_description.json` écrits par `medx`.
//! Sans `git` ou hors d'un dépôt (archive des sources), la variable n'est pas définie et le commit est omis.

fn main() {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|sortie| sortie.status.success())
        .and_then(|sortie| String::from_utf8(sortie.stdout).ok())
        .map(|texte| texte.trim().to_string())
        .filter(|texte| !texte.is_empty());
    if let Some(commit) = commit {
        println!("cargo:rustc-env=MEDX_COMMIT={commit}");
    }
    // relancé quand HEAD change (`git commit`, `git checkout`)
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs");
}
