//! Ce que la machine offre pour le calcul, sans dépendre de Burn : chaque module en déduit son device par défaut.

use std::path::Path;

/// Fichier que le pilote NVIDIA crée sous Linux quand il est chargé.
const PILOTE_NVIDIA: &str = "/proc/driver/nvidia/version";

/// Vrai si un pilote NVIDIA est chargé (Linux) : CUDA peut alors servir, si le binaire a été compilé avec la feature `cuda`.
/// Lecture d'un fichier seulement : aucune bibliothèque CUDA n'est chargée, rien ne peut paniquer sur une machine sans GPU.
pub fn gpu_nvidia() -> bool {
    pilote_present(Path::new(PILOTE_NVIDIA))
}

fn pilote_present(chemin: &Path) -> bool {
    cfg!(target_os = "linux") && chemin.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Le pilote n'est reconnu que par la présence de son fichier, et seulement sous Linux.
    #[test]
    fn the_nvidia_driver_is_detected_by_its_proc_file_on_linux_only() {
        let dossier = std::env::temp_dir().join(format!("medoxide-machine-{}", std::process::id()));
        std::fs::create_dir_all(&dossier).unwrap();
        let fichier = dossier.join("version");
        assert!(!pilote_present(&fichier));
        std::fs::write(&fichier, "NVRM version").unwrap();
        assert_eq!(pilote_present(&fichier), cfg!(target_os = "linux"));
        assert!(!pilote_present(&dossier), "un dossier n'est pas le fichier du pilote");
        std::fs::remove_dir_all(&dossier).unwrap();
    }
}
