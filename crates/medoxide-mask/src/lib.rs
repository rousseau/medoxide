//! `medoxide-mask` : extraction du masque cérébral fœtal.
//!
//! Portage du modèle de segmentation Fetal-BET (PyTorch) vers Burn,
//! via un export ONNX intermédiaire. Voir `docs/LEARNING.md` à la racine
//! du workspace pour le détail de la démarche.
//!
//! Statut : squelette. La logique d'inférence arrive à l'étape suivante.

use std::path::Path;

/// Erreurs possibles lors de la segmentation.
///
/// On définit ce type dès maintenant, même vide de variantes utiles,
/// pour que la signature de `segment` ne change plus une fois l'inférence
/// branchée dessus — c'est l'API qui doit être stable, pas forcément
/// l'implémentation derrière.
#[derive(Debug)]
pub enum MaskError {
    NotImplementedYet,
    /// Échec de lecture d'un fichier NIfTI (fichier absent, format invalide...).
    /// La variante *contient* l'erreur d'origine de la crate `nifti`.
    Nifti(nifti::NiftiError),
}

/// Permet à l'opérateur `?` de convertir automatiquement une erreur de la
/// crate `nifti` en `MaskError::Nifti`, dans toute fonction qui renvoie
/// un `Result<_, MaskError>`.
impl From<nifti::NiftiError> for MaskError {
    fn from(e: nifti::NiftiError) -> Self {
        MaskError::Nifti(e)
    }
}

impl std::fmt::Display for MaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaskError::NotImplementedYet => {
                write!(f, "inférence pas encore branchée (prochaine étape)")
            }
            MaskError::Nifti(e) => write!(f, "lecture NIfTI : {e}"),
        }
    }
}

impl std::error::Error for MaskError {}

/// Informations géométriques d'un volume NIfTI, lues dans son en-tête.
#[derive(Debug, PartialEq)]
pub struct VolumeInfo {
    /// Nombre de voxels par axe (ex. `[256, 256, 44]`).
    pub dim: Vec<u16>,
    /// Taille d'un voxel en mm sur les 3 premiers axes (x, y, z).
    pub spacing: [f32; 3],
}

/// Lit les dimensions et l'espacement d'un volume NIfTI (`.nii` ou `.nii.gz`).
///
/// Seul l'en-tête (348 octets) est lu : les voxels ne sont pas chargés.
///
/// # Erreurs
/// `MaskError::Nifti` si le fichier est absent, illisible ou invalide.
pub fn volume_info(path: &Path) -> Result<VolumeInfo, MaskError> {
    let header = nifti::NiftiHeader::from_file(path)?;
    let dim = header.dim()?.to_vec();
    // pixdim[0] sert à autre chose (sens de rotation) : les tailles de voxel
    // sont dans pixdim[1..=3].
    let spacing = [header.pixdim[1], header.pixdim[2], header.pixdim[3]];
    Ok(VolumeInfo { dim, spacing })
}

/// Calcule le masque cérébral d'un volume IRM fœtal.
///
/// # Étapes prévues (à venir)
/// 1. Charger le volume NIfTI depuis `input_path`.
/// 2. Faire passer chaque coupe (ou le volume) dans le modèle Burn
///    importé depuis l'ONNX de Fetal-BET.
/// 3. Ré-assembler et écrire le masque résultant dans `output_path`.
pub fn segment(_input_path: &Path, _output_path: &Path) -> Result<(), MaskError> {
    Err(MaskError::NotImplementedYet)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Valeurs de référence calculées avec nibabel (arrondies à 2 décimales).
    /// Les volumes se récupèrent avec :
    /// `rclone sync garage:study-x/medoxide-dev/sourcedata data/sourcedata`
    #[test]
    fn volume_info_matches_nibabel() {
        let attendu: [(&str, u16, f32, f32); 8] = [
            ("fetus_03", 44, 1.17, 2.0),
            ("fetus_04", 40, 1.17, 3.0),
            ("fetus_06", 46, 0.94, 2.0),
            ("fetus_07", 50, 1.02, 2.5),
            ("fetus_09", 50, 1.02, 2.5),
            ("fetus_10", 50, 1.02, 2.5),
            ("fetus_11", 50, 1.02, 2.5),
            ("fetus_12", 50, 1.02, 2.5),
        ];
        for (nom, nz, pixel, epaisseur) in attendu {
            let chemin = format!(
                "{}/../../data/sourcedata/{nom}.nii.gz",
                env!("CARGO_MANIFEST_DIR")
            );
            let info = volume_info(Path::new(&chemin)).expect(&chemin);
            assert_eq!(info.dim, vec![256, 256, nz], "{nom}");
            let voulu = [pixel, pixel, epaisseur];
            for (lu, v) in info.spacing.iter().zip(voulu) {
                assert!((lu - v).abs() < 0.005, "{nom} : espacement {lu} != {v}");
            }
        }
    }

    #[test]
    fn volume_info_fails_on_missing_file() {
        let r = volume_info(Path::new("n_existe_pas.nii.gz"));
        assert!(matches!(r, Err(MaskError::Nifti(_))));
    }

    #[test]
    fn segment_reports_not_implemented_for_now() {
        let result = segment(Path::new("in.nii.gz"), Path::new("out.nii.gz"));
        assert!(matches!(result, Err(MaskError::NotImplementedYet)));
    }
}
