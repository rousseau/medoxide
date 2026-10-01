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
}

impl std::fmt::Display for MaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaskError::NotImplementedYet => {
                write!(f, "inférence pas encore branchée (prochaine étape)")
            }
        }
    }
}

impl std::error::Error for MaskError {}

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

    #[test]
    fn segment_reports_not_implemented_for_now() {
        let result = segment(Path::new("in.nii.gz"), Path::new("out.nii.gz"));
        assert!(matches!(result, Err(MaskError::NotImplementedYet)));
    }
}
