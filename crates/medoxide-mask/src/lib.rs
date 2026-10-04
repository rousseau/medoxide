//! `medoxide-mask` : extraction du masque cérébral fœtal.
//!
//! Portage du modèle de segmentation Fetal-BET (PyTorch) vers Burn,
//! via un export ONNX intermédiaire. Voir `docs/LEARNING.md` à la racine
//! du workspace pour le détail de la démarche.
//!
//! Statut : squelette. La logique d'inférence arrive à l'étape suivante.

use std::path::Path;

use ndarray::{Array3, Ix3};

mod model;

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
    /// Le fichier n'est pas un volume à 3 axes ; contient ses dimensions.
    NotVolume3D(Vec<usize>),
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
            MaskError::NotVolume3D(dim) => write!(f, "volume 3D attendu, dimensions : {dim:?}"),
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
    /// Affine voxel → mm (matrice 4×4, ligne par ligne) lue dans le `sform`.
    /// `None` si l'en-tête n'a pas de `sform` (`sform_code == 0`).
    pub affine: Option<[[f32; 4]; 4]>,
}

/// Lit les dimensions, l'espacement et l'affine d'un volume NIfTI
/// (`.nii` ou `.nii.gz`).
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
    // Les 3 premières lignes de l'affine sont stockées dans srow_x/y/z ;
    // la 4e est toujours [0, 0, 0, 1] par convention.
    let affine = if header.sform_code != 0 {
        Some([
            header.srow_x,
            header.srow_y,
            header.srow_z,
            [0.0, 0.0, 0.0, 1.0],
        ])
    } else {
        None
    };
    Ok(VolumeInfo { dim, spacing, affine })
}

/// Lit les voxels d'un volume NIfTI (`.nii` ou `.nii.gz`) en `f32`, axes `[x, y, z]`.
///
/// Le facteur d'échelle de l'en-tête (`scl_slope`, `scl_inter`) est appliqué.
///
/// # Erreurs
/// `MaskError::Nifti` si le fichier est illisible ou invalide ;
/// `MaskError::NotVolume3D` s'il n'a pas exactement 3 axes.
pub fn read_volume(path: &Path) -> Result<Array3<f32>, MaskError> {
    use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

    let objet = ReaderOptions::new().read_file(path)?;
    // Tableau à nombre d'axes dynamique, dans l'ordre mémoire du fichier (Fortran).
    let dynamique = objet.into_volume().into_ndarray::<f32>()?;
    let dim = dynamique.shape().to_vec();
    // `map_err` transforme l'erreur de ndarray en la nôtre, avec les dimensions.
    dynamique
        .into_dimensionality::<Ix3>()
        .map_err(|_| MaskError::NotVolume3D(dim))
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

    /// Affines de référence (nibabel, `img.affine`) pour 3 volumes dont les
    /// orientations diffèrent (LPS, LIP, PIR), arrondis à 4 décimales.
    #[test]
    fn volume_info_affine_matches_nibabel() {
        let attendu: [(&str, [[f32; 4]; 4]); 3] = [
            ("fetus_03", [
                [-1.1719, 0.0, 0.0, 155.1199],
                [0.0, -1.0771, 0.7879, 58.3842],
                [0.0, 0.4617, 1.8383, -254.6115],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            ("fetus_06", [
                [-0.8439, -0.147, -0.8127, 141.5291],
                [0.4083, -0.3038, -1.6797, -19.843],
                [0.0, -0.8747, 0.7199, -69.1684],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            ("fetus_12", [
                [-0.5306, -0.4462, 1.8271, 96.2711],
                [-0.866, 0.2733, -1.1193, 46.4229],
                [0.0, -0.8705, -1.288, -12.331],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        ];
        for (nom, voulu) in attendu {
            let chemin = format!(
                "{}/../../data/sourcedata/{nom}.nii.gz",
                env!("CARGO_MANIFEST_DIR")
            );
            let info = volume_info(Path::new(&chemin)).expect(&chemin);
            let lu = info.affine.expect("sform attendu");
            for (ligne_lue, ligne_voulue) in lu.iter().zip(voulu) {
                for (a, b) in ligne_lue.iter().zip(ligne_voulue) {
                    assert!((a - b).abs() < 1e-3, "{nom} : {lu:?} != {voulu:?}");
                }
            }
        }
    }

    /// Lit un fichier de `f32` bruts en little-endian (4 octets par valeur).
    fn lire_f32(chemin: &str) -> Vec<f32> {
        let octets = std::fs::read(chemin).expect(chemin);
        octets
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Étape 4b : les logits Burn (wgpu) doivent égaler ceux d'onnxruntime.
    /// Références produites par `python scripts/make_reference_slice.py`.
    /// Critère : écart relatif (max |Δ| / max |logit|) < 1e-4, argmax identique.
    #[test]
    fn inference_matches_onnxruntime() {
        use burn::tensor::{Device, Tensor, TensorData};

        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let entree = lire_f32(&format!("{racine}/data/reference/slice_input.f32"));
        let attendu = lire_f32(&format!("{racine}/data/reference/slice_logits.f32"));

        let device = Device::default();
        println!("périphérique : {device:?}");
        let modele = model::Model::from_file(format!("{racine}/models/attunet.bpk"), &device);
        let x = Tensor::<4>::from_data(TensorData::new(entree, [1, 1, 256, 256]), &device);
        let lu = modele.forward(x).into_data().try_to_vec::<f32>().unwrap();

        assert_eq!(lu.len(), attendu.len());
        let max_logit = attendu.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let max_ecart = lu
            .iter()
            .zip(&attendu)
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        let relatif = max_ecart / max_logit;

        // Sortie [1, 2, 256, 256] : canal 0 = fond, canal 1 = cerveau.
        let n = 256 * 256;
        let classe = |v: &[f32], i: usize| usize::from(v[n + i] > v[i]);
        let accord = (0..n).filter(|&i| classe(&lu, i) == classe(&attendu, i)).count();
        let masque = (0..n).filter(|&i| classe(&attendu, i) == 1).count();
        println!("écart relatif {relatif:.2e} ; argmax identique {accord}/{n} ; voxels masque {masque}");

        assert!(masque > 0, "tuile de référence sans masque");
        assert!(relatif < 1e-4, "écart relatif {relatif:.2e} >= 1e-4");
        assert_eq!(accord, n, "argmax différent sur {} voxels", n - accord);
    }

    /// Étape 5a : voxels identiques (bit à bit) à `nibabel.get_fdata()`.
    /// Références produites par `python scripts/make_reference_volumes.py`.
    #[test]
    fn read_volume_matches_nibabel() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        for nom in ["03", "04", "06", "07", "09", "10", "11", "12"] {
            let volume = read_volume(Path::new(&format!(
                "{racine}/data/sourcedata/fetus_{nom}.nii.gz"
            )))
            .unwrap();
            let attendu = lire_f32(&format!("{racine}/data/reference/fetus_{nom}_raw.f32"));
            assert_eq!(volume.len(), attendu.len(), "fetus_{nom} : taille");
            // `iter()` parcourt dans l'ordre logique [x, y, z] (z varie le plus vite),
            // quel que soit l'ordre mémoire : c'est l'ordre du fichier de référence.
            let diff = volume.iter().zip(&attendu).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "fetus_{nom} : {diff} voxels différents");
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
