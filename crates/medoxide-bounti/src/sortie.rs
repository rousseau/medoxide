//! Écriture des résultats de BOUNTI : étiquettes de tissus (`dseg.nii.gz`), table des étiquettes (`dseg.tsv`) et masque du cerveau, dans un dossier au choix ou dans l'arborescence BIDS dérivée que lit guioxide (`docs/donnees.md` de guioxide) :
//! `<étude>/derivatives/medx-bounti/sub-X/ses-Y/anat/sub-X_ses-Y_desc-bounti_dseg.nii.gz`, `…_dseg.tsv`, `…_desc-bounti_mask.nii.gz`, avec `derivatives/medx-bounti/dataset_description.json`.
//! `<étude>` est la racine de l'étude de [`medoxide_core::bids::racine_etude`] : le parent de la racine BIDS si elle s'appelle `sourcedata`, sinon la racine BIDS, ou le dossier donné à `--output-dir`.
use crate::segmentation::Segmentation;
use crate::table::dseg_tsv;
use medoxide_core::bids::{self, ErreurBids, Generateur};
use nalgebra::Matrix4;
use ndarray::Array3;
use std::path::{Path, PathBuf};

/// Erreurs d'écriture.
#[derive(Debug)]
pub enum ErreurSortie {
    /// Identifiant de participant ou de session invalide (BIDS : lettres et chiffres seulement, sans préfixe `sub-` ni `ses-`).
    Identifiant(String),
    /// Écriture d'un fichier ou création d'un dossier.
    Ecriture {
        /// Chemin concerné.
        chemin: PathBuf,
        /// Message de l'erreur.
        message: String,
    },
}

impl std::fmt::Display for ErreurSortie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErreurSortie::Identifiant(v) => write!(
                f,
                "identifiant BIDS invalide « {v} » : lettres et chiffres seulement, sans préfixe sub- ni ses-"
            ),
            ErreurSortie::Ecriture { chemin, message } => write!(f, "écriture de {} : {message}", chemin.display()),
        }
    }
}

impl std::error::Error for ErreurSortie {}

impl From<ErreurBids> for ErreurSortie {
    fn from(e: ErreurBids) -> Self {
        match e {
            ErreurBids::Identifiant(v) => ErreurSortie::Identifiant(v),
            ErreurBids::Fichier { chemin, message } => ErreurSortie::Ecriture { chemin, message },
            // `bounti` ne liste pas de piles
            ErreurBids::AucunePile(chemin) => ErreurSortie::Ecriture {
                chemin,
                message: "aucune pile".to_string(),
            },
        }
    }
}

/// Chemins des fichiers à écrire.
#[derive(Debug, PartialEq)]
pub struct Fichiers {
    /// Étiquettes de tissus (`uint8`).
    pub etiquettes: PathBuf,
    /// Table des étiquettes.
    pub table: PathBuf,
    /// Masque du cerveau (`uint8`, 0 ou 1), s'il est demandé.
    pub masque: Option<PathBuf>,
    /// Racine de l'arborescence dérivée (`<étude>/derivatives/medx-bounti`) en mode BIDS : elle reçoit `dataset_description.json`.
    pub derive: Option<PathBuf>,
}

/// Chemins BIDS des résultats : `<étude>/derivatives/medx-bounti/sub-<participant>[/ses-<session>]/anat/` et les noms `sub-<participant>[_ses-<session>]_desc-bounti_dseg.nii.gz`, `…_dseg.tsv`, `…_desc-bounti_mask.nii.gz`.
/// L'étude est celle de [`medoxide_core::bids::racine_etude`]`(racine_bids, sortie)`.
///
/// # Erreurs
/// [`ErreurSortie::Identifiant`] si le participant ou la session n'est pas alphanumérique (la valeur entre dans un chemin : pas de séparateur ni de `..`).
pub fn fichiers_bids(
    racine_bids: &Path,
    sortie: Option<&Path>,
    participant: &str,
    session: Option<&str>,
) -> Result<Fichiers, ErreurSortie> {
    let d = bids::dossier_derive(racine_bids, sortie, bids::PIPELINE_BOUNTI, participant, session)?;
    Ok(Fichiers {
        etiquettes: d.anat.join(format!("{}_desc-bounti_dseg.nii.gz", d.prefixe)),
        table: d.anat.join(format!("{}_desc-bounti_dseg.tsv", d.prefixe)),
        masque: Some(d.anat.join(format!("{}_desc-bounti_mask.nii.gz", d.prefixe))),
        derive: Some(d.pipeline),
    })
}

/// Chemins hors BIDS : le fichier d'étiquettes donné, la table à côté (`<nom>.tsv` à la place de `.nii.gz` ou `.nii`), et le masque s'il est demandé.
pub fn fichiers_libres(etiquettes: &Path, masque: Option<&Path>) -> Fichiers {
    let nom = etiquettes
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let base = nom
        .strip_suffix(".nii.gz")
        .or_else(|| nom.strip_suffix(".nii"))
        .unwrap_or(&nom);
    Fichiers {
        etiquettes: etiquettes.to_path_buf(),
        table: etiquettes.with_file_name(format!("{base}.tsv")),
        masque: masque.map(Path::to_path_buf),
        derive: None,
    }
}

/// Écrit un volume `uint8` avec l'affine donnée (`sform` et `qform` cohérents, voir [`medoxide_core::definir_affine`]).
pub fn ecrire_volume(chemin: &Path, donnees: &Array3<u8>, affine: &Matrix4<f64>) -> Result<(), ErreurSortie> {
    let mut en_tete = nifti::NiftiHeader::default();
    medoxide_core::definir_affine(
        &mut en_tete,
        std::array::from_fn(|l| std::array::from_fn(|c| affine[(l, c)])),
    );
    nifti::writer::WriterOptions::new(chemin)
        .reference_header(&en_tete)
        .write_nifti(donnees)
        .map_err(|e| ErreurSortie::Ecriture {
            chemin: chemin.to_path_buf(),
            message: e.to_string(),
        })
}

/// Écrit les résultats aux chemins `fichiers` (dossiers créés au besoin) : étiquettes, table, masque du cerveau s'il y a un chemin pour lui, et `dataset_description.json` (décrit par `generateur`) en mode BIDS. `affine` est celle de l'image segmentée.
///
/// # Erreurs
/// [`ErreurSortie::Ecriture`] si un dossier ou un fichier ne peut pas être écrit.
pub fn ecrire(
    fichiers: &Fichiers,
    resultat: &Segmentation,
    affine: &Matrix4<f64>,
    generateur: &Generateur,
) -> Result<(), ErreurSortie> {
    bids::creer_dossier_parent(&fichiers.etiquettes)?;
    ecrire_volume(&fichiers.etiquettes, &resultat.etiquettes, affine)?;
    bids::ecrire_texte(&fichiers.table, &dseg_tsv())?;
    if let Some(masque) = &fichiers.masque {
        bids::creer_dossier_parent(masque)?;
        ecrire_volume(masque, &resultat.cerveau.mapv(u8::from), affine)?;
    }
    if let Some(derive) = &fichiers.derive {
        bids::ecrire_description_derivee(derive, bids::PIPELINE_BOUNTI, generateur)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometrie::Volume;

    /// Chemins BIDS avec session, sans session, et refus des identifiants qui sortiraient de l'arborescence ou portent un préfixe.
    #[test]
    fn the_bids_paths_follow_the_derivative_layout() {
        let f = fichiers_bids(Path::new("/d"), None, "01", Some("A2")).unwrap();
        assert_eq!(
            f.etiquettes,
            Path::new("/d/derivatives/medx-bounti/sub-01/ses-A2/anat/sub-01_ses-A2_desc-bounti_dseg.nii.gz")
        );
        assert_eq!(
            f.table,
            Path::new("/d/derivatives/medx-bounti/sub-01/ses-A2/anat/sub-01_ses-A2_desc-bounti_dseg.tsv")
        );
        assert_eq!(
            f.masque.as_deref(),
            Some(Path::new(
                "/d/derivatives/medx-bounti/sub-01/ses-A2/anat/sub-01_ses-A2_desc-bounti_mask.nii.gz"
            ))
        );
        assert_eq!(f.derive.as_deref(), Some(Path::new("/d/derivatives/medx-bounti")));
        let sans = fichiers_bids(Path::new("/d"), None, "01", None).unwrap();
        assert_eq!(
            sans.etiquettes,
            Path::new("/d/derivatives/medx-bounti/sub-01/anat/sub-01_desc-bounti_dseg.nii.gz")
        );
        for mauvais in ["", "sub-01", "../x", "a/b", "a b", "é"] {
            assert!(
                fichiers_bids(Path::new("/d"), None, mauvais, None).is_err(),
                "participant {mauvais:?}"
            );
            assert!(
                fichiers_bids(Path::new("/d"), None, "01", Some(mauvais)).is_err(),
                "session {mauvais:?}"
            );
        }
    }

    /// Racine de l'étude : sous une racine BIDS `sourcedata/` (buckets Garage), les dérivées vont à côté, dans `<parent>/derivatives/medx-bounti/`, et non dans `sourcedata/derivatives/` ; `--output-dir` remplace la racine de l'étude.
    #[test]
    fn the_derivatives_sit_next_to_sourcedata_not_inside() {
        let f = fichiers_bids(Path::new("/g/fetal-x/sourcedata"), None, "01", Some("1")).unwrap();
        let dossier = "/g/fetal-x/derivatives/medx-bounti/sub-01/ses-1/anat";
        assert_eq!(
            f.etiquettes,
            Path::new(&format!("{dossier}/sub-01_ses-1_desc-bounti_dseg.nii.gz"))
        );
        assert_eq!(
            f.derive.as_deref(),
            Some(Path::new("/g/fetal-x/derivatives/medx-bounti"))
        );
        // une racine BIDS d'un autre nom garde ses dérivées à l'intérieur
        let f = fichiers_bids(Path::new("/g/jeu"), None, "01", None).unwrap();
        assert_eq!(f.derive.as_deref(), Some(Path::new("/g/jeu/derivatives/medx-bounti")));
        // --output-dir remplace la racine de l'étude, `sourcedata` ou non
        let f = fichiers_bids(Path::new("/g/fetal-x/sourcedata"), Some(Path::new("/o")), "01", None).unwrap();
        assert_eq!(f.derive.as_deref(), Some(Path::new("/o/derivatives/medx-bounti")));
    }

    /// Hors BIDS : la table est à côté des étiquettes, `.nii.gz` ou `.nii` remplacé par `.tsv` (un nom sans extension NIfTI reçoit `.tsv`).
    #[test]
    fn the_free_paths_put_the_table_next_to_the_labels() {
        let f = fichiers_libres(Path::new("sortie/seg.nii.gz"), Some(Path::new("m.nii.gz")));
        assert_eq!(f.table, Path::new("sortie/seg.tsv"));
        assert_eq!(f.masque.as_deref(), Some(Path::new("m.nii.gz")));
        assert!(f.derive.is_none());
        assert_eq!(fichiers_libres(Path::new("s.nii"), None).table, Path::new("s.tsv"));
        assert_eq!(fichiers_libres(Path::new("s"), None).table, Path::new("s.tsv"));
        assert!(fichiers_libres(Path::new("s.nii"), None).masque.is_none());
    }

    /// Aller-retour : étiquettes, masque et affine oblique et anisotrope relus à l'identique ; la table est celle de `dseg_tsv()` ; le fichier de description contient la version ; les dossiers sont créés.
    #[test]
    fn the_written_files_read_back_identically() {
        let racine = std::env::temp_dir().join(format!("medoxide_sortie_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&racine);
        let affine = Matrix4::new(
            0.8, 0.05, 0.0, -10.5, -0.03, 0.9, 0.02, 4.25, 0.0, -0.04, 1.1, 30.0, 0.0, 0.0, 0.0, 1.0,
        );
        let etiquettes = Array3::from_shape_fn((5, 4, 3), |(i, j, k)| ((i * 7 + j * 3 + k) % 20) as u8);
        let cerveau = Array3::from_shape_fn((5, 4, 3), |(i, j, k)| (i + j + k) % 3 == 0);
        let resultat = Segmentation {
            etiquettes: etiquettes.clone(),
            cerveau: cerveau.clone(),
        };
        let fichiers = fichiers_bids(&racine, None, "01", Some("1")).unwrap();
        ecrire(&fichiers, &resultat, &affine, &Generateur::new("medx bounti")).unwrap();

        let lu = Volume::lire(&fichiers.etiquettes).unwrap();
        assert_eq!(lu.donnees.mapv(|v| v as u8), etiquettes);
        for l in 0..4 {
            for c in 0..4 {
                assert!((lu.affine[(l, c)] - affine[(l, c)]).abs() < 1e-5, "affine ({l},{c})");
            }
        }
        let masque = Volume::lire(fichiers.masque.as_ref().unwrap()).unwrap();
        assert_eq!(masque.donnees.mapv(|v| v > 0.5), cerveau);
        assert_eq!(std::fs::read_to_string(&fichiers.table).unwrap(), dseg_tsv());
        let description =
            std::fs::read_to_string(racine.join("derivatives/medx-bounti/dataset_description.json")).unwrap();
        assert!(description.contains("\"DatasetType\": \"derivative\""));
        assert!(description.contains(&format!("\"Version\": \"{}\"", env!("CARGO_PKG_VERSION"))));
        let _ = std::fs::remove_dir_all(&racine);
    }

    /// Sans chemin de masque (hors BIDS), aucun masque n'est écrit ; le dossier de sortie absent est créé.
    #[test]
    fn no_mask_is_written_without_a_path_and_missing_folders_are_created() {
        let racine = std::env::temp_dir().join(format!("medoxide_sortie_libre_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&racine);
        let resultat = Segmentation {
            etiquettes: Array3::from_elem((2, 2, 2), 3u8),
            cerveau: Array3::from_elem((2, 2, 2), true),
        };
        let fichiers = fichiers_libres(&racine.join("a/b/seg.nii.gz"), None);
        ecrire(
            &fichiers,
            &resultat,
            &Matrix4::identity(),
            &Generateur::new("medx bounti"),
        )
        .unwrap();
        assert!(racine.join("a/b/seg.nii.gz").is_file() && racine.join("a/b/seg.tsv").is_file());
        assert_eq!(std::fs::read_dir(racine.join("a/b")).unwrap().count(), 2);
        let _ = std::fs::remove_dir_all(&racine);
    }
}
