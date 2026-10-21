//! Écriture des résultats de BOUNTI : étiquettes de tissus (`dseg.nii.gz`), table des étiquettes (`dseg.tsv`) et masque du cerveau, dans un dossier au choix ou dans l'arborescence BIDS dérivée que lit guioxide (`docs/donnees.md` de guioxide) :
//! `<racine>/derivatives/medx-bounti/sub-X/ses-Y/anat/sub-X_ses-Y_desc-bounti_dseg.nii.gz`, `…_dseg.tsv`, `…_desc-bounti_mask.nii.gz`, avec `derivatives/medx-bounti/dataset_description.json`.
use crate::segmentation::Segmentation;
use crate::table::dseg_tsv;
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

/// Chemins des fichiers à écrire.
#[derive(Debug, PartialEq)]
pub struct Fichiers {
    /// Étiquettes de tissus (`uint8`).
    pub etiquettes: PathBuf,
    /// Table des étiquettes.
    pub table: PathBuf,
    /// Masque du cerveau (`uint8`, 0 ou 1), s'il est demandé.
    pub masque: Option<PathBuf>,
    /// Racine de l'arborescence dérivée (`<racine>/derivatives/medx-bounti`) en mode BIDS : elle reçoit `dataset_description.json`.
    pub derive: Option<PathBuf>,
}

/// Un identifiant BIDS : lettres et chiffres ASCII, non vide.
fn verifier_identifiant(valeur: &str) -> Result<(), ErreurSortie> {
    if !valeur.is_empty() && valeur.chars().all(|c| c.is_ascii_alphanumeric()) {
        Ok(())
    } else {
        Err(ErreurSortie::Identifiant(valeur.to_string()))
    }
}

/// Chemins BIDS des résultats : `<racine>/derivatives/medx-bounti/sub-<participant>[/ses-<session>]/anat/` et les noms `sub-<participant>[_ses-<session>]_desc-bounti_dseg.nii.gz`, `…_dseg.tsv`, `…_desc-bounti_mask.nii.gz`.
///
/// # Erreurs
/// [`ErreurSortie::Identifiant`] si le participant ou la session n'est pas alphanumérique (la valeur entre dans un chemin : pas de séparateur ni de `..`).
pub fn fichiers_bids(racine: &Path, participant: &str, session: Option<&str>) -> Result<Fichiers, ErreurSortie> {
    verifier_identifiant(participant)?;
    let derive = racine.join("derivatives/medx-bounti");
    let mut dossier = derive.join(format!("sub-{participant}"));
    let mut prefixe = format!("sub-{participant}");
    if let Some(s) = session {
        verifier_identifiant(s)?;
        dossier = dossier.join(format!("ses-{s}"));
        prefixe = format!("{prefixe}_ses-{s}");
    }
    let dossier = dossier.join("anat");
    Ok(Fichiers {
        etiquettes: dossier.join(format!("{prefixe}_desc-bounti_dseg.nii.gz")),
        table: dossier.join(format!("{prefixe}_desc-bounti_dseg.tsv")),
        masque: Some(dossier.join(format!("{prefixe}_desc-bounti_mask.nii.gz"))),
        derive: Some(derive),
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

fn ecrire_texte(chemin: &Path, texte: &str) -> Result<(), ErreurSortie> {
    std::fs::write(chemin, texte).map_err(|e| ErreurSortie::Ecriture {
        chemin: chemin.to_path_buf(),
        message: e.to_string(),
    })
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

/// Contenu de `dataset_description.json` d'un dossier dérivé BIDS (nom, version de BIDS, générateur et sa version).
fn description_du_jeu() -> String {
    format!(
        "{{\n  \"Name\": \"medx-bounti\",\n  \"BIDSVersion\": \"1.9.0\",\n  \"DatasetType\": \"derivative\",\n  \"GeneratedBy\": [\n    {{\n      \"Name\": \"medx bounti\",\n      \"Version\": \"{}\",\n      \"CodeURL\": \"https://github.com/rousseau/medoxide\"\n    }}\n  ]\n}}\n",
        env!("CARGO_PKG_VERSION")
    )
}

/// Écrit les résultats aux chemins `fichiers` (dossiers créés au besoin) : étiquettes, table, masque du cerveau s'il y a un chemin pour lui, et `dataset_description.json` en mode BIDS. `affine` est celle de l'image segmentée.
///
/// # Erreurs
/// [`ErreurSortie::Ecriture`] si un dossier ou un fichier ne peut pas être écrit.
pub fn ecrire(fichiers: &Fichiers, resultat: &Segmentation, affine: &Matrix4<f64>) -> Result<(), ErreurSortie> {
    let creer = |chemin: &Path| {
        if let Some(parent) = chemin.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| ErreurSortie::Ecriture {
                chemin: parent.to_path_buf(),
                message: e.to_string(),
            })?;
        }
        Ok(())
    };
    creer(&fichiers.etiquettes)?;
    ecrire_volume(&fichiers.etiquettes, &resultat.etiquettes, affine)?;
    creer(&fichiers.table)?;
    ecrire_texte(&fichiers.table, &dseg_tsv())?;
    if let Some(masque) = &fichiers.masque {
        creer(masque)?;
        ecrire_volume(masque, &resultat.cerveau.mapv(u8::from), affine)?;
    }
    if let Some(derive) = &fichiers.derive {
        ecrire_texte(&derive.join("dataset_description.json"), &description_du_jeu())?;
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
        let f = fichiers_bids(Path::new("/d"), "01", Some("A2")).unwrap();
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
        let sans = fichiers_bids(Path::new("/d"), "01", None).unwrap();
        assert_eq!(
            sans.etiquettes,
            Path::new("/d/derivatives/medx-bounti/sub-01/anat/sub-01_desc-bounti_dseg.nii.gz")
        );
        for mauvais in ["", "sub-01", "../x", "a/b", "a b", "é"] {
            assert!(
                fichiers_bids(Path::new("/d"), mauvais, None).is_err(),
                "participant {mauvais:?}"
            );
            assert!(
                fichiers_bids(Path::new("/d"), "01", Some(mauvais)).is_err(),
                "session {mauvais:?}"
            );
        }
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
        let fichiers = fichiers_bids(&racine, "01", Some("1")).unwrap();
        ecrire(&fichiers, &resultat, &affine).unwrap();

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
        ecrire(&fichiers, &resultat, &Matrix4::identity()).unwrap();
        assert!(racine.join("a/b/seg.nii.gz").is_file() && racine.join("a/b/seg.tsv").is_file());
        assert_eq!(std::fs::read_dir(racine.join("a/b")).unwrap().count(), 2);
        let _ = std::fs::remove_dir_all(&racine);
    }
}
