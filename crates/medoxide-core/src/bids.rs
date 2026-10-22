//! Chemins et fichiers communs des sorties BIDS de `medx` (`fetalbet`, `svr recon`, `bounti`), tels que les lit guioxide (`docs/donnees.md` de guioxide).
//!
//! Les dérivées d'un pipeline sont dans `<étude>/derivatives/<pipeline>/sub-X[/ses-Y]/anat/`, avec un `dataset_description.json` à la racine du pipeline. `<étude>` est la racine de l'étude (voir [`racine_etude`]).

use std::path::{Path, PathBuf};

use crate::json::Json;

/// Dossier de dérivées de `medx fetalbet` (masques cérébraux des piles).
pub const PIPELINE_FETALBET: &str = "medx-fetalbet";
/// Dossier de dérivées de `medx svr recon` (volume reconstruit, mouvement, qualité par coupe).
pub const PIPELINE_SVR: &str = "medx-svr";
/// Dossier de dérivées de `medx bounti` (segmentation en tissus).
pub const PIPELINE_BOUNTI: &str = "medx-bounti";

/// Version de BIDS déclarée dans `dataset_description.json`.
const VERSION_BIDS: &str = "1.9.0";
/// Dépôt du code, déclaré dans `GeneratedBy`.
const URL_DU_CODE: &str = "https://github.com/rousseau/medoxide";

/// Erreurs des chemins et des fichiers BIDS.
#[derive(Debug)]
pub enum ErreurBids {
    /// Identifiant de participant ou de session invalide (BIDS : lettres et chiffres seulement, sans préfixe `sub-` ni `ses-`).
    Identifiant(String),
    /// Lecture ou écriture d'un fichier, lecture ou création d'un dossier.
    Fichier {
        /// Chemin concerné.
        chemin: PathBuf,
        /// Message de l'erreur.
        message: String,
    },
    /// Aucune pile `*_T2w.nii.gz` dans ce dossier.
    AucunePile(PathBuf),
}

impl std::fmt::Display for ErreurBids {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErreurBids::Identifiant(v) => write!(
                f,
                "identifiant BIDS invalide « {v} » : lettres et chiffres seulement, sans préfixe sub- ni ses-"
            ),
            ErreurBids::Fichier { chemin, message } => write!(f, "{} : {message}", chemin.display()),
            ErreurBids::AucunePile(dossier) => write!(f, "{} : aucune pile *_T2w.nii.gz", dossier.display()),
        }
    }
}

impl std::error::Error for ErreurBids {}

/// Racine de l'étude, là où vont les `derivatives/` : `sortie` si elle est donnée (`--output-dir`) ; sinon le **parent** de la racine BIDS quand celle-ci s'appelle `sourcedata` (buckets Garage : `sourcedata/` et `derivatives/` côte à côte) ; sinon la racine BIDS elle-même.
pub fn racine_etude(racine_bids: &Path, sortie: Option<&Path>) -> PathBuf {
    if let Some(dossier) = sortie {
        return dossier.to_path_buf();
    }
    if racine_bids.file_name().is_some_and(|nom| nom == "sourcedata") {
        if let Some(parent) = racine_bids.parent() {
            return parent.to_path_buf();
        }
    }
    racine_bids.to_path_buf()
}

/// Un identifiant BIDS : lettres et chiffres ASCII, non vide (la valeur entre dans un chemin : ni séparateur ni `..`).
///
/// # Erreurs
/// [`ErreurBids::Identifiant`] sinon.
pub fn verifier_identifiant(valeur: &str) -> Result<(), ErreurBids> {
    if !valeur.is_empty() && valeur.chars().all(|c| c.is_ascii_alphanumeric()) {
        Ok(())
    } else {
        Err(ErreurBids::Identifiant(valeur.to_string()))
    }
}

/// Sous-dossier d'un participant (et d'une session) et préfixe de nom de fichier : `("sub-X/ses-Y/anat", "sub-X_ses-Y")`, ou `("sub-X/anat", "sub-X")` sans session.
fn sujet(participant: &str, session: Option<&str>) -> Result<(PathBuf, String), ErreurBids> {
    verifier_identifiant(participant)?;
    let mut dossier = PathBuf::from(format!("sub-{participant}"));
    let mut prefixe = format!("sub-{participant}");
    if let Some(s) = session {
        verifier_identifiant(s)?;
        dossier.push(format!("ses-{s}"));
        prefixe = format!("{prefixe}_ses-{s}");
    }
    Ok((dossier.join("anat"), prefixe))
}

/// Où écrit un pipeline pour un participant (et une session).
#[derive(Debug, Clone, PartialEq)]
pub struct DossierDerive {
    /// Racine du pipeline : `<étude>/derivatives/<pipeline>`. Elle reçoit `dataset_description.json`.
    pub pipeline: PathBuf,
    /// Dossier des fichiers : `<pipeline>/sub-X[/ses-Y]/anat`.
    pub anat: PathBuf,
    /// Début des noms de fichier : `sub-X[_ses-Y]`.
    pub prefixe: String,
}

/// Dossier de dérivées de `pipeline` pour un participant et une session éventuelle, dans l'étude de [`racine_etude`]`(racine_bids, sortie)`.
///
/// # Erreurs
/// [`ErreurBids::Identifiant`] si le participant ou la session n'est pas alphanumérique.
pub fn dossier_derive(
    racine_bids: &Path,
    sortie: Option<&Path>,
    pipeline: &str,
    participant: &str,
    session: Option<&str>,
) -> Result<DossierDerive, ErreurBids> {
    let (relatif, prefixe) = sujet(participant, session)?;
    let pipeline = racine_etude(racine_bids, sortie).join("derivatives").join(pipeline);
    Ok(DossierDerive {
        anat: pipeline.join(relatif),
        pipeline,
        prefixe,
    })
}

/// Piles acquises d'un participant (et d'une session) : les fichiers `*_T2w.nii.gz` de `sub-X[/ses-Y]/anat/` sous la racine BIDS, triés par nom.
///
/// # Erreurs
/// [`ErreurBids::Identifiant`] ; [`ErreurBids::Fichier`] si le dossier ne se lit pas ; [`ErreurBids::AucunePile`] s'il ne contient aucune pile.
pub fn piles_bids(racine_bids: &Path, participant: &str, session: Option<&str>) -> Result<Vec<PathBuf>, ErreurBids> {
    let dossier = racine_bids.join(sujet(participant, session)?.0);
    let entrees = std::fs::read_dir(&dossier).map_err(|e| ErreurBids::Fichier {
        chemin: dossier.clone(),
        message: e.to_string(),
    })?;
    let mut piles: Vec<PathBuf> = entrees
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with("_T2w.nii.gz"))
        })
        .collect();
    if piles.is_empty() {
        return Err(ErreurBids::AucunePile(dossier));
    }
    piles.sort();
    Ok(piles)
}

/// Nom d'une pile : son nom de fichier sans l'extension `.nii.gz` (ou `.nii`). C'est ainsi que les tableaux de sortie (mouvement, qualité par coupe) désignent une pile.
pub fn nom_de_pile(chemin: &Path) -> String {
    let nom = chemin
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    nom.strip_suffix(".nii.gz")
        .or_else(|| nom.strip_suffix(".nii"))
        .unwrap_or(&nom)
        .to_string()
}

/// Entités d'une pile : son nom sans le suffixe (`T2w`). `sub-X_ses-Y_acq-ax_run-1_T2w.nii.gz` donne `sub-X_ses-Y_acq-ax_run-1`.
pub fn entites_de_pile(chemin: &Path) -> String {
    let nom = nom_de_pile(chemin);
    match nom.rsplit_once('_') {
        Some((entites, _suffixe)) => entites.to_string(),
        None => nom,
    }
}

/// Masque cérébral d'une pile dans `dossier` (dossier `anat` de [`PIPELINE_FETALBET`]) : `<entités de la pile>_desc-brain_mask.nii.gz`.
pub fn masque_de_pile(dossier: &Path, pile: &Path) -> PathBuf {
    dossier.join(format!("{}_desc-brain_mask.nii.gz", entites_de_pile(pile)))
}

/// Qui a écrit les fichiers : déclaré dans `GeneratedBy` de `dataset_description.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Generateur {
    /// Nom de l'outil, par exemple `medx bounti`.
    pub nom: String,
    /// Version de l'outil.
    pub version: String,
    /// Commit du code, s'il est connu.
    pub commit: Option<String>,
    /// Ligne de commande, si elle est connue.
    pub ligne_de_commande: Option<String>,
}

impl Generateur {
    /// Le générateur `nom`, dans la version de ce workspace (`medx` et ses crates ont la même).
    pub fn new(nom: &str) -> Generateur {
        Generateur {
            nom: nom.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            commit: None,
            ligne_de_commande: None,
        }
    }

    /// Même générateur, avec le commit du code.
    pub fn avec_commit(self, commit: Option<&str>) -> Generateur {
        Generateur {
            commit: commit.map(str::to_string),
            ..self
        }
    }

    /// Même générateur, avec la ligne de commande.
    pub fn avec_ligne_de_commande(self, ligne: &str) -> Generateur {
        Generateur {
            ligne_de_commande: Some(ligne.to_string()),
            ..self
        }
    }

    /// Entrée de `GeneratedBy`.
    fn en_json(&self) -> Json {
        let mut champs = vec![
            ("Name", Json::texte(&self.nom)),
            ("Version", Json::texte(&self.version)),
        ];
        if let Some(c) = &self.commit {
            champs.push(("Commit", Json::texte(c)));
        }
        champs.push(("CodeURL", Json::texte(URL_DU_CODE)));
        if let Some(l) = &self.ligne_de_commande {
            champs.push(("CommandLine", Json::texte(l)));
        }
        Json::objet(champs)
    }
}

/// Crée le dossier parent de `chemin` s'il n'existe pas.
///
/// # Erreurs
/// [`ErreurBids::Fichier`] si le dossier ne peut pas être créé.
pub fn creer_dossier_parent(chemin: &Path) -> Result<(), ErreurBids> {
    if let Some(parent) = chemin.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| ErreurBids::Fichier {
            chemin: parent.to_path_buf(),
            message: e.to_string(),
        })?;
    }
    Ok(())
}

/// Écrit `texte` dans `chemin`, dossiers créés au besoin.
///
/// # Erreurs
/// [`ErreurBids::Fichier`] si un dossier ou le fichier ne peut pas être écrit.
pub fn ecrire_texte(chemin: &Path, texte: &str) -> Result<(), ErreurBids> {
    creer_dossier_parent(chemin)?;
    std::fs::write(chemin, texte).map_err(|e| ErreurBids::Fichier {
        chemin: chemin.to_path_buf(),
        message: e.to_string(),
    })
}

/// Écrit `<pipeline>/dataset_description.json` : `DatasetType: derivative`, version de BIDS et `GeneratedBy` ; `nom_pipeline` est le nom du jeu (`medx-svr`…). Le fichier est réécrit à chaque lancement (il décrit le dernier).
///
/// # Erreurs
/// [`ErreurBids::Fichier`] si le fichier ne peut pas être écrit.
pub fn ecrire_description_derivee(
    pipeline: &Path,
    nom_pipeline: &str,
    generateur: &Generateur,
) -> Result<(), ErreurBids> {
    let description = Json::objet([
        ("Name", Json::texte(nom_pipeline)),
        ("BIDSVersion", Json::texte(VERSION_BIDS)),
        ("DatasetType", Json::texte("derivative")),
        ("GeneratedBy", Json::Liste(vec![generateur.en_json()])),
    ]);
    ecrire_texte(&pipeline.join("dataset_description.json"), &description.en_texte())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Racine de l'étude : `sourcedata` remonte au parent (Garage), toute autre racine reste elle-même, `--output-dir` remplace tout.
    #[test]
    fn the_study_root_is_the_parent_of_sourcedata_else_the_bids_root() {
        assert_eq!(
            racine_etude(Path::new("/g/fetal/sourcedata"), None),
            Path::new("/g/fetal")
        );
        assert_eq!(
            racine_etude(Path::new("/g/fetal/sourcedata/"), None),
            Path::new("/g/fetal")
        );
        assert_eq!(racine_etude(Path::new("sourcedata"), None), Path::new(""));
        assert_eq!(racine_etude(Path::new("/g/jeu"), None), Path::new("/g/jeu"));
        assert_eq!(
            racine_etude(Path::new("/g/sourcedata2"), None),
            Path::new("/g/sourcedata2")
        );
        assert_eq!(
            racine_etude(Path::new("/g/sourcedata/sub-01"), None),
            Path::new("/g/sourcedata/sub-01")
        );
        assert_eq!(
            racine_etude(Path::new("/g/fetal/sourcedata"), Some(Path::new("/tmp/o"))),
            Path::new("/tmp/o")
        );
        assert_eq!(racine_etude(Path::new("/g/jeu"), Some(Path::new("o"))), Path::new("o"));
    }

    /// Chemins de dérivées avec session, sans session, depuis `sourcedata/`, avec `--output-dir`, et refus des identifiants qui sortiraient de l'arborescence.
    #[test]
    fn the_derivative_folder_follows_the_layout() {
        let d = dossier_derive(Path::new("/d"), None, PIPELINE_SVR, "01", Some("A2")).unwrap();
        assert_eq!(d.pipeline, Path::new("/d/derivatives/medx-svr"));
        assert_eq!(d.anat, Path::new("/d/derivatives/medx-svr/sub-01/ses-A2/anat"));
        assert_eq!(d.prefixe, "sub-01_ses-A2");
        let sans = dossier_derive(Path::new("/d"), None, PIPELINE_SVR, "01", None).unwrap();
        assert_eq!(sans.anat, Path::new("/d/derivatives/medx-svr/sub-01/anat"));
        assert_eq!(sans.prefixe, "sub-01");
        let garage = dossier_derive(Path::new("/g/sourcedata"), None, PIPELINE_BOUNTI, "01", Some("1")).unwrap();
        assert_eq!(garage.pipeline, Path::new("/g/derivatives/medx-bounti"));
        let libre = dossier_derive(
            Path::new("/g/sourcedata"),
            Some(Path::new("/o")),
            PIPELINE_BOUNTI,
            "01",
            None,
        )
        .unwrap();
        assert_eq!(libre.anat, Path::new("/o/derivatives/medx-bounti/sub-01/anat"));
        for mauvais in ["", "sub-01", "../x", "a/b", "a b", "é"] {
            assert!(
                dossier_derive(Path::new("/d"), None, PIPELINE_SVR, mauvais, None).is_err(),
                "{mauvais:?}"
            );
            assert!(
                dossier_derive(Path::new("/d"), None, PIPELINE_SVR, "01", Some(mauvais)).is_err(),
                "{mauvais:?}"
            );
        }
    }

    /// Nom de pile, entités et masque : le nom déjà en usage pour les masques de fetalbet.
    #[test]
    fn the_stack_names_and_the_mask_follow_the_contract() {
        let pile = Path::new("/d/sub-X/ses-Y/anat/sub-X_ses-Y_acq-ax_run-1_T2w.nii.gz");
        assert_eq!(nom_de_pile(pile), "sub-X_ses-Y_acq-ax_run-1_T2w");
        assert_eq!(nom_de_pile(Path::new("a/b.nii")), "b");
        assert_eq!(nom_de_pile(Path::new("pile")), "pile");
        assert_eq!(entites_de_pile(pile), "sub-X_ses-Y_acq-ax_run-1");
        assert_eq!(entites_de_pile(Path::new("pile.nii.gz")), "pile");
        assert_eq!(
            masque_de_pile(Path::new("/o/anat"), pile),
            Path::new("/o/anat/sub-X_ses-Y_acq-ax_run-1_desc-brain_mask.nii.gz")
        );
    }

    /// Les piles d'une session sont les `*_T2w.nii.gz` de `anat/`, triées ; les autres fichiers sont ignorés ; un dossier absent ou sans pile est une erreur.
    #[test]
    fn the_stacks_of_a_session_are_listed_in_name_order() {
        let racine = std::env::temp_dir().join(format!("medoxide_bids_piles_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&racine);
        let anat = racine.join("sub-01/ses-1/anat");
        std::fs::create_dir_all(&anat).unwrap();
        for nom in [
            "sub-01_ses-1_acq-sag_T2w.nii.gz",
            "sub-01_ses-1_acq-ax_T2w.nii.gz",
            "sub-01_ses-1_acq-ax_T2w.json",
            "sub-01_ses-1_T1w.nii.gz",
        ] {
            std::fs::write(anat.join(nom), b"").unwrap();
        }
        let piles = piles_bids(&racine, "01", Some("1")).unwrap();
        let noms: Vec<String> = piles.iter().map(|p| nom_de_pile(p)).collect();
        assert_eq!(noms, ["sub-01_ses-1_acq-ax_T2w", "sub-01_ses-1_acq-sag_T2w"]);
        assert!(matches!(
            piles_bids(&racine, "02", Some("1")),
            Err(ErreurBids::Fichier { .. })
        ));
        std::fs::create_dir_all(racine.join("sub-02/anat")).unwrap();
        assert!(matches!(
            piles_bids(&racine, "02", None),
            Err(ErreurBids::AucunePile(_))
        ));
        let _ = std::fs::remove_dir_all(&racine);
    }

    /// `dataset_description.json` : type de jeu, version, générateur avec son commit et sa ligne de commande (échappée) ; sans commit ni ligne, ces deux clés sont absentes.
    #[test]
    fn the_dataset_description_declares_the_generator() {
        let racine = std::env::temp_dir().join(format!("medoxide_bids_description_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&racine);
        let g = Generateur::new("medx svr recon")
            .avec_commit(Some("abc1234"))
            .avec_ligne_de_commande("medx svr recon --bids \"a b\"");
        ecrire_description_derivee(&racine.join("derivatives/medx-svr"), "medx-svr", &g).unwrap();
        let texte = std::fs::read_to_string(racine.join("derivatives/medx-svr/dataset_description.json")).unwrap();
        for attendu in [
            "\"Name\": \"medx-svr\"",
            "\"DatasetType\": \"derivative\"",
            "\"Name\": \"medx svr recon\"",
            &format!("\"Version\": \"{}\"", env!("CARGO_PKG_VERSION")),
            "\"Commit\": \"abc1234\"",
            "\"CommandLine\": \"medx svr recon --bids \\\"a b\\\"\"",
        ] {
            assert!(texte.contains(attendu), "{attendu} absent de {texte}");
        }
        let nu = Generateur::new("medx bounti");
        ecrire_description_derivee(&racine, "medx-bounti", &nu).unwrap();
        let texte = std::fs::read_to_string(racine.join("dataset_description.json")).unwrap();
        assert!(!texte.contains("Commit") && !texte.contains("CommandLine"));
        let _ = std::fs::remove_dir_all(&racine);
    }
}
