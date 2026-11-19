//! Table des entrées et sorties BIDS des commandes de `medx` : source unique des noms de fichiers.
//!
//! Le code d'écriture ([`crate::bids::masque_de_pile`], `medoxide_svr::sortie`, `medoxide_bounti::sortie`) tire ses
//! suffixes des constantes de ce module ; `medx --describe` (schéma `oxide-describe/1`, `docs/donnees.md` de guioxide,
//! section 5) en tire ses motifs de chemin. Un nom ne s'écrit donc qu'ici.
//!
//! Un nom de fichier est un **préfixe** (`sub-X[_ses-Y]`, ou les entités d'une pile pour une sortie par
//! acquisition) suivi d'un **suffixe** : les constantes `SUFFIXE_*`.

/// Suffixe du masque cérébral d'une pile (`medx fetalbet`).
pub const SUFFIXE_MASQUE_PILE: &str = "_desc-brain_mask.nii.gz";
/// Suffixe d'une pile acquise : entrée de `medx fetalbet`, `svr qc` et `svr recon`.
pub const SUFFIXE_PILE: &str = "_T2w.nii.gz";
/// Suffixe du volume reconstruit (`medx svr recon`).
pub const SUFFIXE_SVR_VOLUME: &str = "_desc-svr_T2w.nii.gz";
/// Suffixe du fichier voisin du volume reconstruit (réglages, piles utilisées).
pub const SUFFIXE_SVR_VOLUME_JSON: &str = "_desc-svr_T2w.json";
/// Suffixe du mouvement de chaque coupe.
pub const SUFFIXE_SVR_MOUVEMENT: &str = "_desc-svr_motion.tsv";
/// Suffixe de la qualité de chaque coupe après reconstruction.
pub const SUFFIXE_SVR_QUALITE: &str = "_desc-svr_slicequality.tsv";
/// Suffixe du contrôle qualité des acquisitions (`medx svr qc`).
pub const SUFFIXE_ACQ_QUALITE: &str = "_desc-acq_slicequality.tsv";
/// Suffixe du fichier voisin du contrôle qualité des acquisitions.
pub const SUFFIXE_ACQ_QUALITE_JSON: &str = "_desc-acq_slicequality.json";
/// Suffixe des étiquettes de la segmentation (`medx bounti`).
pub const SUFFIXE_BOUNTI_ETIQUETTES: &str = "_desc-bounti_dseg.nii.gz";
/// Suffixe de la table des étiquettes (`medx bounti`).
pub const SUFFIXE_BOUNTI_TABLE: &str = "_desc-bounti_dseg.tsv";
/// Suffixe du masque du cerveau (`medx bounti`).
pub const SUFFIXE_BOUNTI_MASQUE: &str = "_desc-bounti_mask.nii.gz";

/// Début des motifs de chemin d'une sortie par session, relatif à la racine du pipeline.
pub const PREFIXE_SESSION: &str = "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]";
/// Début des motifs de chemin d'une entrée ou d'une sortie par acquisition.
pub const PREFIXE_ACQUISITION: &str = "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}][_acq-{acq}][_run-{run}]";

/// Combien de fichiers par session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cardinalite {
    /// Un fichier par session.
    Session,
    /// Un fichier par acquisition brute de la session (`{acq}` et `{run}` prennent ceux de l'acquisition).
    ParAcquisition,
}

impl Cardinalite {
    /// Nom dans le document `--describe`.
    pub fn nom(self) -> &'static str {
        match self {
            Cardinalite::Session => "session",
            Cardinalite::ParAcquisition => "per_acquisition",
        }
    }

    fn prefixe(self) -> &'static str {
        match self {
            Cardinalite::Session => PREFIXE_SESSION,
            Cardinalite::ParAcquisition => PREFIXE_ACQUISITION,
        }
    }
}

/// Une sortie d'une commande.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sortie {
    /// Identifiant stable dans la commande.
    pub id: &'static str,
    /// Nature (`volume`, `mask`, `labels`, `table`, `metadata`…), qui décide de l'affichage dans guioxide.
    pub nature: &'static str,
    /// Suffixe du nom de fichier.
    pub suffixe: &'static str,
    /// Écrite à chaque exécution réussie.
    pub requise: bool,
    /// Un fichier par session ou par acquisition.
    pub cardinalite: Cardinalite,
    /// Pour des étiquettes : identifiant de la sortie `table` qui les nomme.
    pub table: Option<&'static str>,
}

impl Sortie {
    /// Motif de chemin relatif à la racine du pipeline (marques `{sub}`, `{ses}`, `{acq}`, `{run}`).
    pub fn motif(&self) -> String {
        format!("{}{}", self.cardinalite.prefixe(), self.suffixe)
    }
}

/// Une entrée d'une commande.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entree {
    /// Identifiant stable dans la commande.
    pub id: &'static str,
    /// Nature.
    pub nature: &'static str,
    /// Dossier de dérivées d'où vient l'entrée, `None` pour les données brutes de la racine BIDS.
    pub pipeline: Option<&'static str>,
    /// Suffixe du nom de fichier.
    pub suffixe: &'static str,
    /// Un fichier par session ou par acquisition.
    pub cardinalite: Cardinalite,
    /// Tous les fichiers qui correspondent (`true`) ou un seul.
    pub plusieurs: bool,
    /// Indispensable.
    pub requise: bool,
}

impl Entree {
    /// Motif de chemin relatif à la racine du pipeline d'origine (ou de la racine BIDS).
    pub fn motif(&self) -> String {
        format!("{}{}", self.cardinalite.prefixe(), self.suffixe)
    }
}

/// Ce que `--describe` sait d'une commande en dehors de clap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Commande {
    /// Identifiant stable (`svr.recon`).
    pub id: &'static str,
    /// Mots après le binaire (`["svr", "recon"]`).
    pub argv: &'static [&'static str],
    /// Étape générique nourrie.
    pub etape: &'static str,
    /// Dossier de dérivées écrit.
    pub pipeline: &'static str,
    /// `Name` écrit dans `GeneratedBy`.
    pub genere_par: &'static str,
    /// Entrées.
    pub entrees: &'static [Entree],
    /// Sorties.
    pub sorties: &'static [Sortie],
    /// Ressource GPU : `none`, `optional` ou `required`.
    pub gpu: &'static str,
    /// Identifiant du paramètre qui choisit le calcul, s'il y en a un.
    pub parametre_calcul: Option<&'static str>,
    /// Paramètres que clap ne range pas parmi les options (chemins du mode libre) : exclus du document.
    pub chemins_libres: &'static [&'static str],
    /// Paramètres affichés d'emblée (`basic`).
    pub basiques: &'static [&'static str],
}

const fn sortie(
    id: &'static str,
    nature: &'static str,
    suffixe: &'static str,
    cardinalite: Cardinalite,
    table: Option<&'static str>,
) -> Sortie {
    Sortie {
        id,
        nature,
        suffixe,
        requise: true,
        cardinalite,
        table,
    }
}

const PILES: Entree = Entree {
    id: "stacks",
    nature: "volume",
    pipeline: None,
    suffixe: SUFFIXE_PILE,
    cardinalite: Cardinalite::ParAcquisition,
    plusieurs: true,
    requise: true,
};

const MASQUES: Entree = Entree {
    id: "masks",
    nature: "mask",
    pipeline: Some(crate::bids::PIPELINE_FETALBET),
    suffixe: SUFFIXE_MASQUE_PILE,
    cardinalite: Cardinalite::ParAcquisition,
    plusieurs: true,
    requise: true,
};

/// `medx fetalbet` : un masque cérébral par pile.
pub const FETALBET: Commande = Commande {
    id: "fetalbet",
    argv: &["fetalbet"],
    etape: "segmentation",
    pipeline: crate::bids::PIPELINE_FETALBET,
    genere_par: "medx fetalbet",
    entrees: &[PILES],
    sorties: &[sortie(
        "mask",
        "mask",
        SUFFIXE_MASQUE_PILE,
        Cardinalite::ParAcquisition,
        None,
    )],
    gpu: "optional",
    parametre_calcul: None,
    chemins_libres: &["input", "output"],
    basiques: &[],
};

/// `medx svr qc` : contrôle qualité des acquisitions, avant reconstruction.
pub const SVR_QC: Commande = Commande {
    id: "svr.qc",
    argv: &["svr", "qc"],
    etape: "acquisition",
    pipeline: crate::bids::PIPELINE_SVR,
    genere_par: "medx svr qc",
    entrees: &[PILES, MASQUES],
    sorties: &[
        sortie("slicequality", "table", SUFFIXE_ACQ_QUALITE, Cardinalite::Session, None),
        sortie(
            "slicequality_sidecar",
            "metadata",
            SUFFIXE_ACQ_QUALITE_JSON,
            Cardinalite::Session,
            None,
        ),
    ],
    gpu: "none",
    parametre_calcul: None,
    chemins_libres: &["input", "mask", "output"],
    basiques: &[],
};

/// `medx svr recon` : reconstruction d'un volume à partir des piles.
pub const SVR_RECON: Commande = Commande {
    id: "svr.recon",
    argv: &["svr", "recon"],
    etape: "reconstruction",
    pipeline: crate::bids::PIPELINE_SVR,
    genere_par: "medx svr recon",
    entrees: &[PILES, MASQUES],
    sorties: &[
        sortie("volume", "volume", SUFFIXE_SVR_VOLUME, Cardinalite::Session, None),
        sortie(
            "volume_sidecar",
            "metadata",
            SUFFIXE_SVR_VOLUME_JSON,
            Cardinalite::Session,
            None,
        ),
        sortie("motion", "table", SUFFIXE_SVR_MOUVEMENT, Cardinalite::Session, None),
        sortie("slicequality", "table", SUFFIXE_SVR_QUALITE, Cardinalite::Session, None),
    ],
    gpu: "optional",
    parametre_calcul: Some("device"),
    chemins_libres: &["input", "mask", "output", "quality", "poses"],
    basiques: &["motion", "resolution_mm", "alpha"],
};

/// `medx bounti` : segmentation en 19 tissus du volume reconstruit.
pub const BOUNTI: Commande = Commande {
    id: "bounti",
    argv: &["bounti"],
    etape: "segmentation",
    pipeline: crate::bids::PIPELINE_BOUNTI,
    genere_par: "medx bounti",
    entrees: &[Entree {
        id: "volume",
        nature: "volume",
        pipeline: Some(crate::bids::PIPELINE_SVR),
        suffixe: SUFFIXE_SVR_VOLUME,
        cardinalite: Cardinalite::Session,
        plusieurs: false,
        requise: true,
    }],
    sorties: &[
        sortie(
            "labels",
            "labels",
            SUFFIXE_BOUNTI_ETIQUETTES,
            Cardinalite::Session,
            Some("labels_table"),
        ),
        sortie(
            "labels_table",
            "table",
            SUFFIXE_BOUNTI_TABLE,
            Cardinalite::Session,
            None,
        ),
        sortie("mask", "mask", SUFFIXE_BOUNTI_MASQUE, Cardinalite::Session, None),
    ],
    gpu: "optional",
    parametre_calcul: Some("device"),
    chemins_libres: &["input", "output", "brain_mask"],
    basiques: &[],
};

/// Les commandes décrites, dans l'ordre du document.
pub const COMMANDES: [&Commande; 4] = [&FETALBET, &SVR_QC, &SVR_RECON, &BOUNTI];

#[cfg(test)]
mod tests {
    use super::*;

    /// Les identifiants de sortie sont uniques dans une commande, et la table d'une sortie d'étiquettes existe.
    #[test]
    fn the_output_ids_are_unique_and_label_tables_exist() {
        for c in COMMANDES {
            for (i, s) in c.sorties.iter().enumerate() {
                assert!(c.sorties[i + 1..].iter().all(|t| t.id != s.id), "{} : {}", c.id, s.id);
                if let Some(t) = s.table {
                    assert!(
                        c.sorties.iter().any(|u| u.id == t && u.nature == "table"),
                        "{} : {t}",
                        c.id
                    );
                }
            }
        }
    }

    /// Les motifs, tels que le document les donne : entités de la session, ou de l'acquisition.
    #[test]
    fn the_patterns_follow_the_contract() {
        assert_eq!(
            FETALBET.sorties[0].motif(),
            "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}][_acq-{acq}][_run-{run}]_desc-brain_mask.nii.gz"
        );
        assert_eq!(
            SVR_RECON.sorties[0].motif(),
            "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_desc-svr_T2w.nii.gz"
        );
        assert_eq!(
            BOUNTI.entrees[0].motif(),
            "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_desc-svr_T2w.nii.gz"
        );
    }

    /// Les noms d'avant la table n'ont pas changé.
    #[test]
    fn the_file_names_are_unchanged() {
        assert_eq!(SUFFIXE_MASQUE_PILE, "_desc-brain_mask.nii.gz");
        assert_eq!(SUFFIXE_SVR_VOLUME, "_desc-svr_T2w.nii.gz");
        assert_eq!(SUFFIXE_SVR_VOLUME_JSON, "_desc-svr_T2w.json");
        assert_eq!(SUFFIXE_SVR_MOUVEMENT, "_desc-svr_motion.tsv");
        assert_eq!(SUFFIXE_SVR_QUALITE, "_desc-svr_slicequality.tsv");
        assert_eq!(SUFFIXE_ACQ_QUALITE, "_desc-acq_slicequality.tsv");
        assert_eq!(SUFFIXE_BOUNTI_ETIQUETTES, "_desc-bounti_dseg.nii.gz");
        assert_eq!(SUFFIXE_BOUNTI_TABLE, "_desc-bounti_dseg.tsv");
        assert_eq!(SUFFIXE_BOUNTI_MASQUE, "_desc-bounti_mask.nii.gz");
    }
}
