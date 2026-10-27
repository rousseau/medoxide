//! Sorties BIDS de `medx svr recon`, dans l'arborescence dérivée que lit guioxide (`docs/donnees.md` de guioxide) :
//! `<étude>/derivatives/medx-svr/sub-X[/ses-Y]/anat/sub-X[_ses-Y]_desc-svr_T2w.nii.gz` (volume reconstruit), `…_desc-svr_T2w.json` (réglages, piles utilisées), `…_desc-svr_motion.tsv` (mouvement de chaque coupe),
//! `…_desc-svr_slicequality.tsv` (qualité de chaque coupe), avec `derivatives/medx-svr/dataset_description.json`.
//!
//! Les piles et leurs masques cérébraux (produits par `medx fetalbet`, dans `derivatives/medx-fetalbet`) se déduisent des mêmes règles de nommage ([`piles_et_masques`]). `<étude>` est la racine de l'étude de [`medoxide_core::bids::racine_etude`].

use std::path::{Path, PathBuf};

use medoxide_core::bids::{self, Generateur};
use medoxide_core::json::Json;

use crate::pipeline::{Motion, ReconstructionConfig, ReconstructionOutput};
use crate::{Stack, SvrError};

/// Chemins des fichiers à écrire.
#[derive(Debug, PartialEq)]
pub struct Fichiers {
    /// Volume reconstruit.
    pub volume: PathBuf,
    /// Fichier voisin du volume : réglages, piles utilisées.
    pub description: PathBuf,
    /// Mouvement de chaque coupe.
    pub mouvement: PathBuf,
    /// Qualité de chaque coupe.
    pub qualite: PathBuf,
    /// Contrôle qualité des acquisitions, avant reconstruction (voir [`crate::qc`]).
    pub qualite_acquisition: PathBuf,
    /// Racine de l'arborescence dérivée (`<étude>/derivatives/medx-svr`) : elle reçoit `dataset_description.json`.
    pub derive: PathBuf,
}

/// Chemins BIDS des résultats d'un participant (et d'une session), dans l'étude de [`medoxide_core::bids::racine_etude`]`(racine_bids, sortie)`.
///
/// # Erreurs
/// [`SvrError::Bids`] si le participant ou la session n'est pas alphanumérique.
pub fn fichiers_bids(
    racine_bids: &Path,
    sortie: Option<&Path>,
    participant: &str,
    session: Option<&str>,
) -> Result<Fichiers, SvrError> {
    let d = bids::dossier_derive(racine_bids, sortie, bids::PIPELINE_SVR, participant, session)?;
    let nom = |suffixe: &str| d.anat.join(format!("{}_desc-svr_{suffixe}", d.prefixe));
    Ok(Fichiers {
        volume: nom("T2w.nii.gz"),
        description: nom("T2w.json"),
        mouvement: nom("motion.tsv"),
        qualite: nom("slicequality.tsv"),
        qualite_acquisition: d.anat.join(format!("{}_desc-acq_slicequality.tsv", d.prefixe)),
        derive: d.pipeline,
    })
}

/// Piles à reconstruire et leurs masques, dans l'ordre : `entrees` si elle n'est pas vide (restriction de la liste), sinon les piles `*_T2w.nii.gz` du participant (et de la session) dans la racine BIDS.
/// Le masque d'une pile est celui de `medx fetalbet` dans l'étude de `sortie` ([`medoxide_core::bids::masque_de_pile`]) ; son existence n'est pas vérifiée ici (la lecture du masque nomme le fichier manquant).
///
/// # Erreurs
/// [`SvrError::Bids`] : identifiant invalide, dossier illisible, aucune pile.
pub fn piles_et_masques(
    racine_bids: &Path,
    sortie: Option<&Path>,
    participant: &str,
    session: Option<&str>,
    entrees: &[PathBuf],
) -> Result<Vec<(PathBuf, PathBuf)>, SvrError> {
    let piles = if entrees.is_empty() {
        bids::piles_bids(racine_bids, participant, session)?
    } else {
        entrees.to_vec()
    };
    let masques = bids::dossier_derive(racine_bids, sortie, bids::PIPELINE_FETALBET, participant, session)?.anat;
    Ok(piles
        .into_iter()
        .map(|pile| {
            let masque = bids::masque_de_pile(&masques, &pile);
            (pile, masque)
        })
        .collect())
}

/// Nom d'un mode de mouvement dans les fichiers (celui de l'option `--motion`).
fn nom_du_mouvement(motion: Motion) -> &'static str {
    match motion {
        Motion::SliceToSlice => "s2s",
        Motion::SliceToVolume => "s2v",
    }
}

/// Réglages de la reconstruction, tous les champs de [`ReconstructionConfig`] sous leur nom (un test vérifie qu'aucun ne manque).
fn reglages(c: &ReconstructionConfig) -> Json {
    let reel = Json::Nombre;
    let entier = |n: usize| Json::Entier(n as i64);
    Json::objet([
        ("motion", Json::texte(nom_du_mouvement(c.motion))),
        ("resolution_mm", reel(c.resolution_mm)),
        ("margin_mm", reel(c.margin_mm)),
        ("alpha", reel(c.alpha)),
        ("min_stack_coverage", entier(c.min_stack_coverage)),
        ("edge_kappa", reel(c.edge_kappa)),
        ("edge_outer", entier(c.edge_outer)),
        ("cg_max_iterations", entier(c.cg_max_iterations)),
        ("cg_tolerance", reel(c.cg_tolerance)),
        ("s2s_sweeps", entier(c.s2s_sweeps)),
        ("s2s_seed", Json::Entier(c.s2s_seed as i64)),
        ("s2s_block_parallel_gradient", Json::Bool(c.s2s_block_parallel_gradient)),
        (
            "s2s_update",
            Json::texte(match c.s2s_update {
                crate::s2s::S2sUpdate::GaussSeidel => "gauss-seidel",
                crate::s2s::S2sUpdate::Jacobi => "jacobi",
            }),
        ),
        ("s2s_coarse_sigma_pixels", reel(c.s2s_coarse_sigma_pixels)),
        ("s2s_coarse_sweeps", entier(c.s2s_coarse_sweeps)),
        ("s2s_coarse_levels", entier(c.s2s_coarse_levels)),
        ("s2s_highpass_mm", reel(c.s2s_highpass_mm)),
        ("s2s_block_levels", entier(c.s2s_block_levels)),
        ("s2s_block_sweeps", entier(c.s2s_block_sweeps)),
        ("s2s_block_search_mm", reel(c.s2s_block_search_mm)),
        ("s2s_bias_rounds", entier(c.s2s_bias_rounds)),
        ("s2s_bias_min_reduction", reel(c.s2s_bias_min_reduction)),
        ("s2s_temporal_weight", reel(c.s2s_temporal_weight)),
        ("s2s_temporal_robust_mm", reel(c.s2s_temporal_robust_mm)),
        ("s2s_temporal_robust_start_mm", reel(c.s2s_temporal_robust_start_mm)),
        ("s2s_temporal_info_exponent", reel(c.s2s_temporal_info_exponent)),
        ("s2s_anchor_stack", c.s2s_anchor_stack.map_or(Json::Null, entier)),
        ("s2s_exclude_fraction", reel(c.s2s_exclude_fraction)),
        ("s2v_refine_cycles", entier(c.s2v_refine_cycles)),
        ("s2v_refine_resolution_mm", reel(c.s2v_refine_resolution_mm)),
        ("s2v_cycles", entier(c.s2v_cycles)),
        ("s2v_extra_starts", entier(c.s2v_extra_starts)),
        ("s2v_start_amplitude_mm", reel(c.s2v_start_amplitude_mm)),
        ("s2v_suspect_ncc", reel(c.s2v_suspect_ncc)),
        ("s2v_leave_slice_out", Json::Bool(c.s2v_leave_slice_out)),
        ("s2v_batched", Json::Bool(c.s2v_batched)),
        (
            "s2v_init",
            Json::texte(match c.s2v_init {
                crate::pipeline::S2vInit::Header => "header",
                crate::pipeline::S2vInit::Stacks => "stacks",
                crate::pipeline::S2vInit::StacksBias => "stacks-bias",
            }),
        ),
        (
            "s2v_skip_unused_reconstructions",
            Json::Bool(c.s2v_skip_unused_reconstructions),
        ),
    ])
}

/// Sens des matrices de `…_desc-svr_motion.tsv`, pour un lecteur qui n'a pas le code : voir [`ReconstructionOutput::write_poses`].
fn convention_du_mouvement() -> Json {
    Json::objet([
        ("File", Json::texte("*_desc-svr_motion.tsv")),
        (
            "Matrix",
            Json::texte(
                "M : mouvement rigide de la coupe (matrice 4x4, mm), absolu par rapport à la géométrie d'en-tête de sa pile et composé dans le repère monde (RAS+, mm)",
            ),
        ),
        (
            "SliceToWorld",
            Json::texte(
                "coupe → monde = M · A · T(0, 0, k), où A est l'affine voxel → monde de l'en-tête de la pile et T(0, 0, k) la translation de k voxels selon la 3e colonne de A (k : indice de la coupe)",
            ),
        ),
        (
            "ColumnOrder",
            Json::texte(
                "colonnes m00 m10 m20 m30 m01 … m33 : m<ligne><colonne>, matrice rangée colonne par colonne (numpy : reshape(4, 4, order=\"F\"))",
            ),
        ),
        (
            "Stack",
            Json::texte("nom du fichier de la pile sans extension, tel que listé dans Stacks"),
        ),
    ])
}

/// Contenu du fichier voisin du volume : résolution, mode de mouvement, réglages, piles utilisées (noms sans extension) et ligne de commande.
fn description_du_volume(stacks: &[Stack], config: &ReconstructionConfig, generateur: &Generateur) -> Json {
    let mut champs = vec![
        (
            "Description",
            Json::texte("Volume T2 reconstruit à partir de piles 2D par reconstruction coupe-vers-volume (SVR)"),
        ),
        ("ResolutionMm", Json::Nombre(config.resolution_mm)),
        ("MotionMode", Json::texte(nom_du_mouvement(config.motion))),
        (
            "Stacks",
            Json::Liste(
                stacks
                    .iter()
                    .map(|s| Json::texte(bids::nom_de_pile(s.path())))
                    .collect(),
            ),
        ),
        ("Settings", reglages(config)),
        ("MotionConvention", convention_du_mouvement()),
    ];
    if let Some(ligne) = &generateur.ligne_de_commande {
        champs.push(("CommandLine", Json::texte(ligne)));
    }
    Json::objet(champs)
}

/// Écrit les résultats aux chemins `fichiers` (dossiers créés au besoin) : volume, fichier voisin du volume, mouvement, qualité par coupe, et `dataset_description.json` (décrit par `generateur`). `stacks` sont les piles de `sortie`, dans le même ordre ; `config` est la configuration effectivement utilisée.
///
/// # Erreurs
/// [`SvrError::Write`] si un fichier ne peut pas être écrit, [`SvrError::PoseMismatch`] si `stacks` n'a pas la forme des poses.
pub fn ecrire(
    fichiers: &Fichiers,
    sortie: &ReconstructionOutput,
    stacks: &[Stack],
    config: &ReconstructionConfig,
    generateur: &Generateur,
) -> Result<(), SvrError> {
    bids::creer_dossier_parent(&fichiers.volume)?;
    sortie.write_volume(&fichiers.volume)?;
    sortie.write_poses(stacks, &fichiers.mouvement)?;
    sortie.write_quality(stacks, &fichiers.qualite)?;
    bids::ecrire_texte(
        &fichiers.description,
        &description_du_volume(stacks, config, generateur).en_texte(),
    )?;
    bids::ecrire_description_derivee(&fichiers.derive, bids::PIPELINE_SVR, generateur)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chemins BIDS avec session, sans session, depuis `sourcedata/` et avec `--output-dir` ; identifiants invalides refusés.
    #[test]
    fn the_bids_paths_follow_the_derivative_layout() {
        let f = fichiers_bids(Path::new("/d"), None, "01", Some("A2")).unwrap();
        let dossier = "/d/derivatives/medx-svr/sub-01/ses-A2/anat";
        assert_eq!(
            f.volume,
            Path::new(&format!("{dossier}/sub-01_ses-A2_desc-svr_T2w.nii.gz"))
        );
        assert_eq!(
            f.description,
            Path::new(&format!("{dossier}/sub-01_ses-A2_desc-svr_T2w.json"))
        );
        assert_eq!(
            f.mouvement,
            Path::new(&format!("{dossier}/sub-01_ses-A2_desc-svr_motion.tsv"))
        );
        assert_eq!(
            f.qualite,
            Path::new(&format!("{dossier}/sub-01_ses-A2_desc-svr_slicequality.tsv"))
        );
        assert_eq!(f.derive, Path::new("/d/derivatives/medx-svr"));
        let sans = fichiers_bids(Path::new("/d"), None, "01", None).unwrap();
        assert_eq!(
            sans.volume,
            Path::new("/d/derivatives/medx-svr/sub-01/anat/sub-01_desc-svr_T2w.nii.gz")
        );
        let garage = fichiers_bids(Path::new("/g/x/sourcedata"), None, "01", None).unwrap();
        assert_eq!(garage.derive, Path::new("/g/x/derivatives/medx-svr"));
        let libre = fichiers_bids(Path::new("/g/x/sourcedata"), Some(Path::new("/o")), "01", None).unwrap();
        assert_eq!(libre.derive, Path::new("/o/derivatives/medx-svr"));
        for mauvais in ["", "sub-01", "../x", "a/b"] {
            assert!(
                fichiers_bids(Path::new("/d"), None, mauvais, None).is_err(),
                "{mauvais:?}"
            );
            assert!(
                fichiers_bids(Path::new("/d"), None, "01", Some(mauvais)).is_err(),
                "{mauvais:?}"
            );
        }
    }

    /// Les masques sont ceux de `medx fetalbet` (même étude, même nom que lui) ; `--input` restreint la liste des piles sans changer la règle des masques ; sans `--input`, toutes les piles `T2w` de la session.
    #[test]
    fn the_masks_are_those_of_fetalbet_and_input_restricts_the_stacks() {
        let racine = std::env::temp_dir().join(format!("medoxide_svr_piles_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&racine);
        let bids_ = racine.join("sourcedata");
        let anat = bids_.join("sub-01/ses-1/anat");
        std::fs::create_dir_all(&anat).unwrap();
        for nom in ["sub-01_ses-1_acq-ax_T2w.nii.gz", "sub-01_ses-1_acq-sag_T2w.nii.gz"] {
            std::fs::write(anat.join(nom), b"").unwrap();
        }
        let masques = racine.join("derivatives/medx-fetalbet/sub-01/ses-1/anat");
        let toutes = piles_et_masques(&bids_, None, "01", Some("1"), &[]).unwrap();
        assert_eq!(toutes.len(), 2);
        assert_eq!(toutes[0].0, anat.join("sub-01_ses-1_acq-ax_T2w.nii.gz"));
        assert_eq!(toutes[0].1, masques.join("sub-01_ses-1_acq-ax_desc-brain_mask.nii.gz"));
        assert_eq!(toutes[1].1, masques.join("sub-01_ses-1_acq-sag_desc-brain_mask.nii.gz"));
        let une = piles_et_masques(&bids_, None, "01", Some("1"), &[toutes[1].0.clone()]).unwrap();
        assert_eq!(une, vec![toutes[1].clone()]);
        let ailleurs = piles_et_masques(&bids_, Some(Path::new("/o")), "01", Some("1"), &[]).unwrap();
        assert_eq!(
            ailleurs[0].1,
            Path::new("/o/derivatives/medx-fetalbet/sub-01/ses-1/anat/sub-01_ses-1_acq-ax_desc-brain_mask.nii.gz")
        );
        assert!(matches!(
            piles_et_masques(&bids_, None, "02", None, &[]),
            Err(SvrError::Bids(_))
        ));
        let _ = std::fs::remove_dir_all(&racine);
    }

    /// Les réglages écrits contiennent tous les champs de la configuration (un champ ajouté à `ReconstructionConfig` et oublié dans `reglages` fait échouer ce test) ; les valeurs par défaut s'y lisent.
    #[test]
    fn the_settings_list_every_field_of_the_configuration() {
        let config = ReconstructionConfig::default();
        let affichage = format!("{config:#?}");
        let champs: Vec<&str> = affichage
            .lines()
            .filter_map(|l| l.strip_prefix("    "))
            .filter(|l| !l.starts_with(' '))
            .filter_map(|l| l.split_once(':').map(|(nom, _)| nom))
            .collect();
        assert!(champs.len() > 30, "champs lus : {champs:?}");
        let Json::Objet(ecrits) = reglages(&config) else {
            panic!("un objet est attendu");
        };
        let ecrits: Vec<&str> = ecrits.iter().map(|(cle, _)| cle.as_str()).collect();
        for champ in &champs {
            assert!(ecrits.contains(champ), "réglage « {champ} » absent du fichier voisin");
        }
        assert_eq!(ecrits.len(), champs.len(), "réglages écrits : {ecrits:?}");
        let texte = reglages(&config).en_texte();
        assert!(texte.contains("\"motion\": \"s2s\"") && texte.contains("\"resolution_mm\": 0.8"));
    }
}
