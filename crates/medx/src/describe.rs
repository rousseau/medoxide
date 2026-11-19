//! `medx --describe` : description JSON des commandes que guioxide peut lancer (schéma `oxide-describe/1`,
//! `docs/donnees.md` de guioxide, section 5).
//!
//! Le document est généré, pas rédigé : les paramètres viennent des définitions clap (`Cli::command()`), les entrées et
//! sorties de la table `medoxide_core::contrat` (celle d'où le code d'écriture tire ses noms de fichiers), le reste de
//! constantes de ce module. Rien n'est lu ni écrit : ni poids, ni GPU, ni jeu de données. Le JSON s'écrit avec
//! `medoxide_core::json::Json` (le workspace n'a pas de dépendance JSON).

use std::any::TypeId;

use clap::{Arg, ArgAction, Command, CommandFactory};
use medoxide_core::contrat::{self, Commande};
use medoxide_core::json::Json;

use crate::Cli;

/// Version du schéma du document.
const SCHEMA: &str = "oxide-describe/1";
/// Version du contrat de données de guioxide respectée par l'outil.
const CONTRAT: &str = "1.0";
/// Dépôt du code.
const URL_DU_CODE: &str = "https://github.com/rousseau/medoxide";

/// Arguments BIDS de clap : ils font la ligne de commande BIDS de guioxide, ce ne sont pas des paramètres.
const ARGUMENTS_BIDS: [&str; 4] = ["bids", "participant", "session", "output_dir"];
/// Bornes et unités connues de l'outil : (paramètre, minimum, unité). Rien n'est deviné.
const BORNES: [(&str, Option<i64>, Option<&str>); 3] = [
    ("resolution_mm", Some(0), Some("mm")),
    ("alpha", Some(0), None),
    ("min_stack_coverage", Some(1), None),
];

/// Document complet : `{"schema", "contract", "tool", "commands"}`.
pub fn document() -> Json {
    let cli = Cli::command();
    Json::objet([
        ("schema", Json::texte(SCHEMA)),
        ("contract", Json::texte(CONTRAT)),
        (
            "tool",
            Json::objet([
                ("name", Json::texte("medx")),
                ("version", Json::texte(env!("CARGO_PKG_VERSION"))),
                ("commit", option_env!("MEDX_COMMIT").map_or(Json::Null, Json::texte)),
                ("code_url", Json::texte(URL_DU_CODE)),
            ]),
        ),
        (
            "commands",
            Json::Liste(
                contrat::COMMANDES
                    .iter()
                    .map(|c| commande(c, sous_commande(&cli, c.argv)))
                    .collect(),
            ),
        ),
    ])
}

/// La sous-commande de clap de chemin `argv` (`["svr", "recon"]`).
fn sous_commande<'a>(cli: &'a Command, argv: &[&str]) -> &'a Command {
    argv.iter().fold(cli, |c, mot| {
        c.get_subcommands()
            .find(|s| s.get_name() == *mot)
            .unwrap_or_else(|| panic!("sous-commande inconnue : {mot}"))
    })
}

/// Défaut de clap, converti selon le type du paramètre.
fn defaut(genre: &str, arg: &Arg) -> Json {
    let Some(texte) = arg.get_default_values().first().and_then(|v| v.to_str()) else {
        return if genre == "boolean" {
            Json::Bool(false)
        } else {
            Json::Null
        };
    };
    match genre {
        "number" => texte.parse::<f64>().map_or(Json::Null, Json::Nombre),
        "integer" => texte.parse::<i64>().map_or(Json::Null, Json::Entier),
        "boolean" => Json::Bool(texte == "true"),
        _ => Json::texte(texte),
    }
}

/// Type du schéma d'après l'analyseur de clap.
fn genre(arg: &Arg, a_des_choix: bool) -> &'static str {
    if matches!(arg.get_action(), ArgAction::SetTrue | ArgAction::SetFalse) {
        return "boolean";
    }
    if a_des_choix {
        return "enum";
    }
    let id = arg.get_value_parser().type_id();
    if id == TypeId::of::<f64>() || id == TypeId::of::<f32>() {
        "number"
    } else if [
        TypeId::of::<usize>(),
        TypeId::of::<u64>(),
        TypeId::of::<u32>(),
        TypeId::of::<i64>(),
        TypeId::of::<i32>(),
    ]
    .iter()
    .any(|t| id == *t)
    {
        "integer"
    } else if id == TypeId::of::<std::path::PathBuf>() {
        "path"
    } else {
        "string"
    }
}

/// Paramètre du schéma pour un argument clap.
fn parametre(arg: &Arg, c: &Commande) -> Json {
    let id = arg.get_id().as_str();
    let choix: Vec<Json> = arg
        .get_possible_values()
        .iter()
        .map(|v| Json::texte(v.get_name()))
        .collect();
    let genre = genre(arg, !choix.is_empty());
    let mut m = vec![
        ("id", Json::texte(id)),
        ("flag", Json::texte(format!("--{}", arg.get_long().unwrap_or(id)))),
        ("type", Json::texte(genre)),
    ];
    if !choix.is_empty() {
        m.push(("choices", Json::Liste(choix)));
    }
    if !matches!(arg.get_action(), ArgAction::SetTrue | ArgAction::SetFalse)
        && arg.get_num_args().is_some_and(|n| n.max_values() > 1)
    {
        m.push(("list", Json::Bool(true)));
    }
    m.push(("default", defaut(genre, arg)));
    if let Some((_, min, unite)) = BORNES.iter().find(|(nom, _, _)| *nom == id) {
        if let Some(min) = min {
            m.push(("min", Json::Entier(*min)));
        }
        if let Some(unite) = unite {
            m.push(("unit", Json::texte(*unite)));
        }
    }
    m.push(("optional", Json::Bool(true)));
    let niveau = if c.basiques.contains(&id) { "basic" } else { "advanced" };
    m.push(("level", Json::texte(niveau)));
    m.push((
        "help",
        Json::texte(arg.get_help().map(|h| h.to_string()).unwrap_or_default()),
    ));
    Json::objet(m)
}

/// Motifs et cardinalité d'une entrée de la table.
fn entree(e: &contrat::Entree) -> Json {
    Json::objet([
        ("id", Json::texte(e.id)),
        ("nature", Json::texte(e.nature)),
        ("pipeline", e.pipeline.map_or(Json::Null, Json::texte)),
        ("pattern", Json::texte(e.motif())),
        ("cardinality", Json::texte(e.cardinalite.nom())),
        ("count", Json::texte(if e.plusieurs { "many" } else { "one" })),
        ("required", Json::Bool(e.requise)),
    ])
}

/// Sortie de la table.
fn sortie(s: &contrat::Sortie) -> Json {
    let mut m = vec![
        ("id", Json::texte(s.id)),
        ("nature", Json::texte(s.nature)),
        ("required", Json::Bool(s.requise)),
        ("cardinality", Json::texte(s.cardinalite.nom())),
        ("pattern", Json::texte(s.motif())),
    ];
    if let Some(t) = s.table {
        m.push(("table", Json::texte(t)));
    }
    Json::objet(m)
}

/// Une commande : clap pour la ligne de commande, la table pour le reste.
fn commande(c: &Commande, clap: &Command) -> Json {
    let parametres: Vec<Json> = clap
        .get_arguments()
        .filter(|a| {
            let id = a.get_id().as_str();
            a.get_long().is_some()
                && !ARGUMENTS_BIDS.contains(&id)
                && !c.chemins_libres.contains(&id)
                && id != "help"
                && id != "version"
        })
        .map(|a| parametre(a, c))
        .collect();
    Json::objet([
        ("id", Json::texte(c.id)),
        ("argv", Json::Liste(c.argv.iter().map(|m| Json::texte(*m)).collect())),
        (
            "summary",
            Json::texte(clap.get_about().map(|a| a.to_string()).unwrap_or_default()),
        ),
        ("stage", Json::texte(c.etape)),
        ("pipeline", Json::texte(c.pipeline)),
        ("generated_by", Json::texte(c.genere_par)),
        (
            "bids",
            Json::objet([
                ("mode", Json::texte("supported")),
                ("root", Json::texte("--bids")),
                ("participant", Json::texte("--participant")),
                ("session", Json::texte("--session")),
                ("output_dir", Json::texte("--output-dir")),
            ]),
        ),
        ("inputs", Json::Liste(c.entrees.iter().map(entree).collect())),
        ("outputs", Json::Liste(c.sorties.iter().map(sortie).collect())),
        ("parameters", Json::Liste(parametres)),
        ("presets", Json::Liste(Vec::new())),
        ("default_preset", Json::Null),
        (
            "resources",
            Json::objet([
                ("gpu", Json::texte(c.gpu)),
                ("device_param", c.parametre_calcul.map_or(Json::Null, Json::texte)),
                ("memory_gb", Json::Null),
            ]),
        ),
        ("progress", Json::Null),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use medoxide_core::bids;
    use medoxide_core::contrat::Cardinalite;

    /// Valeur d'une clé d'un objet.
    fn cle<'a>(j: &'a Json, nom: &str) -> &'a Json {
        match j {
            Json::Objet(paires) => paires
                .iter()
                .find(|(k, _)| k == nom)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("clé absente : {nom}")),
            _ => panic!("pas un objet"),
        }
    }

    fn liste(j: &Json) -> &[Json] {
        match j {
            Json::Liste(v) => v,
            _ => panic!("pas une liste"),
        }
    }

    fn texte(j: &Json) -> &str {
        match j {
            Json::Texte(t) => t,
            _ => panic!("pas un texte"),
        }
    }

    fn commandes(d: &Json) -> &[Json] {
        liste(cle(d, "commands"))
    }

    /// Forme du document : version du schéma, les quatre commandes, motifs de la table.
    #[test]
    fn the_document_has_the_four_bids_commands() {
        let d = document();
        assert_eq!(texte(cle(&d, "schema")), "oxide-describe/1");
        assert_eq!(texte(cle(cle(&d, "tool"), "name")), "medx");
        let ids: Vec<&str> = commandes(&d).iter().map(|c| texte(cle(c, "id"))).collect();
        assert_eq!(ids, ["fetalbet", "svr.qc", "svr.recon", "bounti"]);
        let recon = &commandes(&d)[2];
        assert_eq!(texte(cle(recon, "pipeline")), "medx-svr");
        assert_eq!(texte(cle(recon, "generated_by")), "medx svr recon");
        let volume = &liste(cle(recon, "outputs"))[0];
        assert_eq!(
            texte(cle(volume, "pattern")),
            "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_desc-svr_T2w.nii.gz"
        );
        let masque = &liste(cle(&commandes(&d)[0], "outputs"))[0];
        assert_eq!(texte(cle(masque, "cardinality")), "per_acquisition");
    }

    /// Chaque option de chaque commande décrite figure dans le document, hors BIDS et hors mode libre ; défauts de clap.
    #[test]
    fn every_clap_option_is_described_with_its_default() {
        let cli = Cli::command();
        let d = document();
        for (c, json) in contrat::COMMANDES.iter().zip(commandes(&d)) {
            let clap = sous_commande(&cli, c.argv);
            let params = liste(cle(json, "parameters"));
            for a in clap.get_arguments().filter(|a| a.get_long().is_some()) {
                let id = a.get_id().as_str();
                let attendu = !(ARGUMENTS_BIDS.contains(&id)
                    || c.chemins_libres.contains(&id)
                    || id == "help"
                    || id == "version");
                let present = params.iter().any(|p| texte(cle(p, "id")) == id);
                assert_eq!(present, attendu, "{} : {id}", c.id);
            }
            // les identifiants libres de la table existent dans clap (pas de faute de frappe silencieuse)
            for libre in c.chemins_libres {
                assert!(clap.get_arguments().any(|a| a.get_id() == *libre), "{} : {libre}", c.id);
            }
            for b in c.basiques {
                assert!(clap.get_arguments().any(|a| a.get_id() == *b), "{} : {b}", c.id);
            }
            if let Some(p) = c.parametre_calcul {
                assert!(clap.get_arguments().any(|a| a.get_id() == p), "{} : {p}", c.id);
            }
        }
        // Ni les arguments BIDS ni les chemins du mode libre ne sont des paramètres (liste écrite ici, pas la constante).
        let sans = |c: &Json, absents: &[&str]| {
            for p in liste(cle(c, "parameters")) {
                assert!(
                    !absents.contains(&texte(cle(p, "id"))),
                    "{} ne doit pas être un paramètre",
                    texte(cle(p, "id"))
                );
            }
        };
        let bids = ["bids", "participant", "session", "output_dir"];
        sans(&commandes(&d)[0], &[&bids[..], &["input", "output"]].concat());
        sans(&commandes(&d)[1], &[&bids[..], &["input", "mask", "output"]].concat());
        sans(
            &commandes(&d)[2],
            &[&bids[..], &["input", "mask", "output", "quality", "poses"]].concat(),
        );
        sans(
            &commandes(&d)[3],
            &[&bids[..], &["input", "output", "brain_mask"]].concat(),
        );
        let recon = liste(cle(&commandes(&d)[2], "parameters"));
        let par = |id: &str| recon.iter().find(|p| texte(cle(p, "id")) == id).unwrap();
        assert_eq!(texte(cle(par("motion"), "type")), "enum");
        assert_eq!(cle(par("motion"), "default"), &Json::texte("s2s"));
        assert_eq!(texte(cle(par("resolution_mm"), "type")), "number");
        assert_eq!(texte(cle(par("resolution_mm"), "unit")), "mm");
        assert_eq!(texte(cle(par("min_stack_coverage"), "type")), "integer");
        assert_eq!(cle(par("min_stack_coverage"), "min"), &Json::Entier(1));
        assert_eq!(texte(cle(par("s2s_block_parallel_gradient"), "type")), "boolean");
        assert_eq!(cle(par("s2s_block_parallel_gradient"), "default"), &Json::Bool(false));
        assert_eq!(texte(cle(par("motion"), "level")), "basic");
        assert_eq!(texte(cle(par("s2s_seed"), "level")), "advanced");
        assert_eq!(cle(par("device"), "default"), &Json::Null);
    }

    /// Les chemins de la table sont ceux que construisent les fonctions d'écriture existantes (avec et sans session).
    #[test]
    fn the_table_patterns_equal_the_paths_the_writers_build() {
        for session in [Some("A2"), None] {
            let racine = std::path::Path::new("/d/sourcedata");
            let svr = medoxide_svr::sortie::fichiers_bids(racine, None, "01", session).unwrap();
            let bounti = medoxide_bounti::sortie::fichiers_bids(racine, None, "01", session).unwrap();
            let relatif = |chemin: &std::path::Path, pipeline: &str| {
                chemin
                    .strip_prefix(format!("/d/derivatives/{pipeline}"))
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            };
            let attendus = [
                (&contrat::SVR_RECON, "volume", relatif(&svr.volume, bids::PIPELINE_SVR)),
                (
                    &contrat::SVR_RECON,
                    "volume_sidecar",
                    relatif(&svr.description, bids::PIPELINE_SVR),
                ),
                (
                    &contrat::SVR_RECON,
                    "motion",
                    relatif(&svr.mouvement, bids::PIPELINE_SVR),
                ),
                (
                    &contrat::SVR_RECON,
                    "slicequality",
                    relatif(&svr.qualite, bids::PIPELINE_SVR),
                ),
                (
                    &contrat::SVR_QC,
                    "slicequality",
                    relatif(&svr.qualite_acquisition, bids::PIPELINE_SVR),
                ),
                (
                    &contrat::BOUNTI,
                    "labels",
                    relatif(&bounti.etiquettes, bids::PIPELINE_BOUNTI),
                ),
                (
                    &contrat::BOUNTI,
                    "labels_table",
                    relatif(&bounti.table, bids::PIPELINE_BOUNTI),
                ),
                (
                    &contrat::BOUNTI,
                    "mask",
                    relatif(bounti.masque.as_ref().unwrap(), bids::PIPELINE_BOUNTI),
                ),
            ];
            for (commande, id, chemin) in attendus {
                let s = commande.sorties.iter().find(|s| s.id == id).unwrap();
                assert_eq!(
                    resoudre(&s.motif(), "01", session, None),
                    chemin,
                    "{} : {id}",
                    commande.id
                );
            }
            // masque d'une pile : un par acquisition
            let pile = std::path::Path::new("/d/sourcedata/sub-01_ses-A2_acq-ax_run-1_T2w.nii.gz");
            let masque = bids::masque_de_pile(std::path::Path::new("anat"), pile);
            let s = &contrat::FETALBET.sorties[0];
            let motif = resoudre(&s.motif(), "01", Some("A2"), Some(("ax", "1")));
            assert_eq!(
                motif,
                format!("sub-01/ses-A2/anat/{}", masque.file_name().unwrap().to_string_lossy())
            );
        }
    }

    /// Résout un motif pour le participant `sub` : `ses` (ou sans session) et `acq`/`run` éventuels.
    fn resoudre(motif: &str, sub: &str, ses: Option<&str>, acq: Option<(&str, &str)>) -> String {
        let mut s = motif.to_string();
        // Un groupe facultatif est remplacé par son contenu résolu, ou supprimé si sa marque n'a pas de valeur.
        let groupe = |s: String, ouvert: &str, valeur: Option<String>| s.replace(ouvert, &valeur.unwrap_or_default());
        s = groupe(s, "[/ses-{ses}]", ses.map(|v| format!("/ses-{v}")));
        s = groupe(s, "[_ses-{ses}]", ses.map(|v| format!("_ses-{v}")));
        s = groupe(s, "[_acq-{acq}]", acq.map(|(a, _)| format!("_acq-{a}")));
        s = groupe(s, "[_run-{run}]", acq.map(|(_, r)| format!("_run-{r}")));
        s.replace("{sub}", sub)
    }

    /// Le fichier voisin d'une sortie `metadata` a le suffixe de son image : le motif change seulement l'extension.
    #[test]
    fn the_sidecars_share_the_stem_of_their_image() {
        let recon = &contrat::SVR_RECON;
        let volume = recon.sorties.iter().find(|s| s.id == "volume").unwrap();
        let json = recon.sorties.iter().find(|s| s.id == "volume_sidecar").unwrap();
        assert_eq!(
            volume.suffixe.trim_end_matches(".nii.gz"),
            json.suffixe.trim_end_matches(".json")
        );
        assert_eq!(volume.cardinalite, Cardinalite::Session);
    }

    /// Entités `(acq, run)` des piles `*_T2w.nii.gz` de la session du jeu de test.
    fn acquisitions(anat: &std::path::Path) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = std::fs::read_dir(anat)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(contrat::SUFFIXE_PILE))
            .map(|n| {
                let entite = |cle: &str| {
                    n.split('_')
                        .find_map(|m| m.strip_prefix(cle))
                        .unwrap_or_default()
                        .to_string()
                };
                (entite("acq-"), entite("run-"))
            })
            .collect();
        v.sort();
        v
    }

    /// Sur le jeu de test BIDS (`OXIDE_JEU_TEST`, fantôme de guioxide : sorties de toutes les étapes, écrites par les
    /// outils), chaque entrée et chaque sortie `required` déclarée existe, et chaque fichier des dérivées de `medx` est
    /// déclaré. Lancer : `OXIDE_JEU_TEST=<jeu> cargo test --release -p medx -- --ignored`.
    #[test]
    #[ignore = "jeu BIDS de test (OXIDE_JEU_TEST)"]
    fn every_declared_output_exists_in_the_test_dataset_and_every_file_is_declared() {
        let jeu = std::path::PathBuf::from(std::env::var_os("OXIDE_JEU_TEST").expect("OXIDE_JEU_TEST non défini"));
        let sub = std::env::var("OXIDE_JEU_SUB").unwrap_or_else(|_| "fant01".into());
        let ses = std::env::var("OXIDE_JEU_SES").unwrap_or_else(|_| "01".into());
        let acqs = acquisitions(&jeu.join(format!("sub-{sub}/ses-{ses}/anat")));
        assert!(!acqs.is_empty(), "aucune pile dans le jeu");
        let resolu = |motif: &str, cardinalite: Cardinalite| -> Vec<String> {
            match cardinalite {
                Cardinalite::Session => vec![resoudre(motif, &sub, Some(&ses), None)],
                Cardinalite::ParAcquisition => acqs
                    .iter()
                    .map(|(a, r)| resoudre(motif, &sub, Some(&ses), Some((a, r))))
                    .collect(),
            }
        };
        let mut manquants = Vec::new();
        let mut declares: std::collections::BTreeMap<&str, Vec<String>> = Default::default();
        for c in contrat::COMMANDES {
            for e in c.entrees {
                let racine = match e.pipeline {
                    Some(p) => jeu.join("derivatives").join(p),
                    None => jeu.clone(),
                };
                for chemin in resolu(&e.motif(), e.cardinalite) {
                    if !racine.join(&chemin).is_file() {
                        manquants.push(format!("entrée {} {} : {chemin}", c.id, e.id));
                    }
                }
            }
            for s in c.sorties {
                for chemin in resolu(&s.motif(), s.cardinalite) {
                    if s.requise && !jeu.join("derivatives").join(c.pipeline).join(&chemin).is_file() {
                        manquants.push(format!("sortie {} {} : {chemin}", c.id, s.id));
                    }
                    declares.entry(c.pipeline).or_default().push(chemin);
                }
            }
        }
        assert!(manquants.is_empty(), "déclaré mais absent du jeu : {manquants:#?}");
        let mut non_declares = Vec::new();
        for (pipeline, chemins) in &declares {
            let anat = jeu.join(format!("derivatives/{pipeline}/sub-{sub}/ses-{ses}/anat"));
            for e in std::fs::read_dir(&anat).unwrap().filter_map(Result::ok) {
                let relatif = format!("sub-{sub}/ses-{ses}/anat/{}", e.file_name().to_string_lossy());
                if !chemins.contains(&relatif) {
                    non_declares.push(format!("{pipeline} : {relatif}"));
                }
            }
        }
        assert!(
            non_declares.is_empty(),
            "présent dans le jeu mais non déclaré : {non_declares:#?}"
        );
    }
}
