//! Emplacement des poids convertis de BOUNTI (trois fichiers `.bpk`, hors du dépôt : voir `docs/BOUNTI.md` pour leur licence et leur production).
//!
//! Le dossier des poids contient `bet/bet_64.bpk`, `unet/unet_64.bpk` et `attunet/attunet_64.bpk`. Il est cherché dans cet ordre : l'option de la ligne de commande, la variable d'environnement `MEDOXIDE_BOUNTI_WEIGHTS`, puis le cache de l'utilisateur
//! (`$XDG_CACHE_HOME/medoxide/bounti`, sinon `~/.cache/medoxide/bounti`). Rien n'est téléchargé : la licence des poids n'est pas tranchée.
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Fichiers de poids attendus dans le dossier, relativement à lui : le réseau d'extraction du cerveau d'abord, puis les deux réseaux de tissus.
pub const FICHIERS: [&str; 3] = ["bet/bet_64.bpk", "unet/unet_64.bpk", "attunet/attunet_64.bpk"];

/// Fichiers exigés : les trois avec l'extraction du cerveau de BOUNTI (`avec_extraction`), les deux réseaux de tissus seulement si le masque vient de Fetal-BET.
pub fn requis(avec_extraction: bool) -> &'static [&'static str] {
    if avec_extraction {
        &FICHIERS
    } else {
        &FICHIERS[1..]
    }
}

/// Nom de la variable d'environnement qui donne le dossier des poids.
pub const VARIABLE: &str = "MEDOXIDE_BOUNTI_WEIGHTS";

/// Poids introuvables.
#[derive(Debug, PartialEq)]
pub enum ErreurPoids {
    /// Aucun dossier n'est déterminable (ni option, ni variable, ni dossier de cache).
    AucunDossier,
    /// Le dossier existe mais il manque des fichiers (chemins relatifs).
    Manquants {
        /// Dossier cherché.
        dossier: PathBuf,
        /// Fichiers absents.
        fichiers: Vec<&'static str>,
    },
}

impl std::fmt::Display for ErreurPoids {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let aide = "Ils se produisent depuis l'image Docker de BOUNTI avec scripts/bounti_poids.sh (voir docs/BOUNTI.md) ; ils sont sous licence GPL v3 et ne sont pas distribués avec medoxide.";
        match self {
            ErreurPoids::AucunDossier => write!(
                f,
                "dossier des poids de BOUNTI introuvable : indiquez --weights, ou {VARIABLE}, ou HOME. {aide}"
            ),
            ErreurPoids::Manquants { dossier, fichiers } => write!(
                f,
                "poids de BOUNTI manquants dans {} : {}. {aide}",
                dossier.display(),
                fichiers.join(", ")
            ),
        }
    }
}

impl std::error::Error for ErreurPoids {}

/// Dossier des poids (avec `avec_extraction` : le fichier du réseau d'extraction du cerveau est aussi exigé, voir [`requis`]) : `option` (ligne de commande), sinon `variable` (valeur de `MEDOXIDE_BOUNTI_WEIGHTS`), sinon le cache (`xdg_cache`, sinon `home/.cache`), puis vérification que les trois fichiers y sont.
/// Les valeurs d'environnement sont passées en paramètres pour que la fonction soit testable sans toucher à l'environnement du processus ; [`dossier_des_poids`] les lit.
///
/// # Erreurs
/// [`ErreurPoids::AucunDossier`] si rien ne désigne un dossier ; [`ErreurPoids::Manquants`] s'il manque un fichier. Une valeur vide pour la variable ou pour `xdg_cache` est ignorée.
pub fn choisir_dossier(
    option: Option<&Path>,
    variable: Option<OsString>,
    xdg_cache: Option<OsString>,
    home: Option<OsString>,
    avec_extraction: bool,
) -> Result<PathBuf, ErreurPoids> {
    let non_vide = |v: Option<OsString>| v.filter(|v| !v.is_empty());
    let dossier = if let Some(o) = option {
        o.to_path_buf()
    } else if let Some(v) = non_vide(variable) {
        PathBuf::from(v)
    } else if let Some(x) = non_vide(xdg_cache) {
        PathBuf::from(x).join("medoxide/bounti")
    } else if let Some(h) = non_vide(home) {
        PathBuf::from(h).join(".cache/medoxide/bounti")
    } else {
        return Err(ErreurPoids::AucunDossier);
    };
    let manquants: Vec<&'static str> = requis(avec_extraction)
        .iter()
        .copied()
        .filter(|f| !dossier.join(f).is_file())
        .collect();
    if manquants.is_empty() {
        Ok(dossier)
    } else {
        Err(ErreurPoids::Manquants {
            dossier,
            fichiers: manquants,
        })
    }
}

/// Comme [`choisir_dossier`], avec les variables d'environnement du processus.
///
/// # Erreurs
/// Voir [`choisir_dossier`].
pub fn dossier_des_poids(option: Option<&Path>, avec_extraction: bool) -> Result<PathBuf, ErreurPoids> {
    choisir_dossier(
        option,
        std::env::var_os(VARIABLE),
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
        avec_extraction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dossier temporaire propre à ce test, avec les fichiers demandés (vides).
    fn dossier_avec(nom: &str, fichiers: &[&str]) -> PathBuf {
        let dossier = std::env::temp_dir().join(format!("medoxide_poids_{}_{nom}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dossier);
        for f in fichiers {
            let chemin = dossier.join(f);
            std::fs::create_dir_all(chemin.parent().unwrap()).unwrap();
            std::fs::write(chemin, b"").unwrap();
        }
        std::fs::create_dir_all(&dossier).unwrap();
        dossier
    }

    /// L'option l'emporte sur la variable, qui l'emporte sur `XDG_CACHE_HOME`, qui l'emporte sur `HOME` ; chacun est choisi seulement si ses fichiers y sont (un dossier prioritaire incomplet n'est PAS remplacé par le suivant).
    #[test]
    fn the_folder_is_chosen_in_priority_order() {
        let (a, b, c, d) = (
            dossier_avec("option", &FICHIERS),
            dossier_avec("variable", &FICHIERS),
            dossier_avec("xdg", &[]),
            dossier_avec("home", &[]),
        );
        for (dossier, fichier) in [(&c, "medoxide/bounti"), (&d, ".cache/medoxide/bounti")] {
            for f in FICHIERS {
                let chemin = dossier.join(fichier).join(f);
                std::fs::create_dir_all(chemin.parent().unwrap()).unwrap();
                std::fs::write(chemin, b"").unwrap();
            }
        }
        let os = |p: &PathBuf| Some(p.clone().into_os_string());
        assert_eq!(choisir_dossier(Some(&a), os(&b), os(&c), os(&d), true).unwrap(), a);
        assert_eq!(choisir_dossier(None, os(&b), os(&c), os(&d), true).unwrap(), b);
        assert_eq!(
            choisir_dossier(None, None, os(&c), os(&d), true).unwrap(),
            c.join("medoxide/bounti")
        );
        assert_eq!(
            choisir_dossier(None, None, None, os(&d), true).unwrap(),
            d.join(".cache/medoxide/bounti")
        );
        // une valeur vide est ignorée
        assert_eq!(
            choisir_dossier(None, Some("".into()), Some("".into()), os(&d), true).unwrap(),
            d.join(".cache/medoxide/bounti")
        );
        // un dossier prioritaire incomplet n'est pas contourné
        let incomplet = dossier_avec("incomplet", &FICHIERS[..2]);
        assert!(matches!(
            choisir_dossier(Some(&incomplet), os(&b), None, None, true),
            Err(ErreurPoids::Manquants { .. })
        ));
    }

    /// Sans option, sans variable et sans dossier de cache : `AucunDossier`. Avec un dossier incomplet : la liste exacte des fichiers manquants, et un message qui dit où chercher la procédure.
    #[test]
    fn missing_weights_are_reported_with_the_exact_files() {
        assert_eq!(
            choisir_dossier(None, None, None, None, true),
            Err(ErreurPoids::AucunDossier)
        );
        let incomplet = dossier_avec("manque", &["bet/bet_64.bpk"]);
        let erreur = choisir_dossier(Some(&incomplet), None, None, None, true).unwrap_err();
        assert_eq!(
            erreur,
            ErreurPoids::Manquants {
                dossier: incomplet.clone(),
                fichiers: vec!["unet/unet_64.bpk", "attunet/attunet_64.bpk"]
            }
        );
        let message = erreur.to_string();
        assert!(message.contains("unet/unet_64.bpk") && message.contains("bounti_poids.sh") && message.contains("GPL"));
        assert!(!message.contains("bet/bet_64.bpk"));
    }

    /// Sans l'extraction de BOUNTI (masque de Fetal-BET), seuls les deux réseaux de tissus sont exigés : un dossier sans `bet/bet_64.bpk` est accepté, et les fichiers manquants listés sont ceux des tissus ; avec l'extraction, le même dossier est refusé en nommant `bet/bet_64.bpk`.
    #[test]
    fn the_extraction_weights_are_required_only_with_the_bounti_extraction() {
        assert_eq!(requis(true), &FICHIERS[..]);
        assert_eq!(requis(false), &["unet/unet_64.bpk", "attunet/attunet_64.bpk"]);
        let sans_bet = dossier_avec("sans_bet", &FICHIERS[1..]);
        assert_eq!(
            choisir_dossier(Some(&sans_bet), None, None, None, false).unwrap(),
            sans_bet
        );
        assert_eq!(
            choisir_dossier(Some(&sans_bet), None, None, None, true),
            Err(ErreurPoids::Manquants {
                dossier: sans_bet.clone(),
                fichiers: vec!["bet/bet_64.bpk"]
            })
        );
        let sans_unet = dossier_avec("sans_unet", &["bet/bet_64.bpk", "attunet/attunet_64.bpk"]);
        assert_eq!(
            choisir_dossier(Some(&sans_unet), None, None, None, false),
            Err(ErreurPoids::Manquants {
                dossier: sans_unet.clone(),
                fichiers: vec!["unet/unet_64.bpk"]
            })
        );
    }
}
