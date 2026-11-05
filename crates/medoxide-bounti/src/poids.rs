//! Emplacement des poids convertis de BOUNTI (trois fichiers `.bpk`, hors du dépôt : publiés sur <https://huggingface.co/rousseau/medoxide-bounti> sous licence GPL v3, ou produits par `scripts/bounti_poids.sh` ; voir `docs/BOUNTI.md`).
//!
//! Le dossier des poids contient `bet/bet_64.bpk`, `unet/unet_64.bpk` et `attunet/attunet_64.bpk`. Il est cherché dans cet ordre : l'option de la ligne de commande, la variable d'environnement `MEDOXIDE_BOUNTI_WEIGHTS`, puis le cache de l'utilisateur
//! (`$XDG_CACHE_HOME/medoxide/bounti`, sinon `~/.cache/medoxide/bounti`). Dans le cache, les poids sont **téléchargés au premier usage** depuis Hugging Face, à une révision figée, vérifiés par SHA-256, et mis à jour
//! automatiquement si ce code attend une autre révision ([`JeuDePoids::dans_le_cache`]). Un dossier donné par l'option ou la variable n'est que vérifié (usage hors ligne). Même mécanisme pour Multi-BOUNTI ([`MULTIBOUNTI`]).
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use medoxide_core::cache::{cache_dir_from, cached_verified};

/// Fichiers de poids attendus dans le dossier, relativement à lui.
pub const FICHIERS: [&str; 3] = ["bet/bet_64.bpk", "unet/unet_64.bpk", "attunet/attunet_64.bpk"];

/// Nom de la variable d'environnement qui donne le dossier des poids.
pub const VARIABLE: &str = "MEDOXIDE_BOUNTI_WEIGHTS";

/// Jeu de poids publié sur Hugging Face et figé sur une révision : où le ranger dans le cache, où le télécharger, et
/// l'empreinte attendue de chaque fichier.
pub struct JeuDePoids {
    /// Nom du jeu, pour les messages.
    pub nom: &'static str,
    /// Dépôt Hugging Face (`propriétaire/nom`).
    pub depot: &'static str,
    /// Révision figée (commit du dépôt) : le contenu à cette adresse ne change jamais.
    pub revision: &'static str,
    /// Sous-dossier du cache de medoxide.
    pub sous_dossier: &'static str,
    /// Variable d'environnement qui désigne un dossier de poids local à la place du cache (usage hors ligne).
    pub variable: &'static str,
    /// Fichiers du jeu : chemin relatif et SHA-256.
    pub fichiers: &'static [(&'static str, &'static str)],
}

impl JeuDePoids {
    /// Adresse d'un fichier du jeu, à la révision figée.
    pub fn adresse(&self, fichier: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{fichier}",
            self.depot, self.revision
        )
    }

    /// Dossier du jeu dans le cache de medoxide, avec tous ses fichiers à jour : téléchargés s'ils manquent, mis à jour
    /// si leur empreinte n'est pas celle de la révision figée (voir `medoxide_core::cache::cached_verified`).
    ///
    /// # Erreurs
    /// [`ErreurPoids::AucunDossier`] sans dossier de cache (ni `XDG_CACHE_HOME` ni `HOME`) ; [`ErreurPoids::Telechargement`]
    /// si un fichier ne peut pas être téléchargé ou vérifié (hors ligne : donner un dossier de poids explicitement).
    pub fn dans_le_cache(&self) -> Result<PathBuf, ErreurPoids> {
        let cache = cache_dir_from(std::env::var_os("XDG_CACHE_HOME"), std::env::var_os("HOME"))
            .ok_or(ErreurPoids::AucunDossier)?;
        let dossier = cache.join(self.sous_dossier);
        for (fichier, empreinte) in self.fichiers {
            cached_verified(&dossier, fichier, &self.adresse(fichier), empreinte).map_err(|e| {
                ErreurPoids::Telechargement(format!(
                    "poids de {} : {e}. Hors ligne : indiquez --weights ou {}",
                    self.nom, self.variable
                ))
            })?;
        }
        Ok(dossier)
    }
}

/// Poids de BOUNTI convertis pour Burn (<https://huggingface.co/rousseau/medoxide-bounti>, licence GPL v3), révision du 20 octobre 2026.
pub const BOUNTI: JeuDePoids = JeuDePoids {
    nom: "BOUNTI",
    depot: "rousseau/medoxide-bounti",
    revision: "e45c39eb8ed65d56f3ed52f75390fc77891fe25c",
    sous_dossier: "bounti",
    variable: VARIABLE,
    fichiers: &[
        (
            "bet/bet_64.bpk",
            "3b288ec287cfa42518f6cd5023d23e7cd5387f06175d769038b73781e0d31fe6",
        ),
        (
            "unet/unet_64.bpk",
            "6ac7d9b793c68e307d89d793a7b1ff66916c631d9524b0157c7f0a25f2ac6ae8",
        ),
        (
            "attunet/attunet_64.bpk",
            "a3c9a4c5ed67997d3ce9b283e309818030f82c3e38cfe160f0351f69e45774fa",
        ),
    ],
};

/// Poids de Multi-BOUNTI convertis pour Burn, et le gabarit gauche/droite (<https://huggingface.co/rousseau/medoxide-multibounti>,
/// licence GPL v3), révision du 22 octobre 2026.
pub const MULTIBOUNTI: JeuDePoids = JeuDePoids {
    nom: "Multi-BOUNTI",
    depot: "rousseau/medoxide-multibounti",
    revision: "38aaccd42094ee3a78479a0ec6fa4e50e8dfbfbc",
    sous_dossier: "multibounti",
    variable: "MEDOXIDE_MULTIBOUNTI_WEIGHTS",
    fichiers: &[
        (
            "bet/bet_64.bpk",
            "8f39a5400e6edd147a8296ee35d8b3b27ad18dfe4a3a3e94af0e69619da1d455",
        ),
        (
            "lr/lr_64.bpk",
            "d9c20fe02c4e52ddcc73af38f9357ab47412a87a33a3d712cd31bf9e451436fd",
        ),
        (
            "multi/multi_64.bpk",
            "747e9ecbe47c0650014850532962398d51afdb24aec9c3c2629a0ab1b57e9344",
        ),
        (
            "gabarit/lr-lab.nii.gz",
            "9daf554bf37f7cb607c8c5ba92de611f94b547dbcec9136bf366396b9f2beb7d",
        ),
    ],
};

/// Poids introuvables.
#[derive(Debug, PartialEq)]
pub enum ErreurPoids {
    /// Aucun dossier n'est déterminable (ni option, ni variable, ni dossier de cache).
    AucunDossier,
    /// Le téléchargement ou la vérification d'un fichier du cache a échoué (message complet).
    Telechargement(String),
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
        let aide = "Sans --weights ni MEDOXIDE_BOUNTI_WEIGHTS, ils sont téléchargés automatiquement dans le cache depuis https://huggingface.co/rousseau/medoxide-bounti ; un dossier donné explicitement doit contenir bet/, unet/ et attunet/ (poids produits aussi par scripts/bounti_poids.sh depuis l'image Docker des auteurs, voir docs/BOUNTI.md). Ils sont sous licence GPL v3 et ne sont pas inclus dans le dépôt de medoxide.";
        match self {
            ErreurPoids::AucunDossier => write!(
                f,
                "dossier des poids de BOUNTI introuvable : indiquez --weights, ou {VARIABLE}, ou HOME. {aide}"
            ),
            ErreurPoids::Telechargement(message) => f.write_str(message),
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

/// Dossier des poids : `option` (ligne de commande), sinon `variable` (valeur de `MEDOXIDE_BOUNTI_WEIGHTS`), sinon le cache (`xdg_cache`, sinon `home/.cache`), puis vérification que les trois fichiers y sont.
/// Les valeurs d'environnement sont passées en paramètres pour que la fonction soit testable sans toucher à l'environnement du processus ; [`dossier_des_poids`] les lit.
///
/// # Erreurs
/// [`ErreurPoids::AucunDossier`] si rien ne désigne un dossier ; [`ErreurPoids::Manquants`] s'il manque un fichier. Une valeur vide pour la variable ou pour `xdg_cache` est ignorée.
pub fn choisir_dossier(
    option: Option<&Path>,
    variable: Option<OsString>,
    xdg_cache: Option<OsString>,
    home: Option<OsString>,
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
    let manquants: Vec<&'static str> = FICHIERS
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

/// Dossier des poids de BOUNTI : l'option ou la variable `MEDOXIDE_BOUNTI_WEIGHTS` si l'une est donnée (dossier vérifié
/// par [`choisir_dossier`]), sinon le cache, où les poids sont téléchargés ou mis à jour au besoin ([`BOUNTI`]).
///
/// # Erreurs
/// Voir [`choisir_dossier`] et [`JeuDePoids::dans_le_cache`].
pub fn dossier_des_poids(option: Option<&Path>) -> Result<PathBuf, ErreurPoids> {
    let variable = std::env::var_os(VARIABLE).filter(|v| !v.is_empty());
    if option.is_none() && variable.is_none() {
        return BOUNTI.dans_le_cache();
    }
    choisir_dossier(option, variable, None, None)
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
        assert_eq!(choisir_dossier(Some(&a), os(&b), os(&c), os(&d)).unwrap(), a);
        assert_eq!(choisir_dossier(None, os(&b), os(&c), os(&d)).unwrap(), b);
        assert_eq!(
            choisir_dossier(None, None, os(&c), os(&d)).unwrap(),
            c.join("medoxide/bounti")
        );
        assert_eq!(
            choisir_dossier(None, None, None, os(&d)).unwrap(),
            d.join(".cache/medoxide/bounti")
        );
        // une valeur vide est ignorée
        assert_eq!(
            choisir_dossier(None, Some("".into()), Some("".into()), os(&d)).unwrap(),
            d.join(".cache/medoxide/bounti")
        );
        // un dossier prioritaire incomplet n'est pas contourné
        let incomplet = dossier_avec("incomplet", &FICHIERS[..2]);
        assert!(matches!(
            choisir_dossier(Some(&incomplet), os(&b), None, None),
            Err(ErreurPoids::Manquants { .. })
        ));
    }

    /// Les jeux téléchargés ont exactement les fichiers attendus par le chargement des réseaux (BOUNTI : [`FICHIERS`] ;
    /// Multi-BOUNTI : `segmentation_multi::FICHIERS`), avec des empreintes SHA-256 bien formées.
    #[test]
    fn the_downloaded_sets_match_the_files_the_networks_load() {
        let chemins = |jeu: &JeuDePoids| jeu.fichiers.iter().map(|(f, _)| *f).collect::<Vec<_>>();
        assert_eq!(chemins(&BOUNTI), FICHIERS);
        assert_eq!(chemins(&MULTIBOUNTI), crate::segmentation_multi::FICHIERS);
        for (_, empreinte) in BOUNTI.fichiers.iter().chain(MULTIBOUNTI.fichiers) {
            assert!(
                empreinte.len() == 64
                    && empreinte
                        .chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            );
        }
        assert_eq!(
            BOUNTI.adresse("bet/bet_64.bpk"),
            "https://huggingface.co/rousseau/medoxide-bounti/resolve/e45c39eb8ed65d56f3ed52f75390fc77891fe25c/bet/bet_64.bpk"
        );
    }

    /// Sans option, sans variable et sans dossier de cache : `AucunDossier`. Avec un dossier incomplet : la liste exacte des fichiers manquants, et un message qui dit où chercher la procédure.
    #[test]
    fn missing_weights_are_reported_with_the_exact_files() {
        assert_eq!(choisir_dossier(None, None, None, None), Err(ErreurPoids::AucunDossier));
        let incomplet = dossier_avec("manque", &["bet/bet_64.bpk"]);
        let erreur = choisir_dossier(Some(&incomplet), None, None, None).unwrap_err();
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
}
