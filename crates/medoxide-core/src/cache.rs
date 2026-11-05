//! Dossier de cache de medoxide et téléchargement vérifié des fichiers volumineux (poids des modèles).
//!
//! Partagé par les modules qui téléchargent leurs poids au premier usage (`medoxide-fetalbet`, et bientôt
//! `medoxide-bounti`) : un seul dossier de cache, une seule façon de télécharger et de vérifier un fichier.
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

/// Échec d'un téléchargement vérifié : message lisible, avec l'adresse ou le fichier en cause.
#[derive(Debug, PartialEq)]
pub struct DownloadError(pub String);

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DownloadError {}

/// Dossier de cache de medoxide : `$XDG_CACHE_HOME/medoxide`, sinon
/// `$HOME/.cache/medoxide`. `None` si aucune de ces variables n'est définie.
/// Les valeurs sont passées en paramètres pour pouvoir tester sans toucher à
/// l'environnement du processus.
pub fn cache_dir_from(xdg_cache_home: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    let base = match xdg_cache_home.filter(|v| !v.is_empty()) {
        Some(xdg) => PathBuf::from(xdg),
        // `?` fonctionne aussi sur un `Option` : si `home` est vide ou absent, la
        // fonction renvoie `None` immédiatement.
        None => PathBuf::from(home.filter(|v| !v.is_empty())?).join(".cache"),
    };
    Some(base.join("medoxide"))
}

/// Télécharge `url` vers `dest` et vérifie son SHA-256.
///
/// Le contenu est écrit par blocs dans un fichier temporaire voisin (`.part`),
/// haché au fur et à mesure, puis renommé en `dest` seulement si l'empreinte est
/// la bonne : un téléchargement interrompu ou corrompu ne laisse jamais un faux
/// fichier de poids. Une progression par tranche de 10 % est écrite sur stderr.
///
/// # Erreurs
/// [`DownloadError`] si la requête, la lecture, l'écriture ou la vérification échoue ; le
/// fichier `.part` est alors supprimé.
pub fn download_verified(url: &str, dest: &Path, expected_sha256: &str) -> Result<(), DownloadError> {
    let echec = |contexte: &str, e: &dyn std::fmt::Display| DownloadError(format!("{contexte} : {e}"));
    let partiel = dest.with_extension("part");

    let resultat = (|| {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_global(Some(Duration::from_secs(30 * 60)))
            .build()
            .into();
        let reponse = agent.get(url).call().map_err(|e| echec(url, &e))?;
        let total = reponse.body().content_length();
        let mut lecteur = reponse.into_body().into_reader();

        let mut fichier = File::create(&partiel).map_err(|e| echec(&partiel.display().to_string(), &e))?;
        let mut hachage = Sha256::new();
        let mut tampon = vec![0u8; 64 * 1024];
        let (mut recu, mut dernier_pourcent) = (0u64, 0u64);
        loop {
            let n = lecteur.read(&mut tampon).map_err(|e| echec("lecture réseau", &e))?;
            if n == 0 {
                break;
            }
            hachage.update(&tampon[..n]);
            fichier.write_all(&tampon[..n]).map_err(|e| echec("écriture", &e))?;
            recu += n as u64;
            if let Some(total) = total.filter(|&t| t > 0) {
                let pourcent = recu * 100 / total / 10 * 10;
                if pourcent > dernier_pourcent {
                    dernier_pourcent = pourcent;
                    eprintln!("  {pourcent} %");
                }
            }
        }
        fichier.flush().map_err(|e| echec("écriture", &e))?;

        // Le résultat est un tableau d'octets : chacun s'écrit sur 2 chiffres hexadécimaux.
        let obtenu: String = hachage.finalize().iter().map(|o| format!("{o:02x}")).collect();
        if obtenu != expected_sha256 {
            return Err(DownloadError(format!(
                "SHA-256 inattendu (obtenu {obtenu}, attendu {expected_sha256})"
            )));
        }
        std::fs::rename(&partiel, dest).map_err(|e| echec(&dest.display().to_string(), &e))
    })();

    if resultat.is_err() {
        let _ = std::fs::remove_file(&partiel); // nettoyage ; l'erreur d'origine prime
    }
    resultat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_prefers_xdg_then_home() {
        let os = |v: &str| Some(OsString::from(v));
        assert_eq!(
            cache_dir_from(os("/xdg"), os("/home/u")),
            Some(PathBuf::from("/xdg/medoxide"))
        );
        assert_eq!(
            cache_dir_from(None, os("/home/u")),
            Some(PathBuf::from("/home/u/.cache/medoxide"))
        );
        // Variables vides : traitées comme absentes.
        assert_eq!(
            cache_dir_from(os(""), os("/home/u")),
            Some(PathBuf::from("/home/u/.cache/medoxide"))
        );
        assert_eq!(cache_dir_from(None, None), None);
        assert_eq!(cache_dir_from(os(""), os("")), None);
    }

    /// Réseau : une empreinte fausse est refusée et ne laisse aucun fichier.
    /// Lancer avec `cargo test -p medoxide-core -- --ignored download_rejects`.
    #[test]
    #[ignore = "nécessite le réseau"]
    fn download_rejects_wrong_checksum_and_cleans_up() {
        let dossier = std::env::temp_dir().join(format!("medoxide_dl_{}", std::process::id()));
        std::fs::create_dir_all(&dossier).unwrap();
        let dest = dossier.join("fiche.md");
        // Petit fichier du dépôt Hugging Face (la fiche), avec une empreinte volontairement fausse.
        let url = "https://huggingface.co/rousseau/medoxide-fetalbet/resolve/main/README.md";
        let r = download_verified(url, &dest, &"0".repeat(64));
        let reste = std::fs::read_dir(&dossier).unwrap().count();
        std::fs::remove_dir_all(&dossier).unwrap();
        assert!(matches!(r, Err(DownloadError(ref m)) if m.contains("SHA-256")), "{r:?}");
        assert_eq!(reste, 0, "fichier résiduel après un échec");
    }
}
