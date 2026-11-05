//! Dossier de cache de medoxide et téléchargement vérifié des fichiers volumineux (poids des modèles).
//!
//! Partagé par les modules qui téléchargent leurs poids au premier usage (`medoxide-fetalbet`, `medoxide-bounti`) :
//! un seul dossier de cache, une seule façon de télécharger et de vérifier un fichier. [`cached_verified`] garde à côté
//! de chaque fichier son empreinte vérifiée (`<nom>.sha256`) : un fichier dont l'empreinte attendue a changé (nouvelle
//! révision des poids) est retéléchargé automatiquement.
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
        let obtenu = hexadecimal(&hachage.finalize());
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

/// Empreinte SHA-256 en hexadécimal minuscule : chaque octet s'écrit sur 2 chiffres.
fn hexadecimal(octets: &[u8]) -> String {
    octets.iter().map(|o| format!("{o:02x}")).collect()
}

/// SHA-256 d'un fichier, lu par blocs (les poids pèsent jusqu'à une centaine de mégaoctets).
fn empreinte_fichier(chemin: &Path) -> std::io::Result<String> {
    let mut fichier = File::open(chemin)?;
    let mut hachage = Sha256::new();
    let mut tampon = vec![0u8; 64 * 1024];
    loop {
        let n = fichier.read(&mut tampon)?;
        if n == 0 {
            break;
        }
        hachage.update(&tampon[..n]);
    }
    Ok(hexadecimal(&hachage.finalize()))
}

/// Fichier `dossier/nom` du cache, garanti identique à l'empreinte attendue : utilisé tel quel s'il est à jour, sinon
/// (absent, ou d'une autre révision) téléchargé depuis `url` avec [`download_verified`].
///
/// L'empreinte vérifiée est gardée dans `<nom>.sha256`, à côté du fichier : quand elle est celle qu'attend le code,
/// le fichier n'est pas relu. Sans ce fichier (cache antérieur), l'empreinte est calculée une fois, puis gardée. Une
/// empreinte différente (le code attend une nouvelle révision des poids) provoque un nouveau téléchargement : la mise
/// à jour des poids est automatique. Les sous-dossiers de `nom` (par exemple `bet/bet_64.bpk`) sont créés au besoin.
/// Messages de téléchargement sur stderr.
///
/// # Erreurs
/// [`DownloadError`] si le dossier ne peut pas être créé, si le fichier présent est illisible, ou si le téléchargement
/// échoue (hors ligne, par exemple) : le fichier présent n'est alors pas modifié.
pub fn cached_verified(dossier: &Path, nom: &str, url: &str, expected_sha256: &str) -> Result<PathBuf, DownloadError> {
    let chemin = dossier.join(nom);
    let marque = dossier.join(format!("{nom}.sha256"));
    if chemin.is_file() {
        let gardee = std::fs::read_to_string(&marque).unwrap_or_default();
        if gardee.trim() == expected_sha256 {
            return Ok(chemin);
        }
        let actuelle = empreinte_fichier(&chemin).map_err(|e| DownloadError(format!("{} : {e}", chemin.display())))?;
        if actuelle == expected_sha256 {
            // Le marqueur est une optimisation : s'il ne peut pas être écrit, le fichier reste bon.
            let _ = std::fs::write(&marque, expected_sha256);
            return Ok(chemin);
        }
        eprintln!("Mise à jour de {nom} (nouvelle révision des poids).");
    }
    if let Some(parent) = chemin.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DownloadError(format!("{} : {e}", parent.display())))?;
    }
    eprintln!("Téléchargement de {nom} vers {}\n  depuis {url}", chemin.display());
    download_verified(url, &chemin, expected_sha256)?;
    let _ = std::fs::write(&marque, expected_sha256);
    eprintln!("{nom} téléchargé et vérifié.");
    Ok(chemin)
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

    /// Dossier temporaire vide propre à un test.
    fn dossier_vide(nom: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("medoxide_cache_{}_{nom}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Adresse injoignable : un test qui tenterait un téléchargement échoue aussitôt.
    const INJOIGNABLE: &str = "http://127.0.0.1:9/poids.bpk";

    /// Un fichier présent et juste, sans marqueur (cache antérieur), est utilisé sans téléchargement ; son empreinte
    /// est alors gardée dans `<nom>.sha256`.
    #[test]
    fn a_correct_file_is_kept_and_its_checksum_recorded() {
        let d = dossier_vide("juste");
        std::fs::create_dir_all(d.join("bet")).unwrap();
        std::fs::write(d.join("bet/poids.bpk"), b"poids").unwrap();
        let attendue = hexadecimal(&Sha256::digest(b"poids"));
        let chemin = cached_verified(&d, "bet/poids.bpk", INJOIGNABLE, &attendue).unwrap();
        assert_eq!(chemin, d.join("bet/poids.bpk"));
        assert_eq!(
            std::fs::read_to_string(d.join("bet/poids.bpk.sha256")).unwrap(),
            attendue
        );
        // Second appel : le marqueur suffit.
        assert!(cached_verified(&d, "bet/poids.bpk", INJOIGNABLE, &attendue).is_ok());
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Un fichier d'une autre révision (empreinte différente) déclenche un téléchargement ; s'il échoue, l'erreur est
    /// rendue et le fichier présent n'est pas modifié.
    #[test]
    fn a_stale_file_triggers_a_download_and_survives_its_failure() {
        let d = dossier_vide("perime");
        std::fs::write(d.join("poids.bpk"), b"ancienne revision").unwrap();
        std::fs::write(
            d.join("poids.bpk.sha256"),
            hexadecimal(&Sha256::digest(b"ancienne revision")),
        )
        .unwrap();
        let attendue = hexadecimal(&Sha256::digest(b"nouvelle revision"));
        let r = cached_verified(&d, "poids.bpk", INJOIGNABLE, &attendue);
        assert!(matches!(r, Err(DownloadError(_))), "{r:?}");
        assert_eq!(std::fs::read(d.join("poids.bpk")).unwrap(), b"ancienne revision");
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Fichier absent : téléchargement tenté, les sous-dossiers sont créés.
    #[test]
    fn a_missing_file_is_downloaded_into_its_subfolder() {
        let d = dossier_vide("absent");
        let r = cached_verified(&d, "unet/unet.bpk", INJOIGNABLE, &"0".repeat(64));
        assert!(matches!(r, Err(DownloadError(_))), "{r:?}");
        assert!(d.join("unet").is_dir());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
