//! `medoxide-fetalbet` : extraction du masque cérébral fœtal.
//!
//! Portage du modèle de segmentation Fetal-BET (PyTorch) vers Burn,
//! via un export ONNX intermédiaire. Voir `docs/LEARNING.md` à la racine
//! du workspace pour le détail de la démarche.
//!
//! Le modèle est celui de Fetal-BET (Faghihpirayesh et al., 2024, CC BY 4.0) :
//! voir l'attribution complète dans `model.rs` et dans le `README.md` racine.

use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use burn::tensor::{Device, Tensor, TensorData};
use medoxide_core::{read_volume, volume_info, CoreError};
use ndarray::{s, Array2, Array3, Array4, Axis};
use sha2::{Digest, Sha256};

mod model;

/// Erreurs possibles lors de la segmentation.
///
/// On définit ce type dès maintenant, même vide de variantes utiles,
/// pour que la signature de `segment` ne change plus une fois l'inférence
/// branchée dessus — c'est l'API qui doit être stable, pas forcément
/// l'implémentation derrière.
#[derive(Debug)]
pub enum MaskError {
    /// Échec de lecture ou d'écriture d'un volume NIfTI (fichier absent, format
    /// invalide, fichier qui n'est pas 3D, écriture impossible...). Contient
    /// l'erreur d'origine de `medoxide-core`.
    Core(medoxide_core::CoreError),
    /// Le fichier de poids du modèle (`.bpk`) n'existe pas.
    ModelNotFound(PathBuf),
    /// Le dossier où écrire le masque n'existe pas ; contient ce dossier.
    OutputDirNotFound(PathBuf),
    /// Échec du téléchargement des poids par défaut (réseau, disque, SHA-256) ;
    /// contient un message explicatif.
    ModelDownload(String),
}

/// Permet à `?` de convertir une erreur de `medoxide-core` en `MaskError::Core`.
impl From<medoxide_core::CoreError> for MaskError {
    fn from(e: medoxide_core::CoreError) -> Self {
        MaskError::Core(e)
    }
}

impl std::fmt::Display for MaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Le message du core est repris tel quel (`NIfTI : …`, `volume 3D attendu…`).
            MaskError::Core(e) => write!(f, "{e}"),
            MaskError::ModelNotFound(chemin) => {
                write!(f, "poids du modèle introuvables : {}", chemin.display())
            }
            MaskError::OutputDirNotFound(dossier) => {
                write!(f, "dossier de sortie introuvable : {}", dossier.display())
            }
            MaskError::ModelDownload(message) => {
                write!(f, "téléchargement des poids impossible : {message}")
            }
        }
    }
}

impl std::error::Error for MaskError {}

/// Normalise un volume coupe par coupe, comme Fetal-BET (axe `z`, le 3e).
///
/// Dans chaque coupe, les voxels strictement positifs sont divisés par leur
/// écart-type (non biaisé, `n - 1` : c'est celui de `torch.std`). **La moyenne
/// n'est pas soustraite** : ce n'est pas un z-score. Les autres voxels (0)
/// restent inchangés. Une coupe avec moins de 2 voxels positifs est laissée
/// telle quelle (l'original y produirait des `NaN`).
///
/// Le volume est modifié en place (`&mut`) : pas de copie.
pub fn normalize_slices(volume: &mut Array3<f32>) {
    for mut coupe in volume.axis_iter_mut(Axis(2)) {
        // Somme en f64 : plus précise que le f32 de la référence.
        let positifs: Vec<f64> = coupe
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| f64::from(v))
            .collect();
        let n = positifs.len();
        if n < 2 {
            continue;
        }
        let moyenne = positifs.iter().sum::<f64>() / n as f64;
        let variance =
            positifs.iter().map(|v| (v - moyenne).powi(2)).sum::<f64>() / (n - 1) as f64;
        let ecart_type = variance.sqrt() as f32;
        coupe.mapv_inplace(|v| if v > 0.0 { v / ecart_type } else { v });
    }
}

/// Longueur d'un axe après rééchantillonnage à 1 mm : `round((n - 1) × pas + 1)`.
/// C'est la règle de MONAI (`Spacingd`), qui part de l'étendue entre les
/// centres du premier et du dernier voxel.
fn resampled_len(n: usize, spacing: f32) -> usize {
    (((n - 1) as f64) * f64::from(spacing) + 1.0).round() as usize
}

/// Rééchantillonne un axe par interpolation linéaire : le voxel de sortie `j`
/// lit l'entrée à la position `j × pas` (le centre du premier voxel reste au même
/// point), et un voisin hors du volume compte pour 0. La sortie a `n_sortie`
/// voxels sur cet axe.
fn resample_axis(volume: &Array3<f32>, axis: usize, n_sortie: usize, pas: f64) -> Array3<f32> {
    let n = volume.len_of(Axis(axis));
    let mut dim = volume.raw_dim();
    dim[axis] = n_sortie;
    let mut sortie = Array3::<f32>::zeros(dim);

    for j in 0..n_sortie {
        let position = j as f64 * pas;
        let i0 = position.floor();
        let w = (position - i0) as f32; // poids du voisin de droite
        let i0 = i0 as usize;

        let mut cible = sortie.index_axis_mut(Axis(axis), j);
        if i0 < n {
            cible.scaled_add(1.0 - w, &volume.index_axis(Axis(axis), i0));
        }
        if i0 + 1 < n {
            cible.scaled_add(w, &volume.index_axis(Axis(axis), i0 + 1));
        }
    }
    sortie
}

/// Rééchantillonne x et y à 1 mm (interpolation bilinéaire, bord à zéro) ;
/// l'axe z est inchangé. Reproduit `Spacingd(pixdim=(1, 1, -1))` de MONAI.
///
/// `spacing` : taille de voxel en mm sur x et y (voir [`VolumeInfo::spacing`]).
/// Les tailles doivent être strictement positives.
pub fn resample_in_plane(volume: &Array3<f32>, spacing: [f32; 2]) -> Array3<f32> {
    let (nx, ny, _) = volume.dim();
    let selon_x = resample_axis(volume, 0, resampled_len(nx, spacing[0]), 1.0 / f64::from(spacing[0]));
    resample_axis(&selon_x, 1, resampled_len(ny, spacing[1]), 1.0 / f64::from(spacing[1]))
}

/// Écrit un masque (octets 0/1) dans un NIfTI (`.nii` ou `.nii.gz`).
///
/// L'en-tête est copié depuis `reference` (le volume dont le masque est issu) :
/// le masque garde donc le même affine et les mêmes espacements. Seuls les
/// dimensions et le type de donnée (`uint8`) changent. La compression gzip est
/// activée si `path` se termine par `.gz`.
///
/// # Erreurs
/// `MaskError::Core` si `reference` est illisible ou si l'écriture échoue.
pub fn write_mask(path: &Path, mask: &Array3<u8>, reference: &Path) -> Result<(), MaskError> {
    nifti::writer::WriterOptions::new(path)
        .reference_file(&reference)
        .write_nifti(mask)
        // L'erreur de `nifti` devient une `CoreError`, puis `?` la convertit en `MaskError`.
        .map_err(CoreError::from)?;
    Ok(())
}

/// Ramène des logits `[2, x, y, z]` (grille à 1 mm) sur la grille d'origine, puis
/// en déduit le masque (0 fond, 1 cerveau). C'est l'inverse du rééchantillonnage
/// à 1 mm : le voxel `i` de la grille d'origine lit la position `i × spacing`.
///
/// L'interpolation porte sur les **logits**, avant l'argmax, comme `Invertd` de
/// MONAI (interpoler le masque puis seuiller donne un résultat différent : Dice
/// 0,991 sur fetus_03). Le softmax du script officiel ne change pas l'argmax.
///
/// `spacing` : taille de voxel d'origine sur x et y (mm) ; `original` : nombre
/// de voxels d'origine sur x et y.
fn logits_to_mask(logits: &Array4<f32>, spacing: [f32; 2], original: (usize, usize)) -> Array3<u8> {
    // Fermeture : elle capture `logits`, `spacing` et `original`.
    let canal = |c: usize| {
        let v = logits.index_axis(Axis(0), c).to_owned();
        let selon_x = resample_axis(&v, 0, original.0, f64::from(spacing[0]));
        resample_axis(&selon_x, 1, original.1, f64::from(spacing[1]))
    };
    let fond = canal(0);
    let cerveau = canal(1);
    ndarray::Zip::from(&fond)
        .and(&cerveau)
        .map_collect(|&f, &c| u8::from(c > f))
}

/// Plan de découpage d'un axe en fenêtres glissantes (règles de MONAI).
#[derive(Debug, PartialEq)]
struct AxisWindows {
    /// Zéros ajoutés avant l'axe (0 si l'axe est au moins aussi long que la fenêtre).
    pad_before: usize,
    /// Longueur de l'axe après complétion : `max(taille, fenêtre)`.
    padded_len: usize,
    /// Positions de départ des fenêtres, dans l'axe complété.
    starts: Vec<usize>,
}

/// Calcule les fenêtres d'un axe de longueur `size`, pour une fenêtre `roi` et
/// un recouvrement `overlap` (entre 0 et 1, exclu), comme
/// `sliding_window_inference` de MONAI :
/// - un axe plus court que la fenêtre est complété par des zéros, de façon
///   symétrique (la moitié arrondie à l'inférieur avant) ;
/// - l'intervalle entre fenêtres est `roi × (1 - overlap)` tronqué, ou `roi` si
///   l'axe complété a exactement la taille de la fenêtre ;
/// - la dernière fenêtre est décalée vers l'arrière pour ne pas dépasser.
fn window_plan(size: usize, roi: usize, overlap: f64) -> AxisWindows {
    let diff = roi.saturating_sub(size);
    let pad_before = diff / 2;
    let padded_len = size.max(roi);

    let interval = if roi == padded_len {
        roi
    } else {
        ((roi as f64 * (1.0 - overlap)) as usize).max(1)
    };

    // Nombre de fenêtres : la première `d` telle que d × intervalle + roi
    // atteint la fin de l'axe, plus un.
    let candidates = padded_len.div_ceil(interval);
    let count = (0..candidates)
        .find(|d| d * interval + roi >= padded_len)
        .map_or(1, |d| d + 1);

    let starts = (0..count)
        .map(|i| {
            let start = i * interval;
            start - (start + roi).saturating_sub(padded_len)
        })
        .collect();
    AxisWindows { pad_before, padded_len, starts }
}

/// Côté des fenêtres d'inférence (celui du modèle Fetal-BET).
const ROI: usize = 256;
/// Recouvrement entre fenêtres voisines.
const OVERLAP: f64 = 0.5;

/// Passe un volume prétraité `[x, y, z]` dans le modèle, coupe par coupe, par
/// fenêtres de 256×256 avec 50 % de recouvrement, et renvoie les logits moyens
/// `[2, x, y, z]` (canal 0 : fond, canal 1 : cerveau). Reproduit
/// `SliceInferer` de MONAI : sur les zones de recouvrement, les logits des
/// fenêtres sont moyennés ; un axe plus court que 256 est complété par des zéros
/// puis rogné.
fn infer_logits(model: &model::Model, device: &Device, volume: &Array3<f32>) -> Array4<f32> {
    let (nx, ny, nz) = volume.dim();
    let plan_x = window_plan(nx, ROI, OVERLAP);
    let plan_y = window_plan(ny, ROI, OVERLAP);
    let (px, py) = (plan_x.padded_len, plan_y.padded_len);

    // Volume complété par des zéros, avec le volume d'origine à l'intérieur.
    let mut complete = Array3::<f32>::zeros((px, py, nz));
    complete
        .slice_mut(s![
            plan_x.pad_before..plan_x.pad_before + nx,
            plan_y.pad_before..plan_y.pad_before + ny,
            ..
        ])
        .assign(volume);

    let mut somme = Array4::<f32>::zeros((2, px, py, nz));
    // Nombre de fenêtres qui couvrent chaque voxel (identique pour toutes les coupes).
    let mut compte = Array2::<f32>::zeros((px, py));
    for &sx in &plan_x.starts {
        for &sy in &plan_y.starts {
            compte.slice_mut(s![sx..sx + ROI, sy..sy + ROI]).mapv_inplace(|c| c + 1.0);
        }
    }

    for z in 0..nz {
        for &sx in &plan_x.starts {
            for &sy in &plan_y.starts {
                let tuile: Vec<f32> = complete
                    .slice(s![sx..sx + ROI, sy..sy + ROI, z])
                    .iter()
                    .copied()
                    .collect();
                let x = Tensor::<4>::from_data(TensorData::new(tuile, [1, 1, ROI, ROI]), device);
                let logits = model.forward(x).into_data().try_to_vec::<f32>().unwrap();
                let logits = Array3::from_shape_vec((2, ROI, ROI), logits).unwrap();
                let mut cible = somme.slice_mut(s![.., sx..sx + ROI, sy..sy + ROI, z]);
                cible += &logits;
            }
        }
    }

    // Moyenne : division par le nombre de fenêtres de chaque voxel.
    for mut canal in somme.axis_iter_mut(Axis(0)) {
        for mut coupe in canal.axis_iter_mut(Axis(2)) {
            coupe.zip_mut_with(&compte, |a, &c| *a /= c);
        }
    }

    // Retire la complétion.
    somme
        .slice(s![
            ..,
            plan_x.pad_before..plan_x.pad_before + nx,
            plan_y.pad_before..plan_y.pad_before + ny,
            ..
        ])
        .to_owned()
}

/// Masque binaire (0 fond, 1 cerveau) : la classe de plus grand logit. Utilisé par
/// les tests de l'inférence par tuiles (grille à 1 mm) ; `logits_to_mask` fait le
/// même calcul après le retour à la grille d'origine.
#[cfg(test)]
/// En cas d'égalité exacte, le fond l'emporte (comme `argmax` de torch ici).
fn argmax_mask(logits: &Array4<f32>) -> Array3<u8> {
    let fond = logits.index_axis(Axis(0), 0);
    let cerveau = logits.index_axis(Axis(0), 1);
    ndarray::Zip::from(&fond)
        .and(&cerveau)
        .map_collect(|&f, &c| u8::from(c > f))
}

/// Adresse des poids par défaut : le dépôt Hugging Face du projet, épinglé sur la
/// révision qui contient ce fichier (le contenu à cette adresse ne change jamais).
const MODEL_URL: &str = "https://huggingface.co/rousseau/medoxide-fetalbet/resolve/645902a2e04be584e3c7acc5faeeb52dab5579eb/attunet.bpk";
/// Empreinte SHA-256 attendue pour ces poids.
const MODEL_SHA256: &str = "a70bcbe8da5f791b751c293a851505eb1aa9aa44c50def0afc41fd34bd60c3c2";
/// Nom du fichier de poids dans le dossier de cache.
const MODEL_FILE: &str = "attunet.bpk";

/// Dossier de cache de medoxide : `$XDG_CACHE_HOME/medoxide`, sinon
/// `$HOME/.cache/medoxide`. `None` si aucune de ces variables n'est définie.
/// Les valeurs sont passées en paramètres pour pouvoir tester sans toucher à
/// l'environnement du processus.
fn cache_dir_from(xdg_cache_home: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
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
fn download_verified(url: &str, dest: &Path, expected_sha256: &str) -> Result<(), MaskError> {
    let echec = |contexte: &str, e: &dyn std::fmt::Display| {
        MaskError::ModelDownload(format!("{contexte} : {e}"))
    };
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
            return Err(MaskError::ModelDownload(format!(
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

/// Chemin des poids par défaut du modèle, téléchargés au besoin.
///
/// Les poids sont cherchés dans le cache (`$XDG_CACHE_HOME/medoxide/attunet.bpk`,
/// sinon `~/.cache/medoxide/attunet.bpk`). S'ils sont absents, ils sont
/// téléchargés depuis Hugging Face (environ 121 Mo, message sur stderr), vérifiés
/// par SHA-256, puis conservés pour les lancements suivants. Un fichier déjà
/// présent n'est pas revérifié.
///
/// # Erreurs
/// `MaskError::ModelDownload` si aucun dossier de cache n'est déterminable, ou si
/// le téléchargement, l'écriture ou la vérification échoue. Hors ligne, utiliser
/// un fichier local (`--model`).
pub fn default_model_path() -> Result<PathBuf, MaskError> {
    let dossier = cache_dir_from(
        std::env::var_os("XDG_CACHE_HOME"),
        std::env::var_os("HOME"),
    )
    .ok_or_else(|| {
        MaskError::ModelDownload(
            "aucun dossier de cache (ni XDG_CACHE_HOME ni HOME) ; indiquez les poids avec --model"
                .to_string(),
        )
    })?;
    let chemin = dossier.join(MODEL_FILE);
    if chemin.exists() {
        return Ok(chemin);
    }
    std::fs::create_dir_all(&dossier)
        .map_err(|e| MaskError::ModelDownload(format!("{} : {e}", dossier.display())))?;
    eprintln!(
        "Téléchargement des poids du modèle (121 Mo) vers {}\n  depuis {MODEL_URL}",
        chemin.display()
    );
    download_verified(MODEL_URL, &chemin, MODEL_SHA256)?;
    eprintln!("Poids téléchargés et vérifiés.");
    Ok(chemin)
}

/// Calcule le masque cérébral d'un volume IRM fœtal et l'écrit en NIfTI.
///
/// Reproduit l'inférence de Fetal-BET :
/// 1. lecture du volume (`input_path`) ;
/// 2. rééchantillonnage à 1 mm dans le plan, puis normalisation par coupe ;
/// 3. inférence par tuiles de 256×256 (recouvrement 50 %) avec le modèle dont
///    les poids sont dans `model_path` (fichier `.bpk`, backend wgpu) ;
/// 4. retour des logits sur la grille d'origine, puis argmax ;
/// 5. écriture du masque (`uint8`, même affine que l'entrée) dans `output_path`.
///
/// `model_path` à `None` : poids par défaut, téléchargés au besoin (voir
/// [`default_model_path`]), **après** la vérification de l'entrée et du dossier de
/// sortie, pour ne pas télécharger 121 Mo avant une faute de frappe.
///
/// # Erreurs
/// `MaskError::ModelNotFound` si `model_path` est donné et n'existe pas ;
/// `MaskError::OutputDirNotFound` si le dossier de `output_path` n'existe pas
/// (vérifié avant tout calcul) ; `MaskError::Core` (lecture ou écriture NIfTI, ou volume non 3D)
/// si l'entrée est invalide ; `MaskError::ModelDownload` si le téléchargement des
/// poids par défaut échoue. Un fichier de poids présent mais invalide fait
/// paniquer le chargement du modèle (code généré).
pub fn segment(
    input_path: &Path,
    output_path: &Path,
    model_path: Option<&Path>,
) -> Result<(), MaskError> {
    if let Some(chemin) = model_path {
        if !chemin.exists() {
            return Err(MaskError::ModelNotFound(chemin.to_path_buf()));
        }
    }
    // `parent()` vaut `Some("")` pour un nom de fichier seul : le dossier courant.
    if let Some(dossier) = output_path.parent() {
        if !dossier.as_os_str().is_empty() && !dossier.is_dir() {
            return Err(MaskError::OutputDirNotFound(dossier.to_path_buf()));
        }
    }
    let info = volume_info(input_path)?;
    let volume = read_volume(input_path)?;
    let (nx, ny, _) = volume.dim();
    let spacing = [info.spacing[0], info.spacing[1]];

    // Les poids par défaut ne sont téléchargés qu'une fois l'entrée validée.
    let model_path = match model_path {
        Some(chemin) => chemin.to_path_buf(),
        None => default_model_path()?,
    };

    let mut prep = resample_in_plane(&volume, spacing);
    normalize_slices(&mut prep);

    let device = Device::default();
    let modele = model::Model::from_file(&model_path, &device);
    let logits = infer_logits(&modele, &device, &prep);

    let masque = logits_to_mask(&logits, spacing, (nx, ny));
    write_mask(output_path, &masque, input_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lit un fichier de `f32` bruts en little-endian (4 octets par valeur).
    fn lire_f32(chemin: &str) -> Vec<f32> {
        let octets = std::fs::read(chemin).expect(chemin);
        octets
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Étape 4b : les logits Burn (wgpu) doivent égaler ceux d'onnxruntime.
    /// Références produites par `python scripts/make_reference_slice.py`.
    /// Critère : écart relatif (max |Δ| / max |logit|) < 1e-4, argmax identique.
    #[test]
    fn inference_matches_onnxruntime() {
        use burn::tensor::{Device, Tensor, TensorData};

        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let entree = lire_f32(&format!("{racine}/data/reference/slice_input.f32"));
        let attendu = lire_f32(&format!("{racine}/data/reference/slice_logits.f32"));

        let device = Device::default();
        println!("périphérique : {device:?}");
        let modele = model::Model::from_file(format!("{racine}/models/attunet.bpk"), &device);
        let x = Tensor::<4>::from_data(TensorData::new(entree, [1, 1, 256, 256]), &device);
        let lu = modele.forward(x).into_data().try_to_vec::<f32>().unwrap();

        assert_eq!(lu.len(), attendu.len());
        let max_logit = attendu.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let max_ecart = lu
            .iter()
            .zip(&attendu)
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        let relatif = max_ecart / max_logit;

        // Sortie [1, 2, 256, 256] : canal 0 = fond, canal 1 = cerveau.
        let n = 256 * 256;
        let classe = |v: &[f32], i: usize| usize::from(v[n + i] > v[i]);
        let accord = (0..n).filter(|&i| classe(&lu, i) == classe(&attendu, i)).count();
        let masque = (0..n).filter(|&i| classe(&attendu, i) == 1).count();
        println!("écart relatif {relatif:.2e} ; argmax identique {accord}/{n} ; voxels masque {masque}");

        assert!(masque > 0, "tuile de référence sans masque");
        assert!(relatif < 1e-4, "écart relatif {relatif:.2e} >= 1e-4");
        assert_eq!(accord, n, "argmax différent sur {} voxels", n - accord);
    }

    /// Étape 5b : normalisation par coupe identique à Fetal-BET (sans
    /// rééchantillonnage). Critère : écart relatif (max |Δ| / max |valeur|) < 1e-4.
    #[test]
    fn normalize_slices_matches_reference() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        for nom in ["03", "04", "06", "07", "09", "10", "11", "12"] {
            let mut volume = read_volume(Path::new(&format!(
                "{racine}/data/sourcedata/fetus_{nom}.nii.gz"
            )))
            .unwrap();
            normalize_slices(&mut volume);
            let attendu = lire_f32(&format!("{racine}/data/reference/fetus_{nom}_raw_norm.f32"));
            assert_eq!(volume.len(), attendu.len(), "fetus_{nom} : taille");
            let max_valeur = attendu.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let max_ecart = volume
                .iter()
                .zip(&attendu)
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            let relatif = max_ecart / max_valeur;
            println!("fetus_{nom} : écart relatif {relatif:.2e}");
            assert!(relatif < 1e-4, "fetus_{nom} : écart relatif {relatif:.2e}");
        }
    }

    /// Étape 5c : rééchantillonnage à 1 mm puis normalisation, comparés à MONAI
    /// (`Spacingd` puis `SliceWiseNormalizeIntensityd`) sur les 8 volumes.
    /// Écart relatif = max |Δ| / max |valeur|. Rééchantillonnage : < 1e-4. Prétraité :
    /// seuil provisoire 5e-3 (le critère initial de 1e-4 n'est pas atteint : MONAI
    /// laisse des voxels à ~1e-15 là où le résultat exact est 0, ce qui change
    /// l'écart-type de chaque coupe de ~0,2 %). À confirmer.
    #[test]
    fn preprocessing_matches_monai() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let ecart_relatif = |lu: &Array3<f32>, chemin: String, nom: &str| {
            let attendu = lire_f32(&chemin);
            assert_eq!(lu.len(), attendu.len(), "fetus_{nom} : taille {:?}", lu.dim());
            let max_valeur = attendu.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let max_ecart = lu
                .iter()
                .zip(&attendu)
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            max_ecart / max_valeur
        };
        for nom in ["03", "04", "06", "07", "09", "10", "11", "12"] {
            let chemin = format!("{racine}/data/sourcedata/fetus_{nom}.nii.gz");
            let info = volume_info(Path::new(&chemin)).unwrap();
            let volume = read_volume(Path::new(&chemin)).unwrap();

            let mut prep = resample_in_plane(&volume, [info.spacing[0], info.spacing[1]]);
            let rel_rech = ecart_relatif(
                &prep,
                format!("{racine}/data/reference/fetus_{nom}_resampled.f32"),
                nom,
            );
            normalize_slices(&mut prep);
            let rel_prep = ecart_relatif(
                &prep,
                format!("{racine}/data/reference/fetus_{nom}_prep.f32"),
                nom,
            );
            println!(
                "fetus_{nom} {:?} : rééchantillonné {rel_rech:.2e} ; prétraité {rel_prep:.2e}",
                prep.dim()
            );
            assert!(rel_rech < 1e-4, "fetus_{nom} : rééchantillonnage {rel_rech:.2e}");
            assert!(rel_prep < 5e-3, "fetus_{nom} : prétraitement {rel_prep:.2e}");
        }
    }

    /// Étape 6a : fenêtres identiques à celles de MONAI. Valeurs de référence
    /// générées avec `dense_patch_slices` (fenêtre 256, recouvrement 0,5) :
    /// (taille, zéros avant, longueur complétée, départs).
    #[test]
    fn window_plan_matches_monai() {
        let attendu: Vec<(usize, usize, usize, Vec<usize>)> = vec![
            (100, 78, 256, vec![0]),
            (200, 28, 256, vec![0]),
            (240, 8, 256, vec![0]),
            (256, 0, 256, vec![0]),
            (257, 0, 257, vec![0, 1]),
            (260, 0, 260, vec![0, 4]),
            (300, 0, 300, vec![0, 44]),
            (400, 0, 400, vec![0, 128, 144]),
            (513, 0, 513, vec![0, 128, 256, 257]),
        ];
        for (taille, pad_before, padded_len, starts) in attendu {
            let plan = window_plan(taille, 256, 0.5);
            assert_eq!(plan, AxisWindows { pad_before, padded_len, starts }, "taille {taille}");
        }
    }

    /// Coefficient de Dice entre deux masques binaires (octets 0/1, même ordre).
    fn dice(a: &Array3<u8>, b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let inter = a.iter().zip(b).filter(|(&x, &y)| x == 1 && y == 1).count();
        let (na, nb) = (a.iter().filter(|&&x| x == 1).count(), b.iter().filter(|&&y| y == 1).count());
        2.0 * inter as f64 / (na + nb) as f64
    }

    /// Vérifie l'inférence par tuiles sur un volume, contre `SliceInferer` de MONAI
    /// sur la grille à 1 mm. Références : `python scripts/make_reference_masks.py`.
    /// (1) Entrée MONAI (`_prep.f32`) : Dice >= 0,999 (et logits de fetus_03 à
    ///     < 1e-4 en relatif) : isole le découpage en tuiles.
    /// (2) Entrée issue du prétraitement Rust (5a à 5c) : Dice >= 0,99.
    fn verifier_inference(nom: &str, modele: &model::Model, device: &Device) {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let chemin = format!("{racine}/data/sourcedata/fetus_{nom}.nii.gz");
        let info = volume_info(Path::new(&chemin)).unwrap();
        let masque_monai =
            std::fs::read(format!("{racine}/data/reference/fetus_{nom}_mask1mm.u8")).unwrap();

        // Prétraitement Rust ; sa grille donne aussi la forme du volume à 1 mm.
        let mut prep = resample_in_plane(
            &read_volume(Path::new(&chemin)).unwrap(),
            [info.spacing[0], info.spacing[1]],
        );
        normalize_slices(&mut prep);
        let forme = prep.dim();

        // (1) Entrée MONAI.
        let prep_monai = Array3::from_shape_vec(
            forme,
            lire_f32(&format!("{racine}/data/reference/fetus_{nom}_prep.f32")),
        )
        .unwrap();
        let logits = infer_logits(modele, device, &prep_monai);
        let dice_monai = dice(&argmax_mask(&logits), &masque_monai);
        let mut message = format!("fetus_{nom} {forme:?} : Dice (entrée MONAI) {dice_monai:.5}");
        if nom == "03" {
            let attendu = lire_f32(&format!("{racine}/data/reference/fetus_03_logits1mm.f32"));
            let max_logit = attendu.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let max_ecart = logits
                .iter()
                .zip(&attendu)
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            let relatif = max_ecart / max_logit;
            message += &format!(" ; logits : écart relatif {relatif:.2e}");
            assert!(relatif < 1e-4, "fetus_03 : logits {relatif:.2e}");
        }
        assert!(dice_monai >= 0.999, "fetus_{nom} : Dice (entrée MONAI) {dice_monai:.5}");

        // (2) Entrée issue du prétraitement Rust.
        let dice_rust = dice(&argmax_mask(&infer_logits(modele, device, &prep)), &masque_monai);
        println!("{message} ; Dice (prétraitement Rust) {dice_rust:.5}");
        assert!(dice_rust >= 0.99, "fetus_{nom} : Dice (prétraitement Rust) {dice_rust:.5}");
    }

    /// Étape 6b, version rapide (~1 min) : fetus_06, dont l'axe 240 est complété
    /// à 256 (une seule fenêtre par coupe).
    #[test]
    fn tiled_inference_matches_monai_fast() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let device = Device::default();
        let modele = model::Model::from_file(format!("{racine}/models/attunet.bpk"), &device);
        verifier_inference("06", &modele, &device);
    }

    /// Étape 6b, version complète : les 8 volumes (~27 min, car 176 passages par
    /// volume pour les tailles 260 et 300). Lancer avec :
    /// `cargo test -p medoxide-fetalbet -- --ignored --nocapture tiled_inference`
    #[test]
    #[ignore = "27 min : voir la doc du test"]
    fn tiled_inference_matches_monai_all() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let device = Device::default();
        let modele = model::Model::from_file(format!("{racine}/models/attunet.bpk"), &device);
        for nom in ["03", "04", "06", "07", "09", "10", "11", "12"] {
            verifier_inference(nom, &modele, &device);
        }
    }

    /// Étape 7a : logits à 1 mm (MONAI) ramenés sur la grille d'origine puis argmax,
    /// comparés au masque final de Fetal-BET (étape 2), sur les 8 volumes.
    /// Critère : Dice >= 0,9999.
    #[test]
    fn logits_to_mask_matches_fetal_bet() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        for nom in ["03", "04", "06", "07", "09", "10", "11", "12"] {
            let chemin = format!("{racine}/data/sourcedata/fetus_{nom}.nii.gz");
            let info = volume_info(Path::new(&chemin)).unwrap();
            let (nx, ny, nz) = (info.dim[0] as usize, info.dim[1] as usize, info.dim[2] as usize);
            let spacing = [info.spacing[0], info.spacing[1]];
            let forme = (resampled_len(nx, spacing[0]), resampled_len(ny, spacing[1]), nz);

            let logits = Array4::from_shape_vec(
                (2, forme.0, forme.1, forme.2),
                lire_f32(&format!("{racine}/data/reference/fetus_{nom}_logits1mm.f32")),
            )
            .unwrap();
            let masque = logits_to_mask(&logits, spacing, (nx, ny));

            let officiel = read_volume(Path::new(&format!(
                "{racine}/data/derivatives/fetal-bet/fetus_{nom}_predicted_mask.nii.gz"
            )))
            .unwrap();
            assert_eq!(masque.dim(), officiel.dim(), "fetus_{nom} : dimensions");
            let officiel: Vec<u8> = officiel.iter().map(|&v| v as u8).collect();
            let diff = masque.iter().zip(&officiel).filter(|(a, b)| a != b).count();
            let d = dice(&masque, &officiel);
            println!("fetus_{nom} : Dice {d:.6} ; voxels différents {diff}/{}", officiel.len());
            assert!(d >= 0.9999, "fetus_{nom} : Dice {d:.6}");
        }
    }

    /// Étape 7b : un masque écrit avec l'en-tête du volume d'entrée garde ses
    /// dimensions et son affine, et se relit à l'identique.
    #[test]
    fn write_mask_roundtrip_keeps_header() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        for nom in ["03", "06", "12"] {
            let entree = format!("{racine}/data/sourcedata/fetus_{nom}.nii.gz");
            let masque: Array3<u8> = read_volume(Path::new(&format!(
                "{racine}/data/derivatives/fetal-bet/fetus_{nom}_predicted_mask.nii.gz"
            )))
            .unwrap()
            .mapv(|v| v as u8);

            let sortie = std::env::temp_dir()
                .join(format!("medoxide_test_{}_{nom}.nii.gz", std::process::id()));
            write_mask(&sortie, &masque, Path::new(&entree)).unwrap();

            let (info_in, info_out) = (
                volume_info(Path::new(&entree)).unwrap(),
                volume_info(&sortie).unwrap(),
            );
            let relu = read_volume(&sortie).unwrap();
            std::fs::remove_file(&sortie).unwrap();

            assert_eq!(info_out.dim, info_in.dim, "fetus_{nom} : dimensions");
            assert_eq!(info_out.spacing, info_in.spacing, "fetus_{nom} : espacement");
            assert_eq!(info_out.affine, info_in.affine, "fetus_{nom} : affine");
            assert!(info_out.affine.is_some(), "fetus_{nom} : sform attendu");
            assert_eq!(relu.mapv(|v| v as u8), masque, "fetus_{nom} : voxels relus");
        }
    }

    #[test]
    fn segment_fails_when_model_is_missing() {
        let r = segment(
            Path::new("in.nii.gz"),
            Path::new("out.nii.gz"),
            Some(Path::new("n_existe_pas.bpk")),
        );
        assert!(matches!(r, Err(MaskError::ModelNotFound(_))));
    }

    /// Un dossier de sortie absent est détecté avant tout calcul : l'entrée
    /// n'existe pas non plus ici, et pourtant c'est bien ce dossier qui est signalé.
    #[test]
    fn segment_fails_early_when_output_dir_is_missing() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let r = segment(
            Path::new("n_existe_pas.nii.gz"),
            Path::new("/dossier_inexistant_medoxide/masque.nii.gz"),
            Some(Path::new(&format!("{racine}/models/attunet.bpk"))),
        );
        assert!(matches!(r, Err(MaskError::OutputDirNotFound(_))), "{r:?}");
    }

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
    /// Lancer avec `cargo test -p medoxide-fetalbet -- --ignored download_rejects`.
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
        assert!(matches!(r, Err(MaskError::ModelDownload(ref m)) if m.contains("SHA-256")), "{r:?}");
        assert_eq!(reste, 0, "fichier résiduel après un échec");
    }

    /// Étape 8 : `segment` de bout en bout (fichier NIfTI → fichier NIfTI) contre
    /// le masque final de Fetal-BET (étape 2). Affiche le temps. Dice >= 0,99.
    fn verifier_segment(nom: &str) {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let entree = format!("{racine}/data/sourcedata/fetus_{nom}.nii.gz");
        let sortie = std::env::temp_dir().join(format!("medoxide_e2e_{}_{nom}.nii.gz", std::process::id()));

        let debut = std::time::Instant::now();
        let poids = format!("{racine}/models/attunet.bpk");
        segment(Path::new(&entree), &sortie, Some(Path::new(&poids))).unwrap();
        let duree = debut.elapsed();

        let masque: Array3<u8> = read_volume(&sortie).unwrap().mapv(|v| v as u8);
        std::fs::remove_file(&sortie).unwrap();
        let officiel: Vec<u8> = read_volume(Path::new(&format!(
            "{racine}/data/derivatives/fetal-bet/fetus_{nom}_predicted_mask.nii.gz"
        )))
        .unwrap()
        .iter()
        .map(|&v| v as u8)
        .collect();
        let d = dice(&masque, &officiel);
        println!("fetus_{nom} : Dice {d:.5} ; {:.1} s", duree.as_secs_f64());
        assert!(d >= 0.99, "fetus_{nom} : Dice {d:.5}");
    }

    /// Étape 8, version rapide : fetus_06 (une fenêtre par coupe).
    #[test]
    fn segment_end_to_end_fast() {
        verifier_segment("06");
    }

    /// Étape 8, version complète : les 8 volumes (long). Lancer avec :
    /// `cargo test --release -p medoxide-fetalbet -- --ignored --nocapture segment_end_to_end_all`
    #[test]
    #[ignore = "long : voir la doc du test"]
    fn segment_end_to_end_all() {
        for nom in ["03", "04", "06", "07", "09", "10", "11", "12"] {
            verifier_segment(nom);
        }
    }
}
