//! Chaîne complète de Multi-BOUNTI fœtal (43 étiquettes, voxels de 0,5 mm) : de la reconstruction aux étiquettes sur la grille de cette reconstruction.
//!
//! Les étapes sont celles de la chaîne officielle (voir `docs/MULTIBOUNTI.md`), chacune validée séparément contre MIRTK et PyTorch :
//! 1. phase A et réseau d'extraction du cerveau (cube de 128³), phase B (masque nettoyé, image masquée), réseau gauche/droite : [`preparer`] ;
//! 2. recalage affine du gabarit gauche/droite sur la carte prédite ([`crate::recalage::recaler`], départ choisi d'après [`Preparation::volume_ml`]) ;
//! 3. phase C (image ramenée dans l'orientation du gabarit, cube de 256³), réseau à 43 étiquettes par fenêtres glissantes, phase D (retour dans la grille d'origine, amincissement du cortex) : [`segmenter_apres_recalage`].
use crate::fenetres::etiquettes_par_fenetres;
use crate::fusion::etiquettes;
use crate::geometrie::{ErreurGeometrie, Volume};
use crate::multibounti::{phase_a, phase_b, phase_c, phase_d, PhaseA, PhaseB, COTE_PETIT_CUBE};
use crate::recalage::{depart_selon_le_volume, recaler};
use crate::{bet, lr, multi};
use burn::tensor::{Device, Tensor, TensorData};
use nalgebra::Matrix4;
use ndarray::Array3;
use std::path::Path;

/// Côté et recouvrement des fenêtres du réseau à 43 étiquettes (`sliding_window_inference` du script officiel).
const FENETRE: usize = 128;
const RECOUVREMENT: f64 = 0.8;
/// Canaux de sortie du réseau à 43 étiquettes : le fond et les 43 étiquettes.
const CANAUX_TISSUS: usize = 44;

/// Les trois réseaux de la chaîne, chargés une fois pour plusieurs volumes.
pub struct ReseauxMulti {
    extraction: bet::Model,
    gauche_droite: lr::Model,
    tissus: multi::Model,
    gabarit: Volume,
}

impl ReseauxMulti {
    /// Charge les poids convertis depuis `dossier`, qui contient `bet/bet_64.bpk`, `lr/lr_64.bpk`, `multi/multi_64.bpk` et le gabarit gauche/droite `gabarit/lr-lab.nii.gz` (voir `docs/MULTIBOUNTI.md`).
    ///
    /// # Panique
    /// Si un fichier est absent ou illisible.
    pub fn charger(dossier: &Path, device: &Device) -> ReseauxMulti {
        ReseauxMulti {
            extraction: bet::Model::from_file(dossier.join("bet/bet_64.bpk"), device),
            gauche_droite: lr::Model::from_file(dossier.join("lr/lr_64.bpk"), device),
            tissus: multi::Model::from_file(dossier.join("multi/multi_64.bpk"), device),
            gabarit: Volume::lire(&dossier.join("gabarit/lr-lab.nii.gz")).expect("gabarit gauche/droite lisible"),
        }
    }
}

/// Ce que produisent les phases A et B et les deux petits réseaux, avant le recalage.
pub struct Preparation {
    /// Phase A.
    pub a: PhaseA,
    /// Phase B.
    pub b: PhaseB,
    /// Carte gauche (1) / droite (2) prédite sur le cube de `b.cube`, limitée au masque du cerveau (`lr-lab-pad-masked-crop-t2-128`).
    pub gauche_droite: Array3<u8>,
    /// Volume du masque du cerveau en mL, arrondi (ce qu'affiche `guess-ga`) : au-dessus de 170, la transformation de départ du recalage est l'identité rigide, sinon une mise à l'échelle de 40 %.
    pub volume_ml: u32,
}

/// Image normalisée dans [0, 1] sur tout le volume (comme le fait le script officiel avant chaque réseau) : `(v − min) / (max − min)`.
fn normaliser(cube: &Volume) -> Array3<f32> {
    let (min, max) = cube
        .donnees
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let plage = max - min;
    cube.donnees.mapv(|v| if plage > 0.0 { (v - min) / plage } else { 0.0 })
}

/// Tenseur `[1, 1, a, b, c]` d'un cube.
fn vers_tenseur(cube: &Array3<f32>, device: &Device) -> Tensor<5> {
    let (a, b, c) = cube.dim();
    let valeurs: Vec<f32> = cube.iter().copied().collect();
    Tensor::<5>::from_data(TensorData::new(valeurs, [1, 1, a, b, c]), device)
}

/// Volume d'un masque en mL, arrondi à l'entier (voxels × volume du voxel de `volume`).
fn volume_en_ml(masque: &Array3<bool>, volume: &Volume) -> u32 {
    let voxel_mm3 = volume.affine.fixed_view::<3, 3>(0, 0).determinant().abs();
    (masque.iter().filter(|&&m| m).count() as f64 * voxel_mm3 / 1000.0).round() as u32
}

/// Phases A et B, réseaux d'extraction du cerveau et gauche/droite.
///
/// # Erreurs
/// [`ErreurGeometrie::Vide`] si l'image ou le masque du cerveau est vide.
pub fn preparer(entree: &Volume, reseaux: &ReseauxMulti, device: &Device) -> Result<Preparation, ErreurGeometrie> {
    let a = phase_a(entree)?;
    let n = COTE_PETIT_CUBE;
    let sortie = reseaux.extraction.forward(vers_tenseur(&normaliser(&a.cube), device));
    let brut: Vec<bool> = etiquettes(sortie).into_iter().map(|e| e > 0).collect();
    let brut = Array3::from_shape_vec((n, n, n), brut).expect("cube complet");
    let b = phase_b(&a, &brut)?;
    let sortie = reseaux
        .gauche_droite
        .forward(vers_tenseur(&normaliser(&b.cube), device));
    let mut gauche_droite = Array3::from_shape_vec((n, n, n), etiquettes(sortie)).expect("cube complet");
    for (e, &dedans) in gauche_droite.iter_mut().zip(b.masque.iter()) {
        if !dedans {
            *e = 0;
        }
    }
    let volume_ml = volume_en_ml(&b.masque, &a.cube);
    Ok(Preparation {
        a,
        b,
        gauche_droite,
        volume_ml,
    })
}

/// Segmente `entree` (reconstruction 3D T2 fœtale à 0,5 mm) en 43 étiquettes, de bout en bout : [`preparer`], recalage affine, [`segmenter_apres_recalage`].
///
/// # Erreurs
/// [`ErreurGeometrie::Vide`] si l'image, le masque du cerveau ou l'image réorientée est vide.
pub fn segmenter(
    entree: &Volume,
    reseaux: &ReseauxMulti,
    device: &Device,
) -> Result<SegmentationMulti, ErreurGeometrie> {
    let preparation = preparer(entree, reseaux, device)?;
    let transformation = transformation_du_recalage(&preparation, reseaux);
    segmenter_apres_recalage(&preparation, &transformation, reseaux, device)
}

/// Recalage affine du gabarit gauche/droite (centré sur l'image masquée, comme `edit-image -copy-origin` dans la chaîne officielle) sur la carte gauche/droite prédite, depuis le départ choisi d'après le volume du cerveau.
pub fn transformation_du_recalage(preparation: &Preparation, reseaux: &ReseauxMulti) -> Matrix4<f64> {
    let gabarit = reseaux.gabarit.recentrer(&preparation.b.masquee.centre());
    let sujet = Volume {
        donnees: preparation.gauche_droite.mapv(f32::from),
        affine: preparation.b.cube.affine,
    };
    recaler(&gabarit, &sujet, &depart_selon_le_volume(preparation.volume_ml))
}

/// Segmentation en 43 étiquettes sur la grille de l'image d'entrée.
pub struct SegmentationMulti {
    /// Étiquette de chaque voxel (0 : fond, 1 à 43 : structures), cortex aminci.
    pub etiquettes: Array3<u8>,
}

/// Phases C et D et réseau à 43 étiquettes, avec la transformation affine `transformation` du recalage (de la grille du gabarit vers celle de l'image : voir [`crate::affine::Parametres::matrice`]).
///
/// # Erreurs
/// [`ErreurGeometrie::Vide`] si l'image réorientée est vide.
pub fn segmenter_apres_recalage(
    preparation: &Preparation,
    transformation: &Matrix4<f64>,
    reseaux: &ReseauxMulti,
    device: &Device,
) -> Result<SegmentationMulti, ErreurGeometrie> {
    let c = phase_c(&preparation.a, &preparation.b, transformation)?;
    let image = normaliser(&c.cube);
    let reseau = |fenetre: &Array3<f32>| -> Vec<f32> {
        reseaux
            .tissus
            .forward(vers_tenseur(fenetre, device))
            .into_data()
            .try_to_vec::<f32>()
            .expect("logits réels")
    };
    let sur_le_cube = etiquettes_par_fenetres(&image, FENETRE, RECOUVREMENT, CANAUX_TISSUS, reseau)
        .expect("le cube de 256³ contient des fenêtres de 128³");
    let (_, finales) = phase_d(&preparation.a, &c, &sur_le_cube, transformation);
    Ok(SegmentationMulti { etiquettes: finales })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn racine() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/multibounti")
    }

    fn appareil() -> Device {
        match std::env::var("MEDOXIDE_DEVICE").as_deref() {
            Ok("flex") => Device::flex(),
            _ => Device::default(),
        }
    }

    /// Coefficient de Dice de l'étiquette `e` entre deux cartes de même taille, et nombre de voxels de `e` dans la première ; `None` si l'étiquette est absente des deux.
    fn dice(a: &Array3<u8>, b: &Array3<u8>, e: u8) -> Option<(f64, usize)> {
        let (mut dans_a, mut dans_b, mut commun) = (0usize, 0usize, 0usize);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dans_a += usize::from(x == e);
            dans_b += usize::from(y == e);
            commun += usize::from(x == e && y == e);
        }
        (dans_a + dans_b > 0).then(|| (2.0 * commun as f64 / (dans_a + dans_b) as f64, dans_a))
    }

    /// Le Dice vaut 1 pour des cartes identiques, 0 pour des supports disjoints, 2/3 pour un support inclus dans un support deux fois plus grand ; `None` pour une étiquette absente des deux.
    #[test]
    fn the_dice_coefficient_is_computed_from_the_overlap() {
        let a = Array3::from_shape_vec((1, 1, 4), vec![1u8, 1, 0, 0]).unwrap();
        let b = Array3::from_shape_vec((1, 1, 4), vec![0u8, 0, 1, 1]).unwrap();
        let c = Array3::from_shape_vec((1, 1, 4), vec![1u8, 0, 0, 0]).unwrap();
        assert_eq!(dice(&a, &a, 1), Some((1.0, 2)));
        assert_eq!(dice(&a, &b, 1), Some((0.0, 2)));
        let (d, _) = dice(&a, &c, 1).unwrap();
        assert!((d - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(dice(&a, &a, 7), None);
    }

    /// Dice moyen sur les 43 étiquettes présentes, et le plus petit Dice parmi celles d'au moins 1 000 voxels (avec son numéro), de `obtenu` contre `officiel`.
    fn resume(id: &str, nom: &str, obtenu: &Array3<u8>, officiel: &Array3<u8>) -> (f64, f64, u8) {
        let (mut somme, mut compte) = (0.0, 0usize);
        let mut pire = (f64::INFINITY, 0u8);
        for e in 1..=43u8 {
            if let Some((d, voxels)) = dice(obtenu, officiel, e) {
                println!("MULTI {id} {nom} étiquette {e:2} : Dice {d:.4} ({voxels} voxels)");
                somme += d;
                compte += 1;
                if voxels >= 1000 && d < pire.0 {
                    pire = (d, e);
                }
            }
        }
        let accord = obtenu.iter().zip(officiel.iter()).filter(|(a, b)| a == b).count() as f64 / obtenu.len() as f64;
        println!(
            "MULTI {id} {nom} : accord de voxels {accord:.5}, Dice moyen {:.4} sur {compte} étiquettes, pire (≥ 1 000 voxels) {:.4} (étiquette {})",
            somme / compte as f64,
            pire.0,
            pire.1
        );
        (somme / compte as f64, pire.0, pire.1)
    }

    fn transformation(chemin: &Path) -> Matrix4<f64> {
        let octets = std::fs::read(chemin).unwrap();
        crate::affine::Parametres::depuis_octets(&octets).unwrap().matrice()
    }

    fn lancer(id: &str, dofs: &[(&str, PathBuf)]) -> Vec<(String, Array3<u8>)> {
        let device = appareil();
        let reseaux = ReseauxMulti::charger(&racine().join("burn"), &device);
        let entree = Volume::lire(&racine().join("ref").join(id).join("entree.nii.gz")).unwrap();
        let preparation = preparer(&entree, &reseaux, &device).unwrap();
        let officiel_ml: u32 = std::fs::read_to_string(racine().join("ref").join(id).join("guess-ga.txt"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        println!(
            "MULTI {id} : volume du masque {} mL (guess-ga officiel : {officiel_ml})",
            preparation.volume_ml
        );
        assert_eq!(preparation.volume_ml, officiel_ml, "{id} : guess-ga");
        dofs.iter()
            .map(|(nom, chemin)| {
                let debut = std::time::Instant::now();
                let r = segmenter_apres_recalage(&preparation, &transformation(chemin), &reseaux, &device).unwrap();
                println!("MULTI {id} {nom} : {:.0} s", debut.elapsed().as_secs_f64());
                (nom.to_string(), r.etiquettes)
            })
            .collect()
    }

    /// **Critère M5a** (`docs/MULTIBOUNTI.md`), recalage de MIRTK conservé (`aff-d.dof` officiel) : la chaîne en Burn, des fichiers d'entrée aux étiquettes, donne un Dice moyen ≥ 0,98 sur les étiquettes présentes et un Dice ≥ 0,95 pour toute étiquette d'au moins 1 000 voxels, contre `sortie_officielle` de `calme1`.
    /// Très long (343 fenêtres). À lancer sous le garde-fou mémoire : `python scripts/garde_memoire.py 18 cargo test --release -p medoxide-bounti the_full_multibounti_chain -- --ignored --nocapture`.
    #[test]
    #[ignore = "demande les poids de Multi-BOUNTI et les références de data/multibounti (hors dépôt) ; long"]
    fn the_full_multibounti_chain_matches_the_official_one_with_the_official_registration() {
        let id = "calme1";
        let dossier = racine().join("ref").join(id);
        let obtenus = lancer(id, &[("recalage officiel", dossier.join("aff-d.dof"))]);
        let officiel = Volume::lire(&dossier.join("sortie_officielle.nii.gz"))
            .unwrap()
            .donnees
            .mapv(|v| v.round() as u8);
        let (moyen, pire, e) = resume(id, "recalage officiel", &obtenus[0].1, &officiel);
        assert!(moyen >= 0.98, "Dice moyen {moyen}");
        assert!(pire >= 0.95, "étiquette {e} : Dice {pire}");
    }

    /// Mesure (sans critère) : sensibilité de la sortie officielle au recalage. La chaîne en Burn est lancée avec des recalages de MIRTK obtenus à partir de départs perturbés (`scripts/multibounti_repetabilite.py`, essais 2 et 0, écarts de 2,4 et 5,4 mm à l'officiel),
    /// et chaque sortie est comparée à la sortie officielle de `calme1` : c'est le plancher de bruit que le recalage de medoxide ne pourra pas descendre sous.
    #[test]
    #[ignore = "demande les poids de Multi-BOUNTI, les références de data/multibounti et les essais de répétabilité (hors dépôt) ; très long"]
    fn diagnostic_sensitivity_of_the_segmentation_to_the_registration() {
        let id = "calme1";
        let dossier = racine().join("ref").join(id);
        let essais = racine().join("essais/repet").join(id);
        let obtenus = lancer(
            id,
            &[
                ("essai 2", essais.join("essai-2.dof")),
                ("essai 0", essais.join("essai-0.dof")),
            ],
        );
        let officiel = Volume::lire(&dossier.join("sortie_officielle.nii.gz"))
            .unwrap()
            .donnees
            .mapv(|v| v.round() as u8);
        for (nom, etiquettes) in &obtenus {
            resume(id, nom, etiquettes, &officiel);
        }
    }

    /// **Critère M5** (`docs/MULTIBOUNTI.md`), chaîne de bout en bout avec NOTRE recalage : Dice moyen ≥ 0,98 sur les étiquettes présentes et Dice ≥ 0,95 pour toute étiquette d'au moins 1 000 voxels contre `sortie_officielle`, pour chaque volume de `MEDOXIDE_MULTI_IDS` (séparés par des virgules, `calme1` par défaut).
    /// Très long (343 fenêtres par volume). À lancer sous le garde-fou mémoire : `python scripts/garde_memoire.py 18 cargo test --release -p medoxide-bounti with_our_registration -- --ignored --nocapture`.
    #[test]
    #[ignore = "demande les poids de Multi-BOUNTI et les références de data/multibounti (hors dépôt) ; très long"]
    fn the_full_multibounti_chain_matches_the_official_one_with_our_registration() {
        let device = appareil();
        let reseaux = ReseauxMulti::charger(&racine().join("burn"), &device);
        let ids = std::env::var("MEDOXIDE_MULTI_IDS").unwrap_or_else(|_| "calme1".to_string());
        for id in ids.split(',') {
            let dossier = racine().join("ref").join(id);
            let entree = Volume::lire(&dossier.join("entree.nii.gz")).unwrap();
            let debut = std::time::Instant::now();
            let resultat = segmenter(&entree, &reseaux, &device).unwrap();
            println!("MULTI {id} : chaîne complète en {:.0} s", debut.elapsed().as_secs_f64());
            let officiel = Volume::lire(&dossier.join("sortie_officielle.nii.gz"))
                .unwrap()
                .donnees
                .mapv(|v| v.round() as u8);
            let (moyen, pire, e) = resume(id, "notre recalage", &resultat.etiquettes, &officiel);
            assert!(moyen >= 0.98, "{id} : Dice moyen {moyen}");
            assert!(pire >= 0.95, "{id} : étiquette {e} : Dice {pire}");
        }
    }
}
