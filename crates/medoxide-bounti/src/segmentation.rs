//! Chaîne complète de BOUNTI, avec deux choix d'extraction du cerveau ([`Extraction`] : le réseau 3D de BOUNTI, ou Fetal-BET) : de la reconstruction 3D aux étiquettes de tissus sur la grille de cette reconstruction.
//!
//! Les étapes sont celles de la chaîne officielle (voir `docs/BOUNTI.md`), chacune validée séparément contre MIRTK et PyTorch :
//! 1. [`Volume::preparer`] : recadrage, grille de référence à 0,6 mm ;
//! 2. extraction du cerveau (mode `bounti` : cube de 128³ et réseau 3D ; mode `fetalbet` : Fetal-BET sur le volume d'entrée), plus grande composante, retour dans la grille préparée, dilatation de 4 et érosion de 2 ;
//! 3. image masquée et recadrée, cube de 256³, fusion des quatre sorties (U-Net et Attention U-Net, image normale et retournée) ;
//! 4. retour des étiquettes dans la grille de l'image d'entrée ([`etiquettes_vers_la_grille`]).
use crate::etiquettes::etiquettes_vers_la_grille;
use crate::fusion::{etiquettes, logits_fusionnes};
use crate::geometrie::{ErreurGeometrie, Volume};
use crate::masque::{composante_du_cerveau, nettoyer, plus_grande_composante, vers_la_grille};
use crate::{attunet, bet, unet};
use burn::tensor::{Device, Tensor, TensorData};
use ndarray::Array3;
use std::path::{Path, PathBuf};

/// Côté du cube de l'extraction du cerveau.
const COTE_EXTRACTION: usize = 128;
/// Côté du cube de la segmentation des tissus.
const COTE_TISSUS: usize = 256;

/// Machine de calcul des réseaux.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Calcul {
    /// GPU (wgpu : Metal sur Mac, Vulkan sous Linux).
    Gpu,
    /// CPU de la machine (backend `flex` de Burn), beaucoup plus lent.
    Cpu,
}

impl Calcul {
    /// Appareil Burn correspondant.
    pub fn appareil(self) -> Device {
        match self {
            Calcul::Gpu => Device::default(),
            Calcul::Cpu => Device::flex(),
        }
    }
}

/// Les deux réseaux de segmentation des tissus, chargés une fois pour plusieurs volumes.
pub struct Reseaux {
    unet: unet::Model,
    attunet: attunet::Model,
}

impl Reseaux {
    /// Charge les poids convertis depuis `dossier`, qui contient `unet/unet_64.bpk` et `attunet/attunet_64.bpk` (voir `docs/BOUNTI.md`).
    ///
    /// # Panique
    /// Si un fichier de poids est absent ou illisible.
    pub fn charger(dossier: &Path, device: &Device) -> Reseaux {
        Reseaux {
            unet: unet::Model::from_file(dossier.join("unet/unet_64.bpk"), device),
            attunet: attunet::Model::from_file(dossier.join("attunet/attunet_64.bpk"), device),
        }
    }
}

/// Extraction du cerveau, première moitié de la chaîne : elle donne le masque qui délimite le cerveau avant la segmentation des tissus.
pub enum Extraction {
    /// Réseau 3D d'extraction de BOUNTI (chaîne officielle), appliqué à un cube de 128³ de l'image préparée (dans un `Box` : le réseau est grand, l'autre variante ne porte qu'un chemin).
    Bounti(Box<bet::Model>),
    /// Fetal-BET (`medoxide-fetalbet`), réseau 2D appliqué au volume d'entrée sur sa propre grille ; contient le fichier de poids de Fetal-BET.
    Fetalbet(PathBuf),
}

impl Extraction {
    /// Extraction par le réseau de BOUNTI : charge `bet/bet_64.bpk` depuis `dossier`.
    ///
    /// # Panique
    /// Si le fichier de poids est absent ou illisible.
    pub fn bounti(dossier: &Path, device: &Device) -> Extraction {
        Extraction::Bounti(Box::new(bet::Model::from_file(dossier.join("bet/bet_64.bpk"), device)))
    }

    /// Extraction par Fetal-BET avec le fichier de poids `poids` (format `.bpk` de `medoxide-fetalbet`).
    pub fn fetalbet(poids: PathBuf) -> Extraction {
        Extraction::Fetalbet(poids)
    }
}

/// Résultat de la chaîne, sur la grille de l'image d'entrée.
pub struct Segmentation {
    /// Étiquette de tissu de chaque voxel (0 : fond, 1 à 19 : tissus).
    pub etiquettes: Array3<u8>,
    /// Masque du cerveau : plus grande composante connexe de l'extraction, avant dilatation et érosion (comme le masque `mask-bet` de la chaîne officielle).
    pub cerveau: Array3<bool>,
}

/// Tenseur `[1, 1, n, n, n]` d'un cube.
fn vers_tenseur(cube: &Volume, device: &Device) -> Tensor<5> {
    let (a, b, c) = cube.donnees.dim();
    let valeurs: Vec<f32> = cube.donnees.iter().copied().collect();
    Tensor::<5>::from_data(TensorData::new(valeurs, [1, 1, a, b, c]), device)
}

/// Plus grande composante du cerveau trouvée par l'extraction, dans la grille de l'image préparée (`.0`) et dans celle de l'image d'entrée (`.1`).
fn composante_du_cerveau_extrait(
    entree: &Volume,
    preparee: &Volume,
    extraction: &Extraction,
    device: &Device,
) -> (Array3<bool>, Array3<bool>) {
    match extraction {
        Extraction::Bounti(reseau) => {
            let cube = preparee.cube(COTE_EXTRACTION);
            let n = COTE_EXTRACTION;
            let voxels: Vec<f32> = etiquettes(reseau.forward(vers_tenseur(&cube, device)))
                .into_iter()
                .map(f32::from)
                .collect();
            let voxels = Array3::from_shape_vec((n, n, n), voxels).expect("cube complet");
            let composante = composante_du_cerveau(&voxels, &cube.affine, preparee);
            let sur_l_entree =
                etiquettes_vers_la_grille(&composante.mapv(u8::from), &preparee.affine, entree).mapv(|v| v == 1);
            (composante, sur_l_entree)
        }
        Extraction::Fetalbet(poids) => {
            let pas = |axe: usize| entree.affine.fixed_view::<3, 1>(0, axe).norm() as f32;
            let brut = medoxide_fetalbet::mask_from_volume(&entree.donnees, [pas(0), pas(1)], poids);
            let sur_l_entree = plus_grande_composante(&brut.mapv(|v| v == 1));
            let composante = vers_la_grille(&sur_l_entree, &entree.affine, preparee);
            (composante, sur_l_entree)
        }
    }
}

/// Segmente `entree` (reconstruction 3D T2 d'un cerveau fœtal) en 19 tissus, avec l'extraction du cerveau choisie.
///
/// # Erreurs
/// [`ErreurGeometrie::Vide`] si aucun voxel de `entree` ne dépasse le seuil de recadrage, ou si l'extraction ne trouve aucun cerveau.
pub fn segmenter(
    entree: &Volume,
    reseaux: &Reseaux,
    extraction: &Extraction,
    device: &Device,
) -> Result<Segmentation, ErreurGeometrie> {
    let preparee = entree.preparer()?;

    // extraction du cerveau, puis nettoyage du masque
    let (composante, cerveau) = composante_du_cerveau_extrait(entree, &preparee, extraction, device);
    let masque = nettoyer(&composante);

    // segmentation des tissus sur l'image masquée
    let masquee = preparee.masquer_et_recadrer(&masque)?;
    let cube_tissus = masquee.cube(COTE_TISSUS);
    let image = vers_tenseur(&cube_tissus, device);
    let reseau_unet = |x: Tensor<5>| reseaux.unet.forward(x);
    let reseau_attunet = |x: Tensor<5>| reseaux.attunet.forward(x);
    let logits = logits_fusionnes(image, &[&reseau_unet, &reseau_attunet], device);
    let m = COTE_TISSUS;
    let sur_le_cube = Array3::from_shape_vec((m, m, m), etiquettes(logits)).expect("cube complet");

    // retour dans la grille de l'image d'entrée
    let etiquettes_finales = etiquettes_vers_la_grille(&sur_le_cube, &cube_tissus.affine, entree);
    Ok(Segmentation {
        etiquettes: etiquettes_finales,
        cerveau,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Coefficient de Dice de l'étiquette `e` entre deux images d'étiquettes de même taille : `2 |A ∩ B| / (|A| + |B|)` ; `None` si l'étiquette est absente des deux.
    fn dice(a: &Array3<u8>, b: &Array3<u8>, e: u8) -> Option<f64> {
        let (mut dans_a, mut dans_b, mut commun) = (0usize, 0usize, 0usize);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dans_a += usize::from(x == e);
            dans_b += usize::from(y == e);
            commun += usize::from(x == e && y == e);
        }
        (dans_a + dans_b > 0).then(|| 2.0 * commun as f64 / (dans_a + dans_b) as f64)
    }

    /// Le Dice vaut 1 pour des images identiques, 0 pour des supports disjoints, 2/3 pour un support inclus dans un support deux fois plus grand ; `None` si l'étiquette est absente des deux.
    #[test]
    fn the_dice_coefficient_is_computed_from_the_overlap() {
        let a = Array3::from_shape_vec((1, 1, 4), vec![1u8, 1, 0, 0]).unwrap();
        let b = Array3::from_shape_vec((1, 1, 4), vec![0u8, 0, 1, 1]).unwrap();
        let c = Array3::from_shape_vec((1, 1, 4), vec![1u8, 0, 0, 0]).unwrap();
        assert_eq!(dice(&a, &a, 1), Some(1.0));
        assert_eq!(dice(&a, &b, 1), Some(0.0));
        assert!((dice(&a, &c, 1).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(dice(&a, &a, 7), None);
    }

    /// **Critère C3**, mode `bounti`, contre la chaîne officielle (script Python officiel et commandes MIRTK ; références de `scripts/bounti_references_mirtk.py` : `etiquettes_officiel` et `cerveau_officiel` dans `data/bounti/mirtk/<id>`) sur trois volumes :
    /// Dice de chaque étiquette présente ≥ 0,98 et aucune sous 0,95 ; Dice du masque du cerveau ≥ 0,99 (voir `docs/BOUNTI.md`, sous-étape 5d-2).
    /// À lancer sous le garde-fou mémoire : `python scripts/garde_memoire.py 18 cargo test --release -p medoxide-bounti the_full_chain -- --ignored --nocapture`.
    #[test]
    #[ignore = "demande les poids, les volumes et les sorties du script officiel dans data/bounti (hors dépôt)"]
    fn the_full_chain_matches_the_official_chain() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti");
        let device = match std::env::var("MEDOXIDE_DEVICE").as_deref() {
            Ok("flex") => Device::flex(),
            _ => Device::default(),
        };
        let reseaux = Reseaux::charger(&racine.join("burn"), &device);
        let extraction = Extraction::bounti(&racine.join("burn"), &device);
        for id in ["calme1", "calme2", "calme3"] {
            let entree = Volume::lire(&racine.join("mirtk").join(id).join("entree.nii.gz")).unwrap();
            let dossier = racine.join("mirtk").join(id);
            let officiel_tissus = Volume::lire(&dossier.join("etiquettes_officiel.nii.gz")).unwrap();
            let officiel_cerveau = Volume::lire(&dossier.join("cerveau_officiel.nii.gz")).unwrap();
            let debut = std::time::Instant::now();
            let resultat = segmenter(&entree, &reseaux, &extraction, &device).unwrap();
            let duree = debut.elapsed().as_secs_f64();
            let tissus_officiels = officiel_tissus.donnees.mapv(|v| v as u8);
            assert_eq!(resultat.etiquettes.dim(), tissus_officiels.dim(), "{id} : dimensions");
            let accord = resultat
                .etiquettes
                .iter()
                .zip(tissus_officiels.iter())
                .filter(|(a, b)| a == b)
                .count() as f64
                / tissus_officiels.len() as f64;
            let mut pire = (f64::INFINITY, 0u8);
            let (mut somme, mut compte) = (0.0, 0usize);
            for e in 1..=19u8 {
                if let Some(d) = dice(&resultat.etiquettes, &tissus_officiels, e) {
                    println!("C3 {id} étiquette {e:2} : Dice {d:.4}");
                    somme += d;
                    compte += 1;
                    if d < pire.0 {
                        pire = (d, e);
                    }
                    assert!(d >= 0.95, "{id} : étiquette {e} sous 0,95 : {d}");
                }
            }
            let cerveau_officiel = officiel_cerveau.donnees.mapv(|v| u8::from(v > 0.5));
            let cerveau = resultat.cerveau.mapv(u8::from);
            let dice_cerveau = dice(&cerveau, &cerveau_officiel, 1).unwrap();
            println!(
                "C3 {id} : accord de voxels {accord:.5}, Dice moyen {:.4} sur {compte} étiquettes, pire {:.4} (étiquette {}), Dice du masque du cerveau {dice_cerveau:.4}, durée {duree:.1} s",
                somme / compte as f64,
                pire.0,
                pire.1
            );
            let voxels_differents = resultat
                .cerveau
                .iter()
                .zip(officiel_cerveau.donnees.iter())
                .filter(|(&a, &b)| a != (b > 0.5))
                .count();
            println!(
                "C3 {id} : masque du cerveau, {voxels_differents} voxels différents sur {} dans le masque officiel",
                cerveau_officiel.iter().filter(|&&v| v == 1).count()
            );
            assert!(dice_cerveau >= 0.99, "{id} : Dice du masque du cerveau {dice_cerveau}");
            assert!(pire.0 >= 0.98, "{id} : Dice minimal {} (étiquette {})", pire.0, pire.1);
        }
    }

    /// Réseau d'extraction du cerveau en Rust contre les étiquettes du réseau officiel (`bet_etiquettes_cube` de MIRTK/Docker) sur le même cube de 128³ (`cube128`, remis à l'échelle [0, 1]) : nombre de voxels d'étiquette différente, et leur marge de logits.
    #[test]
    #[ignore = "demande les poids et les références MIRTK de data/bounti (hors dépôt)"]
    fn the_brain_extraction_labels_match_the_official_ones() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti");
        let device = Device::default();
        let extraction = Extraction::bounti(&racine.join("burn"), &device);
        let Extraction::Bounti(reseau) = &extraction else {
            unreachable!()
        };
        for id in ["calme1", "calme2", "calme3"] {
            let dossier = racine.join("mirtk").join(id);
            let cube = Volume::lire(&dossier.join("cube128.nii.gz")).unwrap();
            let officiel = Volume::lire(&dossier.join("bet_etiquettes_cube.nii.gz")).unwrap();
            let (min, max) = cube
                .donnees
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
            let normalise = Volume {
                donnees: cube.donnees.mapv(|v| (v - min) / (max - min)),
                affine: cube.affine,
            };
            let logits = reseau.forward(vers_tenseur(&normalise, &device));
            let valeurs = logits.clone().into_data().convert::<f32>().to_vec::<f32>().unwrap();
            let voxels = 128 * 128 * 128;
            let mut differents = 0usize;
            let mut marge_max = 0.0_f32;
            for (position, (&mien, &attendu)) in etiquettes(logits).iter().zip(officiel.donnees.iter()).enumerate() {
                if f32::from(mien) != attendu {
                    differents += 1;
                    marge_max = marge_max.max((valeurs[position] - valeurs[voxels + position]).abs());
                }
            }
            println!("BET {id} : {differents} voxels d'étiquette différente sur {voxels} (marge de logits maximale des voxels différents {marge_max:.2e})");
        }
    }

    /// **Critère du mode `fetalbet`** (`docs/BOUNTI.md`, sous-étape 5d-4), sur trois volumes, contre la même référence officielle que le mode `bounti` : Dice moyen sur les 19 étiquettes d'au moins 0,95 pour chaque volume ; Dice par étiquette et Dice du masque du cerveau rapportés sans seuil.
    /// Demande les poids de Fetal-BET (`models/attunet.bpk` à la racine du dépôt). À lancer sous le garde-fou mémoire : `python scripts/garde_memoire.py 18 cargo test --release -p medoxide-bounti the_fetalbet_mode -- --ignored --nocapture`.
    #[test]
    #[ignore = "demande les poids, les volumes et les références officielles dans data/bounti, et les poids de Fetal-BET"]
    fn the_fetalbet_mode_meets_the_dice_criterion_against_the_official_chain() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti");
        let poids_fetalbet = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/attunet.bpk");
        let device = Device::default();
        let reseaux = Reseaux::charger(&racine.join("burn"), &device);
        let extraction = Extraction::fetalbet(poids_fetalbet);
        let mut moyennes = Vec::new();
        for id in ["calme1", "calme2", "calme3"] {
            let dossier = racine.join("mirtk").join(id);
            let entree = Volume::lire(&dossier.join("entree.nii.gz")).unwrap();
            let officiel = Volume::lire(&dossier.join("etiquettes_officiel.nii.gz"))
                .unwrap()
                .donnees
                .mapv(|v| v as u8);
            let cerveau_officiel = Volume::lire(&dossier.join("cerveau_officiel.nii.gz"))
                .unwrap()
                .donnees
                .mapv(|v| u8::from(v > 0.5));
            let resultat = segmenter(&entree, &reseaux, &extraction, &device).unwrap();
            let (mut somme, mut compte, mut pire) = (0.0, 0usize, (f64::INFINITY, 0u8));
            let mut lignes = Vec::new();
            for e in 1..=19u8 {
                if let Some(d) = dice(&resultat.etiquettes, &officiel, e) {
                    somme += d;
                    compte += 1;
                    if d < pire.0 {
                        pire = (d, e);
                    }
                    lignes.push(format!("{e}:{d:.3}"));
                }
            }
            let moyen = somme / compte as f64;
            let dice_masque = dice(&resultat.cerveau.mapv(u8::from), &cerveau_officiel, 1).unwrap();
            println!("FETALBET {id} : Dice moyen {moyen:.4} sur {compte} étiquettes, pire {:.4} (étiquette {}), Dice du masque du cerveau contre l'officiel {dice_masque:.4}", pire.0, pire.1);
            println!("FETALBET {id} par étiquette : {}", lignes.join(" "));
            moyennes.push((id, moyen));
        }
        // le critère n'est jugé qu'après avoir affiché les trois volumes
        for (id, moyen) in moyennes {
            assert!(moyen >= 0.95, "{id} : Dice moyen {moyen}");
        }
    }

    /// Diagnostic : Fetal-BET sur le volume `BOUNTI_VOLUME` (défaut `calme1`) en prenant chacun des trois axes comme axe des coupes ; affiche le Dice de chaque masque contre le masque du cerveau officiel et la part de ce masque couverte.
    #[test]
    #[ignore = "diagnostic : demande les références de data/bounti et les poids de Fetal-BET"]
    fn diagnostic_fetalbet_slice_axis() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti");
        let poids = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/attunet.bpk");
        let id = std::env::var("BOUNTI_VOLUME").unwrap_or_else(|_| "calme1".into());
        let dossier = racine.join("mirtk").join(&id);
        let entree = Volume::lire(&dossier.join("entree.nii.gz")).unwrap();
        let officiel = Volume::lire(&dossier.join("cerveau_officiel.nii.gz"))
            .unwrap()
            .donnees
            .mapv(|v| v > 0.5);
        for (nom, ordre) in [
            ("z (axes x,y,z)", [0usize, 1, 2]),
            ("y (axes x,z,y)", [0, 2, 1]),
            ("x (axes y,z,x)", [1, 2, 0]),
        ] {
            let permute = entree
                .donnees
                .clone()
                .permuted_axes(ordre)
                .as_standard_layout()
                .to_owned();
            let masque = medoxide_fetalbet::mask_from_volume(&permute, [0.8, 0.8], &poids);
            let mut inverse = [0usize; 3];
            for (position, &axe) in ordre.iter().enumerate() {
                inverse[axe] = position;
            }
            let masque = masque
                .permuted_axes(inverse)
                .as_standard_layout()
                .to_owned()
                .mapv(|v| v == 1);
            let commun = masque.iter().zip(officiel.iter()).filter(|(&a, &b)| a && b).count() as f64;
            let (a, b) = (
                masque.iter().filter(|&&v| v).count() as f64,
                officiel.iter().filter(|&&v| v).count() as f64,
            );
            println!("AXE {id} coupes selon {nom} : Dice {:.4}, masque {a} voxels, couverture du masque officiel {:.3}, part dans l'officiel {:.3}", 2.0 * commun / (a + b), commun / b, commun / a);
        }
    }
}
