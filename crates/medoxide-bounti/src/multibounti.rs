//! Prétraitement géométrique de Multi-BOUNTI fœtal (phases A et B de `scripts/multibounti_mirtk_phases.sh`, voir `docs/MULTIBOUNTI.md`) : de la reconstruction à 0,5 mm aux cubes de 128³ que reçoivent le réseau d'extraction du cerveau puis le réseau gauche/droite.
//!
//! Les opérations sont celles du script officiel, avec les règles de MIRTK déjà établies pour BOUNTI (recadrages, grille de référence centrée, `-labels`, dilatation et érosion) ; seuls changent la grille de référence (359 × 359 × 359 voxels de 0,5 mm), le seuil (0,5)
//! et `pad-3d` (cube de 128 voxels sans mise à l'échelle des intensités).
use crate::cortex::amincir_le_cortex;
use crate::etiquettes::{etiquettes_vers_la_grille, etiquettes_vers_la_grille_par};
use crate::geometrie::{Boite, ErreurGeometrie, Volume};
use crate::masque::{dilater, eroder, plus_grande_composante};
use nalgebra::Matrix4;
use ndarray::Array3;

/// Dimensions de la grille de référence `ref-082025` du gabarit (voxels de 0,5 mm, orientation LAS).
pub const DIMENSIONS_GABARIT: [usize; 3] = [359, 359, 359];
/// Taille du voxel de la grille de référence (mm).
pub const VOXEL_GABARIT_MM: f64 = 0.5;
/// Seuil du masque de l'image : les voxels strictement supérieurs en font partie.
pub const SEUIL_MASQUE: f32 = 0.5;
/// Valeur absolue au-delà de laquelle `nan` met un voxel à 0 dans les images (phase A).
pub const SEUIL_VALEURS_EXTREMES: f32 = 100_000.0;
/// Côté des cubes d'entrée des réseaux d'extraction du cerveau et gauche/droite.
pub const COTE_PETIT_CUBE: usize = 128;
/// Côté du cube que reçoit le réseau à 43 étiquettes.
pub const COTE_GRAND_CUBE: usize = 256;
/// Côté de la grille du gabarit de réorientation (`dl-ref-t2` : voxels de 0,5 mm, orientation LAS), sur laquelle l'image est ramenée par la transformation affine.
pub const COTE_GRILLE_REORIENTEE: usize = 410;
/// Valeur absolue au-delà de laquelle `nan` met un voxel à 0 dans l'image réorientée (phase C).
pub const SEUIL_VALEURS_EXTREMES_REORIENTEE: f32 = 50_000.0;

/// Résultat de la phase A.
pub struct PhaseA {
    /// Image d'entrée après `nan` (`org-t2`).
    pub origine: Volume,
    /// Plus grande composante connexe des voxels > 0,5, sur la grille de l'entrée (`m-t2`).
    pub masque: Array3<bool>,
    /// Boîte englobante de `masque`, en voxels de l'entrée.
    pub boite: Boite,
    /// Entrée recadrée sur cette boîte (`crop-org-t2`).
    pub recadree: Volume,
    /// Entrée dans la grille de référence, recadrée sur la boîte physique du masque (`crop-tr-t2`).
    pub dans_le_gabarit: Volume,
    /// Cube de 128³ de `dans_le_gabarit`, intensités inchangées (`pad-crop-tr-t2-128`) : entrée du réseau d'extraction du cerveau.
    pub cube: Volume,
}

/// **Phase A** : prétraitement de `entree` (reconstruction à 0,5 mm) jusqu'au cube de 128³.
///
/// # Erreurs
/// [`ErreurGeometrie::Vide`] si aucun voxel ne dépasse le seuil.
pub fn phase_a(entree: &Volume) -> Result<PhaseA, ErreurGeometrie> {
    let mut origine = entree.clone();
    origine.annuler_valeurs_extremes(SEUIL_VALEURS_EXTREMES);
    let masque = plus_grande_composante(&origine.donnees.mapv(|v| v > SEUIL_MASQUE));
    let boite = Volume::boite_du_masque(&masque).ok_or(ErreurGeometrie::Vide)?;
    let recadree = origine.recadrer(boite.0, boite.1);
    // la grille de référence est centrée sur l'image recadrée, mais l'image rééchantillonnée est l'image ENTIÈRE (`transform-image org-t2 … -target tmp-ref`)
    let affine = Volume::affine_de_grille_las(&recadree.centre(), DIMENSIONS_GABARIT, VOXEL_GABARIT_MM);
    let mut dans_le_gabarit = origine.reechantillonner(&affine, DIMENSIONS_GABARIT);
    dans_le_gabarit.annuler_valeurs_extremes(SEUIL_VALEURS_EXTREMES);
    let dans_le_gabarit = dans_le_gabarit.recadrer_sur_boite_physique(&origine, boite)?;
    let cube = dans_le_gabarit.cube_sans_echelle(COTE_PETIT_CUBE);
    Ok(PhaseA {
        origine,
        masque,
        boite,
        recadree,
        dans_le_gabarit,
        cube,
    })
}

/// Résultat de la phase B.
pub struct PhaseB {
    /// Masque du cerveau nettoyé sur le cube de 128³ (`bet-lab-pad-crop-t2-128` après érosion, plus grande composante, dilatation).
    pub masque: Array3<bool>,
    /// Ce masque dilaté de 2 itérations (`dl-bet-lab-pad-crop-t2-128`).
    pub masque_dilate: Array3<bool>,
    /// Image dans la grille de référence, masquée par `masque_dilate` puis recadrée (`masked-n4-crop-tr-t2`).
    pub masquee: Volume,
    /// Cube de 128³ de `masquee`, intensités inchangées (`pad-masked-n4-crop-tr-t2-128`) : entrée du réseau gauche/droite.
    pub cube: Volume,
}

/// **Phase B** : nettoyage de l'extraction du cerveau (`brut` : étiquettes du réseau sur la grille de `a.cube`, `true` pour le cerveau), puis image masquée et cube de 128³ pour le réseau gauche/droite.
///
/// # Erreurs
/// [`ErreurGeometrie::Vide`] si le masque nettoyé est vide.
pub fn phase_b(a: &PhaseA, brut: &Array3<bool>) -> Result<PhaseB, ErreurGeometrie> {
    let masque = dilater(&plus_grande_composante(&eroder(brut, 1)), 1);
    let masque_dilate = dilater(&masque, 2);
    let volume_masque = Volume {
        donnees: masque_dilate.mapv(|m| if m { 1.0 } else { 0.0 }),
        affine: a.cube.affine,
    };
    let boite = Volume::boite_du_masque(&masque_dilate).ok_or(ErreurGeometrie::Vide)?;
    let masquee = a
        .dans_le_gabarit
        .masquer_au_plus_proche(&volume_masque)
        .recadrer_sur_boite_physique(&volume_masque, boite)?;
    let cube = masquee.cube_sans_echelle(COTE_PETIT_CUBE);
    Ok(PhaseB {
        masque,
        masque_dilate,
        masquee,
        cube,
    })
}

/// Résultat de la phase C.
pub struct PhaseC {
    /// Masque du cerveau passé dans la grille de `a.recadree`, dilaté de 2 itérations (`dl-bet-org-t2` avant recadrage).
    pub masque_origine: Array3<bool>,
    /// Image d'origine masquée par `masque_origine` puis recadrée sur sa boîte (`masked-crop-org-t2`).
    pub masquee_origine: Volume,
    /// Cette image ramenée par la transformation affine dans la grille du gabarit, recadrée sur les voxels > 0,5, valeurs extrêmes annulées (`reo-n4-masked-t2`).
    pub reorientee: Volume,
    /// Cube de 256³ de `reorientee`, intensités inchangées (`pad-reo-n4-masked-t2-256`) : entrée du réseau à 43 étiquettes.
    pub cube: Volume,
}

/// **Phase C** (hors recalage) : l'image masquée est ramenée dans l'orientation du gabarit par la transformation affine `transformation` (`aff-d.dof` : voir [`crate::affine::Parametres::matrice`]), puis mise en cube de 256³.
/// `b` donne le masque nettoyé du cerveau et l'image `masquee` sur laquelle est centrée la grille du gabarit.
///
/// # Erreurs
/// [`ErreurGeometrie::Vide`] si un masque ou l'image réorientée est vide.
pub fn phase_c(a: &PhaseA, b: &PhaseB, transformation: &Matrix4<f64>) -> Result<PhaseC, ErreurGeometrie> {
    let etiquettes = b.masque.mapv(u8::from);
    let sur_origine = etiquettes_vers_la_grille(&etiquettes, &a.cube.affine, &a.recadree);
    let masque_origine = dilater(&sur_origine.mapv(|e| e > 0), 2);
    let masquee_origine = a.recadree.masquer_et_recadrer(&masque_origine)?;
    let affine_gabarit =
        Volume::affine_de_grille_las(&b.masquee.centre(), [COTE_GRILLE_REORIENTEE; 3], VOXEL_GABARIT_MM);
    let ramenee =
        masquee_origine.reechantillonner_transforme(transformation, &affine_gabarit, [COTE_GRILLE_REORIENTEE; 3]);
    let (mut reorientee, _) = ramenee.recadrer_sur_seuil(SEUIL_MASQUE)?;
    reorientee.annuler_valeurs_extremes(SEUIL_VALEURS_EXTREMES_REORIENTEE);
    let cube = reorientee.cube_sans_echelle(COTE_GRAND_CUBE);
    Ok(PhaseC {
        masque_origine,
        masquee_origine,
        reorientee,
        cube,
    })
}

/// **Phase D** : étiquettes du réseau à 43 étiquettes (sur la grille de `c.cube`) ramenées dans la grille de l'image d'origine par l'INVERSE de la transformation (`-dofin_i`), puis cortex aminci (`thin-cortex-thick`).
/// Rend aussi les étiquettes avant amincissement (`tr-multi-lab-reo-t2`).
///
/// # Panique
/// Si `transformation` n'est pas inversible.
pub fn phase_d(
    a: &PhaseA,
    c: &PhaseC,
    etiquettes: &Array3<u8>,
    transformation: &Matrix4<f64>,
) -> (Array3<u8>, Array3<u8>) {
    let inverse = transformation.try_inverse().expect("transformation inversible");
    let avant = etiquettes_vers_la_grille_par(etiquettes, &c.cube.affine, &inverse, &a.origine);
    let apres = amincir_le_cortex(&avant);
    (avant, apres)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dossier(id: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/multibounti/ref")
            .join(id)
    }

    fn lire(id: &str, nom: &str) -> Volume {
        Volume::lire(&dossier(id).join(format!("{nom}.nii.gz"))).unwrap_or_else(|e| panic!("{id}/{nom} : {e}"))
    }

    /// Même grille (dimensions identiques, affine à `tol_affine` mm près) et intensités à `tol` de la plage près.
    fn comparer(id: &str, nom: &str, mien: &Volume, officiel: &Volume, tol: f64, tol_affine: f64) {
        assert_eq!(
            mien.dimensions(),
            officiel.dimensions(),
            "{id}/{nom} : dimensions (affines : {} contre {})",
            mien.affine.column(3).transpose(),
            officiel.affine.column(3).transpose()
        );
        let ecart_affine = (mien.affine - officiel.affine).abs().max();
        assert!(
            ecart_affine <= tol_affine,
            "{id}/{nom} : affine, écart {ecart_affine:e}"
        );
        let (min, max) = officiel
            .donnees
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let plage = f64::from(max - min).max(1e-30);
        let ecart = mien
            .donnees
            .iter()
            .zip(officiel.donnees.iter())
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max)
            / plage;
        println!("MULTI {id}/{nom} : dimensions {:?}, écart d'affine {ecart_affine:.1e} mm, écart d'intensité {ecart:.1e} de la plage", mien.dimensions());
        assert!(ecart <= tol, "{id}/{nom} : écart d'intensité {ecart:e}");
    }

    /// **Critère de la phase A** (`docs/MULTIBOUNTI.md`, M3) contre MIRTK (image ARM) sur `calme1` à `calme3` : masque `m-t2` identique à 99,9 % près, `crop-org-t2`, `crop-tr-t2` et le cube de 128³ de même grille (affine à 1e-3 mm près) et d'intensités à 1e-3 de la plage près.
    #[test]
    #[ignore = "demande les références de data/multibounti/ref (hors dépôt) : voir scripts/multibounti_reference.py"]
    fn phase_a_matches_mirtk() {
        for id in ["calme1", "calme2", "calme3"] {
            let a = phase_a(&lire(id, "entree")).unwrap();
            let m = lire(id, "m-t2");
            let differents = a
                .masque
                .iter()
                .zip(m.donnees.iter())
                .filter(|(&a, &b)| a != (b > 0.5))
                .count();
            println!(
                "MULTI {id}/m-t2 : {differents} voxels différents sur {}",
                a.masque.len()
            );
            assert!(differents as f64 <= 1e-3 * a.masque.len() as f64);
            comparer(id, "crop-org-t2", &a.recadree, &lire(id, "crop-org-t2"), 1e-3, 1e-3);
            comparer(
                id,
                "crop-tr-t2",
                &a.dans_le_gabarit,
                &lire(id, "crop-tr-t2"),
                1e-3,
                1e-3,
            );
            comparer(
                id,
                "pad-crop-tr-t2-128",
                &a.cube,
                &lire(id, "pad-crop-tr-t2-128"),
                1e-3,
                1e-3,
            );
        }
    }

    /// **Critère de la phase B** contre MIRTK sur `calme1` à `calme3`, en partant de la sortie BRUTE du réseau d'extraction du cerveau officiel (`bet-reseau-brut`) : masques nettoyés identiques à 99,9 % près, image masquée et cube de 128³ de même grille et d'intensités à 1e-3 de la plage près.
    #[test]
    #[ignore = "demande les références de data/multibounti/ref (hors dépôt) : voir scripts/multibounti_reference.py"]
    fn phase_b_matches_mirtk() {
        for id in ["calme1", "calme2", "calme3"] {
            let a = phase_a(&lire(id, "entree")).unwrap();
            let brut = lire(id, "bet-reseau-brut").donnees.mapv(|v| v > 0.5);
            let b = phase_b(&a, &brut).unwrap();
            for (nom, mien) in [
                ("bet-lab-pad-crop-t2-128", &b.masque),
                ("dl-bet-lab-pad-crop-t2-128", &b.masque_dilate),
            ] {
                let officiel = lire(id, nom);
                let differents = mien
                    .iter()
                    .zip(officiel.donnees.iter())
                    .filter(|(&a, &o)| a != (o > 0.5))
                    .count();
                println!("MULTI {id}/{nom} : {differents} voxels différents sur {}", mien.len());
                assert!(differents as f64 <= 1e-3 * mien.len() as f64, "{id}/{nom}");
            }
            comparer(
                id,
                "masked-n4-crop-tr-t2",
                &b.masquee,
                &lire(id, "masked-n4-crop-tr-t2"),
                1e-3,
                1e-3,
            );
            comparer(
                id,
                "pad-masked-n4-crop-tr-t2-128",
                &b.cube,
                &lire(id, "pad-masked-n4-crop-tr-t2-128"),
                1e-3,
                1e-3,
            );
        }
    }

    /// Transformation officielle `aff-d.dof` d'un volume de référence.
    fn transformation(id: &str) -> Matrix4<f64> {
        let octets = std::fs::read(dossier(id).join("aff-d.dof")).unwrap();
        crate::affine::Parametres::depuis_octets(&octets).unwrap().matrice()
    }

    /// **Critère de la phase C** (hors recalage : la transformation est celle de MIRTK) : masque d'origine identique à 99,9 % près, `masked-crop-org-t2`, `reo-n4-masked-t2` et le cube de 256³ de même grille (affine à 1e-3 mm près) et d'intensités à 1e-3 de la plage près.
    #[test]
    #[ignore = "demande les références de data/multibounti/ref (hors dépôt) : voir scripts/multibounti_reference.py"]
    fn phase_c_matches_mirtk() {
        for id in ["calme1", "calme2"] {
            let a = phase_a(&lire(id, "entree")).unwrap();
            let brut = lire(id, "bet-reseau-brut").donnees.mapv(|v| v > 0.5);
            let b = phase_b(&a, &brut).unwrap();
            let c = phase_c(&a, &b, &transformation(id)).unwrap();
            let officiel = lire(id, "dl-bet-org-t2");
            let boite = Volume::boite_du_masque(&c.masque_origine).unwrap();
            let mien = c.masque_origine.slice(ndarray::s![
                boite.0[0]..=boite.1[0],
                boite.0[1]..=boite.1[1],
                boite.0[2]..=boite.1[2]
            ]);
            assert_eq!(mien.dim(), officiel.donnees.dim(), "{id}/dl-bet-org-t2");
            let differents = mien
                .iter()
                .zip(officiel.donnees.iter())
                .filter(|(&a, &o)| a != (o > 0.5))
                .count();
            println!(
                "MULTI {id}/dl-bet-org-t2 : {differents} voxels différents sur {}",
                mien.len()
            );
            assert!(differents as f64 <= 1e-3 * mien.len() as f64, "{id}/dl-bet-org-t2");
            comparer(
                id,
                "masked-crop-org-t2",
                &c.masquee_origine,
                &lire(id, "masked-crop-org-t2"),
                1e-3,
                1e-3,
            );
            comparer(
                id,
                "reo-n4-masked-t2",
                &c.reorientee,
                &lire(id, "reo-n4-masked-t2"),
                1e-3,
                1e-3,
            );
            comparer(
                id,
                "pad-reo-n4-masked-t2-256",
                &c.cube,
                &lire(id, "pad-reo-n4-masked-t2-256"),
                1e-3,
                1e-3,
            );
        }
    }

    /// **Critère de la phase D** : à partir des étiquettes OFFICIELLES du réseau (`multi-lab-reo-t2`) et de la transformation officielle, les étiquettes dans la grille d'origine (`tr-multi-lab-reo-t2`) puis la sortie finale (`sortie_officielle`) diffèrent d'au plus 0,1 % des voxels NON NULS de la référence (même grille).
    #[test]
    #[ignore = "demande les références de data/multibounti/ref (hors dépôt) : voir scripts/multibounti_reference.py"]
    fn phase_d_matches_mirtk() {
        for id in ["calme1"] {
            let a = phase_a(&lire(id, "entree")).unwrap();
            let brut = lire(id, "bet-reseau-brut").donnees.mapv(|v| v > 0.5);
            let b = phase_b(&a, &brut).unwrap();
            let c = phase_c(&a, &b, &transformation(id)).unwrap();
            let etiquettes = lire(id, "multi-lab-reo-t2").donnees.mapv(|v| v.round() as u8);
            let (avant, apres) = phase_d(&a, &c, &etiquettes, &transformation(id));
            for (nom, mien) in [("tr-multi-lab-reo-t2", &avant), ("sortie_officielle", &apres)] {
                let officiel = lire(id, nom).donnees.mapv(|v| v.round() as u8);
                assert_eq!(mien.dim(), officiel.dim(), "{id}/{nom}");
                let non_nuls = officiel.iter().filter(|&&e| e > 0).count();
                let differents = mien.iter().zip(officiel.iter()).filter(|(a, b)| a != b).count();
                println!("MULTI {id}/{nom} : {differents} voxels différents sur {non_nuls} non nuls");
                assert!(differents as f64 <= 1e-3 * non_nuls as f64, "{id}/{nom}");
            }
        }
    }

    /// `edit-image gabarit -copy-origin image` : le gabarit gauche/droite (`lr-lab.nii.gz`) centré sur `masked-n4-crop-tr-t2` est identique à `tmp-lr-ref` de MIRTK (mêmes dimensions, affine ≤ 1e-4 mm, mêmes valeurs).
    #[test]
    #[ignore = "demande les références et les gabarits de data/multibounti (hors dépôt)"]
    fn the_recentred_template_matches_mirtks_copy_origin() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/multibounti");
        let gabarit = Volume::lire(&racine.join("burn/gabarit/lr-lab.nii.gz")).unwrap();
        for id in ["calme1", "calme2", "calme3"] {
            let centre = lire(id, "masked-n4-crop-tr-t2").centre();
            let mien = gabarit.recentrer(&centre);
            let officiel = lire(id, "tmp-lr-ref");
            comparer(id, "tmp-lr-ref", &mien, &officiel, 0.0, 1e-4);
        }
    }
}
