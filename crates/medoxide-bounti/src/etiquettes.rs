//! Retour des étiquettes de tissus dans la grille de l'image d'origine (dernier pas de la chaîne officielle : `transform-image … -labels` de MIRTK).
//!
//! Une image d'étiquettes ne s'interpole pas comme une image d'intensités : la moyenne de deux étiquettes (3 et 7) n'est pas une étiquette. Règle de MIRTK, établie par comparaison (voir `docs/BOUNTI.md`, sous-étape 5d-1) :
//! chaque étiquette présente est rééchantillonnée comme une carte binaire (1 sur ses voxels, 0 ailleurs, interpolation linéaire) ; un voxel d'arrivée prend l'étiquette dont la valeur interpolée **dépasse 0,5**
//! (il ne peut y en avoir qu'une), et, si aucune ne la dépasse (jonction de plusieurs étiquettes), l'étiquette du **plus proche voisin** dans la grille de départ (0 hors de cette grille). Ce n'est donc pas « l'étiquette de plus grande valeur interpolée ».
use crate::geometrie::Volume;
use ndarray::Array3;

/// Seuil au-dessus duquel la valeur interpolée d'une étiquette l'emporte.
const SEUIL: f32 = 0.5;

/// Étiquette du voxel de `etiquettes` le plus proche de la position `voxel` (en coordonnées de voxel de la grille de départ), 0 hors de la grille.
fn plus_proche_voisin(etiquettes: &Array3<u8>, voxel: &nalgebra::Vector3<f64>) -> u8 {
    let (a, b, c) = etiquettes.dim();
    let (i, j, k) = (voxel.x.round(), voxel.y.round(), voxel.z.round());
    let dedans = i >= 0.0 && j >= 0.0 && k >= 0.0 && i < a as f64 && j < b as f64 && k < c as f64;
    if dedans {
        etiquettes[[i as usize, j as usize, k as usize]]
    } else {
        0
    }
}

/// Passe une image d'étiquettes définie sur la grille d'affine `affine_source` (mêmes axes que `etiquettes`) dans la grille de `cible` : étiquette de valeur interpolée supérieure à 0,5, à défaut du plus proche voisin (voir le module).
pub fn etiquettes_vers_la_grille(
    etiquettes: &Array3<u8>,
    affine_source: &nalgebra::Matrix4<f64>,
    cible: &Volume,
) -> Array3<u8> {
    let dimensions = cible.dimensions();
    let vers_source = affine_source.try_inverse().expect("affine inversible") * cible.affine;
    let mut resultat = Array3::<u8>::zeros((dimensions[0], dimensions[1], dimensions[2]));
    for ((i, j, k), sortie) in resultat.indexed_iter_mut() {
        let voxel = (vers_source * nalgebra::Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
        *sortie = plus_proche_voisin(etiquettes, &voxel);
    }
    let mut presentes: Vec<u8> = etiquettes.iter().copied().collect();
    presentes.sort_unstable();
    presentes.dedup();
    for etiquette in presentes {
        let carte = Volume {
            donnees: etiquettes.mapv(|e| if e == etiquette { 1.0 } else { 0.0 }),
            affine: *affine_source,
        };
        let interpolee = carte.reechantillonner(&cible.affine, dimensions).donnees;
        for (&valeur, sortie) in interpolee.iter().zip(resultat.iter_mut()) {
            if valeur > SEUIL {
                *sortie = etiquette;
            }
        }
    }
    resultat
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;
    use std::path::PathBuf;

    /// Grille d'un seul voxel, à la position `decalage` (en voxels de la source, affine de la source = identité) ; renvoie son étiquette.
    fn etiquette_en(etiquettes: &Array3<u8>, decalage: [f64; 3]) -> u8 {
        let mut affine_cible = Matrix4::identity();
        for axe in 0..3 {
            affine_cible[(axe, 3)] = decalage[axe];
        }
        let cible = Volume {
            donnees: Array3::zeros((1, 1, 1)),
            affine: affine_cible,
        };
        etiquettes_vers_la_grille(etiquettes, &Matrix4::identity(), &cible)[[0, 0, 0]]
    }

    /// Cellule de 2 × 2 × 2 voxels d'étiquettes données dans l'ordre `[x][y][z]` aplati (x lent).
    fn cellule(valeurs: [u8; 8]) -> Array3<u8> {
        Array3::from_shape_vec((2, 2, 2), valeurs.to_vec()).unwrap()
    }

    /// Sur la même grille, les étiquettes sont inchangées.
    #[test]
    fn the_same_grid_keeps_the_labels() {
        let etiquettes = Array3::from_shape_fn((4, 3, 5), |(i, j, k)| ((i * 7 + j * 3 + k) % 6) as u8);
        let cible = Volume {
            donnees: Array3::zeros((4, 3, 5)),
            affine: Matrix4::identity(),
        };
        assert_eq!(
            etiquettes_vers_la_grille(&etiquettes, &Matrix4::identity(), &cible),
            etiquettes
        );
    }

    /// Jonction de trois étiquettes à (0,3 ; 0,3 ; 0,3) de la cellule : B (2) au coin le plus proche pèse 0,343, A (1) sur deux coins 0,294, C (3) sur cinq coins 0,363 ; aucune ne dépasse 0,5 : plus proche voisin (B), et non la plus grande valeur (C).
    #[test]
    fn a_junction_takes_the_nearest_neighbour() {
        //                 000 001 010 011 100 101 110 111
        let etiquettes = cellule([2, 3, 1, 3, 1, 3, 3, 3]);
        assert_eq!(etiquette_en(&etiquettes, [0.3, 0.3, 0.3]), 2);
    }

    /// À (0,45 ; 0,45 ; 0) : l'étiquette A (1) couvre trois coins sur quatre, valeur 0,6975 > 0,5 : elle l'emporte, bien que le plus proche voisin soit B (2).
    #[test]
    fn a_label_above_one_half_wins_over_the_nearest_neighbour() {
        //                 000 001 010 011 100 101 110 111
        let etiquettes = cellule([2, 2, 1, 1, 1, 1, 1, 1]);
        assert_eq!(etiquette_en(&etiquettes, [0.45, 0.45, 0.0]), 1);
    }

    /// Entre deux voxels d'étiquettes différentes, la plus proche l'emporte (0,7 de poids), de chaque côté du milieu.
    #[test]
    fn between_two_voxels_the_closer_label_wins() {
        let etiquettes = cellule([4, 4, 4, 4, 9, 9, 9, 9]);
        assert_eq!(etiquette_en(&etiquettes, [0.3, 0.0, 0.0]), 4);
        assert_eq!(etiquette_en(&etiquettes, [0.7, 0.0, 0.0]), 9);
    }

    /// Hors de la grille de départ (au-delà d'un demi-voxel), l'étiquette est 0 ; dans le demi-voxel de bord, celle du voxel de bord.
    #[test]
    fn outside_the_source_grid_the_label_is_zero() {
        let etiquettes = Array3::from_elem((3, 3, 3), 5u8);
        assert_eq!(etiquette_en(&etiquettes, [-0.4, 1.0, 1.0]), 5);
        assert_eq!(etiquette_en(&etiquettes, [-0.6, 1.0, 1.0]), 0);
        assert_eq!(etiquette_en(&etiquettes, [1.0, 2.4, 1.0]), 5);
        assert_eq!(etiquette_en(&etiquettes, [1.0, 2.6, 1.0]), 0);
        assert_eq!(etiquette_en(&etiquettes, [2.4, 1.0, 1.0]), 5);
        assert_eq!(etiquette_en(&etiquettes, [2.6, 1.0, 1.0]), 0);
        assert_eq!(etiquette_en(&etiquettes, [1.0, 1.0, 2.4]), 5);
        assert_eq!(etiquette_en(&etiquettes, [1.0, 1.0, 2.6]), 0);
        assert_eq!(etiquette_en(&etiquettes, [1.0, 1.0, -0.6]), 0);
    }

    /// Le plus proche voisin suit les axes de la source : une source anisotrope et décalée (affine non identité) donne l'étiquette du voxel qui contient la position, selon chaque axe.
    #[test]
    fn the_nearest_neighbour_follows_the_source_affine() {
        let etiquettes = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| (1 + i + 3 * j + 12 * k) as u8);
        let mut affine = Matrix4::identity();
        affine[(0, 0)] = 2.0;
        affine[(1, 1)] = 0.5;
        affine[(2, 2)] = 3.0;
        affine[(0, 3)] = 10.0;
        affine[(1, 3)] = -1.0;
        affine[(2, 3)] = 4.0;
        // voxel (1, 2, 3) de la source : monde (12, 0, 13) ; la cible est un seul voxel à ce point, légèrement décalé
        let cible = Volume {
            donnees: Array3::zeros((1, 1, 1)),
            affine: Matrix4::new(
                1.0, 0.0, 0.0, 12.2, 0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 1.0, 12.5, 0.0, 0.0, 0.0, 1.0,
            ),
        };
        let sortie = etiquettes_vers_la_grille(&etiquettes, &affine, &cible);
        assert_eq!(sortie[[0, 0, 0]], etiquettes[[1, 2, 3]]);
    }

    /// Jonction de trois étiquettes autour de chacun des huit coins de la cellule : le décalage vaut 0,3 ou 0,7 sur chaque axe (le plus proche voisin est donc chacun des huit coins) ; B (2) est mis sur ce coin, A (1) sur ses voisins selon x et y, C (3) ailleurs (poids 0,343 / 0,294 / 0,363).
    /// Contrôle l'arrondi du plus proche voisin sur chaque axe, des deux côtés du milieu.
    #[test]
    fn a_junction_takes_the_nearest_neighbour_around_each_corner() {
        for coin in 0..8usize {
            let bits = [coin & 1, (coin >> 1) & 1, (coin >> 2) & 1];
            let mut etiquettes = Array3::from_elem((2, 2, 2), 3u8);
            etiquettes[[bits[0], bits[1], bits[2]]] = 2;
            etiquettes[[1 - bits[0], bits[1], bits[2]]] = 1;
            etiquettes[[bits[0], 1 - bits[1], bits[2]]] = 1;
            let decalage = [0, 1, 2].map(|axe| if bits[axe] == 1 { 0.7 } else { 0.3 });
            assert_eq!(etiquette_en(&etiquettes, decalage), 2, "coin {bits:?}");
        }
    }

    /// Jonction à (0,35 ; 0,35 ; 0) : A (1) pèse 0,5775 (trois coins) : il l'emporte sur le plus proche voisin B (2). À (0,45 ; 0,45 ; 0), avec A (1) sur deux coins (0,45), B (2) au coin le plus proche (0,3025) et C (3) sur le quatrième (0,2475), aucune étiquette ne dépasse 0,5 : plus proche voisin (B).
    /// Encadre le seuil de 0,5 de part et d'autre.
    #[test]
    fn the_threshold_is_one_half() {
        //                        000 001 010 011 100 101 110 111
        let trois_coins = cellule([2, 2, 1, 1, 1, 1, 1, 1]);
        assert_eq!(etiquette_en(&trois_coins, [0.35, 0.35, 0.0]), 1);
        let partage = cellule([2, 2, 3, 3, 1, 1, 1, 1]);
        assert_eq!(etiquette_en(&partage, [0.45, 0.45, 0.0]), 2);
    }

    /// Dans une source anisotrope et décalée, la jonction donne encore le plus proche voisin, la cible ayant sa propre affine (translation, échelle) : contrôle le sens de la composition des affines.
    #[test]
    fn the_nearest_neighbour_composes_the_affines_in_the_right_order() {
        let etiquettes = cellule([2, 3, 1, 3, 1, 3, 3, 3]);
        let mut source = Matrix4::identity();
        source[(0, 0)] = 2.0;
        source[(1, 1)] = 0.5;
        source[(2, 2)] = 3.0;
        source[(0, 3)] = 10.0;
        source[(1, 3)] = -1.0;
        source[(2, 3)] = 4.0;
        // monde du point (0,3 ; 0,3 ; 0,3) en voxels de la source : (10,6 ; −0,85 ; 4,9)
        let mut cible = Matrix4::identity();
        cible[(0, 0)] = 0.8;
        cible[(1, 1)] = 1.5;
        cible[(2, 2)] = 0.5;
        cible[(0, 3)] = 10.6 - 0.8 * 3.0;
        cible[(1, 3)] = -0.85 - 1.5 * 2.0;
        cible[(2, 3)] = 4.9 - 0.5 * 5.0;
        // le voxel (3, 2, 5) de la cible tombe sur ce point
        let grille = Volume {
            donnees: Array3::zeros((4, 3, 6)),
            affine: cible,
        };
        let sortie = etiquettes_vers_la_grille(&etiquettes, &source, &grille);
        assert_eq!(sortie[[3, 2, 5]], 2);
    }

    /// Comparaison avec MIRTK (références de `scripts/bounti_references_mirtk.py`, étape `etiquettes`) sur trois volumes : `cargo test --release -p medoxide-bounti the_labels -- --ignored --nocapture`.
    /// Tolérance : 0,1 % des voxels (critère de `docs/BOUNTI.md`) ; les écarts observés sont de 0 à 4 voxels sur 2,5 millions, tous à une valeur interpolée de 0,500008 (MIRTK calcule ses coordonnées en simple précision).
    #[test]
    #[ignore = "demande les références MIRTK de data/bounti/mirtk (voir docs/BOUNTI.md)"]
    fn the_labels_back_to_the_grid_match_mirtk() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti/mirtk");
        for id in ["calme1", "calme2", "calme3"] {
            let lire = |nom: &str| {
                Volume::lire(&racine.join(id).join(format!("{nom}.nii.gz")))
                    .unwrap_or_else(|e| panic!("{id}/{nom} : {e}"))
            };
            let source = lire("etiquettes_cube256");
            let entree = lire("entree");
            let mirtk = lire("etiquettes_grille");
            let etiquettes = source.donnees.mapv(|e| e as u8);
            let mien = etiquettes_vers_la_grille(&etiquettes, &source.affine, &entree);
            assert_eq!(mien.dim(), mirtk.donnees.dim());
            let differents = mien
                .indexed_iter()
                .filter(|(position, &e)| e != mirtk.donnees[*position] as u8)
                .count();
            println!("ÉTIQUETTES {id} : {differents} voxels différents sur {}", mien.len());
            assert!(
                differents as f64 <= 1e-3 * mien.len() as f64,
                "{id} : {differents} voxels différents"
            );
        }
    }
}
