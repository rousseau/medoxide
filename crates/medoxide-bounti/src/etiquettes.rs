//! Retour des étiquettes de tissus dans la grille de l'image d'origine (dernier pas de la chaîne officielle : `transform-image … -labels` de MIRTK).
//!
//! Une image d'étiquettes ne s'interpole pas comme une image d'intensités : la moyenne de deux étiquettes (3 et 7) n'est pas une étiquette. Règle de MIRTK, établie par comparaison (voir `docs/BOUNTI.md`, sous-étape 5d-1) :
//! chaque étiquette présente est rééchantillonnée comme une carte binaire (1 sur ses voxels, 0 ailleurs, interpolation linéaire) ; un voxel d'arrivée prend l'étiquette dont la valeur interpolée **dépasse 0,5**
//! (il ne peut y en avoir qu'une), et, si aucune ne la dépasse (jonction de plusieurs étiquettes), l'étiquette du **plus proche voisin** dans la grille de départ. Hors de la grille de départ, le voxel de bord est prolongé d'un voxel entier au-delà du dernier centre (et non d'un demi-voxel comme pour les intensités), puis 0. Ce n'est donc pas « l'étiquette de plus grande valeur interpolée ».
use crate::geometrie::{Volume, MARGE_ETIQUETTES};
use ndarray::Array3;

/// Seuil au-dessus duquel la valeur interpolée d'une étiquette l'emporte.
const SEUIL: f32 = 0.5;

/// Étiquette du voxel de `etiquettes` le plus proche de la position `voxel` (en coordonnées de voxel de la grille de départ) : celle du voxel de bord jusqu'à [`MARGE_ETIQUETTES`] voxels au-delà du dernier centre, 0 au-delà.
fn plus_proche_voisin(etiquettes: &Array3<u8>, voxel: &nalgebra::Vector3<f64>) -> u8 {
    let (a, b, c) = etiquettes.dim();
    let dimensions = [a, b, c];
    let coordonnees = [voxel.x, voxel.y, voxel.z];
    let mut indices = [0usize; 3];
    for axe in 0..3 {
        let dernier = dimensions[axe] as f64 - 1.0;
        if coordonnees[axe] < -MARGE_ETIQUETTES || coordonnees[axe] > dernier + MARGE_ETIQUETTES {
            return 0;
        }
        indices[axe] = coordonnees[axe].round().clamp(0.0, dernier) as usize;
    }
    etiquettes[indices]
}

/// Passe une image d'étiquettes définie sur la grille d'affine `affine_source` (mêmes axes que `etiquettes`) dans la grille de `cible` : étiquette de valeur interpolée supérieure à 0,5, à défaut du plus proche voisin (voir le module).
pub fn etiquettes_vers_la_grille(
    etiquettes: &Array3<u8>,
    affine_source: &nalgebra::Matrix4<f64>,
    cible: &Volume,
) -> Array3<u8> {
    etiquettes_vers_la_grille_par(etiquettes, affine_source, &nalgebra::Matrix4::identity(), cible)
}

/// Comme [`etiquettes_vers_la_grille`] avec une transformation (`transform-image … -labels -dofin`) : la lecture se fait au point `transformation × (position du voxel de la cible)`, exprimé dans le repère de la grille source.
/// Pour appliquer l'INVERSE d'une transformation (`-dofin_i`), passer son inverse.
pub fn etiquettes_vers_la_grille_par(
    etiquettes: &Array3<u8>,
    affine_source: &nalgebra::Matrix4<f64>,
    transformation: &nalgebra::Matrix4<f64>,
    cible: &Volume,
) -> Array3<u8> {
    let dimensions = cible.dimensions();
    let affine_cible = transformation * cible.affine;
    let vers_source = affine_source.try_inverse().expect("affine inversible") * affine_cible;
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
        let interpolee = carte
            .reechantillonner_avec_marge(&affine_cible, dimensions, MARGE_ETIQUETTES)
            .donnees;
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

    /// Hors de la grille de départ : l'étiquette du voxel de bord est prolongée d'un voxel entier au-delà du dernier centre (règle de MIRTK pour les étiquettes, relevée à 0,83 voxel : 73 voxels sur 74 sur calme 2), puis 0 ; sur chacun des trois axes, des deux côtés.
    /// Une source à bord non uniforme (étiquettes 5 au bord bas, 6 au bord haut) distingue le bord prolongé du voxel le plus proche d'un autre bord.
    #[test]
    fn the_border_label_extends_one_voxel_then_zero() {
        let etiquettes = Array3::from_elem((3, 3, 3), 5u8);
        for axe in 0..3 {
            for (position, attendu) in [(-0.9, 5), (-1.1, 0), (2.9, 5), (3.1, 0)] {
                let mut decalage = [1.0; 3];
                decalage[axe] = position;
                assert_eq!(
                    etiquette_en(&etiquettes, decalage),
                    attendu,
                    "axe {axe}, position {position}"
                );
            }
        }
        let mut deux_bords = Array3::from_elem((4, 1, 1), 5u8);
        deux_bords[[3, 0, 0]] = 6;
        assert_eq!(etiquette_en(&deux_bords, [-0.9, 0.0, 0.0]), 5);
        assert_eq!(etiquette_en(&deux_bords, [3.9, 0.0, 0.0]), 6);
    }

    /// La prolongation du bord s'applique aussi aux valeurs interpolées : à 0,9 voxel au-delà du dernier centre sur x, avec des étiquettes identiques sur les deux dernières couches (A (1) sur trois des quatre positions de y et z, B (2) sur la quatrième) et un décalage de (0,35 ; 0,35) sur y et z,
    /// A pèse 0,5775 > 0,5 et l'emporte sur le plus proche voisin B ; avec une prolongation d'un demi-voxel seulement, les valeurs seraient nulles et le plus proche voisin B serait retenu.
    #[test]
    fn the_border_extension_applies_to_the_interpolated_values() {
        let mut etiquettes = Array3::from_elem((2, 2, 2), 1u8);
        for x in 0..2 {
            etiquettes[[x, 0, 0]] = 2;
        }
        assert_eq!(etiquette_en(&etiquettes, [1.9, 0.35, 0.35]), 1);
        assert_eq!(etiquette_en(&etiquettes, [-0.9, 0.35, 0.35]), 1);
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

    /// Diagnostic : dernier pas du masque du cerveau (`bet_s3` → grille d'entrée) à partir de l'intermédiaire officiel ; affiche les voxels différents par rapport à la grille de départ.
    #[test]
    #[ignore = "diagnostic : demande les références MIRTK de data/bounti/mirtk"]
    fn diagnostic_brain_mask_last_step() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti/mirtk");
        for id in ["calme1", "calme2", "calme3"] {
            let lire = |nom: &str| Volume::lire(&racine.join(id).join(format!("{nom}.nii.gz"))).unwrap();
            let (source, entree, mirtk) = (lire("bet_s3"), lire("entree"), lire("cerveau_officiel"));
            for perturbation in [1e-7, 1e-5] {
                let mut affine = source.affine;
                for axe in 0..3 {
                    affine[(axe, 3)] += perturbation;
                }
                let perturbe = etiquettes_vers_la_grille(&source.donnees.mapv(|v| u8::from(v > 0.5)), &affine, &entree);
                let differents = perturbe
                    .indexed_iter()
                    .filter(|(p, &m)| m != mirtk.donnees[*p] as u8)
                    .count();
                println!("DIAGPERTURB {id} : affine décalée de {perturbation:e} mm : {differents} voxels différents de MIRTK");
            }
            let mien = etiquettes_vers_la_grille(&source.donnees.mapv(|v| u8::from(v > 0.5)), &source.affine, &entree);
            let vers_source = source.affine.try_inverse().unwrap() * entree.affine;
            let d = source.dimensions();
            let (mut total, mut hors, mut proche_bord) = (0usize, 0usize, 0usize);
            let mut exemples = Vec::new();
            for (position, &m) in mien.indexed_iter() {
                if m != mirtk.donnees[position] as u8 {
                    total += 1;
                    let v = vers_source
                        * nalgebra::Vector4::new(position.0 as f64, position.1 as f64, position.2 as f64, 1.0);
                    let coord = [v.x, v.y, v.z];
                    let dehors = (0..3).any(|a| coord[a] < -0.5 || coord[a] > d[a] as f64 - 0.5);
                    let bord = (0..3).any(|a| coord[a] < 1.0 || coord[a] > d[a] as f64 - 2.0);
                    hors += usize::from(dehors);
                    proche_bord += usize::from(bord);
                    if exemples.len() < 4 {
                        exemples.push((
                            position,
                            m,
                            mirtk.donnees[position],
                            [v.x as f32, v.y as f32, v.z as f32],
                        ));
                    }
                }
            }
            println!("DIAGMASQUE {id} : {total} différents (hors de la grille de départ {hors}, à moins d'un voxel du bord {proche_bord}), source {d:?}, exemples {exemples:?}");
        }
    }
}
