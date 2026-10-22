//! Port de `thin-cortex-thick`, l'outil MIRTK propre à l'image de Multi-BOUNTI qui termine la chaîne officielle (voir `docs/MULTIBOUNTI.md`, section 3 : l'outil n'a pas de source, sa règle a été lue dans le désassemblage puis vérifiée contre l'outil réel).
//!
//! L'outil applique **douze fois** la même opération [`passe`], avec les paramètres de [`PASSES`] : `thick_gm(image, g, a, w)`. Pour chaque voxel V d'étiquette `w` situé à au moins 2 voxels du bord, s'il existe un voxel d'étiquette `a` à distance exactement 2 sur un axe (±x, ±y, ±z),
//! alors chacun des six voisins de V d'étiquette `g` prend l'étiquette `w`. Les lectures se font dans l'image d'entrée de la passe et les écritures dans une copie : l'ordre de parcours est sans effet, et une passe ne propage pas son propre résultat.
use ndarray::Array3;

/// Les douze passes `(g, a, w)` appliquées dans l'ordre : `g` vaut 37 (six premières) puis 38 (six dernières), `a` va de 13 à 24 et `w` de 1 à 12 en alternant les deux séries.
pub const PASSES: [(u8, u8, u8); 12] = [
    (37, 13, 1),
    (37, 15, 3),
    (37, 17, 5),
    (37, 19, 7),
    (37, 21, 9),
    (37, 23, 11),
    (38, 14, 2),
    (38, 16, 4),
    (38, 18, 6),
    (38, 20, 8),
    (38, 22, 10),
    (38, 24, 12),
];

/// Les six directions des voisins sur les axes.
const DIRECTIONS: [(isize, isize, isize); 6] = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];

/// Une passe `thick_gm(image, g, a, w)` (voir le module). Une image d'au plus 4 voxels sur un axe est rendue inchangée (la boucle de l'outil ne s'exécute pas).
pub fn passe(entree: &Array3<u8>, g: u8, a: u8, w: u8) -> Array3<u8> {
    let (nx, ny, nz) = entree.dim();
    let mut sortie = entree.clone();
    // Garde-fou contre le dépassement de `n - 2` (usize) ; au-delà de n = 2 la boucle `2..n - 2` est déjà vide jusqu'à n = 4, donc le seuil 5 est indiscernable de 4 (mutant équivalent).
    if nx < 5 || ny < 5 || nz < 5 {
        return sortie;
    }
    let lire = |x: usize, y: usize, z: usize, d: (isize, isize, isize), pas: isize| -> u8 {
        entree[[
            (x as isize + pas * d.0) as usize,
            (y as isize + pas * d.1) as usize,
            (z as isize + pas * d.2) as usize,
        ]]
    };
    for x in 2..nx - 2 {
        for y in 2..ny - 2 {
            for z in 2..nz - 2 {
                if entree[[x, y, z]] != w || !DIRECTIONS.iter().any(|&d| lire(x, y, z, d, 2) == a) {
                    continue;
                }
                for &d in &DIRECTIONS {
                    if lire(x, y, z, d, 1) == g {
                        sortie[[
                            (x as isize + d.0) as usize,
                            (y as isize + d.1) as usize,
                            (z as isize + d.2) as usize,
                        ]] = w;
                    }
                }
            }
        }
    }
    sortie
}

/// `thin-cortex-thick` : les douze [`PASSES`] à la suite.
pub fn amincir_le_cortex(etiquettes: &Array3<u8>) -> Array3<u8> {
    PASSES
        .iter()
        .fold(etiquettes.clone(), |image, &(g, a, w)| passe(&image, g, a, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometrie::Volume;
    use std::path::PathBuf;

    /// Volume de `n` voxels de côté rempli de `fond`.
    fn volume(n: usize, fond: u8) -> Array3<u8> {
        Array3::from_elem((n, n, n), fond)
    }

    /// Les douze passes sont celles de `main` de l'outil : `g` 37 puis 38, `a` de 13 à 24, `w` de 1 à 12, chaque `(a, w)` apparié (a = 12 + 2w − 1 pour les impairs, 12 + 2w pour les pairs).
    #[test]
    fn the_twelve_passes_are_the_ones_of_the_tool() {
        assert_eq!(PASSES.len(), 12);
        for (i, &(g, a, w)) in PASSES.iter().enumerate() {
            assert_eq!(g, if i < 6 { 37 } else { 38 }, "g de la passe {i}");
            assert_eq!(a, 13 + 2 * (i % 6) as u8 + (i / 6) as u8, "a de la passe {i}");
            assert_eq!(w, 1 + 2 * (i % 6) as u8 + (i / 6) as u8, "w de la passe {i}");
        }
    }

    /// Une passe : le voxel V (w = 1) est entouré d'un voisin g = 37 sur chacun des six axes ; un voxel a = 13 à distance 2 sur UN axe suffit à les convertir tous, quel que soit l'axe (six cas) ;
    /// sans voxel a à distance 2 (il est à distance 1 ou 3, ou d'une autre étiquette), rien ne change ; un voisin d'une autre étiquette que g n'est pas converti.
    #[test]
    fn a_pass_converts_all_g_neighbours_when_an_a_voxel_is_two_voxels_away() {
        let centre = 4usize;
        for (d, &(dx, dy, dz)) in DIRECTIONS.iter().enumerate() {
            let mut image = volume(9, 0);
            image[[centre, centre, centre]] = 1;
            for &(ex, ey, ez) in &DIRECTIONS {
                image[[
                    (centre as isize + ex) as usize,
                    (centre as isize + ey) as usize,
                    (centre as isize + ez) as usize,
                ]] = 37;
            }
            // un voisin d'une autre étiquette : celui de la direction opposée à d
            let oppose = (centre as isize - dx, centre as isize - dy, centre as isize - dz);
            image[[oppose.0 as usize, oppose.1 as usize, oppose.2 as usize]] = 99;
            // le voxel a à distance 2 dans la direction d (il remplace le zéro à cet endroit, il n'est pas un voisin)
            image[[
                (centre as isize + 2 * dx) as usize,
                (centre as isize + 2 * dy) as usize,
                (centre as isize + 2 * dz) as usize,
            ]] = 13;
            let sortie = passe(&image, 37, 13, 1);
            for (e, &(ex, ey, ez)) in DIRECTIONS.iter().enumerate() {
                let position = [
                    (centre as isize + ex) as usize,
                    (centre as isize + ey) as usize,
                    (centre as isize + ez) as usize,
                ];
                let attendu = if (ex, ey, ez) == (-dx, -dy, -dz) { 99 } else { 1 };
                assert_eq!(sortie[position], attendu, "direction de a {d}, voisin {e}");
            }
            assert_eq!(sortie[[centre, centre, centre]], 1);
        }
        // sans voxel a à distance 2 : inchangé
        let mut image = volume(9, 0);
        image[[4, 4, 4]] = 1;
        image[[5, 4, 4]] = 37;
        image[[4, 4, 3]] = 13; // distance 1 : ce n'est pas la bonne distance
        image[[4, 7, 4]] = 13; // distance 3
        image[[2, 4, 4]] = 14; // distance 2 mais mauvaise étiquette
        assert_eq!(passe(&image, 37, 13, 1), image);
    }

    /// Bords, sur CHACUN des trois axes : un voxel V d'étiquette w à moins de 2 voxels d'un bord (indice 0, 1, ou `n − 2`, `n − 1`) n'est pas examiné, même avec un voxel a à distance 2 ; aux indices permis, 2 et `n − 3`, il l'est.
    /// Le voisin g et le voxel a sont placés sur un axe perpendiculaire, à mi-hauteur, donc dans l'image. Une image d'au plus 4 voxels sur un axe est rendue inchangée.
    #[test]
    fn the_two_voxel_margin_is_respected_on_each_axis_and_small_images_are_unchanged() {
        let n = 9usize;
        for axe in 0..3usize {
            let perpendiculaire = (axe + 1) % 3;
            for (position, examine) in [
                (0, false),
                (1, false),
                (2, true),
                (n - 3, true),
                (n - 2, false),
                (n - 1, false),
            ] {
                let mut image = volume(n, 0);
                let mut v = [4usize; 3];
                v[axe] = position;
                let mut g = v;
                g[perpendiculaire] += 1;
                let mut a = v;
                a[perpendiculaire] += 2;
                image[v] = 1;
                image[g] = 37;
                image[a] = 13;
                let sortie = passe(&image, 37, 13, 1);
                assert_eq!(
                    sortie[g],
                    if examine { 1 } else { 37 },
                    "axe {axe}, V à l'indice {position}"
                );
            }
        }
        for dimensions in [(4, 9, 9), (9, 4, 9), (9, 9, 4), (3, 3, 3), (1, 1, 1)] {
            let petit = Array3::from_elem(dimensions, 1u8);
            assert_eq!(passe(&petit, 37, 13, 1), petit, "{dimensions:?}");
        }
    }

    /// Les lectures se font dans l'entrée de la passe : un voisin converti pendant la passe ne sert pas de voxel V à son tour (pas de propagation en chaîne).
    /// Les passes ne dépendent les unes des autres que par `g` : un voxel g convoité par deux passes revient à la première de la suite, car elle le convertit avant que la seconde ne le voie.
    #[test]
    fn a_pass_does_not_propagate_its_own_result_and_the_first_pass_takes_a_contested_voxel() {
        // pas de propagation : V (w = 1) en (4,4,4), g = 37 en (5,4,4) et (6,4,4), a = 13 en (4,4,2) ; seul (5,4,4) est converti, (6,4,4) ne l'est pas
        let mut image = volume(11, 0);
        image[[4, 4, 4]] = 1;
        image[[5, 4, 4]] = 37;
        image[[6, 4, 4]] = 37;
        image[[4, 4, 2]] = 13;
        let sortie = passe(&image, 37, 13, 1);
        assert_eq!((sortie[[5, 4, 4]], sortie[[6, 4, 4]]), (1, 37));
        // voxel contesté : g = 37 en (5,5,5), entre V1 (étiquette 1, a = 13 à distance 2) et V3 (étiquette 3, a = 15 à distance 2) ; la passe 1 (w = 1) le prend avant la passe 2 (w = 3)
        let mut image = volume(11, 0);
        image[[5, 5, 5]] = 37;
        image[[4, 5, 5]] = 1;
        image[[2, 5, 5]] = 13;
        image[[6, 5, 5]] = 3;
        image[[8, 5, 5]] = 15;
        let fin = amincir_le_cortex(&image);
        assert_eq!(fin[[5, 5, 5]], 1, "la première passe de la suite l'emporte");
        // en changeant seulement la seconde passe seule : le même voxel irait à l'étiquette 3
        assert_eq!(passe(&image, 37, 15, 3)[[5, 5, 5]], 3);
        // l'ordre est bien celui de la suite : lancer les passes à l'envers donnerait 3
        let a_l_envers = PASSES
            .iter()
            .rev()
            .fold(image.clone(), |im, &(g, a, w)| passe(&im, g, a, w));
        assert_eq!(a_l_envers[[5, 5, 5]], 3);
    }

    /// Le port est identique à l'outil réel (`thin-cortex-thick` de l'image de Multi-BOUNTI, exécuté dans un conteneur AMD64) : six volumes d'étiquettes aléatoires de 14×15×13 à 24×15×18 voxels (44 à 143 voxels modifiés chacun), puis les sorties officielles de la chaîne (`tr-multi-lab-reo-t2` → `sortie_officielle`), 0 voxel de différence attendu.
    #[test]
    #[ignore = "demande les sorties de l'outil dans data/multibounti (hors dépôt) : voir docs/MULTIBOUNTI.md, section 3"]
    fn the_port_is_identical_to_the_real_tool() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/multibounti");
        for n in 0..6 {
            let lire = |nom: String| Volume::lire(&racine.join("essais/tct").join(nom)).unwrap();
            let entree = lire(format!("in_{n}.nii.gz")).donnees.mapv(|v| v as u8);
            let officiel = lire(format!("out_{n}.nii.gz")).donnees.mapv(|v| v as u8);
            let differents = amincir_le_cortex(&entree)
                .iter()
                .zip(officiel.iter())
                .filter(|(a, b)| a != b)
                .count();
            let modifies = entree.iter().zip(officiel.iter()).filter(|(a, b)| a != b).count();
            println!("CORTEX aléatoire {n} : {differents} voxels différents ({modifies} modifiés par l'outil)");
            assert_eq!(differents, 0, "volume aléatoire {n}");
        }
        for id in ["calme1", "calme2", "calme3"] {
            let dossier = racine.join("ref").join(id);
            let (entree, sortie) = (
                dossier.join("tr-multi-lab-reo-t2.nii.gz"),
                dossier.join("sortie_officielle.nii.gz"),
            );
            if !entree.exists() || !sortie.exists() {
                println!("CORTEX {id} : références absentes (réseau 43 étiquettes pas encore calculé)");
                continue;
            }
            let entree = Volume::lire(&entree).unwrap().donnees.mapv(|v| v as u8);
            let officiel = Volume::lire(&sortie).unwrap().donnees.mapv(|v| v as u8);
            let differents = amincir_le_cortex(&entree)
                .iter()
                .zip(officiel.iter())
                .filter(|(a, b)| a != b)
                .count();
            let modifies = entree.iter().zip(officiel.iter()).filter(|(a, b)| a != b).count();
            println!("CORTEX {id} : {differents} voxels différents ({modifies} modifiés par l'outil)");
            assert_eq!(differents, 0, "{id}");
        }
    }
}
