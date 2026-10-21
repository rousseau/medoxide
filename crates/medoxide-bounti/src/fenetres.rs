//! Inférence par **fenêtres glissantes** (`sliding_window_inference` de MONAI, mode `constant`), comme dans le script officiel de Multi-BOUNTI : fenêtres cubiques de 128 voxels qui se recouvrent de 80 %, le réseau est appliqué à chacune,
//! les logits sont additionnés là où les fenêtres se chevauchent, divisés par le nombre de fenêtres qui couvrent chaque voxel, et l'étiquette est le canal de plus grand logit moyen.
//!
//! Pourquoi : le réseau 43 étiquettes ne voit qu'un cube de 128³ ; le volume de 256³ est parcouru par 7 × 7 × 7 = 343 fenêtres. Les poids de MONAI sont uniformes (mode `constant`) : chaque fenêtre compte pour autant partout.
//!
//! Le plan des fenêtres est celui de `dense_patch_slices` de MONAI (voir [`departs`]), vérifié contre MONAI sur plusieurs tailles. Le réseau est un paramètre (une fermeture), ce qui permet de tester la logique d'accumulation sans GPU.
use ndarray::{s, Array3, Array4, ArrayView4};

/// Erreurs de l'inférence par fenêtres.
#[derive(Debug, PartialEq)]
pub enum ErreurFenetres {
    /// L'image est plus petite que la fenêtre sur un axe (MONAI la complèterait de zéros ; ici elle est toujours déjà mise à la taille voulue par `pad-3d`).
    ImageTropPetite {
        /// Taille de l'image.
        image: [usize; 3],
        /// Taille de la fenêtre.
        fenetre: usize,
    },
    /// Le réseau a rendu un nombre de valeurs différent de `canaux × fenêtre³`.
    SortieInattendue {
        /// Nombre de valeurs attendu.
        attendu: usize,
        /// Nombre de valeurs obtenu.
        obtenu: usize,
    },
}

impl std::fmt::Display for ErreurFenetres {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErreurFenetres::ImageTropPetite { image, fenetre } => {
                write!(f, "image {image:?} plus petite que la fenêtre de {fenetre} voxels")
            }
            ErreurFenetres::SortieInattendue { attendu, obtenu } => {
                write!(f, "le réseau a rendu {obtenu} valeurs au lieu de {attendu}")
            }
        }
    }
}

impl std::error::Error for ErreurFenetres {}

/// Départs des fenêtres sur UN axe de `taille` voxels, pour une fenêtre de `fenetre` voxels et un `recouvrement` entre 0 et 1 : `dense_patch_slices` de MONAI.
///
/// Le pas est `int(fenetre × (1 − recouvrement))` (au moins 1 ; la fenêtre entière si elle couvre déjà l'axe). Il y a autant de fenêtres qu'il en faut pour atteindre le bord, et la dernière est ramenée contre le bord (elle recouvre alors davantage l'avant-dernière).
/// Pour 256 voxels, une fenêtre de 128 et un recouvrement de 0,8 : pas de 25, départs 0, 25, 50, 75, 100, 125 et 128.
pub fn departs(taille: usize, fenetre: usize, recouvrement: f64) -> Vec<usize> {
    let fenetre = fenetre.min(taille);
    let pas = if fenetre == taille {
        fenetre
    } else {
        ((fenetre as f64 * (1.0 - recouvrement)) as usize).max(1)
    };
    let candidats = taille.div_ceil(pas);
    let nombre = (0..candidats)
        .find(|d| d * pas + fenetre >= taille)
        .map_or(1, |d| d + 1);
    (0..nombre)
        .map(|i| {
            let depart = i * pas;
            depart - (depart + fenetre).saturating_sub(taille)
        })
        .collect()
}

/// Étiquettes de `image` (`[X, Y, Z]`, chaque axe d'au moins `fenetre` voxels) par fenêtres glissantes : `reseau` reçoit une fenêtre (`[fenetre, fenetre, fenetre]`, copie contiguë) et rend ses logits
/// `[canaux, fenetre, fenetre, fenetre]` à plat, dans cet ordre (canal le plus lent, z le plus rapide). Les fenêtres sont parcourues avec x le plus lent, comme MONAI.
/// Résultat : le canal de plus grand logit moyen (le premier en cas d'égalité, comme `torch.argmax`), moyenne = somme des logits des fenêtres divisée par le nombre de fenêtres qui couvrent le voxel.
///
/// # Erreurs
/// [`ErreurFenetres::ImageTropPetite`] si un axe est plus court que la fenêtre ; [`ErreurFenetres::SortieInattendue`] si le réseau ne rend pas `canaux × fenêtre³` valeurs.
pub fn etiquettes_par_fenetres(
    image: &Array3<f32>,
    fenetre: usize,
    recouvrement: f64,
    canaux: usize,
    mut reseau: impl FnMut(&Array3<f32>) -> Vec<f32>,
) -> Result<Array3<u8>, ErreurFenetres> {
    let (nx, ny, nz) = image.dim();
    if nx < fenetre || ny < fenetre || nz < fenetre {
        return Err(ErreurFenetres::ImageTropPetite {
            image: [nx, ny, nz],
            fenetre,
        });
    }
    let (dx, dy, dz) = (
        departs(nx, fenetre, recouvrement),
        departs(ny, fenetre, recouvrement),
        departs(nz, fenetre, recouvrement),
    );
    let mut somme = Array4::<f32>::zeros((canaux, nx, ny, nz));
    let mut couverture = Array3::<f32>::zeros((nx, ny, nz));
    for &x in &dx {
        for &y in &dy {
            for &z in &dz {
                let fenetre_image = image
                    .slice(s![x..x + fenetre, y..y + fenetre, z..z + fenetre])
                    .to_owned();
                let logits = reseau(&fenetre_image);
                let attendu = canaux * fenetre * fenetre * fenetre;
                if logits.len() != attendu {
                    return Err(ErreurFenetres::SortieInattendue {
                        attendu,
                        obtenu: logits.len(),
                    });
                }
                let vue = ArrayView4::from_shape((canaux, fenetre, fenetre, fenetre), &logits).expect("forme vérifiée");
                let mut cible = somme.slice_mut(s![.., x..x + fenetre, y..y + fenetre, z..z + fenetre]);
                cible += &vue;
                let mut compte = couverture.slice_mut(s![x..x + fenetre, y..y + fenetre, z..z + fenetre]);
                compte += 1.0;
            }
        }
    }
    // moyenne puis argmax, canal par canal pour parcourir la mémoire dans l'ordre
    let mut meilleur = Array3::<f32>::from_elem((nx, ny, nz), f32::NEG_INFINITY);
    let mut etiquettes = Array3::<u8>::zeros((nx, ny, nz));
    for c in 0..canaux {
        let plan = somme.index_axis(ndarray::Axis(0), c);
        for (((valeur, compte), meilleur), etiquette) in plan
            .iter()
            .zip(couverture.iter())
            .zip(meilleur.iter_mut())
            .zip(etiquettes.iter_mut())
        {
            let moyenne = valeur / compte;
            if moyenne > *meilleur {
                *meilleur = moyenne;
                *etiquette = c as u8;
            }
        }
    }
    Ok(etiquettes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Départs sur un axe identiques à ceux de `dense_patch_slices` de MONAI 1.4.0 (valeurs calculées par MONAI lui-même, `scripts/` : voir `docs/MULTIBOUNTI.md`), pour des tailles, des fenêtres et des recouvrements variés :
    /// image égale à la fenêtre, un peu plus grande, grande, avec et sans reste, recouvrement de 0 à 0,9, pas arrondi à 1.
    #[test]
    fn the_window_starts_match_monai() {
        let cas: &[(usize, usize, f64, &[usize])] = &[
            (256, 128, 0.8, &[0, 25, 50, 75, 100, 125, 128]),
            (128, 128, 0.8, &[0]),
            (200, 128, 0.8, &[0, 25, 50, 72]),
            (257, 128, 0.8, &[0, 25, 50, 75, 100, 125, 129]),
            (130, 128, 0.8, &[0, 2]),
            (300, 128, 0.5, &[0, 64, 128, 172]),
            (256, 96, 0.25, &[0, 72, 144, 160]),
            (
                150,
                64,
                0.9,
                &[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 86],
            ),
            (64, 64, 0.5, &[0]),
            (65, 64, 0.0, &[0, 1]),
            (
                100,
                10,
                0.95,
                &[
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                ],
            ),
        ];
        for &(taille, fenetre, recouvrement, attendu) in cas {
            assert_eq!(
                departs(taille, fenetre, recouvrement),
                attendu,
                "taille {taille}, fenêtre {fenetre}, recouvrement {recouvrement}"
            );
        }
    }

    /// Volume dont chaque voxel porte un entier distinct de 0 à 4 (motif déterministe), pour des tests exacts.
    fn volume_a_motif(n: (usize, usize, usize)) -> Array3<f32> {
        Array3::from_shape_fn(n, |(i, j, k)| ((i * 3 + j * 5 + k * 7) % 5) as f32)
    }

    /// Réseau PONCTUEL (le logit d'un voxel ne dépend que de sa valeur) : chaque voxel a le même logit dans toutes les fenêtres qui le couvrent, donc la moyenne est exacte et le résultat est celui du réseau appliqué voxel par voxel ;
    /// ce qui contrôle qu'aucun voxel n'est oublié (couverture complète), que la division par le nombre de fenêtres est juste et que l'argmax prend le premier canal en cas d'égalité (le canal 3 est identique au canal 1 : il l'égale partout et n'est jamais retenu).
    #[test]
    fn a_pointwise_network_gives_the_pointwise_labels_with_full_coverage() {
        let image = volume_a_motif((13, 11, 12));
        let canaux = 4usize;
        let f = 8usize;
        // logits : canal 0 = valeur, canal 1 = 3 − valeur, canal 2 = 2,5 (constant), canal 3 = 3 − valeur (identique au canal 1)
        let reseau = |fenetre: &Array3<f32>| -> Vec<f32> {
            let mut sortie = Vec::with_capacity(canaux * f * f * f);
            for c in 0..canaux {
                for &v in fenetre.iter() {
                    sortie.push(match c {
                        0 => v,
                        1 => 3.0 - v,
                        2 => 2.5,
                        _ => 3.0 - v,
                    });
                }
            }
            sortie
        };
        let obtenu = etiquettes_par_fenetres(&image, f, 0.5, canaux, reseau).unwrap();
        let attendu = image.mapv(|v| {
            let logits = [v, 3.0 - v, 2.5, 3.0 - v];
            let mut meilleur = 0;
            for c in 1..4 {
                if logits[c] > logits[meilleur] {
                    meilleur = c;
                }
            }
            meilleur as u8
        });
        assert_eq!(obtenu, attendu);
        // les étiquettes 0, 1 et 2 sont présentes (le test serait vide autrement) et l'étiquette 3 ne l'est jamais (égalités : premier canal)
        for e in [0u8, 1, 2] {
            assert!(attendu.iter().any(|&v| v == e), "étiquette {e} absente du test");
        }
        assert!(attendu.iter().all(|&v| v != 3));
    }

    /// Réseau qui dépend de la FENÊTRE (logit = valeur + moyenne de la fenêtre) : la moyenne sur les fenêtres change selon leur position ; résultat comparé à un calcul naïf voxel par voxel (pour chaque voxel, on parcourt les fenêtres du plan de MONAI qui le couvrent).
    /// Volume de 12 × 10 × 11 voxels, fenêtre de 8, recouvrement 0,5 (pas de 4) : départs {0, 4}, {0, 2}, {0, 3}.
    #[test]
    // calcul naïf volontaire : les indices (c, i, j, k) servent à indexer plusieurs tableaux en parallèle, comme dans la description de MONAI
    #[allow(clippy::needless_range_loop)]
    fn a_window_dependent_network_matches_a_naive_per_voxel_average() {
        let image = Array3::from_shape_fn((12, 10, 11), |(i, j, k)| (((i * 7 + j * 3 + k * 5) % 11) as f32) * 0.5);
        let (canaux, f) = (3usize, 8usize);
        let fenetre_logits = |fenetre: &Array3<f32>| -> Vec<f32> {
            let moyenne = fenetre.iter().sum::<f32>() / fenetre.len() as f32;
            let mut sortie = Vec::new();
            for c in 0..canaux {
                for &v in fenetre.iter() {
                    sortie.push(match c {
                        0 => v + moyenne,
                        1 => 2.0 * v - moyenne,
                        _ => 3.0 - (v - moyenne).abs(),
                    });
                }
            }
            sortie
        };
        let obtenu = etiquettes_par_fenetres(&image, f, 0.5, canaux, fenetre_logits).unwrap();
        // calcul naïf
        let (dx, dy, dz) = (departs(12, f, 0.5), departs(10, f, 0.5), departs(11, f, 0.5));
        assert_eq!(
            (dx.clone(), dy.clone(), dz.clone()),
            (vec![0, 4], vec![0, 2], vec![0, 3])
        );
        let mut somme = vec![vec![0.0f32; 12 * 10 * 11]; canaux];
        let mut compte = vec![0.0f32; 12 * 10 * 11];
        for &x in &dx {
            for &y in &dy {
                for &z in &dz {
                    let fen = image.slice(s![x..x + f, y..y + f, z..z + f]).to_owned();
                    let logits = fenetre_logits(&fen);
                    for i in 0..f {
                        for j in 0..f {
                            for k in 0..f {
                                let g = ((x + i) * 10 + (y + j)) * 11 + (z + k);
                                compte[g] += 1.0;
                                for c in 0..canaux {
                                    somme[c][g] += logits[((c * f + i) * f + j) * f + k];
                                }
                            }
                        }
                    }
                }
            }
        }
        for i in 0..12 {
            for j in 0..10 {
                for k in 0..11 {
                    let g = (i * 10 + j) * 11 + k;
                    let mut meilleur = (f32::NEG_INFINITY, 0u8);
                    for c in 0..canaux {
                        let m = somme[c][g] / compte[g];
                        if m > meilleur.0 {
                            meilleur = (m, c as u8);
                        }
                    }
                    assert_eq!(obtenu[[i, j, k]], meilleur.1, "voxel ({i}, {j}, {k})");
                }
            }
        }
        // le test discrimine : au moins deux étiquettes différentes
        let distinctes: std::collections::HashSet<u8> = obtenu.iter().copied().collect();
        assert!(distinctes.len() >= 2, "{distinctes:?}");
    }

    /// Le réseau est appelé une fois par fenêtre (7 × 7 × 7 = 343 à 256³ avec recouvrement 0,8, comme MONAI ; 1 fenêtre à la taille exacte), chaque fenêtre est une copie de `fenetre³` voxels, et les erreurs sont signalées : image trop petite, sortie de mauvaise longueur.
    #[test]
    fn the_network_is_called_once_per_window_and_errors_are_reported() {
        let mut appels = 0usize;
        let image = Array3::<f32>::zeros((20, 20, 20));
        let _ = etiquettes_par_fenetres(&image, 8, 0.8, 2, |fen| {
            appels += 1;
            assert_eq!(fen.dim(), (8, 8, 8));
            vec![0.0; 2 * 512]
        })
        .unwrap();
        assert_eq!(appels, departs(20, 8, 0.8).len().pow(3));
        assert_eq!(departs(256, 128, 0.8).len().pow(3), 343);
        let exact = Array3::<f32>::zeros((8, 8, 8));
        let mut un = 0;
        etiquettes_par_fenetres(&exact, 8, 0.8, 1, |_| {
            un += 1;
            vec![0.0; 512]
        })
        .unwrap();
        assert_eq!(un, 1);
        let petite = Array3::<f32>::zeros((8, 9, 7));
        assert_eq!(
            etiquettes_par_fenetres(&petite, 8, 0.5, 1, |_| vec![0.0; 512]),
            Err(ErreurFenetres::ImageTropPetite {
                image: [8, 9, 7],
                fenetre: 8
            })
        );
        assert_eq!(
            etiquettes_par_fenetres(&exact, 8, 0.5, 2, |_| vec![0.0; 10]),
            Err(ErreurFenetres::SortieInattendue {
                attendu: 1024,
                obtenu: 10
            })
        );
        assert!(ErreurFenetres::SortieInattendue { attendu: 1, obtenu: 2 }
            .to_string()
            .contains("2 valeurs"));
    }
}
