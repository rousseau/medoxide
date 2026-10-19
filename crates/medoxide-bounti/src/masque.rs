//! Nettoyage du masque du cerveau, entre l'extraction (réseau 3D sur le cube de 128) et la segmentation des tissus (cube de 256).
//!
//! Étapes, dans l'ordre de la chaîne officielle (relevées et vérifiées contre les outils MIRTK de l'image Docker, voir `docs/BOUNTI.md`) :
//! 1. **plus grande composante connexe** du masque (connexité 26) ;
//! 2. **passage dans la grille du volume recadré** (celle de [`crate::geometrie::Volume::preparer`]) : le masque binaire est rééchantillonné par interpolation linéaire, et un voxel est dans le masque si la valeur interpolée dépasse 0,5 (MIRTK : conversion d'étiquettes `-labels`, qui garde l'étiquette de plus grande pseudo-probabilité) ;
//! 3. **dilatation de 4 itérations puis érosion de 2** (voisinage de 26 voxels, c'est-à-dire un cube 3 × 3 × 3 par itération) ;
//! 4. **masquage** du volume (0 hors du masque) puis **recadrage** sur la boîte englobante du masque.
//!
//! Particularité de MIRTK reproduite ici : à chaque itération de dilatation ou d'érosion, la couche extérieure de voxels (les voxels sur une face de l'image) **garde sa valeur** ; seuls les voxels intérieurs sont mis à jour, à partir de tous leurs voisins (y compris ceux de la couche extérieure).
use crate::geometrie::{Boite, ErreurGeometrie, Volume};
use ndarray::Array3;

/// Plus grande composante connexe (voisinage de 26 voxels) d'un masque ; à taille égale, celle dont le premier voxel (ordre `x`, `y`, `z`) vient en premier. Masque vide : masque vide.
pub fn plus_grande_composante(masque: &Array3<bool>) -> Array3<bool> {
    let (nx, ny, nz) = masque.dim();
    let mut etiquette = Array3::<u32>::zeros((nx, ny, nz));
    let mut tailles: Vec<usize> = Vec::new();
    let mut pile: Vec<(usize, usize, usize)> = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if !masque[[x, y, z]] || etiquette[[x, y, z]] != 0 {
                    continue;
                }
                let numero = tailles.len() as u32 + 1;
                let mut taille = 0usize;
                etiquette[[x, y, z]] = numero;
                pile.push((x, y, z));
                while let Some((i, j, k)) = pile.pop() {
                    taille += 1;
                    for di in -1i64..=1 {
                        for dj in -1i64..=1 {
                            for dk in -1i64..=1 {
                                let (a, b, c) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                                if a < 0 || b < 0 || c < 0 || a >= nx as i64 || b >= ny as i64 || c >= nz as i64 {
                                    continue;
                                }
                                let p = [a as usize, b as usize, c as usize];
                                if masque[p] && etiquette[p] == 0 {
                                    etiquette[p] = numero;
                                    pile.push((p[0], p[1], p[2]));
                                }
                            }
                        }
                    }
                }
                tailles.push(taille);
            }
        }
    }
    // la plus grande ; à taille égale, la première trouvée
    let mut meilleur: Option<(usize, usize)> = None;
    for (i, &t) in tailles.iter().enumerate() {
        if meilleur.is_none_or(|(_, mt)| t > mt) {
            meilleur = Some((i, t));
        }
    }
    match meilleur {
        None => Array3::from_elem((nx, ny, nz), false),
        Some((i, _)) => etiquette.mapv(|e| e == i as u32 + 1),
    }
}

/// Une itération de dilatation (`maximum = true`) ou d'érosion (`maximum = false`) en voisinage de 26 voxels, couche extérieure figée (voir la documentation du module).
/// Le maximum (ou le minimum) sur le cube 3 × 3 × 3 se calcule axe par axe : trois passes de 3 voxels.
fn iteration_morphologique(masque: &Array3<bool>, maximum: bool) -> Array3<bool> {
    let (nx, ny, nz) = masque.dim();
    let mut courant = masque.clone();
    for axe in 0..3 {
        let precedent = courant.clone();
        for ((i, j, k), valeur) in courant.indexed_iter_mut() {
            let n = [nx, ny, nz][axe];
            let p = [i, j, k][axe];
            let voisin = |decalage: i64| {
                let q = p as i64 + decalage;
                if q < 0 || q >= n as i64 {
                    // voisin hors de l'image : valeur neutre (faux pour un maximum, vrai pour un minimum). Elle n'influence que les voxels de la couche extérieure, remplacés plus bas par leur valeur d'entrée : aucun voxel intérieur ne la lit.
                    return !maximum;
                }
                let mut c = [i, j, k];
                c[axe] = q as usize;
                precedent[c]
            };
            let (a, b, c) = (voisin(-1), precedent[[i, j, k]], voisin(1));
            *valeur = if maximum { a || b || c } else { a && b && c };
        }
    }
    // la couche extérieure garde sa valeur d'entrée
    for ((i, j, k), valeur) in courant.indexed_iter_mut() {
        if i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 {
            *valeur = masque[[i, j, k]];
        }
    }
    courant
}

/// Dilatation de `iterations` itérations (voisinage de 26 voxels, couche extérieure figée).
pub fn dilater(masque: &Array3<bool>, iterations: usize) -> Array3<bool> {
    (0..iterations).fold(masque.clone(), |m, _| iteration_morphologique(&m, true))
}

/// Érosion de `iterations` itérations (voisinage de 26 voxels, couche extérieure figée).
pub fn eroder(masque: &Array3<bool>, iterations: usize) -> Array3<bool> {
    (0..iterations).fold(masque.clone(), |m, _| iteration_morphologique(&m, false))
}

/// Passe un masque binaire défini sur la grille de `source` (`affine_source`, mêmes axes que `masque`) dans la grille de `cible` : interpolation linéaire du masque (convention d'interpolation de [`Volume`]), voxel dans le masque si la valeur dépasse 0,5.
pub fn vers_la_grille(masque: &Array3<bool>, affine_source: &nalgebra::Matrix4<f64>, cible: &Volume) -> Array3<bool> {
    let source = Volume {
        donnees: masque.mapv(|m| if m { 1.0 } else { 0.0 }),
        affine: *affine_source,
    };
    source
        .reechantillonner(&cible.affine, cible.dimensions())
        .donnees
        .mapv(|v| v > 0.5)
}

/// Masque du cerveau dans la grille de `volume` à partir des étiquettes de l'extraction (`etiquettes` : valeur 1 pour le cerveau, sur une grille d'affine `affine_etiquettes`) : plus grande composante, passage dans la grille de `volume`, dilatation de 4 puis érosion de 2.
pub fn masque_du_cerveau(
    etiquettes: &Array3<f32>,
    affine_etiquettes: &nalgebra::Matrix4<f64>,
    volume: &Volume,
) -> Array3<bool> {
    let cerveau = etiquettes.mapv(|e| e == 1.0);
    let composante = plus_grande_composante(&cerveau);
    let dans_la_grille = vers_la_grille(&composante, affine_etiquettes, volume);
    eroder(&dilater(&dans_la_grille, 4), 2)
}

impl Volume {
    /// Volume dont les voxels hors de `masque` sont mis à 0.
    pub fn masquer(&self, masque: &Array3<bool>) -> Volume {
        let mut donnees = self.donnees.clone();
        for (valeur, &dedans) in donnees.iter_mut().zip(masque.iter()) {
            if !dedans {
                *valeur = 0.0;
            }
        }
        Volume {
            donnees,
            affine: self.affine,
        }
    }

    /// Boîte englobante (bornes incluses) des voxels vrais d'un masque, ou `None` s'il est vide.
    pub fn boite_du_masque(masque: &Array3<bool>) -> Option<Boite> {
        let mut boite: Option<Boite> = None;
        for ((i, j, k), &dedans) in masque.indexed_iter() {
            if dedans {
                boite = Some(match boite {
                    None => ([i, j, k], [i, j, k]),
                    Some((lo, hi)) => (
                        [lo[0].min(i), lo[1].min(j), lo[2].min(k)],
                        [hi[0].max(i), hi[1].max(j), hi[2].max(k)],
                    ),
                });
            }
        }
        boite
    }

    /// Masque le volume puis le recadre sur la boîte englobante du masque (dernière étape du nettoyage).
    ///
    /// # Erreurs
    /// [`ErreurGeometrie::Vide`] si le masque est vide.
    pub fn masquer_et_recadrer(&self, masque: &Array3<bool>) -> Result<Volume, ErreurGeometrie> {
        let boite = Volume::boite_du_masque(masque).ok_or(ErreurGeometrie::Vide)?;
        Ok(self.masquer(masque).recadrer(boite.0, boite.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    fn vide(n: usize) -> Array3<bool> {
        Array3::from_elem((n, n, n), false)
    }

    /// Composantes : un voxel relié seulement par une diagonale (voisinage de 26) appartient à la composante ; la plus grande est gardée, l'autre disparaît ; masque vide → masque vide ; à taille égale, la première trouvée.
    #[test]
    fn the_largest_component_uses_26_connectivity() {
        let mut m = vide(12);
        // grande composante : bloc 3×3×3 plus un voxel relié par une diagonale pure (offset (1,1,1))
        for x in 2..5 {
            for y in 2..5 {
                for z in 2..5 {
                    m[[x, y, z]] = true;
                }
            }
        }
        m[[5, 5, 5]] = true;
        // petite composante isolée
        m[[9, 9, 9]] = true;
        m[[9, 9, 10]] = true;
        let c = plus_grande_composante(&m);
        assert_eq!(c.iter().filter(|&&v| v).count(), 27 + 1, "bloc et voxel diagonal");
        assert!(c[[5, 5, 5]] && !c[[9, 9, 9]] && !c[[9, 9, 10]]);
        assert_eq!(plus_grande_composante(&vide(5)).iter().filter(|&&v| v).count(), 0);
        let mut egal = vide(8);
        egal[[1, 1, 1]] = true;
        egal[[6, 6, 6]] = true;
        let c = plus_grande_composante(&egal);
        assert!(c[[1, 1, 1]] && !c[[6, 6, 6]], "à taille égale, la première");
    }

    /// Dilatation et érosion : un voxel intérieur devient un cube de côté 2n+1 après n itérations (voisinage de 26), et l'érosion de ce cube le ramène exactement au voxel ; un voxel d'une face n'est ni dilaté vers l'extérieur ni érodé : la couche extérieure garde sa valeur (dilatation : un voxel de face voisin d'un voxel du masque reste faux ; érosion : un voxel de face vrai reste vrai).
    #[test]
    fn dilation_and_erosion_use_a_3x3x3_neighbourhood_and_freeze_the_outer_layer() {
        let mut m = vide(15);
        m[[7, 7, 7]] = true;
        for n in 1..=4usize {
            let d = dilater(&m, n);
            let cote = 2 * n + 1;
            assert_eq!(
                d.iter().filter(|&&v| v).count(),
                cote * cote * cote,
                "cube après {n} itérations"
            );
            assert!(d[[7 + n, 7 + n, 7 + n]] && d[[7 - n, 7 - n, 7 - n]] && !d[[7 + n + 1, 7, 7]]);
            let retour = eroder(&d, n);
            assert_eq!(retour, m, "l'érosion du cube de {n} itérations ramène le voxel");
        }
        // couche extérieure : voxel du masque sur la face x = 0, son voisin sur la face reste faux
        let mut face = vide(6);
        face[[0, 3, 3]] = true;
        let d = dilater(&face, 1);
        assert!(d[[0, 3, 3]], "le voxel de face garde sa valeur");
        assert!(!d[[0, 3, 4]] && !d[[0, 2, 2]], "la couche extérieure n'est pas dilatée");
        assert!(
            d[[1, 3, 3]] && d[[1, 2, 2]] && d[[1, 4, 4]],
            "l'intérieur voisin est dilaté"
        );
        assert!(
            !d[[1, 5, 5]] && !d[[1, 1, 1]] && !d[[2, 3, 3]],
            "loin du masque, rien n'est dilaté (la valeur neutre hors image ne doit pas contaminer l'intérieur)"
        );
        // les six faces : un voxel de face du masque garde sa valeur à la dilatation et son voisin de face ne devient pas vrai ; à l'érosion un voxel de face vrai aux voisins faux reste vrai
        for axe in 0..3 {
            for cote in [0usize, 5] {
                let place = |decalage: usize| {
                    let mut c = [3usize, 3, 3];
                    c[axe] = cote;
                    c[(axe + 1) % 3] += decalage;
                    c
                };
                let mut m = vide(6);
                m[place(0)] = true;
                let d = dilater(&m, 1);
                assert!(
                    d[place(0)] && !d[place(1)],
                    "face (axe {axe}, côté {cote}) : dilatation"
                );
                assert!(eroder(&m, 1)[place(0)], "face (axe {axe}, côté {cote}) : érosion");
            }
        }
        // érosion : tout le volume vrai ; la couche extérieure reste vraie et l'intérieur aussi (aucun voisin faux)
        let plein = Array3::from_elem((6, 6, 6), true);
        assert_eq!(eroder(&plein, 3), plein);
        // un voxel de face vrai dont les voisins sont faux reste vrai
        let mut f = vide(6);
        f[[5, 2, 2]] = true;
        assert!(eroder(&f, 2)[[5, 2, 2]], "voxel de face vrai : non érodé");
    }

    /// `masque_du_cerveau` : sur une grille identique à celle des étiquettes, un seul voxel du cerveau donne, après dilatation de 4 puis érosion de 2, exactement un cube de 5 × 5 × 5 voxels (9 − 2 × 2) centré sur ce voxel ; un second voxel isolé plus petit que la composante est écarté.
    #[test]
    fn the_brain_mask_dilates_by_four_and_erodes_by_two() {
        let affine = Matrix4::<f64>::identity();
        let volume = Volume {
            donnees: Array3::<f32>::zeros((24, 24, 24)),
            affine,
        };
        let mut etiquettes = Array3::<f32>::zeros((24, 24, 24));
        etiquettes[[12, 12, 12]] = 1.0;
        etiquettes[[12, 12, 13]] = 1.0; // composante de deux voxels
        etiquettes[[2, 2, 2]] = 1.0; // voxel isolé, plus petit
        let m = masque_du_cerveau(&etiquettes, &affine, &volume);
        // composante de 2 voxels : dilatation de 4 → 9 × 9 × 10, érosion de 2 → 5 × 5 × 6
        assert_eq!(m.iter().filter(|&&v| v).count(), 5 * 5 * 6);
        assert!(m[[10, 10, 10]] && m[[14, 14, 15]] && !m[[9, 12, 12]] && !m[[2, 2, 2]]);
    }

    /// Passage dans la grille : un cube plein de 8 voxels de 2 mm devient, dans une grille de 1 mm de même repère, le bloc de 16 voxels de côté qui le recouvre jusqu'à l'interpolation (à l'intérieur toujours vrai, à l'extérieur lointain faux) ; la valeur 0,5 exacte n'est pas dans le masque (seuil strict).
    #[test]
    fn the_mask_is_resampled_by_linear_interpolation_and_a_strict_threshold() {
        let affine_grossier = Matrix4::new(
            2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let mut grossier = Array3::from_elem((10, 10, 10), false);
        for x in 3..7 {
            for y in 3..7 {
                for z in 3..7 {
                    grossier[[x, y, z]] = true;
                }
            }
        }
        let fin = Volume {
            donnees: Array3::<f32>::zeros((20, 20, 20)),
            affine: Matrix4::new(
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ),
        };
        let m = vers_la_grille(&grossier, &affine_grossier, &fin);
        assert!(m[[10, 10, 10]] && m[[8, 8, 8]], "intérieur");
        assert!(!m[[0, 0, 0]] && !m[[19, 19, 19]] && !m[[2, 10, 10]], "extérieur");
        // sur l'axe x à y = z = 10 : positions fines x -> voxel grossier x/2 ; le masque grossier est vrai pour 3 ≤ i ≤ 6 : valeur interpolée > 0,5 pour x/2 dans ]2,5 ; 6,5[
        assert!(
            !m[[5, 10, 10]],
            "x = 5 → i = 2,5 : valeur 0,5 exactement, hors masque (seuil strict)"
        );
        assert!(m[[6, 10, 10]] && m[[12, 10, 10]], "x = 6 (i = 3) et x = 12 (i = 6)");
        assert!(!m[[13, 10, 10]], "x = 13 (i = 6,5) : valeur 0,5, hors masque");
        // valeurs interpolées de 0,45 et de 0,55 (grille fine de 0,1 mm le long de x, à y = z = 10 mm) : le seuil est 0,5, ni 0,4 ni 0,6
        let ligne = Volume {
            donnees: Array3::<f32>::zeros((100, 3, 3)),
            affine: Matrix4::new(
                0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 9.0, 0.0, 0.0, 1.0, 9.0, 0.0, 0.0, 0.0, 1.0,
            ),
        };
        let l = vers_la_grille(&grossier, &affine_grossier, &ligne);
        assert!(!l[[49, 1, 1]], "x = 4,9 (i = 2,45) : valeur 0,45, hors masque");
        assert!(l[[51, 1, 1]], "x = 5,1 (i = 2,55) : valeur 0,55, dans le masque");
    }

    /// Masquage et recadrage : hors du masque, 0 ; dans le masque, valeur d'origine ; le recadrage prend la boîte englobante du masque et décale l'origine de l'affine ; masque vide → erreur.
    #[test]
    fn masking_zeroes_the_outside_and_cropping_follows_the_mask() {
        let affine = Matrix4::new(
            0.5, 0.0, 0.0, 10.0, 0.0, 0.5, 0.0, 20.0, 0.0, 0.0, 0.5, 30.0, 0.0, 0.0, 0.0, 1.0,
        );
        let v = Volume {
            donnees: Array3::from_shape_fn((8, 9, 10), |(i, j, k)| 1.0 + (i + 10 * j + 100 * k) as f32),
            affine,
        };
        let mut m = Array3::from_elem((8, 9, 10), false);
        for (i, j, k) in [(2, 3, 4), (5, 4, 7), (3, 6, 5)] {
            m[[i, j, k]] = true;
        }
        let r = v.masquer_et_recadrer(&m).unwrap();
        assert_eq!(r.dimensions(), [4, 4, 4], "boîte (2..=5, 3..=6, 4..=7)");
        assert_eq!(r.donnees[[0, 0, 0]], v.donnees[[2, 3, 4]]);
        assert_eq!(r.donnees[[3, 1, 3]], v.donnees[[5, 4, 7]]);
        assert_eq!(
            r.donnees.iter().filter(|&&x| x != 0.0).count(),
            3,
            "seuls les voxels du masque survivent"
        );
        let attendu = v.monde(&nalgebra::Vector3::new(2.0, 3.0, 4.0));
        assert!((r.monde(&nalgebra::Vector3::zeros()) - attendu).norm() < 1e-12);
        assert!(v.masquer_et_recadrer(&Array3::from_elem((8, 9, 10), false)).is_err());
    }

    /// **Sous-étape 5c** : le nettoyage du masque contre les sorties de MIRTK (`scripts/bounti_mirtk_masque.sh`, dans `data/bounti/mirtk/<id>/`, hors dépôt) pour trois volumes. **Chaque étape part de l'entrée de MIRTK** (pas de la sortie de l'étape précédente du code Rust : une différence d'un voxel s'élargirait à la dilatation) :
    /// plus grande composante à partir des étiquettes du réseau d'extraction OFFICIEL, passage dans la grille, dilatation de 4, érosion de 2 ; au plus 1e-5 des voxels différents à chaque étape (MIRTK calcule les coordonnées en simple précision : un voxel de valeur interpolée à 4e-6 du seuil de 0,5 peut basculer) ;
    /// chaîne complète `masque_du_cerveau` : au moins 99,9 % de voxels identiques à MIRTK (critère de la sous-étape) ; image masquée et recadrée de mêmes dimensions et affine, valeurs égales à 1e-3 près ; cube de 256³ de l'image masquée à 1e-5 près après mise à l'échelle dans [0, 1].
    #[test]
    #[ignore = "demande les références MIRTK dans data/bounti/mirtk (hors dépôt)"]
    fn the_mask_cleaning_matches_mirtk() {
        let racine = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti/mirtk");
        for id in ["calme1", "calme2", "calme3"] {
            let lire = |nom: &str| {
                Volume::lire(&racine.join(id).join(format!("{nom}.nii.gz")))
                    .unwrap_or_else(|e| panic!("{id}/{nom} : {e}"))
            };
            let masque_de = |v: &Volume| v.donnees.mapv(|x| x > 0.0);
            let comparer = |nom: &str, mien: &Array3<bool>, mirtk: &Volume, tolerance: f64| {
                let different = mien
                    .iter()
                    .zip(mirtk.donnees.iter())
                    .filter(|(&a, &b)| a != (b > 0.0))
                    .count();
                println!(
                    "MASQUE {id} {nom} : {different} voxels différents sur {} ({} dans le masque)",
                    mien.len(),
                    mien.iter().filter(|&&v| v).count()
                );
                assert!(
                    different as f64 <= tolerance * mien.len() as f64,
                    "{id} {nom} : {different} voxels différents"
                );
            };
            let s3 = lire("s3_recadre");
            let etiquettes = lire("bet_etiquettes_cube");
            let composante_mirtk = lire("bet_composante");
            let bet_s3 = lire("bet_s3");
            let dilate_mirtk = lire("bet_dilate");
            let erode_mirtk = lire("bet_erode");
            comparer(
                "composante",
                &plus_grande_composante(&etiquettes.donnees.mapv(|e| e == 1.0)),
                &composante_mirtk,
                0.0,
            );
            comparer(
                "grille",
                &vers_la_grille(&masque_de(&composante_mirtk), &composante_mirtk.affine, &s3),
                &bet_s3,
                1e-5,
            );
            comparer("dilatation", &dilater(&masque_de(&bet_s3), 4), &dilate_mirtk, 0.0);
            comparer("érosion", &eroder(&masque_de(&dilate_mirtk), 2), &erode_mirtk, 0.0);
            // chaîne complète : critère de la sous-étape
            let complet = masque_du_cerveau(&etiquettes.donnees, &etiquettes.affine, &s3);
            comparer("chaîne complète", &complet, &erode_mirtk, 1e-3);
            // masquage, recadrage, cube : à partir du masque de MIRTK
            let recadre = s3.masquer_et_recadrer(&masque_de(&erode_mirtk)).unwrap();
            let mirtk = lire("masque_recadre");
            assert_eq!(
                recadre.dimensions(),
                mirtk.dimensions(),
                "{id} masque_recadre : dimensions"
            );
            assert!(
                (recadre.affine - mirtk.affine).abs().max() < 1e-3,
                "{id} masque_recadre : affine"
            );
            let ecart = recadre
                .donnees
                .iter()
                .zip(mirtk.donnees.iter())
                .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
            println!(
                "MASQUE {id} masque_recadre : dimensions {:?}, écart d'intensité {ecart:.2e}",
                recadre.dimensions()
            );
            assert!(ecart < 1e-3, "{id} masque_recadre : écart {ecart}");
            let cube = mirtk.cube(256);
            let mirtk_cube = lire("cube256_masque");
            let (min, max) = mirtk_cube
                .donnees
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
            assert_eq!(cube.dimensions(), mirtk_cube.dimensions());
            assert!(
                (cube.affine - mirtk_cube.affine).abs().max() < 1e-3,
                "{id} cube256_masque : affine"
            );
            let ecart = cube
                .donnees
                .iter()
                .zip(mirtk_cube.donnees.iter())
                .fold(0.0_f32, |m, (a, b)| m.max((a - (b - min) / (max - min)).abs()));
            println!("MASQUE {id} cube256_masque : écart d'intensité {ecart:.2e}");
            assert!(ecart < 1e-5, "{id} cube256_masque : écart {ecart}");
        }
    }
}
