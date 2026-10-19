//! Géométrie de la chaîne BOUNTI : de la reconstruction 3D au cube que lisent les réseaux.
//!
//! Étapes, dans l'ordre de la chaîne officielle (relevées et vérifiées contre l'outil MIRTK de l'image Docker, voir `docs/BOUNTI.md`) :
//! 1. **recadrage** sur la boîte englobante des voxels strictement supérieurs à 0,005 ([`Volume::recadrer_sur_seuil`]) ;
//! 2. **rééchantillonnage linéaire dans la grille de référence** : 300 × 369 × 300 voxels de 0,6 mm, orientation LAS, centrée sur le centre du volume recadré ([`Volume::vers_grille_de_reference`]) ;
//! 3. **second recadrage** sur l'étendue physique du masque de l'étape 1 : on garde les voxels dont le centre est dans cette boîte ([`Volume::recadrer_sur_boite_physique`]) ;
//! 4. **cube** de `n`³ voxels isotropes de taille `1,01 × plus petit voxel × plus grande dimension / n`, centré sur le centre du volume, intensités ramenées dans [0, 1] ([`Volume::cube`]).
//!
//! Convention d'interpolation : trilinéaire en coordonnées monde ; une position à moins d'un demi-voxel du bord du volume prend la valeur du voxel de bord (le domaine d'une image s'étend d'un demi-voxel au-delà de ses centres extrêmes) ; au-delà, 0.
use medoxide_core::{affine_nifti, read_volume, CoreError};
use nalgebra::{Matrix4, Vector3, Vector4};
use ndarray::Array3;
use std::path::Path;

/// Seuil de la chaîne officielle : les voxels d'intensité strictement supérieure définissent la boîte du premier recadrage.
pub const SEUIL_RECADRAGE: f32 = 0.005;

/// Dimensions de la grille de référence (voxels).
pub const DIMENSIONS_REFERENCE: [usize; 3] = [300, 369, 300];

/// Taille d'un voxel de la grille de référence (mm).
pub const VOXEL_REFERENCE_MM: f64 = 0.6;

/// Boîte de voxels (bornes basse et haute, incluses) d'un volume.
pub type Boite = ([usize; 3], [usize; 3]);

/// Erreurs de la géométrie.
#[derive(Debug)]
pub enum ErreurGeometrie {
    /// Aucun voxel au-dessus du seuil : volume vide.
    Vide,
    /// Lecture du fichier NIfTI.
    Lecture(CoreError),
}

impl From<CoreError> for ErreurGeometrie {
    fn from(e: CoreError) -> Self {
        ErreurGeometrie::Lecture(e)
    }
}

impl std::fmt::Display for ErreurGeometrie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErreurGeometrie::Vide => write!(f, "volume vide (aucun voxel au-dessus du seuil)"),
            ErreurGeometrie::Lecture(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ErreurGeometrie {}

/// Volume 3D : voxels `[x, y, z]` et affine voxel → mm.
#[derive(Debug, Clone)]
pub struct Volume {
    /// Intensités.
    pub donnees: Array3<f32>,
    /// Affine voxel → mm : `[x, y, z, 1]ᵀ = affine · [i, j, k, 1]ᵀ`.
    pub affine: Matrix4<f64>,
}

impl Volume {
    /// Lit un volume NIfTI : voxels et affine (règle `sform`, sinon `qform`, sinon `pixdim`, voir [`medoxide_core::affine_nifti`]). Les valeurs non finies sont remplacées par 0.
    ///
    /// # Erreurs
    /// [`ErreurGeometrie::Lecture`] si le fichier est illisible ou n'est pas un volume 3D.
    pub fn lire(chemin: &Path) -> Result<Volume, ErreurGeometrie> {
        let mut donnees = read_volume(chemin)?;
        donnees.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
        let m = affine_nifti(chemin)?.matrice;
        let affine = Matrix4::from_fn(|ligne, colonne| m[ligne][colonne]);
        Ok(Volume { donnees, affine })
    }

    /// Nombre de voxels par axe.
    pub fn dimensions(&self) -> [usize; 3] {
        let d = self.donnees.dim();
        [d.0, d.1, d.2]
    }

    /// Position en mm du voxel (fractionnaire) `voxel`.
    pub fn monde(&self, voxel: &Vector3<f64>) -> Vector3<f64> {
        (self.affine * Vector4::new(voxel.x, voxel.y, voxel.z, 1.0)).xyz()
    }

    /// Position en mm du centre de la grille, c'est-à-dire du voxel fractionnaire `(dimensions − 1) / 2`.
    pub fn centre(&self) -> Vector3<f64> {
        let d = self.dimensions();
        self.monde(&Vector3::new(
            (d[0] as f64 - 1.0) / 2.0,
            (d[1] as f64 - 1.0) / 2.0,
            (d[2] as f64 - 1.0) / 2.0,
        ))
    }

    /// Intensité interpolée (trilinéaire) à la position `voxel` exprimée en coordonnées de voxel de ce volume : valeur du voxel de bord à moins d'un demi-voxel du bord, 0 au-delà.
    fn echantillonner_voxel(&self, voxel: &Vector3<f64>) -> f32 {
        let d = self.dimensions();
        let mut base = [0usize; 3];
        let mut fraction = [0.0_f64; 3];
        for axe in 0..3 {
            let v = voxel[axe];
            let dernier = d[axe] as f64 - 1.0;
            if v < -0.5 || v > dernier + 0.5 {
                return 0.0;
            }
            let v = v.clamp(0.0, dernier);
            let plancher = (v.floor() as usize).min(d[axe].saturating_sub(2));
            base[axe] = plancher;
            fraction[axe] = if d[axe] == 1 { 0.0 } else { v - plancher as f64 };
        }
        let mut somme = 0.0_f64;
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    let poids = (if dx == 1 { fraction[0] } else { 1.0 - fraction[0] })
                        * (if dy == 1 { fraction[1] } else { 1.0 - fraction[1] })
                        * (if dz == 1 { fraction[2] } else { 1.0 - fraction[2] });
                    if poids != 0.0 {
                        let (i, j, k) = (
                            (base[0] + dx).min(d[0] - 1),
                            (base[1] + dy).min(d[1] - 1),
                            (base[2] + dz).min(d[2] - 1),
                        );
                        somme += poids * f64::from(self.donnees[[i, j, k]]);
                    }
                }
            }
        }
        somme as f32
    }

    /// Intensité interpolée à la position `monde` (mm) ; voir la convention d'interpolation du module.
    pub fn echantillonner(&self, monde: &Vector3<f64>) -> f32 {
        let inverse = self.affine.try_inverse().expect("affine inversible");
        let v = (inverse * Vector4::new(monde.x, monde.y, monde.z, 1.0)).xyz();
        self.echantillonner_voxel(&v)
    }

    /// Rééchantillonne ce volume sur la grille d'affine `affine_cible` et de dimensions `dimensions` (interpolation trilinéaire, voir la convention du module).
    pub fn reechantillonner(&self, affine_cible: &Matrix4<f64>, dimensions: [usize; 3]) -> Volume {
        let vers_source = self.affine.try_inverse().expect("affine inversible") * affine_cible;
        let mut donnees = Array3::<f32>::zeros((dimensions[0], dimensions[1], dimensions[2]));
        for ((i, j, k), valeur) in donnees.indexed_iter_mut() {
            let v = (vers_source * Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
            *valeur = self.echantillonner_voxel(&v);
        }
        Volume {
            donnees,
            affine: *affine_cible,
        }
    }

    /// Sous-volume des voxels `debut..=fin` (bornes incluses) ; l'affine est décalée en conséquence.
    pub fn recadrer(&self, debut: [usize; 3], fin: [usize; 3]) -> Volume {
        let donnees = self
            .donnees
            .slice(ndarray::s![debut[0]..=fin[0], debut[1]..=fin[1], debut[2]..=fin[2]])
            .to_owned();
        let mut affine = self.affine;
        let origine = self.monde(&Vector3::new(debut[0] as f64, debut[1] as f64, debut[2] as f64));
        affine.fixed_view_mut::<3, 1>(0, 3).copy_from(&origine);
        Volume { donnees, affine }
    }

    /// Boîte englobante (bornes incluses) des voxels strictement supérieurs à `seuil`, ou `None` s'il n'y en a pas.
    pub fn boite_englobante(&self, seuil: f32) -> Option<Boite> {
        let mut boite: Option<Boite> = None;
        for ((i, j, k), &v) in self.donnees.indexed_iter() {
            if v > seuil {
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

    /// **Étape 1** : recadre sur la boîte englobante des voxels > `seuil` (la chaîne officielle prend [`SEUIL_RECADRAGE`]). Rend aussi cette boîte (voxels de ce volume), utile à l'étape 3.
    ///
    /// # Erreurs
    /// [`ErreurGeometrie::Vide`] si aucun voxel ne dépasse le seuil.
    pub fn recadrer_sur_seuil(&self, seuil: f32) -> Result<(Volume, Boite), ErreurGeometrie> {
        let boite = self.boite_englobante(seuil).ok_or(ErreurGeometrie::Vide)?;
        Ok((self.recadrer(boite.0, boite.1), boite))
    }

    /// **Étape 2** : rééchantillonne dans la grille de référence (300 × 369 × 300 voxels de 0,6 mm, orientation LAS, c'est-à-dire axe x de sens opposé), centrée sur le centre de ce volume.
    pub fn vers_grille_de_reference(&self) -> Volume {
        let mut affine = Matrix4::<f64>::identity();
        affine[(0, 0)] = -VOXEL_REFERENCE_MM;
        affine[(1, 1)] = VOXEL_REFERENCE_MM;
        affine[(2, 2)] = VOXEL_REFERENCE_MM;
        let milieu = Vector3::new(
            (DIMENSIONS_REFERENCE[0] as f64 - 1.0) / 2.0,
            (DIMENSIONS_REFERENCE[1] as f64 - 1.0) / 2.0,
            (DIMENSIONS_REFERENCE[2] as f64 - 1.0) / 2.0,
        );
        let decalage = self.centre() - affine.fixed_view::<3, 3>(0, 0) * milieu;
        affine.fixed_view_mut::<3, 1>(0, 3).copy_from(&decalage);
        self.reechantillonner(&affine, DIMENSIONS_REFERENCE)
    }

    /// **Étape 3** : garde les voxels de ce volume dont le centre est dans la boîte physique `boite` (bornes de voxels d'un volume `origine`, élargies d'un demi-voxel : bords des voxels). Sur chaque axe, début = ⌈borne basse⌉ et fin = ⌊borne haute⌋ en indices de ce volume.
    ///
    /// # Erreurs
    /// [`ErreurGeometrie::Vide`] si la boîte ne contient aucun centre de voxel.
    pub fn recadrer_sur_boite_physique(&self, origine: &Volume, boite: Boite) -> Result<Volume, ErreurGeometrie> {
        let inverse = self.affine.try_inverse().expect("affine inversible");
        let (mut bas, mut haut) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for coin in 0..8 {
            let choix = |axe: usize, bit: usize| {
                if (coin >> bit) & 1 == 0 {
                    boite.0[axe] as f64 - 0.5
                } else {
                    boite.1[axe] as f64 + 0.5
                }
            };
            let monde = origine.monde(&Vector3::new(choix(0, 0), choix(1, 1), choix(2, 2)));
            let v = (inverse * Vector4::new(monde.x, monde.y, monde.z, 1.0)).xyz();
            for axe in 0..3 {
                bas[axe] = bas[axe].min(v[axe]);
                haut[axe] = haut[axe].max(v[axe]);
            }
        }
        let d = self.dimensions();
        let (mut debut, mut fin) = ([0usize; 3], [0usize; 3]);
        for axe in 0..3 {
            // petite tolérance : un bord tombant exactement sur un centre de voxel (à l'arrondi flottant près) l'inclut
            let b = (bas[axe] - 1e-9).ceil().max(0.0);
            let h = (haut[axe] + 1e-9).floor().min(d[axe] as f64 - 1.0);
            if h < b {
                return Err(ErreurGeometrie::Vide);
            }
            (debut[axe], fin[axe]) = (b as usize, h as usize);
        }
        Ok(self.recadrer(debut, fin))
    }

    /// **Étape 4** (`prepare-for-monai` de la chaîne officielle) : cube de `n`³ voxels isotropes de taille `1,01 × plus petit voxel × plus grande dimension / n`, de même orientation que ce volume et centré sur son centre, interpolation trilinéaire,
    /// puis intensités ramenées dans [0, 1] par `(v − min) / (max − min)` (un volume constant donne 0).
    pub fn cube(&self, n: usize) -> Volume {
        let colonnes: Vec<Vector3<f64>> = (0..3)
            .map(|c| self.affine.fixed_view::<3, 1>(0, c).into_owned())
            .collect();
        let tailles: Vec<f64> = colonnes.iter().map(Vector3::norm).collect();
        let plus_petit = tailles.iter().cloned().fold(f64::INFINITY, f64::min);
        let plus_grande_dimension = *self.dimensions().iter().max().expect("trois axes") as f64;
        let nouvelle = 1.01 * (plus_petit * plus_grande_dimension) / n as f64;
        let mut affine = Matrix4::<f64>::identity();
        for c in 0..3 {
            let direction = colonnes[c] / tailles[c];
            affine.fixed_view_mut::<3, 1>(0, c).copy_from(&(direction * nouvelle));
        }
        let milieu = Vector3::new(1.0, 1.0, 1.0) * ((n as f64 - 1.0) / 2.0);
        let decalage = self.centre() - affine.fixed_view::<3, 3>(0, 0) * milieu;
        affine.fixed_view_mut::<3, 1>(0, 3).copy_from(&decalage);
        let mut cube = self.reechantillonner(&affine, [n, n, n]);
        let (min, max) = cube
            .donnees
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let plage = max - min;
        cube.donnees
            .mapv_inplace(|v| if plage > 0.0 { (v - min) / plage } else { 0.0 });
        cube
    }

    /// Étapes 1 à 3 : volume recadré dans la grille de référence, prêt pour [`Volume::cube`] (voir la documentation du module).
    ///
    /// # Erreurs
    /// [`ErreurGeometrie::Vide`] si le volume est vide.
    pub fn preparer(&self) -> Result<Volume, ErreurGeometrie> {
        let (recadre, boite_physique) = self.recadrer_sur_seuil(SEUIL_RECADRAGE)?;
        // la boîte est exprimée dans les voxels de `self` : l'étape 3 la mesure dans l'espace de `self`
        let reference = recadre.vers_grille_de_reference();
        reference.recadrer_sur_boite_physique(self, boite_physique)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(dimensions: [usize; 3], affine: Matrix4<f64>, f: impl Fn(usize, usize, usize) -> f32) -> Volume {
        Volume {
            donnees: Array3::from_shape_fn((dimensions[0], dimensions[1], dimensions[2]), |(i, j, k)| f(i, j, k)),
            affine,
        }
    }

    fn diagonale(x: f64, y: f64, z: f64, origine: [f64; 3]) -> Matrix4<f64> {
        Matrix4::new(
            x, 0.0, 0.0, origine[0], 0.0, y, 0.0, origine[1], 0.0, 0.0, z, origine[2], 0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Interpolation : exacte sur une fonction linéaire à l'intérieur (voxels anisotropes, origine quelconque, positions fractionnaires) ; à moins d'un demi-voxel du bord, valeur du voxel de bord (interpolée sur les autres axes) ; au-delà d'un demi-voxel, 0.
    #[test]
    fn the_interpolation_is_exact_inside_and_extends_half_a_voxel() {
        let affine = diagonale(2.0, 3.0, 4.0, [10.0, 20.0, 30.0]);
        let lineaire = |i: f64, j: f64, k: f64| (1.0 + 2.0 * i + 3.0 * j + 5.0 * k) as f32;
        let v = volume([5, 6, 7], affine, |i, j, k| lineaire(i as f64, j as f64, k as f64));
        let au_voxel = |i: f64, j: f64, k: f64| v.echantillonner(&v.monde(&Vector3::new(i, j, k)));
        for (i, j, k) in [(0.0, 0.0, 0.0), (1.3, 2.6, 3.9), (3.99, 0.01, 5.5), (4.0, 5.0, 6.0)] {
            assert!(
                (au_voxel(i, j, k) - lineaire(i, j, k)).abs() < 1e-4,
                "intérieur {i} {j} {k}"
            );
        }
        // bord étendu : à -0.3 sur x, la valeur est celle de x = 0 (j et k restent interpolés)
        assert!(
            (au_voxel(-0.3, 2.5, 3.5) - lineaire(0.0, 2.5, 3.5)).abs() < 1e-4,
            "bord bas"
        );
        assert!(
            (au_voxel(4.0 + 0.4, 2.5, 3.5) - lineaire(4.0, 2.5, 3.5)).abs() < 1e-4,
            "bord haut"
        );
        // au-delà d'un demi-voxel : 0
        assert_eq!(au_voxel(-0.6, 2.5, 3.5), 0.0);
        assert_eq!(au_voxel(4.0 + 0.6, 2.5, 3.5), 0.0);
        assert_eq!(au_voxel(2.0, 2.0, -0.51), 0.0);
        assert_eq!(au_voxel(2.0, 5.0 + 0.51, 3.0), 0.0);
    }

    /// Boîte englobante : seuil strict (une valeur égale au seuil n'est pas retenue, une valeur juste au-dessus l'est), bornes incluses, `None` si rien ne dépasse ; `recadrer` rend les bonnes données et une affine dont l'origine est le voxel de début.
    #[test]
    fn the_bounding_box_is_strict_and_cropping_moves_the_origin() {
        let mut v = volume([8, 9, 10], diagonale(0.8, 0.8, 0.8, [-5.0, 6.0, 7.0]), |_, _, _| 0.0);
        v.donnees[[2, 3, 4]] = 1.0;
        v.donnees[[5, 6, 7]] = 0.0051;
        v.donnees[[7, 0, 0]] = SEUIL_RECADRAGE;
        assert_eq!(v.boite_englobante(SEUIL_RECADRAGE), Some(([2, 3, 4], [5, 6, 7])));
        assert_eq!(v.boite_englobante(5.0), None);
        let (recadre, boite) = v.recadrer_sur_seuil(SEUIL_RECADRAGE).unwrap();
        assert_eq!(recadre.dimensions(), [4, 4, 4]);
        assert_eq!(recadre.donnees[[0, 0, 0]], 1.0);
        assert_eq!(recadre.donnees[[3, 3, 3]], 0.0051);
        assert!(
            (recadre.monde(&Vector3::zeros())
                - v.monde(&Vector3::new(boite.0[0] as f64, boite.0[1] as f64, boite.0[2] as f64)))
            .norm()
                < 1e-12
        );
        assert!(volume([3, 3, 3], Matrix4::identity(), |_, _, _| 0.0)
            .recadrer_sur_seuil(SEUIL_RECADRAGE)
            .is_err());
    }

    /// Grille de référence : 300 × 369 × 300, voxels de 0,6 mm, orientation LAS (axe x de sens opposé), centre égal à celui du volume d'entrée (même pour une entrée d'orientation et de taille de voxel différentes) ; une région constante au centre garde sa valeur et l'extérieur lointain vaut 0.
    #[test]
    fn the_reference_grid_is_centred_on_the_volume() {
        let v = volume([20, 22, 24], diagonale(0.8, 0.8, 0.8, [-8.0, 3.0, 11.0]), |_, _, _| 5.0);
        let r = v.vers_grille_de_reference();
        assert_eq!(r.dimensions(), DIMENSIONS_REFERENCE);
        assert!(
            (r.centre() - v.centre()).norm() < 1e-9,
            "centre {:?} contre {:?}",
            r.centre(),
            v.centre()
        );
        for (colonne, attendu) in [(0, -0.6), (1, 0.6), (2, 0.6)] {
            assert!((r.affine[(colonne, colonne)] - attendu).abs() < 1e-12);
        }
        assert!(
            (r.donnees[[149, 184, 149]] - 5.0).abs() < 1e-5,
            "centre : {}",
            r.donnees[[149, 184, 149]]
        );
        assert_eq!(r.donnees[[0, 0, 0]], 0.0);
        assert_eq!(r.donnees[[299, 368, 299]], 0.0);
    }

    /// Second recadrage, par propriété : les voxels gardés ont tous leur centre dans la boîte physique (bords de voxels du masque), et les voxels immédiatement avant et après sur chaque axe ont leur centre en dehors.
    #[test]
    fn the_physical_crop_keeps_exactly_the_voxel_centres_inside_the_box() {
        let origine = volume([30, 28, 26], diagonale(0.8, 0.8, 0.8, [-12.0, 7.0, 3.0]), |i, j, k| {
            if (6..21).contains(&i) && (5..19).contains(&j) && (4..17).contains(&k) {
                1.0 + (i + j + k) as f32
            } else {
                0.0
            }
        });
        let (recadre, boite) = origine.recadrer_sur_seuil(SEUIL_RECADRAGE).unwrap();
        let reference = recadre.vers_grille_de_reference();
        let s3 = reference.recadrer_sur_boite_physique(&origine, boite).unwrap();
        // coordonnées de voxel de `origine` de chaque centre de voxel de s3 : la boîte est [bas - 0.5, haut + 0.5]
        let inverse = origine.affine.try_inverse().unwrap();
        let dans_origine = |i: f64, j: f64, k: f64| {
            let m = reference.affine * Vector4::new(i, j, k, 1.0);
            let v = inverse * m;
            [v.x, v.y, v.z]
        };
        let d = s3.dimensions();
        // décalage de s3 dans la grille de référence
        let decalage_monde = s3.monde(&Vector3::zeros());
        let decalage = (reference.affine.try_inverse().unwrap()
            * Vector4::new(decalage_monde.x, decalage_monde.y, decalage_monde.z, 1.0))
        .xyz();
        let debut = [decalage.x.round(), decalage.y.round(), decalage.z.round()];
        let dedans = |p: [f64; 3]| {
            (0..3).all(|a| p[a] >= boite.0[a] as f64 - 0.5 - 1e-6 && p[a] <= boite.1[a] as f64 + 0.5 + 1e-6)
        };
        for coin in 0..8 {
            let c = |axe: usize, bit: usize| {
                if (coin >> bit) & 1 == 0 {
                    debut[axe]
                } else {
                    debut[axe] + d[axe] as f64 - 1.0
                }
            };
            assert!(
                dedans(dans_origine(c(0, 0), c(1, 1), c(2, 2))),
                "coin {coin} gardé hors de la boîte"
            );
        }
        for axe in 0..3 {
            for (cote, pas) in [(debut[axe] - 1.0, 0usize), (debut[axe] + d[axe] as f64, 1usize)] {
                let _ = pas;
                let mut p = [debut[0], debut[1], debut[2]];
                p[axe] = cote;
                // les deux autres coordonnées au milieu de s3 pour ne tester que l'axe
                for autre in 0..3 {
                    if autre != axe {
                        p[autre] = debut[autre] + (d[autre] as f64 - 1.0) / 2.0;
                    }
                }
                assert!(
                    !dedans(dans_origine(p[0], p[1], p[2])),
                    "voxel voisin de s3 sur l'axe {axe} (indice {cote}) encore dans la boîte"
                );
            }
        }
    }

    /// Cube : géométrie documentée pour un volume de 168 × 149 × 146 voxels de 0,6 mm en orientation LAS (voxel `1,01 × 0,6 × 168 / n`, centre inchangé, directions des axes inchangées, dimensions n³) et intensités exactement dans [0, 1] (min 0, max 1) ; un volume constant donne un cube nul.
    #[test]
    fn the_cube_has_the_documented_geometry() {
        let v = volume(
            [168, 149, 146],
            diagonale(-0.6, 0.6, 0.6, [20.0, -30.0, 5.0]),
            |i, j, k| 3.0 + ((i * 7 + j * 3 + k) % 11) as f32,
        );
        for n in [128usize, 256] {
            let c = v.cube(n);
            assert_eq!(c.dimensions(), [n, n, n]);
            let voxel = 1.01 * (0.6 * 168.0) / n as f64;
            for colonne in 0..3 {
                let attendu = [-1.0, 1.0, 1.0][colonne] * voxel;
                assert!(
                    (c.affine[(colonne, colonne)] - attendu).abs() < 1e-12,
                    "n={n} colonne {colonne}"
                );
            }
            assert!((c.centre() - v.centre()).norm() < 1e-9, "n={n} centre");
            let (min, max) = c
                .donnees
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
            assert_eq!((min, max), (0.0, 1.0), "n={n}");
        }
        assert!(volume([10, 10, 10], Matrix4::identity(), |_, _, _| 4.0)
            .cube(16)
            .donnees
            .iter()
            .all(|&x| x == 0.0));
        // voxels anisotropes (0,5 / 0,8 / 1,2 mm) : le voxel du cube se calcule avec le PLUS PETIT voxel (0,5) et la PLUS GRANDE dimension (50 voxels)
        let anisotrope = volume([40, 50, 30], diagonale(0.5, 0.8, 1.2, [0.0, 0.0, 0.0]), |i, j, k| {
            ((i + 2 * j + 3 * k) % 7) as f32 + 1.0
        });
        let c = anisotrope.cube(64);
        let attendu = 1.01 * (0.5 * 50.0) / 64.0;
        for colonne in 0..3 {
            assert!(
                (c.affine[(colonne, colonne)] - attendu).abs() < 1e-12,
                "colonne {colonne} : {} contre {attendu}",
                c.affine[(colonne, colonne)]
            );
        }
    }

    /// **Sous-étape 5a** : étapes 1 à 4 contre les sorties de MIRTK (outils de l'image Docker de BOUNTI, voir `docs/BOUNTI.md`) pour trois reconstructions (`data/bounti/mirtk/<id>/`, hors dépôt, produites par `preparer.sh`) : même dimensions et même affine (0,001 mm près) à chaque étape ;
    /// écart maximal d'intensité ≤ 1e-3 de la plage pour le recadrage et la grille de référence, et ≤ 1e-5 pour les cubes de 128³ et 256³ après mise à l'échelle dans [0, 1].
    #[test]
    #[ignore = "demande les références MIRTK dans data/bounti/mirtk (hors dépôt)"]
    fn the_geometry_matches_mirtk() {
        let racine = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti/mirtk");
        for id in ["calme1", "calme2", "calme3"] {
            let lire = |nom: &str| {
                Volume::lire(&racine.join(id).join(format!("{nom}.nii.gz")))
                    .unwrap_or_else(|e| panic!("{id}/{nom} : {e}"))
            };
            let comparer = |nom: &str, mien: &Volume, mirtk: &Volume, seuil_relatif: f64| {
                assert_eq!(mien.dimensions(), mirtk.dimensions(), "{id} {nom} : dimensions");
                let ecart_affine = (mien.affine - mirtk.affine).abs().max();
                assert!(ecart_affine < 1e-3, "{id} {nom} : affine, écart {ecart_affine}");
                let plage = f64::from(
                    mirtk.donnees.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                        - mirtk.donnees.iter().cloned().fold(f32::INFINITY, f32::min),
                );
                let ecart = mien
                    .donnees
                    .iter()
                    .zip(mirtk.donnees.iter())
                    .fold(0.0_f64, |m, (a, b)| m.max(f64::from((a - b).abs())));
                println!("GÉOMÉTRIE {id} {nom} : dimensions {:?}, écart d'affine {ecart_affine:.1e} mm, écart d'intensité {:.2e} de la plage", mien.dimensions(), ecart / plage);
                assert!(
                    ecart <= seuil_relatif * plage,
                    "{id} {nom} : écart d'intensité {:.2e} de la plage",
                    ecart / plage
                );
            };
            let entree = lire("entree");
            let (s1, _) = entree.recadrer_sur_seuil(SEUIL_RECADRAGE).unwrap();
            comparer("s1_recadre", &s1, &lire("s1_recadre"), 1e-6);
            let s3 = entree.preparer().unwrap();
            comparer("s3_recadre", &s3, &lire("s3_recadre"), 1e-3);
            for n in [128usize, 256] {
                // le cube MIRTK est mis à l'échelle dans [0, 500] : même mise à l'échelle dans [0, 1] des deux côtés
                let mirtk = lire(&format!("cube{n}"));
                let (min, max) = mirtk
                    .donnees
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
                let mirtk = Volume {
                    donnees: mirtk.donnees.mapv(|v| (v - min) / (max - min)),
                    affine: mirtk.affine,
                };
                comparer(&format!("cube{n}"), &s3.cube(n), &mirtk, 1e-5);
            }
        }
    }
}
