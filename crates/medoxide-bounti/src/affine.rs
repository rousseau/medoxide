//! Transformations affines de MIRTK (fichiers `.dof`) : lecture des douze paramètres et matrice 4 × 4 correspondante.
//!
//! Format (relevé sur les fichiers de la chaîne de Multi-BOUNTI) : en-tête de 12 octets en grand-boutiste (nombre magique `0x000c6f9f`, type de transformation, nombre de paramètres), puis les paramètres en `f64` grand-boutistes.
//! Type 2 : rigide, 6 paramètres (translation `tx ty tz` en mm, rotation `rx ry rz` en degrés). Type 3 : affine, 12 paramètres : les six précédents, les échelles `sx sy sz` en POURCENT (100 = identité), puis les cisaillements `sxy syz sxz` en degrés.
//!
//! Matrice (reconstituée par ajustement sur trois transformations obtenues par `mirtk register`, puis vérifiée : écart de 5e-7 avec la matrice que MIRTK affiche avec six chiffres) : `A = T · R · K · S`, avec `T` la translation,
//! `R = Rx(−rx) · Ry(−ry) · Rz(−rz)` (rotations d'angle en degrés), `K` la matrice identité avec `tan(sxy)` en (0, 1), `tan(sxz)` en (0, 2) et `tan(syz)` en (1, 2), et `S = diag(sx, sy, sz, 100) / 100`.
//! `A` envoie un point du repère de l'image **cible** du recalage (le gabarit) vers le repère de l'image **source** (le sujet), en millimètres, repère RAS des fichiers NIfTI : `transform-image source sortie -target cible -dofin dof` donne `sortie(x) = source(A · x)`.
use nalgebra::Matrix4;

/// Nombre magique des fichiers `.dof` de MIRTK.
const MAGIQUE: u32 = 0x000c_6f9f;

/// Erreurs de lecture d'un fichier `.dof`.
#[derive(Debug, PartialEq)]
pub enum ErreurDof {
    /// Le fichier est trop court ou n'a pas le nombre magique.
    Invalide,
    /// Type de transformation non pris en charge (seuls le rigide, 2, et l'affine, 3, le sont).
    TypeNonPris(u32),
}

impl std::fmt::Display for ErreurDof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErreurDof::Invalide => write!(f, "fichier .dof invalide (trop court ou sans nombre magique)"),
            ErreurDof::TypeNonPris(t) => write!(
                f,
                "type de transformation {t} non pris en charge (rigide : 2, affine : 3)"
            ),
        }
    }
}

impl std::error::Error for ErreurDof {}

/// Les douze paramètres d'une transformation affine de MIRTK (voir le module).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parametres {
    /// Translation en mm.
    pub translation: [f64; 3],
    /// Rotation en degrés.
    pub rotation: [f64; 3],
    /// Échelles en pourcent (100 = identité).
    pub echelle: [f64; 3],
    /// Cisaillements `sxy`, `syz`, `sxz` en degrés.
    pub cisaillement: [f64; 3],
}

impl Parametres {
    /// L'identité (translation et rotation nulles, échelles à 100 %, pas de cisaillement).
    pub fn identite() -> Parametres {
        Parametres {
            translation: [0.0; 3],
            rotation: [0.0; 3],
            echelle: [100.0; 3],
            cisaillement: [0.0; 3],
        }
    }

    /// Lit un fichier `.dof` rigide (échelles à 100 %, pas de cisaillement) ou affine.
    ///
    /// # Erreurs
    /// [`ErreurDof::Invalide`] si le fichier est trop court ou n'a pas le nombre magique ; [`ErreurDof::TypeNonPris`] pour un autre type de transformation.
    pub fn depuis_octets(octets: &[u8]) -> Result<Parametres, ErreurDof> {
        let entier = |i: usize| -> Option<u32> { Some(u32::from_be_bytes(octets.get(i..i + 4)?.try_into().ok()?)) };
        let reel = |i: usize| -> Option<f64> { Some(f64::from_be_bytes(octets.get(i..i + 8)?.try_into().ok()?)) };
        if entier(0) != Some(MAGIQUE) {
            return Err(ErreurDof::Invalide);
        }
        let type_ = entier(4).ok_or(ErreurDof::Invalide)?;
        let nombre = match type_ {
            2 => 6,
            3 => 12,
            autre => return Err(ErreurDof::TypeNonPris(autre)),
        };
        if entier(8) != Some(nombre as u32) {
            return Err(ErreurDof::Invalide);
        }
        let mut valeurs = [0.0f64; 12];
        valeurs[6..9].copy_from_slice(&[100.0; 3]);
        for (i, v) in valeurs.iter_mut().take(nombre).enumerate() {
            *v = reel(12 + 8 * i).ok_or(ErreurDof::Invalide)?;
        }
        Ok(Parametres {
            translation: [valeurs[0], valeurs[1], valeurs[2]],
            rotation: [valeurs[3], valeurs[4], valeurs[5]],
            echelle: [valeurs[6], valeurs[7], valeurs[8]],
            cisaillement: [valeurs[9], valeurs[10], valeurs[11]],
        })
    }

    /// Matrice 4 × 4 `A = T · R · K · S` (voir le module) : envoie un point du repère de l'image cible vers celui de l'image source.
    pub fn matrice(&self) -> Matrix4<f64> {
        let degres = std::f64::consts::PI / 180.0;
        let rotation_x = |a: f64| {
            let (s, c) = (a * degres).sin_cos();
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0)
        };
        let rotation_y = |a: f64| {
            let (s, c) = (a * degres).sin_cos();
            Matrix4::new(c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0)
        };
        let rotation_z = |a: f64| {
            let (s, c) = (a * degres).sin_cos();
            Matrix4::new(c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        };
        let [tx, ty, tz] = self.translation;
        let translation = Matrix4::new(
            1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz, 0.0, 0.0, 0.0, 1.0,
        );
        let rotation = rotation_x(-self.rotation[0]) * rotation_y(-self.rotation[1]) * rotation_z(-self.rotation[2]);
        let [sxy, syz, sxz] = self.cisaillement;
        let mut cisaillement = Matrix4::identity();
        cisaillement[(0, 1)] = (sxy * degres).tan();
        cisaillement[(0, 2)] = (sxz * degres).tan();
        cisaillement[(1, 2)] = (syz * degres).tan();
        let echelle = Matrix4::from_diagonal(&nalgebra::Vector4::new(
            self.echelle[0] / 100.0,
            self.echelle[1] / 100.0,
            self.echelle[2] / 100.0,
            1.0,
        ));
        translation * rotation * cisaillement * echelle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Fichier `.dof` fabriqué à la main : en-tête (magique, type, nombre) puis les paramètres en `f64` grand-boutistes.
    fn dof(type_: u32, parametres: &[f64]) -> Vec<u8> {
        let mut octets = Vec::new();
        octets.extend_from_slice(&MAGIQUE.to_be_bytes());
        octets.extend_from_slice(&type_.to_be_bytes());
        octets.extend_from_slice(&(parametres.len() as u32).to_be_bytes());
        for p in parametres {
            octets.extend_from_slice(&p.to_be_bytes());
        }
        octets
    }

    /// Lecture : un fichier rigide donne des échelles de 100 % et pas de cisaillement ; un fichier affine rend ses douze paramètres dans l'ordre ; en-tête invalide, fichier tronqué et type inconnu sont refusés.
    #[test]
    fn the_dof_files_are_read_in_the_documented_order() {
        let rigide = Parametres::depuis_octets(&dof(2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        assert_eq!(rigide.translation, [1.0, 2.0, 3.0]);
        assert_eq!(rigide.rotation, [4.0, 5.0, 6.0]);
        assert_eq!(rigide.echelle, [100.0; 3]);
        assert_eq!(rigide.cisaillement, [0.0; 3]);
        let affine = Parametres::depuis_octets(&dof(
            3,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 70.0, 80.0, 90.0, 10.0, 11.0, 12.0],
        ))
        .unwrap();
        assert_eq!(affine.echelle, [70.0, 80.0, 90.0]);
        assert_eq!(affine.cisaillement, [10.0, 11.0, 12.0]);
        assert_eq!(Parametres::depuis_octets(&[0u8; 120]), Err(ErreurDof::Invalide));
        assert_eq!(
            Parametres::depuis_octets(&dof(3, &[0.0; 12])[..60]),
            Err(ErreurDof::Invalide)
        );
        assert_eq!(
            Parametres::depuis_octets(&dof(4, &[0.0; 12])),
            Err(ErreurDof::TypeNonPris(4))
        );
        assert_eq!(
            Parametres::depuis_octets(&dof(3, &[0.0; 6])),
            Err(ErreurDof::Invalide),
            "nombre de paramètres incohérent avec le type"
        );
    }

    /// Cas analytiques : l'identité, une échelle de 50 % (deux fois plus petit), une translation, une rotation de 90° autour de z d'angle `rz` (le signe moins de la formule : `Rz(−90°)` envoie x vers −y), un cisaillement de 45° (`tan = 1`).
    #[test]
    fn the_matrix_follows_the_documented_formula_on_analytic_cases() {
        assert_eq!(Parametres::identite().matrice(), Matrix4::identity());
        let mut p = Parametres::identite();
        p.echelle = [50.0, 25.0, 100.0];
        p.translation = [1.0, -2.0, 3.0];
        let a = p.matrice();
        assert_eq!(a[(0, 0)], 0.5);
        assert_eq!(a[(1, 1)], 0.25);
        assert_eq!(a[(2, 2)], 1.0);
        assert_eq!((a[(0, 3)], a[(1, 3)], a[(2, 3)]), (1.0, -2.0, 3.0));
        let mut r = Parametres::identite();
        r.rotation = [0.0, 0.0, 90.0];
        let x = r.matrice() * nalgebra::Vector4::new(1.0, 0.0, 0.0, 1.0);
        assert!((x.x - 0.0).abs() < 1e-12 && (x.y + 1.0).abs() < 1e-12, "x → {x:?}");
        let mut k = Parametres::identite();
        k.cisaillement = [45.0, 0.0, 0.0];
        assert!((k.matrice()[(0, 1)] - 1.0).abs() < 1e-12);
        // l'ordre T · R · K · S : la translation n'est ni tournée ni mise à l'échelle
        let mut tout = Parametres::identite();
        tout.translation = [10.0, 0.0, 0.0];
        tout.rotation = [0.0, 0.0, 90.0];
        tout.echelle = [200.0, 200.0, 200.0];
        let origine = tout.matrice() * nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0);
        assert!((origine.x - 10.0).abs() < 1e-12 && origine.y.abs() < 1e-12);
        let unite = tout.matrice() * nalgebra::Vector4::new(1.0, 0.0, 0.0, 1.0);
        // S puis R : x → 2 x (échelle) puis Rz(−90°) : (2, 0) → (0, −2), puis translation : (10, −2)
        assert!(
            (unite.x - 10.0).abs() < 1e-12 && (unite.y + 2.0).abs() < 1e-12,
            "{unite:?}"
        );
    }

    /// **Vérification contre MIRTK** : les paramètres des `.dof` de trois recalages (`aff-d.dof`) redonnent la matrice affichée par `convert-dof -output-format aladin` à 1e-5 près (six chiffres affichés), et celle des départs fournis (`i-fetal.dof`, rigide `i.dof`…) est cohérente.
    #[test]
    #[ignore = "demande les références de data/multibounti/ref (hors dépôt) : voir scripts/multibounti_reference.py"]
    fn the_matrix_matches_the_one_printed_by_mirtk() {
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/multibounti");
        for id in ["calme1", "calme2", "calme3"] {
            let dossier = racine.join("ref").join(id);
            let p = Parametres::depuis_octets(&std::fs::read(dossier.join("aff-d.dof")).unwrap()).unwrap();
            let texte = std::fs::read_to_string(dossier.join("aff-d.mat")).unwrap();
            let valeurs: Vec<f64> = texte.split_whitespace().map(|v| v.parse().unwrap()).collect();
            let attendue = Matrix4::from_row_slice(&valeurs);
            let ecart = (p.matrice() - attendue).abs().max();
            println!("DOF {id} : écart à la matrice de MIRTK {ecart:.1e}");
            assert!(ecart < 1e-5, "{id} : {ecart:e}");
        }
        let modele = racine.join("image/home/perinatal-brain-mri-analysis/templates/ref-bounti");
        let rigide = Parametres::depuis_octets(&std::fs::read(modele.join("i.dof")).unwrap()).unwrap();
        assert_eq!(rigide, Parametres::identite(), "i.dof est l'identité");
        let tres_precoce =
            Parametres::depuis_octets(&std::fs::read(modele.join("i-very-early-ga.dof")).unwrap()).unwrap();
        assert_eq!(tres_precoce.echelle, [40.0; 3]);
    }
}
