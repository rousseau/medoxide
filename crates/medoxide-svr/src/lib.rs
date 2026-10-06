//! `medoxide-svr` : reconstruction coupe-vers-volume (SVR) en IRM fœtale.
//!
//! Pour l'instant : la lecture d'un stack (volume 2D multi-coupes) avec sa géométrie.
//!
//! # Conventions
//!
//! - Repère monde : **RAS+ en mm**, celui du `sform` NIfTI (comme `nibabel`). ITK et
//!   SimpleITK travaillent en LPS (signes de x et y inversés) : à convertir aux frontières.
//! - Un indice de voxel `(i, j, k)` désigne le **centre** du voxel (indices entiers).
//! - Une coupe est un `k` fixé sur le **3ᵉ axe** du fichier.
//! - L'affine voxel → monde peut avoir un **déterminant négatif** (repère d'indices « main
//!   gauche » : c'est le cas courant, y compris des 96 stacks du jeu de développement) :
//!   cela n'est pas une erreur, et l'image n'est jamais retournée.

use std::path::{Path, PathBuf};

use medoxide_core::{read_volume, volume_info, CoreError};
use nalgebra::{Matrix3, Matrix4};
use ndarray::Array3;

/// Cosinus maximal toléré entre deux colonnes de l'affine : au-delà, les axes ne sont pas
/// orthogonaux (cisaillement) et la coupe n'a plus de repère rigide.
const COSINE_MAX: f64 = 1e-3;
/// Écart maximal toléré, en mm, entre `pixdim` et la norme d'une colonne de l'affine.
const SPACING_TOLERANCE_MM: f64 = 1e-3;

/// Erreurs de lecture d'un stack.
#[derive(Debug)]
pub enum SvrError {
    /// Lecture NIfTI impossible, ou fichier qui n'est pas un volume 3D.
    Core(CoreError),
    /// L'en-tête n'a pas de `sform` (`sform_code == 0`) : pas d'affine exploitable.
    NoSform(PathBuf),
    /// Une colonne de l'affine est nulle ou non finie : géométrie dégénérée.
    DegenerateAffine(PathBuf),
    /// Deux axes ne sont pas orthogonaux ; contient le plus grand cosinus mesuré.
    ShearedAffine { path: PathBuf, cosine: f64 },
    /// `pixdim` et les normes des colonnes de l'affine diffèrent de plus de 1e-3 mm.
    InconsistentSpacing {
        path: PathBuf,
        pixdim: [f64; 3],
        columns: [f64; 3],
    },
}

impl From<CoreError> for SvrError {
    fn from(e: CoreError) -> Self {
        SvrError::Core(e)
    }
}

impl std::fmt::Display for SvrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SvrError::Core(e) => write!(f, "{e}"),
            SvrError::NoSform(p) => write!(f, "{} : pas de sform (affine absente)", p.display()),
            SvrError::DegenerateAffine(p) => {
                write!(f, "{} : affine dégénérée (colonne nulle ou non finie)", p.display())
            }
            SvrError::ShearedAffine { path, cosine } => write!(
                f,
                "{} : axes non orthogonaux (cosinus {cosine:.2e} > {COSINE_MAX:.0e})",
                path.display()
            ),
            SvrError::InconsistentSpacing { path, pixdim, columns } => write!(
                f,
                "{} : pixdim {pixdim:?} incohérent avec les normes de l'affine {columns:?}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SvrError {}

/// Un stack : un volume 3D de coupes 2D, avec sa géométrie voxel → monde.
///
/// Les champs sont privés : un `Stack` ne se construit que par [`Stack::read`], qui vérifie la
/// géométrie. Tout `Stack` existant a donc une affine finie, de colonnes orthogonales et de
/// normes cohérentes avec `pixdim`.
#[derive(Debug)]
pub struct Stack {
    path: PathBuf,
    data: Array3<f32>,
    affine: Matrix4<f64>,
    spacing: [f64; 3],
}

impl Stack {
    /// Lit un stack NIfTI (`.nii` ou `.nii.gz`) et valide sa géométrie.
    ///
    /// Les voxels sont lus en `f32`, axes `[x, y, z]`, facteur d'échelle de l'en-tête appliqué.
    /// L'affine est celle du `sform`, convertie en `f64` sans perte. L'espacement est la norme
    /// de chaque colonne de l'affine (la géométrie réelle) ; il doit coïncider avec `pixdim`
    /// à 1e-3 mm près. Un déterminant négatif est accepté.
    ///
    /// # Erreurs
    /// [`SvrError::Core`] (fichier illisible, non 3D) ; [`SvrError::NoSform`] ;
    /// [`SvrError::DegenerateAffine`] ; [`SvrError::ShearedAffine`] ;
    /// [`SvrError::InconsistentSpacing`].
    pub fn read(path: &Path) -> Result<Stack, SvrError> {
        let info = volume_info(path)?;
        let lignes = info.affine.ok_or_else(|| SvrError::NoSform(path.to_path_buf()))?;
        // `f64::from` : conversion exacte du `f32` de l'en-tête.
        let affine = Matrix4::from_fn(|r, c| f64::from(lignes[r][c]));

        // Partie linéaire 3x3 : colonne `a` = déplacement monde (mm) d'un pas d'indice sur l'axe `a`.
        let lineaire: Matrix3<f64> = affine.fixed_view::<3, 3>(0, 0).into_owned();
        let colonnes = [lineaire.column(0), lineaire.column(1), lineaire.column(2)];
        let normes = [colonnes[0].norm(), colonnes[1].norm(), colonnes[2].norm()];
        // `!(n > 0.0)` est vrai aussi pour NaN, ce que `n <= 0.0` ne détecterait pas.
        if normes.iter().any(|n| !(*n > 0.0) || !n.is_finite()) {
            return Err(SvrError::DegenerateAffine(path.to_path_buf()));
        }

        let mut cosinus_max = 0.0_f64;
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let cosinus = colonnes[a].dot(&colonnes[b]).abs() / (normes[a] * normes[b]);
            cosinus_max = cosinus_max.max(cosinus);
        }
        if cosinus_max > COSINE_MAX {
            return Err(SvrError::ShearedAffine { path: path.to_path_buf(), cosine: cosinus_max });
        }

        let pixdim = [
            f64::from(info.spacing[0]),
            f64::from(info.spacing[1]),
            f64::from(info.spacing[2]),
        ];
        if (0..3).any(|a| (pixdim[a] - normes[a]).abs() > SPACING_TOLERANCE_MM) {
            return Err(SvrError::InconsistentSpacing {
                path: path.to_path_buf(),
                pixdim,
                columns: normes,
            });
        }

        let data = read_volume(path)?;
        Ok(Stack { path: path.to_path_buf(), data, affine, spacing: normes })
    }

    /// Chemin du fichier lu.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Voxels, axes `[x, y, z]` ; la coupe `k` est le plan `z = k`.
    pub fn data(&self) -> &Array3<f32> {
        &self.data
    }

    /// Affine voxel → monde (RAS+, mm), 4×4, coordonnées homogènes.
    pub fn affine(&self) -> &Matrix4<f64> {
        &self.affine
    }

    /// Taille d'un pas d'indice sur chaque axe, en mm (normes des colonnes de l'affine).
    pub fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    /// Nombre de voxels sur chaque axe `(x, y, z)` ; `z` est le nombre de coupes.
    pub fn dim(&self) -> (usize, usize, usize) {
        self.data.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4};
    use nifti::{writer::WriterOptions, NiftiHeader};

    fn racine() -> String {
        format!("{}/../..", env!("CARGO_MANIFEST_DIR"))
    }

    /// Dossier temporaire propre à un test (nom + processus), supprimé en fin de test.
    struct Temp(PathBuf);
    impl Temp {
        fn new(nom: &str) -> Temp {
            let d = std::env::temp_dir().join(format!("medoxide_svr_{nom}_{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Temp(d)
        }
        fn fichier(&self, nom: &str) -> PathBuf {
            self.0.join(nom)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// En-tête synthétique : `sform` donné par ses 3 premières lignes, `pixdim` donné.
    fn en_tete(lignes: [[f32; 4]; 3], pixdim: [f32; 3], sform_code: i16) -> NiftiHeader {
        let mut h = NiftiHeader::default();
        h.sform_code = sform_code;
        h.srow_x = lignes[0];
        h.srow_y = lignes[1];
        h.srow_z = lignes[2];
        h.pixdim = [1.0, pixdim[0], pixdim[1], pixdim[2], 1.0, 1.0, 1.0, 1.0];
        h
    }

    fn ecrire_3d(chemin: &Path, h: &NiftiHeader) {
        let data = Array3::<f32>::from_elem((4, 3, 2), 1.0);
        WriterOptions::new(chemin).reference_header(h).write_nifti(&data).unwrap();
    }

    const DIAG: [[f32; 4]; 3] = [[1.0, 0.0, 0.0, 10.0], [0.0, 2.0, 0.0, 20.0], [0.0, 0.0, 3.0, 30.0]];

    #[test]
    fn reads_a_valid_synthetic_stack() {
        let t = Temp::new("valide");
        let f = t.fichier("s.nii.gz");
        ecrire_3d(&f, &en_tete(DIAG, [1.0, 2.0, 3.0], 1));
        let s = Stack::read(&f).unwrap();
        assert_eq!(s.dim(), (4, 3, 2));
        assert_eq!(s.spacing(), [1.0, 2.0, 3.0]);
        assert_eq!(s.affine()[(0, 3)], 10.0);
        assert_eq!(s.affine()[(3, 3)], 1.0);
    }

    /// Un déterminant négatif n'est pas une erreur (cas de tous les stacks du jeu de développement).
    #[test]
    fn accepts_a_left_handed_affine() {
        let t = Temp::new("gauche");
        let f = t.fichier("s.nii.gz");
        let lignes = [[-1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]];
        ecrire_3d(&f, &en_tete(lignes, [1.0, 2.0, 3.0], 1));
        let s = Stack::read(&f).unwrap();
        assert!(s.affine().fixed_view::<3, 3>(0, 0).determinant() < 0.0);
        assert_eq!(s.spacing(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_a_missing_sform() {
        let t = Temp::new("sans_sform");
        let f = t.fichier("s.nii.gz");
        ecrire_3d(&f, &en_tete(DIAG, [1.0, 2.0, 3.0], 0));
        assert!(matches!(Stack::read(&f), Err(SvrError::NoSform(_))));
    }

    #[test]
    fn rejects_a_sheared_affine() {
        let t = Temp::new("cisaille");
        let f = t.fichier("s.nii.gz");
        // 2e colonne = (0,1,0) + 0,2 × 1re colonne : cosinus entre colonnes ≈ 0,2.
        let lignes = [[1.0, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let norme = (1.0_f32 + 0.04).sqrt();
        ecrire_3d(&f, &en_tete(lignes, [1.0, norme, 1.0], 1));
        match Stack::read(&f) {
            Err(SvrError::ShearedAffine { cosine, .. }) => assert!(cosine > 0.1, "{cosine}"),
            autre => panic!("attendu ShearedAffine, obtenu {autre:?}"),
        }
    }

    #[test]
    fn rejects_inconsistent_pixdim() {
        let t = Temp::new("pixdim");
        let f = t.fichier("s.nii.gz");
        ecrire_3d(&f, &en_tete(DIAG, [1.0, 2.0, 3.5], 1)); // l'affine dit 3, pixdim dit 3,5
        assert!(matches!(Stack::read(&f), Err(SvrError::InconsistentSpacing { .. })));
    }

    #[test]
    fn rejects_a_degenerate_affine() {
        let t = Temp::new("degenere");
        let f = t.fichier("s.nii.gz");
        let lignes = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]; // colonne 1 nulle
        ecrire_3d(&f, &en_tete(lignes, [1.0, 0.0, 1.0], 1));
        assert!(matches!(Stack::read(&f), Err(SvrError::DegenerateAffine(_))));
    }

    #[test]
    fn rejects_a_4d_file_and_a_missing_file() {
        let t = Temp::new("erreurs");
        let f = t.fichier("s4d.nii.gz");
        let data = Array4::<f32>::from_elem((4, 3, 2, 2), 1.0);
        WriterOptions::new(&f)
            .reference_header(&en_tete(DIAG, [1.0, 2.0, 3.0], 1))
            .write_nifti(&data)
            .unwrap();
        assert!(matches!(
            Stack::read(&f),
            Err(SvrError::Core(CoreError::NotVolume3D(_)))
        ));
        assert!(matches!(
            Stack::read(&t.fichier("absent.nii.gz")),
            Err(SvrError::Core(CoreError::Nifti(_)))
        ));
    }

    /// Affines de référence de nibabel (`img.affine`) pour 3 volumes de Fetal-BET, arrondis à 4
    /// décimales : mêmes valeurs que dans `medoxide-core`, lues cette fois par `Stack`.
    #[test]
    fn reads_fetal_bet_volumes_with_nibabel_affines() {
        let attendu: [(&str, [[f64; 4]; 3]); 3] = [
            ("fetus_03", [[-1.1719, 0.0, 0.0, 155.1199], [0.0, -1.0771, 0.7879, 58.3842], [0.0, 0.4617, 1.8383, -254.6115]]),
            ("fetus_06", [[-0.8439, -0.147, -0.8127, 141.5291], [0.4083, -0.3038, -1.6797, -19.843], [0.0, -0.8747, 0.7199, -69.1684]]),
            ("fetus_12", [[-0.5306, -0.4462, 1.8271, 96.2711], [-0.866, 0.2733, -1.1193, 46.4229], [0.0, -0.8705, -1.288, -12.331]]),
        ];
        for (nom, voulu) in attendu {
            let s = Stack::read(Path::new(&format!("{}/data/sourcedata/{nom}.nii.gz", racine()))).unwrap();
            for r in 0..3 {
                for c in 0..4 {
                    assert!((s.affine()[(r, c)] - voulu[r][c]).abs() < 1e-3, "{nom} ({r},{c})");
                }
            }
            assert_eq!(s.affine().row(3).iter().copied().collect::<Vec<_>>(), vec![0.0, 0.0, 0.0, 1.0]);
        }
    }

    /// Parcourt récursivement un dossier et collecte les fichiers dont le nom se termine par `fin`.
    fn trouver(dossier: &Path, fin: &str, sortie: &mut Vec<PathBuf>) {
        for entree in std::fs::read_dir(dossier).unwrap().flatten() {
            let chemin = entree.path();
            if chemin.is_dir() {
                trouver(&chemin, fin, sortie);
            } else if chemin.to_string_lossy().ends_with(fin) {
                sortie.push(chemin);
            }
        }
    }

    /// Jeu de développement : les 96 stacks se chargent, tous à déterminant négatif, géométrie
    /// valide (orthogonale, pixdim cohérent). Données locales (`data/`, hors de Git).
    #[test]
    fn reads_all_development_stacks_even_left_handed() {
        let mut fichiers = Vec::new();
        trouver(&Path::new(&racine()).join("data/svr/jeu_reel_tru_haste"), "_T2w.nii.gz", &mut fichiers);
        fichiers.retain(|f| !f.to_string_lossy().contains("/derivatives/") && !f.to_string_lossy().contains("/sourcedata/"));
        assert_eq!(fichiers.len(), 96, "jeu de développement incomplet");
        let mut gauches = 0;
        for f in &fichiers {
            let s = Stack::read(f).unwrap_or_else(|e| panic!("{e}"));
            gauches += usize::from(s.affine().fixed_view::<3, 3>(0, 0).determinant() < 0.0);
            assert_eq!(s.dim().2, s.data().dim().2);
        }
        assert_eq!(gauches, 96, "tous les stacks du jeu sont censés être à déterminant négatif");
    }
}
