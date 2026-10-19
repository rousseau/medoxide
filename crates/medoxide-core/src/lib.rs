//! `medoxide-core` : briques communes aux modules de medoxide.
//!
//! Pour l'instant : la lecture de volumes NIfTI (en-tête et voxels). Ce crate ne
//! dépend d'aucun module (ni de Burn, ni d'un modèle) : un module peut lire un
//! volume sans en tirer les dépendances d'un autre.

use std::path::Path;

use ndarray::{Array3, Ix3};

/// Erreurs de lecture d'un volume NIfTI.
#[derive(Debug)]
pub enum CoreError {
    /// Échec de lecture ou d'écriture d'un fichier NIfTI (fichier absent, format
    /// invalide, écriture impossible...). La variante *contient* l'erreur d'origine
    /// de la crate `nifti`.
    Nifti(nifti::NiftiError),
    /// Le fichier n'est pas un volume à 3 axes ; contient ses dimensions.
    NotVolume3D(Vec<usize>),
}

/// Permet à l'opérateur `?` de convertir automatiquement une erreur de la
/// crate `nifti` en `CoreError::Nifti`, dans toute fonction qui renvoie
/// un `Result<_, CoreError>`.
impl From<nifti::NiftiError> for CoreError {
    fn from(e: nifti::NiftiError) -> Self {
        CoreError::Nifti(e)
    }
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreError::Nifti(e) => write!(f, "NIfTI : {e}"),
            CoreError::NotVolume3D(dim) => write!(f, "volume 3D attendu, dimensions : {dim:?}"),
        }
    }
}

impl std::error::Error for CoreError {}

/// Informations géométriques d'un volume NIfTI, lues dans son en-tête.
#[derive(Debug, PartialEq)]
pub struct VolumeInfo {
    /// Nombre de voxels par axe (ex. `[256, 256, 44]`).
    pub dim: Vec<u16>,
    /// Taille d'un voxel en mm sur les 3 premiers axes (x, y, z).
    pub spacing: [f32; 3],
    /// Affine voxel → mm (matrice 4×4, ligne par ligne) lue dans le `sform`.
    /// `None` si l'en-tête n'a pas de `sform` (`sform_code == 0`).
    pub affine: Option<[[f32; 4]; 4]>,
}

/// Lit les dimensions, l'espacement et l'affine d'un volume NIfTI
/// (`.nii` ou `.nii.gz`).
///
/// Seul l'en-tête (348 octets) est lu : les voxels ne sont pas chargés.
///
/// # Erreurs
/// `CoreError::Nifti` si le fichier est absent, illisible ou invalide.
pub fn volume_info(path: &Path) -> Result<VolumeInfo, CoreError> {
    let header = nifti::NiftiHeader::from_file(path)?;
    let dim = header.dim()?.to_vec();
    // pixdim[0] sert à autre chose (sens de rotation) : les tailles de voxel
    // sont dans pixdim[1..=3].
    let spacing = [header.pixdim[1], header.pixdim[2], header.pixdim[3]];
    // Les 3 premières lignes de l'affine sont stockées dans srow_x/y/z ;
    // la 4e est toujours [0, 0, 0, 1] par convention.
    let affine = if header.sform_code != 0 {
        Some([
            header.srow_x,
            header.srow_y,
            header.srow_z,
            [0.0, 0.0, 0.0, 1.0],
        ])
    } else {
        None
    };
    Ok(VolumeInfo { dim, spacing, affine })
}

/// D'où vient l'affine d'un volume NIfTI (voir [`affine_nifti`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceAffine {
    /// Le `sform` (`sform_code > 0`) : matrice stockée telle quelle.
    Sform,
    /// Le `qform` (`qform_code > 0`) : quaternion, tailles de voxel (`pixdim`) et sens (`qfac`).
    Qform,
    /// Ni l'un ni l'autre : diagonale des `pixdim`, volume centré sur l'origine (comme nibabel).
    Pixdim,
}

/// Affine voxel → mm d'un volume NIfTI, en double précision, et sa source.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineNifti {
    /// Matrice 4×4, ligne par ligne : `[x, y, z, 1]ᵀ = matrice · [i, j, k, 1]ᵀ`.
    pub matrice: [[f64; 4]; 4],
    /// Champ de l'en-tête dont elle est tirée.
    pub source: SourceAffine,
}

/// Affine d'un en-tête NIfTI par la règle de nibabel (`get_best_affine`) : le `sform` si `sform_code > 0`, sinon le `qform` si `qform_code > 0`, sinon la diagonale des `pixdim`.
///
/// Le `qform` est converti selon la spécification NIfTI-1 : `a = √(1 − b² − c² − d²)` (quaternion normalisé si cette quantité est négative), matrice de rotation `R` du quaternion `(a, b, c, d)`, colonnes multipliées par `pixdim[1..=3]`,
/// la troisième colonne l'étant en plus par `qfac` (`pixdim[0]`, pris égal à −1 s'il est négatif, 1 sinon) ; translation `(quatern_x, quatern_y, quatern_z)` (le `qoffset` de la spécification).
///
/// # Erreurs
/// `CoreError::Nifti` si les dimensions de l'en-tête sont invalides (cas sans `sform` ni `qform`, qui centre le volume).
pub fn affine_depuis_en_tete(header: &nifti::NiftiHeader) -> Result<AffineNifti, CoreError> {
    if header.sform_code > 0 {
        let ligne = |l: [f32; 4]| [f64::from(l[0]), f64::from(l[1]), f64::from(l[2]), f64::from(l[3])];
        return Ok(AffineNifti { matrice: [ligne(header.srow_x), ligne(header.srow_y), ligne(header.srow_z), [0.0, 0.0, 0.0, 1.0]], source: SourceAffine::Sform });
    }
    let taille = [f64::from(header.pixdim[1]), f64::from(header.pixdim[2]), f64::from(header.pixdim[3])];
    if header.qform_code > 0 {
        let (b, c, d) = (f64::from(header.quatern_b), f64::from(header.quatern_c), f64::from(header.quatern_d));
        let reste = 1.0 - (b * b + c * c + d * d);
        let (a, b, c, d) = if reste < 1e-7 {
            let norme = (b * b + c * c + d * d).sqrt();
            (0.0, b / norme, c / norme, d / norme)
        } else {
            (reste.sqrt(), b, c, d)
        };
        let rotation = [
            [a * a + b * b - c * c - d * d, 2.0 * (b * c - a * d), 2.0 * (b * d + a * c)],
            [2.0 * (b * c + a * d), a * a + c * c - b * b - d * d, 2.0 * (c * d - a * b)],
            [2.0 * (b * d - a * c), 2.0 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ];
        let qfac = if header.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        let decalage = [f64::from(header.quatern_x), f64::from(header.quatern_y), f64::from(header.quatern_z)];
        let mut matrice = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        for ligne in 0..3 {
            for colonne in 0..3 {
                let facteur = if colonne == 2 { qfac } else { 1.0 };
                matrice[ligne][colonne] = rotation[ligne][colonne] * taille[colonne] * facteur;
            }
            matrice[ligne][3] = decalage[ligne];
        }
        return Ok(AffineNifti { matrice, source: SourceAffine::Qform });
    }
    let dim = header.dim()?;
    let mut matrice = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    for axe in 0..3 {
        matrice[axe][axe] = taille[axe];
        matrice[axe][3] = -(f64::from(dim[axe]) - 1.0) / 2.0 * taille[axe];
    }
    Ok(AffineNifti { matrice, source: SourceAffine::Pixdim })
}

/// Affine d'un fichier NIfTI (`.nii` ou `.nii.gz`) par la règle de [`affine_depuis_en_tete`] ; seul l'en-tête est lu.
///
/// # Erreurs
/// `CoreError::Nifti` si le fichier est absent, illisible ou invalide.
pub fn affine_nifti(path: &Path) -> Result<AffineNifti, CoreError> {
    affine_depuis_en_tete(&nifti::NiftiHeader::from_file(path)?)
}

/// Fixe dans `header` la géométrie `matrice` (affine voxel → mm, 4×4, ligne par ligne) de façon **cohérente** : le `sform` (code 1) et le `qform` (code 1) décrivent la même transformation, et `pixdim[1..=3]` (normes des colonnes) et `pixdim[0]` (`qfac`) sont mis à jour.
///
/// Pourquoi les deux : les outils ne lisent pas le même champ (MIRTK lit le `qform`, nibabel et FSL préfèrent le `sform`) ; un `qform` laissé à sa valeur par défaut (sans origine) fait placer le volume avec son coin à l'origine.
/// Le `qform` suit la spécification NIfTI-1 (`nifti_mat44_to_quatern`) : colonnes normalisées, `qfac = −1` et troisième colonne inversée si le déterminant est négatif, puis quaternion de la rotation ; `a` n'est pas stocké (il vaut `√(1 − b² − c² − d²) ≥ 0`).
pub fn definir_affine(header: &mut nifti::NiftiHeader, matrice: [[f64; 4]; 4]) {
    header.sform_code = 1;
    header.srow_x = std::array::from_fn(|c| matrice[0][c] as f32);
    header.srow_y = std::array::from_fn(|c| matrice[1][c] as f32);
    header.srow_z = std::array::from_fn(|c| matrice[2][c] as f32);
    // colonnes de la partie linéaire : tailles de voxel et directions
    let mut r = [[0.0_f64; 3]; 3];
    let mut taille = [0.0_f64; 3];
    for colonne in 0..3 {
        taille[colonne] = (0..3).map(|l| matrice[l][colonne] * matrice[l][colonne]).sum::<f64>().sqrt();
        for ligne in 0..3 {
            r[ligne][colonne] = if taille[colonne] > 0.0 { matrice[ligne][colonne] / taille[colonne] } else { 0.0 };
        }
    }
    let determinant = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    let qfac = if determinant < 0.0 { -1.0 } else { 1.0 };
    if qfac < 0.0 {
        for ligne in r.iter_mut() {
            ligne[2] = -ligne[2];
        }
    }
    let a0 = r[0][0] + r[1][1] + r[2][2] + 1.0;
    let (a, b, c, d);
    if a0 > 0.5 {
        let a_ = 0.5 * a0.sqrt();
        (a, b, c, d) = (a_, 0.25 * (r[2][1] - r[1][2]) / a_, 0.25 * (r[0][2] - r[2][0]) / a_, 0.25 * (r[1][0] - r[0][1]) / a_);
    } else {
        let xd = 1.0 + r[0][0] - (r[1][1] + r[2][2]);
        let yd = 1.0 + r[1][1] - (r[0][0] + r[2][2]);
        let zd = 1.0 + r[2][2] - (r[0][0] + r[1][1]);
        if xd > 1.0 {
            let b_ = 0.5 * xd.sqrt();
            (a, b, c, d) = (0.25 * (r[2][1] - r[1][2]) / b_, b_, 0.25 * (r[0][1] + r[1][0]) / b_, 0.25 * (r[0][2] + r[2][0]) / b_);
        } else if yd > 1.0 {
            let c_ = 0.5 * yd.sqrt();
            (a, b, c, d) = (0.25 * (r[0][2] - r[2][0]) / c_, 0.25 * (r[0][1] + r[1][0]) / c_, c_, 0.25 * (r[1][2] + r[2][1]) / c_);
        } else {
            let d_ = 0.5 * zd.sqrt();
            (a, b, c, d) = (0.25 * (r[1][0] - r[0][1]) / d_, 0.25 * (r[0][2] + r[2][0]) / d_, 0.25 * (r[1][2] + r[2][1]) / d_, d_);
        }
    }
    let signe = if a < 0.0 { -1.0 } else { 1.0 };
    header.qform_code = 1;
    (header.quatern_b, header.quatern_c, header.quatern_d) = ((signe * b) as f32, (signe * c) as f32, (signe * d) as f32);
    (header.quatern_x, header.quatern_y, header.quatern_z) = (matrice[0][3] as f32, matrice[1][3] as f32, matrice[2][3] as f32);
    header.pixdim[0] = qfac as f32;
    for (colonne, t) in taille.iter().enumerate() {
        header.pixdim[colonne + 1] = *t as f32;
    }
}

/// Lit les voxels d'un volume NIfTI (`.nii` ou `.nii.gz`) en `f32`, axes `[x, y, z]`.
///
/// Le facteur d'échelle de l'en-tête (`scl_slope`, `scl_inter`) est appliqué.
///
/// # Erreurs
/// `CoreError::Nifti` si le fichier est illisible ou invalide ;
/// `CoreError::NotVolume3D` s'il n'a pas exactement 3 axes.
pub fn read_volume(path: &Path) -> Result<Array3<f32>, CoreError> {
    use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

    let objet = ReaderOptions::new().read_file(path)?;
    // Tableau à nombre d'axes dynamique, dans l'ordre mémoire du fichier (Fortran).
    let dynamique = objet.into_volume().into_ndarray::<f32>()?;
    let dim = dynamique.shape().to_vec();
    // `map_err` transforme l'erreur de ndarray en la nôtre, avec les dimensions.
    dynamique
        .into_dimensionality::<Ix3>()
        .map_err(|_| CoreError::NotVolume3D(dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Valeurs de référence calculées avec nibabel (arrondies à 2 décimales).
    /// Les volumes (`data/sourcedata/`) sont les 8 volumes d'exemple du dépôt
    /// Fetal-BET (`dataset/f0832s1`) ; voir `book/src/validation.md`.
    #[test]
    fn volume_info_matches_nibabel() {
        let attendu: [(&str, u16, f32, f32); 8] = [
            ("fetus_03", 44, 1.17, 2.0),
            ("fetus_04", 40, 1.17, 3.0),
            ("fetus_06", 46, 0.94, 2.0),
            ("fetus_07", 50, 1.02, 2.5),
            ("fetus_09", 50, 1.02, 2.5),
            ("fetus_10", 50, 1.02, 2.5),
            ("fetus_11", 50, 1.02, 2.5),
            ("fetus_12", 50, 1.02, 2.5),
        ];
        for (nom, nz, pixel, epaisseur) in attendu {
            let chemin = format!(
                "{}/../../data/sourcedata/{nom}.nii.gz",
                env!("CARGO_MANIFEST_DIR")
            );
            let info = volume_info(Path::new(&chemin)).expect(&chemin);
            assert_eq!(info.dim, vec![256, 256, nz], "{nom}");
            let voulu = [pixel, pixel, epaisseur];
            for (lu, v) in info.spacing.iter().zip(voulu) {
                assert!((lu - v).abs() < 0.005, "{nom} : espacement {lu} != {v}");
            }
        }
    }

    /// Affines de référence (nibabel, `img.affine`) pour 3 volumes dont les
    /// orientations diffèrent (LPS, LIP, PIR), arrondis à 4 décimales.
    #[test]
    fn volume_info_affine_matches_nibabel() {
        let attendu: [(&str, [[f32; 4]; 4]); 3] = [
            ("fetus_03", [
                [-1.1719, 0.0, 0.0, 155.1199],
                [0.0, -1.0771, 0.7879, 58.3842],
                [0.0, 0.4617, 1.8383, -254.6115],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            ("fetus_06", [
                [-0.8439, -0.147, -0.8127, 141.5291],
                [0.4083, -0.3038, -1.6797, -19.843],
                [0.0, -0.8747, 0.7199, -69.1684],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            ("fetus_12", [
                [-0.5306, -0.4462, 1.8271, 96.2711],
                [-0.866, 0.2733, -1.1193, 46.4229],
                [0.0, -0.8705, -1.288, -12.331],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        ];
        for (nom, voulu) in attendu {
            let chemin = format!(
                "{}/../../data/sourcedata/{nom}.nii.gz",
                env!("CARGO_MANIFEST_DIR")
            );
            let info = volume_info(Path::new(&chemin)).expect(&chemin);
            let lu = info.affine.expect("sform attendu");
            for (ligne_lue, ligne_voulue) in lu.iter().zip(voulu) {
                for (a, b) in ligne_lue.iter().zip(ligne_voulue) {
                    assert!((a - b).abs() < 1e-3, "{nom} : {lu:?} != {voulu:?}");
                }
            }
        }
    }

    /// Lit un fichier de `f32` bruts en little-endian (4 octets par valeur).
    fn lire_f32(chemin: &str) -> Vec<f32> {
        let octets = std::fs::read(chemin).expect(chemin);
        octets
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Étape 5a : voxels identiques (bit à bit) à `nibabel.get_fdata()`.
    /// Références produites par `python scripts/make_reference_volumes.py`.
    #[test]
    fn read_volume_matches_nibabel() {
        let racine = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        for nom in ["03", "04", "06", "07", "09", "10", "11", "12"] {
            let volume = read_volume(Path::new(&format!(
                "{racine}/data/sourcedata/fetus_{nom}.nii.gz"
            )))
            .unwrap();
            let attendu = lire_f32(&format!("{racine}/data/reference/fetus_{nom}_raw.f32"));
            assert_eq!(volume.len(), attendu.len(), "fetus_{nom} : taille");
            // `iter()` parcourt dans l'ordre logique [x, y, z] (z varie le plus vite),
            // quel que soit l'ordre mémoire : c'est l'ordre du fichier de référence.
            let diff = volume.iter().zip(&attendu).filter(|(a, b)| a != b).count();
            assert_eq!(diff, 0, "fetus_{nom} : {diff} voxels différents");
        }
    }

    #[test]
    fn volume_info_fails_on_missing_file() {
        let r = volume_info(Path::new("n_existe_pas.nii.gz"));
        assert!(matches!(r, Err(CoreError::Nifti(_))));
    }

    /// Règle `sform` / `qform` / `pixdim` (valeurs de référence calculées avec nibabel `Nifti1Header.set_qform` et `get_best_affine`, voir `docs/BOUNTI.md`) : quatre `qform` (identité à voxel de 0,8 mm avec translation ; rotation de 90° autour de z avec voxels anisotropes ;
    /// inversion de l'axe x donc `qfac = −1` ; rotation oblique), le `sform` prioritaire quand les deux existent, et l'absence des deux (diagonale centrée).
    #[test]
    fn the_affine_rule_matches_nibabel() {
        let en_tete = |code_q: i16, b: f32, c: f32, d: f32, decalage: [f32; 3], qfac: f32, pixdim: [f32; 3]| {
            let mut h = nifti::NiftiHeader::default();
            h.dim = [3, 4, 5, 6, 1, 1, 1, 1];
            h.sform_code = 0; // `NiftiHeader::default()` initialise un sform : on veut le qform seul
            h.qform_code = code_q;
            (h.quatern_b, h.quatern_c, h.quatern_d) = (b, c, d);
            (h.quatern_x, h.quatern_y, h.quatern_z) = (decalage[0], decalage[1], decalage[2]);
            h.pixdim = [qfac, pixdim[0], pixdim[1], pixdim[2], 0.0, 0.0, 0.0, 0.0];
            h
        };
        let proche = |a: [[f64; 4]; 4], b: [[f64; 4]; 4]| a.iter().flatten().zip(b.iter().flatten()).all(|(x, y)| (x - y).abs() < 1e-5);
        let cas: [(nifti::NiftiHeader, [[f64; 4]; 4]); 6] = [
            (en_tete(1, 0.0, 0.0, 0.0, [1.0, 2.0, 3.0], 1.0, [0.8, 0.8, 0.8]), [[0.8, 0.0, 0.0, 1.0], [0.0, 0.8, 0.0, 2.0], [0.0, 0.0, 0.8, 3.0], [0.0, 0.0, 0.0, 1.0]]),
            (en_tete(1, 0.0, 0.0, 0.707106769, [10.0, -20.0, 30.0], 1.0, [0.5, 0.5, 2.0]), [[0.0, -0.5, 0.0, 10.0], [0.5, 0.0, 0.0, -20.0], [0.0, 0.0, 2.0, 30.0], [0.0, 0.0, 0.0, 1.0]]),
            (en_tete(1, 0.0, 1.0, 0.0, [89.7, -112.65, -91.95], -1.0, [0.6, 0.6, 0.6]), [[-0.6, 0.0, 0.0, 89.7], [0.0, 0.6, 0.0, -112.65], [0.0, 0.0, 0.6, -91.95], [0.0, 0.0, 0.0, 1.0]]),
            (en_tete(1, 0.0, 0.0, 0.422386974, [0.0, 0.0, 0.0], 1.0, [0.942583799, 0.959917367, 1.5]), [[0.6062496, -0.7350244, 0.0, 0.0], [0.7217518, 0.6173982, 0.0, 0.0], [0.0, 0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            // rotation de 90° autour de x (b non nul) avec voxel de 3 mm en z
            (en_tete(1, 0.707106769, 0.0, 0.0, [5.0, 6.0, 7.0], 1.0, [1.0, 1.0, 3.0]), [[1.0, 0.0, 0.0, 5.0], [0.0, 0.0, -3.0, 6.0], [0.0, 1.0, 0.0, 7.0], [0.0, 0.0, 0.0, 1.0]]),
            // rotation générale en 3D (b, c et d non nuls), voxels anisotropes
            (en_tete(1, 0.239298344, 0.189307854, 0.038134575, [-3.0, 4.0, -5.0], 1.0, [0.800000012, 1.0, 2.0]), [[0.7403333, 0.0180283, 0.7570446, -3.0], [0.1305407, 0.8825641, -0.8819392, 4.0], [-0.2736161, 0.4698463, 1.6275954, -5.0], [0.0, 0.0, 0.0, 1.0]]),
        ];
        for (i, (h, attendu)) in cas.iter().enumerate() {
            let a = affine_depuis_en_tete(h).unwrap();
            assert_eq!(a.source, SourceAffine::Qform, "cas {i}");
            assert!(proche(a.matrice, *attendu), "cas {i} : {:?} contre {attendu:?}", a.matrice);
        }
        // le sform est prioritaire, même quand le qform existe et diffère
        let mut h = en_tete(1, 0.0, 0.0, 0.0, [1.0, 2.0, 3.0], 1.0, [0.8, 0.8, 0.8]);
        h.sform_code = 1;
        h.srow_x = [0.8, 0.0, 0.0, -38.285];
        h.srow_y = [0.0, 0.8, 0.0, -49.232];
        h.srow_z = [0.0, 0.0, 0.8, -64.298];
        let a = affine_depuis_en_tete(&h).unwrap();
        assert_eq!(a.source, SourceAffine::Sform);
        assert!(proche(a.matrice, [[0.8, 0.0, 0.0, -38.285], [0.0, 0.8, 0.0, -49.232], [0.0, 0.0, 0.8, -64.298], [0.0, 0.0, 0.0, 1.0]]));
        // ni sform ni qform : diagonale des pixdim, volume centré (nibabel : -(dim - 1) / 2 · pixdim), ici dim = (4, 5, 6) et pixdim 0,8
        let h = en_tete(0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0], 1.0, [0.8, 0.8, 0.8]);
        let a = affine_depuis_en_tete(&h).unwrap();
        assert_eq!(a.source, SourceAffine::Pixdim);
        assert!(proche(a.matrice, [[0.8, 0.0, 0.0, -1.2], [0.0, 0.8, 0.0, -1.6], [0.0, 0.0, 0.8, -2.0], [0.0, 0.0, 0.0, 1.0]]), "{:?}", a.matrice);
    }

    /// `definir_affine` (écriture) est l'inverse de `affine_depuis_en_tete` (lecture) : pour les six affines de référence (dont inversion d'axe et rotations d'axes variés) et deux cas limites (rotation de 180° autour de chaque axe, volume à une taille de voxel inégale), le `sform` relu est exactement
    /// la matrice donnée (à la précision `f32`), le `qform` relu seul (`sform_code = 0`) aussi, et `pixdim` contient les tailles de voxel et `qfac`. Les quaternions des cas de référence sont ceux de nibabel (`set_qform`).
    #[test]
    fn writing_the_affine_is_consistent_between_sform_and_qform() {
        let matrices: Vec<[[f64; 4]; 4]> = vec![
            [[0.8, 0.0, 0.0, 1.0], [0.0, 0.8, 0.0, 2.0], [0.0, 0.0, 0.8, 3.0], [0.0, 0.0, 0.0, 1.0]],
            [[0.0, -0.5, 0.0, 10.0], [0.5, 0.0, 0.0, -20.0], [0.0, 0.0, 2.0, 30.0], [0.0, 0.0, 0.0, 1.0]],
            [[-0.6, 0.0, 0.0, 89.7], [0.0, 0.6, 0.0, -112.65], [0.0, 0.0, 0.6, -91.95], [0.0, 0.0, 0.0, 1.0]],
            [[0.6062496, -0.7350244, 0.0, 0.0], [0.7217518, 0.6173982, 0.0, 0.0], [0.0, 0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0, 5.0], [0.0, 0.0, -3.0, 6.0], [0.0, 1.0, 0.0, 7.0], [0.0, 0.0, 0.0, 1.0]],
            [[0.7403333, 0.0180283, 0.7570446, -3.0], [0.1305407, 0.8825641, -0.8819392, 4.0], [-0.2736161, 0.4698463, 1.6275954, -5.0], [0.0, 0.0, 0.0, 1.0]],
            // rotations de 180° autour de x, y et z (a = 0 : branches du quaternion hors du cas général)
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, -1.0, 2.0], [0.0, 0.0, 0.0, 1.0]],
            [[-1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
            [[-2.0, 0.0, 0.0, 0.5], [0.0, -2.0, 0.0, 0.5], [0.0, 0.0, 3.0, 0.5], [0.0, 0.0, 0.0, 1.0]],
            // rotations d'axes obliques (voxels 1 / 1,5 / 2 mm) qui exercent chaque branche du quaternion hors du cas général, termes croisés non nuls ; les trois dernières (150°, composante négative) imposent le changement de signe du quaternion
            // 180° autour de (1, 1, 0)
            [[0.0, 1.5, 0.0, 0.5], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, -2.0, 1.5], [0.0, 0.0, 0.0, 1.0]],
            // 180°, axe à composante y dominante
            [[-0.8200720, 0.7197121, 0.6237505, 0.5], [0.4798081, 0.4192323, 1.6633347, 1.0], [0.3118752, 1.2475010, -0.9188325, 1.5], [0.0, 0.0, 0.0, 1.0]],
            // 180°, axe à composante z dominante
            [[-0.82, 0.36, 1.0392305, 0.5], [0.24, -1.02, 1.3856406, 1.0], [0.5196152, 1.0392305, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]],
            // 150°, axe à composante x négative
            [[0.3282011, -1.3683593, -0.4903186, 0.5], [-0.5805436, -0.5992960, 1.4189353, 1.0], [-0.7451531, -0.1357836, -1.3214425, 1.5], [0.0, 0.0, 0.0, 1.0]],
            // 150°, axe à composante y négative
            [[-0.6690257, -0.9997080, -0.6579353, 0.5], [-0.4498599, 1.0731672, -1.0691395, 1.0], [0.5916340, -0.3144773, -1.5569399, 1.5], [0.0, 0.0, 0.0, 1.0]],
            // 150°, axe à composante z négative
            [[-0.7866201, 0.8748713, -0.4052217, 0.5], [-0.3450316, -1.0310451, -1.2782563, 1.0], [-0.5120372, -0.6492658, 1.4838653, 1.5], [0.0, 0.0, 0.0, 1.0]],
        ];
        let proche = |a: [[f64; 4]; 4], b: [[f64; 4]; 4]| a.iter().flatten().zip(b.iter().flatten()).all(|(x, y)| (x - y).abs() < 2e-5 * (1.0 + y.abs()));
        for (i, m) in matrices.iter().enumerate() {
            let mut h = nifti::NiftiHeader::default();
            h.dim = [3, 4, 5, 6, 1, 1, 1, 1];
            definir_affine(&mut h, *m);
            assert_eq!((h.sform_code, h.qform_code), (1, 1), "cas {i} : codes");
            let par_sform = affine_depuis_en_tete(&h).unwrap();
            assert_eq!(par_sform.source, SourceAffine::Sform);
            assert!(proche(par_sform.matrice, *m), "cas {i} : sform {:?}", par_sform.matrice);
            h.sform_code = 0;
            let par_qform = affine_depuis_en_tete(&h).unwrap();
            assert_eq!(par_qform.source, SourceAffine::Qform);
            assert!(proche(par_qform.matrice, *m), "cas {i} : qform {:?} contre {m:?}", par_qform.matrice);
            let tailles: Vec<f64> = (0..3).map(|c| (0..3).map(|l| m[l][c] * m[l][c]).sum::<f64>().sqrt()).collect();
            for c in 0..3 {
                assert!((f64::from(h.pixdim[c + 1]) - tailles[c]).abs() < 1e-5, "cas {i} : pixdim {c}");
            }
        }
        // valeurs de nibabel pour les quaternions : identité, rotation de 90° autour de z, inversion de l'axe x (qfac = -1)
        let mut h = nifti::NiftiHeader::default();
        definir_affine(&mut h, matrices[1]);
        assert!(h.quatern_b.abs() < 1e-6 && h.quatern_c.abs() < 1e-6 && (h.quatern_d - 0.707106769).abs() < 1e-6 && h.pixdim[0] == 1.0, "rotation de 90° : {} {} {} {}", h.quatern_b, h.quatern_c, h.quatern_d, h.pixdim[0]);
        definir_affine(&mut h, matrices[2]);
        assert!(h.quatern_b.abs() < 1e-6 && (h.quatern_c - 1.0).abs() < 1e-6 && h.quatern_d.abs() < 1e-6 && h.pixdim[0] == -1.0, "inversion de x : {} {} {} {}", h.quatern_b, h.quatern_c, h.quatern_d, h.pixdim[0]);
        assert_eq!((h.quatern_x, h.quatern_y, h.quatern_z), (89.7, -112.65, -91.95));
    }
}
