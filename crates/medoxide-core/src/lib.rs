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
}
