//! Lecture des acquisitions cartésiennes 3D de l'IRM ultra-bas champ (48 mT), pour la reconstruction (docs/ULF.md, P3).
//!
//! Deux formats, reconnus aux jeux de données présents dans le fichier HDF5 :
//! - **Spinloop** (`spinloop_recon.mrd.write`) : ISMRMRD 1.x, `dataset/xml` et `dataset/data`, une acquisition par
//!   écho ;
//! - **fmi** (`fmi.io.writers.write_hdf5_file`, dépôt fridge-magnet-imaging) : `xml_header` et `kspace` (tableau dense).
//!
//! Les deux rendent une [`Acquisition`] : k-space `[lecture, phase 1, phase 2]` en `Complex32`, nombre de lectures par
//! ligne, et ce que dit l'en-tête. Les axes de la machine (`axes` de Spinloop, `plane` de fmi) sont rendus tels quels :
//! la convention commune des repères n'est pas encore fixée.

pub mod entete;

use std::path::{Path, PathBuf};

use hdf5::types::{VarLenArray, VarLenAscii, VarLenUnicode};
use hdf5::H5Type;
use ndarray::{Array2, Array3};
use num_complex::Complex32;

pub use entete::{Entete, ErreurEntete};

/// Format du fichier lu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// ISMRMRD 1.x écrit par Spinloop.
    Spinloop,
    /// HDF5 de fridge-magnet-imaging (`xml_header`, `kspace`).
    Fmi,
}

/// Polarité du gradient de lecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarite {
    /// Lecture dans le sens habituel (Spinloop : `read_sign` = 1 ; fmi : `invert_ro_polarity` = False).
    Normale,
    /// Lecture inversée (Spinloop : `read_sign` = -1 ; fmi : `invert_ro_polarity` = True).
    Inversee,
}

/// Une acquisition 3D cartésienne, prête pour la reconstruction.
#[derive(Debug, Clone)]
pub struct Acquisition {
    /// Format du fichier d'origine.
    pub format: Format,
    /// k-space `[lecture, phase 1, phase 2]` ; les lignes non acquises valent 0.
    pub kspace: Array3<Complex32>,
    /// Nombre de lectures moyennées dans chaque ligne `[phase 1, phase 2]` (0 : ligne non acquise). fmi : 1 ou 0.
    pub lectures_par_ligne: Array2<u16>,
    /// Ce que dit l'en-tête XML (matrice, champ de vue, centre, porteuse, TR, paramètres utilisateur).
    pub entete: Entete,
    /// En-tête XML brut (provenance).
    pub xml: String,
    /// Temps d'échantillonnage en µs (Spinloop : `dwell_us` ; fmi : `dwellTime`, en ms).
    pub dwell_us: f64,
    /// Polarité de la lecture ; `None` si le fichier ne la dit pas (Spinloop sans `read_sign`).
    pub polarite_lecture: Option<Polarite>,
    /// Vrai si l'écrivain a retourné le k-space le long de la lecture pour le ramener à la polarité normale (fmi, lecture
    /// inversée) : les échantillons ne sont alors plus dans l'ordre d'acquisition.
    pub lecture_retournee_a_l_ecriture: bool,
    /// Axes de la machine `[lecture, phase 1, phase 2]` (Spinloop, paramètre `axes`, ex. « yzx »).
    pub axes: Option<String>,
    /// Plan d'acquisition au sens de fmi (paramètre `plane`, ex. « yz »).
    pub plan: Option<String>,
}

/// Erreur de lecture d'une acquisition.
#[derive(Debug)]
pub enum ErreurLecture {
    /// Fichier illisible ou jeu de données attendu illisible.
    Hdf5 {
        /// Fichier concerné.
        chemin: PathBuf,
        /// Message de HDF5.
        message: String,
    },
    /// Ni `dataset/data` (Spinloop) ni `kspace` (fmi) dans le fichier.
    FormatInconnu(PathBuf),
    /// En-tête XML illisible ou incomplet.
    Entete(ErreurEntete),
    /// Contenu incohérent avec l'en-tête (taille, indice hors de la matrice, paramètre manquant).
    Incoherence(String),
}

impl std::fmt::Display for ErreurLecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErreurLecture::Hdf5 { chemin, message } => write!(f, "{} : {message}", chemin.display()),
            ErreurLecture::FormatInconnu(chemin) => write!(
                f,
                "{} : ni dataset/data (ISMRMRD de Spinloop) ni kspace (fridge-magnet-imaging)",
                chemin.display()
            ),
            ErreurLecture::Entete(e) => write!(f, "{e}"),
            ErreurLecture::Incoherence(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ErreurLecture {}

impl From<ErreurEntete> for ErreurLecture {
    fn from(e: ErreurEntete) -> Self {
        ErreurLecture::Entete(e)
    }
}

/// Lit une acquisition, dans l'un des deux formats (reconnu aux jeux de données présents).
pub fn lire(chemin: &Path) -> Result<Acquisition, ErreurLecture> {
    let h5 = |e: hdf5::Error| ErreurLecture::Hdf5 {
        chemin: chemin.to_path_buf(),
        message: e.to_string(),
    };
    let fichier = hdf5::File::open(chemin).map_err(h5)?;
    if fichier.link_exists("dataset/data") {
        lire_spinloop(&fichier).map_err(|e| e.avec_chemin(chemin))
    } else if fichier.link_exists("kspace") {
        lire_fmi(&fichier).map_err(|e| e.avec_chemin(chemin))
    } else {
        Err(ErreurLecture::FormatInconnu(chemin.to_path_buf()))
    }
}

impl ErreurLecture {
    fn avec_chemin(self, chemin: &Path) -> Self {
        match self {
            ErreurLecture::Hdf5 { message, .. } => ErreurLecture::Hdf5 {
                chemin: chemin.to_path_buf(),
                message,
            },
            autre => autre,
        }
    }
}

fn erreur_hdf5(e: hdf5::Error) -> ErreurLecture {
    ErreurLecture::Hdf5 {
        chemin: PathBuf::new(),
        message: e.to_string(),
    }
}

/// Chaîne XML d'un jeu de données de longueur variable : h5py écrit de l'UTF-8 (fmi), ismrmrd de l'ASCII (Spinloop) ;
/// scalaire (fmi) ou tableau d'un élément (Spinloop).
fn lire_xml(jeu: &hdf5::Dataset) -> Result<String, ErreurLecture> {
    let premier = |v: Vec<String>| {
        v.into_iter()
            .next()
            .ok_or_else(|| ErreurLecture::Incoherence("en-tête XML vide".into()))
    };
    match jeu.read_raw::<VarLenUnicode>() {
        Ok(v) => premier(v.into_iter().map(|s| s.as_str().to_string()).collect()),
        Err(_) => premier(
            jeu.read_raw::<VarLenAscii>()
                .map_err(erreur_hdf5)?
                .into_iter()
                .map(|s| s.as_str().to_string())
                .collect(),
        ),
    }
}

fn parametre_requis<'a>(entete: &'a Entete, nom: &str) -> Result<&'a str, ErreurLecture> {
    entete
        .parametres
        .get(nom)
        .map(String::as_str)
        .ok_or_else(|| ErreurLecture::Incoherence(format!("paramètre utilisateur « {nom} » absent")))
}

// --- fmi --------------------------------------------------------------------------------------------------------------

fn lire_fmi(fichier: &hdf5::File) -> Result<Acquisition, ErreurLecture> {
    let xml = lire_xml(&fichier.dataset("xml_header").map_err(erreur_hdf5)?)?;
    let entete = Entete::depuis_xml(&xml)?;
    let kspace: Array3<Complex32> = fichier
        .dataset("kspace")
        .map_err(erreur_hdf5)?
        .read::<Complex32, ndarray::Ix3>()
        .map_err(erreur_hdf5)?;
    if kspace.shape() != entete.matrice {
        return Err(ErreurLecture::Incoherence(format!(
            "kspace {:?}, matrice de l'en-tête {:?}",
            kspace.shape(),
            entete.matrice
        )));
    }
    // une ligne est acquise si l'un de ses échantillons est non nul (l'écrivain met à zéro les autres)
    let lectures_par_ligne = kspace.map_axis(ndarray::Axis(0), |ligne| {
        u16::from(ligne.iter().any(|z| z.re != 0.0 || z.im != 0.0))
    });
    let inversee = match parametre_requis(&entete, "invert_ro_polarity")? {
        "True" => true,
        "False" => false,
        autre => return Err(ErreurLecture::Incoherence(format!("invert_ro_polarity = « {autre} »"))),
    };
    let dwell_ms: f64 = parametre_requis(&entete, "dwellTime")?
        .parse()
        .map_err(|_| ErreurLecture::Incoherence("dwellTime n'est pas un nombre".into()))?;
    let plan = entete.parametres.get("plane").cloned();
    Ok(Acquisition {
        format: Format::Fmi,
        kspace,
        lectures_par_ligne,
        dwell_us: dwell_ms * 1e3,
        polarite_lecture: Some(if inversee {
            Polarite::Inversee
        } else {
            Polarite::Normale
        }),
        lecture_retournee_a_l_ecriture: inversee,
        axes: None,
        plan,
        entete,
        xml,
    })
}

// --- Spinloop (ISMRMRD 1.x) -------------------------------------------------------------------------------------------

/// Indices d'une acquisition ISMRMRD : seuls les champs lus sont déclarés, HDF5 convertit le type composé du fichier
/// champ par champ, **par nom** (l'ordre et les champs absents d'ici n'importent pas).
#[derive(H5Type, Clone, Debug)]
#[repr(C)]
struct Indices {
    kspace_encode_step_1: u16,
    kspace_encode_step_2: u16,
}

/// En-tête d'une acquisition ISMRMRD, réduit aux champs lus.
#[derive(H5Type, Clone, Debug)]
#[repr(C)]
struct EnteteAcquisition {
    flags: u64,
    number_of_samples: u16,
    active_channels: u16,
    idx: Indices,
}

/// Une acquisition ISMRMRD : en-tête et échantillons (réels et imaginaires entrelacés, longueur variable).
#[derive(H5Type, Clone, Debug)]
#[repr(C)]
struct AcquisitionBrute {
    head: EnteteAcquisition,
    data: VarLenArray<f32>,
}

/// `ACQ_IS_DUMMYSCAN_DATA` (bit 27 d'ISMRMRD, numéroté à partir de 1) : train joué pour atteindre l'équilibre.
const DRAPEAU_EQUILIBRE: u64 = 1 << (27 - 1);

fn lire_spinloop(fichier: &hdf5::File) -> Result<Acquisition, ErreurLecture> {
    let xml = lire_xml(&fichier.dataset("dataset/xml").map_err(erreur_hdf5)?)?;
    let entete = Entete::depuis_xml(&xml)?;
    let [n_lecture, n1, n2] = entete.matrice;
    let brutes = fichier
        .dataset("dataset/data")
        .map_err(erreur_hdf5)?
        .read_raw::<AcquisitionBrute>()
        .map_err(erreur_hdf5)?;
    // somme en f64 puis moyenne, comme spinloop_recon.cartesian.grid (complexes de numpy en double précision)
    let mut somme = Array3::<num_complex::Complex64>::zeros((n_lecture, n1, n2));
    let mut compte = Array2::<u16>::zeros((n1, n2));
    for (n, a) in brutes.iter().enumerate() {
        if a.head.flags & DRAPEAU_EQUILIBRE != 0 {
            continue;
        }
        let (i1, i2) = (
            a.head.idx.kspace_encode_step_1 as usize,
            a.head.idx.kspace_encode_step_2 as usize,
        );
        if i1 >= n1 || i2 >= n2 {
            return Err(ErreurLecture::Incoherence(format!(
                "acquisition {n} : ligne ({i1}, {i2}) hors de la matrice {n1} × {n2}"
            )));
        }
        if a.head.active_channels != 1
            || a.head.number_of_samples as usize != n_lecture
            || a.data.len() != 2 * n_lecture
        {
            return Err(ErreurLecture::Incoherence(format!(
                "acquisition {n} : {} canaux, {} échantillons ({} valeurs) ; attendu 1 canal, {n_lecture} échantillons",
                a.head.active_channels,
                a.head.number_of_samples,
                a.data.len()
            )));
        }
        for (s, paire) in a.data.as_slice().as_chunks::<2>().0.iter().enumerate() {
            somme[[s, i1, i2]] += num_complex::Complex64::new(paire[0] as f64, paire[1] as f64);
        }
        compte[[i1, i2]] += 1;
    }
    let kspace = Array3::from_shape_fn((n_lecture, n1, n2), |(s, i1, i2)| {
        let c = compte[[i1, i2]];
        if c == 0 {
            Complex32::new(0.0, 0.0)
        } else {
            let m = somme[[s, i1, i2]] / c as f64;
            Complex32::new(m.re as f32, m.im as f32)
        }
    });
    let dwell_us: f64 = parametre_requis(&entete, "dwell_us")?
        .parse()
        .map_err(|_| ErreurLecture::Incoherence("dwell_us n'est pas un nombre".into()))?;
    let polarite_lecture = match entete.parametres.get("read_sign").map(String::as_str) {
        Some("1") => Some(Polarite::Normale),
        Some("-1") => Some(Polarite::Inversee),
        None => None,
        Some(autre) => return Err(ErreurLecture::Incoherence(format!("read_sign = « {autre} »"))),
    };
    let axes = entete.parametres.get("axes").cloned();
    Ok(Acquisition {
        format: Format::Spinloop,
        kspace,
        lectures_par_ligne: compte,
        dwell_us,
        polarite_lecture,
        lecture_retournee_a_l_ecriture: false,
        axes,
        plan: None,
        entete,
        xml,
    })
}
