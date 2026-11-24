//! Entrées de l'étape 1 depuis une paire de fichiers ISMRMRD (P6, docs/ULF.md) : portage de `utils_io.load_pair` et de
//! `denoise_joint.Context` (fmi `cb7174b`).
//!
//! Lecture par `medoxide-ismrmrd` (format fmi : inv déjà retourné le long de la lecture), position de l'écho, fenêtre
//! de Tukey commune autour de l'écho, temps de lecture, bornes de l'amplitude de B0, espacement et masque de l'objet.
//! Tout est calculé en `f64`, comme dans fmi.

use std::f64::consts::PI;
use std::path::Path;

use ndarray::{Array3, Axis, Zip};
use rustfft::num_complex::Complex64;

use medoxide_ismrmrd::{lire, Acquisition, Format, Polarite};

use crate::ModeleCpr;

/// Largeur (échantillons) de la transition cosinus en bord de fenêtre (`symmetric_readout_window`, `taper`).
pub const TAPER: f64 = 6.0;
/// Durée ajoutée à la lecture (`build_time_array`, `readout_padding`), en secondes.
pub const REMPLISSAGE_LECTURE_S: f64 = 500e-6;
/// Seuil du masque de l'objet, relatif au maximum de l'image moyenne non corrigée (`denoise_joint.Context`).
pub const SEUIL_MASQUE: f64 = 0.2;

/// Entrées de la méthode pour une paire reg / inv.
#[derive(Debug, Clone)]
pub struct Entrees {
    /// k-space de reg, fenêtré autour de son écho, `[lecture, phase 1, phase 2]`.
    pub k_reg: Array3<Complex64>,
    /// k-space de inv (retourné à l'écriture), fenêtré autour de son écho.
    pub k_inv: Array3<Complex64>,
    /// Temps de lecture (s), un par échantillon de lecture, nul au milieu de la lecture.
    pub temps: Vec<f64>,
    /// Temps d'échantillonnage (s).
    pub dwell_s: f64,
    /// Bornes de l'amplitude de B0 (Hz) : `[min, max]`.
    pub bornes_b0: [f64; 2],
    /// Espacement des voxels (mm), arrondi `f32` de fmi compris.
    pub espacement_mm: [f64; 3],
    /// Masque de l'objet (évaluation).
    pub masque: Array3<bool>,
    /// Positions de l'écho (échantillon, sous-pixel) de reg et de inv.
    pub echo_reg: f64,
    pub echo_inv: f64,
    /// Demi-largeur commune de la fenêtre (échantillons).
    pub demi_largeur: f64,
    /// Plan d'acquisition (paramètre `plane` de fmi).
    pub plan: Option<String>,
}

/// Paire illisible ou incohérente.
#[derive(Debug)]
pub struct ErreurEntrees(pub String);

impl std::fmt::Display for ErreurEntrees {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ErreurEntrees {}

fn en_f64(a: &Acquisition) -> Array3<Complex64> {
    a.kspace.mapv(|c| Complex64::new(c.re as f64, c.im as f64))
}

/// Position sous-pixel de l'écho le long de la lecture (`utils_io.echo_position`) : maximum du logarithme de l'énergie
/// moyenne par échantillon de lecture, affiné par une parabole sur les trois points autour du maximum.
pub fn position_echo(k: &Array3<Complex64>) -> f64 {
    let n_autres = (k.len() / k.len_of(Axis(0))) as f64;
    let log_e: Vec<f64> = k
        .axis_iter(Axis(0))
        .map(|plan| (plan.iter().map(|c| c.norm().powi(2)).sum::<f64>() / n_autres).ln())
        .collect();
    // premier maximum, comme `argmax`
    let mut i = 0;
    for (j, v) in log_e.iter().enumerate() {
        if *v > log_e[i] {
            i = j;
        }
    }
    if i == 0 || i == log_e.len() - 1 {
        return i as f64;
    }
    let (y0, y1, y2) = (log_e[i - 1], log_e[i], log_e[i + 1]);
    i as f64 + 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
}

/// Fenêtre de Tukey 1D centrée sur `echo` (`utils_io.readout_window`) : 1 à moins de `demi_largeur - taper` de l'écho,
/// 0 au-delà de `demi_largeur`, transition cosinus entre les deux.
pub fn fenetre_lecture(n: usize, echo: f64, demi_largeur: f64, taper: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let d = (i as f64 - echo).abs();
            let rampe = ((demi_largeur - d) / taper).clamp(0.0, 1.0);
            0.5 - 0.5 * (PI * rampe).cos()
        })
        .collect()
}

/// Temps de lecture (`cpr.tse_time_array`, `symmetric_time=True`) : pas de `dwell + remplissage / n`, nul à
/// l'échantillon `n / 2`.
pub fn temps_lecture(n: usize, dwell_s: f64, remplissage_s: f64) -> Vec<f64> {
    let pas = dwell_s + remplissage_s / n as f64;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * pas).collect();
    let t0 = t[n / 2];
    t.into_iter().map(|v| v - t0).collect()
}

/// Bornes physiques de l'amplitude de B0 (`utils_cpr.b0_bounds_hz`) : `1 / (2π T_lecture)` et `1 / (2 dwell)`.
pub fn bornes_b0(temps: &[f64], dwell_s: f64) -> [f64; 2] {
    let duree = (temps[temps.len() - 1] - temps[0]) + dwell_s;
    [1.0 / (2.0 * PI * duree), 0.5 / dwell_s]
}

/// Espacement (mm) comme `denoise_joint.Context` : FOV en mètres converti en `f32` (`scan_params`), puis
/// `1e3 · fov / n` calculé en `f32`.
pub fn espacement_mm(champ_de_vue_mm: [f64; 3], dims: [usize; 3]) -> [f64; 3] {
    std::array::from_fn(|a| {
        let fov_m = (champ_de_vue_mm[a] * 1e-3) as f32;
        ((1e3_f32 * fov_m) / dims[a] as f32) as f64
    })
}

/// Masque de l'objet (`denoise_joint.Context.mask`) : moyenne des magnitudes de Aᴴ(0) k_reg et Aᴴ(0) k_inv, seuillée à
/// [`SEUIL_MASQUE`] fois son maximum.
pub fn masque_objet(k_reg: &Array3<Complex64>, k_inv: &Array3<Complex64>, temps: &[f64]) -> Array3<bool> {
    let (moyenne, seuil) = image_masque(k_reg, k_inv, temps);
    moyenne.mapv(|v| v > seuil)
}

/// Image moyenne non corrigée et seuil du masque de l'objet (le test compare au seuil les voxels qui diffèrent).
pub fn image_masque(k_reg: &Array3<Complex64>, k_inv: &Array3<Complex64>, temps: &[f64]) -> (Array3<f64>, f64) {
    let dims: [usize; 3] = k_reg.dim().into();
    let modele = ModeleCpr::<f64>::new(dims, temps);
    let zero = Array3::<f64>::zeros(dims);
    let a = modele.adjoint(k_reg, &zero);
    let b = modele.adjoint(k_inv, &zero);
    let moyenne = Zip::from(&a).and(&b).map_collect(|x, y| (x.norm() + y.norm()) / 2.0);
    let max = moyenne.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (moyenne, SEUIL_MASQUE * max)
}

/// Temps d'échantillonnage (s) tel que fmi le lit : texte de `dwellTime` (ms) · 1e-3 pour le format fmi.
fn dwell_s(a: &Acquisition) -> Result<f64, ErreurEntrees> {
    match a.format {
        Format::Fmi => a
            .entete
            .parametres
            .get("dwellTime")
            .ok_or_else(|| ErreurEntrees("dwellTime absent".into()))?
            .trim()
            .parse::<f64>()
            .map(|ms| ms * 1e-3)
            .map_err(|_| ErreurEntrees("dwellTime n'est pas un nombre".into())),
        Format::Spinloop => Ok(a.dwell_us * 1e-6),
    }
}

/// Vérifie que `reg` et `inv` forment une paire que la méthode sait traiter.
fn verifier_paire(reg: &Acquisition, inv: &Acquisition) -> Result<(), ErreurEntrees> {
    let refus = |m: String| Err(ErreurEntrees(m));
    if reg.format != Format::Fmi || inv.format != Format::Fmi {
        return refus("seul le format fmi est pris en charge (P6 ; Spinloop à lecture inversée : hors P6)".into());
    }
    if reg.polarite_lecture != Some(Polarite::Normale) || inv.polarite_lecture != Some(Polarite::Inversee) {
        return refus(format!(
            "polarités attendues : reg normale, inv inversée (lues : {:?}, {:?})",
            reg.polarite_lecture, inv.polarite_lecture
        ));
    }
    if !inv.lecture_retournee_a_l_ecriture {
        return refus("inv n'a pas été retourné le long de la lecture à l'écriture".into());
    }
    if reg.phases_retournees_a_l_ecriture != inv.phases_retournees_a_l_ecriture {
        return refus(
            "axes de phase retournés à l'écriture dans un seul des deux fichiers (invert_ph_polarity)".into(),
        );
    }
    if reg.plan != inv.plan {
        return refus(format!("plans différents : {:?}, {:?}", reg.plan, inv.plan));
    }
    if reg.kspace.dim() != inv.kspace.dim() || reg.entete.matrice != inv.entete.matrice {
        return refus("matrices différentes".into());
    }
    if reg.entete.champ_de_vue_mm != inv.entete.champ_de_vue_mm {
        return refus("champs de vue différents".into());
    }
    if dwell_s(reg)? != dwell_s(inv)? {
        return refus("temps d'échantillonnage différents".into());
    }
    Ok(())
}

/// Entrées de la méthode à partir de deux acquisitions déjà lues.
pub fn depuis_acquisitions(reg: &Acquisition, inv: &Acquisition) -> Result<Entrees, ErreurEntrees> {
    verifier_paire(reg, inv)?;
    let (mut k_reg, mut k_inv) = (en_f64(reg), en_f64(inv));
    let n = k_reg.len_of(Axis(0));
    let (echo_reg, echo_inv) = (position_echo(&k_reg), position_echo(&k_inv));
    let demi_largeur = [echo_reg, echo_inv]
        .iter()
        .map(|e| e.min((n - 1) as f64 - e))
        .fold(f64::INFINITY, f64::min);
    for (k, echo) in [(&mut k_reg, echo_reg), (&mut k_inv, echo_inv)] {
        let w = fenetre_lecture(n, echo, demi_largeur, TAPER);
        for (i, mut plan) in k.axis_iter_mut(Axis(0)).enumerate() {
            plan.mapv_inplace(|c| c * w[i]);
        }
    }
    let dwell = dwell_s(reg)?;
    let temps = temps_lecture(n, dwell, REMPLISSAGE_LECTURE_S);
    let masque = masque_objet(&k_reg, &k_inv, &temps);
    let dims: [usize; 3] = k_reg.dim().into();
    Ok(Entrees {
        bornes_b0: bornes_b0(&temps, dwell),
        espacement_mm: espacement_mm(reg.entete.champ_de_vue_mm, dims),
        k_reg,
        k_inv,
        temps,
        dwell_s: dwell,
        masque,
        echo_reg,
        echo_inv,
        demi_largeur,
        plan: reg.plan.clone(),
    })
}

/// Entrées de la méthode à partir d'une paire de fichiers (reg : lecture normale ; inv : lecture inversée).
pub fn depuis_fichiers(reg: &Path, inv: &Path) -> Result<Entrees, ErreurEntrees> {
    let lire = |p: &Path| lire(p).map_err(|e| ErreurEntrees(format!("{} : {e:?}", p.display())));
    depuis_acquisitions(&lire(reg)?, &lire(inv)?)
}
