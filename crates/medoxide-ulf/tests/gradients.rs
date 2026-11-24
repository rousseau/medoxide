//! Critère 2 de P4 bis (docs/ULF.md) : gradients écrits à la main du modèle direct et de l'adjoint, comparés aux
//! différences finies et à l'autograd de PyTorch (fmi).
//!
//! Cas synthétique versionné (`tests/donnees/cpr_gradients_synthetique.h5`, écrit par `scripts/cas_cpr.py gradients`) ;
//! plans de P1 si `MEDOXIDE_ULF_P1_GRAD` désigne un dossier de fichiers écrits de même (hors dépôt).
//!
//! Pertes : L_y = Σ Re(conj(w_y) · A(b0) x) et L_z = Σ Re(conj(w_z) · Aᴴ(b0) k), d'où ∂L/∂y = w_y et ∂L/∂z = w_z.
//! Seuils (écarts relatifs L2) : PyTorch ≤ 1e-10 (`f64`) et ≤ 1e-5 (`f32`) sur le cas synthétique, ≤ 1e-8 et ≤ 1e-4
//! sur P1 ; différences finies (`f64`) ≤ 1e-6.

use std::path::Path;

use medoxide_ulf::{ModeleCpr, Reel};
use ndarray::{Array1, Array3, Ix1, Ix3, Zip};
use rustfft::num_complex::{Complex, Complex64};

struct Cas {
    x: Array3<Complex64>,
    k: Array3<Complex64>,
    b0: Array3<f64>,
    temps: Array1<f64>,
    w_y: Array3<Complex64>,
    w_z: Array3<Complex64>,
    grad_y_x: Array3<Complex64>,
    grad_y_b0: Array3<f64>,
    grad_z_b0: Array3<f64>,
}

fn lire(chemin: &Path) -> Cas {
    let f = hdf5::File::open(chemin).unwrap();
    let c = |nom: &str| f.dataset(nom).unwrap().read::<Complex64, Ix3>().unwrap();
    let r = |nom: &str| f.dataset(nom).unwrap().read::<f64, Ix3>().unwrap();
    Cas {
        x: c("x"),
        k: c("k"),
        w_y: c("w_y"),
        w_z: c("w_z"),
        grad_y_x: c("grad_y_x"),
        b0: r("b0"),
        grad_y_b0: r("grad_y_b0"),
        grad_z_b0: r("grad_z_b0"),
        temps: f.dataset("temps").unwrap().read::<f64, Ix1>().unwrap(),
    }
}

fn vers<T: Reel>(a: &Array3<Complex64>) -> Array3<Complex<T>> {
    a.mapv(|z| Complex::new(T::from_f64(z.re).unwrap(), T::from_f64(z.im).unwrap()))
}

fn ecart_c<T: Reel>(a: &Array3<Complex<T>>, r: &Array3<Complex64>) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (x, y) in a.iter().zip(r) {
        let x = Complex64::new(x.re.to_f64().unwrap(), x.im.to_f64().unwrap());
        num += (x - y).norm_sqr();
        den += y.norm_sqr();
    }
    (num / den).sqrt()
}

fn ecart_r<T: Reel>(a: &Array3<T>, r: &Array3<f64>) -> f64 {
    let num: f64 = a.iter().zip(r).map(|(x, y)| (x.to_f64().unwrap() - y).powi(2)).sum();
    (num / r.iter().map(|y| y * y).sum::<f64>()).sqrt()
}

/// Écarts à PyTorch des trois gradients (∂L_y/∂x, ∂L_y/∂b0, ∂L_z/∂b0) dans la précision T.
fn ecarts_pytorch<T: Reel>(cas: &Cas) -> [f64; 3] {
    let temps: Vec<T> = cas.temps.iter().map(|&t| T::from_f64(t).unwrap()).collect();
    let modele = ModeleCpr::new(cas.x.dim().into(), &temps);
    let b0 = cas.b0.mapv(|v| T::from_f64(v).unwrap());
    let (gx, gb0) = modele.gradient_direct(&vers::<T>(&cas.x), &b0, &vers::<T>(&cas.w_y));
    let (_, atk) = modele.adjoint_et_pondere(&vers::<T>(&cas.k), &b0);
    let gz = modele.gradient_adjoint_b0(&atk, &vers::<T>(&cas.w_z));
    [
        ecart_c(&gx, &cas.grad_y_x),
        ecart_r(&gb0, &cas.grad_y_b0),
        ecart_r(&gz, &cas.grad_z_b0),
    ]
}

fn re_produit(w: &Array3<Complex64>, y: &Array3<Complex64>) -> f64 {
    w.iter().zip(y).map(|(w, y)| (w.conj() * y).re).sum()
}

/// Différences finies centrées le long de directions pseudo-aléatoires, comparées au produit scalaire du gradient avec
/// la direction : écart relatif le plus grand des quatre (b0 pour L_y, x réel et imaginaire pour L_y, b0 pour L_z).
fn ecart_differences_finies(cas: &Cas) -> f64 {
    let modele = ModeleCpr::new(cas.x.dim().into(), cas.temps.as_slice().unwrap());
    let dims = cas.x.dim();
    let mut etat = 7u64;
    let mut alea = move || {
        etat = etat.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((etat >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let d: Array3<f64> = Array3::from_shape_simple_fn(dims, &mut alea);
    let (gx, gb0) = modele.gradient_direct(&cas.x, &cas.b0, &cas.w_y);
    let (_, atk) = modele.adjoint_et_pondere(&cas.k, &cas.b0);
    let gz = modele.gradient_adjoint_b0(&atk, &cas.w_z);
    let l_y = |x: &Array3<Complex64>, b0: &Array3<f64>| re_produit(&cas.w_y, &modele.direct(x, b0));
    let l_z = |b0: &Array3<f64>| re_produit(&cas.w_z, &modele.adjoint(&cas.k, b0));
    let h = 1e-2; // Hz : phase 2π · t · h ≈ 1e-4 rad
    let (bp, bm) = (&cas.b0 + &(&d * h), &cas.b0 - &(&d * h));
    let fd_y_b0 = (l_y(&cas.x, &bp) - l_y(&cas.x, &bm)) / (2.0 * h);
    let fd_z_b0 = (l_z(&bp) - l_z(&bm)) / (2.0 * h);
    let scal = |g: &Array3<f64>| Zip::from(g).and(&d).fold(0.0, |s, g, d| s + g * d);
    let mut ecarts = vec![
        ((fd_y_b0 - scal(&gb0)) / scal(&gb0)).abs(),
        ((fd_z_b0 - scal(&gz)) / scal(&gz)).abs(),
    ];
    // L_y est linéaire en x : la différence finie est exacte à l'arrondi près.
    for unite in [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)] {
        let dx = d.mapv(|v| unite * v);
        let hx = 1e-3;
        let fd = (l_y(&(&cas.x + &(&dx * hx)), &cas.b0) - l_y(&(&cas.x - &(&dx * hx)), &cas.b0)) / (2.0 * hx);
        let s: f64 = gx.iter().zip(&dx).map(|(g, dx)| (g.conj() * dx).re).sum();
        ecarts.push(((fd - s) / s).abs());
    }
    println!("  différences finies : {ecarts:?}");
    ecarts.into_iter().fold(0.0, f64::max)
}

#[test]
fn gradients_comme_pytorch_cas_synthetique() {
    let cas = lire(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/cpr_gradients_synthetique.h5"));
    let (e64, e32) = (ecarts_pytorch::<f64>(&cas), ecarts_pytorch::<f32>(&cas));
    println!("cas synthétique : f64 {e64:?}, f32 {e32:?}");
    assert!(e64.iter().all(|&e| e <= 1e-10), "f64 : {e64:?}");
    assert!(e32.iter().all(|&e| e <= 1e-5), "f32 : {e32:?}");
}

#[test]
fn gradients_comme_differences_finies() {
    let cas = lire(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/cpr_gradients_synthetique.h5"));
    let e = ecart_differences_finies(&cas);
    assert!(e <= 1e-6, "différences finies : {e:e}");
}

#[test]
fn gradients_comme_pytorch_plans_p1() {
    let Ok(dossier) = std::env::var("MEDOXIDE_ULF_P1_GRAD") else {
        eprintln!("MEDOXIDE_ULF_P1_GRAD absent : test sur les plans de P1 sauté");
        return;
    };
    let mut fichiers: Vec<_> = std::fs::read_dir(&dossier)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "h5"))
        .collect();
    fichiers.sort();
    assert!(!fichiers.is_empty(), "{dossier} : aucun fichier .h5");
    for chemin in fichiers {
        let cas = lire(&chemin);
        let (e64, e32) = (ecarts_pytorch::<f64>(&cas), ecarts_pytorch::<f32>(&cas));
        let fd = ecart_differences_finies(&cas);
        println!("{} : f64 {e64:?}, f32 {e32:?}", chemin.display());
        assert!(e64.iter().all(|&e| e <= 1e-8), "{} f64 : {e64:?}", chemin.display());
        assert!(e32.iter().all(|&e| e <= 1e-4), "{} f32 : {e32:?}", chemin.display());
        assert!(fd <= 1e-6, "{} différences finies : {fd:e}", chemin.display());
    }
}
