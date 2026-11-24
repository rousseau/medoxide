//! Critères 2 et 3 de P4 (docs/ULF.md) : modèle direct et adjoint comparés au code Python de fmi, en `f64` et en `f32`.
//!
//! Cas synthétique versionné (`tests/donnees/cpr_synthetique.h5`, écrit par `scripts/cas_cpr.py synthetique`) ; plans
//! de P1 si `MEDOXIDE_ULF_P1` désigne un dossier de fichiers écrits par `scripts/cas_cpr.py p1` (hors dépôt).

use std::path::Path;
use std::time::Instant;

use medoxide_ulf::{ModeleCpr, Reel};
use ndarray::{Array1, Array3, Ix1, Ix3};
use rustfft::num_complex::{Complex, Complex64};

struct Cas {
    x: Array3<Complex64>,
    k: Array3<Complex64>,
    b0: Array3<f64>,
    temps: Array1<f64>,
    y: Array3<Complex64>,
    z: Array3<Complex64>,
}

fn lire(chemin: &Path) -> Cas {
    let f = hdf5::File::open(chemin).unwrap();
    let c = |nom: &str| f.dataset(nom).unwrap().read::<Complex64, Ix3>().unwrap();
    Cas {
        x: c("x"),
        k: c("k"),
        y: c("y"),
        z: c("z"),
        b0: f.dataset("b0").unwrap().read::<f64, Ix3>().unwrap(),
        temps: f.dataset("temps").unwrap().read::<f64, Ix1>().unwrap(),
    }
}

/// Écart relatif L2 de `a` (précision T) à la référence `r` (complex128).
fn ecart<T: Reel>(a: &Array3<Complex<T>>, r: &Array3<Complex64>) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (x, y) in a.iter().zip(r) {
        let x = Complex64::new(x.re.to_f64().unwrap(), x.im.to_f64().unwrap());
        num += (x - y).norm_sqr();
        den += y.norm_sqr();
    }
    (num / den).sqrt()
}

/// Applique A et Aᴴ en précision T ; rend les écarts à la référence et la durée moyenne d'une application (s).
fn comparer<T: Reel>(cas: &Cas) -> (f64, f64, f64) {
    let vers = |a: &Array3<Complex64>| a.mapv(|z| Complex::new(T::from_f64(z.re).unwrap(), T::from_f64(z.im).unwrap()));
    let temps: Vec<T> = cas.temps.iter().map(|&t| T::from_f64(t).unwrap()).collect();
    let dims = [cas.x.shape()[0], cas.x.shape()[1], cas.x.shape()[2]];
    let modele = ModeleCpr::new(dims, &temps);
    let b0 = cas.b0.mapv(|v| T::from_f64(v).unwrap());
    let debut = Instant::now();
    let y = modele.direct(&vers(&cas.x), &b0);
    let z = modele.adjoint(&vers(&cas.k), &b0);
    let duree = debut.elapsed().as_secs_f64() / 2.0;
    (ecart(&y, &cas.y), ecart(&z, &cas.z), duree)
}

#[test]
fn cas_synthetique_comme_fmi() {
    let cas = lire(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/cpr_synthetique.h5"));
    let (ey, ez, _) = comparer::<f64>(&cas);
    assert!(ey < 1e-12 && ez < 1e-12, "f64 : A x à {ey:e}, Aᴴ k à {ez:e}");
    let (ey, ez, _) = comparer::<f32>(&cas);
    assert!(ey < 1e-5 && ez < 1e-5, "f32 : A x à {ey:e}, Aᴴ k à {ez:e}");
}

#[test]
fn plans_de_p1_comme_fmi() {
    let Ok(dossier) = std::env::var("MEDOXIDE_ULF_P1") else {
        eprintln!("MEDOXIDE_ULF_P1 absent : test sur les plans de P1 sauté");
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
        let (ey64, ez64, d64) = comparer::<f64>(&cas);
        let (ey32, ez32, d32) = comparer::<f32>(&cas);
        let nom = chemin.file_name().unwrap().to_string_lossy();
        eprintln!("{nom} : f64 A {ey64:.1e} Aᴴ {ez64:.1e} ({d64:.2} s) ; f32 A {ey32:.1e} Aᴴ {ez32:.1e} ({d32:.2} s)");
        assert!(ey64 < 1e-10 && ez64 < 1e-10, "{nom} : f64");
        assert!(ey32 < 1e-5 && ez32 < 1e-5, "{nom} : f32");
    }
}
