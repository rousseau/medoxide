//! Critère 3 de P4 bis (docs/ULF.md) : pertes de l'estimation jointe et leurs gradients par rapport à (u, b0), à travers
//! la coupure du graphe, comparés à PyTorch (fmi).
//!
//! Cas synthétique versionné (`tests/donnees/pertes_synthetique.h5`, écrit par `scripts/cas_pertes.py synthetique`) ;
//! sorties de P1 si `MEDOXIDE_ULF_P1_PERTES` désigne un dossier de fichiers écrits par `scripts/cas_pertes.py p1` (hors
//! dépôt). Seuils : valeurs à ≤ 1e-10 (`f64`) et 1e-5 (`f32`), gradients à ≤ 1e-8 et 1e-4 (écarts relatifs), masque de
//! support identique voxel pour voxel, facteur de normalisation à 1e-12. Amendement du 2026-11-22 (docs/ULF.md, même
//! règle qu'au critère 1) : en `f32`, une quantité passe aussi si elle reste sous deux fois l'écart de PyTorch float32
//! au float64 sur le même cas (`ecart_pytorch_f32/…`) ; le gradient du terme image dépasse 1e-4 en PyTorch f32 aussi.

use std::path::Path;

use burn::prelude::*;
use medoxide_ulf::pertes::{depuis_tenseur, evaluer, vers_tenseur, Donnees, Termes};
use medoxide_ulf::Reel;
use ndarray::{Array1, Array3, Ix1, Ix3};
use rustfft::num_complex::{Complex, Complex64};

const TERMES: [&str; 5] = ["fidelity", "image", "b0", "smooth", "support"];

struct Cas {
    k_reg: Array3<Complex64>,
    k_inv: Array3<Complex64>,
    temps: Array1<f64>,
    espacement: [f64; 3],
    u: Array3<Complex64>,
    b0: Array3<f64>,
    echelle: f64,
    masque: Array3<u8>,
    /// Par terme : valeur, ∂/∂u, ∂/∂b0.
    references: Vec<(f64, Array3<Complex64>, Array3<f64>)>,
    /// Par terme : écarts de PyTorch float32 (valeur, ∂/∂u, ∂/∂b0).
    planchers: Vec<[f64; 3]>,
}

fn lire(chemin: &Path) -> Cas {
    let f = hdf5::File::open(chemin).unwrap();
    let c = |nom: &str| f.dataset(nom).unwrap().read::<Complex64, Ix3>().unwrap();
    let r = |nom: &str| f.dataset(nom).unwrap().read::<f64, Ix3>().unwrap();
    let s = |nom: &str| f.dataset(nom).unwrap().read_scalar::<f64>().unwrap();
    let e = f.dataset("espacement_mm").unwrap().read_raw::<f64>().unwrap();
    Cas {
        k_reg: c("k_reg"),
        k_inv: c("k_inv"),
        temps: f.dataset("temps").unwrap().read::<f64, Ix1>().unwrap(),
        espacement: [e[0], e[1], e[2]],
        u: c("u"),
        b0: r("b0"),
        echelle: s("echelle"),
        masque: f.dataset("masque_support").unwrap().read::<u8, Ix3>().unwrap(),
        references: TERMES
            .iter()
            .map(|t| {
                (
                    s(&format!("terme_{t}/valeur")),
                    c(&format!("terme_{t}/grad_u")),
                    r(&format!("terme_{t}/grad_b0")),
                )
            })
            .collect(),
        planchers: TERMES
            .iter()
            .map(|t| ["valeur", "grad_u", "grad_b0"].map(|q| s(&format!("ecart_pytorch_f32/terme_{t}/{q}"))))
            .collect(),
    }
}

fn terme(t: &Termes, nom: &str) -> Tensor<1> {
    match nom {
        "fidelity" => t.fidelite.clone(),
        "image" => t.image.clone(),
        "b0" => t.b0.clone(),
        "smooth" => t.lissage.clone(),
        "support" => t.support.clone(),
        _ => unreachable!(),
    }
}

/// Écart relatif L2, ou norme absolue si la référence est nulle (gradient qui ne dépend pas de cette variable).
fn ecart(a: impl Iterator<Item = f64> + Clone, r: impl Iterator<Item = f64> + Clone) -> f64 {
    let num: f64 = a
        .clone()
        .zip(r.clone())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    let den: f64 = r.map(|y| y * y).sum::<f64>().sqrt();
    if den == 0.0 {
        num
    } else {
        num / den
    }
}

fn plat_c<T: Reel>(a: &Array3<Complex<T>>) -> impl Iterator<Item = f64> + Clone + '_ {
    a.iter().flat_map(|z| [z.re.to_f64().unwrap(), z.im.to_f64().unwrap()])
}

/// Écarts (valeur, ∂/∂u, ∂/∂b0) de chaque terme dans la précision T ; vérifie le masque et le facteur d'échelle.
fn ecarts<T: Reel>(cas: &Cas) -> Vec<[f64; 3]> {
    let device = Device::flex().autodiff();
    let d = Donnees::<T>::nouveau(
        &cas.k_reg,
        &cas.k_inv,
        cas.temps.as_slice().unwrap(),
        cas.espacement,
        &device,
    );
    assert!(
        (d.echelle - cas.echelle).abs() <= 1e-12 * cas.echelle,
        "échelle {} contre {}",
        d.echelle,
        cas.echelle
    );
    let differents = d
        .masque
        .iter()
        .zip(&cas.masque)
        .filter(|(a, b)| **a != (**b != 0))
        .count();
    assert_eq!(differents, 0, "masque de support : {differents} voxels différents");
    let dtype = if std::mem::size_of::<T>() == 8 {
        burn::tensor::DType::F64
    } else {
        burn::tensor::DType::F32
    };
    let vers = |a: Array3<f64>| vers_tenseur(&a, &device, dtype).require_grad();
    TERMES
        .iter()
        .zip(&cas.references)
        .map(|(nom, (valeur, ref_gu, ref_gb0))| {
            let (u_re, u_im, b0) = (
                vers(cas.u.mapv(|z| z.re)),
                vers(cas.u.mapv(|z| z.im)),
                vers(cas.b0.clone()),
            );
            let (termes, coupure) = evaluer(&d, &u_re, &u_im, &b0);
            let t = terme(&termes, nom);
            let v = t
                .clone()
                .try_into_data_dtype(burn::tensor::DType::F64)
                .unwrap()
                .try_to_vec::<f64>()
                .unwrap()[0];
            let grads = t.backward();
            let (gu_op, gb0_op) = coupure.gradients(&d, &grads);
            let (gu_op_re, gu_op_im) = (depuis_tenseur::<T>(gu_op.re), depuis_tenseur::<T>(gu_op.im));
            let gb0_op = depuis_tenseur::<T>(gb0_op);
            let direct = |x: &Tensor<3>| {
                x.grad(&grads)
                    .map_or_else(|| Array3::<T>::zeros(d.dims), depuis_tenseur::<T>)
            };
            let (gre, gim) = (direct(&u_re), direct(&u_im));
            let gu = Array3::from_shape_fn(d.dims, |i| Complex::new(gu_op_re[i] + gre[i], gu_op_im[i] + gim[i]));
            let gb0 = &direct(&b0) + &gb0_op;
            let e = [
                ((v - valeur) / valeur).abs(),
                ecart(plat_c(&gu), ref_gu.iter().flat_map(|z| [z.re, z.im])),
                ecart(gb0.iter().map(|x| x.to_f64().unwrap()), ref_gb0.iter().copied()),
            ];
            println!(
                "  {nom:<9} valeur {v:+.6e} (PyTorch {valeur:+.6e})  écarts {:.2e} {:.2e} {:.2e}",
                e[0], e[1], e[2]
            );
            e
        })
        .collect()
}

fn verifier(chemin: &Path) {
    let cas = lire(chemin);
    println!("{} — f64", chemin.display());
    let e64 = ecarts::<f64>(&cas);
    println!("{} — f32", chemin.display());
    let e32 = ecarts::<f32>(&cas);
    let mut echecs = Vec::new();
    for ((nom, (a, b)), p) in TERMES.iter().zip(e64.iter().zip(&e32)).zip(&cas.planchers) {
        for (q, (seuil64, seuil32)) in [(1e-10, 1e-5), (1e-8, 1e-4), (1e-8, 1e-4)].into_iter().enumerate() {
            if a[q] > seuil64 {
                echecs.push(format!("{nom} f64 [{q}] : {:.2e} > {seuil64:.0e}", a[q]));
            }
            let limite = f64::max(seuil32, 2.0 * p[q]);
            if b[q] > limite {
                echecs.push(format!(
                    "{nom} f32 [{q}] : {:.2e} > {limite:.2e} (PyTorch f32 {:.2e})",
                    b[q], p[q]
                ));
            }
        }
        println!(
            "  {nom:<9} f32 / PyTorch f32 : {:.2e}/{:.2e} {:.2e}/{:.2e} {:.2e}/{:.2e}",
            b[0], p[0], b[1], p[1], b[2], p[2]
        );
    }
    assert!(
        echecs.is_empty(),
        "{} hors seuil :\n{}",
        chemin.display(),
        echecs.join("\n")
    );
}

#[test]
fn pertes_comme_pytorch_cas_synthetique() {
    verifier(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/pertes_synthetique.h5"));
}

#[test]
fn pertes_comme_pytorch_plans_p1() {
    let Ok(dossier) = std::env::var("MEDOXIDE_ULF_P1_PERTES") else {
        eprintln!("MEDOXIDE_ULF_P1_PERTES absent : test sur les sorties de P1 sauté");
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
        verifier(&chemin);
    }
}
