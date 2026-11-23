//! Critère 2 de P5 (docs/ULF.md) : méthode B (`compare_methods.estimate_inr`), perte, gradient et pas d'optimisation
//! comparés à PyTorch (fmi), à partir des mêmes poids initiaux, en `f64`.
//!
//! Cas synthétique versionné (`tests/donnees/methode_b_synthetique.h5`, écrit par `scripts/cas_jointe.py methode_b
//! synthetique`) ; plans de P1 si `MEDOXIDE_ULF_P1_B` désigne un dossier de fichiers écrits par `scripts/cas_jointe.py
//! methode_b <inputs.npz>` (hors dépôt).
//!
//! Seuils : à l'état initial, terme b0 et lissage à ≤ 1e-10 (écart relatif), gradient de b0 + W_SMOOTH · lissage par
//! rapport à b0 à ≤ 1e-8 (écart relatif L2) ; après chacun des deux premiers pas, chaque paramètre à ≤ 1e-10 de
//! PyTorch, ou sous deux fois l'écart de PyTorch à lui-même entre un thread et le nombre de threads par défaut
//! (`plancher_pytorch/…`), comme l'amendement du critère 4 de P4 bis. Les termes de la perte avant le second pas sont
//! aussi vérifiés (≤ 1e-10).

use std::path::Path;

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use medoxide_ulf::inr::{ChampB0, ConfigGrille, Encodage, InrHachage};
use medoxide_ulf::jointe::Adam;
use medoxide_ulf::methode_b::{pas_b, perte_et_gradient, ReglagesB};
use medoxide_ulf::pertes::Donnees;
use ndarray::{Ix1, Ix3};
use rustfft::num_complex::Complex64;

const SEUIL: f64 = 1e-10;

fn lire(g: &hdf5::Group, nom: &str) -> (Vec<f64>, Vec<usize>) {
    let d = g.dataset(nom).unwrap();
    (d.read_raw::<f64>().unwrap(), d.shape())
}

fn n_membres(g: &hdf5::Group, prefixe: &str) -> usize {
    g.member_names()
        .unwrap()
        .iter()
        .filter(|n| n.starts_with(prefixe))
        .count()
}

fn config(g: &hdf5::Group) -> ConfigGrille {
    let f = |n: &str| g.attr(n).unwrap().read_scalar::<f64>().unwrap();
    let i = |n: &str| g.attr(n).unwrap().read_scalar::<i64>().unwrap();
    ConfigGrille {
        n_niveaux: i("n_levels") as usize,
        n_caracteristiques: i("n_features_per_level") as usize,
        log2_table: i("log2_hashmap_size") as u32,
        resolution_base: f("base_resolution"),
        facteur: f("per_level_scale"),
    }
}

/// Noms des paramètres du champ, dans l'ordre de [`parametres_rust`].
fn noms(g: &hdf5::Group) -> Vec<String> {
    let mut n: Vec<String> = (0..n_membres(g, "niveau_"))
        .map(|l| format!("niveau_{l}/table"))
        .collect();
    for c in 0..n_membres(g, "couche_") {
        n.push(format!("couche_{c}/poids"));
        n.push(format!("couche_{c}/biais"));
    }
    n.push("s".into());
    n.push("o".into());
    n
}

fn champ_h5(g: &hdf5::Group, bornes: &[f64], device: &Device) -> ChampB0 {
    let tables = (0..n_membres(g, "niveau_"))
        .map(|l| {
            let (v, f) = lire(g, &format!("niveau_{l}/table"));
            (v, [f[0], f[1]])
        })
        .collect();
    let couches = (0..n_membres(g, "couche_"))
        .map(|c| {
            let (w, f) = lire(g, &format!("couche_{c}/poids"));
            (w, [f[0], f[1]], lire(g, &format!("couche_{c}/biais")).0)
        })
        .collect();
    let mut champ = ChampB0::nouveau(
        InrHachage::depuis_poids(tables, couches, device, DType::F64),
        bornes[0],
        bornes[1],
    );
    let scalaire = |nom: &str| {
        let v = g.dataset(nom).unwrap().read_scalar::<f64>().unwrap();
        Param::from_tensor(Tensor::from_data(TensorData::new(vec![v], [1]), (device, DType::F64)))
    };
    champ.s = scalaire("s");
    champ.o = scalaire("o");
    champ
}

fn en_f64<const D: usize>(t: Tensor<D>) -> Vec<f64> {
    t.try_into_data_dtype(DType::F64).unwrap().try_to_vec::<f64>().unwrap()
}

fn parametres_rust(c: &ChampB0) -> Vec<Vec<f64>> {
    let mut p: Vec<Vec<f64>> = c.reseau.tables.iter().map(|t| en_f64(t.val())).collect();
    for couche in &c.reseau.couches {
        p.push(en_f64(couche.weight.val().transpose()));
        p.push(en_f64(couche.bias.as_ref().unwrap().val()));
    }
    p.push(en_f64(c.s.val()));
    p.push(en_f64(c.o.val()));
    p
}

fn ecart(a: &[f64], r: &[f64]) -> f64 {
    assert_eq!(a.len(), r.len());
    let num: f64 = a.iter().zip(r).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = r.iter().map(|y| y * y).sum();
    (num / den).sqrt()
}

fn noter(echecs: &mut Vec<String>, nom: String, e: f64, seuil: f64) {
    println!("  {nom:<40} {e:.2e}");
    if e > seuil || e.is_nan() {
        echecs.push(format!("{nom} : {e:.2e} > {seuil:.2e}"));
    }
}

fn verifier(chemin: &Path) -> Vec<String> {
    let nom_cas = chemin.file_stem().unwrap().to_string_lossy().to_string();
    println!("{nom_cas}");
    let f = hdf5::File::open(chemin).unwrap();
    let c = |nom: &str| f.dataset(nom).unwrap().read::<Complex64, Ix3>().unwrap();
    let temps = f.dataset("temps").unwrap().read::<f64, Ix1>().unwrap();
    let e = lire(&f, "espacement_mm").0;
    let bornes = lire(&f, "bornes_b0").0;
    let device = Device::flex().autodiff();
    let d = Donnees::<f64>::nouveau(
        &c("k_reg"),
        &c("k_inv"),
        temps.as_slice().unwrap(),
        [e[0], e[1], e[2]],
        &device,
    );
    let cfg = config(&f.group("champ_config").unwrap());
    let enc = Encodage::nouveau(&cfg, d.dims, &device, DType::F64);
    let attr = |n: &str| f.attr(n).unwrap().read_scalar::<f64>().unwrap();
    let reglages = ReglagesB {
        n_iter: f.attr("n_iter").unwrap().read_scalar::<i64>().unwrap() as usize,
        poids_lissage: attr("w_smooth"),
        poids_scalaires: attr("w_scalaires"),
        ..ReglagesB::default()
    };
    assert_eq!(reglages.n_iter, ReglagesB::default().n_iter);
    let mut echecs = Vec::new();
    let mut noter = |nom: String, e: f64, s: f64| noter(&mut echecs, format!("{nom_cas} {nom}"), e, s);

    // état initial : b0, termes, gradient
    let g0 = f.group("etat_0/champ").unwrap();
    let mut champ = champ_h5(&g0, &bornes, &device);
    let b0 = champ.b0(&enc).reshape(d.dims);
    // contrôle, hors critère : à l'initialisation de fmi la sortie du réseau est presque constante et son centrage perd
    // des chiffres (P4 bis, critère 4) ; le terme, le gradient et les pas, eux, sont tenus aux seuils du critère.
    noter(
        "b0 initial (contrôle)".into(),
        ecart(&en_f64(b0.clone()), &lire(&f, "b0_etat_0").0),
        1e-8,
    );
    let (t_b0, lissage, grad) = perte_et_gradient(&d, &b0, &reglages);
    let pertes = |n: &str| lire(&f, &format!("pertes_b/{n}")).0;
    noter(
        "terme b0 (état 0)".into(),
        ((t_b0 - pertes("b0")[0]) / pertes("b0")[0]).abs(),
        SEUIL,
    );
    noter(
        "lissage (état 0)".into(),
        ((lissage - pertes("smooth")[0]) / pertes("smooth")[0]).abs(),
        SEUIL,
    );
    noter(
        "gradient par rapport à b0".into(),
        ecart(&en_f64(grad), &lire(&f, "grad_b0").0),
        1e-8,
    );

    // pas
    let n_pas = f.attr("n_pas").unwrap().read_scalar::<i64>().unwrap() as usize;
    let mut adam = Adam::default();
    let noms = noms(&g0);
    for p in 0..n_pas {
        let (c2, bilan) = pas_b(&d, champ, &enc, &mut adam, &reglages, reglages.facteur_lr(p));
        champ = c2;
        noter(
            format!("perte totale avant le pas {}", p + 1),
            ((bilan.totale - pertes("totale")[p]) / pertes("totale")[p]).abs(),
            SEUIL,
        );
        let ge = f.group(&format!("etat_{}/champ", p + 1)).unwrap();
        for (nom, v) in noms.iter().zip(parametres_rust(&champ)) {
            let plancher = f
                .dataset(&format!("plancher_pytorch/etat_{}/champ/{nom}", p + 1))
                .map_or(0.0, |d| d.read_scalar::<f64>().unwrap());
            noter(
                format!("pas {} {nom}", p + 1),
                ecart(&v, &lire(&ge, nom).0),
                SEUIL.max(2.0 * plancher),
            );
        }
    }
    echecs
}

#[test]
fn methode_b_comme_pytorch_cas_synthetique() {
    let echecs = verifier(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/donnees/methode_b_synthetique.h5"));
    assert!(echecs.is_empty(), "hors seuil :\n{}", echecs.join("\n"));
}

#[test]
fn methode_b_comme_pytorch_plans_p1() {
    let Ok(dossier) = std::env::var("MEDOXIDE_ULF_P1_B") else {
        eprintln!("MEDOXIDE_ULF_P1_B non défini : plans de P1 sautés");
        return;
    };
    let mut fichiers: Vec<_> = std::fs::read_dir(dossier)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "h5"))
        .collect();
    fichiers.sort();
    assert!(!fichiers.is_empty());
    let echecs: Vec<String> = fichiers.iter().flat_map(|p| verifier(p)).collect();
    assert!(echecs.is_empty(), "hors seuil :\n{}", echecs.join("\n"));
}
