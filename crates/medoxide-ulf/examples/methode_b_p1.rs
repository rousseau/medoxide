//! Méthode B (champ INR seul, `compare_methods.estimate_inr`) de bout en bout sur un plan : critère 3 de P5
//! (docs/ULF.md). 400 pas, champ initialisé par Burn.
//!
//!     cargo run --release -p medoxide-ulf --example methode_b_p1 [--features cuda] -- \
//!         <entrées.h5> <sortie.h5> [--dispositif cpu|wgpu|cuda] [--f64] [--graine n] [--poids <cas.h5>]
//!
//! Entrées : celles de `jointe_p1` (`scripts/cas_jointe.py entrees` ; `b0_init` n'est pas lu). Sortie : `b0_hz`,
//! attributs `secondes`, `graine`, `dispositif`, `precision`, `poids_fmi`. Elle peut servir de `--b0-init` à
//! `jointe_p1`. `--poids` : champ initial lu dans `etat_0/champ` d'un fichier de `scripts/cas_jointe.py methode_b`
//! (poids de la graine 0 de fmi, critère 3a amendé) au lieu d'être tiré par Burn.

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::DType;
use medoxide_ulf::inr::{ChampB0, ConfigGrille, Encodage, InrHachage};
use medoxide_ulf::methode_b::{estimer_b, ReglagesB};
use medoxide_ulf::pertes::Donnees;
use medoxide_ulf::Reel;
use ndarray::{Ix1, Ix3};
use rustfft::num_complex::Complex64;

struct Options {
    entree: String,
    sortie: String,
    dispositif: String,
    f64: bool,
    graine: u64,
    poids: Option<String>,
}

fn options() -> Options {
    let mut args = std::env::args().skip(1);
    let mut o = Options {
        entree: args.next().expect("entrées.h5"),
        sortie: args.next().expect("sortie.h5"),
        dispositif: "cpu".into(),
        f64: false,
        graine: 0,
        poids: None,
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dispositif" => o.dispositif = args.next().expect("cpu, wgpu ou cuda"),
            "--f64" => o.f64 = true,
            "--graine" => o.graine = args.next().expect("graine").parse().expect("entier"),
            "--poids" => o.poids = Some(args.next().expect("cas.h5")),
            _ => panic!("option inconnue : {a}"),
        }
    }
    o
}

fn dispositif(nom: &str) -> Device {
    match nom {
        "cpu" => Device::flex().autodiff(),
        "wgpu" => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice).autodiff(),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0).autodiff(),
        _ => panic!("dispositif {nom} indisponible (cuda : compiler avec --features cuda)"),
    }
}

/// Champ au format de `cas_jointe.py` (tables, couches au format PyTorch, scalaires s et o, nuls à l'état initial).
fn lire_champ(g: &hdf5::Group, device: &Device, dtype: DType) -> InrHachage {
    let lire = |nom: &str| {
        let d = g.dataset(nom).unwrap();
        (d.read_raw::<f64>().unwrap(), d.shape())
    };
    let compter = |prefixe: &str| {
        g.member_names()
            .unwrap()
            .iter()
            .filter(|n| n.starts_with(prefixe))
            .count()
    };
    for nom in ["s", "o"] {
        assert_eq!(lire(nom).0, vec![0.0], "{nom} initial non nul");
    }
    let tables = (0..compter("niveau_"))
        .map(|l| {
            let (v, f) = lire(&format!("niveau_{l}/table"));
            (v, [f[0], f[1]])
        })
        .collect();
    let couches = (0..compter("couche_"))
        .map(|c| {
            let (w, f) = lire(&format!("couche_{c}/poids"));
            (w, [f[0], f[1]], lire(&format!("couche_{c}/biais")).0)
        })
        .collect();
    InrHachage::depuis_poids(tables, couches, device, dtype)
}

fn lancer<T: Reel>(o: &Options, dtype: DType) {
    let f = hdf5::File::open(&o.entree).unwrap();
    let c = |nom: &str| f.dataset(nom).unwrap().read::<Complex64, Ix3>().unwrap();
    let temps = f.dataset("temps").unwrap().read::<f64, Ix1>().unwrap();
    let e = f.dataset("espacement_mm").unwrap().read_raw::<f64>().unwrap();
    let bornes = f.dataset("bornes_b0").unwrap().read_raw::<f64>().unwrap();
    let device = dispositif(&o.dispositif);
    device.seed(o.graine);

    let t0 = Instant::now();
    let d = Donnees::<T>::nouveau(
        &c("k_reg"),
        &c("k_inv"),
        temps.as_slice().unwrap(),
        [e[0], e[1], e[2]],
        &device,
    );
    let cfg = ConfigGrille::champ_b();
    let enc = Encodage::nouveau(&cfg, d.dims, &device, dtype);
    let reseau = match &o.poids {
        Some(cas) => {
            let f = hdf5::File::open(cas).unwrap();
            let g = f.group("champ_config").unwrap();
            let facteur = g.attr("per_level_scale").unwrap().read_scalar::<f64>().unwrap();
            assert!(
                (facteur - cfg.facteur).abs() < 1e-12,
                "{cas} : grille autre que champ_b()"
            );
            lire_champ(&f.group("etat_0/champ").unwrap(), &device, dtype)
        }
        None => InrHachage::nouveau(&cfg, 1, 64, 2, &device, dtype),
    };
    let champ = ChampB0::nouveau(reseau, bornes[0], bornes[1]);
    let reglages = ReglagesB::default();
    let mut debut = Instant::now();
    let (b0, _) = estimer_b(&d, champ, &enc, &reglages, |it, b| {
        if it % 50 == 0 || it + 1 == reglages.n_iter {
            println!(
                "pas {it:>3} : perte {:+.5e} (b0 {:+.5}, lissage {:.3e}) {:.3} s/pas",
                b.totale,
                b.b0,
                b.lissage,
                debut.elapsed().as_secs_f64() / if it == 0 { 1.0 } else { 50.0 }
            );
            debut = Instant::now();
        }
    });
    let secondes = t0.elapsed().as_secs_f64();
    println!("{} : {secondes:.1} s", o.entree);

    let s = hdf5::File::create(&o.sortie).unwrap();
    s.new_dataset_builder().with_data(&b0).create("b0_hz").unwrap();
    let attr = |nom: &str, v: f64| s.new_attr::<f64>().create(nom).unwrap().write_scalar(&v).unwrap();
    attr("secondes", secondes);
    attr("graine", o.graine as f64);
    attr("poids_fmi", if o.poids.is_some() { 1.0 } else { 0.0 });
    let texte = |nom: &str, v: &str| {
        let v: hdf5::types::VarLenUnicode = v.parse().unwrap();
        s.new_attr::<hdf5::types::VarLenUnicode>()
            .create(nom)
            .unwrap()
            .write_scalar(&v)
            .unwrap()
    };
    texte("dispositif", &o.dispositif);
    texte("precision", if o.f64 { "f64" } else { "f32" });
}

fn main() {
    let o = options();
    if o.f64 {
        lancer::<f64>(&o, DType::F64)
    } else {
        lancer::<f32>(&o, DType::F32)
    }
}
