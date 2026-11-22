//! Estimation jointe B0 + image de bout en bout sur un plan (critère 5 de P4 bis, docs/ULF.md) : démarrage à chaud
//! (300 pas par réseau) puis 200 pas joints, réseaux initialisés par Burn.
//!
//!     cargo run --release -p medoxide-ulf --example jointe_p1 [--features cuda] -- \
//!         <entrées.h5> <sortie.h5> [--dispositif cpu|wgpu|cuda] [--f64] [--graine n]
//!
//! Entrées (HDF5, écrites par `scripts/cas_jointe.py entrees`) : `k_reg`, `k_inv` (complexes bruts), `temps`,
//! `espacement_mm`, `b0_init` (champ de départ), `bornes_b0`. Sortie : `b0_hz`, `u` (complexe, à l'échelle des
//! données), attributs `secondes`, `dispositif`, `precision`, `graine`.
//!
//! Le modèle CPR tourne sur CPU dans la précision choisie ; les réseaux et les pertes sur le dispositif choisi.

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::DType;
use medoxide_ulf::inr::{ChampB0, ConfigGrille, Encodage, ImageInr, InrHachage};
use medoxide_ulf::jointe::{estimer, Reglages};
use medoxide_ulf::pertes::Donnees;
use medoxide_ulf::Reel;
use ndarray::{Array3, Ix1, Ix3, Zip};
use rustfft::num_complex::Complex64;

struct Options {
    entree: String,
    sortie: String,
    dispositif: String,
    f64: bool,
    graine: u64,
}

fn options() -> Options {
    let mut args = std::env::args().skip(1);
    let mut o = Options {
        entree: args.next().expect("entrées.h5"),
        sortie: args.next().expect("sortie.h5"),
        dispositif: "cpu".into(),
        f64: false,
        graine: 0,
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dispositif" => o.dispositif = args.next().expect("cpu, wgpu ou cuda"),
            "--f64" => o.f64 = true,
            "--graine" => o.graine = args.next().expect("graine").parse().expect("entier"),
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

fn lancer<T: Reel>(o: &Options, dtype: DType) {
    let f = hdf5::File::open(&o.entree).unwrap();
    let c = |nom: &str| f.dataset(nom).unwrap().read::<Complex64, Ix3>().unwrap();
    let (k_reg, k_inv) = (c("k_reg"), c("k_inv"));
    let temps = f.dataset("temps").unwrap().read::<f64, Ix1>().unwrap();
    let e = f.dataset("espacement_mm").unwrap().read_raw::<f64>().unwrap();
    let b0_init: Array3<f64> = f.dataset("b0_init").unwrap().read::<f64, Ix3>().unwrap();
    let bornes = f.dataset("bornes_b0").unwrap().read_raw::<f64>().unwrap();
    let device = dispositif(&o.dispositif);
    device.seed(o.graine);

    let t0 = Instant::now();
    let d = Donnees::<T>::nouveau(&k_reg, &k_inv, temps.as_slice().unwrap(), [e[0], e[1], e[2]], &device);
    let dims = d.dims;
    let (cfg_champ, cfg_image) = (ConfigGrille::champ_p1(), ConfigGrille::image_p1());
    let enc_champ = Encodage::nouveau(&cfg_champ, dims, &device, dtype);
    let enc_image = Encodage::nouveau(&cfg_image, dims, &device, dtype);
    let champ = ChampB0::nouveau(
        InrHachage::nouveau(&cfg_champ, 1, 64, 2, &device, dtype),
        bornes[0],
        bornes[1],
    );
    let image = ImageInr {
        reseau: InrHachage::nouveau(&cfg_image, 2, 64, 2, &device, dtype),
    };
    let reglages = Reglages::default();
    let mut debut = Instant::now();
    let r = estimer(
        &d,
        champ,
        image,
        &enc_champ,
        &enc_image,
        &b0_init,
        &reglages,
        |it, b| {
            if it % 20 == 0 || it + 1 == reglages.n_iter {
                println!(
                "pas joint {it:>3} : perte {:+.4e} (fidélité {:.3e}, image {:+.4}, b0 {:+.4}, lissage {:.3e}, support {:.3e}) {:.2} s/pas",
                b.totale,
                b.fidelite,
                b.image,
                b.b0,
                b.lissage,
                b.support,
                debut.elapsed().as_secs_f64() / if it == 0 { 1.0 } else { 20.0 }
            );
                debut = Instant::now();
            }
        },
    );
    let secondes = t0.elapsed().as_secs_f64();
    println!("{} : {secondes:.1} s", o.entree);

    let s = hdf5::File::create(&o.sortie).unwrap();
    s.new_dataset_builder().with_data(&r.b0).create("b0_hz").unwrap();
    let u = Zip::from(&r.u_re)
        .and(&r.u_im)
        .map_collect(|a, b| Complex64::new(*a, *b));
    s.new_dataset_builder().with_data(&u).create("u").unwrap();
    let attr = |nom: &str, v: f64| s.new_attr::<f64>().create(nom).unwrap().write_scalar(&v).unwrap();
    attr("secondes", secondes);
    attr("graine", o.graine as f64);
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
