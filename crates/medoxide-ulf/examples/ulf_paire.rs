//! Étape 1 de bout en bout depuis une paire de fichiers ISMRMRD (P6, docs/ULF.md) : lecture, entrées (fenêtre autour de
//! l'écho, temps, bornes, espacement, masque), méthode B (champ de départ), puis estimation jointe B0 + image.
//!
//!     cargo run --release -p medoxide-ulf --example ulf_paire [--features cuda] -- \
//!         <reg.h5> <inv.h5> <sortie.h5> [--dispositif cpu|wgpu|cuda] [--f64] [--graine n] [--ecrire-entrees]
//!
//! `reg.h5` : lecture normale ; `inv.h5` : lecture inversée (format fmi). Sortie : `b0_hz`, `u` (complexe, à l'échelle
//! des données), `b0_methode_b`, `masque` (uint8) ; attributs `secondes`, `secondes_methode_b`, `secondes_jointe`,
//! `graine`, `dispositif`, `precision`, `plan`, `echo_reg`, `echo_inv`, `demi_largeur`. `--ecrire-entrees` ajoute les
//! entrées de la méthode (`k_reg`, `k_inv`, `temps`, `espacement_mm`, `bornes_b0`), pour le critère 4 de P6.

use std::path::Path;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::DType;
use medoxide_ulf::entrees::depuis_fichiers;
use medoxide_ulf::inr::{ChampB0, ConfigGrille, Encodage, ImageInr, InrHachage};
use medoxide_ulf::jointe::{estimer, Reglages};
use medoxide_ulf::methode_b::{estimer_b, ReglagesB};
use medoxide_ulf::pertes::Donnees;
use medoxide_ulf::Reel;
use ndarray::Zip;
use rustfft::num_complex::Complex64;

struct Options {
    reg: String,
    inv: String,
    sortie: String,
    dispositif: String,
    f64: bool,
    graine: u64,
    ecrire_entrees: bool,
}

fn options() -> Options {
    let mut args = std::env::args().skip(1);
    let mut o = Options {
        reg: args.next().expect("reg.h5"),
        inv: args.next().expect("inv.h5"),
        sortie: args.next().expect("sortie.h5"),
        dispositif: "cpu".into(),
        f64: false,
        graine: 0,
        ecrire_entrees: false,
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dispositif" => o.dispositif = args.next().expect("cpu, wgpu ou cuda"),
            "--f64" => o.f64 = true,
            "--graine" => o.graine = args.next().expect("graine").parse().expect("entier"),
            "--ecrire-entrees" => o.ecrire_entrees = true,
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
    let t0 = Instant::now();
    let e = depuis_fichiers(Path::new(&o.reg), Path::new(&o.inv)).unwrap_or_else(|err| panic!("{err}"));
    println!(
        "plan {:?} : écho reg {:.3}, inv {:.3}, demi-largeur {:.3} ; bornes {:.2} – {:.1} Hz ; espacement {:?} mm",
        e.plan, e.echo_reg, e.echo_inv, e.demi_largeur, e.bornes_b0[0], e.bornes_b0[1], e.espacement_mm
    );
    let device = dispositif(&o.dispositif);
    device.seed(o.graine);
    let d = Donnees::<T>::nouveau(&e.k_reg, &e.k_inv, &e.temps, e.espacement_mm, &device);
    let dims = d.dims;

    // méthode B : champ de départ
    let t_b = Instant::now();
    let cfg_b = ConfigGrille::champ_b();
    let enc_b = Encodage::nouveau(&cfg_b, dims, &device, dtype);
    let champ_b = ChampB0::nouveau(
        InrHachage::nouveau(&cfg_b, 1, 64, 2, &device, dtype),
        e.bornes_b0[0],
        e.bornes_b0[1],
    );
    let reglages_b = ReglagesB::default();
    let (b0_init, _) = estimer_b(&d, champ_b, &enc_b, &reglages_b, |it, b| {
        if it % 100 == 0 || it + 1 == reglages_b.n_iter {
            println!(
                "méthode B, pas {it:>3} : perte {:+.5e} (b0 {:+.5}, lissage {:.3e})",
                b.totale, b.b0, b.lissage
            );
        }
    });
    let secondes_b = t_b.elapsed().as_secs_f64();

    // estimation jointe
    let t_j = Instant::now();
    let (cfg_champ, cfg_image) = (ConfigGrille::champ_p1(), ConfigGrille::image_p1());
    let enc_champ = Encodage::nouveau(&cfg_champ, dims, &device, dtype);
    let enc_image = Encodage::nouveau(&cfg_image, dims, &device, dtype);
    let champ = ChampB0::nouveau(
        InrHachage::nouveau(&cfg_champ, 1, 64, 2, &device, dtype),
        e.bornes_b0[0],
        e.bornes_b0[1],
    );
    let image = ImageInr {
        reseau: InrHachage::nouveau(&cfg_image, 2, 64, 2, &device, dtype),
    };
    let reglages = Reglages::default();
    let r = estimer(
        &d,
        champ,
        image,
        &enc_champ,
        &enc_image,
        &b0_init,
        &reglages,
        |it, b| {
            if it % 50 == 0 || it + 1 == reglages.n_iter {
                println!(
                    "jointe, pas {it:>3} : perte {:+.4e} (fidélité {:.3e}, image {:+.4}, b0 {:+.4})",
                    b.totale, b.fidelite, b.image, b.b0
                );
            }
        },
    );
    let secondes_j = t_j.elapsed().as_secs_f64();
    let secondes = t0.elapsed().as_secs_f64();
    println!(
        "{} + {} : {secondes:.1} s (méthode B {secondes_b:.1} s, jointe {secondes_j:.1} s)",
        o.reg, o.inv
    );

    let s = hdf5::File::create(&o.sortie).unwrap();
    s.new_dataset_builder().with_data(&r.b0).create("b0_hz").unwrap();
    let u = Zip::from(&r.u_re)
        .and(&r.u_im)
        .map_collect(|a, b| Complex64::new(*a, *b));
    s.new_dataset_builder().with_data(&u).create("u").unwrap();
    s.new_dataset_builder()
        .with_data(&b0_init)
        .create("b0_methode_b")
        .unwrap();
    s.new_dataset_builder()
        .with_data(&e.masque.mapv(u8::from))
        .create("masque")
        .unwrap();
    if o.ecrire_entrees {
        s.new_dataset_builder().with_data(&e.k_reg).create("k_reg").unwrap();
        s.new_dataset_builder().with_data(&e.k_inv).create("k_inv").unwrap();
        s.new_dataset_builder()
            .with_data(e.temps.as_slice())
            .create("temps")
            .unwrap();
        s.new_dataset_builder()
            .with_data(&e.espacement_mm)
            .create("espacement_mm")
            .unwrap();
        s.new_dataset_builder()
            .with_data(&e.bornes_b0)
            .create("bornes_b0")
            .unwrap();
    }
    let attr = |nom: &str, v: f64| s.new_attr::<f64>().create(nom).unwrap().write_scalar(&v).unwrap();
    attr("secondes", secondes);
    attr("secondes_methode_b", secondes_b);
    attr("secondes_jointe", secondes_j);
    attr("graine", o.graine as f64);
    attr("echo_reg", e.echo_reg);
    attr("echo_inv", e.echo_inv);
    attr("demi_largeur", e.demi_largeur);
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
    texte("plan", e.plan.as_deref().unwrap_or(""));
}

fn main() {
    let o = options();
    if o.f64 {
        lancer::<f64>(&o, DType::F64)
    } else {
        lancer::<f32>(&o, DType::F32)
    }
}
