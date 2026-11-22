//! Durée d'un pas de démarrage à chaud (champ, image) aux réglages de P1 sur 115 × 115 × 45, sans données.
//!
//!     cargo run --release -p medoxide-ulf --example chrono_pas [--features cuda] -- <cpu|wgpu|cuda> [f64]

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::DType;
use medoxide_ulf::inr::{ChampB0, ConfigGrille, Encodage, ImageInr, InrHachage};
use medoxide_ulf::jointe::{pas_prefit_champ, Adam, CibleChamp};
use ndarray::Array3;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let device = match args.get(1).map(String::as_str).unwrap_or("cpu") {
        "wgpu" => Device::wgpu(burn::tensor::DeviceKind::DefaultDevice).autodiff(),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0).autodiff(),
        _ => Device::flex().autodiff(),
    };
    let dtype = if args.get(2).is_some_and(|a| a == "f64") { DType::F64 } else { DType::F32 };
    let dims = [115, 115, 45];
    let t = Instant::now();
    let (cc, ci) = (ConfigGrille::champ_p1(), ConfigGrille::image_p1());
    let (ec, ei) = (Encodage::nouveau(&cc, dims, &device, dtype), Encodage::nouveau(&ci, dims, &device, dtype));
    println!("encodages : {:.2} s", t.elapsed().as_secs_f64());
    let mut champ = ChampB0::nouveau(InrHachage::nouveau(&cc, 1, 64, 2, &device, dtype), 12.5, 108.0);
    let image = ImageInr { reseau: InrHachage::nouveau(&ci, 2, 64, 2, &device, dtype) };
    let cible = CibleChamp::nouvelle(&Array3::from_shape_fn(dims, |(i, _, _)| i as f64), &device, dtype);
    let mut adam = Adam::default();
    for k in 0..4 {
        let t = Instant::now();
        let (c, p) = pas_prefit_champ(champ, &ec, &cible, &mut adam, 1e-2);
        champ = c;
        println!("pas du champ {k} : {:.3} s (perte {p:.4})", t.elapsed().as_secs_f64());
    }
    for k in 0..3 {
        let t = Instant::now();
        let (re, im) = image.image(&ei);
        let v = (re.powi_scalar(2) + im.powi_scalar(2)).sum();
        let s = v.clone().into_data().convert::<f64>().to_vec::<f64>().unwrap()[0];
        println!("image, avant {k} : {:.3} s ({s:?})", t.elapsed().as_secs_f64());
        let t = Instant::now();
        let g = v.backward();
        let gt = image.reseau.tables[15].val().grad(&g).unwrap().sum().into_data().convert::<f64>().to_vec::<f64>().unwrap()[0];
        println!("image, arrière {k} : {:.3} s ({gt:?})", t.elapsed().as_secs_f64());
    }
}
