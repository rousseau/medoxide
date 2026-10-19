//! Sonde de performance : une convolution 3D 3×3×3 (`Conv3d`) de `canaux` vers `canaux` canaux sur un cube de côté `n`, pour décider du backend avant de porter les réseaux 3D de BOUNTI (voir `docs/BOUNTI.md`).
//!
//! Une convolution 3D fait, pour chaque voxel et chaque canal de sortie, une somme pondérée sur tous les canaux d'entrée et les 27 voisins : `n³ × canaux² × 27` multiplications-additions. La mémoire d'une carte de caractéristiques est `canaux × n³ × 4` octets en f32.
//!
//! Usage : `cargo run --release -p medoxide-svr --example sonde_conv3d -- <n> <canaux> <wgpu|flex>`
//! (mesure de la mémoire maximale : `/usr/bin/time -l` sous macOS). Un premier passage sur un petit cube (n = 16) amorce les noyaux GPU, il n'est pas chronométré.

use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::PaddingConfig3d;
use burn::tensor::{Device, DeviceKind, Distribution, Tensor};
use std::time::Instant;

/// Passe avant d'une convolution et attente de sa fin : la lecture d'un scalaire force le calcul (les backends GPU sont asynchrones).
fn passe_avant(conv: &Conv3d, entree: Tensor<5>) -> f32 {
    conv.forward(entree).sum().into_scalar()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map_or(64, |v| v.parse().expect("n entier"));
    let canaux: usize = args.get(2).map_or(32, |v| v.parse().expect("canaux entier"));
    let backend = args.get(3).map_or("wgpu", String::as_str);
    let device = match backend {
        "flex" => Device::flex(),
        "wgpu" => Device::wgpu(DeviceKind::DefaultDevice),
        autre => panic!("backend inconnu : {autre} (wgpu ou flex)"),
    };
    let conv: Conv3d = Conv3dConfig::new([canaux, canaux], [3, 3, 3])
        .with_padding(PaddingConfig3d::Same)
        .init(&device);

    // amorçage (compilation des noyaux GPU) sur un petit cube, non chronométré
    let petit = Tensor::<5>::random([1, canaux, 16, 16, 16], Distribution::Default, &device);
    let _ = passe_avant(&conv, petit);

    let entree = Tensor::<5>::random([1, canaux, n, n, n], Distribution::Default, &device);
    let debut = Instant::now();
    let somme = passe_avant(&conv, entree);
    let duree = debut.elapsed().as_secs_f64();
    let mac = (n as f64).powi(3) * (canaux * canaux * 27) as f64;
    println!("{backend} n={n} canaux={canaux} : {duree:.2} s, {:.1} G multiplications-additions/s (somme de contrôle {somme:.3e})", mac / duree / 1e9);
}
