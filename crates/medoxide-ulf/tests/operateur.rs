//! Critère 1 de P4 ter (docs/ULF.md) : `OperateurCpr` (tenseurs sur le dispositif) contre `ModeleCpr` (CPU, validé
//! contre fmi) : direct, adjoint, adjoint pondéré, transformées sans champ, deux gradients. Grilles paires et impaires ;
//! taille P1 (115 × 115 × 45) si `MEDOXIDE_ULF_OP_P1` est défini. Seuils (écarts relatifs L2) : ≤ 1e-12 en `f64` (CPU),
//! ≤ 1e-5 en `f32` (CPU ; wgpu si `MEDOXIDE_ULF_GPU` ; CUDA si `MEDOXIDE_ULF_CUDA` avec la feature `cuda`).

use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use medoxide_ulf::operateur::{Complexe, OperateurCpr};
use medoxide_ulf::{ModeleCpr, Reel};
use ndarray::Array3;
use rustfft::num_complex::Complex;

fn aleatoire(dims: [usize; 3], graine: u64) -> Array3<f64> {
    let mut e = graine;
    Array3::from_shape_simple_fn(dims, move || {
        e = e.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((e >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

fn vers(a: &Array3<f64>, device: &Device, dtype: DType) -> Tensor<3> {
    let (n0, n1, n2) = a.dim();
    Tensor::from_data(
        TensorData::new(a.iter().copied().collect::<Vec<_>>(), [n0, n1, n2]),
        (device, dtype),
    )
}

fn en_f64(t: &Tensor<3>) -> Vec<f64> {
    t.clone()
        .try_into_data_dtype(DType::F64)
        .unwrap()
        .try_to_vec::<f64>()
        .unwrap()
}

fn ecart_c<T: Reel>(a: &Complexe<3>, r: &Array3<Complex<T>>) -> f64 {
    let (re, im) = (en_f64(&a.re), en_f64(&a.im));
    let (mut num, mut den) = (0.0, 0.0);
    for ((x, y), z) in re.iter().zip(&im).zip(r) {
        let (zr, zi) = (z.re.to_f64().unwrap(), z.im.to_f64().unwrap());
        num += (x - zr).powi(2) + (y - zi).powi(2);
        den += zr * zr + zi * zi;
    }
    (num / den).sqrt()
}

fn ecart_r<T: Reel>(a: &Tensor<3>, r: &Array3<T>) -> f64 {
    let v = en_f64(a);
    let num: f64 = v.iter().zip(r).map(|(x, y)| (x - y.to_f64().unwrap()).powi(2)).sum();
    let den: f64 = r.iter().map(|y| y.to_f64().unwrap().powi(2)).sum();
    (num / den).sqrt()
}

/// Écarts des sept quantités, calculées dans la précision T par les deux opérateurs.
fn ecarts<T: Reel>(dims: [usize; 3], device: &Device, dtype: DType) -> Vec<(&'static str, f64)> {
    let temps: Vec<f64> = (0..dims[0]).map(|i| -3.6e-3 + 6.25e-5 * i as f64).collect();
    let temps_t: Vec<T> = temps.iter().map(|&t| T::from_f64(t).unwrap()).collect();
    let cpu = ModeleCpr::<T>::new(dims, &temps_t);
    let op = OperateurCpr::new(dims, &temps, device, dtype);
    let (ure, uim, kre, kim, gre, gim) = (
        aleatoire(dims, 1),
        aleatoire(dims, 2),
        aleatoire(dims, 3),
        aleatoire(dims, 4),
        aleatoire(dims, 5),
        aleatoire(dims, 6),
    );
    let b0 = aleatoire(dims, 7).mapv(|v| 150.0 * v);
    let hote = |re: &Array3<f64>, im: &Array3<f64>| {
        ndarray::Zip::from(re)
            .and(im)
            .map_collect(|a, b| Complex::new(T::from_f64(*a).unwrap(), T::from_f64(*b).unwrap()))
    };
    let disp = |re: &Array3<f64>, im: &Array3<f64>| Complexe::new(vers(re, device, dtype), vers(im, device, dtype));
    let (u, k, g) = (hote(&ure, &uim), hote(&kre, &kim), hote(&gre, &gim));
    let (ud, kd, gd) = (disp(&ure, &uim), disp(&kre, &kim), disp(&gre, &gim));
    let b0h = b0.mapv(|v| T::from_f64(v).unwrap());
    let b0d = vers(&b0, device, dtype);
    let zero = Array3::<T>::zeros(dims);
    let (a, at) = cpu.adjoint_et_pondere(&k, &b0h);
    let (ad, atd) = op.adjoint_et_pondere(&kd, &b0d);
    let (gu, gb) = cpu.gradient_direct(&u, &b0h, &g);
    let (gud, gbd) = op.gradient_direct(&ud, &b0d, &gd);
    vec![
        ("direct", ecart_c(&op.direct(&ud, &b0d), &cpu.direct(&u, &b0h))),
        ("adjoint", ecart_c(&op.adjoint(&kd, &b0d), &cpu.adjoint(&k, &b0h))),
        ("adjoint (pondéré, 1)", ecart_c(&ad, &a)),
        ("adjoint pondéré", ecart_c(&atd, &at)),
        ("F", ecart_c(&op.transformee(&ud), &cpu.direct(&u, &zero))),
        ("F⁻¹", ecart_c(&op.transformee_inverse(&kd), &cpu.adjoint(&k, &zero))),
        ("gradient direct, u", ecart_c(&gud, &gu)),
        ("gradient direct, b0", ecart_r(&gbd, &gb)),
        (
            "gradient adjoint, b0",
            ecart_r(&op.gradient_adjoint_b0(&atd, &gd), &cpu.gradient_adjoint_b0(&at, &g)),
        ),
    ]
}

fn verifier<T: Reel>(nom: &str, device: &Device, dtype: DType, seuil: f64) {
    let mut grilles = vec![[7, 6, 5], [8, 5, 4], [16, 12, 9]];
    if std::env::var("MEDOXIDE_ULF_OP_P1").is_ok() {
        grilles.push([115, 115, 45]);
    }
    for dims in grilles {
        let e = ecarts::<T>(dims, device, dtype);
        let pire = e.iter().map(|x| x.1).fold(0.0, f64::max);
        println!("{nom} {dims:?} : {pire:.2e}  {e:?}");
        assert!(e.iter().all(|x| x.1 <= seuil), "{nom} {dims:?} : {e:?}");
    }
}

#[test]
fn comme_le_modele_cpu_f64() {
    verifier::<f64>("f64 cpu", &Device::flex(), DType::F64, 1e-12);
}

#[test]
fn comme_le_modele_cpu_f32() {
    verifier::<f32>("f32 cpu", &Device::flex(), DType::F32, 1e-5);
}

#[test]
fn comme_le_modele_cpu_f32_gpu() {
    if std::env::var("MEDOXIDE_ULF_GPU").is_ok() {
        verifier::<f32>(
            "f32 wgpu",
            &Device::wgpu(burn::tensor::DeviceKind::DefaultDevice),
            DType::F32,
            1e-5,
        );
    }
    #[cfg(feature = "cuda")]
    if std::env::var("MEDOXIDE_ULF_CUDA").is_ok() {
        verifier::<f32>("f32 cuda", &Device::cuda(0), DType::F32, 1e-5);
    }
}
