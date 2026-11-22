//! Durée des passes du modèle CPR aux dimensions de P1 (115 × 115 × 45), `f32`, données aléatoires.
//!
//!     cargo run --release -p medoxide-ulf --example chrono_cpr

use std::time::Instant;

use medoxide_ulf::ModeleCpr;
use ndarray::Array3;
use rustfft::num_complex::Complex;

fn main() {
    let dims = [115, 115, 45];
    let temps: Vec<f32> = (0..115).map(|i| -3.6e-3 + 6.25e-5 * i as f32).collect();
    let m = ModeleCpr::<f32>::new(dims, &temps);
    let mut e = 1u64;
    let mut a = move || {
        e = e.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((e >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let u = Array3::from_shape_simple_fn(dims, || Complex::new(a(), a()));
    let b0 = Array3::from_shape_simple_fn(dims, || 100.0 * a());
    let zero = Array3::<f32>::zeros(dims);
    let n = 5;
    let chrono = |nom: &str, f: &dyn Fn()| {
        f();
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        println!("{nom:<28} {:.1} ms", 1e3 * t.elapsed().as_secs_f64() / n as f64);
    };
    chrono("direct(u, b0)", &|| drop(m.direct(&u, &b0)));
    chrono("direct(u, 0)", &|| drop(m.direct(&u, &zero)));
    chrono("adjoint(k, b0)", &|| drop(m.adjoint(&u, &b0)));
    chrono("adjoint_et_pondere(k, b0)", &|| drop(m.adjoint_et_pondere(&u, &b0)));
    println!("fils rayon : {}", rayon::current_num_threads());
}
