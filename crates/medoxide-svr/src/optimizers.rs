//! Optimiseurs comparés sur l'objectif de reconstruction (étape 4a) : tous consomment la **valeur et le gradient analytique** de
//! [`ReconstructionProblem`] et partent du même point.
//!
//! L'unité de coût est la **passe** : une simulation plus une rétroprojection de toutes les coupes, qui domine tout le reste. Chaque
//! fonction rend une [`Trace`] : l'objectif après chaque passe, et des instantanés de l'itéré à des budgets de passes donnés.

use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams, LBFGSConfig, LineSearchFn};
use burn::prelude::*;
use ndarray::Array3;

use crate::recon::ReconstructionProblem;

/// Trace d'une optimisation.
#[derive(Debug, Clone)]
pub struct Trace {
    /// Dernier itéré.
    pub x: Array3<f64>,
    /// Valeur de l'objectif de chaque itéré, dans l'ordre où ils sont calculés : `objective[k]` est la valeur de l'itéré disponible après
    /// `k + 1` passes (`objective[0]` : le départ). Pour les méthodes à recherche linéaire (L-BFGS de Wolfe), ce sont les points d'essai.
    pub objective: Vec<f64>,
    /// Instantanés `(passes, x)` : l'itéré dont la valeur est `objective[passes - 1]`, pour chaque budget demandé atteint.
    pub snapshots: Vec<(usize, Array3<f64>)>,
    /// Nombre de passes effectuées.
    pub passes: usize,
}

fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Enregistre l'itéré courant si `passes` est un budget demandé.
fn instantane(traces: &mut Vec<(usize, Array3<f64>)>, checkpoints: &[usize], passes: usize, x: &Array3<f64>) {
    if checkpoints.contains(&passes) {
        traces.push((passes, x.clone()));
    }
}

/// Départ restreint au domaine du problème.
fn restreint(p: &ReconstructionProblem, x0: &Array3<f64>) -> Array3<f64> {
    let mut x = x0.clone();
    for (v, &d) in x.iter_mut().zip(p.domain().iter()) {
        if !d {
            *v = 0.0;
        }
    }
    x
}

/// **Gradient conjugué**, avec le même comptage de passes que les autres : une passe pour le second membre, une pour l'évaluation du départ,
/// puis une par itération. `objective[k]` correspond à l'itéré après `k` itérations, soit `k + 2` passes.
pub fn conjugate_gradient(p: &ReconstructionProblem, x0: &Array3<f64>, max_passes: usize, checkpoints: &[usize]) -> Trace {
    let mut snapshots = Vec::new();
    let max_iterations = max_passes.saturating_sub(2);
    let r = p.conjugate_gradient_observed(x0, max_iterations, 0.0, |k, x| instantane(&mut snapshots, checkpoints, k + 2, x));
    Trace { x: r.x, passes: r.iterations + 2, objective: r.objective_history, snapshots }
}

/// **Plus forte pente à pas exact** : `x ← x − α g` avec `α = g·g / g·Hg`, qui minimise `f` le long de `−g` (l'objectif est quadratique).
/// Le gradient suivant se déduit sans nouvelle passe : `g ← g − α H g`. Une passe de départ, puis une (le produit `H g`) par itération.
pub fn steepest_descent(p: &ReconstructionProblem, x0: &Array3<f64>, max_passes: usize, checkpoints: &[usize]) -> Trace {
    let mut x = restreint(p, x0);
    let e = p.evaluate(&x);
    let (mut f, mut g) = (e.value(), e.gradient);
    let mut objective = vec![f];
    let mut snapshots = Vec::new();
    let mut passes = 1;
    instantane(&mut snapshots, checkpoints, passes, &x);
    while passes < max_passes {
        let hg = p.normal_operator(&g);
        passes += 1;
        let gg = dot(&g, &g);
        let pas = gg / dot(&g, &hg);
        x.scaled_add(-pas, &g);
        g.scaled_add(-pas, &hg);
        f -= 0.5 * pas * gg;
        objective.push(f);
        instantane(&mut snapshots, checkpoints, passes, &x);
    }
    Trace { x, objective, snapshots, passes }
}

/// **Barzilai-Borwein** : `x ← x − α g` avec `α = s·s / s·y` (`s` : dernier pas, `y` : dernière variation du gradient). Une passe par
/// itération, non monotone. Le premier pas vaut `1 / (support max + 6α/h²)`, une borne sûre de la courbure.
pub fn barzilai_borwein(p: &ReconstructionProblem, x0: &Array3<f64>, max_passes: usize, checkpoints: &[usize]) -> Trace {
    let mut x = restreint(p, x0);
    let e = p.evaluate(&x);
    let mut objective = vec![e.value()];
    let mut g = e.gradient;
    let mut snapshots = Vec::new();
    let mut passes = 1;
    instantane(&mut snapshots, checkpoints, passes, &x);
    let support_max = f64::from(p.support().iter().fold(0.0_f32, |m, &v| m.max(v)));
    let pas0 = 1.0 / (support_max + 6.0 * p.alpha() / p.resolution_mm().powi(2));
    let mut pas = pas0;
    while passes < max_passes {
        let x_precedent_g = g.clone();
        x.scaled_add(-pas, &g);
        let e = p.evaluate(&x);
        passes += 1;
        objective.push(e.value());
        g = e.gradient;
        instantane(&mut snapshots, checkpoints, passes, &x);
        let s = &x_precedent_g * (-pas); // s = x_{k+1} − x_k
        let y = &g - &x_precedent_g;
        let sy = dot(&s, &y);
        pas = if sy > 0.0 { dot(&s, &s) / sy } else { pas0 };
    }
    Trace { x, objective, snapshots, passes }
}

/// **Jacobi préconditionné**, schéma de SVRTK et de NeSVoR : `x ← x − ω D ∘ g` avec `D = 1 / (support + 6α/h²)` : chaque voxel est corrigé par
/// la moyenne, pondérée, des résidus qui le touchent. Une passe par itération.
pub fn jacobi(p: &ReconstructionProblem, x0: &Array3<f64>, omega: f64, max_passes: usize, checkpoints: &[usize]) -> Trace {
    let mut x = restreint(p, x0);
    let courbure_reg = 6.0 * p.alpha() / p.resolution_mm().powi(2);
    let d = p.support().mapv(|s| 1.0 / (f64::from(s) + courbure_reg));
    let mut objective = Vec::new();
    let mut snapshots = Vec::new();
    let mut passes = 0;
    loop {
        let e = p.evaluate(&x);
        passes += 1;
        objective.push(e.value());
        instantane(&mut snapshots, checkpoints, passes, &x);
        if passes >= max_passes {
            break;
        }
        for ((v, g), dd) in x.iter_mut().zip(e.gradient.iter()).zip(d.iter()) {
            *v -= omega * dd * g;
        }
    }
    Trace { x, objective, snapshots, passes }
}

// ------------------------------------------------------------------ optimiseurs de Burn (f32)

/// L'inconnue `x` dans un `Module`, comme Burn l'exige pour ses optimiseurs.
#[derive(Module, Debug)]
struct VolumeModule {
    x: Param<Tensor<3>>,
}

fn vers_tenseur(a: &Array3<f64>, device: &Device) -> Tensor<3> {
    let (nx, ny, nz) = a.dim();
    let v: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    Tensor::<3>::from_floats(TensorData::new(v, [nx, ny, nz]), device)
}

fn vers_tableau(t: Tensor<3>, dims: (usize, usize, usize)) -> Array3<f64> {
    let v: Vec<f32> = t.into_data().try_to_vec::<f32>().unwrap();
    Array3::from_shape_vec(dims, v.into_iter().map(f64::from).collect()).unwrap()
}

/// **Adam de Burn** (`burn-optim`), `f32` sur `device`, gradient analytique fourni par `p`. `device` doit avoir l'autodiff activé
/// (`Device::….autodiff()`) : l'optimiseur de Burn l'exige pour ses paramètres, même quand le gradient vient de l'extérieur. Chaque coordonnée avance d'environ `lr` par itération, quelle
/// que soit l'échelle de son gradient. Une passe par itération.
pub fn adam(p: &ReconstructionProblem, x0: &Array3<f64>, lr: f64, max_passes: usize, checkpoints: &[usize], device: &Device) -> Trace {
    let dims = x0.dim();
    let mut module = VolumeModule { x: Param::from_tensor(vers_tenseur(&restreint(p, x0), device)) };
    let mut optimiseur = AdamConfig::new().init();
    let (mut objective, mut snapshots) = (Vec::new(), Vec::new());
    let mut passes = 0;
    let mut x = vers_tableau(module.x.val(), dims);
    loop {
        let e = p.evaluate(&x);
        passes += 1;
        objective.push(e.value());
        instantane(&mut snapshots, checkpoints, passes, &x);
        if passes >= max_passes {
            break;
        }
        let mut grads = GradientsParams::new();
        grads.register(module.x.id, vers_tenseur(&e.gradient, device));
        module = optimiseur.step(lr, module, grads);
        x = vers_tableau(module.x.val(), dims);
    }
    Trace { x, objective, snapshots, passes }
}

/// **L-BFGS de Burn** (`burn-optim`), `f32` sur `device` (autodiff activé, voir [`adam`]), gradient analytique fourni par `p` : approximation de l'inverse du hessien par les `history`
/// derniers couples (pas, variation du gradient), avec ou sans recherche linéaire de Wolfe forte. Chaque appel de la fermeture est une passe.
/// `echelle` divise la valeur et le gradient fournis à Burn (1 : aucun changement).
pub fn lbfgs(p: &ReconstructionProblem, x0: &Array3<f64>, lr: f64, history: usize, wolfe: bool, echelle: f64, max_passes: usize, checkpoints: &[usize], device: &Device) -> Trace {
    let dims = x0.dim();
    let module = VolumeModule { x: Param::from_tensor(vers_tenseur(&restreint(p, x0), device)) };
    let config = LBFGSConfig::new()
        .with_max_iter(max_passes)
        .with_max_eval(Some(max_passes))
        .with_history_size(history)
        .with_tolerance_grad(0.0)
        .with_tolerance_change(0.0)
        .with_line_search_fn(if wolfe { LineSearchFn::StrongWolfe } else { LineSearchFn::None });
    let mut optimiseur = config.init();
    let (mut objective, mut snapshots) = (Vec::new(), Vec::new());
    let mut passes = 0;
    let mut dernier = x0.clone();
    let (module, _) = optimiseur.step(lr, module, |m: VolumeModule| {
        let x = vers_tableau(m.x.val(), dims);
        let e = p.evaluate(&x);
        passes += 1;
        objective.push(e.value());
        instantane(&mut snapshots, checkpoints, passes, &x);
        dernier = x;
        let mut grads = GradientsParams::new();
        // L'optimiseur de Burn travaille en f32 avec des tolérances absolues : `echelle` divise la valeur et le gradient (par exemple par f₀) pour
        // les ramener à un ordre de grandeur ordinaire. Le minimum ne change pas ; la trace garde les valeurs d'origine.
        grads.register(m.x.id, vers_tenseur(&(&e.gradient / echelle), device));
        (e.value() / echelle, grads)
    });
    let x = vers_tableau(module.x.val(), dims);
    let _ = dernier;
    Trace { x, objective, snapshots, passes }
}
