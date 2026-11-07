//! Générateurs de mouvement vrai des bancs d'essai (E2 et banc de vérité terrain, docs/ROADMAP.md F4.0.1) : poses rigides par coupe, avec une graine.
//!
//! Remontés tels quels du module de tests de `recon.rs` (même code, mêmes bits : voir le test
//! `the_true_motion_models_keep_their_recorded_fingerprints`) pour que le simulateur de `medx svr simulate` et les diagnostics d'E2
//! utilisent les mêmes mouvements.

use crate::recon::SlicePoses;
use crate::Stack;
use nalgebra::{Matrix4, Rotation3, Vector3};

/// Petit générateur pseudo-aléatoire xorshift64 : déterministe, sans dépendance ; `suivant` rend un flottant dans [0, 1).
pub(crate) struct Alea(pub(crate) u64);

impl Alea {
    pub(crate) fn suivant(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Mouvement rigide `M = T(c + t) · R(ω) · T(−c)` : rotation du vecteur `omega` (rad) autour du point `c`, puis translation `t`.
pub(crate) fn mouvement_rigide(omega: &Vector3<f64>, t: &Vector3<f64>, c: &Vector3<f64>) -> Matrix4<f64> {
    let r = Rotation3::from_scaled_axis(*omega);
    let mut m = Matrix4::identity();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(&(c + t - r * c));
    m
}

/// Les coupes d'un stack dans l'ordre d'acquisition : celui des instants `SliceTiming` s'ils sont attachés au stack, sinon l'ordre des indices (le tri est stable : sans instants, rien ne change).
pub(crate) fn coupes_dans_l_ordre(stack: &Stack) -> Vec<crate::Slice<'_>> {
    let mut coupes: Vec<_> = stack.slices().collect();
    if let Some(instants) = stack.slice_times() {
        coupes.sort_by(|a, b| instants[a.index()].total_cmp(&instants[b.index()]));
    }
    coupes
}

/// Poses aléatoires par coupe (autour du centre de la coupe) : rotation et translation tirées uniformément dans ± `degres` et ± `mm`.
pub(crate) fn poses_aleatoires(stacks: &[Stack], degres: f64, mm: f64, graine: u64) -> SlicePoses {
    let mut alea = Alea(graine);
    let mut poses = SlicePoses::identity(stacks);
    let mut tire = |a: f64| (alea.suivant() * 2.0 - 1.0) * a;
    for (si, stack) in stacks.iter().enumerate() {
        for coupe in stack.slices() {
            let omega = Vector3::new(tire(degres), tire(degres), tire(degres)).map(f64::to_radians);
            let t = Vector3::new(tire(mm), tire(mm), tire(mm));
            poses.set(
                si,
                coupe.index(),
                mouvement_rigide(&omega, &t, &coupe.geometric_center()),
            );
        }
    }
    poses
}

/// Mouvement par coupe À SAUTS le long de chaque stack : comme [`poses_lisses`] (marche autorégressive de coefficient `rho`, bornée à ± amplitude), mais à chaque coupe, avec la probabilité `proba`, chaque paramètre est retiré uniformément dans ± amplitude (un saut qui persiste ensuite).
pub(crate) fn poses_a_sauts(stacks: &[Stack], degres: f64, mm: f64, graine: u64, rho: f64, proba: f64) -> SlicePoses {
    let mut alea = Alea(graine);
    let mut poses = SlicePoses::identity(stacks);
    for (si, stack) in stacks.iter().enumerate() {
        let mut x = [0.0_f64; 6];
        for coupe in coupes_dans_l_ordre(stack) {
            let saut = alea.suivant() < proba;
            for (p, v) in x.iter_mut().enumerate() {
                let amplitude = if p < 3 { degres } else { mm };
                let (u, w) = (alea.suivant().max(1e-12), alea.suivant());
                let gauss = (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * w).cos();
                *v = if saut {
                    (alea.suivant() * 2.0 - 1.0) * amplitude
                } else {
                    (rho * *v + (1.0 - rho * rho).sqrt() * amplitude / 3.0_f64.sqrt() * gauss)
                        .clamp(-amplitude, amplitude)
                };
            }
            let omega = Vector3::new(x[0], x[1], x[2]).map(f64::to_radians);
            poses.set(
                si,
                coupe.index(),
                mouvement_rigide(&omega, &Vector3::new(x[3], x[4], x[5]), &coupe.geometric_center()),
            );
        }
    }
    poses
}

/// Lit un fichier de poses (stack, coupe, 16 valeurs en ordre colonne) pour des stacks donnés ; les coupes absentes gardent l'identité.
/// Un mouvement rigide par STACK (le même pour toutes ses coupes), autour du barycentre cérébral du stack, tiré dans ± `degres` et ± `mm` : un décalage de stack entier, ce que l'alignement par blocs doit corriger.
pub(crate) fn poses_par_stack_reel(stacks: &[Stack], degres: f64, mm: f64, graine: u64) -> SlicePoses {
    let mut alea = Alea(graine);
    let mut poses = SlicePoses::identity(stacks);
    let mut tire = |a: f64| (alea.suivant() * 2.0 - 1.0) * a;
    for (si, stack) in stacks.iter().enumerate() {
        let centre = stack.brain_barycenter_world().unwrap_or_else(Vector3::zeros);
        let omega = Vector3::new(tire(degres), tire(degres), tire(degres)).map(f64::to_radians);
        let m = mouvement_rigide(&omega, &Vector3::new(tire(mm), tire(mm), tire(mm)), &centre);
        for k in 0..stack.dim().2 {
            poses.set(si, k, m);
        }
    }
    poses
}

/// Mouvement par coupe LISSE le long de chaque stack : chaque paramètre suit une marche autorégressive d'ordre 1 de coefficient `rho` (écart type `amplitude / √3`, comme un tirage uniforme dans ± `amplitude`), borné à ± `amplitude` ; les coupes voisines ont donc des poses proches.
pub(crate) fn poses_lisses(stacks: &[Stack], degres: f64, mm: f64, graine: u64, rho: f64) -> SlicePoses {
    let mut alea = Alea(graine);
    let mut gauss = move || {
        let (u, v) = (alea.suivant().max(1e-12), alea.suivant());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    let mut poses = SlicePoses::identity(stacks);
    for (si, stack) in stacks.iter().enumerate() {
        let mut x = [0.0_f64; 6];
        for coupe in coupes_dans_l_ordre(stack) {
            for (p, v) in x.iter_mut().enumerate() {
                let amplitude = if p < 3 { degres } else { mm };
                let sigma = amplitude / 3.0_f64.sqrt();
                *v = (rho * *v + (1.0 - rho * rho).sqrt() * sigma * gauss()).clamp(-amplitude, amplitude);
            }
            let omega = Vector3::new(x[0], x[1], x[2]).map(f64::to_radians);
            poses.set(
                si,
                coupe.index(),
                mouvement_rigide(&omega, &Vector3::new(x[3], x[4], x[5]), &coupe.geometric_center()),
            );
        }
    }
    poses
}

/// Mouvement vrai d'un banc d'essai selon son modèle : `independante`, `lisse` (marche autorégressive, ρ = 0,9), `sauts` (lisse avec sauts de probabilité 0,08 par coupe) ou `stack` (un décalage rigide par stack).
pub(crate) fn mouvement_vrai(stacks: &[Stack], modele: &str, amplitude: f64, graine: u64) -> SlicePoses {
    match modele {
        "lisse" => poses_lisses(stacks, amplitude, amplitude, graine, 0.9),
        "sauts" => poses_a_sauts(stacks, amplitude, amplitude, graine, 0.9, 0.08),
        "stack" => poses_par_stack_reel(stacks, amplitude, amplitude, graine),
        _ => poses_aleatoires(stacks, amplitude, amplitude, graine),
    }
}

/// Instants d'acquisition d'un stack entrelacé de `n` coupes : les coupes paires d'abord (0, 2, 4, …), puis les impaires.
pub(crate) fn instants_entrelaces(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| if j % 2 == 0 { j / 2 } else { n.div_ceil(2) + j / 2 } as f64)
        .collect()
}
