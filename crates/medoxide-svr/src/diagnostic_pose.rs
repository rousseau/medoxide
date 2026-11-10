//! **Diagnostic de F4.0.6 (étape E43)** : pourquoi les poses `analytic` et `autodiff` divergent sur certaines coupes (test ignoré,
//! lancé à la main sur des données locales ; il n'écrit que dans le dossier de `MEDOXIDE_DIAG_SORTIE`).
//!
//! Ce n'est pas un test de non-régression : il mesure. Il rejoue la boucle coupe-volume cycle par cycle, en trois trajectoires :
//! `A` (autodiff), `N` (analytique) et `I` (« isolé » : l'analytique, cycle par cycle, **depuis les poses de la trajectoire `A`**, donc avec
//! exactement le même point de départ, la même référence et le même a priori que l'autodiff). L'écart `I` contre `A` d'un cycle ne vient
//! donc que du recalage lui-même (relances, minima locaux, bruit `f32` de l'autodiff, gradient), non d'une référence différente.
//!
//! Pour chaque coupe dont l'écart isolé dépasse 0,1 mm, le diagnostic refait à part, pour les deux gradients, le premier recalage et les six
//! relances (que la coupe ait été relancée ou non), et mesure en `f64` indépendant : la perte `−NCC + λ·pénalité` de chaque pose finale
//! (laquelle est le meilleur minimum), et l'écart du gradient analytique à des différences finies centrées `f64` (pose nulle et poses
//! finales des deux premiers recalages).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use nalgebra::{Matrix4, Vector3};

use crate::align::{
    cibles_temporelles, preparer_coupe, reference_de_la_coupe, register_slices_leave_window_out, tenseur_1d,
    vecteurs_vers_tenseur, CoupePreparee,
};
use crate::diff::analytique::{
    cout_et_gradient, register_slice_with_prior_analytique, CoupeHote, PrioriHote, VolumeHote,
};
use crate::diff::{
    departs_deterministes, register_slice_with_prior, PoseGradient, RegistrationConfig, RobustConfig, TemporalPrior,
    VolumeTensors,
};
use crate::pipeline::{autodiff_device, loop_config, s2v_start, DeviceKind, Motion, ReconstructionConfig};
use crate::recon::{adjoint_sums, reconstruction_grid, SlicePoses};
use crate::{Stack, Volume};

/// Écart quadratique moyen (mm) entre deux poses d'une coupe, sur ses points nominaux.
fn ecart_rms(points: &[Vector3<f64>], a: &Matrix4<f64>, b: &Matrix4<f64>) -> f64 {
    let s: f64 = points
        .iter()
        .map(|p| ((a * p.push(1.0)).xyz() - (b * p.push(1.0)).xyz()).norm_squared())
        .sum();
    (s / points.len().max(1) as f64).sqrt()
}

/// Écart quadratique moyen (mm) entre deux jeux de paramètres de pose d'une même coupe (même pivot, même échelle).
fn ecart_params(p: &CoupePreparee, a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let ma = crate::diff::pose_to_matrix(a, &p.pivot, p.echelle);
    let mb = crate::diff::pose_to_matrix(b, &p.pivot, p.echelle);
    ecart_rms(&p.points, &ma, &mb)
}

/// Gradient par différences finies centrées `f64` de la perte complète au point `params`, au pas `h` (mm pour les translations, mm
/// aussi pour les rotations : les paramètres sont φ = ω · échelle).
fn gradient_differences_finies(
    vol: &VolumeHote,
    coupe: &CoupeHote,
    prior: Option<&PrioriHote>,
    params: &[f64; 6],
    h: f64,
) -> [f64; 6] {
    let mut g = [0.0; 6];
    for k in 0..6 {
        let (mut p, mut m) = (*params, *params);
        p[k] += h;
        m[k] -= h;
        g[k] =
            (cout_et_gradient(vol, coupe, &p, prior).perte - cout_et_gradient(vol, coupe, &m, prior).perte) / (2.0 * h);
    }
    g
}

fn norme(v: &[f64; 6]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Erreur relative `‖g − fd‖ / ‖fd‖` du gradient analytique contre les différences finies, la meilleure des trois pas essayés.
fn erreur_gradient(vol: &VolumeHote, coupe: &CoupeHote, prior: Option<&PrioriHote>, params: &[f64; 6]) -> (f64, f64) {
    let g = cout_et_gradient(vol, coupe, params, prior).gradient;
    let mut meilleure = f64::INFINITY;
    for h in [1e-3, 1e-4, 1e-5] {
        let fd = gradient_differences_finies(vol, coupe, prior, params, h);
        let d: [f64; 6] = std::array::from_fn(|k| g[k] - fd[k]);
        meilleure = meilleure.min(norme(&d) / norme(&fd).max(1e-300));
    }
    (meilleure, norme(&g))
}

fn lire_stacks(entrees: &str, masques: &str) -> Vec<Stack> {
    let chemins: Vec<PathBuf> = entrees.split(',').map(PathBuf::from).collect();
    let mut stacks = crate::read_stacks(&chemins).expect("lecture des stacks");
    for (s, m) in stacks.iter_mut().zip(masques.split(',')) {
        s.set_brain_mask(std::path::Path::new(m)).expect("masque");
    }
    for s in &mut stacks {
        if let Some(instants) = crate::read_slice_timing(s.path()) {
            s.set_slice_times(instants);
        }
    }
    stacks
}

#[test]
#[ignore = "diagnostic de F4.0.6 sur données locales : MEDOXIDE_DIAG_ENTREES, MEDOXIDE_DIAG_MASQUES, MEDOXIDE_DIAG_SORTIE"]
fn diagnostic_des_poses_divergentes() {
    let entrees =
        std::env::var("MEDOXIDE_DIAG_ENTREES").expect("MEDOXIDE_DIAG_ENTREES : stacks séparés par des virgules");
    let masques = std::env::var("MEDOXIDE_DIAG_MASQUES").expect("MEDOXIDE_DIAG_MASQUES");
    let sortie = PathBuf::from(std::env::var("MEDOXIDE_DIAG_SORTIE").expect("MEDOXIDE_DIAG_SORTIE : dossier"));
    std::fs::create_dir_all(&sortie).unwrap();
    let seuil_mm: f64 = std::env::var("MEDOXIDE_DIAG_SEUIL_MM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);

    let stacks = lire_stacks(&entrees, &masques);
    let config = ReconstructionConfig {
        motion: Motion::SliceToVolume,
        ..ReconstructionConfig::default()
    };
    let grille = reconstruction_grid(&stacks, config.resolution_mm, config.margin_mm).unwrap();
    let (stacks, departs) = s2v_start(&stacks, SlicePoses::identity(&stacks), &config).unwrap();
    let cpu = autodiff_device(DeviceKind::Cpu);
    let boucle = loop_config(&config);
    let robuste_a: RobustConfig = RobustConfig {
        pose_gradient: PoseGradient::Autodiff,
        ..boucle.robust
    };
    let robuste_n = RobustConfig {
        pose_gradient: PoseGradient::Analytic,
        ..boucle.robust
    };
    let geometrie = grille.zeros();
    let identite = Matrix4::<f64>::identity();

    let mut tableau = String::from(
        "cycle\tstack\tcoupe\tpixels\tncc_first_A\tncc_first_I\trelances_A\trelances_I\tncc_final_A\tncc_final_I\tdepl_iso_mm\tdepl_chaine_mm\tncc_first_N\trelances_N\tncc_final_N\tdepl_iso_P_mm\tdepl_chaine_P_mm\tncc_first_P\trelances_P\n",
    );
    let mut detail = String::from("cycle\tstack\tcoupe\tdepart\tdepl_A_I_mm\tncc_A\tncc_I\tperte_f64_A\tperte_f64_I\n");
    let mut resume_coupes = String::from(
        "cycle\tstack\tcoupe\tdepl_iso_mm\tncc_first_A\tncc_first_I\trelances_A\trelances_I\tperte_meilleure_A\tperte_meilleure_I\tmeilleur\terr_grad_zero\terr_grad_A\terr_grad_I\tpremiers_recalages_mm\tpixels_hors_grille\n",
    );

    let stacks_p: Vec<Stack> = stacks
        .iter()
        .map(|st| st.divided_by_slice_scales(&vec![1.0 + 7e-7; st.dim().2]).unwrap())
        .collect();
    let (mut poses_a, mut poses_n, mut poses_p) = (departs.clone(), departs.clone(), departs.clone());
    for cycle in 0..boucle.cycles {
        let debut = std::time::Instant::now();
        let rap_a = register_slices_leave_window_out(&grille, &stacks, &poses_a, 0, 0.0, robuste_a, &cpu).unwrap();
        let rap_i = register_slices_leave_window_out(&grille, &stacks, &poses_a, 0, 0.0, robuste_n, &cpu).unwrap();
        let rap_n = if cycle == 0 {
            rap_i.clone()
        } else {
            register_slices_leave_window_out(&grille, &stacks, &poses_n, 0, 0.0, robuste_n, &cpu).unwrap()
        };
        // contrôle de bruit : l'autodiff lui-même, sur des données multipliées par 1 + 7e-7 (NCC invariante : seul l'arrondi `f32` change)
        let rap_pi = register_slices_leave_window_out(&grille, &stacks_p, &poses_a, 0, 0.0, robuste_a, &cpu).unwrap();
        let rap_p = if cycle == 0 {
            rap_pi.clone()
        } else {
            register_slices_leave_window_out(&grille, &stacks_p, &poses_p, 0, 0.0, robuste_a, &cpu).unwrap()
        };
        eprintln!(
            "cycle {cycle} : cinq recalages en {:.0} s",
            debut.elapsed().as_secs_f64()
        );
        let (num, den) = adjoint_sums(&grille, &stacks, &poses_a).unwrap();
        let mut divergentes = 0usize;
        for (idx, ra) in rap_a.slices.iter().enumerate() {
            let (si, k) = (ra.stack, ra.slice);
            let ri = &rap_i.slices[idx];
            let rn = &rap_n.slices[idx];
            let rp = &rap_p.slices[idx];
            if !ra.registered {
                continue;
            }
            let courante = *poses_a.get(si, k);
            let Some(p) = preparer_coupe(&stacks[si], k, courante) else {
                continue;
            };
            let Some(nominale) = preparer_coupe(&stacks[si], k, identite) else {
                continue;
            };
            let depl_iso = ecart_rms(&nominale.points, rap_a.poses.get(si, k), rap_i.poses.get(si, k));
            let depl_chaine = ecart_rms(&nominale.points, rap_a.poses.get(si, k), rap_n.poses.get(si, k));
            let depl_iso_p = ecart_rms(&nominale.points, rap_a.poses.get(si, k), rap_pi.poses.get(si, k));
            let depl_chaine_p = ecart_rms(&nominale.points, rap_a.poses.get(si, k), rap_p.poses.get(si, k));
            writeln!(
                tableau,
                "{cycle}\t{si}\t{k}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{:.6}\t{:.6}\t{depl_iso:.6}\t{depl_chaine:.6}\t{:.6}\t{}\t{:.6}\t{depl_iso_p:.6}\t{depl_chaine_p:.6}\t{:.6}\t{}",
                p.points.len(),
                ra.ncc_first,
                ri.ncc_first,
                ra.runs,
                ri.runs,
                ra.ncc_final,
                ri.ncc_final,
                rn.ncc_first,
                rn.runs,
                rn.ncc_final,
                rp.ncc_first,
                rp.runs
            )
            .unwrap();
            if depl_iso <= seuil_mm {
                continue;
            }
            divergentes += 1;
            // ---- diagnostic à part de cette coupe, depuis le même état que le cycle ----
            let image = reference_de_la_coupe(&grille, &geometrie, &stacks[si], si, k, &poses_a, 0, &num, &den);
            let reference = Volume::new(image, grille.affine).unwrap();
            let cibles = cibles_temporelles(
                &stacks[si],
                &poses_a,
                si,
                k,
                courante,
                &p.points,
                config.s2v_temporal_weight,
            );
            let hote = VolumeHote::new(&reference);
            let masque = vec![1.0; p.points.len()];
            let coupe = CoupeHote {
                points: &p.points,
                pivot: p.pivot,
                echelle_mm: p.echelle,
                intensites: &p.intensites,
                masque: &masque,
            };
            let prior_h = cibles.as_ref().map(|c| PrioriHote {
                cibles: c,
                poids: config.s2v_temporal_weight,
            });
            let vt = VolumeTensors::new(&reference, &cpu);
            let prior_t = cibles.as_ref().map(|c| TemporalPrior {
                targets: vecteurs_vers_tenseur(c, &cpu),
                weight: config.s2v_temporal_weight,
            });
            let mut departs_k = vec![[0.0; 6]];
            departs_k.extend(departs_deterministes(
                boucle.robust.extra_starts,
                boucle.robust.start_amplitude_mm,
                boucle.robust.seed,
            ));
            // une pose de départ = une paire de recalages (autodiff, analytique) ; la perte est évaluée en f64 indépendant
            let mut essais: Vec<([f64; 6], [f64; 6], f64, f64, f64, f64)> = Vec::new();
            for (n, depart) in departs_k.iter().enumerate() {
                let reglage = RegistrationConfig {
                    initial: *depart,
                    ..boucle.robust.base
                };
                let a = register_slice_with_prior(
                    &vt,
                    vecteurs_vers_tenseur(&p.points, &cpu),
                    tenseur_1d(&[p.pivot.x, p.pivot.y, p.pivot.z], &cpu),
                    p.echelle,
                    tenseur_1d(&p.intensites, &cpu),
                    tenseur_1d(&masque, &cpu),
                    reglage,
                    prior_t.as_ref(),
                );
                let i = register_slice_with_prior_analytique(&hote, &coupe, reglage, prior_h.as_ref());
                let perte_a = cout_et_gradient(&hote, &coupe, &a.params, prior_h.as_ref()).perte;
                let perte_i = cout_et_gradient(&hote, &coupe, &i.params, prior_h.as_ref()).perte;
                writeln!(
                    detail,
                    "{cycle}\t{si}\t{k}\t{n}\t{:.6}\t{:.6}\t{:.6}\t{perte_a:.8}\t{perte_i:.8}",
                    ecart_params(&p, &a.params, &i.params),
                    a.ncc_history.last().unwrap(),
                    i.ncc_history.last().unwrap(),
                )
                .unwrap();
                essais.push((
                    a.params,
                    i.params,
                    perte_a,
                    perte_i,
                    *a.ncc_history.last().unwrap(),
                    *i.ncc_history.last().unwrap(),
                ));
            }
            // meilleure perte de chaque côté, parmi les essais que la coupe a réellement faits (1 si NCC du premier ≥ seuil, tous sinon)
            let jusque = |ncc_first: f64| {
                if ncc_first >= boucle.robust.suspect_ncc {
                    1
                } else {
                    essais.len()
                }
            };
            let meilleure = |n: usize, f: fn(&([f64; 6], [f64; 6], f64, f64, f64, f64)) -> f64| {
                essais[..n].iter().map(f).fold(f64::INFINITY, f64::min)
            };
            let pa = meilleure(jusque(ra.ncc_first), |e| e.2);
            let pi = meilleure(jusque(ri.ncc_first), |e| e.3);
            let verdict = if (pa - pi).abs() <= 1e-6 {
                "egal"
            } else if pi < pa {
                "analytique"
            } else {
                "autodiff"
            };
            let (err0, _) = erreur_gradient(&hote, &coupe, prior_h.as_ref(), &[0.0; 6]);
            let (err_a, _) = erreur_gradient(&hote, &coupe, prior_h.as_ref(), &essais[0].0);
            let (err_i, _) = erreur_gradient(&hote, &coupe, prior_h.as_ref(), &essais[0].1);
            let premiers = ecart_params(&p, &essais[0].0, &essais[0].1);
            let hors = {
                let inverse = reference.inverse;
                let (nx, ny, nz) = reference.dim();
                p.points
                    .iter()
                    .filter(|x| {
                        let v = (inverse * x.push(1.0)).xyz();
                        v.x < 0.0
                            || v.y < 0.0
                            || v.z < 0.0
                            || v.x > (nx - 1) as f64
                            || v.y > (ny - 1) as f64
                            || v.z > (nz - 1) as f64
                    })
                    .count()
            };
            writeln!(
                resume_coupes,
                "{cycle}\t{si}\t{k}\t{depl_iso:.4}\t{:.5}\t{:.5}\t{}\t{}\t{pa:.8}\t{pi:.8}\t{verdict}\t{err0:.2e}\t{err_a:.2e}\t{err_i:.2e}\t{premiers:.5}\t{hors}",
                ra.ncc_first, ri.ncc_first, ra.runs, ri.runs
            )
            .unwrap();
        }
        eprintln!("cycle {cycle} : {divergentes} coupes à plus de {seuil_mm} mm (isolé)");
        poses_a = rap_a.poses;
        poses_n = rap_n.poses;
        poses_p = rap_p.poses;
        std::fs::write(sortie.join("cycles.tsv"), &tableau).unwrap();
        std::fs::write(sortie.join("essais.tsv"), &detail).unwrap();
        std::fs::write(sortie.join("coupes.tsv"), &resume_coupes).unwrap();
    }
    // écart final des deux trajectoires (celui que mesurait (c)(1) de 0.6c)
    let mut finale: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    for (si, s) in stacks.iter().enumerate() {
        for k in 0..s.dim().2 {
            if let Some(nominale) = preparer_coupe(s, k, identite) {
                finale.insert(
                    (si, k),
                    (
                        ecart_rms(&nominale.points, poses_a.get(si, k), poses_n.get(si, k)),
                        ecart_rms(&nominale.points, poses_a.get(si, k), poses_p.get(si, k)),
                    ),
                );
            }
        }
    }
    let mut texte = String::from("stack\tcoupe\tdepl_final_mm\tdepl_final_P_mm\n");
    for ((si, k), (d, dp)) in finale {
        writeln!(texte, "{si}\t{k}\t{d:.6}\t{dp:.6}").unwrap();
    }
    std::fs::write(sortie.join("finale.tsv"), texte).unwrap();
}
