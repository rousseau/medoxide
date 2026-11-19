//! **Consensus de graines du recalage coupe-coupe** (F4.1.3) : fusionner K lancements de [`crate::s2s::register_slices_s2s`]
//! à graines différentes en un seul jeu de poses.
//!
//! Le coupe-coupe dépend de sa graine (ordre aléatoire des coupes dans les balayages Gauss–Seidel). Chaque lancement est
//! défini à un mouvement rigide global près : on les met d'abord dans un même repère (Kabsch sur des pixels de masque),
//! puis on prend, coupe par coupe, la **médiane** des K poses :
//!
//! - rotation : médiane de Fréchet sur SO(3) (distance géodésique, l'angle de la rotation relative), calculée par
//!   l'algorithme de Weiszfeld dans l'espace tangent (log et exp de SO(3), par le vecteur « axe × angle ») ;
//! - translation : médiane géométrique, dans R³, de la position monde du centre de la coupe (pas de médiane par
//!   coordonnée, qui dépendrait des axes du repère).
//!
//! La médiane est robuste : avec K = 3 et deux lancements d'accord, la pose consensus est exactement la leur, quelle que
//! soit la troisième. Le mode [`S2sConsensus::Best`] (témoin) ne fusionne rien : il garde le lancement le plus central.

use nalgebra::{Matrix3, Matrix4, Rotation3, Vector3};

use crate::align::{AlignmentReport, SliceReport};
use crate::recon::SlicePoses;
use crate::{Stack, SvrError};

/// Fusion des lancements du coupe-coupe à graines différentes (`--s2s-consensus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum S2sConsensus {
    /// Un seul lancement (comportement historique, chemin de code inchangé).
    #[default]
    None,
    /// Médiane, coupe par coupe, des poses des K lancements (rotation de Fréchet sur SO(3), position géométrique du centre).
    Median,
    /// Témoin : les poses du lancement le plus central (médoïde de l'écart quadratique moyen entre lancements), sans fusion.
    Best,
}

/// Nombre maximal de pixels de masque par coupe utilisés pour l'alignement des lancements et les écarts entre poses
/// (sous-échantillonnage régulier au-delà ; même valeur que `scripts/analyse_accord_poses.py`).
pub const MASK_POINTS_PER_SLICE: usize = 250;
/// Arrêt de Weiszfeld : norme du pas (radians pour les rotations, mm pour les positions).
const WEISZFELD_TOLERANCE: f64 = 1e-10;
/// Nombre maximal de pas de Weiszfeld (la section F4.1.3 en prévoyait 100 : sur 2 000 triplets de rotations, sans le test préalable des données, médiane 42 pas, 99e centile 542, maximum 2 394, la limite de 100 laissait des médianes à 1e-7 rad de leur valeur ; le coût reste négligeable pour K = 3).
const WEISZFELD_MAX_STEPS: usize = 10_000;
/// Un alignement `G_j` dont aucun coefficient ne s'écarte de l'identité de plus de cette valeur est remplacé par l'identité exacte (écart d'arrondi de Kabsch, de l'ordre de 1e-14).
const ALIGNMENT_SNAP: f64 = 1e-12;
/// En deçà de cette distance, l'estimation courante est considérée comme confondue avec une donnée (correction de Vardi et Zhang).
const COINCIDENCE: f64 = 1e-12;

/// Résultat de [`fuse_runs`].
#[derive(Debug, Clone)]
pub struct Consensus {
    /// Poses fusionnées (dans le repère du premier lancement) et rapport par coupe (voir [`fuse_runs`]).
    pub report: AlignmentReport,
    /// Indice du lancement le plus central (médoïde) ; c'est celui que garde [`S2sConsensus::Best`].
    pub medoid_run: usize,
    /// Coupes `(stack, coupe)` dont les rotations des K lancements s'écartent de plus de π/2 : la médiane n'y est pas sûre,
    /// la coupe garde la pose de son lancement le plus central. Toujours vide avec [`S2sConsensus::Best`].
    pub flagged: Vec<(usize, usize)>,
    /// Mouvement rigide `G_j` appliqué à chaque lancement pour le mettre dans le repère du premier (`G_0` = identité).
    pub alignments: Vec<Matrix4<f64>>,
}

/// Positions monde, **aux poses d'en-tête**, d'au plus [`MASK_POINTS_PER_SLICE`] pixels de masque de chaque coupe : `[stack][coupe]`.
/// Une coupe sans pixel de masque a une liste vide.
///
/// # Erreurs
/// [`SvrError::NoMask`] si un stack n'a pas de masque.
pub fn mask_points(stacks: &[Stack]) -> Result<Vec<Vec<Vec<Vector3<f64>>>>, SvrError> {
    stacks
        .iter()
        .map(|stack| {
            let masque = stack
                .brain_mask()
                .ok_or_else(|| SvrError::NoMask(stack.path().to_path_buf()))?
                .voxels();
            Ok(stack
                .slices()
                .map(|coupe| {
                    let k = coupe.index();
                    let dedans: Vec<(usize, usize)> = masque
                        .index_axis(ndarray::Axis(2), k)
                        .indexed_iter()
                        .filter(|(_, &v)| v)
                        .map(|(ij, _)| ij)
                        .collect();
                    let n = dedans.len();
                    let garder = n.min(MASK_POINTS_PER_SLICE);
                    (0..garder)
                        .map(|m| {
                            let (i, j) = dedans[m * n / garder];
                            coupe.pixel_to_world(i as f64, j as f64)
                        })
                        .collect()
                })
                .collect())
        })
        .collect()
}

/// **Kabsch** : le mouvement rigide `G` (rotation propre et translation) qui minimise `Σ |G · source_i − cible_i|²`.
/// Moins de trois points : l'identité (le mouvement n'est pas déterminé).
pub fn kabsch(source: &[Vector3<f64>], cible: &[Vector3<f64>]) -> Matrix4<f64> {
    assert_eq!(source.len(), cible.len(), "kabsch : nuages de tailles différentes");
    if source.len() < 3 {
        return Matrix4::identity();
    }
    let n = source.len() as f64;
    let cs = source.iter().sum::<Vector3<f64>>() / n;
    let cc = cible.iter().sum::<Vector3<f64>>() / n;
    let h: Matrix3<f64> = source
        .iter()
        .zip(cible)
        .map(|(s, c)| (s - cs) * (c - cc).transpose())
        .sum();
    let svd = h.svd(true, true);
    let (u, v) = (svd.u.unwrap(), svd.v_t.unwrap().transpose());
    // correction de signe : sans elle, un nuage presque plan peut donner une réflexion (déterminant −1)
    let d = (v * u.transpose()).determinant().signum();
    let r = v * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * u.transpose();
    let mut g = Matrix4::identity();
    g.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    g.fixed_view_mut::<3, 1>(0, 3).copy_from(&(cc - r * cs));
    g
}

/// Positions des `points` (aux poses d'en-tête) déplacés par les poses `poses`, dans l'ordre stacks, coupes, points.
fn placer(points: &[Vec<Vec<Vector3<f64>>>], poses: &SlicePoses) -> Vec<Vector3<f64>> {
    points
        .iter()
        .enumerate()
        .flat_map(|(si, coupes)| {
            coupes
                .iter()
                .enumerate()
                .flat_map(move |(k, pts)| pts.iter().map(move |x| (poses.get(si, k) * x.push(1.0)).xyz()))
        })
        .collect()
}

/// Un pas de Weiszfeld dans un espace tangent : `v` sont les données vues depuis l'estimation courante (vecteurs `donnée − estimation`).
/// Rend `None` si l'estimation courante est la médiane (cas d'une estimation confondue avec des données, correction de
/// Vardi et Zhang : elle l'est si la résultante des directions vers les autres données est de norme ≤ leur multiplicité),
/// sinon le pas à faire.
fn weiszfeld_step(v: &[Vector3<f64>]) -> Option<Vector3<f64>> {
    let mut confondues = 0.0;
    let (mut resultante, mut somme_poids) = (Vector3::zeros(), 0.0);
    for vj in v {
        let d = vj.norm();
        if d < COINCIDENCE {
            confondues += 1.0;
        } else {
            resultante += vj / d;
            somme_poids += 1.0 / d;
        }
    }
    if somme_poids == 0.0 {
        return None; // toutes les données sont confondues avec l'estimation
    }
    let pas = resultante / somme_poids;
    if confondues == 0.0 {
        return Some(pas);
    }
    let r = resultante.norm();
    if r <= confondues {
        None
    } else {
        Some(pas * (1.0 - confondues / r))
    }
}

/// Indice de la donnée de plus petite somme de distances aux autres (le médoïde ; le premier en cas d'égalité).
fn medoide(n: usize, distance: impl Fn(usize, usize) -> f64) -> usize {
    (0..n)
        .map(|a| (a, (0..n).map(|b| distance(a, b)).sum::<f64>()))
        .fold((0, f64::INFINITY), |m, (a, s)| if s < m.1 { (a, s) } else { m })
        .0
}

/// Résultat d'une médiane : la valeur, et `Some(j)` si c'est exactement la donnée `j` (reconnue par le critère de Vardi et Zhang, sans aucun pas de Weiszfeld).
type Mediane<T> = (T, Option<usize>);

/// **Logarithme de SO(3)** : le vecteur « axe × angle » `ω` tel que `exp(ω) = r`, angle dans [0, π]. Précis aux petits angles (là où Weiszfeld converge) : l'angle vient de `atan2(|w|, (trace − 1)/2)`, où
/// `w = vee(r − rᵀ)/2 = sin θ · axe`, et non d'un `acos` de la trace, qui perd la moitié des chiffres significatifs près de l'identité. Près de π (sin θ → 0), l'axe se lit sur la partie symétrique.
pub fn log_so3(r: &Rotation3<f64>) -> Vector3<f64> {
    let m = r.matrix();
    let w = Vector3::new(m[(2, 1)] - m[(1, 2)], m[(0, 2)] - m[(2, 0)], m[(1, 0)] - m[(0, 1)]) / 2.0;
    let s = w.norm();
    let c = (m.trace() - 1.0) / 2.0;
    let theta = s.atan2(c);
    if s < 1e-12 && c < 0.0 {
        // θ ≈ π : r ≈ 2 a aᵀ − I ; l'axe est la colonne de r + I de plus grande norme, son signe est sans importance à π
        let b = m + Matrix3::identity();
        let j = (0..3)
            .max_by(|&a, &c| b.column(a).norm().total_cmp(&b.column(c).norm()))
            .unwrap();
        return b.column(j).normalize() * theta;
    }
    // θ / sin θ, développé en série quand θ est petit (1 + θ²/6)
    let facteur = if theta < 1e-6 {
        1.0 + theta * theta / 6.0
    } else {
        theta / s
    };
    w * facteur
}

/// **Médiane de Fréchet** de rotations : la rotation `M` qui minimise `Σ_j angle(Mᵀ R_j)`, par Weiszfeld dans l'espace
/// tangent. Chaque donnée est d'abord testée (critère de Vardi et Zhang) : si l'une est la médiane, elle est rendue exactement ; sinon Weiszfeld part du médoïde. `None` si deux rotations s'écartent de plus de π/2 (la médiane n'est alors plus sûrement
/// unique ni atteinte par Weiszfeld).
pub fn rotation_median(rotations: &[Rotation3<f64>]) -> Option<Mediane<Rotation3<f64>>> {
    let n = rotations.len();
    assert!(n > 0, "médiane de zéro rotation");
    let distance = |a: usize, b: usize| log_so3(&(rotations[a].inverse() * rotations[b])).norm();
    if (0..n).any(|a| (0..n).any(|b| distance(a, b) > std::f64::consts::FRAC_PI_2)) {
        return None;
    }
    // une donnée est-elle la médiane ? (critère de Vardi et Zhang en chaque donnée : Weiszfeld n'approche une médiane située sur une donnée que très lentement)
    for (j, rj) in rotations.iter().enumerate() {
        let v: Vec<Vector3<f64>> = rotations.iter().map(|r| log_so3(&(rj.inverse() * r))).collect();
        if weiszfeld_step(&v).is_none() {
            return Some((*rj, Some(j)));
        }
    }
    // sinon la médiane est strictement entre les données : Weiszfeld depuis le médoïde
    let mut m = rotations[medoide(n, distance)];
    let exacte = None;
    for _ in 0..WEISZFELD_MAX_STEPS {
        // log : chaque donnée vue depuis M, en vecteur « axe × angle » (espace tangent en M)
        let v: Vec<Vector3<f64>> = rotations.iter().map(|r| log_so3(&(m.inverse() * r))).collect();
        let Some(pas) = weiszfeld_step(&v) else { break };
        // exp : on revient sur SO(3) en suivant le pas depuis M
        m *= Rotation3::from_scaled_axis(pas);
        if pas.norm() < WEISZFELD_TOLERANCE {
            break;
        }
    }
    Some((m, exacte))
}

/// **Médiane géométrique** de points de R³ : le point `y` qui minimise `Σ_j |p_j − y|` ; une donnée qui l'est est rendue exactement (critère de Vardi et Zhang), sinon Weiszfeld depuis le médoïde.
pub fn point_median(points: &[Vector3<f64>]) -> Mediane<Vector3<f64>> {
    let n = points.len();
    assert!(n > 0, "médiane de zéro point");
    for (j, pj) in points.iter().enumerate() {
        let v: Vec<Vector3<f64>> = points.iter().map(|p| p - pj).collect();
        if weiszfeld_step(&v).is_none() {
            return (*pj, Some(j));
        }
    }
    let mut y = points[medoide(n, |a, b| (points[a] - points[b]).norm())];
    let exacte = None;
    for _ in 0..WEISZFELD_MAX_STEPS {
        let v: Vec<Vector3<f64>> = points.iter().map(|p| p - y).collect();
        let Some(pas) = weiszfeld_step(&v) else { break };
        y += pas;
        if pas.norm() < WEISZFELD_TOLERANCE {
            break;
        }
    }
    (y, exacte)
}

/// Rotation (partie 3 × 3) d'une pose rigide.
fn rotation_de(pose: &Matrix4<f64>) -> Rotation3<f64> {
    Rotation3::from_matrix(&pose.fixed_view::<3, 3>(0, 0).into_owned())
}

/// Pose rigide qui tourne de `r` autour de `centre` puis l'envoie en `position` : `x ↦ r · (x − centre) + position`.
fn pose_de(r: &Rotation3<f64>, centre: &Vector3<f64>, position: &Vector3<f64>) -> Matrix4<f64> {
    let mut m = Matrix4::identity();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(&(position - r * centre));
    m
}

/// Écart quadratique moyen (mm), sur les `points` d'une coupe, entre deux poses de cette coupe.
fn ecart_rms(points: &[Vector3<f64>], a: &Matrix4<f64>, b: &Matrix4<f64>) -> f64 {
    let s: f64 = points
        .iter()
        .map(|x| ((a - b) * x.push(1.0)).xyz().norm_squared())
        .sum();
    (s / points.len().max(1) as f64).sqrt()
}

/// **Fusionne K lancements** du coupe-coupe (mêmes stacks, graines différentes) selon `mode` ([`S2sConsensus::Median`] ou
/// [`S2sConsensus::Best`]).
///
/// 1. Alignement : chaque lancement `j` est remplacé par `G_j · M_j`, `G_j` étant le Kabsch qui envoie ses pixels de masque
///    ([`mask_points`]) sur ceux du premier lancement (l'identité exacte s'il en est à moins de 1e-12) ; le premier ne bouge pas.
/// 2. `Median` : pour chaque coupe, rotation médiane ([`rotation_median`]) et position médiane du centre géométrique de
///    la coupe ([`point_median`]) ; si les K poses d'une coupe sont exactement la même donnée médiane (même lancement
///    pour la rotation et la position), sa pose alignée est rendue telle quelle. Une coupe dont les rotations s'écartent de
///    plus de π/2 garde la pose de son médoïde et est signalée (`flagged`).
///    `Best` : les poses alignées du lancement médoïde (écart quadratique moyen moyen, sur les pixels de masque, aux K − 1 autres).
/// 3. Rapport par coupe : celui du lancement dont la pose de cette coupe est la plus proche de la pose rendue (écart
///    quadratique moyen sur ses pixels de masque, ou sur son centre si elle n'en a pas). Il ne sert qu'à l'exclusion de
///    coupes (`s2s_exclude_fraction`).
///
/// # Erreurs
/// [`SvrError::NoMask`] si un stack n'a pas de masque ; [`SvrError::PoseMismatch`] si un jeu de poses n'a pas la forme de
/// `stacks` ou si un rapport n'a pas une entrée par coupe.
///
/// # Panique
/// Si `runs` est vide ou si `mode` vaut [`S2sConsensus::None`].
pub fn fuse_runs(stacks: &[Stack], runs: &[AlignmentReport], mode: S2sConsensus) -> Result<Consensus, SvrError> {
    assert!(!runs.is_empty(), "fuse_runs : aucun lancement");
    assert!(mode != S2sConsensus::None, "fuse_runs : pas de fusion demandée");
    let nombre_coupes: usize = stacks.iter().map(|s| s.dim().2).sum();
    if runs
        .iter()
        .any(|r| !r.poses.matches(stacks) || r.slices.len() != nombre_coupes)
    {
        return Err(SvrError::PoseMismatch);
    }
    let points = mask_points(stacks)?;
    // 1. alignement sur le premier lancement
    let reference = placer(&points, &runs[0].poses);
    let alignments: Vec<Matrix4<f64>> = runs
        .iter()
        .enumerate()
        .map(|(j, r)| {
            if j == 0 {
                return Matrix4::identity();
            }
            let g = kabsch(&placer(&points, &r.poses), &reference);
            // à l'arrondi près de l'identité (lancements déjà dans le même repère) : l'identité exacte, pour que des poses identiques restent identiques au bit près
            if (g - Matrix4::identity()).abs().max() < ALIGNMENT_SNAP {
                Matrix4::identity()
            } else {
                g
            }
        })
        .collect();
    let mut alignes: Vec<SlicePoses> = runs.iter().map(|r| r.poses.clone()).collect();
    for (j, poses) in alignes.iter_mut().enumerate().skip(1) {
        for (si, stack) in stacks.iter().enumerate() {
            for k in 0..stack.dim().2 {
                let p = alignments[j] * runs[j].poses.get(si, k);
                poses.set(si, k, p);
            }
        }
    }
    // médoïde des lancements
    let ecart_lancements = |a: usize, b: usize| {
        let (mut s, mut n) = (0.0, 0usize);
        for (si, coupes) in points.iter().enumerate() {
            for (k, pts) in coupes.iter().enumerate() {
                let d = alignes[a].get(si, k) - alignes[b].get(si, k);
                s += pts.iter().map(|x| (d * x.push(1.0)).xyz().norm_squared()).sum::<f64>();
                n += pts.len();
            }
        }
        (s / n.max(1) as f64).sqrt()
    };
    let medoid_run = medoide(runs.len(), ecart_lancements);
    // 2. poses rendues
    let mut flagged = Vec::new();
    let poses = match mode {
        S2sConsensus::Best => alignes[medoid_run].clone(),
        _ => {
            let mut fusion = alignes[0].clone();
            for (si, stack) in stacks.iter().enumerate() {
                for coupe in stack.slices() {
                    let k = coupe.index();
                    let centre = coupe.geometric_center();
                    let candidates: Vec<&Matrix4<f64>> = alignes.iter().map(|p| p.get(si, k)).collect();
                    let rotations: Vec<Rotation3<f64>> = candidates.iter().map(|m| rotation_de(m)).collect();
                    let positions: Vec<Vector3<f64>> =
                        candidates.iter().map(|m| (*m * centre.push(1.0)).xyz()).collect();
                    let pose = match rotation_median(&rotations) {
                        None => {
                            flagged.push((si, k));
                            let pts = if points[si][k].is_empty() {
                                vec![centre]
                            } else {
                                points[si][k].clone()
                            };
                            *candidates[medoide(candidates.len(), |a, b| ecart_rms(&pts, candidates[a], candidates[b]))]
                        }
                        Some((r, exacte_r)) => {
                            let (p, exacte_p) = point_median(&positions);
                            match (exacte_r, exacte_p) {
                                (Some(a), Some(b)) if a == b => *candidates[a],
                                _ => pose_de(&r, &centre, &p),
                            }
                        }
                    };
                    fusion.set(si, k, pose);
                }
            }
            fusion
        }
    };
    // 3. rapport par coupe
    let mut slices: Vec<SliceReport> = Vec::with_capacity(nombre_coupes);
    let mut indice = 0;
    for (si, stack) in stacks.iter().enumerate() {
        for coupe in stack.slices() {
            let k = coupe.index();
            let pts = if points[si][k].is_empty() {
                vec![coupe.geometric_center()]
            } else {
                points[si][k].clone()
            };
            let plus_proche = (0..runs.len())
                .map(|j| (j, ecart_rms(&pts, alignes[j].get(si, k), poses.get(si, k))))
                .fold((0, f64::INFINITY), |m, (j, e)| if e < m.1 { (j, e) } else { m })
                .0;
            slices.push(runs[plus_proche].slices[indice].clone());
            indice += 1;
        }
    }
    Ok(Consensus {
        report: AlignmentReport { poses, slices },
        medoid_run,
        flagged,
        alignments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn rot(axe: [f64; 3], angle: f64) -> Rotation3<f64> {
        Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(Vector3::from(axe)), angle)
    }

    /// (2) Oracle : des rotations d'un même axe ; la médiane de Fréchet est la rotation d'angle égal à la médiane scalaire
    /// des angles (calculée à la main : 0,1 ; 0,15 ; 0,4 ; −0,2 ; 0,3 → 0,15).
    #[test]
    fn the_median_of_coaxial_rotations_is_the_scalar_median_of_angles() {
        let angles = [0.1, 0.15, 0.4, -0.2, 0.3];
        let rs: Vec<_> = angles.iter().map(|&a| rot([1.0, 2.0, -0.5], a)).collect();
        let (m, _) = rotation_median(&rs).unwrap();
        assert!(m.angle_to(&rot([1.0, 2.0, -0.5], 0.15)) < 1e-9);
        // quatre angles (médiane non unique entre 0,1 et 0,15) : la médiane est dans l'intervalle
        let (m4, _) = rotation_median(&rs[..4]).unwrap();
        let a = (rot([1.0, 2.0, -0.5], 0.0).inverse() * m4).scaled_axis().norm();
        assert!((0.1 - 1e-9..=0.15 + 1e-9).contains(&a), "angle {a}");
    }

    /// (2) K = 3 avec une rotation aberrante (60°) : la médiane est EXACTEMENT la paire identique (un des points de données) ;
    /// une moyenne pondérée de façon égale s'en écarterait d'environ 20°.
    #[test]
    fn the_median_ignores_one_outlying_rotation() {
        let r = rot([0.3, -1.0, 0.2], 0.2);
        let aberrante = r * rot([1.0, 0.0, 0.0], PI / 3.0);
        for ordre in [[r, r, aberrante], [aberrante, r, r], [r, aberrante, r]] {
            let (m, exacte) = rotation_median(&ordre).unwrap();
            assert_eq!(m, r);
            assert!(exacte.is_some());
        }
        // deux rotations proches et une aberrante : la médiane reste entre les deux proches, loin de l'aberrante
        let r2 = r * rot([0.0, 0.0, 1.0], 0.01);
        let (m, _) = rotation_median(&[r, r2, aberrante]).unwrap();
        assert!(m.angle_to(&r) <= 0.01 + 1e-9 && m.angle_to(&r2) <= 0.01 + 1e-9);
    }

    /// Médiane géométrique de R³ : oracle à la main (trois points alignés : le point du milieu ; quatre sommets d'un carré :
    /// son centre) ; un point aberrant ne la déplace pas au-delà de la paire confondue.
    #[test]
    fn the_point_median_matches_hand_computed_cases() {
        let p = |x, y, z| Vector3::new(x, y, z);
        let (m, exacte) = point_median(&[p(0.0, 0.0, 0.0), p(5.0, 0.0, 0.0), p(1.0, 0.0, 0.0)]);
        assert_eq!((m, exacte), (p(1.0, 0.0, 0.0), Some(2)));
        let (m, _) = point_median(&[
            p(1.0, 1.0, 3.0),
            p(-1.0, 1.0, 3.0),
            p(-1.0, -1.0, 3.0),
            p(1.0, -1.0, 3.0),
        ]);
        assert!((m - p(0.0, 0.0, 3.0)).norm() < 1e-9, "{m:?}");
        let (m, _) = point_median(&[p(2.0, 0.0, 0.0), p(2.0, 0.0, 0.0), p(100.0, 50.0, -3.0)]);
        assert_eq!(m, p(2.0, 0.0, 0.0));
        // triangle équilatéral : le point de Fermat est son centre (aucun angle ≥ 120°)
        let s = 3.0_f64.sqrt();
        let (m, _) = point_median(&[p(0.0, 0.0, 0.0), p(2.0, 0.0, 0.0), p(1.0, s, 0.0)]);
        assert!((m - p(1.0, s / 3.0, 0.0)).norm() < 1e-8, "{m:?}");
    }

    /// Deux rotations à plus de π/2 l'une de l'autre : la médiane n'est pas calculée (coupe signalée par `fuse_runs`).
    #[test]
    fn rotations_too_far_apart_have_no_median() {
        let r = rot([0.0, 0.0, 1.0], 0.0);
        assert!(rotation_median(&[r, r, rot([0.0, 0.0, 1.0], 1.7)]).is_none());
        assert!(rotation_median(&[r, r, rot([0.0, 0.0, 1.0], 1.5)]).is_some());
    }

    /// (3) Kabsch retrouve un mouvement rigide connu (écart < 1e-9), y compris d'un nuage presque plan, et rend une rotation propre quand la cible est le miroir de la source (correction de signe).
    #[test]
    fn kabsch_recovers_a_known_rigid_motion() {
        let g = pose_de(
            &rot([0.2, 1.0, 0.4], 0.7),
            &Vector3::new(3.0, -2.0, 1.0),
            &Vector3::new(-10.0, 4.0, 7.5),
        );
        let source: Vec<Vector3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64;
                Vector3::new((t * 0.37).sin() * 20.0, (t * 0.91).cos() * 15.0, 1e-3 * (t * 1.3).sin())
            })
            .collect();
        let cible: Vec<Vector3<f64>> = source.iter().map(|x| (g * x.push(1.0)).xyz()).collect();
        assert!((kabsch(&source, &cible) - g).abs().max() < 1e-9);
        // cible miroir d'un nuage non plan : la meilleure transformation orthogonale serait une réflexion ; Kabsch rend une rotation propre (déterminant +1)
        let epais: Vec<Vector3<f64>> = (0..40)
            .map(|i| {
                let t = i as f64;
                Vector3::new((t * 0.37).sin() * 20.0, (t * 0.91).cos() * 15.0, (t * 1.3).sin() * 10.0)
            })
            .collect();
        let miroir: Vec<Vector3<f64>> = epais.iter().map(|x| Vector3::new(-x.x, x.y, x.z)).collect();
        let r = kabsch(&epais, &miroir).fixed_view::<3, 3>(0, 0).into_owned();
        assert!((r.determinant() - 1.0).abs() < 1e-9);
    }

    /// `weiszfeld_step` avec l'estimation confondue avec la donnée du milieu : elle est la médiane (pas de pas) ;
    /// confondue avec une extrémité : le pas va vers l'intérieur.
    #[test]
    fn the_vardi_zhang_correction_stops_at_a_median_data_point() {
        let v = |x| Vector3::new(x, 0.0, 0.0);
        assert_eq!(weiszfeld_step(&[v(-1.0), v(0.0), v(2.0)]), None);
        let pas = weiszfeld_step(&[v(0.0), v(1.0), v(3.0)]).unwrap();
        assert!(pas.x > 0.0);
    }

    /// Le logarithme de SO(3) inverse l'exponentielle de nalgebra (Rodrigues) à 1e-9 près en relatif, de 1e-12 rad à presque π, y compris aux petits angles où un `acos` de la trace perdrait la moitié des chiffres.
    #[test]
    fn log_so3_inverts_exp() {
        for (i, a) in [1e-12, 1e-9, 1e-6, 1e-3, 0.1, 1.0, 2.0, 3.0, PI - 1e-5]
            .iter()
            .enumerate()
        {
            let axe = Vector3::new(0.3 + i as f64, -1.0, 0.5).normalize();
            let w = log_so3(&Rotation3::from_scaled_axis(axe * *a));
            let e = (w - axe * *a).norm() / a;
            assert!(e < 1e-9, "angle {a} : erreur relative {e:.2e}");
        }
        assert_eq!(log_so3(&Rotation3::identity()), Vector3::zeros());
        // à π exactement l'axe est défini au signe près
        let axe = Vector3::new(1.0, 2.0, 2.0) / 3.0;
        let w = log_so3(&Rotation3::from_scaled_axis(axe * PI));
        assert!((w - axe * PI).norm() < 1e-7 || (w + axe * PI).norm() < 1e-7, "{w:?}");
    }

    /// Triplets aléatoires de rotations voisines (±0,05 rad autour d'une base) : (a) quand la médiane n'est pas une donnée, elle est localement OPTIMALE (la somme des distances ne baisse dans aucune
    /// des six directions, à 1e-5 rad ; vérifié indépendamment de l'arrêt de Weiszfeld) ; (b) elle est équivariante : la médiane des `g · R_j` est `g ·` la médiane, à 1e-14 rad (1,6e-16 mesuré ;
    /// 3,1e-13 avec le logarithme de nalgebra par `acos`, ce qui justifie [`log_so3`]).
    #[test]
    fn the_rotation_median_is_stationary_and_equivariant() {
        let mut graine = 12345u64;
        let mut alea = || {
            graine = graine
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((graine >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let g = Rotation3::from_scaled_axis(Vector3::new(0.1, -0.12, 0.05));
        let (mut interieures, mut pire_eq) = (0, 0.0_f64);
        for _ in 0..200 {
            let base = Rotation3::from_scaled_axis(Vector3::new(alea(), alea(), alea()) * 0.3);
            let rs: Vec<_> = (0..3)
                .map(|_| base * Rotation3::from_scaled_axis(Vector3::new(alea(), alea(), alea()) * 0.05))
                .collect();
            let (m, exacte) = rotation_median(&rs).unwrap();
            if exacte.is_none() {
                interieures += 1;
                // optimalité locale : la somme des distances ne baisse dans aucune des six directions (pas de 1e-5 rad)
                let f = |q: &Rotation3<f64>| rs.iter().map(|r| log_so3(&(q.inverse() * r)).norm()).sum::<f64>();
                for k in 0..6 {
                    let mut d = Vector3::zeros();
                    d[k % 3] = if k < 3 { 1e-5 } else { -1e-5 };
                    let ecart = f(&(m * Rotation3::from_scaled_axis(d))) - f(&m);
                    assert!(
                        ecart > -1e-12,
                        "médiane non optimale : la direction {k} baisse de {:.2e}",
                        -ecart
                    );
                }
            }
            let (m2, _) = rotation_median(&rs.iter().map(|r| g * r).collect::<Vec<_>>()).unwrap();
            pire_eq = pire_eq.max(log_so3(&((g * m).inverse() * m2)).norm());
        }
        assert!(
            interieures > 50,
            "trop peu de médianes intérieures ({interieures}) : le test (a) ne porte pas"
        );
        assert!(pire_eq < 1e-14, "équivariance : {pire_eq:.2e} rad");
    }
}
