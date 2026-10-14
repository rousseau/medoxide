//! Estimation d'un champ de biais multiplicatif lisse **par stack**, sans volume intermédiaire, à partir des intersections de coupes.
//!
//! # Modèle
//!
//! L'image du stack `s` est le tissu multiplié par un champ lisse `b_s`, **fixe dans le repère du scanner** : il dépend de la position NOMINALE (en-tête) du pixel, pas de l'anatomie qui a bougé. Deux coupes de stacks différents qui se croisent (à leurs poses) voient le même
//! tissu le long de leur droite d'intersection ; les deux pixels d'un point de cette droite ont des positions nominales `x₁` et `x₂` différentes, donc
//!
//! `ln y₁ − ln y₂ = φ(x₁)·c_{s₁} − φ(x₂)·c_{s₂}`,
//!
//! où `φ` est une base de monômes des coordonnées nominales normalisées et `c_s` les coefficients du stack `s`. Toutes les paires donnent un système **linéaire** aux moindres carrés (équations normales accumulées, petite régularisation des coefficients non constants) ; le terme constant du
//! stack 0 est fixé à 0 : seuls les rapports de gain entre stacks sont identifiables. Référence Python : `scripts/prototype_champ_biais.py`.

use nalgebra::{DMatrix, DVector, Vector3, Vector4};

use crate::align::verifier_entrees;
use crate::recon::SlicePoses;
use crate::s2s::{plan_de, Plan};
use crate::{Stack, SvrError};

/// Réglages de l'estimation du champ de biais.
#[derive(Debug, Clone, Copy)]
pub struct BiasConfig {
    /// Degré total maximal des monômes de la base (3 : 20 coefficients par stack).
    pub degree: usize,
    /// Échelle (mm) de normalisation des coordonnées : `u = (x − centre) / scale_mm`.
    pub scale_mm: f64,
    /// Angle minimal (°) entre les plans de deux coupes pour qu'elles forment une paire.
    pub min_angle_deg: f64,
    /// Pas d'échantillonnage le long de la droite d'intersection (mm).
    pub step_mm: f64,
    /// Demi-longueur de la droite échantillonnée (mm).
    pub reach_mm: f64,
    /// Nombre minimal de points valides (dans les deux coupes, leurs masques, et au-dessus du seuil d'intensité) pour qu'une paire compte.
    pub min_points: usize,
    /// Nombre maximal de points gardés par paire, régulièrement espacés parmi les points valides.
    pub points_per_pair: usize,
    /// Seuil d'intensité d'un stack, en fraction de son 99ᵉ centile dans le masque : les points plus sombres (log bruité) sont ignorés.
    pub threshold_fraction: f64,
    /// Régularisation des coefficients non constants : `ridge² × (nombre d'équations)` ajouté à la diagonale des équations normales.
    pub ridge: f64,
}

impl Default for BiasConfig {
    fn default() -> Self {
        BiasConfig { degree: 3, scale_mm: 60.0, min_angle_deg: 30.0, step_mm: 0.5, reach_mm: 150.0, min_points: 20, points_per_pair: 40, threshold_fraction: 0.1, ridge: 1e-3 }
    }
}

/// Exposants `(a, b, c)` des monômes `uᵃ vᵇ wᶜ` de degré total ≤ `degree`, triés par degré puis par exposants : le premier est la constante `(0, 0, 0)`.
pub(crate) fn monomial_exponents(degree: usize) -> Vec<[usize; 3]> {
    let mut exposants = Vec::new();
    for a in 0..=degree {
        for b in 0..=degree {
            for c in 0..=degree {
                if a + b + c <= degree {
                    exposants.push([a, b, c]);
                }
            }
        }
    }
    exposants.sort_by_key(|e| (e[0] + e[1] + e[2], *e));
    exposants
}

/// Les logarithmes des champs de biais estimés, un polynôme par stack.
#[derive(Debug, Clone)]
pub struct BiasFields {
    exponents: Vec<[usize; 3]>,
    centre: Vector3<f64>,
    scale_mm: f64,
    /// `coefficients[stack][monôme]` ; le terme constant du stack 0 est nul.
    coefficients: Vec<Vec<f64>>,
    /// Incohérence d'intensité (écart quadratique moyen du log du rapport des intensités aux points d'intersection) avant et après correction, sur les points de l'estimation.
    inconsistency: (f64, f64),
}

impl BiasFields {
    /// Valeurs `φ(x)` de la base en une position nominale `x` (mm).
    fn basis(&self, x: &Vector3<f64>) -> Vec<f64> {
        let u = (x - self.centre) / self.scale_mm;
        self.exponents.iter().map(|e| u.x.powi(e[0] as i32) * u.y.powi(e[1] as i32) * u.z.powi(e[2] as i32)).collect()
    }

    /// Logarithme du champ de biais du stack `stack` à la position NOMINALE `x` (monde, mm, avant tout mouvement estimé). Le terme constant du stack 0 est 0.
    pub fn log_field(&self, stack: usize, x: &Vector3<f64>) -> f64 {
        self.basis(x).iter().zip(&self.coefficients[stack]).map(|(p, c)| p * c).sum()
    }

    /// Incohérence d'intensité entre stacks, avant et après la correction estimée (voir [`intensity_inconsistency`]), sur les points de l'estimation.
    pub fn inconsistency(&self) -> (f64, f64) {
        self.inconsistency
    }

    /// Réduction relative de l'incohérence d'intensité : `1 − après / avant` (0 : le champ estimé n'explique rien ; 1 : tout). Mesure de la PREUVE d'un biais : sans biais, le champ ne fait que s'ajuster au bruit d'anatomie et la réduction est de l'ordre du pour cent.
    pub fn inconsistency_reduction(&self) -> f64 {
        if self.inconsistency.0 > 0.0 {
            1.0 - self.inconsistency.1 / self.inconsistency.0
        } else {
            0.0
        }
    }

    /// Coefficients (un vecteur par stack) des logarithmes des champs.
    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.coefficients
    }
}

/// Percentile `q` (entre 0 et 1) par interpolation linéaire entre les statistiques d'ordre (comme `numpy.percentile`).
fn percentile(valeurs: &mut [f32], q: f64) -> f64 {
    valeurs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let position = q * (valeurs.len() - 1) as f64;
    let (bas, haut) = (position.floor() as usize, position.ceil() as usize);
    f64::from(valeurs[bas]) + (position - bas as f64) * (f64::from(valeurs[haut]) - f64::from(valeurs[bas]))
}

/// Valeur bilinéaire de la coupe `k` du stack en `(i, j)` fractionnaires (bords : pixel le plus proche).
fn bilinear(stack: &Stack, k: usize, i: f64, j: f64) -> f64 {
    let (nx, ny, _) = stack.dim();
    let (i, j) = (i.clamp(0.0, (nx - 1) as f64), j.clamp(0.0, (ny - 1) as f64));
    let (i0, j0) = (i.floor() as usize, j.floor() as usize);
    let (i1, j1) = ((i0 + 1).min(nx - 1), (j0 + 1).min(ny - 1));
    let (fi, fj) = (i - i0 as f64, j - j0 as f64);
    let d = stack.data();
    f64::from(d[[i0, j0, k]]) * (1.0 - fi) * (1.0 - fj) + f64::from(d[[i1, j0, k]]) * fi * (1.0 - fj) + f64::from(d[[i0, j1, k]]) * (1.0 - fi) * fj + f64::from(d[[i1, j1, k]]) * fi * fj
}

/// Un point d'intersection : le même tissu vu par deux coupes de stacks différents (positions NOMINALES des deux pixels et intensités).
struct Sample {
    s1: usize,
    x1: Vector3<f64>,
    y1: f64,
    s2: usize,
    x2: Vector3<f64>,
    y2: f64,
}

/// Les points d'intersection de toutes les paires de coupes de stacks différents, et le centre de normalisation des positions (barycentre des positions nominales des pixels de masque de tous les stacks).
fn collect_samples(stacks: &[Stack], poses: &SlicePoses, config: &BiasConfig) -> Result<(Vec<Sample>, Vector3<f64>), SvrError> {
    if stacks.len() < 2 {
        return Err(SvrError::NotEnoughStacks);
    }
    verifier_entrees(stacks, poses)?;
    let (mut seuils, mut somme, mut n_pixels) = (Vec::new(), Vector3::zeros(), 0usize);
    for stack in stacks {
        let masque = stack.brain_mask().expect("vérifié").voxels();
        let mut valeurs: Vec<f32> = stack.data().iter().zip(masque.iter()).filter(|(_, &m)| m).map(|(&v, _)| v).collect();
        seuils.push(config.threshold_fraction * percentile(&mut valeurs, 0.99));
        for ((i, j, k), _) in masque.indexed_iter().filter(|(_, &m)| m) {
            somme += (stack.affine() * Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
            n_pixels += 1;
        }
    }
    let plans: Vec<Vec<Plan>> = stacks.iter().enumerate().map(|(s, stack)| (0..stack.dim().2).map(|k| plan_de(stack, k, poses.get(s, k))).collect()).collect();
    let sin_min = config.min_angle_deg.to_radians().sin();
    let n_t = (2.0 * config.reach_mm / config.step_mm).round() as usize + 1;
    let mut points = Vec::new();
    for s1 in 0..stacks.len() {
        for s2 in (s1 + 1)..stacks.len() {
            let (m1, m2) = (stacks[s1].brain_mask().expect("vérifié").voxels(), stacks[s2].brain_mask().expect("vérifié").voxels());
            let ((nx1, ny1, _), (nx2, ny2, _)) = (stacks[s1].dim(), stacks[s2].dim());
            for (k1, p1) in plans[s1].iter().enumerate() {
                for (k2, p2) in plans[s2].iter().enumerate() {
                    let d = p1.normale.cross(&p2.normale);
                    if d.norm() < sin_min {
                        continue;
                    }
                    let x0 = (p1.normale.dot(&p1.centre) * p2.normale.cross(&d) + p2.normale.dot(&p2.centre) * d.cross(&p1.normale)) / d.norm_squared();
                    let dh = d / d.norm();
                    let mut valides = Vec::new();
                    for it in 0..n_t {
                        let p = (x0 + (-config.reach_mm + it as f64 * config.step_mm) * dh).push(1.0);
                        let (a, b) = (p1.inverse * p, p2.inverse * p);
                        if a.x < 0.0 || a.x > (nx1 - 1) as f64 || a.y < 0.0 || a.y > (ny1 - 1) as f64 || b.x < 0.0 || b.x > (nx2 - 1) as f64 || b.y < 0.0 || b.y > (ny2 - 1) as f64 {
                            continue;
                        }
                        if !m1[[a.x.round() as usize, a.y.round() as usize, k1]] || !m2[[b.x.round() as usize, b.y.round() as usize, k2]] {
                            continue;
                        }
                        let (y1, y2) = (bilinear(&stacks[s1], k1, a.x, a.y), bilinear(&stacks[s2], k2, b.x, b.y));
                        if y1 > seuils[s1] && y2 > seuils[s2] {
                            valides.push((a, b, y1, y2));
                        }
                    }
                    if valides.len() < config.min_points {
                        continue;
                    }
                    let garde = config.points_per_pair.min(valides.len());
                    for g in 0..garde {
                        let (a, b, y1, y2) = valides[g * valides.len() / garde];
                        // positions NOMINALES des deux pixels (affine d'en-tête, sans la pose)
                        let x1 = (stacks[s1].affine() * Vector4::new(a.x, a.y, k1 as f64, 1.0)).xyz();
                        let x2 = (stacks[s2].affine() * Vector4::new(b.x, b.y, k2 as f64, 1.0)).xyz();
                        points.push(Sample { s1, x1, y1, s2, x2, y2 });
                    }
                }
            }
        }
    }
    if points.is_empty() {
        return Err(SvrError::NoIntersections);
    }
    Ok((points, somme / n_pixels as f64))
}

/// **Estime le champ de biais de chaque stack** aux poses `poses` (voir le modèle en tête de module).
///
/// Le champ n'est identifiable qu'**à un champ commun près, attaché à l'anatomie** : ajouter à tous les stacks un même champ lisse de la position VRAIE ne change aucun rapport d'intensité entre coupes qui se croisent (et un polynôme reste un polynôme après un mouvement rigide). Seule l'incohérence
/// entre stacks est donc estimée, ce qui suffit pour le recalage ; elle se mesure par [`intensity_inconsistency`], pas par la ressemblance de `log_field` à un champ vrai.
///
/// # Erreurs
/// [`SvrError::NotEnoughStacks`] avec moins de deux stacks ; [`SvrError::NoMask`] ou [`SvrError::PoseMismatch`] si les entrées sont invalides ; [`SvrError::NoIntersections`] si aucune paire de coupes ne donne assez de points valides.
pub fn estimate_bias_fields(stacks: &[Stack], poses: &SlicePoses, config: &BiasConfig) -> Result<BiasFields, SvrError> {
    let (points, centre) = collect_samples(stacks, poses, config)?;
    let exposants = monomial_exponents(config.degree);
    let (ns, nb) = (stacks.len(), exposants.len());
    let champs = BiasFields { exponents: exposants, centre, scale_mm: config.scale_mm, coefficients: vec![vec![0.0; nb]; ns], inconsistency: (0.0, 0.0) };
    let (mut ata, mut atb) = (DMatrix::<f64>::zeros(ns * nb, ns * nb), DVector::<f64>::zeros(ns * nb));
    for p in &points {
        let (f1, f2) = (champs.basis(&p.x1), champs.basis(&p.x2));
        let cible = p.y1.ln() - p.y2.ln();
        for u in 0..nb {
            atb[p.s1 * nb + u] += f1[u] * cible;
            atb[p.s2 * nb + u] -= f2[u] * cible;
            for v in 0..nb {
                ata[(p.s1 * nb + u, p.s1 * nb + v)] += f1[u] * f1[v];
                ata[(p.s2 * nb + u, p.s2 * nb + v)] += f2[u] * f2[v];
                ata[(p.s1 * nb + u, p.s2 * nb + v)] -= f1[u] * f2[v];
                ata[(p.s2 * nb + u, p.s1 * nb + v)] -= f2[u] * f1[v];
            }
        }
    }
    // inconnues : tous les coefficients sauf le terme constant du stack 0 ; régularisation des seuls coefficients non constants
    let libres: Vec<usize> = (1..ns * nb).collect();
    let mut a = DMatrix::<f64>::zeros(libres.len(), libres.len());
    let mut b = DVector::<f64>::zeros(libres.len());
    for (r, &c) in libres.iter().enumerate() {
        b[r] = atb[c];
        for (q, &d) in libres.iter().enumerate() {
            a[(r, q)] = ata[(c, d)];
        }
        if c % nb != 0 {
            a[(r, r)] += config.ridge * config.ridge * points.len() as f64;
        }
    }
    let solution = a.cholesky().ok_or(SvrError::NoIntersections)?.solve(&b);
    let mut coefficients = vec![vec![0.0; nb]; ns];
    for (r, &c) in libres.iter().enumerate() {
        coefficients[c / nb][c % nb] = solution[r];
    }
    let mut ajuste = BiasFields { coefficients, ..champs };
    let (mut avant, mut apres) = (0.0, 0.0);
    for p in &points {
        let cible = p.y1.ln() - p.y2.ln();
        avant += cible * cible;
        apres += (cible - (ajuste.log_field(p.s1, &p.x1) - ajuste.log_field(p.s2, &p.x2))).powi(2);
    }
    ajuste.inconsistency = ((avant / points.len() as f64).sqrt(), (apres / points.len() as f64).sqrt());
    Ok(ajuste)
}

/// **Incohérence d'intensité entre stacks** : écart quadratique moyen, sur les points d'intersection, du logarithme du rapport des intensités de deux coupes qui se croisent, après correction par `champs` (`None` : sans correction). Nulle si les stacks sont parfaitement cohérents.
/// Les points sont ceux de l'estimation (mêmes réglages `config`).
///
/// # Erreurs
/// Les mêmes que [`estimate_bias_fields`].
pub fn intensity_inconsistency(stacks: &[Stack], poses: &SlicePoses, config: &BiasConfig, champs: Option<&BiasFields>) -> Result<f64, SvrError> {
    let (points, _) = collect_samples(stacks, poses, config)?;
    let somme: f64 = points
        .iter()
        .map(|p| {
            let correction = champs.map_or(0.0, |c| c.log_field(p.s1, &p.x1) - c.log_field(p.s2, &p.x2));
            (p.y1.ln() - p.y2.ln() - correction).powi(2)
        })
        .sum();
    Ok((somme / points.len() as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Base polynomiale (valeurs exactes) : degré 3 : 1 + 3 + 6 + 10 = 20 monômes, la constante en premier, puis les trois monômes de degré 1 dans l'ordre `w`, `v`, `u` ; `φ` et `log_field` en un point où `u = 1, v = 0, w = 1` valent la somme des coefficients de `1`, `w` et `u`.
    #[test]
    fn the_polynomial_basis_is_exact() {
        let e = monomial_exponents(3);
        assert_eq!(e.len(), 20);
        assert_eq!(e[0], [0, 0, 0]);
        assert_eq!(&e[1..4], &[[0, 0, 1], [0, 1, 0], [1, 0, 0]]);
        assert_eq!(e.iter().filter(|m| m.iter().sum::<usize>() == 2).count(), 6);
        assert_eq!(monomial_exponents(1).len(), 4);
        assert_eq!(monomial_exponents(0), vec![[0, 0, 0]]);
        let mut coefficients = vec![0.0; 20];
        (coefficients[0], coefficients[1], coefficients[2], coefficients[3]) = (0.5, 0.25, 7.0, 0.125);
        let champs = BiasFields { exponents: e, centre: Vector3::new(1.0, 2.0, 3.0), scale_mm: 10.0, coefficients: vec![vec![0.0; 20], coefficients], inconsistency: (0.5, 0.125) };
        assert_eq!(champs.log_field(1, &Vector3::new(11.0, 2.0, 13.0)), 0.5 + 0.25 + 0.125);
        assert_eq!(champs.log_field(0, &Vector3::new(11.0, 2.0, 13.0)), 0.0);
        assert_eq!(champs.log_field(1, &Vector3::new(1.0, 2.0, 3.0)), 0.5);
        assert_eq!(champs.inconsistency(), (0.5, 0.125));
        assert_eq!(champs.inconsistency_reduction(), 0.75);
        assert_eq!(BiasFields { inconsistency: (0.0, 0.0), ..champs }.inconsistency_reduction(), 0.0);
    }

    /// Percentile à interpolation linéaire (comme `numpy.percentile`) : médiane de 1..5 = 3, 99ᵉ centile de 0..100 = 99, 25ᵉ centile de 0, 10, 20, 30 = 7,5.
    #[test]
    fn the_percentile_interpolates_linearly() {
        assert_eq!(percentile(&mut [5.0, 1.0, 3.0, 2.0, 4.0], 0.5), 3.0);
        assert_eq!(percentile(&mut (0..=100).map(|v| v as f32).collect::<Vec<_>>(), 0.99), 99.0);
        assert_eq!(percentile(&mut [0.0, 10.0, 20.0, 30.0], 0.25), 7.5);
    }
}
