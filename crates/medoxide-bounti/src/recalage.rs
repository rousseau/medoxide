//! Recalage affine du gabarit gauche/droite sur la carte gauche/droite prédite par le petit réseau (le `register -model Affine` de MIRTK dans la chaîne officielle de Multi-BOUNTI, voir `docs/MULTIBOUNTI.md`, sous-étape M4).
//!
//! Principe, repris de ce que fait MIRTK (sortie détaillée de `register`, qui fait presque tout son travail au premier niveau) :
//! 1. les deux cartes sont floutées (écart-type : la moitié du pas) et le gabarit est échantillonné sur une grille de [`PAS_MM`] mm ;
//! 2. la similarité est l'**information mutuelle normalisée** (NMI) de ces valeurs échantillonnées, `(H(T) + H(S)) / H(T, S)`, avec un histogramme conjoint de [`CLASSES`] classes ;
//! 3. les douze paramètres (translation, rotation, échelles, cisaillements : voir [`crate::affine`]) sont améliorés par **descente de gradient** : le gradient est estimé par différences centrées, puis une recherche de pas le long de la direction de plus forte décroissance.
//!
//! La transformation rendue envoie un point du repère du gabarit vers celui du sujet, comme celle d'un `.dof` (voir [`crate::affine::Parametres::matrice`]). Les réglages sont ceux figés sur les volumes de mise au point (`docs/MULTIBOUNTI.md`).
use crate::affine::Parametres;
use crate::geometrie::Volume;
use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use ndarray::{Array3, Axis};

/// Pas de l'échantillonnage du gabarit (mm), celui du premier niveau de MIRTK.
pub const PAS_MM: f64 = 3.9;
/// Nombre de classes de l'histogramme conjoint.
pub const CLASSES: usize = 16;
/// Gain relatif d'énergie (par rapport à l'énergie de départ) sous lequel la descente s'arrête.
pub const TOLERANCE: f64 = 1e-5;
/// Nombre maximal d'itérations de la descente.
pub const ITERATIONS_MAX: usize = 100;
/// Demi-largeur des différences centrées (mm pour une translation, degrés pour un angle, pour cent pour une échelle).
const DIFFERENCE: f64 = 0.5;
/// Nombre maximal d'essais de longueur de pas dans une recherche, et nombre de doublements permis.
const ESSAIS_MAX: usize = 25;
const DOUBLEMENTS_MAX: usize = 4;
/// Les valeurs des cartes vont de 0 (fond) à 2 (étiquette droite).
const VALEUR_MAX: f32 = 2.0;
/// Volume du masque du cerveau (mL) au-dessus duquel le départ est l'identité ; en dessous, une réduction à 40 %.
pub const VOLUME_DEPART_ML: u32 = 170;

/// Transformation de départ du recalage selon le volume du masque du cerveau arrondi en mL (voir `guess-ga`) : l'identité rigide au-dessus de [`VOLUME_DEPART_ML`], sinon une mise à l'échelle de 40 % (`i-very-early-ga.dof` du gabarit).
pub fn depart_selon_le_volume(volume_ml: u32) -> Matrix4<f64> {
    let mut parametres = Parametres::identite();
    if volume_ml <= VOLUME_DEPART_ML {
        parametres.echelle = [40.0; 3];
    }
    parametres.matrice()
}

/// Indice du voxel qui prolonge une image de `n` voxels par symétrie au bord (`d c b a | a b c d | d c b a`, le mode `reflect` de SciPy) quand on sort de `0..n`.
fn reflechir(i: isize, n: isize) -> usize {
    let periode = 2 * n;
    let r = i.rem_euclid(periode);
    (if r < n { r } else { periode - 1 - r }) as usize
}

/// Flou gaussien d'écart-type `sigma` voxels sur un axe (noyau tronqué à `4 σ`, normalisé).
fn flou_sur_un_axe(a: &Array3<f32>, axe: usize, sigma: f64) -> Array3<f32> {
    if sigma <= 0.0 {
        return a.clone();
    }
    let rayon = (4.0 * sigma + 0.5) as isize;
    let mut noyau: Vec<f64> = (-rayon..=rayon)
        .map(|x| (-0.5 * (x as f64 / sigma).powi(2)).exp())
        .collect();
    let somme: f64 = noyau.iter().sum();
    noyau.iter_mut().for_each(|w| *w /= somme);
    let n = a.shape()[axe] as isize;
    let mut sortie = Array3::<f32>::zeros(a.raw_dim());
    for (lecture, mut ecriture) in a.lanes(Axis(axe)).into_iter().zip(sortie.lanes_mut(Axis(axe))) {
        for i in 0..n {
            let mut s = 0.0_f64;
            for (k, w) in noyau.iter().enumerate() {
                s += w * f64::from(lecture[reflechir(i + k as isize - rayon, n)]);
            }
            ecriture[i as usize] = s as f32;
        }
    }
    sortie
}

/// Flou gaussien séparable, d'écart-type `sigma` voxels sur chaque axe.
fn flou_gaussien(a: &Array3<f32>, sigma: [f64; 3]) -> Array3<f32> {
    let mut resultat = flou_sur_un_axe(a, 0, sigma[0]);
    resultat = flou_sur_un_axe(&resultat, 1, sigma[1]);
    flou_sur_un_axe(&resultat, 2, sigma[2])
}

/// Information mutuelle normalisée `(H(T) + H(S)) / H(T, S)` de deux listes de valeurs de même longueur dans `[0, VALEUR_MAX]`, par un histogramme conjoint de `CLASSES` classes où chaque échantillon se répartit linéairement entre les deux classes les plus proches sur chaque axe
/// (la fonction est ainsi lisse en les paramètres de la transformation, ce qui permet d'en estimer le gradient par différences finies). Vaut 2 pour deux listes identiques et non constantes dont les valeurs sont au centre des classes (entre deux classes, la répartition linéaire fait un peu baisser la valeur), 1 si l'une est constante.
fn nmi(t: &[f32], s: &[f32]) -> f64 {
    let mut h = [0.0_f64; CLASSES * CLASSES];
    let position = |x: f32| -> (usize, f64) {
        let u = (f64::from(x) / f64::from(VALEUR_MAX) * CLASSES as f64 - 0.5).clamp(0.0, CLASSES as f64 - 1.0 - 1e-9);
        let i = u.floor();
        (i as usize, u - i)
    };
    for (&a, &b) in t.iter().zip(s.iter()) {
        let ((i, fi), (j, fj)) = (position(a), position(b));
        for (di, wi) in [(0, 1.0 - fi), (1, fi)] {
            for (dj, wj) in [(0, 1.0 - fj), (1, fj)] {
                h[(i + di).min(CLASSES - 1) * CLASSES + (j + dj).min(CLASSES - 1)] += wi * wj;
            }
        }
    }
    let total: f64 = h.iter().sum();
    let entropie = |valeurs: &mut dyn Iterator<Item = f64>| -> f64 {
        valeurs
            .map(|v| v / total)
            .filter(|&p| p > 0.0)
            .map(|p| -p * p.ln())
            .sum()
    };
    let conjointe = entropie(&mut h.iter().copied());
    let marginale_t = entropie(&mut (0..CLASSES).map(|i| h[i * CLASSES..(i + 1) * CLASSES].iter().sum::<f64>()));
    let marginale_s = entropie(&mut (0..CLASSES).map(|j| (0..CLASSES).map(|i| h[i * CLASSES + j]).sum::<f64>()));
    (marginale_t + marginale_s) / conjointe.max(1e-12)
}

/// Transformation affine définie par les douze paramètres `p` (translation, rotation, échelles en pour cent, cisaillements : voir [`Parametres`]) **autour du point** `centre` : ce point n'est déplacé que par la translation.
/// Les paramètres sont ainsi peu corrélés (une rotation ne déplace pas le gabarit loin de lui-même), ce qui rend la descente bien conditionnée.
fn autour_du_centre(p: &[f64; 12], centre: &Vector3<f64>) -> Matrix4<f64> {
    let parametres = Parametres {
        translation: [0.0; 3],
        rotation: [p[3], p[4], p[5]],
        echelle: [p[6], p[7], p[8]],
        cisaillement: [p[9], p[10], p[11]],
    };
    let m: Matrix3<f64> = parametres.matrice().fixed_view::<3, 3>(0, 0).into_owned();
    let mut a = Matrix4::identity();
    a.fixed_view_mut::<3, 3>(0, 0).copy_from(&m);
    let decalage = centre + Vector3::new(p[0], p[1], p[2]) - m * centre;
    a.fixed_view_mut::<3, 1>(0, 3).copy_from(&decalage);
    a
}

/// Paramètres de l'identité (échelles à 100 %).
const IDENTITE: [f64; 12] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0];

/// Recale `gabarit` sur `sujet` (cartes gauche/droite : 0 fond, 1 gauche, 2 droite) à partir de `depart` : rend la transformation affine qui envoie un point du repère du gabarit vers celui du sujet (voir le module).
/// Rend `depart` si le gabarit est vide.
pub fn recaler(gabarit: &Volume, sujet: &Volume, depart: &Matrix4<f64>) -> Matrix4<f64> {
    let Some((minimum, maximum)) = gabarit.boite_englobante(0.0) else {
        return *depart;
    };
    let voxel = |v: &Volume| -> Vector3<f64> { Vector3::from_fn(|axe, _| v.affine.fixed_view::<3, 1>(0, axe).norm()) };
    let (voxel_gabarit, voxel_sujet) = (voxel(gabarit), voxel(sujet));
    let pas = (PAS_MM / voxel_gabarit[0]).round().max(1.0) as usize;
    let sigma_gabarit = Vector3::from_fn(|axe, _| PAS_MM / voxel_gabarit[axe] / 2.0);
    let sigma_sujet = Vector3::from_fn(|axe, _| PAS_MM / voxel_sujet[axe] / 2.0);

    // Le gabarit est recadré avant d'être flouté : hors de la boîte des voxels non nuls élargie du rayon du noyau, le flou vaut 0 de toute façon.
    let rayon = (4.0 * sigma_gabarit.max() + 0.5) as usize;
    let marge = rayon + 2 * pas + 1;
    let dimensions = gabarit.dimensions();
    let debut: [usize; 3] = std::array::from_fn(|a| minimum[a].saturating_sub(marge));
    let fin: [usize; 3] = std::array::from_fn(|a| (maximum[a] + marge).min(dimensions[a] - 1));
    let recadre = gabarit.recadrer(debut, fin);
    let flou_gabarit = Volume {
        donnees: flou_gaussien(&recadre.donnees, [sigma_gabarit[0], sigma_gabarit[1], sigma_gabarit[2]]),
        affine: recadre.affine,
    };
    let flou_sujet = Volume {
        donnees: flou_gaussien(&sujet.donnees, [sigma_sujet[0], sigma_sujet[1], sigma_sujet[2]]),
        affine: sujet.affine,
    };

    // points d'échantillonnage : voxels du gabarit (indices d'origine) tous les `pas` voxels sur la boîte élargie de 2 pas
    let bas: [usize; 3] = std::array::from_fn(|a| minimum[a].saturating_sub(2 * pas));
    let haut: [usize; 3] = std::array::from_fn(|a| (maximum[a] + 2 * pas).min(dimensions[a]));
    let mut points: Vec<Vector4<f64>> = Vec::new();
    let mut valeurs_gabarit: Vec<f32> = Vec::new();
    for i in (bas[0]..haut[0]).step_by(pas) {
        for j in (bas[1]..haut[1]).step_by(pas) {
            for k in (bas[2]..haut[2]).step_by(pas) {
                points.push(gabarit.affine * Vector4::new(i as f64, j as f64, k as f64, 1.0));
                valeurs_gabarit.push(flou_gabarit.donnees[[i - debut[0], j - debut[1], k - debut[2]]]);
            }
        }
    }
    let positions_non_nulles: Vec<Vector3<f64>> = gabarit
        .donnees
        .indexed_iter()
        .filter(|(_, &v)| v > 0.0)
        .map(|((i, j, k), _)| (gabarit.affine * Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz())
        .collect();
    let centre = positions_non_nulles.iter().sum::<Vector3<f64>>() / positions_non_nulles.len() as f64;

    let vers_voxels_sujet = sujet.affine.try_inverse().expect("affine inversible");
    let mut valeurs_sujet = vec![0.0_f32; points.len()];
    let mut energie = |p: &[f64; 12]| -> f64 {
        let transformation = vers_voxels_sujet * depart * autour_du_centre(p, &centre);
        for (valeur, point) in valeurs_sujet.iter_mut().zip(points.iter()) {
            *valeur = flou_sujet.echantillonner_voxel(&(transformation * point).xyz(), 0.0);
        }
        -nmi(&valeurs_gabarit, &valeurs_sujet)
    };

    let mut p = IDENTITE;
    let mut e = energie(&p);
    let e0 = e;
    for _ in 0..ITERATIONS_MAX {
        let mut gradient = [0.0_f64; 12];
        for (k, g) in gradient.iter_mut().enumerate() {
            let (mut plus, mut moins) = (p, p);
            plus[k] += DIFFERENCE;
            moins[k] -= DIFFERENCE;
            *g = (energie(&plus) - energie(&moins)) / (2.0 * DIFFERENCE);
        }
        let norme = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        if norme < 1e-9 {
            break;
        }
        // direction de plus forte décroissance, de déplacement maximal 1 sur un paramètre
        let direction: [f64; 12] = std::array::from_fn(|k| -gradient[k] / norme);
        let vers = |pas: f64| -> [f64; 12] { std::array::from_fn(|k| p[k] + pas * direction[k]) };
        let mut pas = 1.0_f64;
        let mut retenu: Option<(f64, f64)> = None;
        for essai in 0..ESSAIS_MAX {
            let nouvelle = energie(&vers(pas));
            if nouvelle < e {
                retenu = Some((pas, nouvelle));
                if essai >= DOUBLEMENTS_MAX {
                    break;
                }
                pas *= 2.0;
            } else if retenu.is_none() {
                pas /= 2.0;
                continue;
            } else {
                break;
            }
            if pas > 20.0 {
                break;
            }
        }
        let Some((pas, nouvelle)) = retenu else {
            break;
        };
        p = vers(pas);
        let gain = e - nouvelle;
        e = nouvelle;
        if gain < TOLERANCE * e0.abs() {
            break;
        }
    }
    depart * autour_du_centre(&p, &centre)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L'indice qui prolonge l'image par symétrie au bord : `d c b a | a b c d | d c b a` pour n = 4.
    #[test]
    fn the_border_reflection_matches_the_scipy_reflect_mode() {
        let attendu_pour = |i: isize| match i {
            -5 => 3,
            -4 => 3,
            -3 => 2,
            -2 => 1,
            -1 => 0,
            0..=3 => i as usize,
            4 => 3,
            5 => 2,
            6 => 1,
            7 => 0,
            8 => 0,
            _ => unreachable!(),
        };
        for i in -5..=8 {
            assert_eq!(reflechir(i, 4), attendu_pour(i), "indice {i}");
        }
    }

    /// Le flou conserve une image constante, la somme d'une impulsion (loin des bords), et l'étale de façon symétrique ; un écart-type nul ne change rien.
    #[test]
    fn the_gaussian_blur_preserves_constants_and_mass_and_is_symmetric() {
        let constante = Array3::from_elem((9, 9, 9), 3.0_f32);
        let floue = flou_gaussien(&constante, [1.0, 2.0, 0.5]);
        assert!(floue.iter().all(|&v| (v - 3.0).abs() < 1e-5));
        let mut impulsion = Array3::<f32>::zeros((31, 31, 31));
        impulsion[[15, 15, 15]] = 1.0;
        let floue = flou_gaussien(&impulsion, [1.5, 1.5, 1.5]);
        assert!((floue.sum() - 1.0).abs() < 1e-5);
        assert!((floue[[14, 15, 15]] - floue[[16, 15, 15]]).abs() < 1e-7);
        assert!((floue[[15, 12, 15]] - floue[[15, 18, 15]]).abs() < 1e-7);
        assert!(floue[[15, 15, 15]] > floue[[14, 15, 15]]);
        assert_eq!(flou_gaussien(&impulsion, [0.0; 3]), impulsion);
    }

    /// L'information mutuelle normalisée vaut 2 pour deux listes identiques et non constantes dont les valeurs sont au centre des classes, 1 si une liste est constante, un peu plus de 1 pour deux listes indépendantes, et reste entre les deux pour deux listes partiellement liées.
    #[test]
    fn the_normalised_mutual_information_orders_dependent_and_independent_values() {
        // centre de la classe i : (i + 0,5) × 2 / CLASSES
        let centre = |i: usize| (i as f32 + 0.5) * 2.0 / CLASSES as f32;
        let t: Vec<f32> = (0..4000).map(|i| centre(2 * (i % 3))).collect();
        assert!((nmi(&t, &t) - 2.0).abs() < 1e-9);
        let constante = vec![centre(3); t.len()];
        assert!((nmi(&t, &constante) - 1.0).abs() < 1e-9);
        // période différente : valeurs indépendantes en pratique
        let autre: Vec<f32> = (0..4000).map(|i| centre(2 * (i % 7))).collect();
        let independante = nmi(&t, &autre);
        assert!(independante > 1.0 && independante < 1.01, "{independante}");
        // une valeur sur dix remplacée : encore liée, mais moins que l'identité
        let abimee: Vec<f32> = t
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 10 == 0 { centre(13) } else { v })
            .collect();
        let partielle = nmi(&t, &abimee);
        assert!(partielle < 2.0 && partielle > independante, "{partielle}");
    }

    /// Les paramètres de l'identité redonnent l'identité ; une translation de t déplace le centre de t ; une échelle laisse le centre en place ; autour d'un centre, une rotation ne le déplace pas.
    #[test]
    fn the_transformation_around_a_centre_leaves_that_centre_fixed() {
        let centre = Vector3::new(10.0, -20.0, 30.0);
        assert!((autour_du_centre(&IDENTITE, &centre) - Matrix4::identity()).abs().max() < 1e-12);
        let mut p = IDENTITE;
        p[0] = 1.0;
        p[1] = -2.0;
        let deplace = (autour_du_centre(&p, &centre) * Vector4::new(centre.x, centre.y, centre.z, 1.0)).xyz();
        assert!((deplace - (centre + Vector3::new(1.0, -2.0, 0.0))).norm() < 1e-12);
        let mut p = IDENTITE;
        p[3] = 20.0;
        p[4] = -10.0;
        p[5] = 15.0;
        p[6] = 70.0;
        p[9] = 5.0;
        let fixe = (autour_du_centre(&p, &centre) * Vector4::new(centre.x, centre.y, centre.z, 1.0)).xyz();
        assert!((fixe - centre).norm() < 1e-12);
        assert!((autour_du_centre(&p, &centre)[(0, 0)]).abs() > 0.1);
    }

    /// Départ : identité au-dessus de 170 mL, réduction à 40 % sinon (170 mL compris).
    #[test]
    fn the_start_depends_on_the_brain_volume() {
        assert!((depart_selon_le_volume(171) - Matrix4::identity()).abs().max() < 1e-15);
        let petit = depart_selon_le_volume(170);
        assert!(
            (petit[(0, 0)] - 0.4).abs() < 1e-15
                && (petit[(1, 1)] - 0.4).abs() < 1e-15
                && (petit[(2, 2)] - 0.4).abs() < 1e-15
        );
        assert_eq!(depart_selon_le_volume(0), petit);
    }

    /// Carte gauche/droite synthétique : ellipsoïde de demi-axes (35, 25, 28) mm centré en `centre`, étiquette 1 pour x < centre.x, 2 sinon ; voxels de `voxel` mm, `n`³ voxels, centrée sur l'origine ;
    /// `transformation` envoie le repère du volume vers celui de l'ellipsoïde (identité pour le gabarit).
    fn carte(n: usize, voxel: f64, transformation: &Matrix4<f64>) -> Volume {
        let mut affine = Matrix4::identity();
        for a in 0..3 {
            affine[(a, a)] = voxel;
            affine[(a, 3)] = -voxel * (n as f64 - 1.0) / 2.0;
        }
        let mut donnees = Array3::<f32>::zeros((n, n, n));
        for ((i, j, k), v) in donnees.indexed_iter_mut() {
            let x = (transformation * affine * Vector4::new(i as f64, j as f64, k as f64, 1.0)).xyz();
            if (x.x / 35.0).powi(2) + (x.y / 25.0).powi(2) + (x.z / 28.0).powi(2) <= 1.0 {
                *v = if x.x < 0.0 { 1.0 } else { 2.0 };
            }
        }
        Volume { donnees, affine }
    }

    /// Écart moyen, en mm, entre deux transformations sur les voxels non nuls du gabarit.
    fn ecart_moyen(gabarit: &Volume, a: &Matrix4<f64>, b: &Matrix4<f64>) -> f64 {
        let points: Vec<Vector4<f64>> = gabarit
            .donnees
            .indexed_iter()
            .filter(|(_, &v)| v > 0.0)
            .map(|((i, j, k), _)| gabarit.affine * Vector4::new(i as f64, j as f64, k as f64, 1.0))
            .collect();
        points.iter().map(|p| ((a - b) * p).xyz().norm()).sum::<f64>() / points.len() as f64
    }

    /// Retrouve une translation connue de (5, −4, 3) mm entre deux cartes synthétiques (voxels de 2 mm), à partir de l'identité : l'écart moyen à la vraie transformation doit passer sous 1 mm.
    #[test]
    fn a_known_translation_is_recovered_on_synthetic_maps() {
        let vraie = Parametres {
            translation: [5.0, -4.0, 3.0],
            ..Parametres::identite()
        }
        .matrice();
        let gabarit = carte(64, 2.0, &Matrix4::identity());
        // le sujet à la position p vaut le gabarit à l'antécédent : sujet(vraie · p) = gabarit(p), donc sujet(x) = gabarit(vraie⁻¹ x)
        let sujet = carte(64, 2.0, &vraie.try_inverse().unwrap());
        let trouvee = recaler(&gabarit, &sujet, &Matrix4::identity());
        let ecart = ecart_moyen(&gabarit, &trouvee, &vraie);
        println!("translation : écart après recalage {ecart:.2} mm");
        assert!(ecart <= 1.0, "écart {ecart}");
    }

    /// Avec une transformation affine complète (rotations de 8° et 6°, échelles de 85 % à 95 %, translation de (4, −3, 2) mm), l'écart moyen à la vraie transformation doit au moins être divisé par deux
    /// (le recouvrement des cartes d'ellipsoïdes presque symétriques ne contraint pas tous les paramètres : la valeur absolue n'est pas testée ici, mais sur les volumes réels).
    #[test]
    fn a_known_affine_transformation_is_brought_much_closer_on_synthetic_maps() {
        let vraie = Parametres {
            translation: [4.0, -3.0, 2.0],
            rotation: [8.0, 0.0, 6.0],
            echelle: [90.0, 95.0, 85.0],
            cisaillement: [0.0; 3],
        }
        .matrice();
        let gabarit = carte(64, 2.0, &Matrix4::identity());
        let sujet = carte(64, 2.0, &vraie.try_inverse().unwrap());
        let trouvee = recaler(&gabarit, &sujet, &Matrix4::identity());
        let avant = ecart_moyen(&gabarit, &Matrix4::identity(), &vraie);
        let apres = ecart_moyen(&gabarit, &trouvee, &vraie);
        println!("affine : écart au départ {avant:.2} mm, après recalage {apres:.2} mm");
        assert!(apres <= avant / 2.0, "écart {apres} contre {avant} au départ");
    }

    /// Un gabarit vide laisse le départ inchangé.
    #[test]
    fn an_empty_template_returns_the_start() {
        let vide = Volume {
            donnees: Array3::zeros((8, 8, 8)),
            affine: Matrix4::identity(),
        };
        let depart = depart_selon_le_volume(100);
        assert_eq!(recaler(&vide, &vide, &depart), depart);
    }

    /// **Critère de M4** (`docs/MULTIBOUNTI.md`, écrit avant la validation), sur les volumes de VALIDATION `calme3` et `feta005` à `feta010` : (i) pour au moins 6 volumes sur 7, l'écart de notre transformation à celle de MIRTK ne dépasse pas le plus grand écart des huit essais de MIRTK
    /// (départs perturbés, `scripts/multibounti_repetabilite.py`) à la même transformation officielle ; (ii) aucun écart à l'officielle ne dépasse 10 mm. Les volumes de mise au point (`calme1`, `calme2`, `feta001` à `feta004`) sont seulement affichés.
    #[test]
    #[ignore = "demande les références et les essais de répétabilité de data/multibounti (hors dépôt) ; critère M4 (i) non atteint (4 volumes sur 7 dans la dispersion de MIRTK, pire écart 5,00 mm) : voir docs/MULTIBOUNTI.md, section 8"]
    fn the_registration_stays_within_the_dispersion_of_mirtk() {
        use std::path::PathBuf;
        let racine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/multibounti");
        let dof = |chemin: PathBuf| {
            Parametres::depuis_octets(&std::fs::read(&chemin).unwrap_or_else(|e| panic!("{} : {e}", chemin.display())))
                .unwrap()
                .matrice()
        };
        let mut dans_la_dispersion = 0;
        let mut pire = 0.0_f64;
        for (id, validation) in [
            ("calme1", false),
            ("calme2", false),
            ("feta001", false),
            ("feta002", false),
            ("feta003", false),
            ("feta004", false),
            ("calme3", true),
            ("feta005", true),
            ("feta006", true),
            ("feta007", true),
            ("feta008", true),
            ("feta009", true),
            ("feta010", true),
        ] {
            let dossier = racine.join("ref").join(id);
            let gabarit = Volume::lire(&dossier.join("tmp-lr-ref.nii.gz")).unwrap();
            let sujet = Volume::lire(&dossier.join("lr-lab-pad-masked-crop-t2-128.nii.gz")).unwrap();
            let volume_ml: u32 = std::fs::read_to_string(dossier.join("guess-ga.txt"))
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            let debut = std::time::Instant::now();
            let trouvee = recaler(&gabarit, &sujet, &depart_selon_le_volume(volume_ml));
            let duree = debut.elapsed().as_secs_f64();
            let officielle = dof(dossier.join("aff-d.dof"));
            let ecart = ecart_moyen(&gabarit, &trouvee, &officielle);
            let plus_grand_essai = (0..8)
                .map(|k| {
                    ecart_moyen(
                        &gabarit,
                        &dof(racine.join("essais/repet").join(id).join(format!("essai-{k}.dof"))),
                        &officielle,
                    )
                })
                .fold(0.0_f64, f64::max);
            println!(
                "M4 {id} ({}) : écart à l'officielle {ecart:.2} mm, plus grand écart d'un essai de MIRTK {plus_grand_essai:.2} mm, {} ({volume_ml} mL, {duree:.1} s)",
                if validation { "validation" } else { "mise au point" },
                if ecart <= plus_grand_essai { "dans la dispersion" } else { "hors de la dispersion" }
            );
            if validation {
                dans_la_dispersion += usize::from(ecart <= plus_grand_essai);
                pire = pire.max(ecart);
            }
        }
        println!(
            "M4 validation : {dans_la_dispersion} volumes sur 7 dans la dispersion de MIRTK, pire écart {pire:.2} mm"
        );
        assert!(dans_la_dispersion >= 6, "{dans_la_dispersion} volumes sur 7");
        assert!(pire <= 10.0, "pire écart {pire}");
    }
}
