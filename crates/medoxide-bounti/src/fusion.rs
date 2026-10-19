//! Fusion des quatre sorties de BOUNTI : deux réseaux (U-Net et Attention U-Net), chacun appliqué à l'image et à son **retournement gauche-droite**.
//!
//! Principe (augmentation à l'inférence) : une structure de l'hémisphère gauche devient une structure de l'hémisphère droit dans l'image miroir. La sortie du réseau sur l'image retournée est donc retournée en retour, puis les canaux gauche et droit sont échangés (1↔2, 3↔4…), avant de moyenner
//! les **logits** (non normalisés) des quatre sorties ; l'étiquette est le canal de plus grand logit. Cette moyenne réduit un éventuel biais du réseau entre les deux côtés.
//!
//! Les faits (axe retourné, permutation des canaux, moyenne des logits) sont ceux de la chaîne officielle de BOUNTI, relevés et vérifiés dans `docs/BOUNTI.md` ; le code est écrit ici à partir de cette description.
use burn::tensor::{Device, Int, Tensor};

/// Nombre de canaux de sortie des réseaux : 19 étiquettes de tissus et le fond.
pub const CANAUX: usize = 20;

/// Canal de la sortie « image normale » qui correspond au canal `c` de la sortie « image retournée » : les paires gauche/droite 1↔2, 3↔4, 5↔6, 7↔8, 11↔12, 14↔15, 16↔17 sont échangées ; le fond (0) et les structures médianes (9, 10, 13, 18, 19) sont inchangés.
pub const PARTENAIRE: [i64; CANAUX] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 9, 10, 12, 11, 13, 15, 14, 17, 16, 18, 19];

/// Axe du retournement dans un tenseur `[exemples, canaux, X, Y, Z]` : le premier axe spatial (X), le même que `Flip(1)` de MONAI sur un tenseur `[1, 1, X, Y, Z]`.
const AXE_X: usize = 2;

/// Retourne le volume sur l'axe x (gauche-droite).
pub fn retourner_x(volume: Tensor<5>) -> Tensor<5> {
    volume.flip([AXE_X])
}

/// Échange les canaux gauche et droit d'une sortie `[exemples, 20, X, Y, Z]` : le canal `c` du résultat est le canal [`PARTENAIRE`]`[c]` de l'entrée.
pub fn echanger_gauche_droite(logits: Tensor<5>, device: &Device) -> Tensor<5> {
    logits.select(1, Tensor::<1, Int>::from_ints(PARTENAIRE, device))
}

/// Logits fusionnés de `image` (`[1, 1, X, Y, Z]`, intensités dans [0, 1]) pour les `reseaux` donnés (chacun transforme `[1, 1, X, Y, Z]` en `[1, 20, X, Y, Z]`) :
/// moyenne, sur les réseaux et sur l'image normale et retournée, des sorties remises dans le repère de l'image (retour du retournement puis échange gauche/droite).
/// Les sorties sont accumulées une à une : au plus une somme et une sortie sont en mémoire en même temps (1,3 Go chacune à 256³).
pub fn logits_fusionnes(image: Tensor<5>, reseaux: &[&dyn Fn(Tensor<5>) -> Tensor<5>], device: &Device) -> Tensor<5> {
    assert!(!reseaux.is_empty(), "au moins un réseau");
    let mut somme: Option<Tensor<5>> = None;
    let mut ajouter = |sortie: Tensor<5>| {
        somme = Some(match somme.take() {
            Some(s) => s + sortie,
            None => sortie,
        });
    };
    for reseau in reseaux {
        ajouter(reseau(image.clone()));
        let retournee = reseau(retourner_x(image.clone()));
        ajouter(echanger_gauche_droite(retourner_x(retournee), device));
    }
    somme.expect("au moins une sortie") / (2 * reseaux.len()) as f64
}

/// Étiquette de chaque voxel : le canal de plus grand logit, en ordre `[X, Y, Z]` (ordre C) pour un exemple (`[1, 20, X, Y, Z]`).
pub fn etiquettes(logits: Tensor<5>) -> Vec<u8> {
    let indices = logits.argmax(1).into_data().convert::<i64>();
    indices.try_to_vec::<i64>().expect("indices entiers").into_iter().map(|c| c as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    /// La permutation est exactement celle des paires documentées (1↔2, 3↔4, 5↔6, 7↔8, 11↔12, 14↔15, 16↔17), involutive, avec 0, 9, 10, 13, 18 et 19 comme points fixes.
    #[test]
    fn the_left_right_permutation_is_the_documented_one() {
        let paires = [(1usize, 2usize), (3, 4), (5, 6), (7, 8), (11, 12), (14, 15), (16, 17)];
        let mut attendue: Vec<i64> = (0..CANAUX as i64).collect();
        for (a, b) in paires {
            attendue[a] = b as i64;
            attendue[b] = a as i64;
        }
        assert_eq!(PARTENAIRE.to_vec(), attendue);
        for c in 0..CANAUX {
            assert_eq!(PARTENAIRE[PARTENAIRE[c] as usize], c as i64, "involution au canal {c}");
        }
        for fixe in [0usize, 9, 10, 13, 18, 19] {
            assert_eq!(PARTENAIRE[fixe], fixe as i64);
        }
    }

    /// `echanger_gauche_droite` : le canal `c` du résultat est le canal `PARTENAIRE[c]` de l'entrée (valeurs distinctes par canal et par position) ; `retourner_x` renverse exactement l'ordre sur l'axe x et laisse y et z intacts.
    #[test]
    fn the_swap_and_the_flip_move_exactly_what_they_should() {
        let device = Device::default();
        let (x, y, z) = (3usize, 2usize, 2usize);
        let valeur = |c: usize, i: usize, j: usize, k: usize| (((c * x + i) * y + j) * z + k) as f32;
        let donnees: Vec<f32> = (0..CANAUX).flat_map(|c| (0..x).flat_map(move |i| (0..y).flat_map(move |j| (0..z).map(move |k| valeur(c, i, j, k))))).collect();
        let entree = Tensor::<5>::from_data(TensorData::new(donnees, [1, CANAUX, x, y, z]), &device);
        let echange = echanger_gauche_droite(entree.clone(), &device).into_data().try_to_vec::<f32>().unwrap();
        let retourne = retourner_x(entree).into_data().try_to_vec::<f32>().unwrap();
        for c in 0..CANAUX {
            for i in 0..x {
                for j in 0..y {
                    for k in 0..z {
                        let position = ((c * x + i) * y + j) * z + k;
                        assert_eq!(echange[position], valeur(PARTENAIRE[c] as usize, i, j, k), "échange c={c} i={i} j={j} k={k}");
                        assert_eq!(retourne[position], valeur(c, x - 1 - i, j, k), "retournement c={c} i={i} j={j} k={k}");
                    }
                }
            }
        }
    }

    /// Fusion de deux réseaux factices **non symétriques** (sortie `a(c)·image + b(c)·(i + 1)` avec `i` l'indice sur l'axe x, de coefficients différents par réseau et par canal) : la fusion Rust égale à 1e-5 près la formule écrite à la main,
    /// `moyenne sur les réseaux de [ normale(c, i) + retournée_remise_dans_le_repère(PARTENAIRE[c], i) ] / 2`, ce qui contrôle à la fois le sens du retournement, l'ordre « retour puis échange » et le diviseur.
    #[test]
    fn the_fusion_follows_the_hand_written_formula() {
        let device = Device::default();
        let (x, y, z) = (3usize, 2usize, 2usize);
        let image_donnees: Vec<f32> = (0..x * y * z).map(|v| 0.1 + 0.07 * v as f32).collect();
        let coefficients = |graine: f32| -> (Vec<f32>, Vec<f32>) { ((0..CANAUX).map(|c| graine + 0.3 * c as f32).collect(), (0..CANAUX).map(|c| 0.5 - graine * 0.2 * c as f32).collect()) };
        let (a1, b1) = coefficients(1.0);
        let (a2, b2) = coefficients(2.5);
        let fabriquer = |a: Vec<f32>, b: Vec<f32>| {
            let device = Device::default();
            move |image: Tensor<5>| {
                let canaux = Tensor::cat(vec![image; CANAUX], 1);
                let poids_a = Tensor::<5>::from_data(TensorData::new(a.clone(), [1, CANAUX, 1, 1, 1]), &device);
                let mut pos = Vec::with_capacity(CANAUX * x);
                for c in 0..CANAUX {
                    for i in 0..x {
                        pos.push(b[c] * (i + 1) as f32);
                    }
                }
                let terme = Tensor::<5>::from_data(TensorData::new(pos, [1, CANAUX, x, 1, 1]), &device);
                canaux * poids_a + terme
            }
        };
        let (r1, r2) = (fabriquer(a1.clone(), b1.clone()), fabriquer(a2.clone(), b2.clone()));
        let image = Tensor::<5>::from_data(TensorData::new(image_donnees.clone(), [1, 1, x, y, z]), &device);
        let fusion = logits_fusionnes(image, &[&r1, &r2], &device).into_data().try_to_vec::<f32>().unwrap();
        let pixel = |i: usize, j: usize, k: usize| image_donnees[(i * y + j) * z + k];
        for c in 0..CANAUX {
            let p = PARTENAIRE[c] as usize;
            for i in 0..x {
                for j in 0..y {
                    for k in 0..z {
                        let mut attendu = 0.0_f32;
                        for (a, b) in [(&a1, &b1), (&a2, &b2)] {
                            let normale = a[c] * pixel(i, j, k) + b[c] * (i + 1) as f32;
                            // image retournée : x_f[i'] = x[X-1-i'] ; sortie f(x_f)[p, i'] = a[p]·x[X-1-i'] + b[p]·(i'+1) ; retournée en retour : i' = X-1-i
                            let retournee = a[p] * pixel(i, j, k) + b[p] * (x - i) as f32;
                            attendu += normale + retournee;
                        }
                        attendu /= 4.0;
                        let obtenu = fusion[((c * x + i) * y + j) * z + k];
                        assert!((obtenu - attendu).abs() <= 1e-5, "c={c} i={i} j={j} k={k} : {obtenu} contre {attendu}");
                    }
                }
            }
        }
    }

    /// `etiquettes` : le canal de plus grand logit à chaque voxel, en ordre C `[X, Y, Z]`.
    #[test]
    fn the_labels_are_the_channel_of_largest_logit() {
        let device = Device::default();
        let (x, y, z) = (2usize, 2usize, 3usize);
        let voxels = x * y * z;
        let mut donnees = vec![0.0_f32; CANAUX * voxels];
        let attendues: Vec<u8> = (0..voxels).map(|v| ((v * 7 + 3) % CANAUX) as u8).collect();
        for (v, &c) in attendues.iter().enumerate() {
            donnees[c as usize * voxels + v] = 5.0 + v as f32;
        }
        let logits = Tensor::<5>::from_data(TensorData::new(donnees, [1, CANAUX, x, y, z]), &device);
        assert_eq!(etiquettes(logits), attendues);
    }
}
