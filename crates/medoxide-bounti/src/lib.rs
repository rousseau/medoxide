//! Portage en Rust (Burn) des réseaux de **BOUNTI** : segmentation du cerveau fœtal en IRM à partir d'une reconstruction 3D (voir `docs/BOUNTI.md`).
//!
//! Chaque réseau est le code généré par `onnx2burn` à partir de l'export ONNX du réseau MONAI d'origine, sans réentraînement. Les poids (GPL v3) restent hors du dépôt.
//! Cette étape ne contient que l'U-Net 3D à 20 sorties, validé contre PyTorch (critère C1) ; la fusion des deux réseaux, le retournement et le pré/post-traitement viendront ensuite.

pub mod attunet;
pub mod bet;
pub mod etiquettes;
pub mod fusion;
pub mod geometrie;
pub mod masque;
pub mod norme;
pub mod poids;
pub mod segmentation;
pub mod sortie;
pub mod table;
pub mod unet;

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};
    use std::path::{Path, PathBuf};

    /// Dossier des données locales de BOUNTI (poids convertis, références), ignoré par Git.
    fn donnees() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti")
    }

    /// Lit un tableau brut `float32` petit-boutiste.
    fn lire_f32(chemin: &Path) -> Vec<f32> {
        let octets = std::fs::read(chemin)
            .unwrap_or_else(|e| panic!("{} illisible : {e} (voir docs/BOUNTI.md)", chemin.display()));
        octets
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// **Critère C1** pour un réseau (cube de `n`³, `data/bounti/reference/<nom>_<n>_*.f32` produit par `scripts/make_bounti_reference.py`) : les logits du réseau Burn (`passe_avant`, GPU) égalent ceux de PyTorch/MONAI à un écart RELATIF maximal (max |Δ| / max |logit|) de 1e-5,
    /// et au moins 99,9 % des voxels ont la même étiquette (argmax sur les 20 canaux).
    fn verifier_c1(
        nom: &str,
        canaux: usize,
        n: usize,
        tag: &str,
        passe_avant: impl Fn(Tensor<5>) -> Tensor<5>,
        device: &Device,
    ) {
        let entree = lire_f32(&donnees().join(format!("reference/{nom}_{n}{tag}_input.f32")));
        let attendus = lire_f32(&donnees().join(format!("reference/{nom}_{n}{tag}_logits.f32")));
        assert_eq!(entree.len(), n * n * n);
        assert_eq!(attendus.len(), canaux * n * n * n);

        let x = Tensor::<5>::from_data(TensorData::new(entree, [1, 1, n, n, n]), device);
        let obtenus = passe_avant(x).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(obtenus.len(), attendus.len(), "forme de sortie inattendue");

        // diagnostic : `MEDOXIDE_DUMP=1` écrit les logits obtenus (tableau brut f32) pour les comparer à une référence en double précision (voir `scripts/analyse_precision_bounti.py`)
        if std::env::var("MEDOXIDE_DUMP").is_ok() {
            let suffixe = std::env::var("MEDOXIDE_DEVICE").unwrap_or_else(|_| "gpu".into());
            let octets: Vec<u8> = obtenus.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(
                donnees().join(format!("reference/{nom}_{n}{tag}_burn_{suffixe}.f32")),
                octets,
            )
            .unwrap();
        }
        let amplitude = attendus.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let ecart = obtenus
            .iter()
            .zip(&attendus)
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        let voxels = n * n * n;
        let etiquette = |logits: &[f32], v: usize| {
            (0..canaux)
                .max_by(|&c, &d| logits[c * voxels + v].partial_cmp(&logits[d * voxels + v]).unwrap())
                .unwrap()
        };
        let identiques = (0..voxels)
            .filter(|&v| etiquette(&obtenus, v) == etiquette(&attendus, v))
            .count();
        let accord = identiques as f64 / voxels as f64;
        println!("{nom} n={n}{tag} : écart relatif {:.2e} (absolu {ecart:.2e}, |logit| max {amplitude:.1}) ; accord des étiquettes {accord:.6}", ecart / amplitude);
        assert!(
            ecart / amplitude <= 1e-5,
            "écart relatif {:.2e} > 1e-5",
            ecart / amplitude
        );
        assert!(accord >= 0.999, "accord des étiquettes {accord}");
    }

    /// Appareil de calcul des tests : le GPU par défaut, ou le CPU de Burn (`flex`) avec `MEDOXIDE_DEVICE=flex` (diagnostic : sépare l'erreur de conversion de l'arithmétique du GPU).
    fn appareil() -> Device {
        match std::env::var("MEDOXIDE_DEVICE").as_deref() {
            Ok("flex") => Device::flex(),
            _ => Device::default(),
        }
    }

    #[test]
    #[ignore = "demande les poids BOUNTI convertis et la référence PyTorch dans data/bounti (hors dépôt)"]
    fn the_unet_logits_match_pytorch() {
        let device = appareil();
        let modele = crate::unet::Model::from_file(donnees().join("burn/unet/unet_64.bpk"), &device);
        verifier_c1("unet", 20, 64, "", |x| modele.forward(x), &device);
    }

    #[test]
    #[ignore = "demande les poids BOUNTI convertis et la référence PyTorch dans data/bounti (hors dépôt)"]
    fn the_attention_unet_logits_match_pytorch() {
        let device = appareil();
        let modele = crate::attunet::Model::from_file(donnees().join("burn/attunet/attunet_64.bpk"), &device);
        verifier_c1("attunet", 20, 64, "", |x| modele.forward(x), &device);
    }

    /// Même contrôle à 256³ (taille de BOUNTI) sur un volume réel rééchantillonné (`make_bounti_reference.py <réseau> 256 <volume.nii.gz>`) : les statistiques d'activation sont réalistes, l'écart se cumule sur 256³ voxels.
    #[test]
    #[ignore = "demande les poids BOUNTI convertis et la référence PyTorch à 256³ dans data/bounti (hors dépôt)"]
    fn the_unet_logits_match_pytorch_on_a_real_volume_at_256() {
        let device = appareil();
        let modele = crate::unet::Model::from_file(donnees().join("burn/unet/unet_64.bpk"), &device);
        verifier_c1("unet", 20, 256, "_reel", |x| modele.forward(x), &device);
    }

    #[test]
    #[ignore = "demande les poids BOUNTI convertis et la référence PyTorch à 256³ dans data/bounti (hors dépôt)"]
    fn the_attention_unet_logits_match_pytorch_on_a_real_volume_at_256() {
        let device = appareil();
        let modele = crate::attunet::Model::from_file(donnees().join("burn/attunet/attunet_64.bpk"), &device);
        verifier_c1("attunet", 20, 256, "_reel", |x| modele.forward(x), &device);
    }

    /// Diagnostic piloté par l'environnement : `BOUNTI_RESEAU` (`unet`, `attunet` ou `bet`), `BOUNTI_N` (côté du cube), `BOUNTI_TAG` (`""` ou `_reel`). Affiche l'écart ; échoue si le critère C1 n'est pas tenu.
    #[test]
    #[ignore = "diagnostic : demande des références locales et des variables d'environnement"]
    fn diagnostic_logits_against_pytorch() {
        let reseau = std::env::var("BOUNTI_RESEAU").unwrap_or_else(|_| "unet".into());
        let n: usize = std::env::var("BOUNTI_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let tag = std::env::var("BOUNTI_TAG").unwrap_or_default();
        let device = appareil();
        match reseau.as_str() {
            "unet" => {
                let m = crate::unet::Model::from_file(donnees().join("burn/unet/unet_64.bpk"), &device);
                verifier_c1("unet", 20, n, &tag, |x| m.forward(x), &device);
            }
            "bet" => {
                let m = crate::bet::Model::from_file(donnees().join("burn/bet/bet_64.bpk"), &device);
                verifier_c1("bet", 2, n, &tag, |x| m.forward(x), &device);
            }
            _ => {
                let m = crate::attunet::Model::from_file(donnees().join("burn/attunet/attunet_64.bpk"), &device);
                verifier_c1("attunet", 20, n, &tag, |x| m.forward(x), &device);
            }
        }
    }

    /// Précision de [`crate::norme::InstanceNorm`] contre un calcul en f64, sur `[1, 4, n, n, n]` de moyenne 3 et d'écart type 1, pour n = 64, 128 et 256 : écart maximal de la sortie normalisée (valeurs de l'ordre de 1) ≤ 1e-5 à toutes les tailles
    /// (mesuré sur GPU : 1,9e-6, 1,7e-6, 2,8e-6 ; la normalisation de `burn::nn::InstanceNorm` donne 2,8e-5, 4,7e-4 puis 1,0e-2 sur ces mêmes tailles).
    #[test]
    fn the_cascaded_instance_norm_stays_accurate_on_large_volumes() {
        use crate::norme::InstanceNormConfig;
        use burn::tensor::Distribution;
        let device = appareil();
        for n in [64usize, 128, 256] {
            let canaux = 4usize;
            let x = Tensor::<5>::random([1, canaux, n, n, n], Distribution::Normal(3.0, 1.0), &device);
            let entree = x.clone().into_data().try_to_vec::<f32>().unwrap();
            let norme = InstanceNormConfig::new(canaux).with_epsilon(1e-5).init(&device);
            let obtenu = norme.forward(x).into_data().try_to_vec::<f32>().unwrap();
            let voxels = n * n * n;
            let mut pire = 0.0_f64;
            for c in 0..canaux {
                let plan = &entree[c * voxels..(c + 1) * voxels];
                let moyenne = plan.iter().map(|&v| f64::from(v)).sum::<f64>() / voxels as f64;
                let variance = plan.iter().map(|&v| (f64::from(v) - moyenne).powi(2)).sum::<f64>() / voxels as f64;
                let echelle = (variance + 1e-5).sqrt();
                for (i, &v) in plan.iter().enumerate() {
                    pire = pire.max((f64::from(obtenu[c * voxels + i]) - (f64::from(v) - moyenne) / echelle).abs());
                }
            }
            println!("INSTANCE_NORM n={n} : écart maximal à f64 {pire:.2e}");
            assert!(pire <= 1e-5, "n = {n} : écart {pire:.2e} > 1e-5");
        }
    }

    /// Précision de la normalisation selon le rapport moyenne / écart type (10, 100, 1000), sur `[1, 2, 128, 128, 128]`, contre un calcul en f64 : écart maximal de la sortie normalisée ≤ 1e-5 pour chaque rapport
    /// (avant la correction à deux passes : 3,0e-6, 1,2e-5 et 2,4e-4 sur GPU).
    #[test]
    fn the_cascaded_instance_norm_is_accurate_for_large_means() {
        use crate::norme::InstanceNormConfig;
        use burn::tensor::Distribution;
        let device = appareil();
        let (n, canaux) = (128usize, 2usize);
        for rapport in [10.0_f64, 100.0, 1000.0] {
            let x = Tensor::<5>::random([1, canaux, n, n, n], Distribution::Normal(rapport, 1.0), &device);
            let entree = x.clone().into_data().try_to_vec::<f32>().unwrap();
            let obtenu = InstanceNormConfig::new(canaux)
                .with_affine(false)
                .init(&device)
                .forward(x)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let voxels = n * n * n;
            let mut pire = 0.0_f64;
            for c in 0..canaux {
                let plan = &entree[c * voxels..(c + 1) * voxels];
                let moyenne = plan.iter().map(|&v| f64::from(v)).sum::<f64>() / voxels as f64;
                let variance = plan.iter().map(|&v| (f64::from(v) - moyenne).powi(2)).sum::<f64>() / voxels as f64;
                let echelle = (variance + 1e-5).sqrt();
                for (i, &v) in plan.iter().enumerate() {
                    pire = pire.max((f64::from(obtenu[c * voxels + i]) - (f64::from(v) - moyenne) / echelle).abs());
                }
            }
            println!("RAPPORT {rapport:6.0} : écart maximal à f64 {pire:.2e}");
            assert!(pire <= 1e-5, "rapport {rapport} : écart {pire:.2e} > 1e-5");
        }
    }

    /// Même sens que `burn::nn::InstanceNorm` sur une petite entrée (où les deux sont précis) : sortie égale à 1e-5 près, avec `gamma` et `beta` non triviaux ; et propriétés exactes : après normalisation sans paramètres affines, chaque canal a une moyenne de 0 (1e-5 près) et une variance de 1 − ε/(variance + ε) (écart type 1 à 1e-4 près pour une variance d'entrée de 25).
    #[test]
    fn the_cascaded_instance_norm_matches_burns_on_small_inputs() {
        use crate::norme::InstanceNormConfig;
        use burn::tensor::Distribution;
        let device = appareil();
        let x = Tensor::<5>::random([2, 3, 6, 7, 8], Distribution::Normal(1.0, 5.0), &device);
        let gamma = Tensor::<1>::from_floats([0.5, 2.0, -1.0], &device);
        let beta = Tensor::<1>::from_floats([0.1, -0.2, 0.3], &device);
        let mut nous = InstanceNormConfig::new(3).init(&device);
        nous.gamma = Some(burn::module::Param::from_tensor(gamma.clone()));
        nous.beta = Some(burn::module::Param::from_tensor(beta.clone()));
        let mut burn_norm = burn::nn::InstanceNormConfig::new(3).init(&device);
        burn_norm.gamma = Some(burn::module::Param::from_tensor(gamma));
        burn_norm.beta = Some(burn::module::Param::from_tensor(beta));
        let a = nous.forward(x.clone()).into_data().try_to_vec::<f32>().unwrap();
        let b = burn_norm.forward(x.clone()).into_data().try_to_vec::<f32>().unwrap();
        let ecart = a.iter().zip(&b).fold(0.0_f32, |m, (u, v)| m.max((u - v).abs()));
        assert!(ecart <= 1e-5, "écart à la normalisation de Burn {ecart:.2e}");
        // canal constant (variance nulle) : le terme ε évite la division par zéro, la sortie est 0 et non indéfinie
        let constant = Tensor::<5>::full([1, 2, 4, 4, 4], 7.0, &device);
        let sortie = InstanceNormConfig::new(2)
            .with_affine(false)
            .init(&device)
            .forward(constant)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            sortie.iter().all(|v| v.is_finite() && v.abs() <= 1e-3),
            "canal constant : sortie non nulle ou indéfinie"
        );
        // sans paramètres affines : moyenne 0 et écart type 1 par exemple et par canal
        let brute = InstanceNormConfig::new(3).with_affine(false).init(&device).forward(x);
        let donnees = brute.into_data().try_to_vec::<f32>().unwrap();
        let taille = 6 * 7 * 8;
        for plan in donnees.chunks(taille) {
            let moyenne = plan.iter().sum::<f32>() / taille as f32;
            let ecart_type = (plan.iter().map(|v| (v - moyenne).powi(2)).sum::<f32>() / taille as f32).sqrt();
            assert!(
                moyenne.abs() <= 1e-5 && (ecart_type - 1.0).abs() <= 1e-3,
                "moyenne {moyenne:.2e}, écart type {ecart_type}"
            );
        }
    }

    /// **Critère C2** (fusion des quatre sorties) : sur un cube de `BOUNTI_N`³ (128 par défaut) d'un volume réel rééchantillonné (`data/bounti/reference/unet_<n>_reel_input.f32`), la fusion Rust des deux réseaux de BOUNTI (image normale et retournée, échange gauche/droite, moyenne des logits) donne au moins 99,9 % de voxels d'étiquette identique
    /// à la fusion de référence PyTorch/MONAI (`scripts/make_bounti_fusion_reference.py`). Indicatif : écart relatif des logits fusionnés.
    #[test]
    #[ignore = "demande les poids BOUNTI convertis et la fusion de référence PyTorch dans data/bounti (hors dépôt)"]
    fn the_fusion_matches_the_pytorch_reference() {
        let n: usize = std::env::var("BOUNTI_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let device = appareil();
        let unet = crate::unet::Model::from_file(donnees().join("burn/unet/unet_64.bpk"), &device);
        let attunet = crate::attunet::Model::from_file(donnees().join("burn/attunet/attunet_64.bpk"), &device);
        let entree = lire_f32(&donnees().join(format!("reference/unet_{n}_reel_input.f32")));
        let image = Tensor::<5>::from_data(TensorData::new(entree, [1, 1, n, n, n]), &device);
        let logits = crate::fusion::logits_fusionnes(
            image,
            &[&|x: Tensor<5>| unet.forward(x), &|x: Tensor<5>| attunet.forward(x)],
            &device,
        );

        let attendus = lire_f32(&donnees().join(format!("reference/fusion_{n}_reel_logits.f32")));
        let obtenus = logits.clone().into_data().try_to_vec::<f32>().unwrap();
        let amplitude = attendus.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let ecart = obtenus
            .iter()
            .zip(&attendus)
            .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));

        let etiquettes = crate::fusion::etiquettes(logits);
        let reference = std::fs::read(donnees().join(format!("reference/fusion_{n}_reel_labels.u8")))
            .expect("étiquettes de référence illisibles");
        assert_eq!(etiquettes.len(), reference.len());
        let identiques = etiquettes.iter().zip(&reference).filter(|(a, b)| a == b).count();
        let accord = identiques as f64 / reference.len() as f64;
        println!("FUSION n={n} : accord des étiquettes {accord:.6} ({} voxels différents sur {}) ; écart relatif des logits fusionnés {:.2e}", reference.len() - identiques, reference.len(), ecart / amplitude);
        assert!(accord >= 0.999, "accord des étiquettes {accord}");
    }

    /// **Critère C1** pour le réseau d'extraction du cerveau (2 sorties) : sur un cube de 64³ puis de 128³ (la taille de la chaîne officielle) d'un volume réel rééchantillonné (`make_bounti_reference.py bet <n> <volume.nii.gz>`), écart relatif des logits au plus 1e-5 et au moins 99,9 % d'étiquettes identiques à PyTorch/MONAI.
    #[test]
    #[ignore = "demande les poids BOUNTI convertis et la référence PyTorch dans data/bounti (hors dépôt)"]
    fn the_brain_extraction_logits_match_pytorch_on_a_real_volume() {
        let device = appareil();
        let modele = crate::bet::Model::from_file(donnees().join("burn/bet/bet_64.bpk"), &device);
        for n in [64usize, 128] {
            verifier_c1("bet", 2, n, "_reel", |x| modele.forward(x), &device);
        }
    }
}
