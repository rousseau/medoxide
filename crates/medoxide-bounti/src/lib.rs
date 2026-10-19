//! Portage en Rust (Burn) des réseaux de **BOUNTI** : segmentation du cerveau fœtal en IRM à partir d'une reconstruction 3D (voir `docs/BOUNTI.md`).
//!
//! Chaque réseau est le code généré par `onnx2burn` à partir de l'export ONNX du réseau MONAI d'origine, sans réentraînement. Les poids (GPL v3) restent hors du dépôt.
//! Cette étape ne contient que l'U-Net 3D à 20 sorties, validé contre PyTorch (critère C1) ; la fusion des deux réseaux, le retournement et le pré/post-traitement viendront ensuite.

pub mod unet;

#[cfg(test)]
mod tests {
    use super::unet::Model;
    use burn::tensor::{Device, Tensor, TensorData};
    use std::path::{Path, PathBuf};

    /// Dossier des données locales de BOUNTI (poids convertis, références), ignoré par Git.
    fn donnees() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/bounti")
    }

    /// Lit un tableau brut `float32` petit-boutiste.
    fn lire_f32(chemin: &Path) -> Vec<f32> {
        let octets = std::fs::read(chemin).unwrap_or_else(|e| panic!("{} illisible : {e} (voir docs/BOUNTI.md)", chemin.display()));
        octets.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    /// **Critère C1** (U-Net 3D, cube de 64³, `data/bounti/reference/unet_64_*.f32` produit par `scripts/make_bounti_reference.py`) : les logits du réseau Burn (GPU) égalent ceux de PyTorch/MONAI à un écart RELATIF maximal (max |Δ| / max |logit|) de 1e-5,
    /// et au moins 99,9 % des voxels ont la même étiquette (argmax sur les 20 canaux).
    #[test]
    #[ignore = "demande les poids BOUNTI convertis et la référence PyTorch dans data/bounti (hors dépôt)"]
    fn the_unet_logits_match_pytorch() {
        let (n, canaux) = (64usize, 20usize);
        let device = Device::default();
        let modele = Model::from_file(donnees().join("burn/unet/unet_64.bpk"), &device);
        let entree = lire_f32(&donnees().join("reference/unet_64_input.f32"));
        let attendus = lire_f32(&donnees().join("reference/unet_64_logits.f32"));
        assert_eq!(entree.len(), n * n * n);
        assert_eq!(attendus.len(), canaux * n * n * n);

        let x = Tensor::<5>::from_data(TensorData::new(entree, [1, 1, n, n, n]), &device);
        let obtenus = modele.forward(x).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(obtenus.len(), attendus.len(), "forme de sortie inattendue");

        let amplitude = attendus.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let ecart = obtenus.iter().zip(&attendus).fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
        let voxels = n * n * n;
        let etiquette = |logits: &[f32], v: usize| (0..canaux).max_by(|&c, &d| logits[c * voxels + v].partial_cmp(&logits[d * voxels + v]).unwrap()).unwrap();
        let identiques = (0..voxels).filter(|&v| etiquette(&obtenus, v) == etiquette(&attendus, v)).count();
        let accord = identiques as f64 / voxels as f64;
        println!("U-NET écart relatif {:.2e} (absolu {ecart:.2e}, |logit| max {amplitude:.1}) ; accord des étiquettes {accord:.6}", ecart / amplitude);
        assert!(ecart / amplitude <= 1e-5, "écart relatif {:.2e} > 1e-5", ecart / amplitude);
        assert!(accord >= 0.999, "accord des étiquettes {accord}");
    }
}
