# Algorithmes et références

## Fetal-BET

`medx fetalbet` reproduit l'inférence de **Fetal-BET**, un outil d'extraction
du cerveau fœtal en IRM fondé sur un Attention U-Net 2D. Le modèle
(architecture et poids) est celui des auteurs ; nous n'avons rien réentraîné.

> Faghihpirayesh R., Karimi D., Erdoğmuş D., Gholipour A. *Fetal-BET: Brain
> Extraction Tool for Fetal MRI*. IEEE Open Journal of Engineering in Medicine
> and Biology, 2024. Code : <https://github.com/IntelligentImaging/fetal-brain-extraction>

```bibtex
@article{faghihpirayesh2024fetal,
  title={Fetal-bet: Brain extraction tool for fetal mri},
  author={Faghihpirayesh, Razieh and Karimi, Davood and Erdo{\u{g}}mu{\c{s}}, Deniz and Gholipour, Ali},
  journal={IEEE Open Journal of Engineering in Medicine and Biology},
  year={2024},
  publisher={IEEE}
}
```

Si vous utilisez `medx fetalbet`, merci de citer cet article. Il est publié sous
licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), comme le
modèle. Les auteurs précisent que l'outil est destiné à la recherche, pas à un
usage médical ou diagnostique, et qu'il est fourni sans garantie.

## Pipeline

Pour chaque volume, `medx fetalbet` enchaîne :

1. **Rééchantillonnage à 1 × 1 mm** dans le plan (interpolation bilinéaire, zéros
   hors du volume). L'axe des coupes n'est pas modifié.
2. **Normalisation par coupe** : les voxels strictement positifs sont divisés par
   leur écart-type. La moyenne n'est *pas* soustraite : ce n'est pas un z-score.
3. **Inférence par fenêtres** de 256 × 256 voxels, avec 50 % de recouvrement :
   les sorties du modèle (logits) des fenêtres qui se recouvrent sont moyennées.
4. **Retour à la grille d'origine** : les logits sont rééchantillonnés vers la
   grille de l'image, puis on garde la classe de plus grand logit (cerveau ou
   fond). Interpoler le masque au lieu des logits donne un résultat différent.
5. **Écriture** du masque en NIfTI, avec l'en-tête de l'image d'entrée.

Ces étapes suivent le script d'inférence officiel de Fetal-BET. Le modèle est très
sensible à ce prétraitement : sur une coupe de test à 1,17 mm non rééchantillonnée,
il ne détectait aucun voxel de cerveau.

## Ce que nous avons modifié

Les poids PyTorch d'origine ont été exportés en ONNX, puis convertis en code Rust
et en poids au format `burnpack` pour [Burn](https://burn.dev) (backend `wgpu`,
donc exécution sur GPU). Les calculs sont équivalents à l'original
([Validation](validation.md)). Le prétraitement, les fenêtres et le
rééchantillonnage sont réécrits en Rust.

## Autres références

- **MONAI** fournit le modèle d'origine (`AttentionUnet`) et le script de référence
  (`Spacingd`, `SliceInferer`) auxquels nous comparons. Cardoso M. J. et al.,
  *MONAI: An open-source framework for deep learning in healthcare*,
  [arXiv:2211.02701](https://arxiv.org/abs/2211.02701), 2022.
- **Attention U-Net**, l'architecture. Oktay O. et al., *Attention U-Net: Learning
  Where to Look for the Pancreas*, [arXiv:1804.03999](https://arxiv.org/abs/1804.03999), 2018.
- **Burn**, le framework d'apprentissage profond en Rust : <https://burn.dev>.
