# Validation

`medx fetalbet` est comparé à la sortie du script d'inférence officiel de
Fetal-BET (MONAI 1.4.0, CPU), sur les 8 volumes d'exemple fournis dans le dépôt de
Fetal-BET (`dataset/f0832s1`). Le script officiel est reproductible : deux
exécutions donnent des masques identiques voxel pour voxel.

## Résultat de bout en bout

Commande `medx fetalbet` compilée en `release`, GPU Apple. Le Dice compare au
masque de Fetal-BET.

| Volume | Dimensions | Dice | Voxels différents | Durée |
|---|---|---|---|---|
| fetus_03 | 256 × 256 × 44 | 0,999899 | 14 | 99 s |
| fetus_04 | 256 × 256 × 40 | 1,000000 | 0 | 90 s |
| fetus_06 | 256 × 256 × 46 | 0,999987 | 3 | 25 s |
| fetus_07 | 256 × 256 × 50 | 0,999973 | 4 | 112 s |
| fetus_09 | 256 × 256 × 50 | 0,999987 | 2 | 112 s |
| fetus_10 | 256 × 256 × 50 | 0,999955 | 7 | 112 s |
| fetus_11 | 256 × 256 × 50 | 0,999975 | 4 | 112 s |
| fetus_12 | 256 × 256 × 50 | 1,000000 | 0 | 112 s |

Les écarts viennent du prétraitement, pas de l'inférence : avec l'entrée de MONAI,
tout ce qui suit est exact (voir plus bas). D'après nos mesures, MONAI laisse des
valeurs de l'ordre de 1e-15 là où le résultat exact est 0, ce qui décale très
légèrement la normalisation. Sur `fetus_03`, 14 voxels diffèrent sur 2,9 millions.

## Étapes intermédiaires

Chaque étape a été validée séparément contre MONAI ou PyTorch avant d'être
assemblée.

| Étape | Résultat |
|---|---|
| Export ONNX vs PyTorch | écart relatif des logits 8e-6, masques identiques |
| Inférence Burn/wgpu vs onnxruntime (une tuile) | écart relatif 3e-6, masques identiques |
| Lecture des voxels NIfTI vs nibabel | identiques |
| Normalisation par coupe vs MONAI | identique |
| Rééchantillonnage à 1 mm vs MONAI | écart relatif ≤ 8e-6 |
| Inférence par fenêtres vs `SliceInferer` (entrée MONAI) | Dice 1,00000 |
| Retour à la grille d'origine vs masques officiels | 0 voxel de différence |

Les écarts d'intensité sont relatifs au maximum de la valeur de référence.

## Limites

- Les 8 volumes viennent d'un seul jeu de données, et les mesures d'un seul Mac.
  Autres résolutions, orientations, scanners, GPU et systèmes ne sont pas testés.
- La référence est Fetal-BET lui-même : cette validation montre que le portage
  est fidèle, pas que le masque est correct. Elle ne remplace pas une comparaison
  à une segmentation manuelle.
- Plusieurs seuils ont été révisés après les premières mesures (écarts de logits :
  absolu puis relatif ; prétraitement). Le juge final est le Dice.

## Reproduire

Les volumes sont ceux du dépôt Fetal-BET et les masques officiels sont produits
avec leur `inference.py`. Les autres références sont produites par les scripts de
`scripts/` (`make_reference_volumes.py`, `make_reference_masks.py`) et rangées dans
`data/`, hors de Git. Les tests Rust les comparent à `medx` :

```bash
cargo test -p medoxide-fetalbet                       # environ 1 min
cargo test -p medoxide-fetalbet -- --ignored --nocapture   # 8 volumes : plusieurs dizaines de minutes
```
