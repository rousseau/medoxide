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

# SVR, reconstruction coupe-vers-volume

La validation de `medx svr recon` a deux volets : des **briques testées contre des références
indépendantes**, et une **évaluation de bout en bout sur données réelles**, sans vérité terrain. Les mesures
datent du 12 octobre 2026.

## Les briques

Le crate `medoxide-svr` a 117 tests (`cargo test -p medoxide-svr`, environ 1 à 2 minutes). Les briques
principales sont comparées à des références écrites séparément ; la plupart ont en plus subi des tests par
**mutation** (on casse volontairement le code pour vérifier que les tests le voient).

| Brique | Référence | Résultat |
|---|---|---|
| Opérateur d'acquisition | NeSVoR | corrélation 0,9990 |
| Adjoint de l'opérateur | produit scalaire `⟨Ax, y⟩ = ⟨x, Aᵀy⟩` | exact à la précision machine |
| Objectif et gradient analytique | algèbre dense | écart relatif 6e-8 |
| Gradient conjugué | solution directe | écart relatif 3e-8 |
| Reconstruction d'un atlas simulé (stacks simulés, bruit 5 %, vraies poses) | l'atlas | corrélation 0,974 ; +7,4 dB par rapport à l'adjoint normalisé |
| Coût coupe-coupe | référence indépendante en `f64` ; mesure faite en Python | écart 1,5e-7 ; 2 898 paires identiques |
| Gradient automatique (Burn) du coût coupe-coupe | différences finies | cosinus 1,0000 |
| Poses écrites par `--poses` | relecture | exactement identiques |
| `medx svr recon --device cpu` (`s2s`, 3 stacks, sujet réel) | résultat de la version de développement | poses identiques bit à bit, volume à 7e-7 près |
| `medx svr recon --motion s2v --device cpu` (3 stacks, sujet réel) | résultat de la version de développement | poses identiques bit à bit (98 coupes) |

## Évaluation sur données réelles

**Données** : six sujets d'un jeu local (non public) d'IRM fœtale, acquisitions TRUFI axiale, coronale et
sagittale ; trois sujets calmes et trois agités, choisis par une règle fixée avant les mesures.

**Juge** : il n'y a pas de vérité terrain. Pour chaque stack mis de côté, un volume est reconstruit avec
les **deux autres** stacks ; les coupes du stack mis de côté sont ensuite recalées contre ce volume par la
**même** machinerie pour toutes les méthodes. Une corrélation plus haute signifie un volume qui prédit
mieux des coupes qu'il n'a jamais vues. Une valeur par sujet (moyenne des trois repliements) : le sujet est
l'unité statistique.

| Méthode | Corrélation moyenne (6 sujets) |
|---|---|
| medoxide, coupe-volume (`s2v`, « sans la coupe ») | 0,768 |
| medoxide, coupe-coupe (`s2s`) | 0,766 |
| Sans correction du mouvement (poses d'en-tête) | 0,746 |
| SVRTK (réglages par défaut) | 0,730 |
| NeSVoR (réglages par défaut, sans SVoRT, CPU) | 0,708 |

Par sujet : `s2s` fait mieux que ne pas corriger chez 4 sujets sur 6 et mieux que SVRTK chez 5 sur 6 ; il fait
mieux que NeSVoR chez 6 sur 6. **Les deux sujets les plus agités** sont un échec pour toutes les méthodes qui
corrigent le mouvement : elles prédisent moins bien que « sans correction », SVRTK et NeSVoR compris. Avec deux
stacks seulement pour reconstruire, ce protocole est plus dur que l'usage réel (trois stacks ou plus).

**Concordance aux intersections** (mesure neutre du mouvement : les intensités de deux coupes qui se croisent
doivent concorder le long de leur droite d'intersection ; trois stacks, mesurée sur les paires de coupes entre le
2ᵉ et le 3ᵉ stack, moyenne sur les six sujets) :
sans correction 0,455, SVRTK 0,643, NeSVoR 0,644, coupe-volume 0,692. Le coupe-coupe, qui optimise ce même
critère, atteint 0,727 : la comparaison n'est pas neutre pour lui.

**Durées** (CPU, même machine à 12 cœurs ; mesures indicatives, d'autres calculs tournaient en parallèle) :

| Méthode | Un sujet, 3 stacks |
|---|---|
| medoxide `s2s` (mouvement + reconstruction) | 150 s |
| medoxide `s2v` | 8 à 17 minutes (le recalage des coupes est parallèle ; à charge égale, il est environ 20 % plus rapide qu'en série) |
| SVRTK (Docker, arm64) | 1 à 3 minutes |
| NeSVoR (CPU, 2 stacks) | 3 à 5 minutes |

## Limites

- **Six sujets d'un seul jeu**, un seul contraste (TRUFI) : rien ne montre que medoxide soit meilleur que SVRTK
  « en général ».
- **Le juge est notre propre recalage**, validé sur des données simulées (corrélation de rang de 1,00 avec le PSNR
  contre l'atlas, sur 5 volumes), mais ce n'est pas une vérité terrain.
- SVRTK et NeSVoR sont utilisés **sans réglage** ; notre régularisation `α` a été réglée sur du bruit simulé.
- **Les simulations ne prédisent pas le comportement réel** : plusieurs résultats obtenus sur des stacks simulés
  (par exemple l'avantage d'une référence « sans le stack » sur « sans la coupe ») se sont inversés sur les données
  réelles. Les simulations servent aux tests de régression, pas à conclure.
- Un seul Mac a été utilisé. Les données réelles ne sont pas publiques : ces résultats ne sont pas reproductibles
  en l'état par d'autres, seuls les tests sur données simulées le sont.

## Reproduire

Les tests sur données simulées tournent avec `cargo test -p medoxide-svr`. Les tests d'évaluation sur données
réelles ou volumineuses sont marqués `#[ignore]` et se lancent à la main (`cargo test --release -p medoxide-svr -- --ignored`),
avec des variables d'environnement (`MEDOXIDE_DONNEES_REELLES`, `MEDOXIDE_SUJET`, …) qui désignent des données
locales. Les scripts de comparaison sont dans `scripts/` : `make_pyrecon_benchmark.py` (stacks simulés),
`run_svrtk.py`, `run_nesvor.py`, `benchmark_real.py`, `analyse_juge_sujets.py` et `intersection_consistency.py`.
