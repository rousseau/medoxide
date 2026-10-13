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
| Recalage coupe-volume par lots (`--device wgpu`) | recalage coupe par coupe (blobs, CPU) | poses identiques (écart 0) ; sujet réel sur GPU : concordance 0,8513 contre 0,8516 sur CPU |
| Reconstructions intermédiaires sautées | boucle complète (blobs ; sujet réel) | poses identiques bit à bit (blobs et 98 coupes réelles), volume final à 8e-3 près |

## Évaluation sur données réelles

**Données** : six sujets d'un jeu local (non public) d'IRM fœtale, acquisitions TRUFI axiale, coronale et
sagittale ; trois sujets calmes et trois agités, choisis par une règle fixée avant les mesures.

**Juge** : il n'y a pas de vérité terrain. Pour chaque stack mis de côté, un volume est reconstruit avec
les **deux autres** stacks ; les coupes du stack mis de côté sont ensuite recalées contre ce volume par la
**même** machinerie pour toutes les méthodes. Une corrélation plus haute signifie un volume qui prédit
mieux des coupes qu'il n'a jamais vues. Une valeur par sujet (moyenne des trois repliements) : le sujet est
l'unité statistique.

Résultats avec **α = 0,1** pour nos méthodes (voir [Choix de la régularisation α](#choix-de-la-régularisation-α) ; la première évaluation, faite avec α = 0,01, est périmée) :

| Sujet | sans correction | `s2s` | `s2v` | SVRTK | NeSVoR |
|---|---|---|---|---|---|
| calme 1 | 0,853 | 0,877 | 0,873 | 0,815 | 0,855 |
| calme 2 | 0,879 | 0,909 | 0,913 | 0,863 | 0,884 |
| calme 3 | 0,841 | 0,875 | 0,882 | 0,827 | 0,854 |
| agité 1 | 0,779 | 0,774 | 0,830 | 0,675 | 0,694 |
| agité 2 | 0,703 | 0,703 | 0,626 | 0,572 | 0,403 |
| agité 3 | 0,707 | 0,691 | 0,648 | 0,632 | 0,559 |
| **moyenne** | 0,794 | **0,805** | 0,795 | 0,731 | 0,708 |

Par sujet : `s2s` fait mieux que ne pas corriger chez 4 sujets sur 6 (en moyenne +0,011, pire sujet −0,017) ;
`s2v` aussi chez 4 sujets sur 6 (en moyenne +0,002) mais perd 0,076 sur l'agité 2 ; les deux font mieux que SVRTK
et que NeSVoR chez 6 sujets sur 6. **Pour les sujets les plus agités (2 et 3), aucune méthode ne fait mieux que de
ne pas corriger** (`s2s` fait jeu égal, `s2v` et les deux autres sont en dessous). La régularisation forte profite
aussi à « sans correction », ce qui réduit l'avantage apparent de la correction du mouvement. Avec deux stacks
seulement pour reconstruire, ce protocole est plus dur que l'usage réel (trois stacks ou plus).

**Le juge favorise le lissage.** Une NCC sur des coupes bruitées récompense un volume lisse. Contrôle : un simple flou
gaussien de 1 voxel (0,8 mm) des volumes de SVRTK augmente leur score de **+0,04** (calme 1 : 0,815 → 0,857 ;
agité 1 : 0,675 → 0,717 ; deux voxels : trop flou). **Face à SVRTK lissé de façon optimale, l'avantage de medoxide est
donc plus petit que dans le tableau** : environ +0,02 sur le sujet calme (0,877 contre 0,857) et de +0,06 à +0,11
sur le sujet agité (0,774 à 0,830 contre 0,717). Le juge ne mesure pas la netteté : SVRTK est visiblement plus net
que nos volumes à α = 0,1.

**Les sujets très agités, à netteté égale.** Pour écarter le biais de lissage, nous comparons `s2s` et « sans correction » sur la **frontière juge / netteté** (plusieurs valeurs de α pour chaque méthode, et pour chaque valeur de `s2s` le juge de « sans correction » interpolé à la même netteté) : agité 2 : `s2s` +0,005 (équivalent) ; agité 3 : −0,016 (moins bon). Le biais de lissage n'explique donc pas l'échec. En revanche, avec **trois** stacks (usage réel), les images de `s2s` et `s2v` montrent plus de structure que « sans correction » (anneau cortical, ventricules), et la concordance aux intersections des poses est nettement meilleure que celle des poses d'en-tête : le protocole à deux stacks, très peu contraint pour un fœtus très agité, est probablement trop dur pour trancher.

**Concordance aux intersections** (mesure neutre du mouvement : les intensités de deux coupes qui se croisent
doivent concorder le long de leur droite d'intersection ; trois stacks, mesurée sur les paires de coupes entre le
2ᵉ et le 3ᵉ stack, moyenne sur les six sujets) :
sans correction 0,455, SVRTK 0,643, NeSVoR 0,644, coupe-volume 0,692. Le coupe-coupe, qui optimise ce même
critère, atteint 0,727 : la comparaison n'est pas neutre pour lui.

**Durées** (CPU, même machine à 12 cœurs ; mesures indicatives, d'autres calculs tournaient en parallèle) :

| Méthode | Un sujet, 3 stacks |
|---|---|
| medoxide `s2s` (mouvement + reconstruction) | 150 s |
| medoxide `s2v` | 151 s sur CPU (349 s sur GPU par lots) |
| SVRTK (Docker, arm64) | 1 à 3 minutes |
| NeSVoR (CPU, 2 stacks) | 3 à 5 minutes |

## Choix de la régularisation α

Notre premier réglage (α = 0,01) avait été fait sur du bruit simulé. Sur données réelles, il laissait un **hachurage régulier** (période de 2 à 3 voxels) et des valeurs négatives (jusqu'à −30 % du maximum) absents des reconstructions de SVRTK. Nous avons réglé α avec le juge hors échantillon (mêmes repliements, `s2s`, deux stacks pour reconstruire), en séparant **sujets de réglage** et **sujets d'évaluation**.

| Sujet de réglage | α = 0,01 | 0,03 | 0,05 | 0,1 | 0,2 | 0,3 | 0,5 |
|---|---|---|---|---|---|---|---|
| calme | 0,862 | 0,878 | **0,881** | 0,877 | 0,863 | 0,851 | |
| très agité | 0,630 | 0,675 | | 0,703 | 0,712 | 0,716 | **0,717** |

La meilleure moyenne des deux sujets de réglage est α = 0,1. Évaluation sur **quatre autres sujets** (deux calmes, deux agités), NCC moyenne du juge :

| Sujet | α = 0,01 | α = 0,1 | gain |
|---|---|---|---|
| calme 2 | 0,894 | 0,909 | +0,015 |
| calme 3 | 0,857 | 0,875 | +0,017 |
| agité 1 | 0,727 | 0,774 | +0,048 |
| agité 3 | 0,627 | 0,691 | +0,064 |

Gain moyen +0,036 (critère fixé avant : ≥ 0,01), positif pour 4 sujets sur 4 (critère : au moins 3), aucun sujet dégradé. Ces gains étaient plus grands que les écarts entre méthodes de la première évaluation, **faite avec α = 0,01** : elle a été refaite (tableau ci-dessus). **Réserves** : l'optimum varie avec le sujet (le sujet calme préfère ≈ 0,05, le très agité ≥ 0,3) ; une NCC sur des coupes jamais vues peut récompenser un volume trop lissé : à 0,1 l'image est visiblement plus floue qu'à 0,03. Une régularisation qui préserve les contours est la piste suivante.

## Zones vues par un seul stack

Sur les sujets très agités, les volumes montraient des « languettes » rectangulaires qui dépassent du cerveau et un liseré extérieur : ce sont des zones **vues par un seul stack** (environ 15 % du domaine pour le sujet agité examiné), alors qu'un voxel réellement dans le cerveau est atteint par chaque plan. Une première hypothèse (des coupes non recalées de bord) a été **infirmée** : elles ne représentent que 0,4 % des pixels de masque, et aucune des 59 coupes recalées n'a une corrélation simulée/réelle sous 0,7. La règle retenue (`--min-stack-coverage 2`, défaut) ne reconstruit que les voxels atteints par au moins deux stacks.

Vérifié sur le sujet agité à trois stacks (`s2s` et `s2v`) : les poses sont **identiques**, aucun voxel du nouveau volume n'est couvert par un seul stack (contre 14 à 15 % avant), et le volume change **près du bord retiré seulement** (écart relatif 24 % à moins de 2 voxels, 0,8 % à 4-8 voxels, 0,08 % au-delà de 8 voxels). La règle est testée de façon indépendante (comptage par stack, coupe volontairement éloignée de 25 mm, 8 mutants sur 8 détectés). **Le juge hors échantillon ne peut pas arbitrer cette règle** : il lit le volume aux pixels des coupes testées, et un domaine réduit y met des zéros ; les évaluations du juge ci-dessus ont été faites avant la règle.

## Régularisation préservant les contours

L'option expérimentale `--edge-kappa` remplace la pénalité quadratique de Tikhonov par celle de Huber (quadratique pour les petites différences entre voxels voisins, linéaire au-delà, résolue par moindres carrés repondérés itérés), pour lisser le bruit sans aplatir les contours. Les briques sont testées (poids égaux à 1 : identique à Tikhonov au bit près ; gradient égal aux différences finies ; poids de Huber exacts sur des cas calculés à la main ; κ immense : solution de Tikhonov).

- **Atlas simulé (vérité connue)**, réglage sur un atlas (STA21) et évaluation sur un autre (STA31), bruit de 5 % : Huber gagne **+2,1 dB de PSNR** (30,4 contre 28,3 dB) et +0,010 de NCC sur Tikhonov, avec une netteté plus proche de la vérité (0,996 contre 1,157). Réserve : l'optimum de Huber est au bord de la grille essayée.
- **Données réelles**, sujets de réglage (juge hors échantillon + indicateur de netteté `S` : gradient moyen du volume légèrement lissé, normalisé par l'intensité médiane du tissu ; `S` a été validé par dose-réponse sur des volumes floutés, après l'échec d'une première normalisation) : **aucun gain sur la frontière juge / netteté de Tikhonov**. Sujet calme : Tikhonov à α = 0,05 (juge 0,881, netteté 0,109) domine Huber à α = 0,1 et κ = 4 (0,878 ; 0,101). Sujet très agité : Huber à α = 0,3 et κ = 4 (0,714 ; 0,037) égale Tikhonov à la même netteté ; Huber à α = 0,1 est légèrement en dessous.

L'option reste donc désactivée par défaut. Les deux mesures (juge et `S`) sont des substituts, et le juge favorise le lissage : un gain de netteté réel n'est pas exclu, mais il n'est pas démontré. Autre constat : l'optimum de Tikhonov sur l'atlas simulé (α = 0,01) est dix fois plus petit que sur données réelles (0,1), une nouvelle illustration du fait que les simulations ne prédisent pas le réel.

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
