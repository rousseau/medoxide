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

**Les sujets très agités, à netteté égale.** Pour écarter le biais de lissage, nous comparons `s2s` et « sans correction » sur la **frontière juge / netteté** (plusieurs valeurs de α pour chaque méthode, et pour chaque valeur de `s2s` le juge de « sans correction » interpolé à la même netteté) : agité 2 : `s2s` +0,005 (équivalent) ; agité 3 : −0,016 (moins bon). Le biais de lissage n'explique donc pas l'échec à deux stacks. Avec **trois** stacks pour reconstruire (usage réel), les images de `s2s` et `s2v` montrent plus de structure que « sans correction » (anneau cortical, ventricules), la concordance aux intersections des poses est nettement meilleure que celle des poses d'en-tête, et l'évaluation à quatre stacks ci-dessous donne un net avantage à la correction : le protocole à deux stacks, très peu contraint pour un fœtus très agité, est trop dur pour trancher.

**À quatre stacks (trois pour reconstruire).** Un seul de nos sujets (agité 1) a quatre stacks TRUFI : chaque stack est mis de côté à son tour (quatre repliements) et les trois autres servent à reconstruire, ce qui est plus proche de l'usage réel que le protocole à deux stacks. Nos méthodes sont à α = 0,1 (couverture minimale 1, comme le juge l'exige), SVRTK et NeSVoR à leurs défauts :

| Méthode | Juge (moyenne des 4 repliements) | Netteté `S` |
|---|---|---|
| sans correction | 0,779 | 0,063 |
| `s2s` | 0,818 | 0,075 |
| `s2v` | **0,848** | 0,079 |
| SVRTK | 0,738 | 0,085 |
| NeSVoR | 0,769 | 0,081 |

La correction du mouvement **aide ici, et les volumes corrigés sont aussi plus nets** (ce n'est donc pas un effet de lissage). À netteté égale (frontière obtenue en variant α de « sans correction »), `s2s` dépasse « sans correction » de +0,051 (α = 0,1) et de +0,034 (α = 0,3) ; à α = 0,03, `s2s` est plus net que toute la frontière de « sans correction » (0,097 contre 0,082 au maximum) tout en gardant +0,045 de juge. `s2v` dépasse « sans correction » sur les quatre repliements (de +0,062 à +0,075) ; `s2s` échoue sur l'un d'eux (le stack sagittal mis de côté : 0,735 contre 0,761). À netteté comparable, nos volumes sont meilleurs que ceux de SVRTK et de NeSVoR (par exemple `s2s` à α = 0,03 : netteté 0,097, juge 0,806, contre 0,085 et 0,738 pour SVRTK, 0,081 et 0,769 pour NeSVoR). **Réserve** : un seul sujet, donc un indice et non une preuve ; il confirme que le protocole à deux stacks est trop dur pour un fœtus très agité.

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

## Pourquoi certains sujets très agités ne sont pas corrigés

Sur deux de nos six sujets (agité 2 et agité 3), aucune méthode (la nôtre, SVRTK, NeSVoR) ne prédit mieux des coupes jamais vues que de ne pas corriger, avec le protocole à deux stacks. Voici ce que les mesures établissent, dans l'ordre.

**1. Les méthodes ne s'accordent pas sur les poses.** Pour chaque coupe recalée, nous plaçons des pixels de masque avec la pose de chaque méthode, nous alignons rigidement les deux jeux de poses (le repère de sortie est libre) et nous rapportons l'écart quadratique par coupe. L'accord est de 0,5 à 1 mm en médiane sur les sujets calmes (1 à 21 % des coupes à plus de 3 mm d'écart), de 1 à 1,8 mm sur l'agité 1 (celui où la correction marche), de 3 à 4,6 mm sur l'agité 2 (46 à 89 % des coupes à plus de 3 mm) et de **6 à 13 mm sur l'agité 3 (85 à 100 %)** : les quatre méthodes y trouvent des poses différentes. Le mouvement estimé par rapport à l'en-tête y est de 5 à 12 mm en médiane (jusqu'à 29 mm au 90ᵉ centile pour SVRTK). SVRTK déplace des stacks entiers de 10 à 28 mm et garde peu de mouvement par coupe (1 à 3 mm) ; nos méthodes et NeSVoR trouvent des décalages de stack modestes (2,5 à 6 mm) mais un mouvement résiduel par coupe de 4 à 9 mm.

**2. La portée de capture, mesurée avec vérité connue.** Nous simulons des stacks d'un atlas avec un mouvement rigide par coupe, uniforme dans ±A (indépendant ou lisse avec sauts), et nous mesurons la part des coupes dont la pose est retrouvée à 1,5 mm ou moins (après retrait du mouvement commun) :

| Amplitude A | ±2 | ±4 | ±6 | ±8 | ±10 |
|---|---|---|---|---|---|
| `s2s`, indépendant | 96 % | 79 % | 8 % | 0 % | 0 % |
| `s2s`, lisse | | 73 % | 0 % | 0 % | 0 % |

La portée de capture d'un recalage local parti de l'en-tête est donc d'environ ±4 mm, alors que le mouvement estimé des sujets très agités est de 5 à 12 mm.

**3. Élargir la portée marche en simulation propre, pas sur le réel.** Des balayages préliminaires sur des coupes floutées (gros vers fin, σ = 6 pixels puis 3) portent la portée à environ ±8 mm : indépendant 92 / 74 / 59 % à ±4 / ±6 / ±8 (8 % et 0 % à ±6 et ±8 sans), lisse 93 / 79 / 49 % à ±4 / ±6 / ±8. Mais sur données réelles, avec le juge à quatre stacks, la moyenne tombe de 0,818 à **0,701** : l'élargissement dégrade nettement.

**4. Cause : les incohérences d'intensité basses fréquences entre stacks.** Le simulateur d'origine donne la même intensité à tous les stacks. Nous lui avons ajouté un **champ de biais multiplicatif lisse, fixe dans le repère du scanner (donc différent pour chaque stack à mouvement donné) et un gain global par stack**, d'amplitude 30 %. Résultat : `s2s` de base passe de 73 % à **3 %** à ±4 en mouvement lisse (79 % à 65 % en indépendant), et la passe grossière s'effondre (±4 indépendant : 92 % → 4 %), ce qui reproduit le comportement observé sur le réel. Un flou fait correspondre des formes lisses d'intensité, donc le champ de biais plutôt que l'anatomie.

**5. Une brique qui l'atténue : le passe-haut des profils.** Avant la corrélation, on soustrait à chaque profil son lissage gaussien (25 mm, restreint au masque) : les variations lentes d'intensité disparaissent. En simulation avec biais, ±4 : 83 % (indépendant, 65 % sans) et 59 % (lisse, **3 % sans**), et sans biais le résultat est inchangé (79 % et 73 %). Sur données réelles, le juge à quatre stacks passe de 0,818 à **0,833** (+0,015, et le repli qui échouait passe de 0,735 à 0,776) ; à deux stacks sur les six sujets, +0,002 en moyenne (5 sujets sur 6 positifs, le pire −0,007). Il est donc actif par défaut (`--s2s-highpass-mm 25`), pour environ 15 à 25 % de temps en plus. **Il ne suffit pas à rendre la passe grossière utilisable** : avec le passe-haut, la passe grossière reste mauvaise sur le réel (juge à quatre stacks 0,706, contre 0,833 pour le passe-haut seul) et en simulation avec biais (±6 : 4 %).

**Ce que cela dit.** Pour les sujets très agités, le mouvement dépasse la portée des recalages locaux, et la voie évidente pour l'élargir est bloquée par les incohérences d'intensité entre stacks. Les mesures suivantes ([alignement par stack](#alignement-par-stack-ou-par-paquets), [comparaison avec NiftyMIC](#comparaison-avec-niftymic)) montrent que la portée est dépassée pour les **décalages de stack** (jusqu'à 8-10 mm en simulation) mais pas pour le mouvement **par coupe** intra-stack de l'agité 3, où même NiftyMIC rejette les coupes. L'**estimation explicite du champ de biais de chaque stack** (SVRTK la fait ; NeSVoR l'a désactivée par défaut) a été ajoutée depuis ([section](#correction-du-champ-de-biais-par-stack)) ; le retour à la passe grossière a été essayé ([section](#retour-au-gros-vers-fin-après-la-correction-de-biais)) : il réussit en simulation et échoue encore sur le réel avec σ = 6 ; un flou adapté au réel (σ = 1) est activé par défaut. Tout ceci repose sur un simulateur dont le biais est un modèle simple (ondes planes de 100 à 200 mm) : il explique qualitativement l'échec, il ne le prouve pas.

## Combien de coupes sont mal recalées ?

`medx svr recon --quality fichier.tsv` écrit, pour chaque coupe, la concordance de ses profils d'intersection avant et après le recalage (`score_avant`, `score_apres`, plus grand = mieux), le nombre de balayages où elle a été recalée et le déplacement appliqué. Peut-on s'en servir pour **compter** les coupes mal recalées (erreur de pose > 1,5 mm) sans vérité terrain ? Critères fixés avant de regarder les résultats (`scripts/analyse_qualite_coupes.py`) : (1) AUC ≥ 0,8 pour détecter une coupe mal recalée ; (2) nombre prédit à ±25 % (ou ±3 coupes) du nombre vrai **sur chaque jeu**, avec un seuil appris sur les autres jeux (validation « un jeu laissé de côté »).

Mesuré sur 9 jeux simulés avec vérité connue (72 à 74 coupes chacun ; part de coupes mal recalées de 7 % à 100 %, 4 jeux sur 9 à 100 % dès ±6 mm) :

| Indicateur | AUC regroupée | AUC dans un jeu (min–max) |
|---|---|---|
| `score_avant` | 0,95 | 0,65–0,92 |
| `score_apres` | 0,94 | 0,50–0,93 |
| déplacement appliqué | 0,78 | 0,64–0,88 |
| gain (après − avant), nombre de balayages | ≈ 0,5 | — |

- **Critère 1 : atteint en apparence seulement.** L'AUC regroupée est dominée par les différences **entre** jeux (concordance 0,3–0,45 quand tout est mal recalé, 0,8–0,9 quand peu le sont) ; à l'intérieur d'un jeu, l'indicateur classe les coupes de façon bien moins sûre (jusqu'à 0,50).
- **Critère 2 : non atteint.** Avec le meilleur indicateur, 3 jeux sur 9 sont hors tolérance (30 coupes vraies mal recalées, 64 prédites ; 19 contre 10 ; 15 contre 10) : le niveau de concordance dépend du sujet (biais d'intensité, anatomie). **Le nombre n'est pas estimable de façon fiable ; le régime l'est.** Concordance moyenne ≥ 0,8 : au plus un quart des coupes mal recalées ; ≤ 0,45 : toutes.
- **Sur les six sujets réels**, concordance moyenne après `s2s` : 0,85 à 0,89 pour les trois calmes (0 à 3 % des coupes sous 0,65), 0,76 pour l'agité 1 (12 %), 0,60 pour l'agité 2 (54 %) et 0,57 pour l'agité 3 (79 %) : les deux derniers sont, par ce critère, dans un régime où une grande partie des coupes est mal recalée.
- **Une estimation indépendante** : NiftyMIC rejette les coupes de corrélation avec son modèle < 0,5, puis 0,65, puis 0,8 ([comparaison](#comparaison-avec-niftymic)). Il en rejette 0 sur 72 pour le sujet calme 1 (nous : 3 % sous 0,65), 12 % pour l'agité 1 (nous : 12 % ; l'AUC de notre concordance pour prédire son rejet est de 0,78), 80 % pour l'agité 2 (nous : 54 %) et 44 % pour l'agité 3 (nous : 79 %). Les deux estimations s'accordent sur le régime, pas sur le nombre.

## Retour au gros-vers-fin après la correction de biais

Le gros-vers-fin (`--s2s-coarse-sigma 6 --s2s-coarse-sweeps 2 --s2s-coarse-levels 2`) avait étendu la portée de capture en simulation propre mais s'était effondré avec un biais d'intensité, puis dégradé les données réelles. Hypothèse : le biais en était la cause ; il est maintenant corrigé. Critères fixés avant : à ±6 mm avec biais de 30 %, au moins 50 % des coupes à ≤ 1,5 mm (0 à 16 % avec le défaut actuel) ; sans biais, aucune baisse de plus de 5 points ; sur le réel, la concordance ne doit baisser de plus de 0,01 sur aucun sujet.

**Simulation : critères atteints.** Part des coupes à ≤ 1,5 mm, à ±6 mm, défaut actuel (alignement par stack et correction de biais) puis avec le gros-vers-fin :

| Jeu | défaut | + gros vers fin |
|---|---|---|
| lisse, biais 30 % | 0 % | **65 %** |
| indépendant, biais 30 % | 16 % | **68 %** |
| lisse, sans biais | 0 % | 40 % |
| indépendant, sans biais | 0 % | 68 % |

Le 90ᵉ centile reste élevé (14-16 mm) : une part des coupes échoue franchement. Au passage, une erreur de méthode a été corrigée : une première évaluation du défaut avait réutilisé d'anciens fichiers de sortie (le script réutilise les fichiers existants) et donnait 1 % ; la mesure directe donne 0 %.

**Réel : critère manqué, sur les six sujets.** Concordance aux intersections (toutes paires, images brutes) :

| | calme 1 | calme 2 | calme 3 | agité 1 | agité 2 | agité 3 | moyenne |
|---|---|---|---|---|---|---|---|
| défaut | 0,8764 | 0,9117 | 0,8985 | 0,7960 | 0,6922 | 0,5633 | 0,7897 |
| + gros vers fin | 0,8746 | 0,8982 | 0,8425 | 0,7002 | 0,6392 | 0,4909 | 0,7409 |

Baisse sur les six sujets, de −0,002 à **−0,096** (agité 1). L'accord des poses le confirme : sur l'agité 1, le gros-vers-fin s'écarte de NiftyMIC, SVRTK et NeSVoR (écart médian 1,8 à 2,3 mm, 29 à 33 % des coupes à plus de 3 mm) alors que le défaut s'en écarte de 0,7 à 1,3 mm (4 à 11 %).

**Ce que cela dit.** Une troisième fois, une méthode qui gagne en simulation (avec ou sans biais) échoue sur le réel : l'incohérence d'intensité basse fréquence n'était pas la seule cause, et la correction de biais ne suffit pas à rendre le gros-vers-fin utilisable. Il reste une propriété des données réelles absente du simulateur. Candidates, à tester une à une : variations d'intensité plus rapides que le champ polynomial (bandes de la séquence TRUFI, effets de B1), variations d'intensité d'une coupe à l'autre dans un stack (histoire de spin), différences de profil de coupe entre orientations, mouvement dans la coupe. Ce réglage (σ = 6) reste en échec sur le réel ; la section suivante en identifie la cause.

## Un flou adapté aux données réelles : la portée de capture mesurée avec une référence de consensus

**Un banc d'essai sur données réelles.** Pour un sujet calme, les poses de toutes les méthodes s'accordent à ~0,5 mm : on dispose d'une **référence de consensus**. Le diagnostic `diagnose_the_capture_range_on_real_data` perturbe ces poses par un mouvement rigide aléatoire indépendant par coupe (± *A* mm et ± *A* °), lance le recalage coupe-coupe depuis ces poses perturbées (stacks corrigés du biais, sans alignement par stack pour isoler le recalage par coupe) et mesure l'erreur par rapport à la référence. C'est la mesure de la portée de capture que la simulation ne pouvait pas donner (calme 2, part des coupes retrouvées à ≤ 1,5 mm) :

| Amplitude *A* | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| recalage fin seul | 100 % | 97 % | 86 % | 11 % | 1 % |
| gros-vers-fin σ = 6 (2 niveaux) | 97 % | 90 % | 62 % | 4 % | 0 % |

La portée réelle du recalage par coupe est d'environ **±2 mm**, contre ±4 mm en simulation (79-83 % à ±4 mm), et le flou σ = 6 la **réduit** à chaque amplitude (alors qu'il la portait à ±8 mm en simulation) : un flou de 6 pixels efface sur une image réelle, riche en détails (sillons), l'information qui guide le recalage. Partant de la référence (*A* = 0), le flou ne détruit pas la bonne solution (97 %, médiane 0,24 mm).

**Balayage du flou à *A* = 3 mm** (1 niveau, 2 balayages sauf indication) :

| Réglage | recalage fin seul | sans flou, +2 balayages (témoin) | σ = 0,5 | σ = 0,75 | **σ = 1** | σ = 1,5 | σ = 2 | σ = 3 |
|---|---|---|---|---|---|---|---|---|
| part ≤ 1,5 mm | 11 % | 20 % | 31 % | 46 % | **68 %** | 58 % | 46 % | 6 % |

Le témoin sans flou (les mêmes balayages supplémentaires) donne 20 % : le gain du flou (68 %) est bien dû au flou, avec un optimum autour de 1 à 1,5 pixel. Avec 3 balayages : 72 % (σ = 1). À *A* = 4 mm : 13 % (σ = 1, 3 balayages, 1 niveau) et **37 % avec 2 niveaux** (σ = 1 puis 0,5) ; à *A* = 6 mm, 0 % partout. La portée réelle passe donc d'environ ±2 mm à ±3 mm bien couvert et ±4 mm en partie.

**Vérification avec une référence indépendante et plusieurs graines.** Les tableaux ci-dessus sont des tirages uniques (une graine de perturbation) avec, comme référence, notre propre sortie précédente. Deux limites ensuite mesurées : la part des coupes retrouvées à ±4 mm varie énormément d'une graine à l'autre pour la même méthode (24 à 63 %), et une référence qui est la sortie de la méthode testée est circulaire. Refait avec la référence **SVRTK** (indépendante) et trois graines (7, 11, 13) à ±3 mm, part des coupes à ≤ 1,5 mm :

| | recalage fin seul | σ = 1 (3 balayages, 2 niveaux) | σ = 6 (2 balayages, 2 niveaux) |
|---|---|---|---|
| graine 7 | 32 % | 54 % | 6 % |
| graine 11 | 35 % | 80 % | 13 % |
| graine 13 | 21 % | 63 % | 25 % |
| **moyenne** | **29 %** | **66 %** | **15 %** |

L'ordre est le même pour chaque graine : σ = 1 est nettement meilleur que le recalage fin seul, lui-même meilleur que σ = 6. L'effet est plus modeste que sur les tirages uniques ci-dessus (66 % contre 29 %, et non 72 % contre 11 %) ; **ce sont ces chiffres qu'il faut retenir.** Les écarts de quelques points d'un tableau à l'autre (par exemple à ±4 mm) ne sont pas significatifs.

**Adopté par défaut : σ = 1 pixel, 3 balayages, 2 niveaux (σ = 1 puis 0,5).**

- **Réel, depuis l'en-tête, six sujets** (concordance aux intersections) : moyenne 0,797 → **0,811** (σ = 6 : 0,742), en hausse sur **les six sujets** : calmes +0,002 à +0,003, agité 1 +0,013, agité 3 +0,011, **agité 2 +0,051**.
- **Accord des poses avec les autres méthodes** : neutre à mitigé. Mieux sur l'agité 2 (NeSVoR 3,8 → 3,2 mm, SVRTK 2,3 → 1,9 mm), inchangé sur l'agité 1 (NiftyMIC 0,7 mm), un peu moins bon sur l'agité 3 (NeSVoR 4,0 → 4,5 mm). La concordance est l'objectif que `s2s` optimise : le banc d'essai à référence de consensus est la preuve non circulaire, la concordance est un indice.
- **Simulation (±4 mm, part ≤ 1,5 mm), défaut précédent → nouveau** : lisse avec biais 75 → 89 %, indépendant avec biais 79 → 82 %, lisse sans biais 83 → 88 %, indépendant sans biais 83 → 85 %. Aucune régression. À ±6 mm avec biais (lisse), 0 → 7 % (σ = 6 donnait 65 % en simulation, mais échoue sur le réel).
- **Coût** : environ 25 % de temps de calcul en plus.

**Ce que cela dit.** La quatrième occurrence de « le simulateur ne prédit pas le réel » a une cause identifiée : la portée de capture réelle est deux fois plus petite que celle du simulateur (le fantôme d'atlas est un tissu moyen, plus lisse que des images réelles). Le simulateur reste utile pour les régressions mais **le réglage du flou doit se faire sur le banc d'essai réel**. Il reste très en deçà des mouvements des sujets très agités (5-12 mm).

## Alignement par stack ou par paquets

Les recalages locaux ont une portée d'environ ±4 mm ([ci-dessus](#pourquoi-certains-sujets-très-agités-ne-sont-pas-corrigés)). Mais le mouvement d'un fœtus agité est en grande partie un **décalage de stack entier** (jusqu'à 5-6 mm sur l'agité 1) auquel s'ajoute un mouvement par coupe plus petit. La brique `--s2s-block-levels` ([description](algorithmes.md#estimation-du-mouvement)) estime d'abord une transformation rigide **commune** par bloc de coupes consécutives (niveau 0 : le stack entier ; niveau *l* : 2^*l* blocs), par recherche exhaustive puis Adam, avant les balayages par coupe. Elle est **activée par défaut à 1 niveau** (voir la décision plus bas).

**Critère fixé avant, en simulation** (stacks d'un atlas, décalage rigide par stack, sans mouvement par coupe ; `evaluate_simulated_motion.py`) : la part de coupes à ≤ 1,5 mm doit être nettement au-dessus de celle du recalage par coupe seul.

| Décalage de stack, biais d'intensité | par coupe seul | 1 niveau (stack entier) | 3 niveaux |
|---|---|---|---|
| 5°/4 mm, biais 30 % | 3 % | **92 %** | 95 % |
| 10°/8 mm, sans biais | 0 % | **88 %** | 88 % |
| 10°/8 mm, biais 30 % | 0 % | 3 % | **82 %** |

Critère atteint. Le biais d'intensité reste le facteur limitant : avec 10°/8 mm **et** un biais, le stack entier seul échoue (3 %) et la hiérarchie à 3 niveaux le rattrape (82 %). Sur le fantôme de blobs du test unitaire, le critère a dû être révisé explicitement (décalage de 3°/3 mm, erreur médiane ≤ 1,5 mm, au lieu de 8 mm : le fantôme est trop pauvre en structure) ; le grand décalage est jugé sur l'atlas.

Le code est testé par mutation (30 mutants : grille de recherche, choix translation puis rotation, point de départ d'Adam, moyenne de la concordance, partition en blocs, ancre du premier stack, seuil de 3 coupes, garde d'acceptation, ordre de composition, pivot et échelle, câblage des options) : le premier jeu de tests n'en détectait que 16, trois tests ont été ajoutés (recherche sur une fonction synthétique, recherche contre descente courte, propriétés d'un niveau : transformation commune, égalité avec l'appel direct à 1e-17, blocs trop petits ignorés), et **29 sont détectés** ; le survivant est équivalent (`||` au lieu de `&&` dans la condition de recherche : avec une étendue nulle la grille ne contient que 0).

**Portée sur données réelles, avec référence connue** (calme 2, référence SVRTK, perturbation d'un mouvement rigide **par stack** de ± *A* mm et ± *A* °, trois graines ; part des coupes à ≤ 1,5 mm) : à *A* = 4, sans alignement par stack 85 / 96 / 11 %, avec 96 / 96 / 90 % ; à *A* = 8, sans 3 / 76 / 0 %, avec **94 / 94 / 0 %**. Pour la graine qui échoue, la perturbation vaut des rotations de 10-11° et des translations de 19-20 mm, au-delà de la grille de ±12 mm. Élargir la recherche à ±20 ou ±28 mm ne la rattrape pas (7 %), ni une recherche multi-départs à la FLIRT (grille de rotations, translations optimisées à chaque départ : 0 %) : à ces décalages le coût lui-même préfère une mauvaise solution (les stacks se recouvrent à peine). L'alignement par stack actuel suffit donc pour les décalages réalistes (translations jusqu'à 13-16 mm) ; la recherche multi-départs de bloc reste une option désactivée (`block_multistart_deg`), sans gain mesuré.

**Sur données réelles** :

- **Juge à quatre stacks, agité 1** (trois stacks pour reconstruire, α = 0,1) : défaut 0,833 ; **1 niveau 0,849** (+0,016, en hausse sur les quatre replis : +0,006, +0,002, +0,029, +0,029) ; 3 niveaux 0,847 pour un temps de calcul triple. La netteté `S` ne baisse pas (0,0781 et 0,0760 contre 0,0758).
- **Accord des poses avec les méthodes indépendantes** : sur l'agité 1, les décalages de stack de `s2s` passent de 5,2/3,1/2,1/1,7 mm à 5,4/3,9/2,0/2,6 mm (NiftyMIC 5,7/4,1/2,1/2,6 ; SVRTK 6,0/3,9/2,0/2,6) et l'écart avec NiftyMIC de 1,1 mm (90ᵉ centile 5,1) à **0,7 mm (1,3)**. Sur l'agité 2, l'écart avec SVRTK passe de 3,2 à 2,3 mm (51 % à 37 % des coupes à plus de 3 mm) et la concordance moyenne de 0,60 à **0,68**. Sur le sujet calme, aucune régression (concordance 0,85 → 0,86 ; écart avec NiftyMIC 0,4 mm, 0 % au-delà de 3 mm).
- **Agité 3 : toujours indécidé.** La concordance ne bouge pas (0,57 → 0,56). Avec une recherche de ±30 mm, l'alignement retrouve les décalages de SVRTK (17,7/22,4 mm contre 18,5/21,0), alors que NiftyMIC, NeSVoR et `s2s` en trouvent 4 à 5 mm ; notre concordance préfère la solution à 4-5 mm (0,57 contre 0,47), mais c'est ce que `s2s` optimise. Deux familles de solutions, et rien dans nos mesures (trois stacks, pas de vérité terrain) ne permet d'arbitrer.

**Mesure ajoutée avant de décider : trois sujets calmes, juge à 2 stacks** (un stack mis de côté tour à tour, deux pour reconstruire, α = 0,1, couverture 1 ; `benchmark_variantes_s2s.py`, ordre sans/avec alterné d'un repli à l'autre). Critères fixés avant : écart du juge moyen (avec − sans blocs) ≥ −0,01 sur **chaque** sujet, netteté `S` non dégradée de plus de 5 %.

| Sujet calme | Juge sans blocs | Juge avec 1 niveau | Écart (par repli) | Netteté S (avec / sans) | Temps CPU (avec / sans) |
|---|---|---|---|---|---|
| 1 | 0,8784 | 0,8775 | −0,001 (+0,003 ; −0,011 ; +0,005) | 101 % | 375 s / 256 s (×1,46) |
| 2 | 0,9112 | 0,9168 | +0,006 (+0,003 ; −0,004 ; +0,017) | 102 % | 338 s / 229 s (×1,48) |
| 3 | 0,8766 | 0,8970 | +0,020 (+0,008 ; +0,048 ; +0,005) | 102 % | 294 s / 192 s (×1,53) |

Critères atteints sur les trois sujets (un seul repli, sur 9, passe sous −0,01 : −0,011). Le temps est le temps CPU (utilisateur + système), le temps mural n'ayant pas de sens sur une machine partagée.

**Décision.** Brique démontrée en simulation, utile sur le sujet agité à quatre stacks (juge) et sur deux sujets agités (accord des poses), sans régression sur trois sujets calmes (juge) : **elle est activée par défaut, à 1 niveau** (stack entier), pour environ 1,5 fois le temps de calcul. Les 3 niveaux restent en option (utiles seulement en simulation avec décalage de stack **et** biais d'intensité ; sur le réel, 0,847 contre 0,849 pour trois fois plus de temps). Réserve : sur le sujet le plus agité (agité 3), l'alignement ne l'aide pas et s'éloigne du consensus de NiftyMIC et NeSVoR ; une activation conditionnelle (garder la variante de meilleure concordance finale) reste possible mais est circulaire. Les résultats des autres sections de ce chapitre qui précèdent cette décision ont été obtenus **sans** alignement par stack. **Correctif ultérieur** : en présence d'un biais d'intensité (simulation), l'alignement par stack régresse (59 → 3 %) ; la [correction de biais](#correction-du-champ-de-biais-par-stack), activée par défaut avec lui, le rend sûr.

## Correction du champ de biais par stack

Le [passe-haut des profils](#pourquoi-certains-sujets-très-agités-ne-sont-pas-corrigés) **ignore** les variations lentes d'intensité entre stacks sans les **corriger**. La brique `--s2s-bias-rounds` ([description](algorithmes.md#estimation-du-mouvement)) les estime : un champ multiplicatif polynomial par stack (degré 3, fixe dans le repère du scanner), par moindres carrés dans le log sur les intersections de coupes, sans volume intermédiaire ; les stacks sont divisés par leur champ avant le recalage, et le volume final est reconstruit à partir des stacks corrigés. Elle est **activée par défaut** (une alternance), avec une garde.

**Ce qu'on peut estimer, et ce qu'on ne peut pas.** Le champ n'est identifiable qu'**à un champ commun près, attaché à l'anatomie** : ajouter à tous les stacks un même champ lisse de la position vraie ne change aucun rapport d'intensité entre coupes qui se croisent (et un polynôme reste un polynôme après un mouvement rigide). Sur des données exactes, l'erreur absolue du champ est donc de 0,09-0,10 en écart type alors que celle du champ relatif entre stacks est de 0,04. Le critère fixé d'abord (corrélation avec le vrai champ) portait sur une quantité non identifiable ; il a été remplacé par l'**incohérence d'intensité aux intersections** (écart quadratique moyen du log du rapport des intensités de deux coupes qui se croisent). Sur l'atlas avec un biais de 30 %, avec les poses vraies, elle passe de 0,358 à 0,088, soit exactement le plancher sans biais (0,090) ; avec les poses d'en-tête, de 0,446 à 0,287 (plancher dominé par les erreurs de pose : 0,289 sans biais).

**La garde « ne pas nuire ».** Sans biais, avec des poses fausses de 4 mm, l'estimateur ajuste la structure parasite d'anatomie comme si c'était du biais, et dégrade l'estimation du mouvement : sans garde, 83 % → 55 % (lisse) et 83 % → 56 % (indépendant) des coupes à ≤ 1,5 mm. La correction n'est donc appliquée que si elle réduit l'incohérence d'au moins 10 % (`--s2s-bias-min-reduction`, seuil fixé avant de le confronter à ces jeux). Réduction mesurée avec les poses d'en-tête : 0,4 % sans biais, 35,7 % avec un biais de 30 % ; sur les six sujets réels, 29 % (calme 1), 2 % (calme 2), 6 % (calme 3), 17 % (agité 1), 3 % (agité 2) et 20 % (agité 3) : la correction est appliquée pour trois d'entre eux.

**Simulation** (part des coupes à ≤ 1,5 mm, ±4 mm, avec l'alignement par stack par défaut) :

| Jeu | sans alignement par stack | alignement par stack seul | alignement + biais |
|---|---|---|---|
| lisse, biais 30 % | 59 % | **3 %** | **75 %** |
| indépendant, biais 30 % | 83 % | 71 % | **79 %** |
| lisse, sans biais | (73 %) | 83 % | 83 % |
| indépendant, sans biais | (79 %) | 83 % | 83 % |

Deux constats. (1) **L'alignement par stack, activé par défaut, régresse en présence d'un biais** (59 → 3 % et 83 → 71 %) : l'incohérence d'intensité le fait aller vers de mauvais décalages de stack ; la correction de biais le rend sûr (75 % et 79 % ; 79 % reste un peu sous les 83 % du passe-haut seul, écart de l'ordre du bruit sur 72 coupes). Cette régression n'avait pas été vue avant le changement de défaut, qui reposait sur des simulations sans biais et sur les données réelles. (2) Sans biais, la garde laisse le résultat strictement identique (83 % et 83 %). Les valeurs entre parenthèses sont des mesures antérieures sans alignement par stack.

**Données réelles.** Le juge hors échantillon est **invalide pour cette brique** : il compare les coupes du stack mis de côté (non corrigées de son propre biais) à un volume reconstruit à partir de stacks corrigés, et le NCC chute mécaniquement (premier repli à quatre stacks : 0,61 contre 0,85) ; il pénalise par construction ce qu'on corrige. Mesures utilisées à la place, qui ne dépendent que des poses (la concordance aux intersections est calculée sur les images brutes, équitable pour toutes les poses) :

| Concordance (toutes paires) | calme 1 | calme 2 | calme 3 | agité 1 | agité 2 | agité 3 | moyenne |
|---|---|---|---|---|---|---|---|
| alignement par stack seul | 0,8784 | 0,9087 | 0,9004 | 0,7936 | 0,6941 | 0,5647 | 0,7900 |
| + correction de biais (régularisation 0,001) | 0,8770 | 0,9087 | 0,9004 | 0,7967 | 0,6941 | 0,6070 | 0,7973 |
| + correction de biais (réglage final : 0,1 et écrêtage) | 0,8760 | 0,9085 | 0,8985 | 0,7986 | 0,6922 | **0,6350** | 0,8015 |

Aucune baisse supérieure à 0,01 sur les six sujets (le pire : −0,0014) ; les trois sujets où la garde écarte la correction sont inchangés. Sur l'**agité 3**, que l'alignement par stack faisait dériver, les décalages de stack passent de 8,0/9,8 mm à **4,7/5,2 mm** (réglage final ; NiftyMIC 4,2/4,7 ; NeSVoR 4,6/5,3) et l'écart médian avec NiftyMIC de 7,9 à **2,3 mm** (90ᵉ centile 15,6 → 7,4) ; avec SVRTK, qui trouve 18-21 mm, l'écart reste de 17 mm. L'**incohérence d'intensité entre stacks était donc bien la cause** de la dérive observée sur ce sujet. Sur l'agité 1, l'accord avec NiftyMIC est inchangé (0,7 mm), sur le calme 1 et le calme 2 il l'est aussi. Le temps de calcul est inchangé dans le bruit des mesures (quelques secondes pour l'estimation du champ).

**Défaut découvert par la figure, et corrigé.** Une première version (régularisation de 0,001) donnait, sur la figure des reconstructions, des volumes **assombris avec un effet de vignettage** (centre sombre, périphérie claire) sur trois sujets, alors que la concordance, mesurée sur les poses, était bonne. Cause : le polynôme de degré 3 **diverge hors des zones contraintes par des intersections** (champ jusqu'à exp(2,4) sur un sujet calme et exp(10), soit ×22 000, sur l'agité 3), ce qui change l'intensité de quelques voxels de masque. Remèdes : régularisation portée à 0,1 (la réduction d'incohérence est conservée : 5, 29, 15 et 17 % contre 6, 29, 17 et 20 % sur quatre sujets) et **écrêtage du champ à la plage vue aux points d'intersection** (aucune extrapolation) ; la plage reste alors comprise entre 0,13 et 0,51 en log (facteur ≤ 1,7). Effet sur les poses : simulation 65 % → 75 % (lisse, biais) et 86 % → 79 % (indépendant, biais) ; réel : concordance moyenne 0,794 → 0,802, agité 3 0,595 → 0,635, aucune baisse. Leçon : les mesures fondées sur les poses (concordance, accord) ne voient pas un défaut d'intensité du volume ; il faut regarder les volumes.

**Réserve.** La brique corrige un champ polynomial de basse fréquence supposé multiplicatif ; les effets d'intensité qui dépendent de la coupe (perte de signal localisée) ne sont pas corrigés (le seuil d'intensité les ignore). Un champ commun à tous les stacks (la sensibilité de la bobine, la même pour tous) est aussi invisible, comme on l'a vu, et n'est pas corrigé.

Le code est testé par mutation (14 mutants du câblage et de la garde, 19 de l'estimateur et 5 de l'écrêtage).

## Régularisation temporelle

**Idée.** En TRUFI les coupes d'un stack sont acquises l'une après l'autre (≈ 1 s) : deux coupes voisines dans l'espace sont voisines dans le temps, donc leurs mouvements sont presque égaux. Le coupe-coupe, qui estime chaque coupe seule, ne l'utilise pas ; une coupe dont les intersections sont peu informatives dérive. Le coût d'une coupe devient la concordance moins λ · moyenne |P′ − Q|² (mm⁻² pour λ), où P′ sont 64 points du masque déplacés par la pose candidate et Q les positions que donnerait le mouvement relatif de ses voisines directes (k ± 1, recalées elles aussi). Voir `--s2s-temporal-weight` et `docs/RECHERCHE_MULTISTART.md` (§ 11).

**Critères fixés avant la mesure du défaut** (meilleur λ parmi 0,005 / 0,01 / 0,02 ; le plus petit qui satisfait tout) : (1) banc lisse ±6 mm ≥ 80 % des coupes à ≤ 1,5 mm en moyenne ; (2) banc indépendant ±3 mm ≥ 85 % ; (3) réel : concordance moyenne ≥ celle du défaut − 0,005 ; (4) accord des poses avec SVRTK, NeSVoR et NiftyMIC non dégradé ; (5) temps ≤ +10 %.

| λ | lisse ±6 mm | lisse ±4 mm | indépendant ±3 mm | concordance réelle (6 sujets) |
|---|---|---|---|---|
| 0 | 23 % | 68 % | 66 % | 0,8148 |
| 0,005 | 74 % ✗ | 89 % | 88 % | 0,8184 |
| **0,01** | **85 %** | 90 % | **90 %** | **0,8125** |
| 0,02 | 90 % | 90 % | 88 % | 0,8003 ✗ |
| 0,1 | 36 % | 88 % | — | — |

Banc : un sujet réel (géométrie des stacks et contenu), référence SVRTK, moyenne sur trois graines (7, 11, 13) ; la variance entre graines est de ±15 points. À λ = 0,1 le terme domine la concordance et la méthode échoue. **λ = 0,01 est le défaut.** Accord des poses à λ = 0,01 (médiane / part à plus de 3 mm, par rapport au défaut) : égal ou meilleur sur cinq sujets sur six (par exemple NiftyMIC sur l'agité 3 : 41 → 12 % de coupes à plus de 3 mm) ; une dégradation marginale sur l'agité 3 avec NeSVoR (38 → 44 %, médiane inchangée, 34 coupes). Temps CPU sur ce sujet, deux répétitions : 848 s contre 852 s, puis 866 s contre 942 s, soit +5 % en moyenne (+0,5 % et +9 % : le coût du terme est faible mais la mesure est bruitée ; critère ≤ +10 % tenu en moyenne).

Le code est testé par mutation : 10 mutants du câblage (signe du terme, normalisation, poids transmis, cibles, voisines précédente et suivante, composition des poses, filtre des voisines recalables, pas d'échantillonnage, bord du stack, activation) sont tous tués. Un premier passage en laissait survivre cinq : le test de câblage ne vérifiait que l'existence d'un effet ; la construction des cibles a été isolée dans une fonction testée sur des valeurs exactes, et un test impose qu'une coupe laissée à l'identité dans un stack décalé soit tirée vers ses voisines.

**Banc semi-synthétique à contenu réel (E2).** Les coupes sont simulées depuis une reconstruction réelle (géométrie des vrais stacks, mouvement vrai connu, bruit de 5 %), ce qui donne une vérité exacte sur du contenu d'image réel ; le banc est validé (sans correction, 0 % des coupes à ≤ 1,5 mm de la vérité pour 4 mm ; 95 % à 2 mm avec le réglage par défaut). Part des coupes à ≤ 1,5 mm, moyenne de trois graines, sans alignement par stack :

| Variante | indépendant ±3 mm | lisse ±4 mm | lisse ±6 mm | sauts ±4 mm |
|---|---|---|---|---|
| λ = 0 | 56 % | 64 % | 8 % | 30 % |
| **λ = 0,01 (défaut)** | 42 % | 83 % | 43 % | 76 % |
| λ = 0,02 | 20 % | 80 % | 18 % | 64 % |
| robuste, `c` fixe 2 mm | 75 % | 77 % | 3 % | 35 % |
| robuste, `c` de 8 à 2 mm | 85 % | 85 % | 34 % | 77 % |

Avec l'alignement par stack de production, `b=1` seul donne 56 / 54 / 8 / 14 % et `b=1` avec λ = 0,01 donne 45 / 82 / 29 / 61 % : l'a priori garde un grand intérêt. Les deux limites sont connues : (1) **un mouvement indépendant d'une coupe à l'autre** (peu physique en TRUFI, une coupe durant environ 1 s) fait perdre 11 à 14 points à l'a priori quadratique ; (2) la pénalité robuste à `c` décroissant règle (1) mais **échoue sur le banc de portée réel** (lisse ±6 mm : 45 % contre 85 %), car un petit `c` en fin de balayage ne tire plus les coupes encore éloignées de leurs voisines ; elle reste une option désactivée. Des niveaux de blocs supplémentaires (2 ou 3 niveaux) ne sont pas retenus (lisse ±6 mm : 14 et 8 % contre 29 %). **Six sujets (banc E2, SVRTK ou reconstruction propre comme contenu, trois graines).** Gain de l'a priori λ = 0,01 sur l'alignement par stack seul, en points de part des coupes à ≤ 1,5 mm : lisse ±6 mm +22,6 en moyenne (de +5 à +46 selon le sujet), sauts ±4 mm +12,9 (de +2 à +47), mouvement indépendant ±3 mm −9,2 (de −28 à +17). Les critères fixés avant la mesure (gain moyen ≥ +10 en lisse, ≥ +5 en sauts, perte moyenne ≤ 15 en indépendant, aucun sujet en perte en lisse ou en sauts) sont tous tenus. Le gain est d'autant plus grand que la base est mauvaise (jusqu'à +46 sur un sujet agité), et il n'y a pas de marge quand elle atteint déjà le plafond. **Juge hors échantillon et netteté, six sujets (sans correction de biais, trois replis par sujet).** Le juge moyen passe de la base (λ = 0) à λ = 0,01 avec un écart de +0,0170 (de +0,0003 à +0,0598 selon le sujet : neutre sur les sujets calmes, +0,017 à +0,060 sur les trois sujets agités) et la netteté de +1,2 % : l'a priori ne lisse donc pas le volume, il améliore la prédiction d'un stack mis de côté. Aucun sujet n'est en perte. **Où échouent les coupes restantes ?** Sur le banc lisse ±6 mm (production), les échecs forment des plages contiguës de 3 à 13 coupes, plutôt au début ou à la fin d'un stack ; le succès va de 15 % au premier quintile du stack à 69 % au quatrième pour un même déplacement vrai, et les coupes de petite surface de masque (bords du volume) réussissent à 26 % contre 45-47 %. Pondérer l'a priori par le nombre d'échantillons de chaque coupe (`--s2s-temporal-info-exponent`) ne corrige pas ce défaut (pas de gain en lisse ±6 mm, perte jusqu'à 9 points en sauts) : l'option reste désactivée. **Raffinement coupe-volume après le coupe-coupe (option, non retenue).** Sur le banc semi-synthétique à vérité exacte, deux cycles coupe-volume (reconstruction à 2 mm) après le coupe-coupe font passer la part de coupes à ≤ 1,5 mm de 45 à 87 % (mouvement indépendant), de 29 à 77 % (lisse ±6 mm) et de 61 à 88 % (sauts) sur un sujet, et de 76 à 91 % (lisse ±6 mm) sur un second à contenu SVRTK, sans dégrader la médiane. Sur six sujets réels, en revanche, l'accord des poses avec SVRTK, NeSVoR et NiftyMIC baisse légèrement (part à plus de 3 mm : 20,6 → 24,2 % en moyenne, avec une dégradation marquée sur un sujet agité) et la concordance aux intersections de 0,813 à 0,773 : les critères fixés avant la mesure ne sont pas tenus, l'option `--s2v-refine-cycles` reste donc désactivée. C'est la quatrième fois qu'un simulateur partageant le modèle direct de l'estimateur est plus favorable que le réel. L'historique complet des critères fixés avant chaque mesure est dans `docs/RECHERCHE_MULTISTART.md` (§ 11 à 15). Les tests de mutation couvrent aussi l'option robuste (11 mutants tués : formule, échelle décroissante, transmission de la configuration).

**Limite.** Un saut de mouvement entre deux coupes voisines (le fœtus bouge au milieu d'un stack) est pénalisé par le terme ; la brique est faite pour un mouvement lisse par morceaux, et le banc à sauts (semi-synthétique, § 12 du document de recherche) doit mesurer ce que coûte un saut.

## Comparaison avec NiftyMIC

NiftyMIC ([Ebner et al., 2020](algorithmes.md#références)) fait un recalage volume-vers-volume des stacks (recherche jusqu'à 45°), puis un recalage coupe-volume hiérarchique, une correction d'intensité par stack et un rejet des coupes aberrantes. Il est exécuté dans Docker (image `renbem/niftymic`, amd64 émulé : 9 à 13 minutes par sujet ; convention validée sur cinq des six sujets, imposée pour le sixième dont le volume n'a pas été écrit), réglages par défaut sauf la résolution (0,8 mm), la correction du champ de biais (désactivée) et l'utilisation des masques dans la reconstruction (`--use-masks-srr 1`, voir ci-dessous) ; ses transformations par coupe (`.tfm` ITK, LPS) sont lues par `scripts/read_niftymic_poses.py`. **Convention validée par mesure**, comme pour SVRTK : la corrélation entre les pixels de masque d'une coupe et le volume de NiftyMIC échantillonné à sa pose est de 0,935 (sujet calme 1) et 0,921 (agité 1) avec la transformation par coupe lue telle quelle, contre 0,652 et 0,376 sans correction et 0,44 avec l'inverse.

| Sujet | Ce que fait NiftyMIC | Accord avec les autres méthodes |
|---|---|---|
| Calme 1 (3 stacks) | 0 coupe rejetée sur 72 | 0,4-0,6 mm en médiane, comme entre les quatre autres |
| Calme 2 (3 stacks) | garde 71 coupes sur 82 | non mesuré (volume et concordance seulement) |
| Calme 3 (3 stacks) | **rejette tout le stack sagittal** (24/24) et la moitié des autres : garde 45 coupes sur 98 | non mesuré ; son volume garde des zones non recalées hors du cerveau |
| Agité 1 (4 stacks) | 12 % de coupes rejetées | **le plus proche de `s2v` (0,7 mm, 90ᵉ centile 1,5) et de SVRTK (0,9 ; 1,9)** ; `s2s` 1,1 (5,1) ; NeSVoR 1,1 (4,2) |
| Agité 2 (3 stacks) | rejette 80 % des coupes (au dernier seuil : 24/25, 14/26 et 21/21 coupes par stack) ; le stack sagittal n'a plus aucune transformation | impossible à comparer sur tout le sujet |
| Agité 3 (3 stacks) | rejette tout le stack coronal ; la reconstruction finale plante avant d'écrire le volume (2 h 46 de recalage, même avec `--use-masks-srr 1`) | **proche de NeSVoR** (1,7 mm, 31 % des coupes à plus de 3 mm ; décalages de stack 4,1/5,1 contre 4,6/5,3) ; SVRTK s'en écarte de 18 mm |

**Un piège de réglage, corrigé : les masques de NiftyMIC.** Par défaut (`use-masks-srr = 0`), NiftyMIC n'utilise les masques cérébraux que pour le recalage : sa reconstruction finale part des images **entières**, si bien que le tissu non cérébral, qui ne bouge pas comme le cerveau, entre dans le volume (bandes brillantes, doublons). Une première série de volumes, lancée ainsi, avait fait croire à des zones hors du cerveau « non recalées ». Refait avec `--use-masks-srr 1` (les mêmes masques fetalBET que pour toutes les méthodes), les volumes sont propres ; la figure applique de plus le même masque à toutes les colonnes (le domaine de notre reconstruction, issu des masques fetalBET) et recale les volumes avec une métrique évaluée dans ce masque.

**Concordance aux intersections, mêmes coupes pour tous** (figure de reconstructions avec tous les stacks, réglages par défaut actuels de nos méthodes ; NiftyMIC mesuré sur les coupes qu'il a gardées, qui sont les mieux recalées, donc à son avantage ; réserve : notre coupe-coupe optimise lui-même cette concordance) : sur les coupes que NiftyMIC garde, notre coupe-coupe est supérieur ou égal pour chaque sujet (calmes 1, 2 et 3 : 0,867, 0,901 et 0,851 pour NiftyMIC contre 0,879, 0,912 et 0,881 ; agité 1 : 0,789 contre 0,838 ; agité 3 : 0,764 contre 0,828). Pour l'agité 2 il ne reste que 19 coupes, sans paire de stacks différents à mesurer. L'accord des poses avec NiftyMIC est de 0,4 mm en médiane sur le calme 1 et de 0,7 mm (90ᵉ centile 1,5) sur l'agité 1. Visuellement, ses volumes sont plus texturés que les nôtres sur les sujets calmes (nous sommes plus lisses à α = 0,1) ; celui de l'agité 2 est médiocre (19 coupes gardées) et celui de l'agité 3 manque (voir ci-dessus).

Ce que cela apporte à la question « pourquoi n'arrive-t-on pas à estimer le mouvement de certaines coupes ? » : **la méthode la plus complète du lot échoue aussi** sur les deux sujets très agités et sur un sujet calme (stack sagittal rejeté en entier), et de façon explicite (il rejette les coupes ou des stacks entiers au lieu d'en proposer une pose). Sur l'agité 1, où le mouvement est à la limite de la portée, les méthodes qui font un alignement de stack (NiftyMIC, SVRTK, `s2v` avec son initialisation) s'accordent, et `s2s` s'en rapproche quand on lui ajoute l'alignement par stack. Sur l'agité 3, deux familles de solutions coexistent (décalages de stack de 4-5 mm ou de 18-22 mm). Réserve : une seule exécution par sujet, réglages par défaut ; une correction d'intensité de NiftyMIC a échoué sur un stack de l'agité 2 (bibliothèque absente de l'image Docker), le traitement a continué.

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
`run_svrtk.py`, `run_nesvor.py`, `read_niftymic_poses.py`, `benchmark_real.py`, `benchmark_4stacks.py`, `benchmark_variantes_s2s.py`, `analyse_accord_poses.py`, `analyse_qualite_coupes.py`, `analyse_juge_sujets.py` et `intersection_consistency.py`.
