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

**Ce que cela dit.** Pour les sujets très agités, le mouvement dépasse la portée des recalages locaux, et la voie évidente pour l'élargir est bloquée par les incohérences d'intensité entre stacks. Les mesures suivantes ([alignement par stack](#alignement-par-stack-ou-par-paquets), [comparaison avec NiftyMIC](#comparaison-avec-niftymic)) montrent que la portée est dépassée pour les **décalages de stack** (jusqu'à 8-10 mm en simulation) mais pas pour le mouvement **par coupe** intra-stack de l'agité 3, où même NiftyMIC rejette les coupes. L'**estimation explicite du champ de biais de chaque stack** (SVRTK la fait ; NeSVoR l'a désactivée par défaut) a été ajoutée depuis ([section](#correction-du-champ-de-biais-par-stack)) ; la brique suivante est le retour à la passe grossière. Tout ceci repose sur un simulateur dont le biais est un modèle simple (ondes planes de 100 à 200 mm) : il explique qualitativement l'échec, il ne le prouve pas.

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
| lisse, biais 30 % | 59 % | **3 %** | **65 %** |
| indépendant, biais 30 % | 83 % | 71 % | **86 %** |
| lisse, sans biais | (73 %) | 83 % | 83 % |
| indépendant, sans biais | (79 %) | 83 % | 83 % |

Deux constats. (1) **L'alignement par stack, activé par défaut, régresse en présence d'un biais** (59 → 3 % et 83 → 71 %) : l'incohérence d'intensité le fait aller vers de mauvais décalages de stack ; la correction de biais le rend sûr (65 % et 86 %, au-dessus du passe-haut seul). Cette régression n'avait pas été vue avant le changement de défaut, qui reposait sur des simulations sans biais et sur les données réelles. (2) Sans biais, la garde laisse le résultat strictement identique (83 % et 83 %). Les valeurs entre parenthèses sont des mesures antérieures sans alignement par stack.

**Données réelles.** Le juge hors échantillon est **invalide pour cette brique** : il compare les coupes du stack mis de côté (non corrigées de son propre biais) à un volume reconstruit à partir de stacks corrigés, et le NCC chute mécaniquement (premier repli à quatre stacks : 0,61 contre 0,85) ; il pénalise par construction ce qu'on corrige. Mesures utilisées à la place, qui ne dépendent que des poses (la concordance aux intersections est calculée sur les images brutes, équitable pour toutes les poses) :

| Concordance (toutes paires) | calme 1 | calme 2 | calme 3 | agité 1 | agité 2 | agité 3 | moyenne |
|---|---|---|---|---|---|---|---|
| alignement par stack seul | 0,8784 | 0,9087 | 0,9004 | 0,7936 | 0,6941 | 0,5647 | 0,7900 |
| + correction de biais | 0,8770 | 0,9087 | 0,9004 | 0,7967 | 0,6941 | **0,6070** | 0,7973 |

Aucune baisse supérieure à 0,01 sur les six sujets (le pire : −0,0014) ; les trois sujets où la garde écarte la correction sont inchangés. Sur l'**agité 3**, que l'alignement par stack faisait dériver, les décalages de stack passent de 8,0/9,8 mm à **4,1/5,2 mm** (NiftyMIC 4,1/5,1 ; NeSVoR 4,6/5,3) et l'écart médian avec NiftyMIC de 7,9 à 3,6 mm (90ᵉ centile 15,6 → 6,8) ; avec SVRTK, qui trouve 18-21 mm, l'écart reste de 17 mm. L'**incohérence d'intensité entre stacks était donc bien la cause** de la dérive observée sur ce sujet. Sur l'agité 1, l'accord avec NiftyMIC est inchangé (0,7 mm), sur le calme 1 et le calme 2 il l'est aussi. Le temps de calcul est inchangé dans le bruit des mesures (quelques secondes pour l'estimation du champ).

**Réserve.** La brique corrige un champ polynomial de basse fréquence supposé multiplicatif ; les effets d'intensité qui dépendent de la coupe (perte de signal localisée) ne sont pas corrigés (le seuil d'intensité les ignore). Un champ commun à tous les stacks (la sensibilité de la bobine, la même pour tous) est aussi invisible, comme on l'a vu, et n'est pas corrigé.

Le code est testé par mutation (14 mutants du câblage et de la garde et 19 de l'estimateur).

## Comparaison avec NiftyMIC

NiftyMIC ([Ebner et al., 2020](algorithmes.md#références)) fait un recalage volume-vers-volume des stacks (recherche jusqu'à 45°), puis un recalage coupe-volume hiérarchique, une correction d'intensité par stack et un rejet des coupes aberrantes. Il est exécuté dans Docker (image `renbem/niftymic`, amd64 émulé : 9 à 13 minutes par sujet ; convention validée sur cinq des six sujets, imposée pour le sixième dont le volume n'a pas été écrit), réglages par défaut sauf la résolution (0,8 mm) et la correction du champ de biais (désactivée) ; ses transformations par coupe (`.tfm` ITK, LPS) sont lues par `scripts/read_niftymic_poses.py`. **Convention validée par mesure**, comme pour SVRTK : la corrélation entre les pixels de masque d'une coupe et le volume de NiftyMIC échantillonné à sa pose est de 0,935 (sujet calme 1) et 0,921 (agité 1) avec la transformation par coupe lue telle quelle, contre 0,652 et 0,376 sans correction et 0,44 avec l'inverse.

| Sujet | Ce que fait NiftyMIC | Accord avec les autres méthodes |
|---|---|---|
| Calme 1 (3 stacks) | 0 coupe rejetée sur 72 | 0,4-0,6 mm en médiane, comme entre les quatre autres |
| Calme 2 (3 stacks) | garde 71 coupes sur 82 | non mesuré (volume et concordance seulement) |
| Calme 3 (3 stacks) | **rejette tout le stack sagittal** (24/24) et la moitié des autres : garde 45 coupes sur 98 | non mesuré ; son volume garde des zones non recalées hors du cerveau |
| Agité 1 (4 stacks) | 12 % de coupes rejetées | **le plus proche de `s2v` (0,7 mm, 90ᵉ centile 1,5) et de SVRTK (0,9 ; 1,9)** ; `s2s` 1,1 (5,1) ; NeSVoR 1,1 (4,2) |
| Agité 2 (3 stacks) | rejette 80 % des coupes (au dernier seuil : 24/25, 14/26 et 21/21 coupes par stack) ; le stack sagittal n'a plus aucune transformation | impossible à comparer sur tout le sujet |
| Agité 3 (3 stacks) | rejette tout le stack coronal ; la reconstruction finale s'arrête avant d'écrire le volume | **proche de NeSVoR** (1,7 mm, 31 % des coupes à plus de 3 mm ; décalages de stack 4,1/5,1 contre 4,6/5,3) ; SVRTK s'en écarte de 18 mm |

**Concordance aux intersections, mêmes coupes pour tous** (figure de reconstructions avec tous les stacks, réglages par défaut actuels de nos méthodes ; NiftyMIC mesuré sur les coupes qu'il a gardées, qui sont les mieux recalées, donc à son avantage) : sur les coupes que NiftyMIC garde, il ne fait pas mieux que nous (calme 3 : 0,847 contre 0,881 pour notre coupe-coupe ; agité 1 : 0,797 contre 0,806 ; agité 3 : 0,749 contre 0,743 ; calmes 1 et 2 : 0,864 et 0,899 contre 0,874 et 0,902). Pour l'agité 2 il ne reste que 12 coupes, sans paire de stacks différents à mesurer. Visuellement, ses volumes sont plus texturés sur les sujets calmes, mais ils gardent des zones hors du cerveau non recalées (calme 3, agité 1) et le volume de l'agité 2 est inexploitable.

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
