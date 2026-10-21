# Portage de Multi-BOUNTI en Rust / Burn

Suite de `docs/BOUNTI.md` (même méthode : une référence officielle, un critère chiffré **écrit avant le code**, une sous-étape à la fois ; tout écart est consigné tel quel). Journal tenu au fil de l'eau ; les fichiers de travail (poids, scripts officiels, références) sont dans `data/multibounti/` (ignoré par Git).

## 1. Ce qu'est Multi-BOUNTI, faits relevés

Multi-BOUNTI (équipe SVRTK, King's College London) segmente le cerveau périnatal (fœtal et néonatal) en **43 étiquettes**, sur une reconstruction **à 0,5 mm**. Le code et les poids sont dans l'image Docker `fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd` (AMD seulement, 13,2 Go compressés, environ 38 Go décompressés : **non tirée**, le disque du Mac est plein à 93 %). Ils sont lus directement dans les couches du registre, en flux, sans rien stocker (`scratchpad` : outil `couches.py`, parcours par expression régulière). L'image `general_auto_arm` utilisée pour BOUNTI n'a **pas** ces scripts.
Licence : celle de la distribution, GNU GPL v3 (en-tête de chaque script : « Copyright 2026 - King's College London … GNU General Public License v3.0 »). Les poids suivent la même règle que ceux de BOUNTI (voir `docs/BOUNTI.md`) : jamais dans ce dépôt MIT.

**Contenu de `/home/perinatal-brain-mri-analysis/`** (état final de l'image : les couches récentes remplacent des scripts de la couche 46 par des variantes) :
- `scripts/` : `run-multi-bounti-fetal-brain-segmentation-2026-{general,dhcp,with-bet-init}.sh`, `run-multi-bounti-neo-brain-segmentation-2026-{dhcp,with-bet-init}.sh`, `run-fetal-brain-bet-extraction-2026-general.sh`, `run-neo-brain-bet-extraction-2026-general.sh`, `run-internal-capsule-neo-brain-segmentation-3t.sh`, deux scripts de volumétrie ;
- `src/` : `run_monai_patch_atunet_segmentation_1case-2026-{gpu,cpu}.py`, `…-flip-14-gpu.py` (capsule interne, 14 étiquettes) ;
- `models/` : réseaux d'extraction du cerveau (`atunet_bet_brain_{fetal_1lab,153t_svr_2026_128,neo_*}`), réseaux gauche/droite (`atunet_lr_brain_{fetal,neo}_2lab`), **réseaux 43 étiquettes** (`patch_atunet_new_multi_43_brain_neo_fetal_aff_fix_022026` pour le fœtus, `…_neo_aff_fix_…` pour le nouveau-né), capsule interne (`patch_atunet_ic_dgm_14_…`) ; chaque fichier fait environ 94,6 Mo ;
- `templates/ref-bounti/` : `ref-082025.nii.gz` (grille de référence), `dl-ref-t2.nii.gz`, `lr-lab.nii.gz` (gabarit gauche/droite), `lab.nii.gz`, et quatre transformations de départ (`i.dof`, `i-fetal.dof`, `i-early-ga.dof`, `i-very-early-ga.dof`) ;
- `bin/N4BiasFieldCorrection` (correction de biais, optionnelle).

**Les trois réseaux de la chaîne fœtale ont la même architecture** (script `run_monai_patch_atunet_segmentation_1case-2026-gpu.py`) : `AttentionUnet` de MONAI, 1 canal d'entrée, `cl_num + 1` sorties, canaux (32, 64, 128, 256, 512), pas (2, 2, 2, 2), noyaux 3, dropout 0,5, poids chargés avec `strict=False` ; intensités ramenées à [0, 1] (`ScaleIntensity`, sur tout le volume) ; inférence `sliding_window_inference(…, (128, 128, 128), 4, modèle, overlap=0.8)` puis `argmax`. Extraction du cerveau : 1 étiquette (2 sorties) ; gauche/droite : 2 étiquettes (3 sorties) ; tissus : **43 étiquettes (44 sorties)**.

**Chaîne officielle fœtale « general »** (`run-multi-bounti-fetal-brain-segmentation-2026-general.sh`, commandes MIRTK et `python3` ; le code Rust sera écrit à partir de cette description) :
1. Contrôle : la résolution z doit valoir 0,5 mm (±0,01), sinon la chaîne s'arrête.
2. Prétraitement : retrait des dynamiques, `nan` (valeurs non finies), masque `> 0,5`, plus grande composante connexe, recadrage ; rééchantillonnage linéaire dans la grille de `ref-082025` (origine recopiée) ; recadrage ; `pad-3d` à 128 (valeur 1).
3. Extraction du cerveau par le réseau (`atunet_bet_brain_153t_svr_2026_128`, une fenêtre de 128³) ; `erode-image` (1), plus grande composante, `dilate-image` (1) ; version dilatée 2 fois ; retour dans la grille recadrée (`-labels`).
4. Image masquée, recadrée, `pad-3d` à 128.
5. Réseau gauche/droite (`atunet_lr_brain_fetal_2lab`) → carte gauche/droite masquée par le cerveau.
6. **Recalage affine** (`mirtk register gabarit lr-lab -model Affine -dofin …`) du gabarit gauche/droite sur la carte prédite ; transformation de départ choisie par **`guess-ga`** (âge gestationnel estimé à partir du masque : `i.dof` si > 170, sinon `i-very-early-ga.dof`).
7. Retour du masque dans la grille d'origine (`-labels`), dilatation de 2, masquage et recadrage de l'image d'origine ; **réorientation** de l'image dans la grille du gabarit `dl-ref-t2` par la transformation affine ; seuil, recadrage, `nan`, `pad-3d` à 256.
8. Réseau 43 étiquettes (fenêtres de 128³ sur le volume de 256³, recouvrement 0,8) ; retour dans la grille de l'image d'origine par l'**inverse** de la transformation affine (`-dofin_i … -labels`) ; **`thin-cortex-thick`** (outil MIRTK propre à cette image) donne l'étiquetage final.
La variante `-with-bet-init` prend en plus un **masque du cerveau externe** (par exemple celui de la reconstruction) à la place de l'étape 3 ; `-dhcp` utilise le réseau d'extraction `atunet_bet_brain_fetal_1lab`.

## 2. Difficultés identifiées avant de coder

| Élément | Nature | Disponibilité de l'outil officiel |
|---|---|---|
| Réseaux (3) et inférence par fenêtres glissantes | même recette que BOUNTI ; fenêtres MONAI à reproduire | PyTorch / MONAI sur le Mac |
| Géométrie MIRTK (`pad-3d`, recadrages, rééchantillonnage, `-labels`) | extension des règles déjà établies pour BOUNTI | outils dans l'image ARM (`pad-3d`, `get-dz`, `register`, `transform-image`, …) |
| Recalage affine MIRTK (`register -model Affine`) | optimisation numérique : on ne reproduira pas le même minimum au bit près | `register` dans l'image ARM |
| `guess-ga`, `thin-cortex-thick` | outils propres à l'image AMD, sans source | absents de l'image ARM ; binaires AMD64 à extraire et exécuter dans un conteneur AMD64 (Rosetta) |

Entrée des essais : nos reconstructions sont à 0,8 mm et la chaîne exige 0,5 mm ; les volumes d'essai sont donc rééchantillonnés à 0,5 mm (interpolation linéaire). Les valeurs numériques sont réelles, la résolution est seulement interpolée.

## 3. Les deux outils sans source, reconstitués

Les deux binaires AMD64 (`guess-ga`, 46 Ko, couche 46 ; `thin-cortex-thick`, 55 Ko, couche 45) sont extraits dans `data/multibounti/amd/`. Pour les **exécuter** sur le Mac : conteneur AMD64 (Rosetta) `ubuntu:22.04` avec `libtbb2` et `libgomp1` (la version 24.04 n'a pas `libtbb.so.2`), les bibliothèques `libMIRTK*.so` des couches 9 et 25 montées depuis l'extérieur (les liens symboliques `.so.0` sont recréés à la main : l'extraction par `tarfile` ne les garde pas). Aucun de ces fichiers n'est dans le dépôt (GPL v3, non redistribués).

**`guess-ga`** : lit un masque binaire et affiche le **volume du masque en mL, arrondi à l'entier** (et non un âge en jours, malgré son nom). Relevé sur onze ellipsoïdes de 0,8 à 266,5 mL et sur le masque d'un cerveau (353,6 mL) : 1, 3, 6, 12, 21, 33, 50, 71, 97, 168, 266 et 354. Le script compare ce nombre à 170 (`-gt 170`) pour choisir la transformation de départ du recalage : `i.dof` (rigide, identité) si le volume arrondi dépasse 170 mL, sinon `i-very-early-ga.dof` (affine, échelle de 40 %).
**`thin-cortex-thick`** : lu dans le désassemblage (`llvm-objdump` d'Xcode ; les symboles `thick_gm` et `fix_gm` sont conservés). `main` lit l'image en `double`, applique **douze fois** `thick_gm(image, g, a, w)` à la suite, puis écrit le résultat en `double` : (g, a, w) = (37, 13, 1), (37, 15, 3), (37, 17, 5), (37, 19, 7), (37, 21, 9), (37, 23, 11), (38, 14, 2), (38, 16, 4), (38, 18, 6), (38, 20, 8), (38, 22, 10), (38, 24, 12). `fix_gm` n'est jamais appelée.
`thick_gm(image, g, a, w)` : pour chaque voxel V d'étiquette `w` situé à au moins 2 voxels du bord, s'il existe un voxel d'étiquette `a` à **distance exactement 2 sur un axe** (±x, ±y, ±z), alors chacun des six voisins de V d'étiquette `g` prend l'étiquette `w`. Les lectures se font dans l'image d'entrée de l'appel, les écritures dans une copie (l'ordre de parcours est donc sans effet).
**Vérification** (hypothèse tirée du code machine, puis essayée contre l'outil réel) : six volumes d'étiquettes aléatoires (de 14×15×13 à 24×15×18 voxels, 29 valeurs, beaucoup de 37 et 38) ; mon modèle des douze passes donne **0 voxel différent** de l'outil, alors que celui-ci modifie de 44 à 143 voxels par volume.

## 4. Références produites, plan et critères (écrits avant le code)

**Références officielles** (`scripts/multibounti_reference.py`, dossier `data/multibounti/ref/<id>/`, chaque fichier intermédiaire de la chaîne gardé) : `calme1` terminée (réseau 43 étiquettes : **20 023 s sur le CPU partagé**, bien plus que les 70 minutes estimées ; le CPU pur de PyTorch est le seul moyen, MPS n'a pas `ConvTranspose3d`) ; `calme2` et `calme3` lancées à basse priorité. Sortie finale de `calme1` : 43 étiquettes présentes (dont de minuscules : 3, 26 et 1 voxels pour les étiquettes 8, 7 et 43), grille et affine de l'entrée (236 × 212 × 211, 0,5 mm), type `float64`. L'extraction du cerveau a donné un masque de 367 mL (`guess-ga`) donc la transformation de départ `i.dof` ; 319 mL pour `calme2`.

| Sous-étape | Quoi | Critère (fixé avant) |
|---|---|---|
| **M1** réseaux | export ONNX et conversion `onnx2burn` des trois réseaux (extraction du cerveau `153t_svr`, gauche/droite, 43 étiquettes ; même `AttentionUnet`, seul le nombre de sorties change : 2, 3, 44) | critère C1 de BOUNTI : écart relatif des logits ≤ 1e-5 contre PyTorch f32 et ≥ 99,9 % d'étiquettes identiques, à 128³, sur les entrées réelles de la chaîne de `calme1` |
| **M2** fenêtres | `sliding_window_inference` de MONAI en Rust : fenêtres de 128³, recouvrement 0,8, poids uniformes | plan des fenêtres égal à celui de MONAI (départs calculés par MONAI pour plusieurs tailles) ; étiquettes **≥ 99,9 % identiques** à la sortie officielle `multi-lab-reo-t2.nii.gz` de `calme1` (256³) |
| **M3** géométrie | `pad-3d`, recadrages, rééchantillonnage dans `ref-082025`, nettoyage du masque, transformations avec `.dof` | chaque étape contre MIRTK (image ARM) sur `calme1` à `calme3` : dimensions identiques, affine ≤ 1e-3 mm, intensités ≤ 1e-3 de la plage, masques ≥ 99,9 % identiques |
| **M4** recalage | recalage affine du gabarit gauche/droite (`register -model Affine`) et lecture/écriture des `.dof` | à fixer après avoir mesuré la répétabilité de MIRTK lui-même (départs perturbés) : l'écart de notre affine à la sienne ne doit pas dépasser la dispersion de MIRTK ; contrôle indirect par le Dice final |
| **M5** assemblage | chaîne complète, commande `medx`, option « masque du cerveau externe » (variante `-with-bet-init`) | **Dice moyen ≥ 0,98 sur les 43 étiquettes et Dice ≥ 0,95 pour toute étiquette d'au moins 1 000 voxels** contre `sortie_officielle` sur `calme1` à `calme3` ; les étiquettes plus petites sont rapportées sans seuil |

**Ordre des essais pour isoler les causes** : M2 à M3 sont d'abord validés en partant des fichiers intermédiaires **officiels** (y compris `aff-d.dof` de MIRTK), puis M4 remplace le recalage de MIRTK par le nôtre ; sans cela une dérive du recalage ne se distinguerait pas d'un défaut du réseau ou de la géométrie.

## 5. Sous-étape M1 : les trois réseaux en Burn (résultat)

Export ONNX par `scripts/export_multibounti_onnx.py <bet|lr|multi>` (chargement **strict** : aucune clé manquante ni en trop, alors que le script officiel charge avec `strict=False`), vérifié contre onnxruntime : écart relatif 1,9e-6 (`bet`), 3,2e-6 (`lr`), 3,8e-6 (`multi`), accord des étiquettes ≥ 0,99999. Conversion par `onnx2burn` (burn-onnx 0.22.0-pre.4).
**Résultat structurel** : l'extraction du cerveau avec les poids `153t_svr` donne un code **identique** à `bet.rs` de BOUNTI (aux imports, à la mise en forme et au chemin des poids près : même contrôle que `scripts/bounti_poids.sh`) : `bet.rs` sert donc aussi à Multi-BOUNTI, avec d'autres poids. Les réseaux gauche/droite et 43 étiquettes diffèrent de `bet.rs` d'**une seule ligne** (le nombre de sorties de la dernière convolution : 3 et 44, au lieu de 2) : `lr.rs` et `multi.rs` sont `bet.rs` avec cette ligne changée et un en-tête d'attribution propre (code identique au code généré, mêmes contrôles). Modules : `crates/medoxide-bounti/src/{lr,multi}.rs`.
**Critère C1 atteint aux trois réseaux** (test `the_multibounti_networks_match_pytorch_on_real_inputs`, GPU, 128³, entrées réelles de la chaîne de `calme1` mises à l'échelle comme dans le script officiel ; références `scripts/make_multibounti_reference.py`) :

| Réseau | écart relatif des logits (seuil 1e-5) | accord des étiquettes (seuil 0,999) |
|---|---|---|
| extraction du cerveau (`bet`, 2 sorties) | 2,26e-6 | 1,000000 |
| gauche / droite (`lr`, 3 sorties) | 2,74e-6 | 1,000000 |
| 43 étiquettes (`multi`, 44 sorties, |logit| jusqu'à 486) | 2,64e-6 | 0,999999 |

## 6. Sous-étape M2 : fenêtres glissantes (résultat)

`crates/medoxide-bounti/src/fenetres.rs` refait `sliding_window_inference(…, (128, 128, 128), 4, modèle, overlap = 0,8)` de MONAI, mode `constant` : départs de fenêtre de `dense_patch_slices` (vérifiés contre MONAI sur plusieurs tailles), logits additionnés puis divisés par le nombre de fenêtres qui couvrent le voxel, étiquette = premier canal de plus grand logit moyen (comparaison stricte `>`). Le plan des fenêtres et l'accumulation sont testés sans GPU avec un faux réseau (cas d'égalité compris, ce qui a corrigé deux conceptions de test erronées).
**Critère atteint** (test `the_multibounti_windows_match_the_official_labels`, GPU, 343 fenêtres sur `pad-reo-n4-masked-t2-256` de `calme1`, mise à l'échelle sur tout le volume) : **0 voxel différent sur 16 777 216** de `multi-lab-reo-t2` officiel (seuil : 0,1 %), en 1 992 s.

## 7. Sous-étape M3 : géométrie, de MIRTK à Rust (résultat)

Chaque phase de `scripts/multibounti_mirtk_phases.sh` est refaite dans `crates/medoxide-bounti/src/multibounti.rs` (`phase_a`, `phase_b`, `phase_c`, `phase_d`) et comparée aux fichiers de MIRTK, en partant à chaque phase des fichiers officiels de la phase précédente.

| Phase | Fichiers comparés | Résultat (critère : dimensions égales, affine ≤ 1e-3 mm, intensités ≤ 1e-3 de la plage, masques ≥ 99,9 % identiques) |
|---|---|---|
| A (prétraitement, cube de 128³) | `m-t2`, `crop-org-t2`, `crop-tr-t2`, `pad-crop-tr-t2-128` | `calme1` à `calme3` : masque identique (0 voxel différent), affine ≤ 3,8e-6 mm, intensités ≤ 3,9e-6 |
| B (nettoyage du masque, cube du gauche/droite) | masques `bet-lab…`, `dl-bet-lab…`, `masked-n4-crop-tr-t2`, `pad-masked-n4-crop-tr-t2-128` | masques identiques (0 voxel), affine ≤ 5,6e-6 mm, intensités ≤ 3,3e-6 |
| C (réorientation par `aff-d.dof` officiel, cube de 256³) | `dl-bet-org-t2`, `masked-crop-org-t2`, `reo-n4-masked-t2`, `pad-reo-n4-masked-t2-256` | `calme1`, `calme2` : 1 et 2 voxels différents sur 6,4 et 5,6 millions (masque), affine ≤ 6,9e-6 mm, intensités ≤ 3,4e-6 |
| D (retour dans la grille d'origine, cortex aminci) | `tr-multi-lab-reo-t2`, `sortie_officielle` | `calme1` : 6 et 5 voxels différents sur 2 505 150 voxels non nuls (seuil : 0,1 %) |

**Règles de MIRTK établies en chemin** (chacune testée contre l'outil, voir `LEARNING.md`) :
- `mask-image` entre deux grilles différentes : plus proche voisin (`floor(x + 0,5)`) ;
- `crop-image` entre deux grilles : intervalle **semi-ouvert** dans l'espace physique (borne basse incluse, haute exclue, inversé sur un axe de sens négatif) ;
- `pad-3d` : cube isotrope de voxel `1,01 × (plus petit voxel × plus grande dimension) / N`, sans changement d'intensités, interpolation linéaire ;
- `edit-image -copy-origin` : centre sur centre ; `-labels` : indicatrice interpolée > 0,5, sinon plus proche voisin (extension du bord d'un voxel pour les étiquettes, d'un demi-voxel pour les intensités) ;
- dilatation et érosion sur le voisinage de 26 avec couche extérieure figée ; bruit de coordonnées en `float32` aux égalités exactes (voir `LEARNING.md`).

**Fichiers `.dof`** (`affine.rs`) : lecture des `.dof` rigides et affines de MIRTK (douze paramètres, nombres de 64 bits gros-boutistes) et matrice `A = T·R·K·S` identique à celle qu'affiche MIRTK (écart ≤ 5e-7). **`thin-cortex-thick`** (`cortex.rs`) : port des douze passes décrites en section 3, identique à l'outil sur six volumes aléatoires et sur la sortie réelle de `calme1` ; test par mutations à la main : 11 mutants sur 12 tués, le dernier (`< 5` au lieu de `< 4` dans la garde des petites images) est équivalent, la boucle étant déjà vide jusqu'à 4 voxels.
**`guess-ga`** est reproduit (367 mL pour `calme1`, test `the_full_multibounti_chain…`).

## 8. Sous-étape M4 : le recalage affine, mesures préalables

Le recalage compare le gabarit gauche/droite (`lr-lab`, cortex seulement : 185 mL) à la carte prédite par le petit réseau (98 mL) ; il est donc peu contraint. **Répétabilité de MIRTK** (`scripts/multibounti_repetabilite.py` et `…_mesure.py`) : le recalage officiel relancé huit fois à partir de départs tirés ±5 mm, ±5° et ±5 % autour du départ officiel ; écart moyen, sur les voxels du gabarit, entre deux transformations :

| Volume | écart à l'officiel, médiane / max | entre essais, médiane / max |
|---|---|---|
| `calme1` | 4,68 / 5,43 mm | 4,68 / 8,92 mm |
| `calme2` | 2,66 / 4,41 mm | 3,25 / 6,21 mm |

Le recalage de MIRTK n'est donc répétable qu'à quelques millimètres près : c'est la dispersion que le nôtre peut se permettre.

### Critère de M4 (écrit avant toute mesure sur les volumes de validation)

Constat, tiré de la sortie détaillée de `register` (`-v 3`) : MIRTK fait presque tout son travail au **premier niveau** (voxels de 3,9 mm, 16 classes d'histogramme, une quinzaine d'itérations de descente de gradient avec recherche de pas, énergie 1 → 0,94) ; les trois niveaux plus fins ne gagnent presque rien (1 à 2 itérations). Sa solution est donc un point de passage d'une descente non convergée, d'où sa dispersion de quelques millimètres.
Le recalage de medoxide en reprend le principe (NMI de 16 classes, voxels de 3,9 mm, descente de gradient depuis la transformation de départ de `guess-ga`) ; ses réglages (pas, flou, nombre de classes, tolérance) sont choisis sur des volumes de **mise au point** : `calme1`, `calme2`, `feta001` à `feta004`.
**Volumes de validation** (non utilisés pour les réglages) : `calme3`, `feta005` à `feta010` (reconstructions du jeu public FeTA 1.2, rééchantillonnées à 0,5 mm ; références produites par `scripts/multibounti_reference.py <id> --sans-reseau-43`, répétabilité de MIRTK par `scripts/multibounti_repetabilite.py <id> 8`).
Pour chaque volume, `e` est l'écart moyen (voxels du gabarit, en mm) entre deux transformations. **Critère** : (i) pour au moins 6 des 7 volumes de validation, l'écart de notre transformation à l'officielle ne dépasse pas le plus grand écart des huit essais de MIRTK à l'officielle sur le même volume ; (ii) aucun écart à l'officielle ne dépasse 10 mm ; (iii) contrôle indirect : le Dice final de M5.

**Réglages retenus sur les volumes de mise au point** (prototype Python hors dépôt, grille de 18 combinaisons : pas 3,0 / 3,9 / 5,0 mm, 16 ou 32 classes, tolérance 1e-3 / 1e-4 / 1e-5 ; puis tolérances plus serrées jusqu'à la convergence complète) : pas de 3,9 mm (flou gaussien d'écart-type la moitié du pas), **16 classes**, histogramme conjoint à pondération linéaire, gradient par différences centrées de 0,5 (mm, degré ou pour cent), pas de recherche de 1 doublé jusqu'à 4 fois (déplacement maximal de 1 unité par paramètre au départ), arrêt quand le gain relatif d'énergie passe sous **1e-5** ou après 100 itérations. Écart moyen à l'officielle sur les six volumes de mise au point : **1,97 mm** (calme1 2,6 ; calme2 3,8 ; feta001 0,7 ; feta002 1,7 ; feta003 1,2 ; feta004 1,9) contre 3,4 à 5,3 mm pour les réglages grossiers ; une convergence complète ne fait pas mieux (2,2 à 2,4 mm) : le plancher est la différence entre notre NMI et celle de MIRTK. Ces réglages sont figés avant toute mesure sur les volumes de validation.

### Résultat de M4 : le critère (i) n'est PAS atteint

Implémentation : `crates/medoxide-bounti/src/recalage.rs` (flou gaussien, NMI à 16 classes, descente de gradient par différences centrées ; 2 s par volume en `--release`), test `the_registration_stays_within_the_dispersion_of_mirtk`. Tests synthétiques : une translation connue est retrouvée à 0,25 mm ; une affine complète est rapprochée de 6,4 à 2,0 mm (les cartes d'ellipsoïdes presque symétriques ne contraignent pas tous les paramètres).
Écart moyen à la transformation officielle (mm), à côté du plus grand écart d'un des huit essais de MIRTK :

| Volume | notre écart | plus grand écart d'un essai MIRTK | dans la dispersion ? |
|---|---|---|---|
| calme1 (mise au point) | 1,63 | 5,43 | oui |
| calme2 (mise au point) | 3,66 | 4,41 | oui |
| feta001 à feta004 (mise au point) | 0,95 / 1,19 / 2,39 / 1,85 | 1,43 / 2,58 / 1,27 / 1,84 | oui / oui / non / non |
| **calme3 (validation)** | 2,78 | 4,11 | oui |
| **feta005 (validation)** | 5,00 | 3,72 | **non** |
| **feta006 (validation)** | 1,52 | 1,51 | non (de 0,01 mm) |
| **feta007 (validation)** | 2,10 | 5,66 | oui |
| **feta008 (validation)** | 1,39 | 15,23 | oui |
| **feta009 (validation)** | 1,35 | 1,31 | non (de 0,04 mm) |
| **feta010 (validation)** | 1,27 | 3,45 | oui |

**Validation : 4 volumes sur 7 dans la dispersion (critère : 6 sur 7) → critère (i) non atteint.** Le critère (ii) (aucun écart > 10 mm) est atteint : pire écart 5,00 mm (feta005). Deux des trois échecs sont de l'ordre du centième de millimètre ; le seul vrai écart est `feta005` (départ « petit cerveau », 83 mL, échelle initiale de 40 %).
Lecture : la dispersion de MIRTK sur les volumes FeTA est parfois de 1,3 mm seulement, de l'ordre de la différence entre deux implémentations de la même NMI ; sur `feta005`, notre solution a une NMI **plus haute** que celles de MIRTK (1,213 contre 1,17 à 1,20 : MIRTK n'a pas convergé), sur `feta003` et `calme2` c'est l'inverse (1,195 contre 1,200 ; 1,127 contre 1,129 : nous sommes un peu moins bien convergés). Les seuils ne sont pas modifiés ; le contrôle indirect (iii), le Dice final de M5, tranche sur l'utilité réelle.

## 9. Sous-étape M5a : la chaîne complète avec le recalage officiel (résultat)

`crates/medoxide-bounti/src/segmentation_multi.rs` assemble la chaîne : `preparer` (phase A, réseau d'extraction du cerveau, phase B, réseau gauche/droite), puis `segmenter_apres_recalage` (phase C, 343 fenêtres du réseau à 43 étiquettes, phase D). Le volume du masque en mL (`guess-ga`) est retrouvé exactement (367 mL pour `calme1`).
**Critère M5a atteint** (test `the_full_multibounti_chain_matches_the_official_one_with_the_official_registration`, GPU, `aff-d.dof` officiel, `calme1`) : accord de voxels avec `sortie_officielle` **1,00000**, Dice moyen **1,0000** sur les 43 étiquettes, pire Dice parmi les étiquettes d'au moins 1 000 voxels **0,9999** (étiquette 29) ; seuils 0,98 et 0,95. Durée : 2 279 s (GPU partagé avec d'autres calculs). La chaîne en Rust, de la reconstruction aux étiquettes, reproduit donc la chaîne officielle dès que le recalage est le même.

## 10. Sensibilité de la sortie au recalage : le plancher de bruit (mesure)

Test `diagnostic_sensitivity_of_the_segmentation_to_the_registration` : la chaîne en Burn (mêmes réseaux, mêmes phases) lancée sur `calme1` avec deux recalages **de MIRTK**, issus de départs perturbés (`scripts/multibounti_repetabilite.py`), et comparée à la sortie officielle :

| Recalage de MIRTK (départ perturbé) | écart moyen à l'officiel | accord de voxels | Dice moyen (43 étiquettes) | pire Dice (étiquettes ≥ 1 000 voxels) |
|---|---|---|---|---|
| essai 2 | 2,38 mm | 0,970 | **0,708** | 0,558 (étiquette 12) |
| essai 0 | 5,38 mm | 0,951 | **0,566** | 0,053 (étiquette 10) |

**Une différence de recalage de 2,4 mm, bien dans la dispersion de MIRTK lui-même, fait tomber le Dice moyen à 0,71.** Hypothèse (non vérifiée : les noms des étiquettes ne sont pas dans l'image) : la parcellation en régions dépend de la position de la tête dans l'espace du gabarit, le réseau n'est pas équivariant à quelques millimètres près. Les plus grosses étiquettes (37 et 38, plus de 500 000 voxels) gardent un Dice de 0,93 (essai 2) et 0,88 (essai 0) ; les petites perdent le plus (jusqu'à 0,05 pour l'étiquette 10 avec l'essai 0).
**Conséquence pour M5** : le critère « Dice moyen ≥ 0,98 contre `sortie_officielle` » ne peut être tenu que par une chaîne qui retrouve la transformation EXACTE de MIRTK (ce que fait M5a, 1,0000, en la lisant), pas par un recalage indépendant : il mesure la chance de tomber sur la même solution non convergée. Il n'est pas modifié ; le résultat avec notre recalage sera donné contre lui, et contre le critère suivant, écrit avant la mesure.

**Critère M5′ (écrit avant de lancer la chaîne avec notre recalage sur `calme1`)** : le Dice moyen de notre sortie contre la sortie officielle n'est pas inférieur à celui d'un recalage de MIRTK aussi éloigné de l'officiel que le nôtre : ≥ 0,708 si notre écart de recalage est ≤ 2,38 mm (valeur de l'essai 2), ≥ 0,566 s'il est ≤ 5,38 mm. Autrement dit : notre recalage ne perd pas plus de segmentation qu'un autre passage de MIRTK.

## 11. Résultat de M5 avec notre recalage (`calme1`)

Test `the_full_multibounti_chain_matches_the_official_one_with_our_registration` (GPU, 1 785 s, recalage de `recalage.rs` depuis le départ de `guess-ga`) :

| Mesure | valeur | seuil | verdict |
|---|---|---|---|
| accord de voxels avec `sortie_officielle` | 0,972 | — | — |
| **Dice moyen (43 étiquettes)** — critère M5 | **0,715** | ≥ 0,98 | **non atteint** |
| pire Dice, étiquettes ≥ 1 000 voxels (étiquette 12) — critère M5 | **0,627** | ≥ 0,95 | **non atteint** |
| **Dice moyen — critère M5′** (notre écart de recalage à l'officielle sur `calme1` : 1,63 mm ≤ 2,38 mm) | **0,715** | ≥ 0,708 | **atteint** (de 0,007) |

Lecture honnête : le critère M5 d'origine ne pouvait pas être atteint par un recalage indépendant (section 10 : deux autres passages de MIRTK lui-même donnent 0,71 et 0,57) ; il n'a pas été modifié. Le critère M5′ est atteint, mais de peu et avec une seule référence de comparaison (essais de MIRTK à 2,38 et 5,38 mm) : notre recalage, 1,63 mm plus près de l'officielle, ne fait pas mieux que l'essai à 2,38 mm (0,715 contre 0,708). On ne peut donc pas affirmer qu'il est meilleur ou pire qu'un autre passage de MIRTK ; on peut dire qu'il est du même ordre. Sans vérité terrain (les noms des 43 étiquettes ne sont pas dans l'image ; FeTA a des étiquettes manuelles en 7 tissus qu'on ne sait pas croiser sans cette table), la **qualité anatomique** de la segmentation obtenue avec chaque recalage n'est pas mesurée.

**Ce qui est établi** : (1) toute la chaîne en Rust, hors recalage, est identique à la chaîne officielle (M1 à M3, M5a : Dice 1,0000) ; (2) le recalage de MIRTK est peu répétable et la sortie en dépend fortement ; (3) notre recalage est dans la dispersion de MIRTK sur 4 volumes de validation sur 7, à 5 mm au plus.
**Ce qui reste ouvert** : variante néonatale ; table des noms d'étiquettes (`dseg.tsv`) ; publication des poids (GPL v3, décision de l'utilisateur) ; un recalage qui reproduirait la trajectoire exacte de MIRTK (gradient analytique de la NMI, pas de recherche, niveaux plus fins) pour se rapprocher de la sortie officielle plutôt que de la valeur de la NMI.
