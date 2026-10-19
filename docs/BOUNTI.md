# Portage de BOUNTI et Multi-BOUNTI en Rust (Burn)

Objectif : segmenter le cerveau fœtal (tissus puis parcellation) à partir d'une reconstruction 3D, **sans dépendre de Docker**, avec les réseaux des auteurs convertis dans un format lu par Burn. Même démarche que `medx fetalbet` : le modèle (architecture et poids) est celui des auteurs, rien n'est réentraîné, et chaque étape se valide contre la référence par un critère chiffré **écrit avant le code**.

Sources : dépôts `SVRTK/auto-proc-svrtk` (BOUNTI) et `SVRTK/perinatal-brain-mri-analysis` (Multi-BOUNTI), images Docker `fetalsvrtk/svrtk:general_auto_{amd,arm}` et `fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd`. Publications : Uus et al., *BOUNTI*, eLife 2023 (doi 10.7554/eLife.88818) ; Uus et al., *Multi-BOUNTI*, 2026 (doi 10.64898/2026.04.21.26351376).

## Licences et règle de conduite

- Le code et les poids de BOUNTI et de Multi-BOUNTI sont distribués sous **GNU GPL v3** (hypothèse de travail : la licence des poids est celle du logiciel ; à confirmer auprès des auteurs).
- medoxide est sous MIT. Conséquences : (1) les poids convertis **ne sont pas versionnés** dans ce dépôt et restent sous GPL v3 ; ils sont chargés à l'exécution depuis un chemin donné ; (2) le code Rust est écrit à partir de l'architecture (MONAI, Apache-2.0) et des articles, **sans copier** les scripts GPL des dépôts SVRTK ; (3) un script de conversion est fourni, pas les fichiers convertis ; (4) citation des articles dans l'aide de l'outil et du livre.
- Les poids extraits des images Docker sont dans `data/bounti/` (ignoré par Git).

## BOUNTI (19 étiquettes), ce que fait la chaîne officielle

Relevé dans les scripts de l'image `general_auto_arm` :

1. **Prétraitement (MIRTK, `prepare-for-monai`)** : rééchantillonnage de la reconstruction dans un cube de 128 (extraction du cerveau) puis de 256 (tissus).
2. **Étage 1, extraction du cerveau** : Attention U-Net MONAI 3D à 1 étiquette, cube de 128 ; masque nettoyé par composantes connexes et morphologie (dilatation puis érosion).
3. **Étage 2, 19 tissus** : **deux réseaux** appliqués au cube de 256 :
   - `AttentionUnet` MONAI 3D, entrée 1 canal, **20 sorties** (19 étiquettes + fond), canaux (16, 32, 64, 128, 256), pas (2, 2, 2, 2), noyaux de convolution et de convolution transposée 3 ; 5,91 M de paramètres ;
   - `UNet` MONAI 3D, 20 sorties, canaux (32, 64, 128, 256, 512), pas (2, 2, 2, 2), noyaux 3, **1 unité résiduelle**, activation PReLU, normalisation d'instance ; 9,83 M de paramètres.
4. **Intensité** : mise à l'échelle min-max dans [0, 1].
5. **Inférence** : fenêtre glissante de la taille du cube (donc une seule fenêtre à 256³), logits non normalisés.
6. **Augmentation à l'inférence** : retournement de l'axe x, retour, **échange des étiquettes gauche/droite** (1↔2, 3↔4, 5↔6, 7↔8, 11↔12, 14↔15, 16↔17 ; 0, 9, 10, 13, 18, 19 inchangées).
7. **Fusion** : moyenne des **logits** des quatre sorties (deux réseaux × normal et retourné), puis argmax.
8. **Retour à la grille d'origine** par transformation d'étiquettes.

Vérifié ici (MONAI 1.4.0, torch 2.7.1) : les deux architectures ci-dessus **chargent les checkpoints en mode strict, toutes clés comprises** (le script officiel charge avec `strict=False`, qui ignorerait un défaut de correspondance).

### Détail des opérations à réimplémenter

- **Attention U-Net** : blocs de convolution à deux convolutions 3×3×3 (ordre norme, dropout, activation ; **BatchNorm3d** puis **ReLU**) ; remontée par convolution transposée (noyau 3, pas 2) + BatchNorm + ReLU ; **porte d'attention** (convolutions 1×1×1 sur le signal de passage et le signal du niveau inférieur, BatchNorm, somme, ReLU, convolution 1×1×1, BatchNorm, sigmoïde, produit) ; **fusion** par convolution 3×3×3 avec **PReLU et normalisation d'instance** (réglages par défaut de MONAI) ; convolution finale 1×1×1.
- **U-Net** : unités résiduelles (convolution 3×3×3, **PReLU à un paramètre partagé**, **InstanceNorm3d sans paramètres**, plus une convolution de raccourci), convolutions à pas 2 en descente, convolutions transposées en remontée, connexions de saut par concaténation.
- En évaluation, BatchNorm est une transformation affine par canal (statistiques d'exécution) et le dropout est l'identité ; InstanceNorm calcule ses statistiques par volume et par canal, même en évaluation.

## Multi-BOUNTI (43 étiquettes)

D'après le dépôt et l'article : parcellation en **43 étiquettes** (lobes), fœtal et néonatal (21 à 44 semaines d'âge gestationnel ou postmenstruel), **Attention U-Net 3D**, entrée à **0,5 mm isotrope**, protocole dHCP (TE de 250 ms), correction N4 optionnelle. L'image Docker est limitée à l'architecture AMD et annonce plus de 16 Go de mémoire GPU. Les poids ne sont **pas** dans l'image BOUNTI (le dossier `monai-2026` de celle-ci contient deux modèles pour l'IRM à 0,55 T) : ils sont à extraire de l'image `perinatal_brain_mri_analysis_amd`. Architecture exacte, hiérarchie des étiquettes et prétraitement : à relever dans cette image avant de définir les critères.

## Étape 1 : sonde de performance (résultats)

`crates/medoxide-svr/examples/sonde_conv3d.rs` : une convolution 3D 3×3×3 de 32 vers 32 canaux sur un cube de côté `n` (Mac à 24 Go de RAM, Burn 0.22.0-pre.4).

| Backend | n | Temps | Débit | Mémoire maximale |
|---|---|---|---|---|
| GPU (wgpu / Metal) | 64 | 0,09 s | 78 G mult.-add./s | 0,38 Go |
| GPU | 128 | 0,27 s | 217 G/s | 0,65 Go |
| GPU | 192 | 0,99 s | 198 G/s | 2,3 Go |
| GPU | **256** | **1,96 s** | **237 G/s** | **4,4 Go** |
| CPU (`flex`) | 64 | 0,33 s | 22 G/s | 0,11 Go |
| CPU (`flex`) | 128 | 2,58 s | 22 G/s | 0,82 Go |

Le GPU est environ dix fois plus rapide que le CPU. Comptage des multiplications-additions des deux réseaux BOUNTI à 256³ (crochets MONAI sur une petite entrée, mise à l'échelle) : **789 G pour l'Attention U-Net, 479 G pour l'U-Net**, soit environ 3 s et 2 s de convolutions par passe avant au débit mesuré, donc **une dizaine de secondes pour les quatre passes** de BOUNTI (hors normalisations et activations, dont le coût est à mesurer). La plus grande carte de caractéristiques fait 1,3 Go (les logits à 20 canaux). **Décision : GO sur le backend GPU (wgpu) ; le CPU serait de l'ordre de dix fois plus lent.** Les critères C4 (ressources) sont satisfaits à l'échelle de la sonde ; la mesure sur le réseau complet reste à faire.

## Voie de conversion : ONNX plutôt que couche par couche

Fetal-BET est exactement l'`AttentionUnet` de MONAI (en 2D), converti par `onnx2burn` (burn-onnx 0.22.0-pre.4) à partir d'un export ONNX : le code Rust du modèle est généré et les poids sont écrits au format `burnpack`. Les opérations de BOUNTI sont les mêmes en 3D (`Conv3d`, `ConvTranspose3d`, BatchNorm, InstanceNorm, PReLU, sigmoïde, produit, concaténation), tous présents dans Burn. **Voie proposée** : exporter chaque réseau MONAI en ONNX (opset 17, taille d'entrée fixe), générer le code et les poids avec `onnx2burn`, valider C1. Repli si l'import échoue sur une opération 3D : écrire le réseau à la main avec `burn::nn` (plus long, plus formateur).

## Étape 2 : U-Net 3D en Rust (résultat)

Crate `medoxide-bounti` (modèle généré par `onnx2burn`, en-tête d'attribution, poids hors dépôt). Test `the_unet_logits_match_pytorch` (`cargo test -p medoxide-bounti -- --ignored`, demande les fichiers locaux de `data/bounti/`) : cube de 64³ de bruit uniforme, comparé aux logits PyTorch/MONAI (`scripts/make_bounti_reference.py`).

| Critère C1 (amendé, relatif) | Mesure | Seuil |
|---|---|---|
| écart relatif maximal des logits (backend GPU wgpu) | **9,5e-6** (absolu 1,2e-3 pour |logit| max 126) | ≤ 1e-5 |
| voxels d'étiquette identique | **100 %** | ≥ 99,9 % |

**C1 atteint pour l'U-Net**, avec une **marge mince** sur l'écart relatif (9,5e-6 contre 1e-5) : les calculs sont en f32 sur GPU contre CPU pour la référence, et l'écart croît avec la profondeur du réseau et la taille du volume. À 64³ la marge suffit ; à 256³ le comportement est à vérifier (étape suivante) avant de conclure que le critère tient sur un volume réel.

## Étape 3 : Attention U-Net, taille réelle (256³) et précision de la normalisation d'instance

**Attention U-Net** (même voie, `attunet.rs`) : à 64³ sur du bruit uniforme, C1 est d'abord **manqué** (écart relatif 1,17e-5 pour un seuil de 1e-5, avec l'implémentation de `burn::nn::InstanceNorm`). Je n'ai pas relâché le critère avant d'avoir compris l'écart.

**Contrôle à 256³ sur un volume réel** (une de nos reconstructions rééchantillonnée dans un cube de 64³, 128³ ou 256³, puis mise à l'échelle min-max : test numérique, pas le prétraitement officiel). Écart relatif des logits Burn (GPU) à PyTorch f32, avec la normalisation d'instance de Burn :

| Réseau | 64³ | 128³ | 256³ |
|---|---|---|---|
| U-Net | 5,8e-4 | 3,2e-3 | **3,5e-2** (accord des étiquettes 99,28 %) |
| Attention U-Net | 7,4e-6 | 1,9e-4 | **5,6e-3** (99,86 %) |

L'erreur **grandit avec la taille du volume** (de 5 à 30 fois par doublement du côté) et est encore pire sur le CPU de Burn (U-Net 2,1e-1 à 256³, 96,3 % d'accord) : ce n'est donc ni du bruit d'arrondi ni une propriété du GPU. Une convolution somme toujours le même nombre de termes ; seule une opération qui somme sur tout le volume peut croître ainsi. Le suspect (la normalisation d'instance, bien plus présente dans l'U-Net que dans l'Attention U-Net) est testé isolément contre un calcul en f64 sur `[1, 4, n, n, n]` de moyenne 3 et d'écart type 1 :

| Normalisation d'instance | 64³ | 128³ | 256³ |
|---|---|---|---|
| `burn::nn::InstanceNorm` (GPU) | 2,8e-5 | 4,7e-4 | 1,0e-2 |
| `burn::nn::InstanceNorm` (CPU) | 1,7e-4 | 3,3e-3 | 1,6e-1 |
| **`InstanceNorm` en cascade** (GPU) | 1,0e-6 | 1,4e-6 | 2,0e-6 |

Cause : la moyenne et la variance sont une réduction sur 16,7 millions de termes en f32. Remède (`norme.rs`) : calculer la moyenne et la variance **une dimension spatiale après l'autre** (trois réductions de 256 termes), avec la même structure et les mêmes champs que `burn::nn::InstanceNorm` (le code généré s'y branche en changeant deux imports). Tests : précision à 64³, 128³ et 256³ (≤ 1e-5), équivalence avec Burn sur une petite entrée, canal constant, six mutants tués.

**C1 après correction** (écart relatif à PyTorch f32, GPU) :

| Réseau | bruit 64³ | volume réel 64³ | 128³ | 256³ |
|---|---|---|---|---|
| U-Net | 2,5e-6 ✓ | 2,0e-6 ✓ | 5,1e-6 ✓ | **2,6e-5 ✗** (accord 99,9998 %) |
| Attention U-Net | 6,4e-6 ✓ | 2,0e-6 ✓ | 2,4e-6 ✓ | **1,9e-6 ✓** (accord 100 %) |

**Avec la seule cascade, C1 est atteint pour l'Attention U-Net à toutes les tailles et pour l'U-Net jusqu'à 128³, mais manqué pour l'U-Net à 256³ (2,6e-5)** : la suite (section suivante) en trouve et en corrige la cause.

## Référence float64 pour l'U-Net (192³), règle d'interprétation écrite avant la mesure

Question : l'écart de 2,6e-5 de l'U-Net à 256³ contre PyTorch f32 vient-il de Burn ou de PyTorch f32 ? Mesure : à 160³ d'abord (environ 3,5 Go en double précision), puis à 192³ (environ 6 Go) seulement si la mémoire disponible le permet (la double précision à 256³ fait planter le Mac de 24 Go ; environ 9 Go étaient disponibles au moment du choix, Firefox et VS Code ouverts), volume réel, écart relatif de PyTorch f32 et de Burn GPU à une référence PyTorch float64 (`scripts/analyse_precision_bounti.py unet 160 _reel`), sous garde-fou de mémoire (arrêt au-delà de 10 Go).
**Règle** : (a) si l'écart de Burn à f64 est ≤ 2 fois celui de PyTorch f32 à f64, Burn est aussi précis que PyTorch et l'écart à PyTorch f32 est du bruit de la référence : C1 devra alors se définir contre la référence f64 (amendement explicite, avec ce résultat comme motif) ; (b) s'il est supérieur à 2 fois, Burn garde une imprécision propre, à chercher couche par couche (PReLU, convolution transposée, somme résiduelle, convolutions de grande profondeur de canaux) avant d'amender quoi que ce soit.

**Résultat (U-Net, volume réel à 160³, référence float64 de PyTorch/MONAI ; pic de mémoire du calcul f64 : 6,8 Go, à peine sous la limite de 7 Go fixée, l'estimation naïve de 3,5 Go était fausse d'un facteur 2).** Écart relatif à la vérité en f64 :

| Implémentation | écart relatif | accord des étiquettes |
|---|---|---|
| PyTorch f32 | 1,73e-6 | 100 % |
| Burn GPU f32, cascade seule | 1,20e-5 | 99,9999 % |
| Burn CPU (flex) f32, cascade seule | 1,15e-5 | 99,9999 % |

L'écart de Burn est **7 fois celui de PyTorch** : cas (b) de la règle, Burn garde une imprécision propre, que ni le GPU ni le CPU ne distinguent (les deux backends donnent la même erreur : la cause est dans du code partagé, donc dans `norme.rs` elle-même). Hypothèse testée directement : la précision de la normalisation dépend du **rapport moyenne / écart type** des canaux, car sommer des valeurs proches de la moyenne perd de la précision relative à l'écart type. Écart de la sortie normalisée à f64 sur `[1, 2, 128, 128, 128]` (GPU) :

| rapport moyenne / écart type | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| cascade seule | 1,0e-6 | 3,0e-6 | **1,2e-5** | 2,4e-4 |
| **cascade + moyenne corrigée en deux passes** | 1,0e-6 | 9,3e-7 | 1,0e-6 | 7,5e-7 |

Hypothèse confirmée (le niveau de 1,2e-5 à un rapport de 100 est celui observé sur le réseau). Remède : recentrer d'abord sur une première moyenne (soustraction de valeurs voisines, exacte), puis calculer la moyenne résiduelle, minuscule. Tests ajoutés (précision à chaque rapport) et six mutants de `norme.rs` tués.

**C1 final, écart relatif à PyTorch f32 (GPU), volume réel :**

| Réseau | 64³ | 128³ | 160³ | 256³ | accord des étiquettes |
|---|---|---|---|---|---|
| U-Net | 2,2e-6 | 2,1e-6 | 2,2e-6 | **2,7e-6** | 100 % |
| Attention U-Net | 2,1e-6 | 2,4e-6 | — | **1,8e-6** | 100 % |

Contre la vérité en f64 à 160³ : PyTorch f32 1,73e-6, **Burn GPU 1,74e-6** : Burn est désormais aussi précis que PyTorch. **C1 est atteint pour les deux réseaux, à toutes les tailles testées, sans amender le critère de seuil (1e-5) ni la définition (écart relatif à PyTorch f32)** ; seule l'amendement initial (absolu vers relatif, voir C1) reste en vigueur.

**Incident.** La première tentative de référence f64, à 256³ et sans borne de mémoire, a fait planter le Mac. Le calcul se fait désormais à 160³ sous `scripts/garde_memoire.py` (arrêt au-delà d'une limite). La mémoire d'un passage avant f64 est d'environ 6,8 Go à 160³ pour l'U-Net et croît comme n³ : environ 12 Go à 192³ et plus de 28 Go à 256³.

## Étape 4 : fusion des quatre sorties (C2)

**Critère C2, précisé avant le code** : la fusion Rust (U-Net et Attention U-Net, image normale et image retournée sur l'axe x, retour, échange des étiquettes gauche/droite, moyenne des logits, argmax) donne, à entrée identique, **au moins 99,9 % de voxels d'étiquette identique** à la fusion de référence PyTorch/MONAI (`scripts/make_bounti_fusion_reference.py`), mesurée à 128³ et à 160³ sur un volume réel rééchantillonné (la logique de fusion ne dépend pas de la taille ; 256³ coûterait plus de 13 Go de mémoire pour la seule référence). Indicatif : écart relatif des logits fusionnés, attendu de l'ordre de 1e-6 à 1e-5.
Faits relevés dans le script officiel et vérifiés ici : le retournement est celui de `Flip(1)` de MONAI, qui retourne bien le **premier axe spatial** (X) d'un tenseur `[1, 1, X, Y, Z]` (vérifié sur un petit tenseur) ; l'échange des étiquettes est la permutation des canaux 1↔2, 3↔4, 5↔6, 7↔8, 11↔12, 14↔15, 16↔17 (les canaux 0, 9, 10, 13, 18, 19 sont inchangés) ; la fusion est la moyenne des **logits** (non normalisés) des quatre sorties.

**Résultat de l'étape 4 (fusion, `crates/medoxide-bounti/src/fusion.rs`).**

| Mesure | Résultat | Critère |
|---|---|---|
| Fusion Rust contre la fusion PyTorch/MONAI, 128³, volume réel | **0 voxel différent sur 2 097 152** (accord de 100 %) | ≥ 99,9 % |
| Écart relatif des logits fusionnés (indicatif) | 1,6e-6 | — |
| Fusion Rust à 160³ | **non mesurée** (voir ci-dessous) | ≥ 99,9 % |

**C2 est atteint à 128³ ; la mesure à 160³, demandée par le critère, reste à faire.** La référence PyTorch à 160³ demande environ 9 Go (4,5 Go à 128³, environ 1,95 fois plus à 160³) ; elle a été différée parce que la machine était sous forte pression mémoire au moment prévu (swap à 10,7 Go sur 11,3 Go, 8,6 Go compressés, compilations et un test d'une autre session en cours). À lancer sous `scripts/garde_memoire.py 11` quand la mémoire le permet : `python scripts/make_bounti_fusion_reference.py 160`, puis `BOUNTI_N=160 cargo test -p medoxide-bounti the_fusion_matches -- --ignored`.
Tests : quatre tests unitaires sans poids (permutation des étiquettes comparée à la liste des paires, échange et retournement exacts sur un tenseur à valeurs distinctes, formule de fusion à la main avec deux réseaux factices non symétriques, étiquettes) ; **sept mutants sur huit tués** (axe retourné, permutation, diviseur, retour du retournement, image réellement retournée, échange, argmax) ; le huitième est équivalent par construction (retourner puis échanger, ou échanger puis retourner, donne le même résultat).


## Étape 5 : chaîne complète, avec le choix de l'extraction du cerveau (décision de l'utilisateur)

**Deux modes** : `bounti` (extraction du cerveau par le réseau 3D de BOUNTI, comme la chaîne officielle) et `fetalbet` (extraction par Fetal-BET, déjà en Rust dans `medoxide-fetalbet`), suivis dans les deux cas de la même segmentation BOUNTI. En ligne de commande : `medx bounti --extraction bounti|fetalbet`.

**Chaîne officielle, relevée dans le script de l'image Docker** (les commandes MIRTK sont celles du script ; le code Rust sera écrit à partir de cette description) :
1. *Prétraitement* : retrait des dynamiques ; masque binaire des voxels > 0,005 ; recadrage sur la boîte englobante de ce masque ; **rééchantillonnage linéaire dans l'espace de référence** `brain-ref-space.nii.gz` : grille de **300 × 369 × 300 voxels à 0,6 mm, orientation LAS** (axe x de sens opposé, affine diag(−0,6, 0,6, 0,6)), dont l'origine est recopiée de l'image d'entrée (centrée sur son centre) ; nouveau recadrage ; remplacement des valeurs non finies.
2. *Extraction du cerveau (mode `bounti`)* : `prepare-for-monai` à 128 (voir ci-dessous), Attention U-Net 3D à **2 sorties, canaux 32 à 512** (≠ réseaux de tissus), intensité min-max dans [0, 1], une fenêtre de 128³, argmax ; puis étiquette 1, **plus grande composante connexe**, retour dans l'espace de l'image (interpolation d'étiquettes), **dilatation de 4 puis érosion de 2**, masquage de l'image, recadrage.
3. *Segmentation BOUNTI* : `prepare-for-monai` à 256 sur l'image masquée, fusion des quatre sorties (étape 4), retour dans l'espace de l'image (interpolation d'étiquettes).
4. **`prepare-for-monai` (cube de côté `n`)** : voxel isotrope de taille `1,01 × (plus petit voxel × plus grand nombre de voxels) / n`, rééchantillonnage linéaire, cube de n³ voxels centré sur le centre de l'image, mise à l'échelle des intensités. (Le code officiel a une particularité : en cherchant le plus petit voxel, il retient la taille en z quand la taille en y est la plus petite ; sans effet pour nos volumes isotropes, à reproduire ou à documenter selon le cas.)

**Sous-étapes, chacune validée contre la sortie de l'outil officiel exécuté dans l'image Docker sur trois de nos reconstructions (références produites avant le code de chaque sous-étape) :**
- **5a, géométrie** (grille de référence, recadrages, cube) : mêmes dimensions et même affine que MIRTK ; écart maximal d'intensité ≤ 1e-3 de la plage des valeurs. Échantillonnage trilinéaire : celui de `medoxide-svr` (validé contre scipy et SimpleITK) ou une copie ; la dépendance de `medoxide-bounti` envers `medoxide-svr` (autodiff, flex) est à éviter si un petit module commun suffit.
- **5b, réseau d'extraction 3D** : conversion ONNX → Burn comme pour les deux autres, critère C1 (écart relatif ≤ 1e-5 à PyTorch f32, 99,9 % d'étiquettes), à 128³.
- **5c, nettoyage du masque** : composante connexe, dilatation, érosion, masquage : au moins 99,9 % de voxels identiques à MIRTK.
- **5d, assemblage** et commande `medx bounti` avec les deux modes.
- **5e, critère C3** : Dice par étiquette ≥ 0,98 (aucune étiquette sous 0,95) contre la sortie du script Docker complet, sur trois reconstructions, pour le mode `bounti` ; pour le mode `fetalbet`, Dice moyen ≥ 0,95 contre la même référence, et écart entre les deux modes rapporté sans seuil.
Restent à trancher en cours de route : la sortie du mode `fetalbet` (Fetal-BET est une extraction 2D par coupe à 1 mm, avec sa propre normalisation, qui donne un masque à la grille de l'image), et le traitement des valeurs hors plage (négatives, non finies).

## Sous-étape 5a : géométrie (résultat)

Références : le prétraitement officiel rejoué dans l'image Docker (`scripts/bounti_references_mirtk.py`) sur trois de nos reconstructions (volumes de 0,8 mm, environ 148 × 133 × 132 voxels). Code : `crates/medoxide-bounti/src/geometrie.rs` ; test `the_geometry_matches_mirtk` (`cargo test -p medoxide-bounti -- --ignored`).

| Étape | Dimensions | Écart d'affine | Écart d'intensité (part de la plage) |
|---|---|---|---|
| 1 recadrage (calme 1 / 2 / 3) | 126×112×110 / 124×110×101 / 118×110×101, identiques à MIRTK | ≤ 3,2e-6 mm | **0** |
| 2-3 grille de référence puis recadrage | 168×149×146 / 166×147×134 / 158×147×134, identiques | ≤ 8,4e-6 mm | ≤ 9,1e-6 |
| 4 cube de 128³ | identiques | ≤ 8,5e-6 mm | ≤ 7,6e-6 |
| 4 cube de 256³ | identiques | ≤ 9,6e-6 mm | ≤ 7,5e-6 |

**Critère atteint** (mêmes dimensions et même affine à 1e-3 mm près ; écart d'intensité ≤ 1e-3 de la plage pour les recadrages, ≤ 1e-5 pour les cubes) sur les trois volumes ; les écarts restants sont du bruit de précision flottante.

**Règles établies par comparaison étape par étape avec MIRTK** (prototype Python puis Rust) :
- *recadrage* : boîte englobante des voxels **strictement** supérieurs à 0,005 ;
- *grille de référence* : 300 × 369 × 300 voxels de 0,6 mm, orientation LAS, **centre de grille égal au centre du volume recadré** ;
- *interpolation* : trilinéaire en coordonnées monde ; une position à **moins d'un demi-voxel du bord** prend la valeur du voxel de bord, au-delà 0. Sans cette règle, 203 voxels sur 33 millions différaient (tous dans la marge d'un demi-voxel). **Hypothèse réfutée** : exclure les voxels de valeur 0 de l'interpolation (la « valeur de remplissage de la source » de `prepare-for-monai`) donnait 154 329 voxels différents au lieu de 203 : MIRTK interpole le zéro comme une valeur ;
- *second recadrage* : on garde les voxels dont le **centre** est dans la boîte physique du masque (bords de voxels), soit début = ⌈borne basse⌉ et fin = ⌊borne haute⌋ sur chaque axe ; le résultat est une tranche exacte de l'image rééchantillonnée ;
- *cube* : voxel isotrope `1,01 × plus petit voxel × plus grande dimension / n` (0,7954 mm pour n = 128 et 0,3977 mm pour 256 sur le premier volume), même orientation, centré sur le centre du volume, intensités ramenées dans [0, 1].
Tests : cinq tests unitaires sur données synthétiques ; **seize mutants du module tous tués** (dont un cas anisotrope ajouté après qu'un mutant a survécu : tous les volumes de test étaient isotropes, le plus petit et le plus grand voxel y étaient égaux).

**Lecture de l'affine.** Les fichiers de MIRTK n'ont qu'un `qform` : `medoxide-core` gagne `affine_nifti` (règle de nibabel : `sform` si son code est positif, sinon `qform`, sinon diagonale des `pixdim` centrée), validée contre nibabel sur six `qform` (identité, rotation de 90° autour de z et autour de x, inversion d'axe avec `qfac = −1`, oblique, rotation générale) ; neuf mutants tués. C'est la règle demandée par la demande n° 1 de guioxide (`docs/issues/medoxide.md`) ; `volume_info`, qui ne lit que le `sform`, n'a pas été modifiée.

**Constat hors périmètre, corrigé.** Les volumes que `medoxide` écrivait (par exemple `ours_s2s_defaut.nii.gz`) avaient un `sform` correct mais un **`qform` sans origine**, donc incohérent avec le `sform` : les outils qui lisent le `qform` (MIRTK l'a fait) plaçaient le volume avec son coin à l'origine. `medoxide-core::definir_affine` écrit désormais `sform` et `qform` cohérents (quaternion selon la spécification NIfTI-1, `pixdim` et `qfac` mis à jour) et `ReconstructionOutput::write_volume` l'utilise. Tests : aller-retour sur quinze affines (dont inversion d'axe, rotations d'axes variés, rotations de 150° et de 180° autour d'axes obliques qui exercent chaque branche du quaternion), le `qform` relu seul égal à la matrice écrite ; le test d'écriture du SVR relit le fichier et compare le `qform` seul à l'affine de la grille ; douze mutants de `definir_affine` et un mutant de l'écriture tués (cinq avaient d'abord survécu : toutes mes rotations de 180° étaient autour d'axes de coordonnées, où les termes croisés du quaternion valent zéro). **Les fichiers déjà écrits gardent leur ancien `qform`** : seuls les volumes produits à partir de maintenant sont cohérents.

## Sous-étape 5b : réseau d'extraction du cerveau (résultat)

Attention U-Net 3D de l'étage 1 de la chaîne officielle : 1 canal d'entrée, **2 sorties** (fond, cerveau), **canaux 32 à 512** (≠ réseaux de tissus), mêmes noyaux et même dropout, 23,6 M de paramètres ; entrée min-max dans [0, 1], une fenêtre de 128³, argmax. Même voie que les autres : export ONNX (`scripts/export_bounti_onnx.py bet`), `onnx2burn`, `crates/medoxide-bounti/src/bet.rs` (en-tête d'attribution, normalisation d'instance en cascade).

| Mesure | Résultat | Critère C1 |
|---|---|---|
| Export ONNX à 64³ contre `onnxruntime` (écart relatif) | 4,7e-6, étiquettes à 100 % | ≤ 1e-5 |
| Logits Burn contre PyTorch, volume réel à 64³ | **6,2e-7**, 100 % d'étiquettes identiques | ≤ 1e-5, ≥ 99,9 % |
| Logits Burn contre PyTorch, volume réel à 128³ | **7,9e-7**, 100 % d'étiquettes identiques | ≤ 1e-5, ≥ 99,9 % |

**C1 atteint** pour le réseau d'extraction (test `the_brain_extraction_logits_match_pytorch_on_a_real_volume`). **Remarque** : l'export à 128³ manquait le seuil contre `onnxruntime` (1,2e-4) alors que celui à 64³ le tient : la normalisation d'instance d'`onnxruntime` perd elle aussi en précision quand le volume grandit (même défaut que celui trouvé dans Burn). Le contrôle de fidélité d'un export se fait donc à 64³, et la validation à la taille réelle contre PyTorch, côté Rust. Aucun code écrit à la main n'est ajouté par cette sous-étape (le modèle est généré), donc pas de test de mutation.

## Sous-étape 5c : nettoyage du masque (résultat)

Références : l'extraction du cerveau officielle rejouée dans l'image Docker (le réseau officiel en Python, puis les commandes MIRTK du script officiel : `scripts/bounti_mirtk_masque.sh`) sur les trois mêmes volumes. Code : `crates/medoxide-bounti/src/masque.rs` ; test `the_mask_cleaning_matches_mirtk`.

| Étape (chaque étape part de l'entrée de MIRTK) | calme 1 | calme 2 | calme 3 |
|---|---|---|---|
| plus grande composante connexe (étiquettes du réseau officiel, 2 097 152 voxels) | 0 voxel différent | 0 | 0 |
| passage dans la grille du volume (3,1 à 3,7 millions de voxels) | 0 | 1 | 2 |
| dilatation de 4 | 0 | 0 | 0 |
| érosion de 2 | 0 | 0 | 0 |
| **chaîne complète** (du masque du cerveau à la sortie de l'érosion) | **0** | **1** | **2** |
| image masquée et recadrée | mêmes dimensions et affine, écart 0 | idem | idem |
| cube de 256³ de l'image masquée (après mise à l'échelle dans [0, 1]) | 1,2e-7 | 1,2e-7 | 1,2e-7 |

**Critère de la sous-étape atteint** (au moins 99,9 % de voxels identiques à MIRTK) : la chaîne diffère de 0, 1 et 2 voxels sur plus de 3 millions, soit 99,9999 % d'identiques. Les 1 et 2 voxels des volumes 2 et 3 ont une cause identifiée : à l'étape de passage dans la grille, un voxel dont la valeur interpolée est à 4e-6 du seuil de 0,5 (0,4999956 dans le cas examiné) est classé dans le masque par MIRTK, qui calcule les coordonnées en simple précision (position d'environ 68 voxels, incertitude de l'ordre de 4e-6 voxel) : du bruit de précision de MIRTK, pas une règle différente.

**Règles établies par comparaison étape par étape avec MIRTK** (prototype Python puis Rust) :
- *plus grande composante* : connexité 26 (défaut de MIRTK) ;
- *passage dans la grille* : interpolation linéaire du masque binaire (convention d'interpolation de la géométrie) puis seuil à 0,5 ; la conversion d'étiquettes de MIRTK (`-labels`) rééchantillonne chaque étiquette par interpolation linéaire et garde celle de plus grande pseudo-probabilité, ce qui n'est pas un plus proche voisin ;
- *dilatation et érosion* : voisinage de 26 voxels (cube 3 × 3 × 3 par itération), 4 puis 2 itérations ; **à chaque itération, la couche extérieure de voxels (les faces de l'image) garde sa valeur** : sans cette règle ma dilatation avait 11 473 voxels de trop, tous sur une face de l'image (des voisins directs du masque, donc pas un problème de forme du voisinage) ; les voisinages de 6 et de 18 voxels, leurs alternances et une dilatation euclidienne ont été essayés et écartés.
- *masquage* : 0 hors du masque ; *recadrage* : boîte englobante du masque.
Tests : cinq tests unitaires (connexité 26 avec un voxel relié par une diagonale pure, cubes de dilatation et d'érosion, couche figée sur les six faces, interpolation et seuil strict à 0,45, 0,5 et 0,55, paramètres 4 et 2 du masque du cerveau, masquage et recadrage) ; **dix-sept mutants du module : seize tués, un équivalent** (la valeur attribuée aux voisins hors de l'image ne sert qu'aux voxels de la couche extérieure, qui sont ensuite remplacés par leur valeur d'entrée). Trois mutants avaient d'abord survécu : mon test ne vérifiait que certaines faces et ne distinguait pas un seuil de 0,5 d'un seuil de 0,6.

