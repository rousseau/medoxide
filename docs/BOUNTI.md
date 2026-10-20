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

**~~Deux modes~~ Un seul mode** (le second a été essayé puis retiré, voir 5d-4) : `bounti` (extraction du cerveau par le réseau 3D de BOUNTI, comme la chaîne officielle), suivie de la segmentation BOUNTI. Texte d'origine : « deux modes, `bounti` et `fetalbet` (extraction par Fetal-BET), `medx bounti --extraction bounti|fetalbet` ».

**Chaîne officielle, relevée dans le script de l'image Docker** (les commandes MIRTK sont celles du script ; le code Rust sera écrit à partir de cette description) :
1. *Prétraitement* : retrait des dynamiques ; masque binaire des voxels > 0,005 ; recadrage sur la boîte englobante de ce masque ; **rééchantillonnage linéaire dans l'espace de référence** `brain-ref-space.nii.gz` : grille de **300 × 369 × 300 voxels à 0,6 mm, orientation LAS** (axe x de sens opposé, affine diag(−0,6, 0,6, 0,6)), dont l'origine est recopiée de l'image d'entrée (centrée sur son centre) ; nouveau recadrage ; remplacement des valeurs non finies.
2. *Extraction du cerveau (mode `bounti`)* : `prepare-for-monai` à 128 (voir ci-dessous), Attention U-Net 3D à **2 sorties, canaux 32 à 512** (≠ réseaux de tissus), intensité min-max dans [0, 1], une fenêtre de 128³, argmax ; puis étiquette 1, **plus grande composante connexe**, retour dans l'espace de l'image (interpolation d'étiquettes), **dilatation de 4 puis érosion de 2**, masquage de l'image, recadrage.
3. *Segmentation BOUNTI* : `prepare-for-monai` à 256 sur l'image masquée, fusion des quatre sorties (étape 4), retour dans l'espace de l'image (interpolation d'étiquettes).
4. **`prepare-for-monai` (cube de côté `n`)** : voxel isotrope de taille `1,01 × (plus petit voxel × plus grand nombre de voxels) / n`, rééchantillonnage linéaire, cube de n³ voxels centré sur le centre de l'image, mise à l'échelle des intensités. (Le code officiel a une particularité : en cherchant le plus petit voxel, il retient la taille en z quand la taille en y est la plus petite ; sans effet pour nos volumes isotropes, à reproduire ou à documenter selon le cas.)

**Sous-étapes, chacune validée contre la sortie de l'outil officiel exécuté dans l'image Docker sur trois de nos reconstructions (références produites avant le code de chaque sous-étape) :**
- **5a, géométrie** (grille de référence, recadrages, cube) : mêmes dimensions et même affine que MIRTK ; écart maximal d'intensité ≤ 1e-3 de la plage des valeurs. Échantillonnage trilinéaire : celui de `medoxide-svr` (validé contre scipy et SimpleITK) ou une copie ; la dépendance de `medoxide-bounti` envers `medoxide-svr` (autodiff, flex) est à éviter si un petit module commun suffit.
- **5b, réseau d'extraction 3D** : conversion ONNX → Burn comme pour les deux autres, critère C1 (écart relatif ≤ 1e-5 à PyTorch f32, 99,9 % d'étiquettes), à 128³.
- **5c, nettoyage du masque** : composante connexe, dilatation, érosion, masquage : au moins 99,9 % de voxels identiques à MIRTK.
- **5d, assemblage** et commande `medx bounti` (prévue avec deux modes ; un seul finalement, voir 5d-4).
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
- *passage dans la grille* : interpolation linéaire du masque binaire (convention d'interpolation de la géométrie) puis seuil à 0,5 ; la conversion d'étiquettes de MIRTK (`-labels`) rééchantillonne chaque étiquette par interpolation linéaire et garde celle dont la valeur dépasse 0,5, à défaut le plus proche voisin (règle corrigée en 5d-1 : ma première formulation, « la plus grande pseudo-probabilité », n'avait pas été vérifiée pour plusieurs étiquettes), ce qui n'est pas un plus proche voisin ;
- *dilatation et érosion* : voisinage de 26 voxels (cube 3 × 3 × 3 par itération), 4 puis 2 itérations ; **à chaque itération, la couche extérieure de voxels (les faces de l'image) garde sa valeur** : sans cette règle ma dilatation avait 11 473 voxels de trop, tous sur une face de l'image (des voisins directs du masque, donc pas un problème de forme du voisinage) ; les voisinages de 6 et de 18 voxels, leurs alternances et une dilatation euclidienne ont été essayés et écartés.
- *masquage* : 0 hors du masque ; *recadrage* : boîte englobante du masque.
Tests : cinq tests unitaires (connexité 26 avec un voxel relié par une diagonale pure, cubes de dilatation et d'érosion, couche figée sur les six faces, interpolation et seuil strict à 0,45, 0,5 et 0,55, paramètres 4 et 2 du masque du cerveau, masquage et recadrage) ; **dix-sept mutants du module : seize tués, un équivalent** (la valeur attribuée aux voisins hors de l'image ne sert qu'aux voxels de la couche extérieure, qui sont ensuite remplacés par leur valeur d'entrée). Trois mutants avaient d'abord survécu : mon test ne vérifiait que certaines faces et ne distinguait pas un seuil de 0,5 d'un seuil de 0,6.


## Sous-étape 5d-1 : retour des étiquettes dans la grille de l'image d'origine (critère, avant le code)

**Quoi.** À la fin de la chaîne officielle, les étiquettes de tissus (grille du cube de 256³, voxel d'environ 0,4 mm sur nos volumes) sont ramenées dans la grille de l'image d'entrée (0,8 mm) par `mirtk transform-image <étiquettes> <sortie> -target <entrée> -labels`. Ce n'est pas un plus proche voisin. Hypothèse de départ, non vérifiée pour 19 étiquettes (elle l'a été seulement pour un masque binaire, sous-étape 5c) : chaque étiquette est rééchantillonnée en interpolation linéaire de sa carte binaire, et chaque voxel prend l'étiquette de plus grande valeur interpolée (le fond compris ?, ex æquo ?).
**Référence.** Étiquettes du réseau (fusion à 128³ sur un volume réel : `fusion_128_reel_labels.u8`, 19 étiquettes, agrandies d'un facteur 2 par plus proche voisin pour occuper 256³) posées sur la géométrie du cube de 256³ de chacun des trois volumes (`cube256.nii.gz` de MIRTK), puis transformées par MIRTK dans la grille de `entree.nii.gz` (script `scripts/bounti_mirtk_etiquettes.sh`). Les étiquettes ne sont pas alignées avec le volume : sans importance, seules la règle d'interpolation et la gestion des bords sont testées, sur des frontières nombreuses entre vingt valeurs.
**Critère (fixé avant de coder).** Au moins 99,9 % de voxels identiques à MIRTK sur la grille d'entrée de chacun des trois volumes (environ 2,5 millions de voxels, soit au plus 2 500 voxels différents) ; tout écart restant doit se situer sur des frontières entre étiquettes (aucun voxel d'une étiquette absente du voisinage), et l'ensemble des étiquettes présentes doit être le même. Les règles trouvées (ex æquo, fond, bords du cube) sont consignées ici.

### Résultat de 5d-1

Code : `crates/medoxide-bounti/src/etiquettes.rs` (`etiquettes_vers_la_grille`) ; test `the_labels_back_to_the_grid_match_mirtk` (`cargo test --release -p medoxide-bounti the_labels -- --ignored --nocapture`).

| Volume | voxels de la grille d'entrée | voxels différents de MIRTK | taux |
|---|---|---|---|
| calme 1 | 2 598 288 | **0** | 100 % |
| calme 2 | 2 390 750 | **0** | 100 % |
| calme 3 | 2 511 270 | **4** | 99,9998 % |

**Critère atteint** (≥ 99,9 %). **L'hypothèse de départ était fausse pour les jonctions** : « l'étiquette de plus grande valeur interpolée » donnait déjà 0, 102 et 197 voxels différents (sous le seuil, mais avec une cause systématique, pas du bruit : sur les voxels en désaccord les valeurs interpolées différaient de 0,02 à 0,27, ce n'étaient pas des ex æquo). En affichant, pour douze voxels en désaccord, toutes les valeurs interpolées et l'étiquette du plus proche voisin, **MIRTK avait retenu à chaque fois l'étiquette du plus proche voisin**. D'où deux règles essayées sur les 7 millions de voxels : plus proche voisin seul (0, 4 066 et 3 681 voxels différents : faux), et « l'étiquette dont la valeur interpolée dépasse 0,5, à défaut le plus proche voisin » (0, 0 et 4 : retenue).

**Règle retenue** : (1) chaque étiquette présente est rééchantillonnée comme une carte binaire (interpolation linéaire, convention de la géométrie) ; (2) un voxel prend l'étiquette dont la valeur dépasse 0,5 (il ne peut y en avoir qu'une : les valeurs de toutes les étiquettes somment à 1 à l'intérieur de la grille) ; (3) à défaut, l'étiquette du plus proche voisin dans la grille de départ (arrondi de chaque coordonnée de voxel au plus proche), 0 hors de la grille (**corrigé en 5d-2** : le voxel de bord est prolongé d'un voxel entier au-delà du dernier centre, puis 0). Le « fond » n'a pas de rôle particulier : 0 est une étiquette comme les autres, et la valeur par défaut hors de la grille.
**Les 4 voxels restants** de calme 3 ont tous une valeur interpolée de 0,500008 (au-dessus de 0,5 de 8e-6) : MIRTK calcule ses coordonnées en simple précision et reste de l'autre côté du seuil ; même classe d'écart qu'en 5c (inversions d'un voxel près d'un seuil).
**Conséquence pour 5c** : la phrase « la conversion d'étiquettes de MIRTK (`-labels`) garde l'étiquette de plus grande pseudo-probabilité » (règles de 5c) était une hypothèse non testée pour plusieurs étiquettes ; pour un masque binaire elle coïncide avec la règle ci-dessus (une valeur > 0,5 ou ≤ 0,5 contre un fond de valeur complémentaire, plus proche voisin quand la valeur est exactement 0,5 : voir ci-dessous) ; le masque binaire de 5c reste validé par MIRTK (0 à 2 voxels).
**Tests** : neuf tests unitaires (même grille, jonction de trois étiquettes autour de chacun des huit coins, seuil encadré à 0,35 et 0,45, ordre des affines, source anisotrope et décalée, bords du cube, des deux côtés sur chaque axe) ; **dix-huit mutants sur dix-neuf tués** (seuil, `>`/`>=`, arrondi sur chaque axe, bornes, valeur hors grille, surcharge, plus proche voisin, ordre des affines, permutation d'axes). Huit mutants avaient survécu à la première série de tests : le plus proche voisin n'était testé que pour des décalages < 0,5, sur un axe, et le seuil que par des valeurs éloignées de 0,5. Survivant : `>` remplacé par `>=` à exactement 0,5, **non vérifiable** contre MIRTK (ses coordonnées en simple précision rendent un score de 0,5 exact introuvable) ; le cas n'est pas tranché ici.

## Sous-étape 5d-2 : assemblage en mode `bounti` et critère C3 (critères, avant la mesure)

**Code** : `crates/medoxide-bounti/src/segmentation.rs` (`Reseaux::charger`, `segmenter`) : prétraitement, extraction du cerveau (cube de 128³), composante, retour dans la grille préparée, dilatation 4, érosion 2, masquage, cube de 256³, fusion des quatre sorties, retour dans la grille de l'image d'entrée. Rend les étiquettes et le masque du cerveau (composante, avant dilatation, comme `mask-bet` du script officiel).
**Référence** : le script officiel complet de l'image Docker (`auto-brain-bounti-segmentation-fetal.sh`), exécuté tel quel sur les trois mêmes volumes (`data/bounti/officiel/<id>/out`), avec ses deux sorties : étiquettes de tissus et masque du cerveau.
**Critères (fixés avant la mesure ; ce sont ceux de C3 de l'étape 5)** pour chacun des trois volumes, sur la grille de l'image d'entrée :
- **C3 étiquettes** : coefficient de Dice de chaque étiquette de 1 à 19 présente dans la référence ≥ 0,98, et aucune sous 0,95 (une étiquette présente dans un seul des deux résultats compte pour un Dice de 0 : échec) ;
- **masque du cerveau** : Dice ≥ 0,99 ;
- rapportés sans seuil : accord de voxels, Dice moyen sur les étiquettes, durée et mémoire maximale.
**Si C3 échoue** : on ne règle pas le seuil. On localise l'écart par étape (étiquettes du réseau d'extraction contre `bet_etiquettes_cube` ; cube de 256³ contre `cube256_masque` ; logits fusionnés), car chaque étape a déjà son critère contre l'outil officiel et l'écart ne peut venir que de leur composition (par exemple d'une différence d'un voxel dans le masque qui déplace le recadrage et donc la grille du cube de 256³).

### Noms et couleurs des étiquettes (pour `dseg.tsv`)

Source : fichier `info/dhcp-atlas-ITKSNAP-19-labels-colour-scheme.txt` du dépôt de l'atlas fœtal de KCL (`gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas`, licence **CC0**), téléchargé brut et relu : 1 et 2 liquide cérébrospinal externe gauche et droit, 3 et 4 cortex, 5 et 6 substance blanche fœtale, 7 et 8 ventricules latéraux, 9 cavum du septum pellucide, 10 tronc cérébral, 11 et 12 cervelet, 13 vermis, **14 et 15 noyaux gris centraux (« basal ganglia ») gauche et droit, 16 et 17 thalamus gauche et droit**, 18 troisième ventricule, 19 quatrième ventricule ; **dans chaque paire le numéro de gauche précède celui de droite : impair à gauche pour 1 à 12, pair à gauche pour 14 et 15 puis 16 et 17** (une première rédaction de cette section disait « impairs à gauche » pour toutes les paires : erreur, trouvée par le test de la table contre la permutation de la fusion). Les couleurs sont celles du même fichier.
**Vérifications indépendantes** : (1) les regroupements du script de volumétrie officiel de l'image Docker (1-2 liquide externe, 3-4 cortex, 5-6 substance blanche, 7-8 ventricules latéraux, 11-13 cervelet, 14-17 « deep GM », 18-19 ventricules médians) et les paires échangées par le retournement (1↔2 … 14↔15, 16↔17) sont cohérents avec cette table ; (2) la position des centres de gravité de la segmentation officielle de calme 1 donne, pour les cinq paires 1-2, 3-4, 5-6, 7-8 et 11-12, l'étiquette impaire du côté gauche (x plus petit en RAS), sans exception. Cette mesure ne tranche pas 14 à 17 : la segmentation de ce volume est de mauvaise qualité (volumes gauche et droit du simple au double, 0,04 mL pour le cervelet droit), les centres des paires 14-15 et 16-17 ont presque le même x ; la source écrite fait foi pour ces quatre-là.

### Résultat de 5d-2

**Écart de méthode sur la référence (à connaître).** Le script officiel complet n'a pas pu être exécuté d'un seul tenant dans Docker : sa dernière étape (le réseau de tissus à 256³, deux réseaux et quatre passes en PyTorch sur CPU) a été tuée (`Killed`) par le manque de mémoire de la machine virtuelle de Docker Desktop (7,7 Gio), après environ 340 s, sur les trois volumes. Je n'ai pas modifié le réglage de Docker. La référence est donc enchaînée à partir des morceaux officiels : prétraitement, extraction du cerveau, nettoyage et cubes (commandes MIRTK et script Python du BET officiel, dans l'image) ; réseau de tissus (le script Python officiel `run_monai_comb_red_atunet_unet_segmentation-2022-lr.py`, tel quel, **sur le Mac**, sous `garde_memoire.py 19`, environ 20 minutes par volume, sur le cube de 256³ produit par MIRTK) ; retour des étiquettes et du masque dans la grille de l'entrée (MIRTK, dans l'image). Chaque morceau est l'officiel ; seul l'enchaînement est le mien (`scripts/bounti_references_mirtk.py`, étapes `reseau` et `finale`).

| Volume | accord de voxels | Dice moyen (19 étiquettes) | Dice minimal | Dice du masque du cerveau | durée de `segmenter` |
|---|---|---|---|---|---|
| calme 1 | 1,00000 | 1,0000 | 1,0000 (étiquette 5) | 0,9987 | 38,9 s |
| calme 2 | 1,00000 | 1,0000 | 1,0000 (étiquette 3) | 0,9990 | 42,1 s |
| calme 3 | 0,99998 | 0,9996 | **0,9935** (étiquette 18, troisième ventricule) | 0,9984 | 40,1 s |

**C3 atteint, avec une très grande marge** (seuils : Dice ≥ 0,98 par étiquette, aucune sous 0,95 ; masque ≥ 0,99). Durées sur le GPU du Mac (wgpu), de la lecture du volume aux étiquettes sur sa grille, contre environ 20 minutes pour le script officiel sur le CPU. Le test (`the_full_chain_matches_the_official_chain`) passe sous une limite de 18 Go.
**Contrôle du réseau d'extraction** (vérification qui restait de 5b) : sur le cube de 128³ officiel, **0 voxel d'étiquette différente** sur les trois volumes (test `the_brain_extraction_labels_match_the_official_ones`).

**Deux corrections trouvées en comparant le masque du cerveau** (le résultat de 5d-1 était vrai pour son test, incomplet) :
1. **Prolongation du bord des images d'étiquettes : un voxel entier, pas un demi-voxel.** Sur calme 2, 73 voxels hors de la grille de départ (à 0,83 voxel au-delà du dernier centre) étaient à 1 chez MIRTK et à 0 chez moi. Mesure sur les voxels hors des centres dont le voxel de bord vaut 1 : MIRTK met 1 pour 73 sur 74 (excès entre 0,75 et 1,0), 31 sur 32 et 137 sur 139 (excès jusqu'à 0,25), 0 sur 88 entre 1,0 et 1,25 et au-delà. L'interpolation d'**intensités** garde un demi-voxel (validé en 5a). Code : `Volume::reechantillonner_avec_marge`, constantes `MARGE_INTENSITES = 0,5` et `MARGE_ETIQUETTES = 1`, appliquée au passage d'un masque dans la grille (`masque::vers_la_grille`, 5c) et aux étiquettes (5d-1) ; le plus proche voisin prolonge aussi le voxel de bord sur cette distance.
2. **Écarts restants du masque : du bruit d'arrondi sur des ex æquo exacts, pas une règle manquante.** La grille de 0,6 mm passée dans une grille de 0,8 mm (rapport 4/3) place des centaines de voxels à une coordonnée source de `.5` exact. Le résultat de ces ex æquo dépend de l'affine au millionième de millimètre : en décalant l'affine de la source de 1e-7 mm l'écart à MIRTK ne bouge pas (218, 85, 166 voxels), en la décalant de 1e-5 mm il passe à 1736, 206 et 1878 voxels, soit l'écart observé dans la chaîne (1777, 1225 et 1793 voxels sur 692 000, 595 000 et 546 000) où mon affine préparée diffère de celle de MIRTK d'au plus 1e-5 mm (MIRTK stocke ses affines en simple précision dans ses fichiers intermédiaires). Le Dice du masque (0,998 à 0,999) en est la conséquence ; les étiquettes de tissus, elles, n'en dépendent presque pas (dilatation, érosion et masquage absorbent ces voxels).
**Mutants** (règles de bord) : douze essayés, **sept tués**, quatre survivants sur des **valeurs exactement égales à la marge** (position à ±1,0 voxel au-delà du dernier centre) dont la convention (incluse ou exclue) n'est pas vérifiable contre MIRTK, un équivalent (la borne basse du `clamp`, redondante avec la conversion saturante d'un flottant négatif en entier). Un survivant réel (marge d'un demi-voxel dans les valeurs interpolées d'une jonction au bord) a donné un test de plus, après quoi il est tué.
**Tests ajoutés ou modifiés** : marge paramétrable de l'interpolation, bord des étiquettes sur chaque axe des deux côtés, bord du masque, prolongation appliquée aux valeurs interpolées, Dice, et trois tests ignorés (C3 de bout en bout, extraction du cerveau contre l'officielle, diagnostic du dernier pas du masque avec perturbation de l'affine).

## Sous-étape 5d-3 : commande `medx bounti`, sorties et poids (résultat)

**Commande** : `medx bounti --input <reconstruction.nii.gz> (--output <dseg.nii.gz> [--brain-mask <masque>] | --bids <racine> --participant X [--session Y] [--output-dir <racine>]) [--weights <dossier>] [--device wgpu|cpu] [--extraction bounti]`. Les chemins de sortie, les identifiants BIDS et la présence des poids sont vérifiés **avant** tout calcul. Il n'y a pas de choix d'extraction du cerveau : voir la sous-étape 5d-4 (mode `fetalbet` essayé puis retiré).
**Sorties** (contrat de guioxide, `docs/issues/medoxide.md` n° 5) : `<racine>/derivatives/medx-bounti/sub-X/ses-Y/anat/sub-X_ses-Y_desc-bounti_dseg.nii.gz` (`uint8`, grille et affine de l'entrée, `sform` et `qform` cohérents), `…_dseg.tsv` (`index name abbreviation color`, table de `table.rs`), `…_desc-bounti_mask.nii.gz` (masque du cerveau, 0/1, **ajouté** : non demandé par guioxide, à confirmer avec eux), et `derivatives/medx-bounti/dataset_description.json` (nom, version, générateur). Sans session, les fichiers perdent `ses-Y` (BIDS le permet). Les identifiants sont limités aux lettres et chiffres (la valeur entre dans un chemin).
**Poids** : `--weights`, sinon `MEDOXIDE_BOUNTI_WEIGHTS`, sinon `~/.cache/medoxide/bounti` ; trois fichiers `bet/bet_64.bpk`, `unet/unet_64.bpk`, `attunet/attunet_64.bpk`. **Rien n'est téléchargé** (licence GPL v3 supposée, non confirmée ; publier ces poids ailleurs est une décision à prendre avec les auteurs). `scripts/bounti_poids.sh` les produit depuis l'image Docker des auteurs : extraction des trois checkpoints, export ONNX vérifié contre onnxruntime (C1), conversion `onnx2burn`, **contrôle que le code Rust généré est celui qui est compilé** (aux imports, à la mise en forme, aux accolades et au chemin du fichier de poids près), copie. Un message d'erreur clair (fichiers manquants, procédure, licence) remplace la panique du chargement.
**Vérification de bout en bout** (la commande réelle, binaire de production, sur calme 1, 42 s sur le GPU du Mac) : fichiers relus avec nibabel : `uint8`, même forme et même affine que l'entrée (`sform` et `qform`), **1 voxel différent** de la sortie officielle sur 2 598 288 (Dice 1,0000 sur les 19 étiquettes), masque de 690 567 voxels. Avec des poids **reproduits par le script** (nouveaux fichiers, octets différents des précédents) : **0 voxel différent** pour les étiquettes et pour le masque.
**Tests** : 4 tests du module de poids et de sortie (ordre de priorité, valeur vide, dossier incomplet non contourné, fichiers manquants listés exactement ; chemins BIDS avec et sans session, identifiants refusés, aller-retour NIfTI sur une affine oblique et anisotrope, dossiers créés, pas de masque sans chemin), 1 test des arguments de la ligne de commande (treize combinaisons), 6 tests de la table.

## Sous-étape 5d-4 : mode `fetalbet` (conception et critères, avant la mesure)

**Conception.** Le masque du cerveau vient de Fetal-BET (`medoxide-fetalbet`, réseau 2D, rééchantillonnage à 1 mm dans le plan, normalisation par coupe, coupes selon le troisième axe du volume), **appliqué au volume d'entrée sur sa propre grille** (`mask_from_volume`). Il remplace l'extraction par le réseau 3D de BOUNTI ; tout le reste de la chaîne est identique : plus grande composante connexe (connexité 26), passage dans la grille préparée (règle des étiquettes de MIRTK, 5d-1), dilatation de 4 puis érosion de 2, image masquée et recadrée, cube de 256³, fusion des quatre sorties, retour dans la grille de l'entrée. Seuls les deux réseaux de tissus sont donc chargés. Le masque du cerveau rendu est la plus grande composante du masque de Fetal-BET, sur la grille de l'entrée.
**Limite connue, à mesurer et non à supposer** : Fetal-BET a été entraîné sur des piles de coupes 2D, et il coupe selon le troisième axe du volume ; une reconstruction 3D n'a pas d'axe de coupe privilégié (la nôtre garde l'orientation de sa grille).
**Critères (fixés avant la mesure ; ce sont ceux de l'étape 5)** sur les trois volumes, contre la même référence officielle que le mode `bounti` (étiquettes de la chaîne officielle) : **Dice moyen sur les 19 étiquettes ≥ 0,95** pour chaque volume. Rapportés sans seuil : Dice par étiquette, Dice du masque du cerveau contre celui de la chaîne officielle, écart entre les deux modes de `medx bounti`. Si le critère échoue, on le dit et on localise la cause (masque de Fetal-BET contre masque officiel d'abord) ; le seuil n'est pas ajusté.

### Résultat de 5d-4 : **critère non atteint**

Code : `Extraction::Fetalbet` (`crates/medoxide-bounti/src/segmentation.rs`), `medoxide_fetalbet::mask_from_volume` (extraite de `segment` sans changer son comportement), `medx bounti --extraction fetalbet [--model]`. Test : `the_fetalbet_mode_meets_the_dice_criterion_against_the_official_chain` (échouait, comme attendu des résultats ci-dessous). **Ce code a ensuite été retiré (voir la fin de cette section) ; il reste dans l'historique git (commit 068e72a).**

| Volume | Dice moyen (19 étiquettes), critère ≥ 0,95 | pire étiquette | Dice du masque du cerveau contre l'officiel |
|---|---|---|---|
| calme 1 | **0,2668** | 0,000 (étiquette 12) | 0,6125 |
| calme 2 | **0,0116** | 0,000 (étiquette 7) | 0,1048 |
| calme 3 | **0,5602** | 0,138 (étiquette 17) | 0,6486 |

Le seuil n'est pas ajusté (règle posée avant la mesure). **Causes, localisées sur calme 1** (la commande `medx fetalbet` existante, sans mon code, donne le même masque : Dice 0,6255 contre 0,6125 après la plus grande composante, donc ce n'est pas une erreur d'intégration) :
- **Le masque de Fetal-BET est trop étroit pour BOUNTI** : 97,8 % de ses voxels sont dans le masque officiel (précision), mais il n'en couvre que 46 % (325 517 voxels contre 692 344). Le masque de l'extraction de BOUNTI inclut le liquide cérébrospinal externe (étiquettes 1 et 2, 67 mL sur ce volume) que Fetal-BET, entraîné à extraire le cerveau, laisse de côté : ce sont deux définitions différentes du « masque du cerveau », pas un défaut d'un des deux. La dilatation de 4 puis l'érosion de 2 (+1,2 mm net) ne comble pas l'écart.
- **L'axe de coupe compte** (hypothèse de la limite notée avant la mesure : confirmée) : Fetal-BET coupe selon le troisième axe du volume ; en prenant tour à tour chaque axe comme axe des coupes, le Dice du masque est de 0,6255 (selon z), 0,7213 (selon y) et **0,8657 (selon x)**, avec une couverture de 0,46, 0,57 et 0,78 et une précision constante de 97,7 %. Une reconstruction 3D n'a pas d'axe de coupe privilégié ; Fetal-BET est un réseau 2D entraîné sur des piles de coupes.
- **calme 2** est le plus mauvais (Dice du masque 0,10) : cause non analysée.
**Conséquence** : tel quel, `--extraction fetalbet` ne donne pas une segmentation comparable à celle de BOUNTI sur nos reconstructions ; l'option avait été marquée expérimentale dans l'aide de la commande (avant son retrait, ci-dessous). Les pistes (combiner les trois axes de coupe, élargir le masque pour inclure le liquide externe) auraient demandé un nouveau critère fixé à l'avance et des volumes qui n'ont pas servi à les mettre au point.

**Décision de l'utilisateur : mode retiré.** Fetal-BET sert en amont, à produire les masques des piles pour la reconstruction ; BOUNTI s'applique ensuite tel quel à la reconstruction, avec sa propre extraction 3D du cerveau, comme la chaîne officielle. Le choix « full BOUNTI ou FetalBET + BOUNTI » demandé au départ était donc une mauvaise lecture du besoin (je l'avais compris comme un second mode d'extraction dans la segmentation). Retirés : `Extraction` (retour à une chaîne simple), `medoxide_fetalbet::mask_from_volume`, les options `--extraction` et `--model` de `medx bounti`, la dépendance de `medoxide-bounti` envers `medoxide-fetalbet`, et le caractère facultatif des poids du réseau d'extraction. Restent : l'analyse ci-dessus et ses chiffres (utiles si la question revient : Fetal-BET et BOUNTI ne définissent pas le « masque du cerveau » de la même façon).

### C2 à 160³ (complément de l'étape 4)

La référence PyTorch de la fusion à 160³ (`scripts/make_bounti_fusion_reference.py 160`, sous `garde_memoire.py 11`, pic de mémoire de 10,1 Go) et le test `the_fusion_matches_the_pytorch_reference` (`BOUNTI_N=160`) : **0 voxel d'étiquette différent sur 4 096 000** et écart relatif des logits fusionnés de **1,45e-6** (critère : ≥ 99,9 % d'étiquettes identiques). Avec le résultat à 128³ (0 voxel sur 2 097 152, 1,6e-6), C2 est atteint aux deux tailles de référence testées ; la chaîne complète à 256³ est validée par C3 (sous-étape 5d-2).

### Publication des poids sur Hugging Face

Décision de l'utilisateur (le 2026-10-20, après celle de principe de la veille) : publier les poids convertis, en citant le travail, le dépôt GitHub des auteurs et la licence de leur distribution. **Publié** : <https://huggingface.co/rousseau/medoxide-bounti> (public, étiquette `license:gpl-3.0`), contenant `bet/bet_64.bpk`, `unet/unet_64.bpk`, `attunet/attunet_64.bpk`, le texte de la licence (`LICENSE`, copie de celui de `auto-proc-svrtk`) et la fiche (`README.md` : origine, licence, version modifiée, conversion, validation, citation, usage et limites ; source dans `docs/huggingface/bounti-model-card.md`). Les fichiers envoyés sont ceux produits par `scripts/bounti_poids.sh` et validés de bout en bout (0 voxel de différence avec les poids précédents sur calme 1).
| Fichier | taille (octets) | SHA-256 |
|---|---|---|
| `bet/bet_64.bpk` | 94 503 944 | `3b288ec2…e0d31fe6` |
| `unet/unet_64.bpk` | 39 342 160 | `6ac7d9b7…ac6ae8` |
| `attunet/attunet_64.bpk` | 23 646 288 | `a3c9a4c5…e45774fa` |
(sommes complètes dans la fiche.) **Vérification** : dépôt retéléchargé par `snapshot_download`, les cinq fichiers (trois poids, `LICENSE`, `README.md`) sont identiques bit à bit à ceux envoyés. Le jeton utilisé (à portée limitée à l'écriture sur le compte) n'a été passé que par variable d'environnement, jamais écrit dans un fichier ni dans le dépôt.
Le dépôt amont ne mentionne pas les poids séparément (« the auto SVRTK code and all scripts » sous GPL v3) : la fiche le dit et les publie sous la licence de la distribution d'où ils viennent. Reste à faire : téléchargement automatique par `medx bounti` (URL et SHA-256, comme `medx fetalbet`).
