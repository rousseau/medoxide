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

**C1 est atteint pour l'Attention U-Net à toutes les tailles testées et pour l'U-Net jusqu'à 128³ ; il reste manqué pour l'U-Net à 256³ (2,6e-5 contre 1e-5)**, avec un accord des étiquettes de 99,9998 %. Le critère n'est pas relâché. Question ouverte : l'écart restant vient-il de Burn ou de la référence PyTorch f32 elle-même ? Le calcul de référence en float64 à 256³ n'a pas pu être fait : il a **fait planter le Mac par manque de mémoire** (la référence en f32 à 256³ occupait déjà 6,6 Go pour l'U-Net et 13 Go pour l'Attention U-Net, et la double précision demande environ le double ; je l'avais lancé sans borner la mémoire). À refaire à une taille intermédiaire (192³) en surveillant la mémoire, ou par morceaux.

## Critères (écrits avant le code)

BOUNTI, sur un volume de test fixé (cube de 64³ aléatoire puis un volume réel de 256³) :
- **C1, par réseau** : écart **relatif** maximal (max |Δ| / max |logit|) entre les logits Rust (f32) et les logits PyTorch/MONAI ≤ 1e-5, et au moins 99,9 % d'étiquettes identiques après argmax.
  *Amendement (2026-10-19, après la mesure de l'export ONNX)* : la première rédaction fixait un écart **absolu** ≤ 1e-3. Elle est fausse pour les réseaux aux logits de grande amplitude : l'Attention U-Net (logits jusqu'à environ 300) donne 1,34e-3 d'écart absolu pour un écart relatif de 4,4e-6, identique à celui de l'U-Net (6,6e-4 absolu, 4,4e-6 relatif). Le projet avait déjà établi ce point pour Fetal-BET (seuil relatif de 1e-5). Le critère est donc relatif ; le changement a été décidé en voyant ce résultat et est signalé comme tel.
- **C2, fusion** (quatre sorties, retournement et échange des étiquettes) : au moins 99,9 % de voxels d'étiquette identique à la référence Python/MONAI à entrée prétraitée identique.
- **C3, chaîne complète** (prétraitement et post-traitement réécrits en Rust, sur trois de nos reconstructions) : Dice moyen par étiquette ≥ 0,98 contre la sortie du script Docker, aucune étiquette sous 0,95.
- **C4, ressources** (informatif, à mesurer d'abord) : mémoire maximale et temps d'une inférence à 256³ sur le Mac (24 Go de RAM). Si la mémoire dépasse 20 Go, étudier l'inférence par fenêtres ou la réduction de précision avant d'aller plus loin.
- Multi-BOUNTI : critères équivalents, écrits quand son architecture et ses poids auront été relevés.

## Plan, par petites étapes (une chose nouvelle à la fois)

0. **Faisabilité** (fait en partie) : poids et scripts extraits, architectures identifiées, chargement strict vérifié. Reste : référence de sortie sur nos volumes.
1. **Sonde de performance en Burn** (fait, voir ci-dessus) : GO sur le GPU.
2. **Export ONNX et génération** (voie de conversion ci-dessus) : installer `onnx2burn`, exporter l'U-Net MONAI en ONNX, générer le code et les poids `burnpack`, valider C1 sur un cube de 64³ ; repli couche par couche si une opération 3D n'est pas prise en charge.
3. **Attention U-Net** par la même voie, validé par C1.
4. **Fusion et augmentation** (C2).
5. **Pré et post-traitement** en Rust (C3), puis commande `medx`.
6. **Multi-BOUNTI**.

## Risques

- Performance et mémoire des convolutions 3D à 256³ (puis 0,5 mm pour Multi-BOUNTI) sur le Mac.
- Prétraitement MIRTK à reproduire fidèlement (le modèle y est sensible, comme Fetal-BET).
- Licence GPL v3 des poids à confirmer avec les auteurs.
- Poids de Multi-BOUNTI dans une image AMD seulement : extraction sans exécution possible, référence de sortie à produire sur une machine x86.
