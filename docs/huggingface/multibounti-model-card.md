---
license: gpl-3.0
library_name: burn
tags:
  - medical-imaging
  - fetal-mri
  - segmentation
  - brain
  - burn
  - rust
  - onnx
---

# medoxide-multibounti : poids de Multi-BOUNTI (cerveau fœtal, 43 étiquettes) convertis pour Burn

Ces fichiers sont les poids de **Multi-BOUNTI** (parcellation du cerveau fœtal en 43 étiquettes à partir d'une reconstruction 3D T2 à 0,5 mm), pour la chaîne **fœtale « general »**, **convertis au format `burnpack`** pour [Burn](https://burn.dev), afin d'être utilisés par la commande `medx multibounti` de [medoxide](https://github.com/rousseau/medoxide) sans Docker ni PyTorch.

**Ce sont les poids des auteurs, sans réentraînement ni ajustement** : seul le format change (export ONNX puis conversion, en `float32`). Le travail est celui de l'équipe SVRTK (King's College London) : merci de la citer. **Statut : expérimental** (voir « Validation » : le recalage propre à medoxide ne reproduit pas la sortie officielle).

## Origine

- Poids extraits de l'image Docker **`fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd`** (dossier `/home/perinatal-brain-mri-analysis/`).
- Dépôt des auteurs : <https://github.com/SVRTK/auto-proc-svrtk> ; images : <https://hub.docker.com/r/fetalsvrtk/svrtk>.
- Points de contrôle convertis : `atunet_bet_brain_153t_svr_2026_128` (extraction du cerveau), `atunet_lr_brain_fetal_2lab` (gauche/droite), `patch_atunet_new_multi_43_brain_neo_fetal_aff_fix_022026` (43 étiquettes).
- `gabarit/lr-lab.nii.gz` : gabarit gauche/droite `templates/ref-bounti/lr-lab.nii.gz` de la même image, **copié tel quel** (il sert au recalage).
- Aucune publication propre à Multi-BOUNTI n'a été trouvée lors de la conversion : citer le dépôt ci-dessus et l'article de BOUNTI, dont il prolonge la méthode (voir Citation).

## Licence

**GNU General Public License v3.0** : celle des scripts de l'image (« Copyright 2026 - King's College London … GNU General Public License v3.0 » en tête de chaque script). Le dépôt amont ne mentionne pas les poids séparément ; ces fichiers sont publiés sous la licence de la distribution dont ils proviennent.
Ce sont une **version modifiée (convertie de format)** des poids d'origine. Le code qui les utilise (medoxide) est sous licence MIT ; les poids restent sous GPL v3. Le texte de la licence est dans le fichier `LICENSE` de ce dépôt (copie de celui de `auto-proc-svrtk`).

## Fichiers

| Fichier | Réseau | Taille | SHA-256 |
|---|---|---|---|
| `bet/bet_64.bpk` | Attention U-Net 3D, extraction du cerveau (2 sorties) | 94,5 Mo | `8f39a5400e6edd147a8296ee35d8b3b27ad18dfe4a3a3e94af0e69619da1d455` |
| `lr/lr_64.bpk` | Attention U-Net 3D, gauche/droite (3 sorties) | 94,5 Mo | `d9c20fe02c4e52ddcc73af38f9357ab47412a87a33a3d712cd31bf9e451436fd` |
| `multi/multi_64.bpk` | Attention U-Net 3D, 43 étiquettes + fond (44 sorties) | 94,5 Mo | `747e9ecbe47c0650014850532962398d51afdb24aec9c3c2629a0ab1b57e9344` |
| `gabarit/lr-lab.nii.gz` | gabarit gauche/droite (NIfTI) | 0,5 Mo | `9daf554bf37f7cb607c8c5ba92de611f94b547dbcec9136bf366396b9f2beb7d` |

(`_64` : taille du cube factice de l'export ONNX ; les réseaux sont indépendants de la taille spatiale. Les trois réseaux ont la même architecture : canaux 32 à 512, dropout 0,5.)

## Validation (détails : `docs/MULTIBOUNTI.md` de medoxide)

- **Réseaux** : écart relatif des logits contre PyTorch (float32) de 2,3e-6 à 2,7e-6 à 128³ sur des entrées réelles, ≥ 99,9999 % d'étiquettes identiques (seuil fixé à 1e-5).
- **Fenêtres glissantes** (343 fenêtres de 128³, recouvrement 0,8) : 0 voxel différent sur 16,7 millions contre la sortie officielle.
- **Géométrie** (rééchantillonnages, recadrages, masques) : identique à MIRTK (écarts ≤ 7e-6).
- **Chaîne complète avec le recalage de MIRTK** : Dice 1,0000 contre la sortie officielle.
- **Chaîne complète avec le recalage propre à medoxide** : Dice moyen **0,72** sur un volume. La sortie officielle dépend fortement du recalage affine de MIRTK, lui-même peu répétable : deux autres passages de MIRTK donnent un Dice de 0,71 et 0,57 contre la sortie officielle. Les noms des 43 étiquettes ne sont pas fournis ici.

## Utilisation

```bash
# télécharger tout le dépôt (en gardant les sous-dossiers) dans un dossier ;
# vérifier les sommes de contrôle avec : shasum -a 256 bet/bet_64.bpk lr/lr_64.bpk multi/multi_64.bpk gabarit/lr-lab.nii.gz
export MEDOXIDE_MULTIBOUNTI_WEIGHTS=<dossier>
medx multibounti --input reconstruction_0p5mm.nii.gz --output segmentation.nii.gz
```
L'entrée doit être une reconstruction **à 0,5 mm** (isotrope).

## Citation

```bibtex
@article{uus2023bounti,
  title={BOUNTI: Brain vOlumetry and aUtomated parcellatioN for 3D feTal MRI},
  author={Uus, Alena U. and Kyriakopoulou, Vanessa and Makropoulos, Antonios and others},
  journal={bioRxiv},
  year={2023},
  doi={10.1101/2023.04.18.537347}
}
```

Merci aussi de citer le logiciel et les images Docker SVRTK (<https://github.com/SVRTK/auto-proc-svrtk>).

## Usage et limites

Outil destiné à la recherche, **pas à un usage médical ou diagnostique**, fourni sans garantie (mêmes réserves que les auteurs : en aucun cas les auteurs ou distributeurs ne sont responsables des dommages liés à son usage).
