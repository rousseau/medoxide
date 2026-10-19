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

# medoxide-bounti : poids de BOUNTI convertis pour Burn

> Brouillon de la fiche du dépôt Hugging Face (non publié). Les sommes de contrôle sont à renseigner au moment de l'envoi.

Ces fichiers sont les poids de **BOUNTI** (segmentation du cerveau fœtal en 19 tissus à partir d'une reconstruction 3D T2) et du réseau d'extraction du cerveau de sa chaîne, **convertis au format `burnpack`** pour [Burn](https://burn.dev), afin d'être utilisés par la commande `medx bounti` de [medoxide](https://github.com/rousseau/medoxide) sans Docker ni PyTorch.

**Ce sont les poids des auteurs, sans réentraînement ni ajustement** : seul le format change (export ONNX puis conversion, en `float32`). Le travail est celui de l'équipe de Maria Deprez, Alena Uus et collaborateurs (King's College London) : merci de les citer.

## Origine

- Poids extraits de l'image Docker **`fetalsvrtk/svrtk:general_auto_arm`** (dossier `/home/auto-proc-svrtk/trained_models/`), qui contient les points de contrôle MONAI de BOUNTI.
- Dépôt des auteurs : <https://github.com/SVRTK/auto-proc-svrtk> (scripts d'inférence officiels) ; images : <https://hub.docker.com/r/fetalsvrtk/svrtk>.
- Publication : Uus, A. U., Kyriakopoulou, V., Makropoulos, A., *et al.* **BOUNTI: Brain vOlumetry and aUtomated parcellatioN for 3D feTal MRI.** bioRxiv 2023, doi:[10.1101/2023.04.18.537347](https://doi.org/10.1101/2023.04.18.537347).
- Table des 19 étiquettes (noms, couleurs) : atlas fœtal de KCL, <https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas> (CC0).

## Licence

**GNU General Public License v3.0** : la licence des scripts et du code de `auto-proc-svrtk`, dont l'image Docker est issue (texte : `LICENSE` de ce dépôt amont, <https://www.gnu.org/licenses/gpl-3.0.html>). Le dépôt amont ne mentionne pas les poids séparément ; ces fichiers sont publiés sous la licence de la distribution dont ils proviennent.
Ils sont une **version modifiée (convertie de format)** des poids d'origine. Le code qui les utilise (medoxide) est sous licence MIT ; les poids, eux, restent sous GPL v3 et ne sont pas inclus dans le dépôt de code.

## Fichiers

| Fichier | Réseau | Paramètres | Taille | SHA-256 |
|---|---|---|---|---|
| `bet/bet_64.bpk` | Attention U-Net 3D, extraction du cerveau (2 sorties, canaux 32 à 512) | 23,6 M | 94,5 Mo | à renseigner |
| `unet/unet_64.bpk` | U-Net 3D, 19 tissus + fond (20 sorties, canaux 32 à 512) | 9,83 M | 39,3 Mo | à renseigner |
| `attunet/attunet_64.bpk` | Attention U-Net 3D, 19 tissus + fond (canaux 16 à 256) | 5,91 M | 23,6 Mo | à renseigner |

(`_64` : taille du cube factice de l'export ONNX ; les réseaux sont indépendants de la taille spatiale.)

## Conversion et validation

Points de contrôle PyTorch/MONAI → export ONNX (opset 17) → `onnx2burn` (burn-onnx 0.22.0-pre.4) → `burnpack`. La procédure complète est le script `scripts/bounti_poids.sh` de medoxide.

- **Réseaux** : écart relatif maximal des logits contre PyTorch (float32) de 6,2e-7 à 2,7e-6 sur des volumes réels de 64³ à 256³, avec 100 % d'étiquettes identiques (seuil fixé à 1e-5).
- **Chaîne complète** contre la chaîne officielle (script Python des auteurs et commandes MIRTK), sur trois reconstructions : coefficient de Dice d'au moins 0,9935 sur chacune des 19 étiquettes, accord de voxels de 99,998 à 100 %.
- Détails, critères et écarts : `docs/BOUNTI.md` de medoxide.

## Utilisation

```bash
# télécharger les trois fichiers dans un dossier (ex. ~/.cache/medoxide/bounti, cherché par défaut)
medx bounti --input reconstruction.nii.gz --output segmentation.nii.gz --weights <dossier>
```

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
