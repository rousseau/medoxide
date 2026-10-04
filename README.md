# medoxide

Boîte à outils en Rust & [Burn](https://burn.dev) pour l'imagerie médicale.

## Prérequis

- Rust installé via [rustup](https://rustup.rs).
- Un GPU compatible `wgpu` (Metal sur Mac, Vulkan ou DX12 ailleurs).
- Les poids du modèle (`attunet.bpk`, 121 Mo), qui ne sont pas dans Git : à
  placer dans `models/`, ou à indiquer avec `--model` ou la variable
  d'environnement `MEDOXIDE_MODEL`.

## Compiler

```bash
cargo build --release
```

## Lancer

```bash
./target/release/medx mask --input volume.nii.gz --output mask.nii.gz
# poids ailleurs que dans ./models/attunet.bpk :
./target/release/medx mask --input volume.nii.gz --output mask.nii.gz --model chemin/attunet.bpk
```

Le masque est un NIfTI `uint8` (0 fond, 1 cerveau) qui garde l'affine du
volume d'entrée. Il reproduit [Fetal-BET](https://github.com/IntelligentImaging/fetal-brain-extraction) :
sur les 8 volumes de test, il diffère du masque de référence de 0 à 14 voxels
(Dice ≥ 0,9998). Compter environ 2 minutes par volume de 50 coupes
(build `release`, GPU Apple).

## Structure

```
crates/
  medoxide-mask/   bibliothèque : logique de segmentation (masque cérébral fœtal)
  medx/            binaire CLI : point d'entrée unique, une sous-commande par module
```

Un nouveau module (reconstruction, recalage, ...) = un nouveau crate dans
`crates/`, exposé comme nouvelle sous-commande de `medx`. On ajoute un
crate `medoxide-core` partagé seulement quand un deuxième module en a
réellement besoin — pas avant.

