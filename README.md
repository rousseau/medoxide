# medoxide

Boîte à outils en Rust & [Burn](https://burn.dev) pour l'imagerie médicale.

## Prérequis

- Rust installé via [rustup](https://rustup.rs).

## Compiler

```bash
cargo build --release
```

## Lancer

```bash
./target/release/medx mask --input volume.nii.gz --output mask.nii.gz
```

Pour l'instant, cette commande se contente de répondre que l'inférence
n'est pas encore branchée — c'est normal, c'est la prochaine étape
(voir `docs/LEARNING.md`).

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

