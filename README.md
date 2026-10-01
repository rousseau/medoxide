# medoxide

Boîte à outils en Rust & [Burn](https://burn.dev) pour l'imagerie médicale
(reconstruction IRM, segmentation, synthèse, recalage, surface corticale, EEG).
Projet du GIS BeAChild.

Statut : démarrage. Un seul module existe pour l'instant (`mask`), et il
n'est pas encore implémenté — juste câblé. Voir `docs/LEARNING.md` pour le
journal de développement et le raisonnement derrière chaque étape.

## Prérequis

- Rust stable récent, installé via [rustup](https://rustup.rs) (pas via le
  gestionnaire de paquets du système, souvent trop ancien pour Burn).

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

## Roadmap

| Module | Référence Python | Statut |
|---|---|---|
| `mask` — masque cérébral fœtal | [Fetal-BET](https://github.com/IntelligentImaging/fetal-brain-extraction) | 🚧 squelette |
| `register` — recalage inter-coupes | [ROSI / pyrecon](https://github.com/gis-beachild/pyrecon) | ⏳ |
| `recon` — reconstruction 3D | NiftyMIC / SVRTK / NeSVoR | ⏳ |
