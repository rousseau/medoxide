# medoxide

Boîte à outils en Rust & [Burn](https://burn.dev) pour l'imagerie médicale.
Chaque méthode est un module, exposé comme une sous-commande de `medx`.

| Sous-commande | Méthode |
|---|---|
| `medx fetalbet` | Masque cérébral en IRM fœtale (portage de [Fetal-BET](https://github.com/IntelligentImaging/fetal-brain-extraction)) |

## Démarrage rapide

Prérequis : [Rust](https://rustup.rs), un GPU compatible `wgpu` (testé sur Mac
Apple Silicon), et une connexion au premier lancement : les poids du modèle (121 Mo, sur
[Hugging Face](https://huggingface.co/rousseau/medoxide-fetalbet)) sont
téléchargés automatiquement, puis conservés dans `~/.cache/medoxide/`.

```bash
cargo build --release
./target/release/medx fetalbet --input /chemin/vers/mon_image.nii.gz --output /chemin/vers/mon_masque.nii.gz
```

Pour utiliser un fichier de poids local (hors ligne), passez `--model` ou définissez
`MEDOXIDE_MODEL`.

## Documentation

Documentation (installation, utilisation, algorithmes et références, validation) :
<https://rousseau.github.io/medoxide/>. Les sources sont dans [`book/`](book/src/) ;
pour la lire en local :

```bash
cargo install mdbook
mdbook serve book --open
```

## Structure

```
crates/
  medoxide-fetalbet/   bibliothèque : portage de Fetal-BET
  medx/                binaire CLI : une sous-commande par module
book/                  documentation (mdBook)
docs/LEARNING.md       journal d'apprentissage Rust/Burn
```

Un nouveau module = un nouveau crate dans `crates/`, exposé comme nouvelle
sous-commande de `medx`. On ajoute un crate `medoxide-core` partagé seulement
quand un deuxième module en a réellement besoin — pas avant.

## Référence et licence

Le code de medoxide est sous licence MIT (voir `LICENSE`).

`medx fetalbet` est un portage de **Fetal-BET**. Le modèle (architecture et
poids) vient de leurs travaux, publiés sous licence
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) : si vous l'utilisez,
merci de citer :

```bibtex
@article{faghihpirayesh2024fetal,
  title={Fetal-bet: Brain extraction tool for fetal mri},
  author={Faghihpirayesh, Razieh and Karimi, Davood and Erdo{\u{g}}mu{\c{s}}, Deniz and Gholipour, Ali},
  journal={IEEE Open Journal of Engineering in Medicine and Biology},
  year={2024},
  publisher={IEEE}
}
```

Modifications : export ONNX, puis conversion en code Rust pour Burn et poids
`burnpack`. Comme les auteurs, nous précisons que cet outil est destiné à la
recherche et non à un usage médical ou diagnostique, et qu'il est fourni sans
garantie.
