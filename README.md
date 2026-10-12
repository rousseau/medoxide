# medoxide

Boîte à outils en Rust & [Burn](https://burn.dev) pour l'imagerie médicale.
Chaque méthode est un module, exposé comme une sous-commande de `medx`.

| Sous-commande | Méthode |
|---|---|
| `medx fetalbet` | Masque cérébral en IRM fœtale (portage de [Fetal-BET](https://github.com/IntelligentImaging/fetal-brain-extraction)) |
| `medx svr recon` | Reconstruction coupe-vers-volume : volume 3D à partir de stacks 2D d'IRM fœtale, avec estimation du mouvement de chaque coupe (coupe-coupe ou coupe-volume) ; en développement |
| `medx svr info` | Géométrie de stacks NIfTI (dimensions, espacement, normale, boîte dans le monde) et, avec des masques cérébraux, regroupement des stacks qui partagent un repère |

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

Reconstruction d'un volume à partir de stacks (au moins deux, de plans différents, avec un
masque cérébral chacun, par exemple obtenus par `medx fetalbet`) :

```bash
./target/release/medx svr recon --input stack_ax.nii.gz stack_cor.nii.gz stack_sag.nii.gz \
  --mask masque_ax.nii.gz masque_cor.nii.gz masque_sag.nii.gz --output volume.nii.gz --poses poses.tsv
```

Calculs sur le CPU par défaut ; voir [la page SVR](https://rousseau.github.io/medoxide/svr.html) pour les
options, les durées et les limites.

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
  medoxide-core/       briques communes (lecture de volumes NIfTI), sans dépendance vers un module
  medoxide-fetalbet/   bibliothèque : portage de Fetal-BET
  medoxide-svr/        bibliothèque : reconstruction coupe-vers-volume (en développement)
  medx/                binaire CLI : une sous-commande par module
book/                  documentation (mdBook)
docs/LEARNING.md       journal d'apprentissage Rust/Burn
```

Un nouveau module = un nouveau crate dans `crates/`, exposé comme nouvelle
sous-commande de `medx`. `medoxide-core` ne reçoit que ce que deux modules au moins
utilisent réellement.

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
