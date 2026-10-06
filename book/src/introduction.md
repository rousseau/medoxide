# medoxide

medoxide est une boîte à outils en Rust, fondée sur [Burn](https://burn.dev),
pour l'imagerie médicale. Chaque méthode est un module, exposé comme une
sous-commande de l'unique commande `medx`.

## Ce qui existe aujourd'hui

| Sous-commande | Méthode | Usage |
|---|---|---|
| `medx fetalbet` | [Fetal-BET](https://github.com/IntelligentImaging/fetal-brain-extraction) | Masque cérébral en IRM fœtale |

Le modèle de `medx fetalbet` n'est pas le nôtre : c'est celui de Fetal-BET
(Faghihpirayesh et al., 2024), converti pour Burn. Les références complètes
sont dans [Algorithmes et références](algorithmes.md).

## Avertissement

medoxide est destiné à la **recherche**. Comme les auteurs de Fetal-BET, nous
précisons qu'il n'est pas prévu pour un usage médical ou diagnostique et qu'il
est fourni sans garantie.
