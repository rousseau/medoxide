# medx fetalbet : masque cérébral fœtal

Calcule le masque du cerveau fœtal d'un volume IRM, avec le modèle de
[Fetal-BET](https://github.com/IntelligentImaging/fetal-brain-extraction)
(voir [Algorithmes et références](algorithmes.md)).

```bash
./target/release/medx fetalbet --input /chemin/vers/mon_image.nii.gz --output /chemin/vers/mon_masque.nii.gz
```

La commande reste silencieuse pendant le calcul, puis affiche `Masque écrit dans "…"`.

| Option | Rôle |
|---|---|
| `--input` | Volume NIfTI d'entrée. |
| `--output` | Masque NIfTI de sortie. Son dossier doit exister. |
| `--model` | Poids du modèle. À défaut : variable `MEDOXIDE_MODEL`, puis `models/attunet.bpk` (relatif au dossier courant). |

## Entrée et sortie

- **Entrée** : un volume NIfTI-1 à 3 axes (`.nii` ou `.nii.gz`), IRM fœtale
  pondérée T2. Un fichier 4D est refusé. Les coupes sont prises selon le 3ᵉ axe
  du fichier, sans réorientation.
- **Sortie** : un NIfTI `uint8` (0 fond, 1 cerveau), sur la même grille que
  l'entrée : il se superpose directement à l'image. L'extension `.nii.gz`
  compresse le fichier.
- **Durée** : de 25 secondes à environ 2 minutes par volume de 40 à 50 coupes
  (GPU Apple).

Vérifiez le masque à l'œil : il a été comparé à Fetal-BET, pas à une
segmentation manuelle ([Validation](validation.md)).

## Erreurs

La commande se termine par le code 0 si tout va bien, 1 en cas d'échec (message
préfixé par `medx fetalbet :`), 2 si les arguments sont invalides.

| Message | Cause |
|---|---|
| `poids du modèle introuvables : …` | Chemin relatif au dossier courant : lancer depuis la racine du dépôt, ou passer `--model`. |
| `dossier de sortie introuvable : …` | Créer le dossier avant de lancer. |
| `volume 3D attendu, dimensions : [...]` | Fichier 4D ou plus. |
| `NIfTI : Io(… NotFound …)` / `NIfTI : Io(… "invalid gzip header" …)` | Image d'entrée absente / fichier non NIfTI. |
