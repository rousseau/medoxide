#!/bin/bash
# Exécuté DANS l'image Docker de BOUNTI : rejoue le dernier pas de la chaîne officielle (`transform-image ... -target <entrée> -labels`) sur des étiquettes de test posées sur le cube de 256³ (`etiquettes_cube256.nii.gz`, écrit par `bounti_references_mirtk.py`) et garde le résultat dans `etiquettes_grille.nii.gz`.
# La commande est celle du script officiel `auto-brain-bounti-segmentation-fetal.sh` (GPL v3, voir docs/BOUNTI.md) ; ce script ne sert qu'à produire des références de test.
set -e
M=/bin/MIRTK/build/bin/mirtk
cd /work/$1
$M transform-image etiquettes_cube256.nii.gz etiquettes_grille.nii.gz -target entree.nii.gz -labels
ls -la etiquettes_grille.nii.gz
