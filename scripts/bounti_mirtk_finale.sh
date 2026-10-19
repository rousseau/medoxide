#!/bin/bash
# Exécuté DANS l'image Docker de BOUNTI : derniers pas de la chaîne officielle sur les sorties déjà produites (réseau officiel de tissus : `bounti_resultats/cnn-lab-in-res-stack-1000.nii.gz` ; composante du masque : `bet_s3.nii.gz`) :
# `transform-image … -target <entrée> -labels`, pour les étiquettes de tissus (`etiquettes_officiel.nii.gz`) et le masque du cerveau (`cerveau_officiel.nii.gz`).
# Les commandes sont celles du script officiel `auto-brain-bounti-segmentation-fetal.sh` (GPL v3, voir docs/BOUNTI.md) ; ce script ne sert qu'à produire des références de test.
set -e
M=/bin/MIRTK/build/bin/mirtk
cd /work/$1
$M transform-image bounti_resultats/cnn-lab-in-res-stack-1000.nii.gz etiquettes_officiel.nii.gz -target entree.nii.gz -labels
$M transform-image bet_s3.nii.gz cerveau_officiel.nii.gz -target entree.nii.gz -labels
ls -la etiquettes_officiel.nii.gz cerveau_officiel.nii.gz
