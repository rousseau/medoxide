#!/bin/bash
# Exécuté DANS l'image Docker de BOUNTI (par `bounti_references_mirtk.py`) : rejoue, dans l'ordre du script officiel `auto-brain-bounti-segmentation-fetal.sh`,
# le prétraitement MIRTK d'un volume (entrée /work/<id>/entree.nii.gz) et garde chaque étape :
# s1_recadre (recadrage sur le masque > 0,005), ref (grille de référence), s2_reference (rééchantillonnage linéaire), s3_recadre (second recadrage), cube128 et cube256 (`prepare-for-monai`).
# Les commandes MIRTK sont celles du script officiel (GPL v3, voir docs/BOUNTI.md) ; ce script ne sert qu'à produire des références de test.
# Rejoue, dans l'ordre du script officiel, le prétraitement MIRTK d'un volume (entrée /work/<id>/entree.nii.gz) et garde chaque étape.
set -e
M=/bin/MIRTK/build/bin/mirtk
T=/home/auto-proc-svrtk/templates/brain-ref-space.nii.gz
cd /work/$1
cp entree.nii.gz s0.nii.gz
$M threshold-image s0.nii.gz th.nii.gz 0.005 > /dev/null
$M crop-image s0.nii.gz th.nii.gz s1_recadre.nii.gz
$M edit-image $T ref.nii.gz -copy-origin s1_recadre.nii.gz
$M transform-image s1_recadre.nii.gz s2_reference.nii.gz -target ref.nii.gz -interp Linear
$M crop-image s2_reference.nii.gz th.nii.gz s3_recadre.nii.gz
$M nan s3_recadre.nii.gz 1000000
for res in 128 256; do
  $M prepare-for-monai cube$res org$res info$res.json info$res.csv $res 1 s3_recadre.nii.gz > /dev/null
  cp cube$res/in-res-stack-1000.nii.gz cube$res.nii.gz
done
ls -la
