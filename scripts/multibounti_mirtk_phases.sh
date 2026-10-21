#!/bin/bash
# Exécuté DANS l'image Docker ARM de BOUNTI (`fetalsvrtk/svrtk:general_auto_arm`, qui a les outils MIRTK géométriques) : rejoue, phase par phase, les commandes MIRTK du script officiel
# `run-multi-bounti-fetal-brain-segmentation-2026-general.sh` (GPL v3, King's College London, voir docs/MULTIBOUNTI.md), sans correction de biais N4 (premier argument du script officiel : 0).
# Les réseaux (python) tournent entre les phases, sur le Mac (voir `multibounti_reference.py`). Ce script ne sert qu'à produire des références de test.
#
# Usage : multibounti_mirtk_phases.sh <phase> ; dossier de travail monté en /work (fichier d'entrée : /work/entree.nii.gz), gabarits montés en /templates
#   A : prétraitement jusqu'au `pad-3d` à 128         (avant l'extraction du cerveau)
#   B : nettoyage du masque, image masquée, `pad-3d`   (avant le réseau gauche/droite)
#   C : recalage affine, réorientation, `pad-3d` à 256 (avant le réseau 43 étiquettes ; GUESS_GA = âge estimé, calculé hors de ce script)
#   D : retour dans la grille d'origine                (avant `thin-cortex-thick`)
set -e
M=/bin/MIRTK/build/lib/tools
T=/templates/ref-bounti
P=/work
cd $P
case "$1" in
A)
  $M/convert-image entree.nii.gz tmp-z.nii.gz
  $M/get-dz tmp-z.nii.gz > dz.txt
  rm tmp-z.nii.gz
  $M/convert-image entree.nii.gz org-t2.nii.gz
  $M/extract-image-region org-t2.nii.gz org-t2.nii.gz -Rt1 0 -Rt2 0
  $M/nan org-t2.nii.gz 100000
  $M/threshold-image org-t2.nii.gz m-t2.nii.gz 0.5 > t.txt
  $M/extract-connected-components m-t2.nii.gz m-t2.nii.gz
  $M/crop-image org-t2.nii.gz m-t2.nii.gz crop-org-t2.nii.gz
  $M/edit-image $T/ref-082025.nii.gz tmp-ref.nii.gz -copy-origin crop-org-t2.nii.gz
  $M/transform-image org-t2.nii.gz tr-org-t2.nii.gz -target tmp-ref.nii.gz -interp Linear
  $M/nan tr-org-t2.nii.gz 100000
  $M/crop-image tr-org-t2.nii.gz m-t2.nii.gz crop-tr-t2.nii.gz
  $M/pad-3d crop-tr-t2.nii.gz pad-crop-tr-t2-128.nii.gz 128 1
  ;;
B)
  $M/erode-image bet-lab-pad-crop-t2-128.nii.gz bet-lab-pad-crop-t2-128.nii.gz
  $M/extract-connected-components bet-lab-pad-crop-t2-128.nii.gz bet-lab-pad-crop-t2-128.nii.gz
  $M/dilate-image bet-lab-pad-crop-t2-128.nii.gz bet-lab-pad-crop-t2-128.nii.gz
  $M/dilate-image bet-lab-pad-crop-t2-128.nii.gz dl-bet-lab-pad-crop-t2-128.nii.gz -iterations 2
  $M/transform-image bet-lab-pad-crop-t2-128.nii.gz bet-lab-crop-t2.nii.gz -labels -target crop-tr-t2.nii.gz
  cp crop-tr-t2.nii.gz n4-crop-tr-t2.nii.gz
  $M/mask-image n4-crop-tr-t2.nii.gz dl-bet-lab-pad-crop-t2-128.nii.gz masked-n4-crop-tr-t2.nii.gz
  $M/crop-image masked-n4-crop-tr-t2.nii.gz dl-bet-lab-pad-crop-t2-128.nii.gz masked-n4-crop-tr-t2.nii.gz
  $M/pad-3d masked-n4-crop-tr-t2.nii.gz pad-masked-n4-crop-tr-t2-128.nii.gz 128 1
  ;;
C)
  $M/edit-image $T/dl-ref-t2.nii.gz tmp-dl-ref.nii.gz -copy-origin masked-n4-crop-tr-t2.nii.gz
  $M/edit-image $T/lr-lab.nii.gz tmp-lr-ref.nii.gz -copy-origin masked-n4-crop-tr-t2.nii.gz
  $M/mask-image lr-lab-pad-masked-crop-t2-128.nii.gz bet-lab-pad-crop-t2-128.nii.gz lr-lab-pad-masked-crop-t2-128.nii.gz
  if [ "$GUESS_GA" -gt 170 ]; then cp $T/i.dof aff-i.dof; else cp $T/i-very-early-ga.dof aff-i.dof; fi
  $M/register tmp-lr-ref.nii.gz lr-lab-pad-masked-crop-t2-128.nii.gz -model Affine -dofin aff-i.dof -dofout aff-d.dof -v 0
  $M/transform-image bet-lab-pad-crop-t2-128.nii.gz dl-bet-org-t2.nii.gz -target crop-org-t2.nii.gz -labels
  $M/dilate-image dl-bet-org-t2.nii.gz dl-bet-org-t2.nii.gz -iterations 2
  $M/mask-image crop-org-t2.nii.gz dl-bet-org-t2.nii.gz masked-crop-org-t2.nii.gz
  $M/crop-image masked-crop-org-t2.nii.gz dl-bet-org-t2.nii.gz masked-crop-org-t2.nii.gz
  $M/transform-image dl-bet-org-t2.nii.gz dl-bet-org-t2.nii.gz -target masked-crop-org-t2.nii.gz -labels
  cp masked-crop-org-t2.nii.gz n4-masked-crop-org-tr-t2.nii.gz
  $M/transform-image n4-masked-crop-org-tr-t2.nii.gz reo-n4-masked-t2.nii.gz -target tmp-dl-ref.nii.gz -dofin aff-d.dof
  $M/threshold-image reo-n4-masked-t2.nii.gz m.nii.gz 0.5 > t.txt
  $M/crop-image reo-n4-masked-t2.nii.gz m.nii.gz reo-n4-masked-t2.nii.gz
  $M/nan reo-n4-masked-t2.nii.gz 50000
  $M/pad-3d reo-n4-masked-t2.nii.gz pad-reo-n4-masked-t2-256.nii.gz 256 1
  ;;
D)
  $M/transform-image multi-lab-reo-t2.nii.gz tr-multi-lab-reo-t2.nii.gz -target org-t2.nii.gz -dofin_i aff-d.dof -labels
  ;;
*) echo "phase inconnue : $1" >&2; exit 1 ;;
esac
