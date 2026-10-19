#!/bin/bash
# Exécuté DANS l'image Docker de BOUNTI (après `bounti_mirtk_preparer.sh`) : rejoue l'extraction du cerveau de la chaîne officielle sur le cube de 128 et son nettoyage, et garde chaque étape :
# bet_etiquettes_cube (sortie du réseau officiel, étiquettes 0/1 sur le cube), bet_composante (étiquette 1 puis plus grande composante connexe), bet_s3 (retour dans l'espace du volume recadré, interpolation d'étiquettes),
# bet_dilate (dilatation de 4), bet_erode (érosion de 2), masque (image masquée), masque_recadre (recadrage sur le masque), cube256_masque (`prepare-for-monai` à 256 sur l'image masquée recadrée).
# Les commandes sont celles du script officiel `auto-brain-bounti-segmentation-fetal.sh` (GPL v3, voir docs/BOUNTI.md) ; ce script ne sert qu'à produire des références de test.
set -e
M=/bin/MIRTK/build/bin/mirtk
cd /work/$1
mkdir -p bet_resultats
python3 /home/auto-proc-svrtk/src/run_monai_atunet_segmentation-2022.py /work/$1/ /home/auto-proc-svrtk/trained_models/monai-checkpoints-attunet-brain-bet-1-lab/ info128.json /work/$1/bet_resultats 128 1 > bet_python.log 2>&1
cp bet_resultats/cnn-lab-in-res-stack-1000.nii.gz bet_etiquettes_cube.nii.gz
$M extract-label bet_etiquettes_cube.nii.gz bet_composante.nii.gz 1 1
$M extract-connected-components bet_composante.nii.gz bet_composante.nii.gz
$M transform-image bet_composante.nii.gz bet_s3.nii.gz -target s3_recadre.nii.gz -labels
$M dilate-image bet_s3.nii.gz bet_dilate.nii.gz -iterations 4
$M erode-image bet_dilate.nii.gz bet_erode.nii.gz -iterations 2
$M mask-image s3_recadre.nii.gz bet_erode.nii.gz masque.nii.gz
$M crop-image masque.nii.gz bet_erode.nii.gz masque_recadre.nii.gz
$M prepare-for-monai cube256m org256m info256m.json info256m.csv 256 1 masque_recadre.nii.gz > /dev/null
cp cube256m/in-res-stack-1000.nii.gz cube256_masque.nii.gz
ls -la bet_*.nii.gz masque*.nii.gz cube256_masque.nii.gz
