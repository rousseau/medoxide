"""Prépare les atlas T2w CRL2025 (Garage : derivatives/crl2025-t2w-atlas/GA<ss>/t2w_GA<ss>_atlas.nii.gz) pour le banc semi-synthétique de medoxide-svr (aide `stacks_atlas_sans_bruit`).

Cette aide attend un atlas 3D centré sur l'origine du repère (le barycentre du tissu en 0) et des intensités dans l'échelle des atlas de Gholipour (tissu > 150). Les atlas CRL2025 sont en 4D, origine quelconque, intensités de 0 à 100 : on recentre sur le barycentre et on multiplie par 10. Fichier écrit : `<sortie>/CRL<ss>.nii.gz` (le nom de l'atlas pour `MEDOXIDE_ATLAS=CRL<ss>`).

Usage : python scripts/preparer_atlas_crl.py --entree data/atlas/crl2025 --sortie data/atlas/gholipour --ages 23 26 29 32 35 37
(`--entree` contient GA<ss>.nii.gz, rapatriés par rclone ; données publiques, mais hors Git comme toutes les données.)
"""

import argparse

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--entree", required=True)
parser.add_argument("--sortie", required=True)
parser.add_argument("--ages", nargs="+", type=int, required=True)
args = parser.parse_args()
for ga in args.ages:
    im = nib.load(f"{args.entree}/GA{ga}.nii.gz")
    d = np.asarray(im.dataobj, dtype=np.float32).reshape(im.shape[:3])
    barycentre_voxel = np.argwhere(d > 0).mean(axis=0)
    affine = im.affine.copy()
    affine[:3, 3] = -affine[:3, :3] @ barycentre_voxel  # le barycentre du tissu tombe en (0, 0, 0)
    sortie = f"{args.sortie}/CRL{ga}.nii.gz"
    nib.save(nib.Nifti1Image(10.0 * d, affine), sortie)
    print(sortie, d.shape, "intensité max", 10.0 * d.max())
