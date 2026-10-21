"""Prépare les entrées d'essai de Multi-BOUNTI (voir `docs/MULTIBOUNTI.md`) : la chaîne officielle exige une reconstruction à **0,5 mm** (isotrope), alors que les reconstructions de medoxide sont à 0,8 mm.

Chaque volume est rééchantillonné à 0,5 mm par interpolation linéaire, **sans changer l'orientation de la grille** (les colonnes de l'affine sont seulement mises à l'échelle 0,5 / pas d'origine) : les centres de voxel couvrent la même boîte physique (les bords des voxels restent à leur place).
Les valeurs sont réelles, la résolution est seulement interpolée : ces volumes servent à valider la chaîne numériquement, pas à juger la qualité de la segmentation.
`sform` et `qform` sont écrits cohérents (MIRTK lit le `qform`).

Usage : python scripts/multibounti_entrees.py id=volume.nii.gz [id=volume2.nii.gz …]      (sortie : data/multibounti/entrees/<id>.nii.gz)
"""

import os
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

SORTIE = "data/multibounti/entrees"
os.makedirs(SORTIE, exist_ok=True)
for argument in sys.argv[1:]:
    ident, chemin = argument.split("=", 1)
    image = nib.load(chemin)
    donnees = np.asarray(image.dataobj, dtype=np.float32)
    pas = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
    facteur = 0.5 / pas  # pas du nouveau voxel, en voxels de l'ancien volume (par axe)
    dimensions = np.floor(np.array(donnees.shape) / facteur).astype(int)
    # nouvel indice j -> ancien indice facteur * j + decalage ; le premier centre est à un demi-nouveau-voxel du bord de l'ancien
    decalage = -0.5 + facteur / 2
    sortie = affine_transform(donnees, np.diag(facteur), offset=decalage, output_shape=tuple(dimensions), order=1, mode="constant", cval=0.0)
    passage = np.eye(4)
    passage[:3, :3] = np.diag(facteur)
    passage[:3, 3] = decalage
    affine = image.affine @ passage
    nouveau = nib.Nifti1Image(sortie.astype(np.float32), affine)
    nouveau.set_sform(affine, code=1)
    nouveau.set_qform(affine, code=1)
    nib.save(nouveau, f"{SORTIE}/{ident}.nii.gz")
    print(f"{ident} : {donnees.shape} à {np.round(pas, 3)} mm -> {sortie.shape} à {np.round(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)), 4)} mm, axes {nib.aff2axcodes(affine)}")
