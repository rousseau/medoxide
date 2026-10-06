"""Références de coordonnées monde par coupe pour tester le crate medoxide-svr (étape 1b).

Pour chaque stack (volumes de Fetal-BET dans data/sourcedata/ et jeu de développement BIDS dans
data/svr/jeu_reel_tru_haste/), choisit des points en indices de voxel (i, j, k) :
  - les 4 coins et le centre ((nx-1)/2, (ny-1)/2) de chaque coupe k ;
  - 2000 points aléatoires continus dans le volume (graine fixe).
Calcule leurs coordonnées monde RAS+ (mm) de deux façons indépendantes :
  - nibabel : nib.affines.apply_affine(img.affine, p) ;
  - SimpleITK : TransformContinuousIndexToPhysicalPoint (repère LPS), converti en RAS par (-x, -y, z).
Écrit data/reference/slicecoords/<nom>.f64 (lignes de 9 f64 little-endian : i j k, x y z nibabel,
x y z SimpleITK-RAS) et data/reference/slicecoords/index.tsv (chemin du stack, fichier de référence).
Affiche l'écart maximal entre les deux outils.

Usage : python scripts/make_reference_slice_coords.py
"""

import glob
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk

SORTIE = "data/reference/slicecoords"
stacks = sorted(glob.glob("data/sourcedata/*.nii.gz")) + sorted(
    glob.glob("data/svr/jeu_reel_tru_haste/sub-*/ses-*/anat/*_T2w.nii.gz")
)
os.makedirs(SORTIE, exist_ok=True)
rng = np.random.default_rng(12345)
index = []
pire = 0.0
for f in stacks:
    img = nib.load(f)
    A = img.affine
    nx, ny, nz = img.shape[:3]
    pts = []
    for k in range(nz):
        for i, j in ((0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1), ((nx - 1) / 2, (ny - 1) / 2)):
            pts.append((i, j, k))
    for _ in range(2000):
        pts.append((rng.uniform(0, nx - 1), rng.uniform(0, ny - 1), rng.uniform(0, nz - 1)))
    P = np.array(pts, dtype=float)
    w_nib = nib.affines.apply_affine(A, P)
    s = sitk.ReadImage(f)
    w_itk = np.array([s.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in p)) for p in P])
    w_itk_ras = w_itk * np.array([-1.0, -1.0, 1.0])
    pire = max(pire, float(np.abs(w_nib - w_itk_ras).max()))
    nom = os.path.basename(f).removesuffix(".nii.gz")
    np.hstack([P, w_nib, w_itk_ras]).astype("<f8").tofile(f"{SORTIE}/{nom}.f64")
    index.append(f"{f}\t{SORTIE}/{nom}.f64")
open(f"{SORTIE}/index.tsv", "w").write("\n".join(index) + "\n")
print(f"{len(stacks)} stacks, {len(pts)} points chacun ; écart max nibabel / SimpleITK-RAS : {pire:.3e} mm")
