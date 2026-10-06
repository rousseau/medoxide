"""Références de l'échantillonnage trilinéaire d'un volume en des points du monde (étape 2a du SVR).

Pour quelques paires de stacks d'un même sujet (le volume = un stack axial TRUFI, les points = les pixels de
coupes d'un stack coronal TRUFI), calcule la valeur du volume aux positions monde des pixels de ces coupes de deux
façons indépendantes :
  - scipy.ndimage.map_coordinates, order=1, avec les indices continus donnés par l'inverse de l'affine nibabel ;
  - SimpleITK Image.EvaluateAtPhysicalPoint (interpolation linéaire, repère LPS : (x, y, z) RAS devient (-x, -y, z) ;
    sans valeur par défaut : la fonction lève une exception hors de la région, convertie en -1e9).
Écrit data/reference/sampling/<nom>.f32 (par point, 3 flottants : dans le domaine [0, n-1] ? (1 ou 0), valeur scipy,
valeur SimpleITK ou -1e9) et data/reference/sampling/index.tsv (volume, stack des coupes, coupes, pas des pixels).
Points : coupes k = 0, 6, 12, ... ; pixels (i, j) tous les 3 voxels, i en premier (ordre i puis j).

Usage : python scripts/make_reference_sampling.py
"""

import collections
import glob
import os
import re

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

RACINE = "data/svr/jeu_reel_tru_haste"
SORTIE = "data/reference/sampling"
PAS_COUPE, PAS_PIXEL, N_SUJETS = 6, 3, 4
os.makedirs(SORTIE, exist_ok=True)


def valeur_itk(image, point_lps):
    """Valeur linéaire de SimpleITK au point ; -1e9 hors de la région définie (la fonction lève alors une exception)."""
    try:
        return image.EvaluateAtPhysicalPoint(point_lps, sitk.sitkLinear)
    except RuntimeError:
        return -1e9


par_sujet = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{RACINE}/sub-*/ses-*/anat/*_T2w.nii.gz")):
    m = re.search(r"(sub-\d+)_ses-\d+_acq-(trufiax|truficor)_run-1_T2w", f)
    if m:
        par_sujet[m.group(1)][m.group(2)] = f
paires = [(v["trufiax"], v["truficor"]) for _, v in sorted(par_sujet.items()) if len(v) == 2][:N_SUJETS]
index = []
for vol_path, coupes_path in paires:
    vol = nib.load(vol_path)
    A_v = vol.affine
    inv = np.linalg.inv(A_v)
    data = vol.get_fdata(dtype=np.float32)
    itk_vol = sitk.Cast(sitk.ReadImage(vol_path), sitk.sitkFloat32)
    cou = nib.load(coupes_path)
    A_c = cou.affine
    nx, ny, nz = cou.shape[:3]
    ks = list(range(0, nz, PAS_COUPE))
    lignes = []
    for k in ks:
        ii, jj = np.meshgrid(np.arange(0, nx, PAS_PIXEL), np.arange(0, ny, PAS_PIXEL), indexing="ij")
        pts = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k), np.ones(ii.size)], 0)
        monde = (A_c @ pts)[:3].T                                   # RAS+, mm
        c = (inv @ np.vstack([monde.T, np.ones(len(monde))]))[:3].T  # indices continus dans le volume
        dedans = np.all((c >= 0) & (c <= np.array(data.shape) - 1), axis=1)
        sp = ndimage.map_coordinates(data, c.T, order=1, mode="nearest")
        itk = np.array([valeur_itk(itk_vol, (-x, -y, z)) for x, y, z in monde])
        lignes.append(np.stack([dedans.astype(np.float32), sp.astype(np.float32), itk.astype(np.float32)], 1))
    nom = os.path.basename(coupes_path).removesuffix("_T2w.nii.gz")
    np.concatenate(lignes).astype("<f4").tofile(f"{SORTIE}/{nom}.f32")
    index.append("\t".join([vol_path, coupes_path, ",".join(map(str, ks)), str(PAS_PIXEL), f"{SORTIE}/{nom}.f32"]))
open(f"{SORTIE}/index.tsv", "w").write("\n".join(index) + "\n")
print(f"{len(paires)} paires ; points par paire : ~{len(lignes[0]) * len(ks)}")
