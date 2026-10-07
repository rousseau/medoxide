"""Référence de la grille de reconstruction pour tester le crate medoxide-svr (étape 4a).

Pour les stacks TRUFI (trufiax, truficor, trufisag, run-1) de quelques sujets du jeu de développement et leurs masques cérébraux (dérivés
`medx-fetalbet`), calcule de façon indépendante (nibabel, scipy) la grille de reconstruction :
  - masque nettoyé : plus grande composante connexe 3D (scipy.ndimage.label, 26 voisins) ;
  - position monde (RAS+, mm) de chaque voxel du masque nettoyé de chaque stack ;
  - boîte englobante de l'union, élargie de MARGE mm de chaque côté ;
  - grille isotrope de RESOLUTION mm alignée sur les axes du monde, de coin minimal `bas` (centre du voxel 0,0,0), et n = ceil((haut - bas) / RESOLUTION) + 1 voxels par axe.
Écrit data/reference/grid/grilles.tsv : résolution, marge, sujet, bas_x, bas_y, bas_z, nx, ny, nz, stacks (séparés par des virgules).

Usage : python scripts/make_reference_grid.py
"""

import glob
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

RACINE = "data/svr/jeu_reel_tru_haste"
SORTIE = "data/reference/grid"
SUJETS = ["sub-S01", "sub-S02", "sub-S03"]
CONFIGS = [(0.8, 10.0), (0.5, 10.0), (1.0, 0.0)]  # (résolution, marge) en mm
os.makedirs(SORTIE, exist_ok=True)


def points_monde(stack):
    """Positions monde des voxels du masque nettoyé (plus grande composante, 26 voisins)."""
    sub, ses = stack.replace(RACINE + "/", "").split("/")[:2]
    masque = f"{RACINE}/derivatives/medx-fetalbet/{sub}/{ses}/anat/" + os.path.basename(stack).replace("_T2w.nii.gz", "_desc-brain_mask.nii.gz")
    m = np.asarray(nib.load(masque).dataobj) > 0
    lab, _ = ndimage.label(m, structure=np.ones((3, 3, 3)))
    tailles = np.bincount(lab.ravel())[1:]
    garde = lab == (tailles.argmax() + 1)
    idx = np.argwhere(garde)
    A = nib.load(stack).affine
    return (A @ np.c_[idx, np.ones(len(idx))].T)[:3].T


lignes = []
for sujet in SUJETS:
    stacks = [sorted(glob.glob(f"{RACINE}/{sujet}/ses-*/anat/*acq-{acq}_run-1_T2w.nii.gz"))[0] for acq in ("trufiax", "truficor", "trufisag")]
    tous = np.vstack([points_monde(s) for s in stacks])
    for resolution, marge in CONFIGS:
        bas, haut = tous.min(axis=0) - marge, tous.max(axis=0) + marge
        n = np.ceil((haut - bas) / resolution).astype(int) + 1
        lignes.append("\t".join([repr(resolution), repr(marge), sujet] + [repr(float(v)) for v in bas] + [str(int(v)) for v in n] + [",".join(stacks)]))
open(f"{SORTIE}/grilles.tsv", "w").write("\n".join(lignes) + "\n")
print(f"{len(lignes)} grilles écrites")
