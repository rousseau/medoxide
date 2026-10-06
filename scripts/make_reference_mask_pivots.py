"""Références du pivot P3 pour tester le crate medoxide-svr (étape 1d).

Pour chaque stack du jeu de développement et son masque cérébral (dérivés `medx-fetalbet`) :
  - nettoie le masque : plus grande composante connexe 3D (scipy.ndimage.label, 26 voisins) ;
  - barycentre 3D du masque nettoyé, en indices de voxel (bi, bj, bk) ;
  - pivot P3 de chaque coupe k : position monde de (bi, bj, k), soit A @ (bi, bj, k, 1).
Écrit deux fichiers TSV dans data/reference/pivots/ (flottants en repr Python, relus sans perte) :
  masques.tsv : stack, masque, voxels du masque brut, voxels conservés, bi, bj, bk
  pivots.tsv  : stack, k, x, y, z

Usage : python scripts/make_reference_mask_pivots.py
"""

import glob
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

RACINE = "data/svr/jeu_reel_tru_haste"
SORTIE = "data/reference/pivots"
os.makedirs(SORTIE, exist_ok=True)
stacks = sorted(glob.glob(f"{RACINE}/sub-*/ses-*/anat/*_T2w.nii.gz"))
masques, pivots = [], []
for s in stacks:
    sub, ses = s.replace(RACINE + "/", "").split("/")[:2]
    m_path = f"{RACINE}/derivatives/medx-fetalbet/{sub}/{ses}/anat/" + os.path.basename(s).replace("_T2w.nii.gz", "_desc-brain_mask.nii.gz")
    A = nib.load(s).affine
    m = np.asarray(nib.load(m_path).dataobj) > 0
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    tailles = np.bincount(lab.ravel())[1:]
    garde = lab == (tailles.argmax() + 1)
    bi, bj, bk = (float(x) for x in np.argwhere(garde).mean(axis=0))
    masques.append("\t".join([s, m_path, str(int(m.sum())), str(int(garde.sum())), repr(bi), repr(bj), repr(bk)]))
    for k in range(m.shape[2]):
        x, y, z = (float(v) for v in (A @ np.array([bi, bj, k, 1.0]))[:3])
        pivots.append("\t".join([s, str(k), repr(x), repr(y), repr(z)]))
open(f"{SORTIE}/masques.tsv", "w").write("\n".join(masques) + "\n")
open(f"{SORTIE}/pivots.tsv", "w").write("\n".join(pivots) + "\n")
print(f"{len(stacks)} stacks, {len(pivots)} pivots de coupe")
