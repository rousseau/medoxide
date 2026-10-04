"""Écrit les volumes de référence pour valider le code Rust (étape 5).

Pour chaque volume de data/sourcedata/, écrit en float32 little-endian brut,
ordre C sur les axes [X, Y, Z] (donc lisible tel quel avec `Array3::iter()`) :
  data/reference/<nom>_raw.f32   voxels lus par nibabel (get_fdata, float32)

(Les sorties rééchantillonnée et normalisée seront ajoutées aux étapes 5b/5c.)

Usage : python scripts/make_reference_volumes.py
"""

import glob
import os

import nibabel as nib
import numpy as np

os.makedirs("data/reference", exist_ok=True)
for f in sorted(glob.glob("data/sourcedata/*.nii.gz")):
    nom = os.path.basename(f).removesuffix(".nii.gz")
    v = nib.load(f).get_fdata(dtype=np.float32)
    np.ascontiguousarray(v, dtype="<f4").tofile(f"data/reference/{nom}_raw.f32")
    print(nom, v.shape, f"min {v.min():.0f} max {v.max():.0f}")
