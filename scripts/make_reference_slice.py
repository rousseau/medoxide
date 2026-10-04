"""Produit une tuile de référence pour valider l'inférence Rust (étape 4b).

Reproduit le prétraitement de inference.py (Fetal-BET) : chargement,
rééchantillonnage à 1 mm dans le plan (Spacingd, bilinéaire, bord à zéro),
puis division par l'écart-type de la coupe (voxels > 0 seulement, sans
soustraire la moyenne : subtrahend=0, divisor=None dans l'original). On prend la coupe contenant le plus
de masque (d'après le masque de référence dans data/derivatives/), on en
extrait la tuile 256×256 d'origine (0, 0) et on la passe dans
models/attunet.onnx avec onnxruntime.

Sans ce prétraitement exact (1 mm, pas de soustraction de la moyenne), le modèle
ne détecte rien : voir docs/LEARNING.md, étape 4b.

Écrit en float32 little-endian brut (lisible en Rust sans crate) :
  data/reference/slice_input.f32   [1, 1, 256, 256]
  data/reference/slice_logits.f32  [1, 2, 256, 256]

Usage : python scripts/make_reference_slice.py [volume.nii.gz]
"""

import os
import sys

import monai.transforms as tr
import nibabel as nib
import numpy as np
import onnxruntime as ort

volume = sys.argv[1] if len(sys.argv) > 1 else "data/sourcedata/fetus_03.nii.gz"
masque_ref = volume.replace("sourcedata", "derivatives/fetal-bet").replace(
    ".nii.gz", "_predicted_mask.nii.gz"
)

# Même enchaînement que FetalTestData.transformations() (sans la normalisation).
vol = tr.Compose(
    [
        tr.LoadImaged(keys=["image"]),
        tr.EnsureChannelFirstd(keys=["image"]),
        tr.Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear", padding_mode="zeros"),
    ]
)({"image": volume})["image"]
vol = np.asarray(vol[0], dtype=np.float32)  # [X, Y, Z] à 1 mm dans le plan

# Coupe la plus riche en cerveau, repérée sur la grille d'origine (même Z).
z = int(nib.load(masque_ref).get_fdata().sum(axis=(0, 1)).argmax())
sl = vol[:, :, z].copy()
m = sl > 0
sl[m] = sl[m] / sl[m].std()  # ÷ écart-type, voxels > 0, SANS retirer la moyenne
x = sl[None, None, :256, :256].astype("<f4")

sess = ort.InferenceSession("models/attunet.onnx", providers=["CPUExecutionProvider"])
logits = sess.run(None, {"image": x})[0].astype("<f4")

os.makedirs("data/reference", exist_ok=True)
x.tofile("data/reference/slice_input.f32")
logits.tofile("data/reference/slice_logits.f32")
n_masque = int((logits.argmax(1) == 1).sum())
print(f"{volume}, coupe {z}, grille 1 mm {vol.shape[:2]} : tuile {x.shape}, logits {logits.shape}")
print(f"|logit| max = {np.abs(logits).max():.2f}, voxels masque = {n_masque}")
assert n_masque > 0, "tuile sans masque : inutile pour valider l'argmax"
