"""Écrit les volumes de référence pour valider le code Rust (étape 5).

Pour chaque volume de data/sourcedata/, écrit en float32 little-endian brut,
ordre C sur les axes [X, Y, Z] (donc lisible tel quel avec `Array3::iter()`) :
  data/reference/<nom>_raw.f32       voxels lus par nibabel (get_fdata, float32)
  data/reference/<nom>_raw_norm.f32  idem, puis normalisation par coupe de
                                     Fetal-BET (sans rééchantillonnage)

(La sortie rééchantillonnée sera ajoutée à l'étape 5c.)

Usage : python scripts/make_reference_volumes.py
"""

import glob
import os

import monai.transforms as tr
import nibabel as nib
import numpy as np
from monai.transforms import MapTransform


class SliceWiseNormalizeIntensityd(MapTransform):
    """Copie du code de Fetal-BET (src/codes/inference.py), cas utilisé :
    subtrahend=0.0, divisor=None, nonzero=True. Par coupe z : les voxels > 0
    sont divisés par leur écart-type (torch.std : non biaisé, ddof=1) ; la
    moyenne n'est PAS soustraite."""

    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for i in range(image.shape[-1]):
                slice_ = image[..., i]
                mask = slice_ > 0
                if np.any(mask):
                    slice_[mask] = slice_[mask] - self.subtrahend
                    slice_[mask] /= slice_[mask].std()
                image[..., i] = slice_
            d[key] = image
        return d


def ecrire(nom, suffixe, tableau):
    np.ascontiguousarray(tableau, dtype="<f4").tofile(f"data/reference/{nom}_{suffixe}.f32")


os.makedirs("data/reference", exist_ok=True)
for f in sorted(glob.glob("data/sourcedata/*.nii.gz")):
    nom = os.path.basename(f).removesuffix(".nii.gz")
    brut = nib.load(f).get_fdata(dtype=np.float32)
    ecrire(nom, "raw", brut)

    norm = tr.Compose(
        [
            tr.LoadImaged(keys=["image"]),
            tr.EnsureChannelFirstd(keys=["image"]),
            SliceWiseNormalizeIntensityd(keys=["image"]),
        ]
    )({"image": f})["image"]
    ecrire(nom, "raw_norm", np.asarray(norm[0]))
    print(nom, brut.shape, f"brut max {brut.max():.0f} ; normalisé max {float(norm.max()):.2f}")
