"""Références pour l'inférence par tuiles (étape 6b).

Pour chaque volume, prend le volume prétraité par MONAI
(data/reference/<nom>_prep.f32, voir make_reference_volumes.py), lance le vrai
SliceInferer de Fetal-BET (tuiles 256×256, recouvrement 0,5, logits moyennés)
avec models/AttUNet.pth, et écrit, sur la grille à 1 mm :
  data/reference/<nom>_mask1mm.u8       masque (argmax des logits), octets 0/1,
                                        ordre C sur [X, Y, Z]
  data/reference/fetus_03_logits1mm.f32 logits [2, X, Y, Z] (fetus_03 seulement)

Usage : python scripts/make_reference_masks.py
"""

import glob
import os

import monai.transforms as tr
import numpy as np
import torch
from monai.inferers import SliceInferer
from monai.networks.nets import AttentionUnet

model = AttentionUnet(
    spatial_dims=2, in_channels=1, out_channels=2, channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3, dropout=0.15,
)
sd = torch.load("models/AttUNet.pth", map_location="cpu", weights_only=True)
model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
model.eval()

inferer = SliceInferer(roi_size=(256, 256), spatial_dim=2, sw_batch_size=4, overlap=0.50, progress=False)

for f in sorted(glob.glob("data/sourcedata/*.nii.gz")):
    nom = os.path.basename(f).removesuffix(".nii.gz")
    # Forme du volume à 1 mm : on la relit en rejouant le rééchantillonnage MONAI.
    forme = tr.Compose(
        [
            tr.LoadImaged(keys=["image"]),
            tr.EnsureChannelFirstd(keys=["image"]),
            tr.Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear", padding_mode="zeros"),
        ]
    )({"image": f})["image"].shape[1:]
    prep = np.fromfile(f"data/reference/{nom}_prep.f32", dtype="<f4").reshape(forme)
    x = torch.from_numpy(prep)[None, None]  # [1, 1, X, Y, Z]
    with torch.no_grad():
        logits = inferer(x, model)  # [1, 2, X, Y, Z]
    masque = logits.argmax(1)[0].numpy().astype(np.uint8)
    np.ascontiguousarray(masque).tofile(f"data/reference/{nom}_mask1mm.u8")
    if nom == "fetus_03":
        np.ascontiguousarray(logits[0].numpy(), dtype="<f4").tofile("data/reference/fetus_03_logits1mm.f32")
    print(nom, tuple(forme), "voxels masque", int(masque.sum()))
