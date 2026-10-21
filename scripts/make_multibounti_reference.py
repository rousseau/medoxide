"""Écrit une entrée réelle et les logits PyTorch/MONAI correspondants pour valider le portage Rust d'un réseau de Multi-BOUNTI (critère C1, voir docs/MULTIBOUNTI.md, sous-étape M1).

L'entrée est un fichier intermédiaire de la chaîne officielle rejouée par `multibounti_reference.py` (dossier `data/multibounti/ref/<id>/`), mise à l'échelle comme dans le script officiel (`ScaleIntensity(minv=0, maxv=1)` sur TOUT le volume) :
  - bet   : `pad-crop-tr-t2-128.nii.gz` (128³)
  - lr    : `pad-masked-n4-crop-tr-t2-128.nii.gz` (128³)
  - multi : fenêtre de 128³ de `pad-reo-n4-masked-t2-256.nii.gz` (256³), à partir du voxel (64, 64, 64) ; l'échelle est celle du volume entier de 256³, comme l'inférence par fenêtres de MONAI.
Les poids (GPL v3) viennent de `data/multibounti/image/` (voir `export_multibounti_onnx.py`). Les fichiers écrits sont des tableaux bruts `float32` petit-boutiste, en ordre C : entrée [1, 1, 128, 128, 128] et logits [1, sorties, 128, 128, 128].

Usage  : python scripts/make_multibounti_reference.py <bet|lr|multi> [id]      (id : calme1 par défaut)
Sortie : data/multibounti/reference/<réseau>_128_input.f32 et <réseau>_128_logits.f32
"""

import os
import sys

import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import AttentionUnet

MODELES = "data/multibounti/image/home/perinatal-brain-mri-analysis/models"
RESEAUX = {
    "bet": ("atunet_bet_brain_153t_svr_2026_128_best_metric_model.pth", 2, "pad-crop-tr-t2-128.nii.gz"),
    "lr": ("atunet_lr_brain_fetal_2lab_best_metric_model.pth", 3, "pad-masked-n4-crop-tr-t2-128.nii.gz"),
    "multi": ("patch_atunet_new_multi_43_brain_neo_fetal_aff_fix_022026_best_metric_model.pth", 44, "pad-reo-n4-masked-t2-256.nii.gz"),
}
reseau = sys.argv[1]
ident = sys.argv[2] if len(sys.argv) > 2 else "calme1"
fichier, sorties, entree = RESEAUX[reseau]
modele = AttentionUnet(
    spatial_dims=3, in_channels=1, out_channels=sorties, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    kernel_size=3, up_kernel_size=3, dropout=0.5,
)
modele.load_state_dict(torch.load(f"{MODELES}/{fichier}", map_location="cpu", weights_only=True), strict=True)
modele.eval()
volume = np.asarray(nib.load(f"data/multibounti/ref/{ident}/{entree}").dataobj, dtype=np.float32)
volume = (volume - volume.min()) / (volume.max() - volume.min())
if reseau == "multi":
    volume = volume[64:192, 64:192, 64:192]
assert volume.shape == (128, 128, 128), volume.shape
x = torch.from_numpy(np.ascontiguousarray(volume))[None, None]
with torch.no_grad():
    logits = modele(x).numpy().astype("<f4")
os.makedirs("data/multibounti/reference", exist_ok=True)
x.numpy().astype("<f4").tofile(f"data/multibounti/reference/{reseau}_128_input.f32")
logits.tofile(f"data/multibounti/reference/{reseau}_128_logits.f32")
etiquettes = np.bincount(logits.argmax(1).ravel(), minlength=sorties)
print(f"{reseau} ({ident}) : logits {logits.shape}, |logit| max {np.abs(logits).max():.1f}, voxels par étiquette {etiquettes.tolist()}")
