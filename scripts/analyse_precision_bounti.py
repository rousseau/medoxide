"""Situe les écarts de logits entre implémentations f32 d'un réseau BOUNTI par rapport à une référence en double précision (float64).

Pourquoi : deux implémentations f32 correctes d'une même convolution 3D diffèrent par le bruit d'arrondi, qui dépend de l'ordre des sommes. Comparer Burn à PyTorch f32 mélange l'erreur de PyTorch et celle de Burn. Contre une référence float64, on voit laquelle est la plus proche de la vérité.
Écart relatif = max |Δ| / max |logit|. Entrées : `data/bounti/reference/<réseau>_<n>_input.f32`, logits PyTorch f32 `_logits.f32`, logits Burn `_burn_gpu.f32` / `_burn_flex.f32` (écrits par `MEDOXIDE_DUMP=1 cargo test -p medoxide-bounti -- --ignored`), export `data/bounti/onnx/<réseau>_<n>.onnx`.

Usage : python scripts/analyse_precision_bounti.py <unet|attunet> [n] [tag]      (tag : `_reel` pour les références sur volume réel)
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from monai.networks.nets import AttentionUnet, UNet

DONNEES = "data/bounti"
reseau = sys.argv[1] if len(sys.argv) > 1 else "unet"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
tag = sys.argv[3] if len(sys.argv) > 3 else ""
if reseau == "unet":
    modele = UNet(spatial_dims=3, in_channels=1, out_channels=20, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3,
                  num_res_units=1, act="PRELU", norm="INSTANCE", dropout=0.5)
    dossier = "monai-checkpoints-unet-brain-bounti-19-lab"
else:
    modele = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=20, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3, dropout=0.5)
    dossier = "monai-checkpoints-red-attunet-brain_bounti-19-lab"
modele.load_state_dict(torch.load(f"{DONNEES}/models/{dossier}/best_metric_model.pth", map_location="cpu", weights_only=True), strict=True)
modele.eval()


def lire(nom):
    return np.fromfile(f"{DONNEES}/reference/{reseau}_{n}{tag}_{nom}.f32", dtype="<f4")


entree = lire("input").reshape(1, 1, n, n, n)
with torch.no_grad():
    vrai = modele.double()(torch.from_numpy(entree.astype("float64"))).numpy().reshape(-1)  # référence float64
amplitude = np.abs(vrai).max()
voxels = n ** 3
etiquettes = lambda l: l.reshape(20, voxels).argmax(0)
vraies = etiquettes(vrai)

candidats = {"PyTorch f32": lire("logits")}
session = ort.InferenceSession(f"{DONNEES}/onnx/{reseau}_{n}.onnx", providers=["CPUExecutionProvider"])
if n == 64:  # l'export ONNX est tracé à 64³ : onnxruntime n'est comparé que pour cette taille
    candidats["onnxruntime f32"] = session.run(None, {"image": entree})[0].reshape(-1)
for suffixe, nom in [("gpu", "Burn GPU f32"), ("flex", "Burn CPU (flex) f32")]:
    chemin = f"{DONNEES}/reference/{reseau}_{n}{tag}_burn_{suffixe}.f32"
    if os.path.exists(chemin):
        candidats[nom] = np.fromfile(chemin, dtype="<f4")

print(f"{reseau} n={n}{tag}, |logit| max {amplitude:.1f} ; écart à la référence float64")
print(f"{'implémentation':22s} {'écart relatif':>14s} {'écart absolu':>13s} {'accord des étiquettes':>22s}")
for nom, logits in candidats.items():
    ecart = np.abs(logits.astype("float64") - vrai).max()
    print(f"{nom:22s} {ecart / amplitude:14.2e} {ecart:13.2e} {(etiquettes(logits) == vraies).mean():22.6f}")
