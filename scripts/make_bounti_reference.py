"""Écrit une entrée de test et les logits PyTorch/MONAI correspondants pour valider le portage Rust d'un réseau 3D de BOUNTI (critère C1 de docs/BOUNTI.md).

Les poids (GPL v3) ne sont pas dans ce dépôt : ils viennent de `data/bounti/models/` (voir `export_bounti_onnx.py`). Les deux fichiers écrits sont des tableaux bruts `float32` petit-boutiste, en ordre C :
  - entrée  [1, 1, n, n, n], uniforme dans [0, 1] (graine fixe) ;
  - logits  [1, 20, n, n, n].

Usage   : python scripts/make_bounti_reference.py <unet|attunet> [n]      (n : 64 par défaut)
Sortie  : data/bounti/reference/<réseau>_<n>_input.f32 et <réseau>_<n>_logits.f32
"""

import os
import sys

import numpy as np
import torch
from monai.networks.nets import AttentionUnet, UNet

MODELES = "data/bounti/models"
SORTIE = "data/bounti/reference"
SORTIES = 20  # 19 étiquettes + fond

reseau = sys.argv[1] if len(sys.argv) > 1 else "unet"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
if reseau == "unet":
    modele = UNet(
        spatial_dims=3, in_channels=1, out_channels=SORTIES, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
        kernel_size=3, up_kernel_size=3, num_res_units=1, act="PRELU", norm="INSTANCE", dropout=0.5,
    )
    dossier = "monai-checkpoints-unet-brain-bounti-19-lab"
elif reseau == "attunet":
    modele = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=SORTIES, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        kernel_size=3, up_kernel_size=3, dropout=0.5,
    )
    dossier = "monai-checkpoints-red-attunet-brain_bounti-19-lab"
else:
    raise SystemExit(f"réseau inconnu : {reseau} (unet ou attunet)")
modele.load_state_dict(torch.load(f"{MODELES}/{dossier}/best_metric_model.pth", map_location="cpu", weights_only=True), strict=True)
modele.eval()

torch.manual_seed(1)  # graine différente de celle de l'export : l'entrée n'est pas celle qui a servi à tracer le graphe
x = torch.rand(1, 1, n, n, n)
with torch.no_grad():
    logits = modele(x).numpy().astype("<f4")
os.makedirs(SORTIE, exist_ok=True)
x.numpy().astype("<f4").tofile(f"{SORTIE}/{reseau}_{n}_input.f32")
logits.tofile(f"{SORTIE}/{reseau}_{n}_logits.f32")
print(f"{reseau} n={n} : logits {logits.shape}, |logit| max {np.abs(logits).max():.1f}, écrits dans {SORTIE}/")
