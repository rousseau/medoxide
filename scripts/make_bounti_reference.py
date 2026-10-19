"""Écrit une entrée de test et les logits PyTorch/MONAI correspondants pour valider le portage Rust d'un réseau 3D de BOUNTI (critère C1 de docs/BOUNTI.md).

Les poids (GPL v3) ne sont pas dans ce dépôt : ils viennent de `data/bounti/models/` (voir `export_bounti_onnx.py`). Les deux fichiers écrits sont des tableaux bruts `float32` petit-boutiste, en ordre C :
  - entrée  [1, 1, n, n, n], uniforme dans [0, 1] (graine fixe) ;
  - logits  [1, 20, n, n, n] (2 canaux pour le réseau `bet`).

Avec un volume NIfTI en troisième argument, l'entrée est ce volume rééchantillonné (trilinéaire) dans un cube de n³ puis mis à l'échelle min-max dans [0, 1] : test NUMÉRIQUE à statistiques d'activation réalistes, qui n'imite PAS le prétraitement officiel de BOUNTI. Les fichiers portent alors le suffixe `_reel`.

Usage   : python scripts/make_bounti_reference.py <unet|attunet|bet> [n] [volume.nii.gz]      (n : 64 par défaut)
Sortie  : data/bounti/reference/<réseau>_<n>[_reel]_input.f32 et <réseau>_<n>[_reel]_logits.f32
"""

import os
import sys

import numpy as np
import torch
from monai.networks.nets import AttentionUnet, UNet

MODELES = "data/bounti/models"
SORTIE = "data/bounti/reference"
SORTIES = 20  # 19 étiquettes + fond (2 sorties pour le réseau `bet` : fond, cerveau)

reseau = sys.argv[1] if len(sys.argv) > 1 else "unet"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
volume = sys.argv[3] if len(sys.argv) > 3 else None
suffixe = "_reel" if volume else ""
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
elif reseau == "bet":
    modele = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=2, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
        kernel_size=3, up_kernel_size=3, dropout=0.5,
    )
    dossier = "monai-checkpoints-attunet-brain-bet-1-lab"
else:
    raise SystemExit(f"réseau inconnu : {reseau} (unet, attunet ou bet)")
modele.load_state_dict(torch.load(f"{MODELES}/{dossier}/best_metric_model.pth", map_location="cpu", weights_only=True), strict=True)
modele.eval()

if volume:
    import nibabel as nib
    from scipy.ndimage import zoom

    donnees = np.clip(np.asarray(nib.load(volume).dataobj, dtype=np.float32), 0, None)
    donnees = zoom(donnees, [n / s for s in donnees.shape], order=1)
    donnees = (donnees - donnees.min()) / (donnees.max() - donnees.min())
    x = torch.from_numpy(donnees.astype(np.float32))[None, None]
else:
    torch.manual_seed(1)  # graine différente de celle de l'export : l'entrée n'est pas celle qui a servi à tracer le graphe
    x = torch.rand(1, 1, n, n, n)
with torch.no_grad():
    logits = modele(x).numpy().astype("<f4")
os.makedirs(SORTIE, exist_ok=True)
x.numpy().astype("<f4").tofile(f"{SORTIE}/{reseau}_{n}{suffixe}_input.f32")
logits.tofile(f"{SORTIE}/{reseau}_{n}{suffixe}_logits.f32")
print(f"{reseau} n={n}{suffixe} : logits {logits.shape}, |logit| max {np.abs(logits).max():.1f}, écrits dans {SORTIE}/")
