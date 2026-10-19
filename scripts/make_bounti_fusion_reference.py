"""Fusion de référence de BOUNTI (PyTorch/MONAI) pour valider la fusion Rust (critère C2 de docs/BOUNTI.md).

Pour l'entrée `data/bounti/reference/unet_<n>_reel_input.f32` (cube de n³, intensités dans [0, 1], voir `make_bounti_reference.py`), calcule les quatre sorties : U-Net et Attention U-Net (poids dans `data/bounti/models/`, hors dépôt), chacun sur l'image et sur son retournement de l'axe x (le premier axe spatial de `[1, 1, X, Y, Z]`, comme `Flip(1)` de MONAI). Les sorties de l'image retournée sont retournées en retour, puis les canaux gauche/droite sont échangés (1↔2, 3↔4, 5↔6, 7↔8, 11↔12, 14↔15, 16↔17). La fusion est la moyenne des quatre jeux de LOGITS ; l'étiquette est l'argmax.
Écrit : `fusion_<n>_reel_labels.u8` (n³ octets, ordre C) et `fusion_<n>_reel_logits.f32` ([20, n, n, n], float32 petit-boutiste).

MÉMOIRE : chaque passage avant occupe 1,7 Go (U-Net) à 3,3 Go (Attention U-Net) à 128³, environ 4 fois plus à 160³ ; toujours lancer sous `scripts/garde_memoire.py`.

Usage : python scripts/garde_memoire.py 8 python scripts/make_bounti_fusion_reference.py <n>
"""

import sys

import numpy as np
import torch
from monai.networks.nets import AttentionUnet, UNet

DONNEES = "data/bounti"
CANAUX = 20
# canal de sortie c <- canal PARTENAIRE[c] de la sortie de l'image retournée
PARTENAIRE = [0, 2, 1, 4, 3, 6, 5, 8, 7, 9, 10, 12, 11, 13, 15, 14, 17, 16, 18, 19]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
entree = np.fromfile(f"{DONNEES}/reference/unet_{n}_reel_input.f32", dtype="<f4").reshape(1, 1, n, n, n)
x = torch.from_numpy(entree)

unet = UNet(spatial_dims=3, in_channels=1, out_channels=CANAUX, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3,
            num_res_units=1, act="PRELU", norm="INSTANCE", dropout=0.5)
unet.load_state_dict(torch.load(f"{DONNEES}/models/monai-checkpoints-unet-brain-bounti-19-lab/best_metric_model.pth", map_location="cpu", weights_only=True), strict=True)
attunet = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=CANAUX, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3, dropout=0.5)
attunet.load_state_dict(torch.load(f"{DONNEES}/models/monai-checkpoints-red-attunet-brain_bounti-19-lab/best_metric_model.pth", map_location="cpu", weights_only=True), strict=True)

somme = torch.zeros(1, CANAUX, n, n, n)
with torch.no_grad():
    for modele in (unet.eval(), attunet.eval()):
        somme += modele(x)
        sortie_retournee = modele(torch.flip(x, [2]))
        somme += torch.flip(sortie_retournee, [2])[:, PARTENAIRE]
        del sortie_retournee
fusion = somme / 4.0
etiquettes = fusion.argmax(dim=1)[0].numpy().astype(np.uint8)
etiquettes.tofile(f"{DONNEES}/reference/fusion_{n}_reel_labels.u8")
fusion[0].numpy().astype("<f4").tofile(f"{DONNEES}/reference/fusion_{n}_reel_logits.f32")
print(f"fusion n={n} : {np.bincount(etiquettes.ravel(), minlength=CANAUX).tolist()} voxels par étiquette")
