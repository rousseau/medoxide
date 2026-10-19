"""Exporte un réseau 3D de BOUNTI en ONNX et vérifie l'export (voir docs/BOUNTI.md).

Les poids ne sont PAS dans ce dépôt (GPL v3, licence de BOUNTI) : ils viennent de l'image Docker `fetalsvrtk/svrtk:general_auto_arm`, dossier `/home/auto-proc-svrtk/trained_models/`, extrait une fois dans `data/bounti/models/` (ignoré par Git).
Les architectures sont celles du script d'inférence officiel de BOUNTI, écrites ici avec les classes MONAI (Apache-2.0) : `AttentionUnet` et `UNet`, 1 canal d'entrée et 19 étiquettes + fond = 20 sorties.

Critère de l'export (critère C1 de docs/BOUNTI.md, amendé) : écart RELATIF maximal des logits PyTorch vs onnxruntime (max |Δ| / max |logit|) ≤ 1e-5 et au moins 99,9 % d'étiquettes (argmax) identiques, sur plusieurs entrées. Le critère initial était un écart absolu ≤ 1e-3 ; il est faux pour des réseaux aux logits de grande amplitude (Attention U-Net : logits jusqu'à environ 300, écart absolu 1,3e-3 pour un écart relatif de 4,4e-6, identique à celui de l'U-Net), comme déjà établi pour Fetal-BET. Contrairement à Fetal-BET (2 classes, accord exigé à 100 %), avec 20 classes un voxel dont les logits sont quasi à égalité peut basculer sans qu'il y ait d'erreur. L'écart absolu est affiché à titre indicatif.

Usage   : python scripts/export_bounti_onnx.py <unet|attunet|bet> [n]      (n : côté du cube factice, 64 par défaut)
Entrée  : data/bounti/models/<dossier du modèle>/best_metric_model.pth
Sortie  : data/bounti/onnx/<réseau>_<n>.onnx
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from monai.networks.nets import AttentionUnet, UNet

MODELES = "data/bounti/models"
SORTIE = "data/bounti/onnx"
SEUIL_REL = 1e-5
SEUIL_ACCORD = 0.999
N_ENTREES = 4
N_ETIQUETTES = 19


def construire(reseau):
    """Architecture MONAI et dossier du checkpoint du réseau demandé."""
    sorties = N_ETIQUETTES + 1
    if reseau == "unet":
        modele = UNet(
            spatial_dims=3, in_channels=1, out_channels=sorties, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            kernel_size=3, up_kernel_size=3, num_res_units=1, act="PRELU", norm="INSTANCE", dropout=0.5,
        )
        return modele, "monai-checkpoints-unet-brain-bounti-19-lab"
    if reseau == "attunet":
        modele = AttentionUnet(
            spatial_dims=3, in_channels=1, out_channels=sorties, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
            kernel_size=3, up_kernel_size=3, dropout=0.5,
        )
        return modele, "monai-checkpoints-red-attunet-brain_bounti-19-lab"
    if reseau == "bet":
        # extraction du cerveau : Attention U-Net 3D à 2 sorties (fond, cerveau), canaux 32 à 512
        modele = AttentionUnet(
            spatial_dims=3, in_channels=1, out_channels=2, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            kernel_size=3, up_kernel_size=3, dropout=0.5,
        )
        return modele, "monai-checkpoints-attunet-brain-bet-1-lab"
    raise SystemExit(f"réseau inconnu : {reseau} (unet, attunet ou bet)")


reseau = sys.argv[1] if len(sys.argv) > 1 else "unet"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
modele, dossier = construire(reseau)
# chargement strict : une clé manquante ou en trop est une erreur (le script officiel charge avec strict=False)
modele.load_state_dict(torch.load(f"{MODELES}/{dossier}/best_metric_model.pth", map_location="cpu", weights_only=True), strict=True)
modele.eval()  # désactive le dropout

os.makedirs(SORTIE, exist_ok=True)
chemin = f"{SORTIE}/{reseau}_{n}.onnx"
# Entrée factice [N, canaux, X, Y, Z] ; seul l'axe N (batch) est dynamique.
torch.manual_seed(0)
factice = torch.rand(1, 1, n, n, n)
torch.onnx.export(
    modele, factice, chemin, input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
    dynamo=False,  # exporteur historique (TorchScript), le plus répandu
)
print(f"Exporté : {chemin} ({os.path.getsize(chemin) / 1e6:.1f} Mo)")

session = ort.InferenceSession(chemin, providers=["CPUExecutionProvider"])


def comparer(x):
    """Renvoie (écart absolu maximal des logits, écart relatif, fraction de voxels de même argmax)."""
    with torch.no_grad():
        ref = modele(x).numpy()
    sortie = session.run(None, {"image": x.numpy()})[0]
    ecart = float(np.abs(ref - sortie).max())
    return ecart, ecart / float(np.abs(ref).max()), float((ref.argmax(1) == sortie.argmax(1)).mean())


resultats = [comparer(torch.rand(1, 1, n, n, n)) for _ in range(N_ENTREES)]
abs_max = max(e for e, _, _ in resultats)
rel_max = max(r for _, r, _ in resultats)
accord_min = min(a for _, _, a in resultats)
print(f"{N_ENTREES} entrées uniformes dans [0, 1] (comme l'intensité min-max de BOUNTI)")
print(f"Écart relatif max : {rel_max:.2e} (seuil {SEUIL_REL:.0e}) ; écart absolu max : {abs_max:.2e} (indicatif)")
print(f"Accord argmax min : {accord_min:.6f} (seuil {SEUIL_ACCORD})")
ok = rel_max <= SEUIL_REL and accord_min >= SEUIL_ACCORD
print("CRITÈRE ATTEINT" if ok else "CRITÈRE NON ATTEINT")
raise SystemExit(0 if ok else 1)
