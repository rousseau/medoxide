"""Exporte un réseau 3D de Multi-BOUNTI en ONNX et vérifie l'export (voir docs/MULTIBOUNTI.md, sous-étape M1).

Les trois réseaux de la chaîne fœtale sont le MÊME `AttentionUnet` de MONAI (script d'inférence officiel `run_monai_patch_atunet_segmentation_1case-2026-*.py`) : 1 canal d'entrée, canaux (32, 64, 128, 256, 512), pas (2, 2, 2, 2), noyaux 3, dropout 0,5 ; seul le nombre de sorties change.
Les poids ne sont PAS dans ce dépôt (GPL v3) : ils viennent de l'image Docker de Multi-BOUNTI, extraits dans `data/multibounti/image/home/perinatal-brain-mri-analysis/models/` (voir `docs/MULTIBOUNTI.md`).
Chargement STRICT (le script officiel charge avec `strict=False` : ici une clé manquante ou en trop est une erreur ; vérifié, il n'y en a aucune).
Critère de l'export : celui de `export_bounti_onnx.py` (écart relatif maximal des logits PyTorch / onnxruntime ≤ 1e-5, au moins 99,9 % d'étiquettes identiques sur plusieurs entrées).

Usage   : python scripts/export_multibounti_onnx.py <bet|lr|multi> [n]      (n : côté du cube factice, 64 par défaut)
Réseaux : bet   = atunet_bet_brain_153t_svr_2026_128 (2 sorties : fond, cerveau)
          lr    = atunet_lr_brain_fetal_2lab (3 sorties : fond, gauche, droite)
          multi = patch_atunet_new_multi_43_brain_neo_fetal_aff_fix_022026 (44 sorties : fond et 43 étiquettes)
Sortie  : data/multibounti/onnx/<réseau>_<n>.onnx
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from monai.networks.nets import AttentionUnet

MODELES = "data/multibounti/image/home/perinatal-brain-mri-analysis/models"
SORTIE = "data/multibounti/onnx"
RESEAUX = {
    "bet": ("atunet_bet_brain_153t_svr_2026_128_best_metric_model.pth", 2),
    "lr": ("atunet_lr_brain_fetal_2lab_best_metric_model.pth", 3),
    "multi": ("patch_atunet_new_multi_43_brain_neo_fetal_aff_fix_022026_best_metric_model.pth", 44),
}
SEUIL_REL = 1e-5
SEUIL_ACCORD = 0.999
N_ENTREES = 4

reseau = sys.argv[1] if len(sys.argv) > 1 else "multi"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
if reseau not in RESEAUX:
    raise SystemExit(f"réseau inconnu : {reseau} (bet, lr ou multi)")
fichier, sorties = RESEAUX[reseau]
modele = AttentionUnet(
    spatial_dims=3, in_channels=1, out_channels=sorties, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    kernel_size=3, up_kernel_size=3, dropout=0.5,
)
modele.load_state_dict(torch.load(f"{MODELES}/{fichier}", map_location="cpu", weights_only=True), strict=True)
modele.eval()  # désactive le dropout

os.makedirs(SORTIE, exist_ok=True)
chemin = f"{SORTIE}/{reseau}_{n}.onnx"
torch.manual_seed(0)
factice = torch.rand(1, 1, n, n, n)
torch.onnx.export(
    modele, factice, chemin, input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
)
print(f"Exporté : {chemin} ({os.path.getsize(chemin) / 1e6:.1f} Mo), {sorties} sorties")

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
print(f"{N_ENTREES} entrées uniformes dans [0, 1] (comme l'intensité min-max du script officiel)")
print(f"Écart relatif max : {rel_max:.2e} (seuil {SEUIL_REL:.0e}) ; écart absolu max : {abs_max:.2e} (indicatif)")
print(f"Accord argmax min : {accord_min:.6f} (seuil {SEUIL_ACCORD})")
ok = rel_max <= SEUIL_REL and accord_min >= SEUIL_ACCORD
print("CRITÈRE ATTEINT" if ok else "CRITÈRE NON ATTEINT")
raise SystemExit(0 if ok else 1)
