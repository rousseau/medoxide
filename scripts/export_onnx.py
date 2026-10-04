"""Exporte le modèle Fetal-BET (Attention U-Net 2D) en ONNX et vérifie l'export.

Étape 1 de la roadmap, critère (sur au moins 10 coupes) :
  - écart relatif des logits PyTorch vs onnxruntime < 1e-5
    (max |Δ| / max |logit| : un seuil absolu est trop strict, les logits
    atteignent ~40 et le float32 accumule du bruit d'arrondi) ;
  - masque (argmax) identique à 100 %.

Usage   : python scripts/export_onnx.py [dossier_de_volumes_nifti]
Entrée  : models/AttUNet.pth
Sortie  : models/attunet.onnx
Sans dossier : coupes gaussiennes seulement. Avec un dossier : on ajoute de
vraies coupes (volumes 256×256 dans le plan, z-score par coupe sur les
voxels > 0, sans rééchantillonnage : test purement numérique, qui n'imite PAS le
prétraitement officiel ; voir make_reference_slice.py).
"""

import sys
from glob import glob

import nibabel as nib
import numpy as np
import onnxruntime as ort
import torch
from monai.networks.nets import AttentionUnet

PTH = "models/AttUNet.pth"
ONNX = "models/attunet.onnx"
SEUIL_REL = 1e-5
N_COUPES = 12
COUPES_PAR_VOLUME = 3

# Même architecture que src/codes/inference.py dans fetal-brain-extraction.
model = AttentionUnet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    dropout=0.15,
)

# Les poids ont été sauvegardés avec DataParallel : clés préfixées "module.".
sd = torch.load(PTH, map_location="cpu", weights_only=True)
sd = {k.removeprefix("module."): v for k, v in sd.items()}
model.load_state_dict(sd, strict=True)
model.eval()  # désactive le dropout

# Entrée factice [N, canaux, H, W] ; seul l'axe N (batch) est dynamique.
dummy = torch.randn(1, 1, 256, 256)
torch.onnx.export(
    model,
    dummy,
    ONNX,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    dynamo=False,  # exporteur historique (TorchScript), le plus répandu
)
print(f"Exporté : {ONNX}")

# Vérification : mêmes entrées dans PyTorch et onnxruntime.
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])


def compare(x):
    """Renvoie (écart relatif des logits, fraction de voxels au même argmax)."""
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"image": x.numpy()})[0]
    rel = np.abs(ref - out).max() / np.abs(ref).max()
    return float(rel), float((ref.argmax(1) == out.argmax(1)).mean())


def coupes_reelles(dossier):
    """Quelques coupes par volume, normalisées comme dans inference.py."""
    for f in sorted(glob(f"{dossier}/*.nii.gz")):
        vol = nib.load(f).get_fdata(dtype=np.float32)
        for z in np.linspace(0, vol.shape[2] - 1, COUPES_PAR_VOLUME + 2)[1:-1]:
            sl = vol[:, :, int(z)].copy()
            m = sl > 0
            if m.any():
                sl[m] = (sl[m] - sl[m].mean()) / sl[m].std()
            yield torch.from_numpy(sl)[None, None]


torch.manual_seed(0)
entrees = [torch.randn(1, 1, 256, 256) for _ in range(N_COUPES)]
entrees += [torch.randn(3, 1, 256, 256)]  # batch > 1 : axe dynamique
if len(sys.argv) > 1:
    entrees += list(coupes_reelles(sys.argv[1]))

res = [compare(x) for x in entrees]
rel_max = max(r for r, _ in res)
accord_min = min(a for _, a in res)
print(f"{len(entrees)} entrées (dont {len(entrees) - N_COUPES - 1} coupes réelles)")
print(f"Écart relatif max : {rel_max:.2e} (seuil {SEUIL_REL:.0e})")
print(f"Accord argmax min : {accord_min:.6f} (seuil 1)")
ok = rel_max < SEUIL_REL and accord_min == 1.0
print("CRITÈRE ATTEINT" if ok else "CRITÈRE NON ATTEINT")
raise SystemExit(0 if ok else 1)
