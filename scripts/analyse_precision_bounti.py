"""Situe les écarts de logits entre implémentations f32 d'un réseau BOUNTI par rapport à une référence en double précision (float64).

Pourquoi : deux implémentations f32 correctes d'une même convolution 3D diffèrent par le bruit d'arrondi, qui dépend de l'ordre des sommes. Comparer Burn à PyTorch f32 mélange l'erreur de PyTorch et celle de Burn. Contre une référence float64, on voit laquelle est la plus proche de la vérité.
Écart relatif = max |Δ| / max |logit|. Entrées : `data/bounti/reference/<réseau>_<n>[tag]_input.f32`, logits PyTorch f32 `_logits.f32`, logits Burn `_burn_gpu.f32` / `_burn_flex.f32` (écrits par `MEDOXIDE_DUMP=1 cargo test -p medoxide-bounti -- --ignored`), et à 64³ l'export `data/bounti/onnx/<réseau>_64.onnx`.

Deux temps, dans deux processus pour que la mémoire du calcul soit rendue avant la comparaison : (1) le passage avant en float64, écrit dans `_f64.f64` s'il n'existe pas ; (2) la comparaison, par canal, depuis le disque.

MÉMOIRE : le passage avant en float64 occupe environ 6,8 Go à 160³ pour l'U-Net (le double de l'estimation naïve) et croît comme n³ : environ 12 Go à 192³, plus de 28 Go à 256³ (ce qui a fait planter une machine de 24 Go). Toujours le lancer sous `scripts/garde_memoire.py`.

Usage : python scripts/garde_memoire.py 7 python scripts/analyse_precision_bounti.py <unet|attunet> [n] [tag]      (tag : `_reel` pour les références sur volume réel)
"""

import os
import subprocess
import sys

import numpy as np

DONNEES = "data/bounti"
reseau = sys.argv[1] if len(sys.argv) > 1 else "unet"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
tag = sys.argv[3] if len(sys.argv) > 3 else ""
etape = sys.argv[4] if len(sys.argv) > 4 else "tout"
CANAUX = 20
voxels = n ** 3
chemin = lambda nom: f"{DONNEES}/reference/{reseau}_{n}{tag}_{nom}"


def ecrire_reference_f64():
    """Passage avant float64 de PyTorch/MONAI, écrit sur disque (tableau brut float64 petit-boutiste)."""
    import torch
    from monai.networks.nets import AttentionUnet, UNet

    if reseau == "unet":
        modele = UNet(spatial_dims=3, in_channels=1, out_channels=CANAUX, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3,
                      num_res_units=1, act="PRELU", norm="INSTANCE", dropout=0.5)
        dossier = "monai-checkpoints-unet-brain-bounti-19-lab"
    else:
        modele = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=CANAUX, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), kernel_size=3, up_kernel_size=3, dropout=0.5)
        dossier = "monai-checkpoints-red-attunet-brain_bounti-19-lab"
    modele.load_state_dict(torch.load(f"{DONNEES}/models/{dossier}/best_metric_model.pth", map_location="cpu", weights_only=True), strict=True)
    modele.eval()
    entree = np.fromfile(chemin("input.f32"), dtype="<f4").reshape(1, 1, n, n, n)
    with torch.no_grad():
        sortie = modele.double()(torch.from_numpy(entree.astype("float64"))).numpy()
    sortie.astype("<f8").tofile(chemin("f64.f64"))


def comparer():
    """Écart de chaque implémentation à la référence float64, canal par canal (sans copie complète en double précision)."""
    vrai = np.memmap(chemin("f64.f64"), dtype="<f8", mode="r", shape=(CANAUX, voxels))
    amplitude = max(float(np.abs(vrai[c]).max()) for c in range(CANAUX))
    candidats = {"PyTorch f32": chemin("logits.f32")}
    for suffixe, nom in [("gpu", "Burn GPU f32"), ("flex", "Burn CPU (flex) f32")]:
        if os.path.exists(chemin(f"burn_{suffixe}.f32")):
            candidats[nom] = chemin(f"burn_{suffixe}.f32")
    onnx = {}
    if n == 64:  # l'export ONNX est tracé à 64³ : onnxruntime n'est comparé que pour cette taille
        import onnxruntime as ort

        session = ort.InferenceSession(f"{DONNEES}/onnx/{reseau}_{n}.onnx", providers=["CPUExecutionProvider"])
        entree = np.fromfile(chemin("input.f32"), dtype="<f4").reshape(1, 1, n, n, n)
        sortie = session.run(None, {"image": entree})[0].reshape(CANAUX, voxels).astype("<f4")
        sortie.tofile(chemin("onnx.f32"))
        candidats["onnxruntime f32"] = chemin("onnx.f32")
    etiquettes_vraies = np.argmax(vrai, axis=0)
    print(f"{reseau} n={n}{tag}, |logit| max {amplitude:.1f} ; écart à la référence float64")
    print(f"{'implémentation':22s} {'écart relatif':>14s} {'écart absolu':>13s} {'accord des étiquettes':>22s}")
    for nom, fichier in candidats.items():
        logits = np.memmap(fichier, dtype="<f4", mode="r", shape=(CANAUX, voxels))
        ecart = max(float(np.abs(logits[c].astype("float64") - vrai[c]).max()) for c in range(CANAUX))
        accord = float((np.argmax(logits, axis=0) == etiquettes_vraies).mean())
        print(f"{nom:22s} {ecart / amplitude:14.2e} {ecart:13.2e} {accord:22.6f}")


if etape == "ecrire":
    ecrire_reference_f64()
elif etape == "comparer":
    comparer()
else:
    if not os.path.exists(chemin("f64.f64")):
        # processus séparé : la mémoire du passage avant est rendue avant la comparaison
        subprocess.run([sys.executable, __file__, reseau, str(n), tag, "ecrire"], check=True)
    comparer()
