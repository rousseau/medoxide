"""Références de PyTorch pour le critère 3 de P5 (docs/ULF.md) : méthode B de fmi (`compare_methods.estimate_inr`,
référence cb7174b) sur les plans de P1, graines 0, 1, 2 en float64 et graine 0 en float32 / complex64.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # compare_methods.py, utils_cpr.py, inr.py de fmi (cb7174b)
    PYTHONPATH=. uv run python <medoxide>/scripts/graines_b.py --p1 <dossier> --sortie <dossier> [--plans sub000_xy,…]

`--p1` : `<sujet>/<plan>/inputs.npz` de P1. Sortie : `<sujet>/<plan>/b{,_s1,_s2,_f32}.npz` (`b0_hz`) et `.json`
(`secondes`, perte finale).

`INRField` fait `torch.manual_seed(seed)` dans son constructeur, avec 0 par défaut, et `estimate_inr` ne lui passe pas de
graine : la graine est passée explicitement (comme `b0_bases_lisses.py --seed`). Le float32 reprend
`b0_bases_lisses.simple_precision` pour le champ (réseau, scalaires, grille de coordonnées) et passe données et temps en
simple précision ; le type du champ final est vérifié. En float64, les opérateurs CPR sont recalculés au retour
(`cas_pertes.checkpointer`) pour tenir en mémoire ; le float32, qui sert de durée de référence, ne l'est pas.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import compare_methods as cm
import utils_cpr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_pertes import checkpointer, cprmod  # noqa: E402

PLANS = [f"{s}_{q}" for s in ("sub000", "subB") for q in ("xy", "yz", "zx")]


def methode_b(entrees, graine, f32, device):
    i = np.load(entrees)
    k_reg = torch.tensor(i["k_reg"], device=device)
    k_inv = torch.tensor(i["k_inv"], device=device)
    t = torch.tensor(i["time_array"], device=device)
    espacement = i["spacing_mm"].tolist()
    bornes = i["b0_bounds"].tolist()
    if f32:
        k_reg, k_inv, t = k_reg.to(torch.complex64), k_inv.to(torch.complex64), t.float()
    champ0, grille0 = cm.INRField, utils_cpr.make_coordinate_grid

    def champ(**kw):
        f = champ0(**{**kw, "seed": graine})
        if f32:
            f.net.float()
            f.scale_param = f.scale_param.detach().float().requires_grad_()
            f.offset_param = f.offset_param.detach().float().requires_grad_()
        return f

    cm.INRField = champ
    if f32:
        utils_cpr.make_coordinate_grid = lambda shape, **kw: grille0(shape, **{**kw, "dtype": torch.float32})
    try:
        torch.manual_seed(graine)
        lncc_vox = [cm.LNCC_SIGMA_MM / h for h in espacement]
        t0 = time.time()
        b0, historique = cm.estimate_inr(k_reg, k_inv, t, bornes, espacement, lncc_vox, None, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        secondes = time.time() - t0
    finally:
        cm.INRField, utils_cpr.make_coordinate_grid = champ0, grille0
    if f32:
        assert b0.dtype == torch.float32, b0.dtype
    return b0.detach().double().cpu().numpy(), secondes, historique["loss_b0"][-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p1", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--plans", default=",".join(PLANS))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    originaux = (cprmod.time_segmented_cpr, cprmod.time_segmented_cpr_adjoint)
    for plan in args.plans.split(","):
        sujet, q = plan.split("_")
        d = Path(args.sortie) / sujet / q
        d.mkdir(parents=True, exist_ok=True)
        for nom, graine, f32 in (("b_f32", 0, True), ("b", 0, False), ("b_s1", 1, False), ("b_s2", 2, False)):
            # recalcul au retour pour le float64 seulement (mémoire) ; le float32, mesure de durée de référence, sans
            cprmod.time_segmented_cpr, cprmod.time_segmented_cpr_adjoint = originaux
            if not f32:
                checkpointer()
            b0, secondes, perte = methode_b(Path(args.p1) / sujet / q / "inputs.npz", graine, f32, device)
            np.savez_compressed(d / f"{nom}.npz", b0_hz=b0)
            (d / f"{nom}.json").write_text(json.dumps({"secondes": secondes, "perte_b0_finale": perte,
                                                       "dispositif": str(device)}, indent=1))
            print(f"{plan} {nom} : {secondes:.0f} s, perte b0 finale {perte:.5f}", flush=True)


if __name__ == "__main__":
    main()
