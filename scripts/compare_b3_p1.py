"""Critère 3 amendé de P5 (docs/ULF.md, décision du 2026-11-23) : méthode B de medoxide-ulf comparée à PyTorch sur les six
plans de P1, à initialisation égale (3a), sur dix graines de chaque côté (3b), et contrôle du cache de P1 (3c).

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # compare_methods.py et utils_cpr.py de fmi (cb7174b)
    PYTHONPATH=. uv run python <medoxide>/scripts/compare_b3_p1.py --rust <dossier> --graines <dossier> --p1 <dossier> \\
        [--sortie resultats.json]

- `--rust` : sorties de `methode_b_p1` sur la RTX : `<plan>_meme_init_cuda.h5` (`--poids`, 3a) et
  `<plan>_cuda.h5`, `<plan>_cuda_s<g>.h5` (g = 1 … 9) ;
- `--graines` : `<sujet>/<plan>/b.npz` (f64, graine 0), `b_f32.npz` et `b_f32_s<g>.npz` (g = 1 … 9) de `graines_b.py` ;
- `--p1` : `<sujet>/<plan>/inputs.npz` de P1.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import compare_methods as cm

PLANS = [f"{s}_{q}" for s in ("sub000", "subB") for q in ("xy", "yz", "zx")]
N_GRAINES = 10


def rms(a, b, m):
    return float(np.sqrt(np.mean((a - b)[m] ** 2)))


def paires(xs, m):
    return [rms(xs[i], xs[j], m) for i in range(len(xs)) for j in range(i + 1, len(xs))]


def plan(args, nom):
    sujet, q = nom.split("_")
    i = np.load(Path(args.p1) / sujet / q / "inputs.npz")
    k_reg, k_inv = torch.tensor(i["k_reg"]), torch.tensor(i["k_inv"])
    t, mask, esp = torch.tensor(i["time_array"]), torch.tensor(i["mask"]), i["spacing_mm"].tolist()
    m = i["mask"]
    g = Path(args.graines) / sujet / q
    lire_py = lambda n: np.load(g / f"{n}.npz")["b0_hz"].astype(np.float64)  # noqa: E731
    py64 = lire_py("b")
    py32 = [lire_py("b_f32")] + [lire_py(f"b_f32_s{k}") for k in range(1, N_GRAINES)]

    def lire_ru(suf):
        with h5py.File(Path(args.rust) / f"{nom}{suf}.h5") as f:
            return f["b0_hz"][()]

    meme = lire_ru("_meme_init_cuda")
    ru = [lire_ru("_cuda")] + [lire_ru(f"_cuda_s{k}") for k in range(1, N_GRAINES)]

    def lncc(b0):
        return cm.pair_criteria(k_reg, k_inv, t, torch.tensor(b0), mask, esp)["lncc"]

    ref_f32 = rms(py32[0], py64, m)
    lncc_py, lncc_ru = [lncc(b) for b in py32], [lncc(b) for b in ru]
    p_py, p_ru = paires(py32, m), paires(ru, m)
    med = float(np.median(p_py))
    r = {
        "3a_rms_hz": rms(meme, py64, m),
        "3a_seuil_hz": 2 * ref_f32,
        "3a_lncc": lncc(meme),
        "3a_lncc_pytorch_f64": lncc(py64),
        "3b_lncc_rust": lncc_ru,
        "3b_lncc_pytorch": lncc_py,
        "3b_mediane_paires_rust_hz": float(np.median(p_ru)),
        "3b_mediane_paires_pytorch_hz": med,
        "3b_moyennes_hz": rms(sum(ru) / len(ru), sum(py32) / len(py32), m),
        "3c_rms_hz": rms(py64, i["b0_warm_start"].astype(np.float64), m),
        "3c_seuil_hz": 2 * ref_f32,
    }
    r["3a_ok"] = r["3a_rms_hz"] <= r["3a_seuil_hz"]
    r["3b_lncc_ok"] = min(lncc_py) <= float(np.median(lncc_ru)) <= max(lncc_py)
    r["3b_dispersion_ok"] = med / 2 <= r["3b_mediane_paires_rust_hz"] <= 2 * med
    r["3b_moyennes_ok"] = r["3b_moyennes_hz"] <= med
    r["3c_ok"] = r["3c_rms_hz"] <= r["3c_seuil_hz"]
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rust", required=True)
    p.add_argument("--graines", required=True)
    p.add_argument("--p1", required=True)
    p.add_argument("--plans", default=",".join(PLANS))
    p.add_argument("--sortie")
    args = p.parse_args()
    tout = {}
    ok = lambda b: "ok" if b else "HORS"  # noqa: E731
    for nom in args.plans.split(","):
        r = plan(args, nom)
        tout[nom] = r
        print(f"{nom} : 3a {r['3a_rms_hz']:.2f} Hz (seuil {r['3a_seuil_hz']:.2f}) {ok(r['3a_ok'])}, LNCC "
              f"{r['3a_lncc']:.4f} / f64 {r['3a_lncc_pytorch_f64']:.4f} | 3b LNCC médiane "
              f"{np.median(r['3b_lncc_rust']):.4f} [{min(r['3b_lncc_pytorch']):.4f}, {max(r['3b_lncc_pytorch']):.4f}] "
              f"{ok(r['3b_lncc_ok'])}, paires {r['3b_mediane_paires_rust_hz']:.2f} / "
              f"{r['3b_mediane_paires_pytorch_hz']:.2f} Hz {ok(r['3b_dispersion_ok'])}, moyennes "
              f"{r['3b_moyennes_hz']:.2f} Hz {ok(r['3b_moyennes_ok'])} | 3c {r['3c_rms_hz']:.2f} Hz "
              f"(seuil {r['3c_seuil_hz']:.2f}) {ok(r['3c_ok'])}", flush=True)
    if args.sortie:
        Path(args.sortie).write_text(json.dumps(tout, indent=1))


if __name__ == "__main__":
    main()
