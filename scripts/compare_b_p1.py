"""Critère 3 de P5 (docs/ULF.md) : méthode B de medoxide-ulf de bout en bout (trois graines), comparée aux graines de
la méthode B de PyTorch sur les six plans de P1.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # compare_methods.py et utils_cpr.py de fmi (cb7174b)
    PYTHONPATH=. uv run python <medoxide>/scripts/compare_b_p1.py --rust <dossier> --graines <dossier> --p1 <dossier> \\
        [--suffixes _cuda,_cuda_s1,_cuda_s2] [--sortie resultats.json]

- `--rust` : sorties de l'exemple `methode_b_p1` (`<sujet>_<plan><suffixe>.h5`, `b0_hz`) ; la première est la graine 0 ;
- `--graines` : `<sujet>/<plan>/b{,_s1,_s2,_f32}.npz` de `scripts/graines_b.py` ;
- `--p1` : `<sujet>/<plan>/inputs.npz` de P1 (k-space, temps, masque de l'objet, espacement, `b0_warm_start`).

Critères (écrits avant le code) :
- B0 : RMS dans l'objet de la graine 0 de Rust − graine 0 de PyTorch (float64) ≤ plus grand RMS entre deux des graines
  0, 1, 2 de PyTorch ;
- LNCC de la paire corrigée (`pair_criteria`) de la graine 0 de Rust dans l'enveloppe des graines 0, 1, 2 et du float32 de
  PyTorch ;
- dispersion : plus grand RMS entre deux graines de Rust ≤ 2 × celui de PyTorch ; RMS entre la moyenne des graines de
  Rust et celle de PyTorch ≤ plus grand RMS entre deux graines de PyTorch ;
- contrôle : la graine 0 de PyTorch redonne le `b0_warm_start` de P1 à la précision float32 de son stockage.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import compare_methods as cm

PLANS = [f"{s}_{q}" for s in ("sub000", "subB") for q in ("xy", "yz", "zx")]


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
    py = {n: np.load(g / f"{n}.npz")["b0_hz"].astype(np.float64) for n in ("b", "b_s1", "b_s2", "b_f32")}
    ru = []
    for suf in args.suffixes.split(","):
        with h5py.File(Path(args.rust) / f"{nom}{suf}.h5") as f:
            ru.append((f["b0_hz"][()], float(f.attrs["secondes"])))

    def lncc(b0):
        return cm.pair_criteria(k_reg, k_inv, t, torch.tensor(b0), mask, esp)["lncc"]

    graines_py = [py["b"], py["b_s1"], py["b_s2"]]
    seuil = max(paires(graines_py, m))
    lncc_py = {n: lncc(b) for n, b in py.items()}
    b0_ru = [b for b, _ in ru]
    ref_p1 = i["b0_warm_start"].astype(np.float64)
    r = {
        "b0_rms_hz": rms(b0_ru[0], py["b"], m),
        "b0_seuil_hz": seuil,
        "b0_rms_pytorch_f32_hz": rms(py["b_f32"], py["b"], m),
        "lncc": lncc(b0_ru[0]),
        "lncc_enveloppe": [min(lncc_py.values()), max(lncc_py.values())],
        "paires_rust_hz": paires(b0_ru, m),
        "paires_pytorch_hz": paires(graines_py, m),
        "moyennes_hz": rms(sum(b0_ru) / len(b0_ru), sum(graines_py) / 3, m),
        "secondes": [s for _, s in ru],
        "controle_p1_hz": rms(py["b"], ref_p1, m),
        "controle_p1_relatif_max": float(np.max(np.abs(py["b"] - ref_p1)) / np.max(np.abs(ref_p1))),
    }
    r["b0_ok"] = r["b0_rms_hz"] <= seuil
    r["lncc_ok"] = r["lncc_enveloppe"][0] <= r["lncc"] <= r["lncc_enveloppe"][1]
    r["dispersion_ok"] = max(r["paires_rust_hz"]) <= 2 * max(r["paires_pytorch_hz"]) and r["moyennes_hz"] <= seuil
    r["controle_p1_ok"] = r["controle_p1_relatif_max"] <= 1e-6
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rust", required=True)
    p.add_argument("--graines", required=True)
    p.add_argument("--p1", required=True)
    p.add_argument("--suffixes", default="_cuda,_cuda_s1,_cuda_s2")
    p.add_argument("--plans", default=",".join(PLANS))
    p.add_argument("--sortie")
    args = p.parse_args()
    tout = {}
    for nom in args.plans.split(","):
        r = plan(args, nom)
        tout[nom] = r
        ok = lambda b: "ok" if b else "HORS"  # noqa: E731
        print(f"{nom} : B0 {r['b0_rms_hz']:.2f} Hz (seuil {r['b0_seuil_hz']:.2f}, PyTorch f32 "
              f"{r['b0_rms_pytorch_f32_hz']:.2f}) {ok(r['b0_ok'])} | LNCC {r['lncc']:.4f} [{r['lncc_enveloppe'][0]:.4f}, "
              f"{r['lncc_enveloppe'][1]:.4f}] {ok(r['lncc_ok'])} | graines Rust "
              f"{np.round(r['paires_rust_hz'], 2).tolist()} PyTorch {np.round(r['paires_pytorch_hz'], 2).tolist()}, "
              f"moyennes {r['moyennes_hz']:.2f} Hz {ok(r['dispersion_ok'])} | contrôle P1 {r['controle_p1_hz']:.2e} Hz "
              f"({r['controle_p1_relatif_max']:.1e}) {ok(r['controle_p1_ok'])} | {np.round(r['secondes']).tolist()} s",
              flush=True)
    if args.sortie:
        Path(args.sortie).write_text(json.dumps(tout, indent=1))


if __name__ == "__main__":
    main()
