"""Critère 4 de P5, référence chaînée (docs/ULF.md, décision du 2026-11-23) : chaînes B → jointe de medoxide-ulf et de
PyTorch, cinq de chaque côté, sur les six plans de P1.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # compare_methods.py et utils_cpr.py de fmi (cb7174b)
    PYTHONPATH=.:<medoxide>/scripts uv run python <medoxide>/scripts/compare_b4_p1.py --rust <dossier> \\
        --pytorch <dossier> --graines <dossier> --p1 <dossier> [--sortie resultats.json]

- `--rust` : `<plan>_chaine_cuda.h5` (chaîne 0) et `<plan>_chaine_cuda_s<g>.h5` (g = 1 … 4) de `jointe_p1 --b0-init` ;
- `--pytorch` : `g<g>/<sujet>/<plan>/inr32_f32.{npz,json}` (g = 0) et `inr32_s<g>_f32.{npz,json}` de `b0_bases_lisses.py`
  lancé sur les entrées de `chaine_b_pytorch.py` ;
- `--graines` : graines de la jointe de P4 bis (`inr32_f32.{npz,json}`), pour σ, comme `compare_jointe_p1.py` ;
- `--p1` : `<sujet>/<plan>/inputs.npz` de P1.

Les LNCC et écarts aux données de PyTorch sont ceux de ses json (même code que le recalcul : contrôlé à 1e-16 et 4e-9 par
`compare_jointe_p1.py`) ; ceux de Rust sont recalculés.

Deux jugements sont imprimés :
- **tel qu'écrit** (avant la mesure) : 4a = chaîne 0 de Rust contre chaîne 0 de PyTorch, au plus le plus grand RMS entre
  deux chaînes de PyTorch ; 4b = LNCC et écart aux données médians de Rust dans [min, max] de PyTorch, médiane des paires
  de Rust dans [m/2, 2m], RMS entre les B0 moyens ≤ m (m = médiane des dix paires de PyTorch) ;
- **retenu** (décision de François du 2026-11-23, « tenu avec réserves ») : 4a = médiane des 25 RMS croisés Rust × PyTorch
  dans [m/2, 2m] ; écart aux données = différence des moyennes des deux côtés rapportée à son écart-type
  (√(s²_Rust/5 + s²_PyTorch/5)), à moins d'un écart-type. La LNCC, la dispersion et les moyennes de B0 restent jugées
  comme écrit ; l'écart-type réduit de la LNCC est aussi imprimé.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from compare_jointe_p1 import ecart_brut, lncc
from plans_ulf import option_plans

N = 5


def rms(a, b, m):
    return float(np.sqrt(np.mean((a - b)[m] ** 2)))


def z(a, b):
    """Différence des moyennes de a et b rapportée à son écart-type (variances empiriques)."""
    a, b = np.asarray(a), np.asarray(b)
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def paires(xs, m):
    return [rms(xs[i], xs[j], m) for i in range(len(xs)) for j in range(i + 1, len(xs))]


def plan(args, nom):
    sujet, q = nom.split("_")
    i = np.load(Path(args.p1) / sujet / q / "inputs.npz")
    k_reg, k_inv = torch.tensor(i["k_reg"]), torch.tensor(i["k_inv"])
    t, mask, esp = torch.tensor(i["time_array"]), torch.tensor(i["mask"]), i["spacing_mm"].tolist()
    m = i["mask"]

    g = Path(args.graines) / sujet / q
    f32 = np.load(g / "inr32_f32.npz")
    sigma = ecart_brut(k_reg, k_inv, t, mask, torch.tensor(f32["b0_hz"]).double(),
                       torch.tensor(f32["u"]).to(torch.complex128)) \
        / json.loads((g / "inr32_f32.json").read_text())["misfit_over_noise"]

    py, lncc_py, ecart_py = [], [], []
    for k in range(N):
        base = Path(args.pytorch) / f"g{k}" / sujet / q / ("inr32_f32" if k == 0 else f"inr32_s{k}_f32")
        py.append(np.load(f"{base}.npz")["b0_hz"].astype(np.float64))
        j = json.loads(Path(f"{base}.json").read_text())
        lncc_py.append(j["crit_lncc"])
        ecart_py.append(j["misfit_over_noise"])

    ru, lncc_ru, ecart_ru = [], [], []
    for k in range(N):
        with h5py.File(Path(args.rust) / f"{nom}_chaine_cuda{'' if k == 0 else f'_s{k}'}.h5") as f:
            b0, u = f["b0_hz"][()], f["u"][()]
        ru.append(b0.astype(np.float64))
        lncc_ru.append(lncc(k_reg, k_inv, t, torch.tensor(b0), mask, esp))
        ecart_ru.append(ecart_brut(k_reg, k_inv, t, mask, torch.tensor(b0), torch.tensor(u)) / sigma)

    p_py, p_ru = paires(py, m), paires(ru, m)
    med = float(np.median(p_py))
    r = {
        "4a_rms_hz": rms(ru[0], py[0], m),
        "4a_seuil_hz": max(p_py),
        "4b_lncc_rust": lncc_ru,
        "4b_lncc_pytorch": lncc_py,
        "4b_ecart_rust": ecart_ru,
        "4b_ecart_pytorch": ecart_py,
        "4b_paires_rust_hz": p_ru,
        "4b_paires_pytorch_hz": p_py,
        "4b_mediane_paires_rust_hz": float(np.median(p_ru)),
        "4b_mediane_paires_pytorch_hz": med,
        "4b_moyennes_hz": rms(sum(ru) / N, sum(py) / N, m),
        "4a_croisees_hz": [rms(a, b, m) for a in ru for b in py],
        "4b_ecart_z": z(ecart_ru, ecart_py),
        "4b_lncc_z": z(lncc_ru, lncc_py),
    }
    r["4a_mediane_croisees_hz"] = float(np.median(r["4a_croisees_hz"]))
    r["4a_ok"] = r["4a_rms_hz"] <= r["4a_seuil_hz"]
    r["4b_lncc_ok"] = min(lncc_py) <= float(np.median(lncc_ru)) <= max(lncc_py)
    r["4b_ecart_ok"] = min(ecart_py) <= float(np.median(ecart_ru)) <= max(ecart_py)
    r["4b_dispersion_ok"] = med / 2 <= r["4b_mediane_paires_rust_hz"] <= 2 * med
    r["4b_moyennes_ok"] = r["4b_moyennes_hz"] <= med
    r["4a_retenu_ok"] = med / 2 <= r["4a_mediane_croisees_hz"] <= 2 * med
    r["4b_ecart_retenu_ok"] = abs(r["4b_ecart_z"]) <= 1
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rust", required=True)
    p.add_argument("--pytorch", required=True)
    p.add_argument("--graines", required=True)
    p.add_argument("--p1", required=True)
    option_plans(p)
    p.add_argument("--sortie")
    args = p.parse_args()
    tout = {}
    ok = lambda b: "ok" if b else "HORS"  # noqa: E731
    for nom in args.plans.split(","):
        r = plan(args, nom)
        tout[nom] = r
        print(f"{nom} : 4a {r['4a_rms_hz']:.2f} Hz (seuil {r['4a_seuil_hz']:.2f}) {ok(r['4a_ok'])} | 4b LNCC médiane "
              f"{np.median(r['4b_lncc_rust']):.4f} [{min(r['4b_lncc_pytorch']):.4f}, {max(r['4b_lncc_pytorch']):.4f}] "
              f"{ok(r['4b_lncc_ok'])}, écart {np.median(r['4b_ecart_rust']):.4f} [{min(r['4b_ecart_pytorch']):.4f}, "
              f"{max(r['4b_ecart_pytorch']):.4f}] {ok(r['4b_ecart_ok'])}, paires {r['4b_mediane_paires_rust_hz']:.2f} / "
              f"{r['4b_mediane_paires_pytorch_hz']:.2f} Hz {ok(r['4b_dispersion_ok'])}, moyennes "
              f"{r['4b_moyennes_hz']:.2f} Hz {ok(r['4b_moyennes_ok'])}", flush=True)
        print(f"{' ' * len(nom)}   retenu : 4a croisées {r['4a_mediane_croisees_hz']:.2f} Hz [m/2, 2m] = "
              f"[{r['4b_mediane_paires_pytorch_hz'] / 2:.2f}, {2 * r['4b_mediane_paires_pytorch_hz']:.2f}] "
              f"{ok(r['4a_retenu_ok'])} | écart aux données {r['4b_ecart_z']:+.1f} écart-type "
              f"{ok(r['4b_ecart_retenu_ok'])} | LNCC {r['4b_lncc_z']:+.1f} écart-type", flush=True)
    if args.sortie:
        Path(args.sortie).write_text(json.dumps(tout, indent=1))


if __name__ == "__main__":
    main()
