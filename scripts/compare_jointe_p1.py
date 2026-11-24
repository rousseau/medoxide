"""Critère 5 de P4 bis (docs/ULF.md) : estimation jointe de medoxide-ulf de bout en bout, comparée aux graines de la
référence PyTorch sur les six plans de P1.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # compare_methods.py et utils_cpr.py de fmi (cb7174b)
    PYTHONPATH=. uv run python <medoxide>/scripts/compare_jointe_p1.py --rust <dossier> [--suffixe _cuda] \\
        --graines <dossier> --p1 <dossier> [--sortie resultats.json]

- `--rust` : sorties de l'exemple `jointe_p1` (`<sujet>_<plan><suffixe>.h5` : `b0_hz`, `u`) ;
- `--graines` : `<sujet>/<plan>/inr32{,_s1,_s2,_f32}.{npz,json}` de b0_bases_lisses.py (garage:spinloop/references/
  1120-230826-…-p4-graines-inr et 1122-004303-…-p4-float32) ;
- `--p1` : `<sujet>/<plan>/inputs.npz` de P1 (k-space, temps, masque de l'objet, espacement).

Critères (écrits avant le code, docs/ULF.md) :
- B0 : RMS dans l'objet de b0_rust − b0 de la graine 0 (float64) ≤ plus grand RMS entre deux des graines 0, 1, 2 ;
- LNCC reg / inv après correction et écart aux données rapporté au bruit dans l'enveloppe (min, max) des graines 0, 1,
  2 et du calcul float32 de PyTorch.

Les métriques sont celles de b0_sensitivity.metrics (pair_criteria de compare_methods ; écart aux données en magnitude,
dans l'objet, rapporté au bruit σ). σ n'est pas dans les fichiers : il est tiré du calcul float32 de PyTorch, dont u et b0
sont enregistrés (σ = écart brut / écart rapporté de son json). Contrôles : la LNCC recalculée de ce calcul redonne son
json, et σ redonne l'écart aux données de la sortie P1 (graine 0, `outputs.npz`) à la précision de son stockage.
"""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

import compare_methods as cm
import utils_cpr
from plans_ulf import option_plans



def ecart_brut(k_reg, k_inv, t, mask, b0, u):
    res = 0.0
    with torch.no_grad():
        for k, s in ((k_reg, 1), (k_inv, -1)):
            pred = utils_cpr.ifft(utils_cpr.predict_kspace(u, t, s * b0), normalize=True).abs()
            obs = utils_cpr.ifft(k, normalize=True).abs()
            res += (pred - obs)[mask].square().mean().item()
    return math.sqrt(res / 2)


def lncc(k_reg, k_inv, t, b0, mask, espacement):
    return cm.pair_criteria(k_reg, k_inv, t, b0, mask, espacement)["lncc"]


def rms(a, b, mask):
    return float(np.sqrt(np.mean((a - b)[mask] ** 2)))


def plan(args, sujet, nom_plan):
    i = np.load(Path(args.p1) / sujet / nom_plan / "inputs.npz")
    k_reg, k_inv = torch.tensor(i["k_reg"]), torch.tensor(i["k_inv"])
    t, mask, esp = torch.tensor(i["time_array"]), torch.tensor(i["mask"]), i["spacing_mm"].tolist()
    m = i["mask"]
    g = Path(args.graines) / sujet / nom_plan
    noms = ("inr32", "inr32_s1", "inr32_s2", "inr32_f32")
    ref = {n: json.loads((g / f"{n}.json").read_text()) for n in noms}
    b0s = {n: np.load(g / f"{n}.npz")["b0_hz"].astype(np.float64) for n in noms}

    # σ et contrôles, sur le calcul float32 de PyTorch
    f32 = np.load(g / "inr32_f32.npz")
    b0_f32, u_f32 = torch.tensor(f32["b0_hz"]).double(), torch.tensor(f32["u"]).to(torch.complex128)
    sigma = ecart_brut(k_reg, k_inv, t, mask, b0_f32, u_f32) / ref["inr32_f32"]["misfit_over_noise"]
    controle_lncc = lncc(k_reg, k_inv, t, b0_f32, mask, esp) - ref["inr32_f32"]["crit_lncc"]
    o = np.load(Path(args.p1) / sujet / nom_plan / "outputs.npz")
    p1_json = json.loads((Path(args.p1) / sujet / nom_plan / "metrics.json").read_text()) \
        if (Path(args.p1) / sujet / nom_plan / "metrics.json").exists() else None
    controle_sigma = None
    if p1_json is not None:
        e = ecart_brut(k_reg, k_inv, t, mask, torch.tensor(o["b0_hz"]).double(),
                       torch.tensor(o["u"]).to(torch.complex128)) / sigma
        controle_sigma = e - p1_json["misfit_over_noise"]

    # Rust
    with h5py.File(Path(args.rust) / f"{sujet}_{nom_plan}{args.suffixe}.h5") as f:
        b0_r, u_r = f["b0_hz"][()], f["u"][()]
        secondes = float(f.attrs["secondes"])
    b0_rt, u_rt = torch.tensor(b0_r), torch.tensor(u_r)
    res = {
        "b0_rms_hz": rms(b0_r, b0s["inr32"], m),
        "b0_seuil_hz": max(rms(b0s[a], b0s[b], m) for a, b in
                           (("inr32", "inr32_s1"), ("inr32", "inr32_s2"), ("inr32_s1", "inr32_s2"))),
        "b0_rms_pytorch_f32_hz": rms(b0s["inr32_f32"], b0s["inr32"], m),
        "lncc": lncc(k_reg, k_inv, t, b0_rt, mask, esp),
        "lncc_enveloppe": [min(r["crit_lncc"] for r in ref.values()), max(r["crit_lncc"] for r in ref.values())],
        "ecart": ecart_brut(k_reg, k_inv, t, mask, b0_rt, u_rt) / sigma,
        "ecart_enveloppe": [min(r["misfit_over_noise"] for r in ref.values()),
                            max(r["misfit_over_noise"] for r in ref.values())],
        "secondes": secondes,
        "controle_lncc_f32": controle_lncc,
        "controle_sigma_p1": controle_sigma,
    }
    res["b0_ok"] = res["b0_rms_hz"] <= res["b0_seuil_hz"]
    res["lncc_ok"] = res["lncc_enveloppe"][0] <= res["lncc"] <= res["lncc_enveloppe"][1]
    res["ecart_ok"] = res["ecart_enveloppe"][0] <= res["ecart"] <= res["ecart_enveloppe"][1]
    return res


def dispersion(args, suffixes):
    """B0 : RMS dans l'objet entre sorties de Rust (graines) deux à deux, et entre graines de PyTorch ; moyenne des
    sorties de Rust à la graine 0 de PyTorch (écart systématique éventuel), rapportée à celle d'une graine."""
    for nom in args.plans.split(","):
        sujet, q = nom.split("_")
        m = np.load(Path(args.p1) / sujet / q / "inputs.npz")["mask"]
        g = Path(args.graines) / sujet / q
        py = [np.load(g / f"{n}.npz")["b0_hz"].astype(np.float64) for n in ("inr32", "inr32_s1", "inr32_s2")]
        ru = []
        for suf in suffixes:
            with h5py.File(Path(args.rust) / f"{nom}{suf}.h5") as f:
                ru.append(f["b0_hz"][()])
        paires = lambda xs: [rms(xs[i], xs[j], m) for i in range(len(xs)) for j in range(i + 1, len(xs))]  # noqa: E731
        moy_ru, moy_py = sum(ru) / len(ru), sum(py) / len(py)
        print(f"{nom} : entre graines Rust {np.round(paires(ru), 2).tolist()} Hz, PyTorch {np.round(paires(py), 2).tolist()}"
              f" Hz | moyenne Rust − moyenne PyTorch {rms(moy_ru, moy_py, m):.2f} Hz", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rust", required=True)
    p.add_argument("--suffixe", default="")
    p.add_argument("--graines", required=True)
    p.add_argument("--p1", required=True)
    option_plans(p)
    p.add_argument("--sortie")
    p.add_argument("--dispersion", help="suffixes de sorties Rust séparés par des virgules (graines)")
    args = p.parse_args()
    if args.dispersion:
        dispersion(args, args.dispersion.split(","))
        return
    tout = {}
    for nom in args.plans.split(","):
        sujet, q = nom.split("_")
        r = plan(args, sujet, q)
        tout[nom] = r
        print(f"{nom} : B0 {r['b0_rms_hz']:.2f} Hz (seuil {r['b0_seuil_hz']:.2f}, PyTorch f32 "
              f"{r['b0_rms_pytorch_f32_hz']:.2f}) {'ok' if r['b0_ok'] else 'HORS'} | LNCC {r['lncc']:.4f} "
              f"[{r['lncc_enveloppe'][0]:.4f}, {r['lncc_enveloppe'][1]:.4f}] {'ok' if r['lncc_ok'] else 'HORS'} | "
              f"écart {r['ecart']:.4f} [{r['ecart_enveloppe'][0]:.4f}, {r['ecart_enveloppe'][1]:.4f}] "
              f"{'ok' if r['ecart_ok'] else 'HORS'} | {r['secondes']:.0f} s | contrôles LNCC {r['controle_lncc_f32']:.1e}"
              f", σ {r['controle_sigma_p1'] if r['controle_sigma_p1'] is None else format(r['controle_sigma_p1'], '.1e')}",
              flush=True)
    if args.sortie:
        Path(args.sortie).write_text(json.dumps(tout, indent=1))


if __name__ == "__main__":
    main()
