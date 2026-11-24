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

`--dispersion _cuda,_wgpu_g1,_wgpu_g2` juge le critère 5 amendé, point 2 (décision de François du 2026-11-22) : sorties
de Rust à trois graines de Burn contre les graines 0, 1, 2 de PyTorch (float64). Imprimés : RMS de B0 entre graines de
chaque côté, RMS entre les B0 moyens, et pour la LNCC et l'écart aux données la différence des moyennes rapportée à son
écart-type (√(s²_Rust/n + s²_PyTorch/n)) ; jugée tenue à moins de 2 écarts-types, comme dans docs/ULF.md (l'écart aux
données de sub000 xy, à 3,2, y est noté).

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


def z(a, b):
    """Différence des moyennes de a et b rapportée à son écart-type (variances empiriques)."""
    a, b = np.asarray(a), np.asarray(b)
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def dispersion(args, suffixes):
    """Critère 5 amendé, point 2 : B0 entre graines de chaque côté et entre moyennes ; LNCC et écart aux données des
    graines de Rust (recalculés) contre ceux des graines de PyTorch (leurs json), en écarts-types de la différence."""
    tout = {}
    for nom in args.plans.split(","):
        sujet, q = nom.split("_")
        i = np.load(Path(args.p1) / sujet / q / "inputs.npz")
        m = i["mask"]
        k_reg, k_inv = torch.tensor(i["k_reg"]), torch.tensor(i["k_inv"])
        t, mask, esp = torch.tensor(i["time_array"]), torch.tensor(m), i["spacing_mm"].tolist()
        g = Path(args.graines) / sujet / q
        noms_py = ("inr32", "inr32_s1", "inr32_s2")
        py = [np.load(g / f"{n}.npz")["b0_hz"].astype(np.float64) for n in noms_py]
        json_py = [json.loads((g / f"{n}.json").read_text()) for n in noms_py]
        f32 = np.load(g / "inr32_f32.npz")
        sigma = ecart_brut(k_reg, k_inv, t, mask, torch.tensor(f32["b0_hz"]).double(),
                           torch.tensor(f32["u"]).to(torch.complex128)) \
            / json.loads((g / "inr32_f32.json").read_text())["misfit_over_noise"]
        ru, lncc_ru, ecart_ru = [], [], []
        for suf in suffixes:
            with h5py.File(Path(args.rust) / f"{nom}{suf}.h5") as f:
                b0, u = f["b0_hz"][()], f["u"][()]
            ru.append(b0)
            lncc_ru.append(lncc(k_reg, k_inv, t, torch.tensor(b0), mask, esp))
            ecart_ru.append(ecart_brut(k_reg, k_inv, t, mask, torch.tensor(b0), torch.tensor(u)) / sigma)
        lncc_py = [j["crit_lncc"] for j in json_py]
        ecart_py = [j["misfit_over_noise"] for j in json_py]
        paires = lambda xs: [rms(xs[a], xs[b], m) for a in range(len(xs)) for b in range(a + 1, len(xs))]  # noqa: E731
        r = {
            "b0_paires_rust_hz": paires(ru), "b0_paires_pytorch_hz": paires(py),
            "b0_moyennes_hz": rms(sum(ru) / len(ru), sum(py) / len(py), m),
            "lncc_rust": lncc_ru, "lncc_pytorch": lncc_py, "lncc_z": z(lncc_ru, lncc_py),
            "ecart_rust": ecart_ru, "ecart_pytorch": ecart_py, "ecart_z": z(ecart_ru, ecart_py),
        }
        r["lncc_ok"], r["ecart_ok"] = abs(r["lncc_z"]) <= 2, abs(r["ecart_z"]) <= 2
        tout[nom] = r
        ok = lambda b: "ok" if b else "HORS"  # noqa: E731
        print(f"{nom} : B0 entre graines Rust {np.round(r['b0_paires_rust_hz'], 2).tolist()} Hz, PyTorch "
              f"{np.round(r['b0_paires_pytorch_hz'], 2).tolist()} Hz, moyennes {r['b0_moyennes_hz']:.2f} Hz | LNCC "
              f"{r['lncc_z']:+.1f} écart-type {ok(r['lncc_ok'])} | écart aux données {r['ecart_z']:+.1f} écart-type "
              f"({100 * (np.mean(ecart_ru) / np.mean(ecart_py) - 1):+.2f} %) {ok(r['ecart_ok'])}", flush=True)
    return tout


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
        tout = dispersion(args, args.dispersion.split(","))
        if args.sortie:
            Path(args.sortie).write_text(json.dumps(tout, indent=1))
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
