"""Critère 4 de P6 (docs/ULF.md) : étape 1 de bout en bout depuis les fichiers ISMRMRD (`ulf_paire --ecrire-entrees`),
comparée à la voie des `inputs.npz` de P1 et aux cinq chaînes B → jointe de Rust du critère 4 de P5.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # compare_methods.py et utils_cpr.py de fmi (cb7174b)
    PYTHONPATH=.:<medoxide>/scripts uv run python <medoxide>/scripts/compare_p6_p1.py --paires <dossier> \\
        --chaines <dossier> --graines <dossier> --p1 <dossier> [--sortie resultats.json]

- `--paires` : `<sujet>_<plan>_paire_cuda.h5` de `ulf_paire` ;
- `--chaines` : `<plan>_chaine_cuda{,_s1..4}.h5` et `critere4.json` (`compare_b4_p1.py`) du critère 4 de P5 ;
- `--graines` : graines de la jointe de P4 bis (`inr32_f32`), pour σ, comme `compare_jointe_p1.py` ;
- `--p1` : `<sujet>/<plan>/inputs.npz` de P1.

Critères : entrées de la méthode égales à celles des `inputs.npz` à l'arrondi `f32` près (≤ 1e-6 relatif) ; B0 final à
moins du plus grand écart entre deux des cinq chaînes de Rust (RMS dans l'objet) de leur moyenne ; LNCC et écart aux
données dans leur [min, max].
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from compare_jointe_p1 import ecart_brut, lncc
from plans_ulf import option_plans



def rms(a, b, m):
    return float(np.sqrt(np.mean((a - b)[m] ** 2)))


def rel_f32(a, r):
    a, r = np.asarray(a), np.asarray(r)
    if np.iscomplexobj(r):
        a, r = a.astype(np.complex64), r.astype(np.complex64)
    else:
        a, r = a.astype(np.float32), r.astype(np.float32)
    return float(np.abs(a.astype(np.complex128) - r).max() / np.abs(r).max())


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

    with h5py.File(Path(args.paires) / f"{nom}_paire_cuda.h5") as f:
        b0, u = f["b0_hz"][()], f["u"][()]
        entrees = {n: f[n][()] for n in ("k_reg", "k_inv", "temps", "espacement_mm", "bornes_b0")}
        masque = f["masque"][()].astype(bool)
        secondes = {n: float(f.attrs[n]) for n in ("secondes", "secondes_methode_b", "secondes_jointe")}
    ecarts = {
        "k_reg": rel_f32(entrees["k_reg"], i["k_reg"]), "k_inv": rel_f32(entrees["k_inv"], i["k_inv"]),
        "temps": rel_f32(entrees["temps"], i["time_array"][:, 0, 0]),
        "espacement_mm": rel_f32(entrees["espacement_mm"], i["spacing_mm"]),
        "bornes_b0": rel_f32(entrees["bornes_b0"], i["b0_bounds"]),
    }
    chaines = [h5py.File(Path(args.chaines) / f"{nom}_chaine_cuda{'' if k == 0 else f'_s{k}'}.h5")["b0_hz"][()]
               for k in range(5)]
    c4 = json.loads((Path(args.chaines) / "critere4.json").read_text())[nom]
    seuil = max(rms(chaines[a], chaines[b], m) for a in range(5) for b in range(a + 1, 5))
    r = {
        "entrees_ecarts": ecarts,
        "masque_identique": bool((masque == m).all()),
        "b0_rms_moyenne_hz": rms(b0, sum(chaines) / 5, m),
        "b0_seuil_hz": seuil,
        "lncc": lncc(k_reg, k_inv, t, torch.tensor(b0), mask, esp),
        "lncc_chaines": c4["4b_lncc_rust"],
        "ecart": ecart_brut(k_reg, k_inv, t, mask, torch.tensor(b0), torch.tensor(u)) / sigma,
        "ecart_chaines": c4["4b_ecart_rust"],
        **secondes,
    }
    r["entrees_ok"] = max(ecarts.values()) <= 1e-6
    r["b0_ok"] = r["b0_rms_moyenne_hz"] <= seuil
    r["lncc_ok"] = min(r["lncc_chaines"]) <= r["lncc"] <= max(r["lncc_chaines"])
    r["ecart_ok"] = min(r["ecart_chaines"]) <= r["ecart"] <= max(r["ecart_chaines"])
    return r


def main():
    p = argparse.ArgumentParser()
    for n in ("--paires", "--chaines", "--graines", "--p1"):
        p.add_argument(n, required=True)
    option_plans(p)
    p.add_argument("--sortie")
    args = p.parse_args()
    tout = {}
    ok = lambda b: "ok" if b else "HORS"  # noqa: E731
    for nom in args.plans.split(","):
        r = plan(args, nom)
        tout[nom] = r
        print(f"{nom} : entrées {max(r['entrees_ecarts'].values()):.1e} {ok(r['entrees_ok'])}, masque "
              f"{ok(r['masque_identique'])} | B0 {r['b0_rms_moyenne_hz']:.2f} Hz de la moyenne (seuil "
              f"{r['b0_seuil_hz']:.2f}) {ok(r['b0_ok'])} | LNCC {r['lncc']:.4f} [{min(r['lncc_chaines']):.4f}, "
              f"{max(r['lncc_chaines']):.4f}] {ok(r['lncc_ok'])} | écart {r['ecart']:.4f} "
              f"[{min(r['ecart_chaines']):.4f}, {max(r['ecart_chaines']):.4f}] {ok(r['ecart_ok'])} | "
              f"{r['secondes']:.0f} s (B {r['secondes_methode_b']:.0f}, jointe {r['secondes_jointe']:.0f})", flush=True)
    if args.sortie:
        Path(args.sortie).write_text(json.dumps(tout, indent=1))


if __name__ == "__main__":
    main()
