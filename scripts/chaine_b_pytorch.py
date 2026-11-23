"""Critère 4 de P5, référence chaînée (docs/ULF.md, décision du 2026-11-23) : entrées de P1 dont le `b0_warm_start` est
remplacé par le B0 de la méthode B de PyTorch `f32` d'une graine, pour que `b0_bases_lisses.py` (fmi) fasse la jointe
à partir de ce B0.

    python chaine_b_pytorch.py --p1 <dossier contenant stage1/> --graines-b <dossier> --graine g --sortie <dossier>

- `--p1` : `stage1/<sujet>/<plan>/{inputs,outputs}.npz` de P1 ;
- `--graines-b` : `<sujet>/<plan>/b_f32.npz` (graine 0) ou `b_f32_s<g>.npz` de `graines_b.py` ;
- écrit `<sortie>/stage1/<sujet>/<plan>/inputs.npz` (copie, `b0_warm_start` remplacé, même type) et `outputs.npz`
  (copié tel quel), à passer en `--p1` à `b0_bases_lisses.py`.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p1", required=True)
    p.add_argument("--graines-b", required=True)
    p.add_argument("--graine", type=int, required=True)
    p.add_argument("--sortie", required=True)
    args = p.parse_args()
    nom = "b_f32" if args.graine == 0 else f"b_f32_s{args.graine}"
    for d in sorted((Path(args.p1) / "stage1").glob("*/*")):
        sujet, plan = d.parent.name, d.name
        i = dict(np.load(d / "inputs.npz"))
        b = np.load(Path(args.graines_b) / sujet / plan / f"{nom}.npz")["b0_hz"]
        assert b.shape == i["b0_warm_start"].shape, (b.shape, i["b0_warm_start"].shape)
        i["b0_warm_start"] = b.astype(i["b0_warm_start"].dtype)
        o = Path(args.sortie) / "stage1" / sujet / plan
        o.mkdir(parents=True, exist_ok=True)
        np.savez(o / "inputs.npz", **i)
        if (d / "outputs.npz").exists():
            shutil.copy(d / "outputs.npz", o / "outputs.npz")
        print(sujet, plan, nom, flush=True)


if __name__ == "__main__":
    main()
