"""Cas de test du modèle direct et de l'adjoint de medoxide-ulf (docs/ULF.md, P4), en HDF5 lisible par le test Rust.

    cd ~/Exp/fridge-magnet-imaging   # environnement de fmi (fmi.reconstruction.operators.cpr)
    uv run python <medoxide>/scripts/cas_cpr.py synthetique <medoxide>/crates/medoxide-ulf/tests/donnees/cpr_synthetique.h5
    uv run python <medoxide>/scripts/cas_cpr.py p1 <fixtures.npz> <sortie.h5>   # un plan de P1 (garage:spinloop/references)

Chaque fichier contient, en complex128 / float64 : `x` (image), `b0` (Hz), `temps` (s, un par échantillon de lecture),
`k` (k-space), `y` = A(b0) x et `z` = Aᴴ(b0) k calculés par fmi (`time_segmented_cpr_adjoint` et `time_segmented_cpr`,
lecture le long de l'axe 0, `t_shift` = 0). Le cas synthétique (9 × 6 × 5, valeurs déterministes) est versionné ; les
plans de P1 ne le sont pas (données hors dépôt).
"""

import sys

import h5py
import numpy as np


def ecrire(chemin, **tableaux):
    with h5py.File(chemin, "w") as f:
        for nom, v in tableaux.items():
            f.create_dataset(nom, data=v)


def synthetique(sortie):
    from fmi.reconstruction.operators import cpr

    rng = np.random.default_rng(20261120)
    dims = (9, 6, 5)
    temps = -2.5e-3 + 4.4e-5 * np.arange(dims[0])
    t = np.broadcast_to(temps[:, None, None], dims).copy()
    x = rng.standard_normal(dims) + 1j * rng.standard_normal(dims)
    k = rng.standard_normal(dims) + 1j * rng.standard_normal(dims)
    b0 = 600.0 * rng.standard_normal(dims)
    y = cpr.time_segmented_cpr_adjoint(x, t, b0, 0.0, 0)
    z = cpr.time_segmented_cpr(k, t, b0, 0.0, 0)
    ecrire(sortie, x=x, k=k, b0=b0, temps=temps, y=y, z=z)


def p1(fixtures, sortie):
    f = np.load(fixtures)
    t = f["t"]
    temps = t[:, 0, 0]
    assert np.allclose(t, temps[:, None, None]), "temps non constant le long des axes de phase"
    ecrire(sortie, x=f["x"], k=f["k_reg"], b0=f["b0"], temps=temps, y=f["y"], z=f["z"])


if __name__ == "__main__":
    {"synthetique": synthetique, "p1": p1}[sys.argv[1]](*sys.argv[2:])
