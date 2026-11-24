"""Valeurs attendues des entrées de l'étape 1 depuis les fichiers ISMRMRD (P6, docs/ULF.md), calculées par le code de fmi
(`utils_io.load_pair`, `denoise_joint.Context`, référence `cb7174b`).

    cd <fmi>/notebooks/francois && PYTHONPATH=.:../../src uv run python <medoxide>/scripts/cas_entrees.py \\
        synthetique <dossier>
    cd <fmi>/notebooks/francois && PYTHONPATH=.:../../src uv run python <medoxide>/scripts/cas_entrees.py \\
        reel <dossier des fichiers .h5> <dossier P1 contenant <sujet>/<plan>/inputs.npz> <sortie>

`synthetique` écrit dans <dossier> une petite paire `reg.h5` / `inv.h5` (écrivain de fmi, inv retourné le long de la
lecture) et `attendu.h5`, tout calculé par fmi. `reel` écrit `<sortie>/<sujet>_<plan>.h5` pour les six plans de P1 : les
valeurs des `inputs.npz` (k-space fenêtré, temps, bornes, espacement, masque) et les positions d'écho et la demi-largeur
recalculées par `utils_io` sur les fichiers du catalogue `mo_data` (noms en attributs).

Contenu d'un `attendu.h5` : `k_reg`, `k_inv` (complex128, fenêtrés), `temps` (le long de la lecture), `bornes_b0`,
`espacement_mm`, `masque` (uint8) ; attributs `fichier_reg`, `fichier_inv`, `echo_reg`, `echo_inv`, `demi_largeur`,
`taper`.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import utils_io
from utils_cpr import b0_bounds_hz, correct_image

XML = """<?xml version="1.0" encoding="ascii"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
 <experimentalConditions><H1resonanceFrequency_Hz>2063802</H1resonanceFrequency_Hz></experimentalConditions>
 <encoding>
  <encodedSpace><matrixSize><x>{n}</x><y>{n1}</y><z>{n2}</z></matrixSize>
   <fieldOfView_mm><x>230.0</x><y>115.0</y><z>91.7</z></fieldOfView_mm></encodedSpace>
  <reconSpace><matrixSize><x>{n}</x><y>{n1}</y><z>{n2}</z></matrixSize>
   <fieldOfView_mm><x>230.0</x><y>115.0</y><z>91.7</z></fieldOfView_mm></reconSpace>
  <encodingLimits>
   <kspace_encoding_step_1><minimum>0</minimum><maximum>{m1}</maximum><center>{c1}</center></kspace_encoding_step_1>
   <kspace_encoding_step_2><minimum>0</minimum><maximum>{m2}</maximum><center>{c2}</center></kspace_encoding_step_2>
  </encodingLimits>
  <trajectory>cartesian</trajectory>
 </encoding>
 <sequenceParameters><TR>500.0</TR><TE>14.0</TE></sequenceParameters>
 <userParameters>
  <userParameterDouble><name>dwellTime</name><value>0.0408</value></userParameterDouble>
  <userParameterDouble><name>echoShift</name><value>0.0</value></userParameterDouble>
  <userParameterString><name>plane</name><value>yz</value></userParameterString>
  <userParameterString><name>invert_ro_polarity</name><value>{inv}</value></userParameterString>
 </userParameters>
</ismrmrdHeader>
"""


def calculer(reg, inv):
    """Entrées de la méthode par le code de fmi (mêmes appels que mo_joint.context et denoise_joint.Context)."""
    k_reg, k_inv, t, dwell, fov, win = utils_io.load_pair(str(reg), str(inv), "cpu")
    espacement = (1e3 * fov / torch.tensor(k_reg.shape)).tolist()
    bornes = b0_bounds_hz(t, dwell)
    zero = torch.zeros(k_reg.shape, dtype=torch.float64)
    moyenne = (correct_image(k_reg, t, zero).abs() + correct_image(k_inv, t, zero).abs()) / 2
    masque = moyenne > 0.2 * moyenne.max()
    return {"k_reg": k_reg.numpy(), "k_inv": k_inv.numpy(), "temps": t[:, 0, 0].numpy(),
            "bornes_b0": np.array(bornes), "espacement_mm": np.array(espacement),
            "masque": masque.numpy().astype(np.uint8)}, win


def ecrire(chemin, valeurs, win, reg, inv):
    with h5py.File(chemin, "w") as f:
        for n, v in valeurs.items():
            f.create_dataset(n, data=v)
        f.attrs["fichier_reg"], f.attrs["fichier_inv"] = Path(reg).name, Path(inv).name
        f.attrs["echo_reg"], f.attrs["echo_inv"] = win["echo_reg"], win["echo_inv"]
        f.attrs["demi_largeur"], f.attrs["taper"] = win["half_width"], win["taper"]


def synthetique(dossier):
    from fmi.io import writers

    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    n, n1, n2 = 32, 6, 5
    rng = np.random.default_rng(6)
    s = np.arange(n)[:, None, None]
    # écho gaussien décentré (reg vers le début, inv vers la fin avant retournement), objet aléatoire
    objet = rng.normal(size=(1, n1, n2)) + 1j * rng.normal(size=(1, n1, n2)) + 3
    bruit = lambda: 0.02 * (rng.normal(size=(n, n1, n2)) + 1j * rng.normal(size=(n, n1, n2)))  # noqa: E731
    k_reg = (np.exp(-0.5 * ((s - 13.3) / 3.0) ** 2) * objet + bruit()).astype(np.complex64)
    k_inv = (np.exp(-0.5 * ((s - 12.6) / 3.2) ** 2) * objet + bruit()).astype(np.complex64)
    k_reg[:, 0, 0] = k_inv[:, 0, 0] = 0  # ligne non acquise
    for nom, k, sens in (("reg.h5", k_reg, "False"), ("inv.h5", k_inv, "True")):
        (dossier / nom).unlink(missing_ok=True)
        xml = XML.format(n=n, n1=n1, n2=n2, m1=n1 - 1, m2=n2 - 1, c1=n1 // 2, c2=n2 // 2, inv=sens)
        writers.write_hdf5_file(dossier, nom, k, xml)
    valeurs, win = calculer(dossier / "reg.h5", dossier / "inv.h5")
    ecrire(dossier / "attendu.h5", valeurs, win, "reg.h5", "inv.h5")
    print(win)


def reel(fichiers, p1, sortie):
    import mo_data  # noqa: F401  (catalogue ; le sujet ne sert qu'aux noms)

    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    for sujet, cat in mo_data.CATALOG.items():
        for plan, (r, i) in mo_data.PAIRS.items():
            reg, inv = Path(fichiers) / cat[r], Path(fichiers) / cat[i]
            valeurs, win = calculer(reg, inv)
            e = np.load(Path(p1) / sujet / plan / "inputs.npz")
            ecarts = {n: float(np.abs(valeurs[n] - e[m]).max()) for n, m in
                      (("k_reg", "k_reg"), ("k_inv", "k_inv"), ("bornes_b0", "b0_bounds"),
                       ("espacement_mm", "spacing_mm"))}
            ecarts["temps"] = float(np.abs(valeurs["temps"] - e["time_array"][:, 0, 0]).max())
            ecarts["masque"] = int((valeurs["masque"] != e["mask"]).sum())
            print(sujet, plan, win, "recalcul − inputs.npz :", ecarts, flush=True)
            # la référence est l'inputs.npz de P1 ; les positions d'écho viennent du recalcul
            ref = {"k_reg": e["k_reg"], "k_inv": e["k_inv"], "temps": e["time_array"][:, 0, 0],
                   "bornes_b0": e["b0_bounds"], "espacement_mm": e["spacing_mm"],
                   "masque": e["mask"].astype(np.uint8)}
            ecrire(sortie / f"{sujet}_{plan}.h5", ref, win, reg, inv)


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd() / "super_resolution"))
    {"synthetique": synthetique, "reel": reel}[sys.argv[1]](*sys.argv[2:])
