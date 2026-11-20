"""Petits fichiers de test synthétiques pour medoxide-ismrmrd (docs/ULF.md, P3), écrits par les écrivains Python d'origine.

    # environnement de Spinloop (ismrmrd, spinloop_recon) :
    cd ~/Exp/spinloop && uv run python <medoxide>/scripts/ecrire_fixtures_ismrmrd.py spinloop <dossier>
    # environnement de fridge-magnet-imaging (fmi) :
    cd ~/Exp/fridge-magnet-imaging && uv run python <medoxide>/scripts/ecrire_fixtures_ismrmrd.py fmi <dossier>

Chaque commande écrit `<format>.h5` et `<format>_attendu.json` : les valeurs que le lecteur Rust doit retrouver,
calculées par le code Python de référence (Spinloop : `spinloop_recon.mrd.read` puis `spinloop_recon.cartesian.grid` ;
fmi : relecture par h5py, compte des lignes = `np.any(k != 0, axis=0)`). Aucune donnée réelle : volumes de 8 × 4 × 3
échantillons, valeurs déterministes. Le k-space attendu est rangé `[lecture, phase 1, phase 2]`, aplati dans l'ordre C,
en parties réelles et imaginaires séparées.
"""

import json
import sys
from pathlib import Path

import numpy as np

N_LECTURE, N1, N2 = 8, 4, 3


def valeurs(i1, i2, train=0):
    """Échantillons déterministes d'une ligne (complex64)."""
    s = np.arange(N_LECTURE)
    return ((1 + s + 10 * i1 + 100 * i2 + 1000 * train) + 1j * (0.5 - s * i1 + 0.25 * i2)).astype(np.complex64)


def attendu(k, compte, extra):
    return {"kspace_re": k.real.ravel().tolist(), "kspace_im": k.imag.ravel().tolist(),
            "forme": list(k.shape), "lectures_par_ligne": compte.ravel().tolist(), **extra}


def spinloop(dossier):
    from spinloop_recon import cartesian, mrd

    centre = (2, 1)
    lectures = []
    for i1 in range(N1):
        for i2 in range(N2):
            if (i1, i2) == (0, 0):  # ligne non acquise (Fourier partiel)
                continue
            lectures.append(mrd.Readout(k1=i1 - centre[0], k2=i2 - centre[1], train=i1, echo=i2,
                                        data=valeurs(i1, i2)))
    # une ligne répétée (moyennée) et un train d'équilibre (ignoré)
    lectures.append(mrd.Readout(k1=1 - centre[0], k2=2 - centre[1], train=7, echo=0, data=valeurs(1, 2, train=7)))
    lectures.append(mrd.Readout(k1=0, k2=0, train=9, echo=0, data=valeurs(3, 3, train=9) * 50, dummy=True))
    encodage = mrd.Encoding(matrix=(N_LECTURE, N1, N2), fov_mm=(460.0, 230.0, 230.0), center=centre,
                            axes=("y", "z", "x"))
    chemin = dossier / "spinloop.h5"
    chemin.unlink(missing_ok=True)
    mrd.write(chemin, lectures, encoding=encodage, dwell_us=25.0, resonance_hz=2071599.0,
              user={"read_sign": "-1", "phase_sign": "1"})
    entete, relues = mrd.read(chemin)
    grille, compte = cartesian.grid(relues, (N_LECTURE, N1, N2), centre)  # [k1, k2, échantillon]
    k = np.moveaxis(grille, 2, 0)  # [lecture, phase 1, phase 2]
    params = {p.name: p.value for p in entete.userParameters.userParameterString}
    return chemin, attendu(k, compte, {
        "format": "Spinloop", "matrice": [N_LECTURE, N1, N2], "champ_de_vue_mm": [460.0, 230.0, 230.0],
        "centre": list(centre), "porteuse_hz": float(entete.experimentalConditions.H1resonanceFrequency_Hz),
        "dwell_us": float(params["dwell_us"]), "polarite_lecture": "Inversee", "axes": params["axes"],
        "plan": None, "lecture_retournee_a_l_ecriture": False})


XML_FMI = """<?xml version="1.0" encoding="ascii"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
 <experimentalConditions><H1resonanceFrequency_Hz>2063802</H1resonanceFrequency_Hz></experimentalConditions>
 <encoding>
  <encodedSpace><matrixSize><x>8</x><y>4</y><z>3</z></matrixSize>
   <fieldOfView_mm><x>230.0</x><y>230.0</y><z>230.0</z></fieldOfView_mm></encodedSpace>
  <reconSpace><matrixSize><x>8</x><y>4</y><z>3</z></matrixSize>
   <fieldOfView_mm><x>230.0</x><y>230.0</y><z>230.0</z></fieldOfView_mm></reconSpace>
  <encodingLimits>
   <kspace_encoding_step_1><minimum>0</minimum><maximum>3</maximum><center>2</center></kspace_encoding_step_1>
   <kspace_encoding_step_2><minimum>0</minimum><maximum>2</maximum><center>1</center></kspace_encoding_step_2>
  </encodingLimits>
  <trajectory>cartesian</trajectory>
 </encoding>
 <sequenceParameters><TR>500.0</TR><TE>14.0</TE></sequenceParameters>
 <userParameters>
  <userParameterDouble><name>dwellTime</name><value>0.04</value></userParameterDouble>
  <userParameterDouble><name>echoShift</name><value>0.2</value></userParameterDouble>
  <userParameterString><name>plane</name><value>xy</value></userParameterString>
  <userParameterString><name>invert_ro_polarity</name><value>True</value></userParameterString>
 </userParameters>
</ismrmrdHeader>
"""


def fmi(dossier):
    import h5py

    from fmi.io import writers

    k = np.stack([np.stack([valeurs(i1, i2) for i2 in range(N2)], axis=-1) for i1 in range(N1)], axis=1)
    k[:, 0, 0] = 0  # ligne non acquise
    k[:, 3, 2] = 0
    chemin = dossier / "fmi.h5"
    chemin.unlink(missing_ok=True)
    writers.write_hdf5_file(dossier, chemin.name, k, XML_FMI)  # lecture inversée : retournée le long de la lecture
    with h5py.File(chemin) as f:
        relu = f["kspace"][()]
    compte = np.any(relu != 0, axis=0).astype(int)
    return chemin, attendu(relu, compte, {
        "format": "Fmi", "matrice": [N_LECTURE, N1, N2], "champ_de_vue_mm": [230.0, 230.0, 230.0], "centre": [2, 1],
        "porteuse_hz": 2063802.0, "dwell_us": 40.0, "polarite_lecture": "Inversee", "axes": None, "plan": "xy",
        "lecture_retournee_a_l_ecriture": True, "tr_ms": 500.0})


def main():
    format_, dossier = sys.argv[1], Path(sys.argv[2])
    dossier.mkdir(parents=True, exist_ok=True)
    chemin, valeurs_attendues = {"spinloop": spinloop, "fmi": fmi}[format_](dossier)
    (dossier / f"{format_}_attendu.json").write_text(json.dumps(valeurs_attendues, indent=1) + "\n")
    print(chemin, chemin.stat().st_size, "octets")


if __name__ == "__main__":
    main()
