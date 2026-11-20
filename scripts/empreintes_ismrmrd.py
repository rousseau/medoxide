"""Empreintes de fichiers réels pour le test de medoxide-ismrmrd sur données réelles (docs/ULF.md, P3, critère 3).

    # fichiers Spinloop (environnement de Spinloop : ismrmrd, spinloop_recon)
    cd ~/Exp/spinloop && uv run python <medoxide>/scripts/empreintes_ismrmrd.py <sortie.json> <raw.h5>...
    # fichiers de fridge-magnet-imaging (h5py et numpy suffisent)
    cd ~/Exp/fridge-magnet-imaging && uv run python <medoxide>/scripts/empreintes_ismrmrd.py <sortie.json> <f.h5>...

Puis : `MEDOXIDE_ULF_DONNEES=<sortie.json> cargo test -p medoxide-ismrmrd --test donnees_reelles`.

Pour chaque fichier, le code Python de référence calcule le k-space `[lecture, phase 1, phase 2]` (Spinloop :
`spinloop_recon.mrd.read` puis `spinloop_recon.cartesian.grid`, moyenne en complex128 ramenée en complex64 ; fmi :
`h5py`, `f["kspace"][()]`) et le JSON garde son empreinte SHA-256 (octets complex64, ordre C), le compte des lectures par
ligne et les champs d'en-tête ; aucun échantillon n'est copié. Le JSON est ajouté à `<sortie.json>` s'il existe.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np


def empreinte(k):
    return hashlib.sha256(np.ascontiguousarray(k.astype(np.complex64)).tobytes()).hexdigest()


def spinloop(chemin):
    from spinloop_recon import cartesian, mrd

    entete, lectures = mrd.read(chemin)
    enc = entete.encoding[0]
    m = enc.encodedSpace.matrixSize
    lim = enc.encodingLimits
    centre = (lim.kspace_encoding_step_1.center, lim.kspace_encoding_step_2.center)
    grille, compte = cartesian.grid(lectures, (m.x, m.y, m.z), centre)
    k = np.moveaxis(grille, 2, 0)
    params = {p.name: p.value for p in entete.userParameters.userParameterString}
    return {"format": "Spinloop", "matrice": [m.x, m.y, m.z], "kspace_sha256": empreinte(k),
            "lectures_par_ligne": compte.ravel().tolist(), "dwell_us": float(params["dwell_us"]),
            "porteuse_hz": float(entete.experimentalConditions.H1resonanceFrequency_Hz),
            "axes": params.get("axes"), "read_sign": params.get("read_sign")}


def fmi(chemin):
    with h5py.File(chemin) as f:
        k = f["kspace"][()]
        x = f["xml_header"][()]
    x = x.decode() if isinstance(x, bytes) else x
    params = dict(re.findall(r"<name>(.*?)</name>\s*<value>(.*?)</value>", x, re.DOTALL))
    return {"format": "Fmi", "matrice": list(k.shape), "kspace_sha256": empreinte(k),
            "lectures_par_ligne": np.any(k != 0, axis=0).astype(int).ravel().tolist(),
            "dwell_us": float(params["dwellTime"]) * 1e3,
            "porteuse_hz": float(re.findall(r"H1resonanceFrequency_Hz>(.*?)<", x)[0]),
            "plan": params.get("plane"), "invert_ro_polarity": params.get("invert_ro_polarity"),
            "tr_ms": float(re.findall(r"<TR>(.*?)<", x)[0])}


def main():
    sortie = Path(sys.argv[1])
    resultats = json.loads(sortie.read_text()) if sortie.exists() else []
    for nom in sys.argv[2:]:
        chemin = Path(nom).resolve()
        with h5py.File(chemin) as f:
            lire = spinloop if "dataset" in f else fmi
        r = {"chemin": str(chemin), **lire(chemin)}
        resultats = [x for x in resultats if x["chemin"] != r["chemin"]] + [r]
        print(chemin.name, r["format"], r["matrice"], r["kspace_sha256"][:12])
    sortie.write_text(json.dumps(resultats, indent=1) + "\n")


if __name__ == "__main__":
    main()
