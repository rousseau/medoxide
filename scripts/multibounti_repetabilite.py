"""Répétabilité du recalage affine de MIRTK (`register -model Affine`) pour Multi-BOUNTI : même images, départs perturbés (voir `docs/MULTIBOUNTI.md`, sous-étape M4).

Pour chaque volume de référence (`data/multibounti/ref/<id>/`, phase C faite), le recalage officiel est relancé `K` fois dans l'image ARM de MIRTK, avec un départ tiré autour de la transformation de départ officielle (`aff-i.dof`) : translation ±5 mm, rotation ±5°, échelles ±5 %,
graine fixe. Les matrices 4 × 4 obtenues sont écrites dans `data/multibounti/essais/repet/<id>/essai-<k>.mat` (format `aladin`), avec `officiel.mat`.
La dispersion est ensuite mesurée par `scripts/multibounti_repetabilite_mesure.py`.

Usage : python scripts/multibounti_repetabilite.py <id> [K]
"""

import os
import random
import struct
import subprocess
import sys

RACINE = os.getcwd()
IMAGE_ARM = "fetalsvrtk/svrtk:general_auto_arm"
M = "/bin/MIRTK/build/lib/tools"


def parametres_de(dof):
    """Les 12 paramètres (tx ty tz rx ry rz sx sy sz sxy syz sxz) d'un .dof rigide ou affine."""
    octets = open(dof, "rb").read()
    type_, nombre = struct.unpack(">II", octets[4:12])
    valeurs = list(struct.unpack(f">{nombre}d", octets[12 : 12 + 8 * nombre]))
    return valeurs + [100.0] * 3 + [0.0] * 3 if nombre == 6 else valeurs


def main():
    ident = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ref = f"{RACINE}/data/multibounti/ref/{ident}"
    sortie = f"{RACINE}/data/multibounti/essais/repet/{ident}"
    os.makedirs(sortie, exist_ok=True)
    depart = parametres_de(f"{ref}/aff-i.dof")
    hasard = random.Random(20261021)
    lignes = ["set -e", "cd /work"]
    for i in range(k):
        p = list(depart)
        for j in range(3):
            p[j] += hasard.uniform(-5, 5)
            p[3 + j] += hasard.uniform(-5, 5)
            p[6 + j] += hasard.uniform(-5, 5)
        options = " ".join(f"-{n} {v:.6f}" for n, v in zip(["tx", "ty", "tz", "rx", "ry", "rz", "sx", "sy", "sz", "sxy", "syz", "sxz"], p))
        lignes += [
            f"{M}/init-dof /out/depart-{i}.dof -affine {options}",
            f"{M}/register tmp-lr-ref.nii.gz lr-lab-pad-masked-crop-t2-128.nii.gz -model Affine -dofin /out/depart-{i}.dof -dofout /out/essai-{i}.dof -v 0 > /dev/null",
            f"{M}/convert-dof /out/essai-{i}.dof /out/essai-{i}.mat -output-format aladin > /dev/null",
        ]
    lignes += [f"{M}/convert-dof aff-d.dof /out/officiel.mat -output-format aladin > /dev/null"]
    commande = ["docker", "run", "--rm", "--platform", "linux/arm64", "-v", f"{ref}:/work", "-v", f"{sortie}:/out", "--entrypoint", "bash", IMAGE_ARM, "-c", "\n".join(lignes)]
    subprocess.run(commande, check=True)
    print(f"{ident} : {k} essais dans {sortie}")


if __name__ == "__main__":
    main()
