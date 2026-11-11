#!/usr/bin/env python3
"""E45 (F4.0.6) : contrôle de l'autodiff à ε = 0, pour séparer le passage des entiers à `float32` de l'arrondi de 1 + ε.

`--racine … --sujet … --eps0 DIR --autodiff DIR --analytic DIR --controles DIR1 DIR2 DIR3 --medx …` : pour un sujet, part des coupes à plus de
0,1 mm (resp. 1 mm), médiane de l'écart de pose, et Dice moyen (12 étiquettes de Q3) de l'exécution à ε = 0 contre l'autodiff d'origine, contre
chacun des trois contrôles à 1 + ε et contre l'analytique. Aucun critère nouveau : ce sont les mesures de (c)(1′) et (c)(4′), à une référence près.
Une ligne TSV par sujet (--sortie). Aucun identifiant n'est écrit dans le code.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradient_pose_controle as ct  # noqa: E402
import gradient_pose_equivalence as eq  # noqa: E402


def comparer(racine, sujet, eps0, autres, medx):
    """`{nom: (bilan, dice)}` de `eps0` contre chacun des dossiers de `autres` (`{nom: dossier}`)."""
    pixels = eq.pixels_des_stacks(racine, sujet, 250)
    poses0 = eq.lire_poses(f"{eps0}/poses.tsv")
    res = {}
    for nom, d in autres.items():
        b = eq.bilan(eq.deplacements(poses0, eq.lire_poses(f"{d}/poses.tsv"), pixels).values())
        res[nom] = (b, ct.dice_moyen(medx, eps0, d))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("racine", "sujet", "eps0", "autodiff", "analytic", "medx"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--controles", nargs=3, required=True)
    ap.add_argument("--sortie")
    a = ap.parse_args()
    autres = {"autodiff": a.autodiff, "analytique": a.analytic, **{f"controle{i + 1}": d for i, d in enumerate(a.controles)}}
    res = comparer(a.racine, a.sujet, a.eps0, autres, a.medx)
    # b = (n, médiane, maximum, part > 0,1 mm, part > 1 mm, _)
    champs = [f"{n}={b[1]:.4f}:{b[2]:.4f}:{b[3]:.4f}:{b[4]:.4f}:{d:.5f}" for n, (b, d) in res.items()]
    ligne = a.sujet + "\tmediane:max:part0,1:part1:Dice\t" + "\t".join(champs)
    print(ligne)
    if a.sortie:
        with open(a.sortie, "a") as f:
            f.write(ligne + "\n")


if __name__ == "__main__":
    main()
