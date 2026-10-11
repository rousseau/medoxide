"""Écart d'une coupe à la médiane locale des poses de ses voisines (même stack) : calibration d'un seuil de rejet des poses aberrantes.

Pour chaque stack, les coupes recalées (pose différente de l'identité) sont ordonnées par indice ; pour la coupe `k`, la pose « locale » est la médiane, sur les coupes recalées dont l'indice est dans
[k − w, k + w] (la coupe elle-même exclue), du déplacement du centre de la coupe (médiane par composante) et du vecteur de rotation (médiane par composante). L'écart de la coupe est : la distance de son
déplacement du centre à la médiane locale (mm) et l'angle de sa rotation relative à la rotation locale (°). Un mouvement fœtal plausible est cohérent entre coupes voisines (écart petit) ; une pose qui dérive ou
échoue ne l'est pas.

Les seuils se calibrent sur des sujets dont les poses sont jugées plausibles (percentiles des écarts) et s'évaluent sur les autres : la part de coupes au-dessus du seuil.

Usage : python scripts/calibrate_pose_outliers.py --jeu nom=poses.tsv:stack0:stack1:stack2 [--jeu ...] [--fenetre 2]
"""

import argparse

import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation

from characterize_motion import lire_poses


def ecarts_locaux(poses, chemins_stacks, fenetre=2):
    """Liste des (stack, coupe, écart de déplacement en mm, écart de rotation en °) des coupes recalées ayant au moins 3 voisines recalées dans la fenêtre."""
    serie = {}
    for (s, k), m in poses.items():
        if not np.allclose(m, np.eye(4), atol=1e-12):
            serie.setdefault(s, {})[k] = m
    sortie = []
    for s, coupes in serie.items():
        img = nib.load(chemins_stacks[s])
        nx, ny = img.shape[:2]
        dep, rot = {}, {}
        for k, m in coupes.items():
            c = img.affine @ np.array([(nx - 1) / 2, (ny - 1) / 2, k, 1.0])
            dep[k] = (m @ c - c)[:3]
            rot[k] = Rotation.from_matrix(m[:3, :3])
        for k in sorted(coupes):
            voisines = [j for j in coupes if j != k and abs(j - k) <= fenetre]
            if len(voisines) < 3:
                continue
            med_dep = np.median(np.array([dep[j] for j in voisines]), axis=0)
            med_rot = Rotation.from_rotvec(np.median(np.array([rot[j].as_rotvec() for j in voisines]), axis=0))
            sortie.append((s, k, float(np.linalg.norm(dep[k] - med_dep)), float(np.degrees((rot[k] * med_rot.inv()).magnitude()))))
    return sortie


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jeu", action="append", required=True, help="nom=poses.tsv:stack0:stack1:stack2")
    parser.add_argument("--fenetre", type=int, default=3)
    args = parser.parse_args()
    for j in args.jeu:
        nom, reste = j.split("=", 1)
        morceaux = reste.split(":")
        e = np.array([(d, r) for _, _, d, r in ecarts_locaux(lire_poses(morceaux[0]), morceaux[1:], args.fenetre)])
        p = lambda v, q: float(np.percentile(v, q))  # noqa: E731
        print(f"{nom:34} {len(e):4d} coupes | écart de déplacement : méd {p(e[:,0],50):5.2f} p90 {p(e[:,0],90):6.2f} p95 {p(e[:,0],95):6.2f} p99 {p(e[:,0],99):6.2f} mm | écart de rotation : méd {p(e[:,1],50):5.2f} p90 {p(e[:,1],90):6.2f} p95 {p(e[:,1],95):6.2f} p99 {p(e[:,1],99):6.2f}°")
