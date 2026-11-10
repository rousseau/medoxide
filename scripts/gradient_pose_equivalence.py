#!/usr/bin/env python3
"""Équivalence des poses de deux variantes du recalage par coupe (docs/ROADMAP.md, F4.0.6, critère (c)(1)).

Pour chaque coupe recalée dans les deux fichiers de poses (`stack slice m00 … m33`, ordre colonne, même repère puisque les deux variantes sont le
même binaire), déplacement quadratique moyen entre la pose de `--ref` et celle de `--new` sur des pixels du masque de la coupe (au plus
`--pixels-par-coupe`, tirés régulièrement), en mm. Par sujet : médiane, maximum, part des coupes à plus de 0,1 mm et à plus de 1 mm.
Critère (c)(1) : au plus 5 % des coupes à plus de 0,1 mm et au plus 0,5 % à plus de 1 mm.

Aucun identifiant n'est écrit dans le code : racine, sujet et fichiers sont des arguments.

Usage : python3 -I scripts/gradient_pose_equivalence.py --racine data/svr/jeu_reel_tru_haste --sujet sub-XXXX --ref poses_autodiff.tsv --new poses_analytic.tsv [--sortie coupes.tsv]
"""

import argparse
import os
import re

import nibabel as nib
import numpy as np

SEUIL_FIN_MM = 0.1
SEUIL_GROS_MM = 1.0
PART_FIN_MAX = 0.05
PART_GROS_MAX = 0.005


def lire_poses(chemin):
    """`{(nom de stack, coupe): matrice 4×4}` d'un TSV de poses (colonnes `stack slice m00 m10 … m33`, ordre colonne)."""
    poses = {}
    with open(chemin) as f:
        for ligne in f.read().strip().split("\n"):
            c = ligne.split("\t")
            if c[0] == "stack":
                continue
            poses[(c[0], int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    return poses


def deplacements(poses_a, poses_b, pixels):
    """Déplacement quadratique moyen (mm) de chaque coupe présente dans les deux jeux de poses, `pixels[(stack, coupe)]` donnant les pixels (n × 3, mm).

    Une coupe absente d'un jeu, ou sans pixel, est ignorée. Résultat : `{(stack, coupe): mm}`."""
    sortie = {}
    for cle, ma in poses_a.items():
        mb = poses_b.get(cle)
        x = pixels.get(cle)
        if mb is None or x is None or len(x) == 0:
            continue
        h = np.hstack([x, np.ones((len(x), 1))])
        d = (h @ ma.T)[:, :3] - (h @ mb.T)[:, :3]
        sortie[cle] = float(np.sqrt((d**2).sum(axis=1).mean()))
    return sortie


def bilan(ecarts):
    """`(n, médiane, maximum, part > 0,1 mm, part > 1 mm, critère tenu ?)` d'une liste d'écarts en mm."""
    e = np.asarray(list(ecarts), dtype=float)
    if len(e) == 0:
        raise ValueError("aucune coupe en commun")
    fin, gros = float((e > SEUIL_FIN_MM).mean()), float((e > SEUIL_GROS_MM).mean())
    return len(e), float(np.median(e)), float(e.max()), fin, gros, fin <= PART_FIN_MAX and gros <= PART_GROS_MAX


def pixels_des_stacks(racine, sujet, par_coupe):
    """`{(nom de stack, coupe): pixels du masque en mm}` des trois piles TRUFI du run 1 du sujet (comme `evaluation_finale.py`)."""
    session = sorted(d for d in os.listdir(f"{racine}/{sujet}") if d.startswith("ses-"))[0]
    dossier = f"{racine}/{sujet}/{session}/anat"
    masques = f"{racine}/derivatives/medx-fetalbet/{sujet}/{session}/anat"
    sortie = {}
    for plan in ("trufiax", "truficor", "trufisag"):
        nom = f"{sujet}_{session}_acq-{plan}_run-1"
        if not os.path.exists(f"{dossier}/{nom}_T2w.nii.gz"):
            continue
        affine = nib.load(f"{dossier}/{nom}_T2w.nii.gz").affine
        masque = np.asarray(nib.load(f"{masques}/{nom}_desc-brain_mask.nii.gz").dataobj) > 0
        for k in range(masque.shape[2]):
            ij = np.argwhere(masque[:, :, k])
            if len(ij) == 0:
                continue
            if len(ij) > par_coupe:
                ij = ij[np.linspace(0, len(ij) - 1, par_coupe).astype(int)]
            ijk = np.hstack([ij, np.full((len(ij), 1), k), np.ones((len(ij), 1))])
            sortie[(f"{nom}_T2w", k)] = (ijk @ affine.T)[:, :3]
    return sortie


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sujet", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--pixels-par-coupe", type=int, default=250)
    ap.add_argument("--sortie", help="TSV des écarts par coupe")
    a = ap.parse_args()
    ecarts = deplacements(lire_poses(a.ref), lire_poses(a.new), pixels_des_stacks(a.racine, a.sujet, a.pixels_par_coupe))
    n, mediane, maximum, fin, gros, tenu = bilan(ecarts.values())
    if a.sortie:
        with open(a.sortie, "w") as f:
            f.write("stack\tslice\tecart_mm\n")
            for (s, k), e in sorted(ecarts.items()):
                f.write(f"{s}\t{k}\t{e:.6f}\n")
    print(f"{a.sujet}\tcoupes={n}\tmediane_mm={mediane:.4f}\tmax_mm={maximum:.4f}\tpart_sup_0.1mm={fin:.4f}\tpart_sup_1mm={gros:.4f}\tcritere_c1={'tenu' if tenu else 'NON tenu'}")


if __name__ == "__main__":
    main()
