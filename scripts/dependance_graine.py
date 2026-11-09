#!/usr/bin/env python3
"""Qui dépend de la graine ? (docs/ROADMAP.md, F4.0.3, point 0.3b).

Pour chaque méthode (`s2s`, `s2v`), reconstruction avec tous les stacks TRUFI (run 1, axial, coronal, sagittal) d'un sujet, sur CPU, aux défauts :
  A : graine 1, deux fois (déterminisme à graine fixe) ; B : graine 1 contre graine 2.
Écart = valeur absolue maximale de la différence, voxel à voxel pour le volume, nombre à nombre pour les poses (`--poses`). Lecture pré-déclarée :
  A ≠ 0           : non déterministe à graine fixe (le bruit d'exécution est inclus dans la variabilité de 0.3c) ;
  A = 0 et B ≠ 0  : dépend de la graine ;
  A = 0 et B = 0  : ne dépend pas de la graine (variabilité entre graines nulle par construction).
A et B sont les plus grands écarts du volume et des poses. Les fichiers déjà présents sont réutilisés (reprise).

Aucun identifiant n'est écrit dans le code : racine et sujet sont des arguments ; sorties sous `--sortie` (ignoré par Git).
"""

import argparse
import csv
import os
import re
import subprocess
import sys

import nibabel as nib
import numpy as np

PLANS = ("trufiax", "truficor", "trufisag")


def ecart_volumes(a, b):
    """Écart absolu maximal entre deux volumes NIfTI de même forme (ValueError si les formes diffèrent)."""
    x, y = np.asarray(nib.load(a).dataobj, dtype=np.float64), np.asarray(nib.load(b).dataobj, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"formes différentes : {x.shape} et {y.shape}")
    return float(np.max(np.abs(x - y)))


def ecart_poses(a, b):
    """Écart absolu maximal entre deux TSV de poses : colonnes numériques comparées nombre à nombre, autres colonnes égales ou ValueError ; formes différentes : ValueError."""
    with open(a, newline="") as f, open(b, newline="") as g:
        ra, rb = list(csv.reader(f, delimiter="\t")), list(csv.reader(g, delimiter="\t"))
    if len(ra) != len(rb) or any(len(u) != len(v) for u, v in zip(ra, rb)):
        raise ValueError("les fichiers de poses n'ont pas la même forme")
    pire = 0.0
    for u, v in zip(ra[1:], rb[1:]):
        for s, t in zip(u, v):
            try:
                pire = max(pire, abs(float(s) - float(t)))
            except ValueError:
                if s != t:
                    raise ValueError(f"cellules non numériques différentes : {s!r} et {t!r}")
    return pire


def classer(ecart_meme_graine, ecart_graines_differentes):
    """Lecture pré-déclarée à partir de A (même graine deux fois) et B (graine 1 contre 2)."""
    if ecart_meme_graine != 0:
        return "non déterministe à graine fixe"
    return "dépend de la graine" if ecart_graines_differentes != 0 else "ne dépend pas de la graine"


def piles(racine, sujet):
    """Stacks et masques du run 1 axial, coronal, sagittal de la première session du sujet."""
    session = sorted(d for d in os.listdir(f"{racine}/{sujet}") if d.startswith("ses-"))[0]
    dossier, masques = f"{racine}/{sujet}/{session}/anat", f"{racine}/derivatives/medx-fetalbet/{sujet}/{session}/anat"
    noms = [f"{sujet}_{session}_acq-{p}_run-1" for p in PLANS]
    for n in noms:
        if not os.path.exists(f"{dossier}/{n}_T2w.nii.gz"):
            raise FileNotFoundError(f"{dossier}/{n}_T2w.nii.gz")
    return [f"{dossier}/{n}_T2w.nii.gz" for n in noms], [f"{masques}/{n}_desc-brain_mask.nii.gz" for n in noms]


def reconstruire(a, methode, nom, graine, entrees, masques):
    volume, poses = f"{a.sortie}/{methode}_{nom}.nii.gz", f"{a.sortie}/{methode}_{nom}_poses.tsv"
    if not (os.path.exists(volume) and os.path.exists(poses)):
        r = subprocess.run([a.medx, "svr", "recon", "--motion", methode, "--device", "cpu", "--s2s-seed", str(graine), "--input", *entrees, "--mask", *masques, "--output", volume, "--poses", poses],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"échec de {methode} {nom} :\n{r.stderr[-800:]}")
    return volume, poses


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--racine", required=True)
    p.add_argument("--sujet", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--medx", required=True)
    p.add_argument("--methodes", nargs="+", default=["s2s", "s2v"], choices=["s2s", "s2v"])
    a = p.parse_args()
    os.makedirs(a.sortie, exist_ok=True)
    entrees, masques = piles(a.racine, a.sujet)
    for m in a.methodes:
        v1, p1 = reconstruire(a, m, "g1a", 1, entrees, masques)
        v1b, p1b = reconstruire(a, m, "g1b", 1, entrees, masques)
        v2, p2 = reconstruire(a, m, "g2", 2, entrees, masques)
        A = max(ecart_volumes(v1, v1b), ecart_poses(p1, p1b))
        B = max(ecart_volumes(v1, v2), ecart_poses(p1, p2))
        print(f"{m} : A (graine 1 deux fois) volume {ecart_volumes(v1, v1b):.3e} poses {ecart_poses(p1, p1b):.3e} ; B (graine 1 contre 2) volume {ecart_volumes(v1, v2):.3e} poses {ecart_poses(p1, p2):.3e} → {classer(A, B)}", flush=True)


if __name__ == "__main__":
    main()
