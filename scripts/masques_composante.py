#!/usr/bin/env python3
"""Nettoyage des masques par la plus grande composante connexe 3D et contrôle visuel (docs/ROADMAP.md, F4.0.4, point 0.4e).

Règle de F3.2 : on garde la plus grande composante connexe 3D de chaque masque, originaux intacts. Connexité : 26 voisins.
- `nettoyer` : pour chaque masque brut `<racine>/derivatives/medx-fetalbet/<sujet>/<session>/anat/*_desc-brain_mask.nii.gz` des sujets donnés, écrit le
  masque nettoyé sous `<racine>/derivatives/medx-fetalbet-cc/` (même arborescence) et un rapport TSV (composantes, voxels gardés, voxels écartés, part gardée).
- `montrer` : une image PNG par (sujet, contraste) : une ligne par pile, 5 coupes régulièrement espacées dans l'étendue du masque brut sur l'axe des coupes,
  contour du masque brut (rouge) et du masque nettoyé (cyan, tireté) sur la pile.

Aucun identifiant n'est écrit dans le code : sujets et racines sont des arguments.
"""

import argparse
import csv
import glob
import os
import re

import nibabel as nib
import numpy as np
from scipy import ndimage

VOISINS_26 = np.ones((3, 3, 3), dtype=int)
SEUIL_ALERTE = 0.8


def plus_grande_composante(masque):
    """`(masque nettoyé, nombre de composantes, voxels gardés, voxels écartés)` ; masque vide : inchangé, 0 composante."""
    masque = np.asarray(masque) > 0
    etiquettes, n = ndimage.label(masque, structure=VOISINS_26)
    if n == 0:
        return masque, 0, 0, 0
    tailles = np.bincount(etiquettes.ravel())[1:]
    garde = int(np.argmax(tailles)) + 1
    nettoye = etiquettes == garde
    return nettoye, n, int(nettoye.sum()), int(masque.sum() - nettoye.sum())


def axe_des_coupes(zooms):
    """Axe de la plus grande taille de voxel (l'épaisseur de coupe) ; le premier en cas d'égalité."""
    return int(np.argmax(zooms[:3]))


def cinq_coupes(masque, axe):
    """Cinq indices régulièrement espacés entre la première et la dernière coupe qui contiennent le masque ; ValueError si le masque est vide."""
    autres = tuple(i for i in range(masque.ndim) if i != axe)
    presentes = np.flatnonzero(np.any(masque > 0, axis=autres))
    if presentes.size == 0:
        raise ValueError("masque vide")
    return [int(i) for i in np.round(np.linspace(presentes[0], presentes[-1], 5)).astype(int)]


def masques_bruts(racine, sujets):
    """Chemins des masques bruts des sujets donnés, triés."""
    sortie = []
    for s in sujets:
        sortie += sorted(glob.glob(f"{racine}/derivatives/medx-fetalbet/{s}/ses-*/anat/*_desc-brain_mask.nii.gz"))
    return sortie


def chemin_nettoye(chemin_brut):
    return chemin_brut.replace("/derivatives/medx-fetalbet/", "/derivatives/medx-fetalbet-cc/")


def chemin_pile(chemin_brut):
    """Pile source d'un masque brut : `derivatives/medx-fetalbet/<s>/<ses>/anat/<nom>_desc-brain_mask.nii.gz` → `<s>/<ses>/anat/<nom>_T2w.nii.gz`."""
    racine, reste = chemin_brut.split("/derivatives/medx-fetalbet/")
    return f"{racine}/{reste.replace('_desc-brain_mask.nii.gz', '_T2w.nii.gz')}"


def cmd_nettoyer(a):
    lignes = []
    for brut in masques_bruts(a.racine, a.sujets):
        img = nib.load(brut)
        donnees = np.asarray(img.dataobj)
        nettoye, n, gardes, ecartes = plus_grande_composante(donnees)
        sortie = chemin_nettoye(brut)
        os.makedirs(os.path.dirname(sortie), exist_ok=True)
        nib.save(nib.Nifti1Image(nettoye.astype(np.uint8), img.affine, img.header), sortie)
        total = gardes + ecartes
        part = gardes / total if total else float("nan")
        lignes.append({"masque": os.path.basename(brut), "sujet": brut.split("/medx-fetalbet/")[1].split("/")[0], "composantes": n,
                       "voxels_gardes": gardes, "voxels_ecartes": ecartes, "part_gardee": f"{part:.4f}",
                       "alerte": "oui" if (total == 0 or part < SEUIL_ALERTE) else "non"})
    os.makedirs(os.path.dirname(a.rapport) or ".", exist_ok=True)
    with open(a.rapport, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]), delimiter="\t")
        w.writeheader()
        w.writerows(lignes)
    print(f"{len(lignes)} masques ; alertes : {sum(1 for l in lignes if l['alerte'] == 'oui')}")


def cmd_montrer(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.sortie, exist_ok=True)
    groupes = {}
    for brut in masques_bruts(a.racine, a.sujets):
        m = re.search(r"/medx-fetalbet/(sub-[a-z0-9]+)/.*_acq-(trufi|haste)", brut)
        groupes.setdefault((m.group(1), m.group(2)), []).append(brut)
    for (sujet, contraste), bruts in sorted(groupes.items()):
        fig, axes = plt.subplots(len(bruts), 5, figsize=(12, 2.4 * len(bruts)), squeeze=False)
        for ligne, brut in enumerate(bruts):
            pile = nib.load(chemin_pile(brut))
            image = np.asarray(pile.dataobj, dtype=float)
            raw = np.asarray(nib.load(brut).dataobj) > 0
            propre = np.asarray(nib.load(chemin_nettoye(brut)).dataobj) > 0
            axe = axe_des_coupes(pile.header.get_zooms())
            for col, i in enumerate(cinq_coupes(raw, axe)):
                coupe = lambda v: np.take(v, i, axis=axe).T
                ax = axes[ligne][col]
                ax.imshow(coupe(image), cmap="gray", origin="lower")
                ax.contour(coupe(raw).astype(float), levels=[0.5], colors="red", linewidths=0.8)
                ax.contour(coupe(propre).astype(float), levels=[0.5], colors="cyan", linewidths=0.8, linestyles="dashed")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{re.search(r'acq-([a-z0-9]+)', brut).group(1)} c.{i}", fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{a.sortie}/{sujet}_{contraste}.png", dpi=70)
        plt.close(fig)
    print(f"{len(groupes)} images dans {a.sortie}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    n = s.add_parser("nettoyer")
    n.add_argument("--rapport", required=True)
    n.set_defaults(f=cmd_nettoyer)
    m = s.add_parser("montrer")
    m.add_argument("--sortie", required=True)
    m.set_defaults(f=cmd_montrer)
    for q in (n, m):
        q.add_argument("--racine", required=True)
        q.add_argument("--sujets", nargs="+", required=True)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
