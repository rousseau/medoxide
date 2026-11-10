#!/usr/bin/env python3
"""Critères (c)(1) à (c)(4) de F4.0.6 pour un sujet : `analytic` contre `autodiff` du même binaire (docs/ROADMAP.md, F4.0.6, 0.6c).

Entrées : un dossier par variante contenant `volume.nii.gz` et `poses.tsv` (sorties de `medx svr recon --motion s2v ... [--pose-gradient analytic]`).
  (c)(1) équivalence des poses : au plus 5 % des coupes à plus de 0,1 mm et au plus 0,5 % à plus de 1 mm (`gradient_pose_equivalence.py`) ;
  (c)(2) netteté `S` du volume complet à ±3 % (même définition que `benchmark_variantes_s2s.py`) ;
  (c)(3) concordance Q2 (toutes les paires, paires sans le stack 0) : aucune perte de plus de 0,01 ;
  (c)(4) Dice BOUNTI entre les deux volumes : moyenne des 12 étiquettes de Q3 au moins 0,975 ; une étiquette vide dans les deux volumes est écartée,
         vide dans un seul : Dice 0, comptée.
Les temps (c)(5) ne sont pas mesurés ici (étape E31). Aucun identifiant dans le code : racine, sujet et dossiers sont des arguments.

Usage : python3 -I scripts/gradient_pose_mesure.py --racine data/svr/jeu_reel_tru_haste --sujet sub-XXXX --autodiff DIR --analytic DIR --medx data/svr/banc/bin/medx_<commit> [--sortie ligne.tsv]
"""

import argparse
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradient_pose_equivalence as eq  # noqa: E402
from analyse_concordance_sujets import stacks_du_sujet  # noqa: E402
from intersection_consistency import Jeu, concordance  # noqa: E402

LABELS = [3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17]  # ceux de Q3 (analyse_reproductibilite_segmentation.py)
NETTETE_TOLERANCE = 0.03
Q2_PERTE_MAX = 0.01
DICE_MIN = 0.975


def nettete(v):
    """Netteté `S` : gradient moyen du volume lissé d'un voxel, dans le tissu érodé, divisé par l'intensité médiane (définition de `benchmark_variantes_s2s.py`)."""
    v = np.asarray(v, dtype=np.float64)
    tissu = v > 0.05 * np.percentile(v[v > 0], 99.5)
    interieur = binary_erosion(tissu, iterations=3)
    g = np.sqrt(sum(np.gradient(gaussian_filter(v, 1.0), axis=a) ** 2 for a in range(3)))
    return float(g[interieur].mean() / np.median(v[interieur]))


def dice_par_etiquette(a, b, labels=LABELS):
    """`{étiquette: Dice}` ; une étiquette vide dans les deux segmentations est absente du résultat, vide dans une seule : Dice 0."""
    sortie = {}
    for k in labels:
        na, nb = int((a == k).sum()), int((b == k).sum())
        if na == 0 and nb == 0:
            continue
        sortie[k] = 2 * int(((a == k) & (b == k)).sum()) / (na + nb)
    return sortie


def verdicts(c1, nettete_a, nettete_n, q2_a, q2_n, dice_moyen):
    """Verdict de chaque critère, sans réinterprétation : `{(c)(1)…(c)(4): (tenu, valeur)}`.
    `c1` = sortie de `gradient_pose_equivalence.bilan`, `q2_*` = (toutes paires, sans stack 0) de chaque variante."""
    rapport = nettete_n / nettete_a - 1.0
    pertes = [qa - qn for qa, qn in zip(q2_a, q2_n)]
    return {
        "c1": (bool(c1[5]), f"{100 * c1[3]:.2f} % > 0,1 mm ; {100 * c1[4]:.2f} % > 1 mm"),
        "c2": (abs(rapport) <= NETTETE_TOLERANCE, f"{100 * rapport:+.2f} %"),
        "c3": (max(pertes) <= Q2_PERTE_MAX, f"pertes {pertes[0]:+.4f} ; {pertes[1]:+.4f}"),
        "c4": (dice_moyen >= DICE_MIN, f"Dice {dice_moyen:.4f}"),
    }


def segmenter(medx, volume, seg):
    if not os.path.exists(seg):
        r = subprocess.run([medx, "bounti", "--device", "cpu", "--input", volume, "--output", seg], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"medx bounti a échoué sur {volume} : {r.stderr[-300:]}")
    return np.asarray(nib.load(seg).dataobj).astype(np.int16)


def lire_poses_absolues(chemin):
    return {k: m for k, m in eq.lire_poses(chemin).items()}


def concordances(racine, sujet, poses_a, poses_n):
    """Q2 de chaque variante sur les coupes recalées par les deux : `((toutes, hors stack 0) autodiff, idem analytic)`."""
    stacks, masques = stacks_du_sujet(racine, sujet)
    jeu = Jeu(stacks, masques)
    noms = [os.path.basename(s).removesuffix(".nii.gz") for s in stacks]
    indice = {n: i for i, n in enumerate(noms)}

    def convertir(p):
        return {(indice[s], k): m for (s, k), m in p.items() if s in indice and not np.allclose(m, np.eye(4), atol=1e-12)}

    pa, pn = convertir(poses_a), convertir(poses_n)
    communes = sorted(set(pa) & set(pn))
    sortie = []
    for p in (pa, pn):
        tout = concordance(jeu, p, communes)
        hors = concordance(jeu, p, communes, hors_stack=0)
        sortie.append((float(tout.mean()), float(hors.mean())))
    return sortie[0], sortie[1], len(communes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--racine", required=True)
    ap.add_argument("--sujet", required=True)
    ap.add_argument("--autodiff", required=True)
    ap.add_argument("--analytic", required=True)
    ap.add_argument("--medx", required=True)
    ap.add_argument("--sortie")
    a = ap.parse_args()
    poses_a, poses_n = eq.lire_poses(f"{a.autodiff}/poses.tsv"), eq.lire_poses(f"{a.analytic}/poses.tsv")
    ecarts = eq.deplacements(poses_a, poses_n, eq.pixels_des_stacks(a.racine, a.sujet, 250))
    c1 = eq.bilan(ecarts.values())
    va, vn = nib.load(f"{a.autodiff}/volume.nii.gz").get_fdata(), nib.load(f"{a.analytic}/volume.nii.gz").get_fdata()
    na, nn = nettete(va), nettete(vn)
    q2_a, q2_n, n_communes = concordances(a.racine, a.sujet, poses_a, poses_n)
    sa = segmenter(a.medx, f"{a.autodiff}/volume.nii.gz", f"{a.autodiff}/seg.nii.gz")
    sn = segmenter(a.medx, f"{a.analytic}/volume.nii.gz", f"{a.analytic}/seg.nii.gz")
    dices = dice_par_etiquette(sa, sn)
    dice_moyen = float(np.mean(list(dices.values()))) if dices else float("nan")
    v = verdicts(c1, na, nn, q2_a, q2_n, dice_moyen)
    detail = ",".join(f"{k}:{d:.3f}" for k, d in sorted(dices.items()))
    ligne = (
        f"{a.sujet}\tcoupes={c1[0]}\tmediane_mm={c1[1]:.4f}\tmax_mm={c1[2]:.4f}\tpart_0.1mm={c1[3]:.4f}\tpart_1mm={c1[4]:.4f}\t"
        f"S_autodiff={na:.5f}\tS_analytic={nn:.5f}\tQ2_toutes_autodiff={q2_a[0]:.5f}\tQ2_toutes_analytic={q2_n[0]:.5f}\t"
        f"Q2_hors0_autodiff={q2_a[1]:.5f}\tQ2_hors0_analytic={q2_n[1]:.5f}\tcoupes_Q2={n_communes}\tDice_moyen={dice_moyen:.5f}\tDice_par_etiquette={detail}\t"
        + "\t".join(f"{k}={'tenu' if t else 'NON tenu'}" for k, (t, _) in v.items())
    )
    print(ligne)
    for k, (t, val) in v.items():
        print(f"  ({k[0]})({k[1]}) {'tenu' if t else 'NON TENU'} : {val}")
    if a.sortie:
        with open(a.sortie, "a") as f:
            f.write(ligne + "\n")


if __name__ == "__main__":
    main()
