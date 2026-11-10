#!/usr/bin/env python3
"""Critères (c)(1′) et (c)(4′) de F4.0.6 : l'analytique contre le bruit de l'autodiff (docs/ROADMAP.md, F4.0.6, « E43 : ENQUÊTE », validés le 2026-11-10).

Deux sous-commandes.
  perturber  `--entree stack.nii.gz --sortie stack_e.nii.gz --epsilon 7e-7` : copie du stack multipliée par (1 + ε) et enregistrée en `float32`.
             La NCC y est mathématiquement invariante : seul l'arrondi `f32` change (contrôle de bruit de l'autodiff).
  analyser   `--racine … --sujet … --autodiff DIR --analytic DIR --controles DIR1 DIR2 DIR3 --medx …` : pour un sujet,
    (c)(1′) la part de coupes à plus de 0,1 mm (resp. 1 mm) entre l'analytique et l'autodiff est au plus la plus grande part mesurée entre l'autodiff
            et chacune des trois exécutions de contrôle ; la médiane de l'écart est au plus 1,5 fois la plus grande médiane de contrôle ;
    (c)(4′) le Dice moyen (12 étiquettes de Q3, `gradient_pose_mesure.py`) entre les deux volumes est au moins le plus petit Dice moyen entre
            l'autodiff et ses trois contrôles, moins 0,01.
  (c)(2) et (c)(3) restent ceux de `gradient_pose_mesure.py` ; (c)(5) (temps) n'est pas mesuré ici.

Aucun identifiant n'est écrit dans le code : racine, sujet et dossiers sont des arguments.
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradient_pose_equivalence as eq  # noqa: E402
import gradient_pose_mesure as me  # noqa: E402

FACTEUR_MEDIANE = 1.5
MARGE_DICE = 0.01


def perturber(entree, sortie, epsilon):
    """Stack multiplié par (1 + ε), enregistré en `float32` sans échelle (l'original est en entiers : le facteur ne s'y représente pas)."""
    im = nib.load(entree)
    data = np.asarray(im.dataobj, dtype=np.float64) * (1.0 + epsilon)
    entete = im.header.copy()
    entete.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), im.affine, entete), sortie)


def verdict_c1(analytique, controles):
    """`(tenu, détail)` ; `analytique` et chaque élément de `controles` = `(n, médiane, maximum, part > 0,1 mm, part > 1 mm, _)` (`eq.bilan`)."""
    if not controles:
        raise ValueError("aucune exécution de contrôle")
    fin_max, gros_max, med_max = max(c[3] for c in controles), max(c[4] for c in controles), max(c[1] for c in controles)
    ok_fin, ok_gros, ok_med = analytique[3] <= fin_max, analytique[4] <= gros_max, analytique[1] <= FACTEUR_MEDIANE * med_max
    detail = (
        f"> 0,1 mm : {100 * analytique[3]:.2f} % (contrôles au plus {100 * fin_max:.2f} %) ; "
        f"> 1 mm : {100 * analytique[4]:.2f} % (au plus {100 * gros_max:.2f} %) ; "
        f"médiane : {analytique[1]:.4f} mm (au plus {FACTEUR_MEDIANE} × {med_max:.4f})"
    )
    return bool(ok_fin and ok_gros and ok_med), detail


def verdict_c4(dice_analytique, dices_controle):
    """`(tenu, détail)` : Dice de l'analytique au moins le plus petit Dice des contrôles, moins 0,01."""
    if not dices_controle:
        raise ValueError("aucune exécution de contrôle")
    seuil = min(dices_controle) - MARGE_DICE
    return bool(dice_analytique >= seuil), f"Dice {dice_analytique:.4f} (seuil {seuil:.4f} = plus petit contrôle {min(dices_controle):.4f} − {MARGE_DICE})"


def dice_moyen(medx, dossier_a, dossier_b):
    sa = me.segmenter(medx, f"{dossier_a}/volume.nii.gz", f"{dossier_a}/seg.nii.gz")
    sb = me.segmenter(medx, f"{dossier_b}/volume.nii.gz", f"{dossier_b}/seg.nii.gz")
    d = me.dice_par_etiquette(sa, sb)
    return float(np.mean(list(d.values()))) if d else float("nan")


def analyser(racine, sujet, autodiff, analytic, controles, medx):
    pixels = eq.pixels_des_stacks(racine, sujet, 250)
    poses_a = eq.lire_poses(f"{autodiff}/poses.tsv")

    def bilan(d):
        return eq.bilan(eq.deplacements(poses_a, eq.lire_poses(f"{d}/poses.tsv"), pixels).values())

    b_n, b_c = bilan(analytic), [bilan(c) for c in controles]
    d_n, d_c = dice_moyen(medx, autodiff, analytic), [dice_moyen(medx, autodiff, c) for c in controles]
    v1, det1 = verdict_c1(b_n, b_c)
    v4, det4 = verdict_c4(d_n, d_c)
    return {"c1p": (v1, det1), "c4p": (v4, det4), "bilan_analytique": b_n, "bilans_controles": b_c, "dice_analytique": d_n, "dice_controles": d_c}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("perturber")
    a.add_argument("--entree", required=True)
    a.add_argument("--sortie", required=True)
    a.add_argument("--epsilon", type=float, required=True)
    a = sp.add_parser("analyser")
    a.add_argument("--racine", required=True)
    a.add_argument("--sujet", required=True)
    a.add_argument("--autodiff", required=True)
    a.add_argument("--analytic", required=True)
    a.add_argument("--controles", nargs=3, required=True)
    a.add_argument("--medx", required=True)
    a.add_argument("--sortie")
    args = ap.parse_args()
    if args.cmd == "perturber":
        perturber(args.entree, args.sortie, args.epsilon)
        return
    r = analyser(args.racine, args.sujet, args.autodiff, args.analytic, args.controles, args.medx)
    fmt = lambda b: f"{b[0]}:{b[1]:.4f}:{b[2]:.4f}:{b[3]:.4f}:{b[4]:.4f}"  # noqa: E731
    ligne = (
        f"{args.sujet}\tanalytique={fmt(r['bilan_analytique'])}\tcontroles={';'.join(fmt(b) for b in r['bilans_controles'])}\t"
        f"Dice_analytique={r['dice_analytique']:.5f}\tDice_controles={';'.join(f'{d:.5f}' for d in r['dice_controles'])}\t"
        f"c1p={'tenu' if r['c1p'][0] else 'NON tenu'}\tc4p={'tenu' if r['c4p'][0] else 'NON tenu'}"
    )
    print(ligne)
    for k in ("c1p", "c4p"):
        print(f"  ({k}) {'tenu' if r[k][0] else 'NON TENU'} : {r[k][1]}")
    if args.sortie:
        with open(args.sortie, "a") as f:
            f.write(ligne + "\n")


if __name__ == "__main__":
    main()
