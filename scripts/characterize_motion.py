"""Caractérise le mouvement par coupe d'un ensemble de poses (estimées sur des données réelles, ou vraies sur des jeux simulés).

Entrée : un fichier de poses `poses*.tsv` (une ligne par coupe : stack, coupe, puis les 16 valeurs de la matrice monde 4x4 en ordre colonne, le mouvement absolu composé APRÈS l'affine d'en-tête)
et les stacks NIfTI dont il provient (pour la position monde du centre de chaque coupe). Les coupes dont la pose est exactement l'identité (non recalées) sont exclues.

Grandeurs, toutes invariantes par un décalage commun à tout un stack (le repère d'une reconstruction n'est défini qu'à un mouvement rigide près) sauf l'amplitude, mesurée par rapport à la
médiane du stack :
  - incréments entre coupes CONSÉCUTIVES d'un même stack (toutes deux recalées) : angle de rotation relatif (°) et variation du déplacement du centre de la coupe (mm) ; médiane, p90, part des
    incréments de déplacement > 3 mm (« sauts ») ;
  - autocorrélation des trois composantes du déplacement du centre (après retrait de la médiane du stack) aux décalages 1, 2, 3 et 5 coupes, estimateur à moyenne et variance connues par composante
    (moyenne nulle imposée par le retrait de la médiane, variance estimée) calculé sur des paires de coupes recalées ;
  - amplitude : angle de rotation et déplacement du centre par rapport à la médiane du stack, médiane et p90.

Usage : python scripts/characterize_motion.py --poses poses.tsv --stacks s0.nii.gz s1.nii.gz s2.nii.gz [--etiquette texte]
"""

import argparse

import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation


def lire_poses(chemin):
    poses = {}
    for ligne in open(chemin):
        c = ligne.strip().split("\t")
        if len(c) == 18:
            poses[(int(c[0]), int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    return poses


def caracteriser(poses, chemins_stacks):
    """Rend un dictionnaire de statistiques ; `poses` : {(stack, coupe): matrice 4x4}."""
    serie = {}
    for (s, k), m in poses.items():
        if np.allclose(m, np.eye(4), atol=1e-12):
            continue
        serie.setdefault(s, {})[k] = m
    centres, deplacements, rotations = {}, {}, {}
    for s, coupes in serie.items():
        img = nib.load(chemins_stacks[s])
        nx, ny, nz = img.shape[:3]
        for k, m in coupes.items():
            c = img.affine @ np.array([(nx - 1) / 2, (ny - 1) / 2, k, 1.0])
            deplacements[(s, k)] = (m @ c - c)[:3]
            rotations[(s, k)] = Rotation.from_matrix(m[:3, :3])
    out = {"n_coupes": len(deplacements), "n_stacks": len(serie)}
    inc_rot, inc_dep, amp_rot, amp_dep = [], [], [], []
    produits = {lag: [] for lag in (1, 2, 3, 5)}
    for s, coupes in serie.items():
        ks = sorted(coupes)
        dep = np.array([deplacements[(s, k)] for k in ks])
        mediane = np.median(dep, axis=0)
        rel = dep - mediane
        rot_med = Rotation.from_rotvec(np.median(np.array([rotations[(s, k)].as_rotvec() for k in ks]), axis=0))
        amp_dep += list(np.linalg.norm(rel, axis=1))
        amp_rot += [np.degrees((rotations[(s, k)] * rot_med.inv()).magnitude()) for k in ks]
        variance = np.mean(rel**2, axis=0)  # par composante
        rel_par_k = {k: rel[i] for i, k in enumerate(ks)}
        for k in ks:
            if k + 1 in coupes:
                inc_dep.append(np.linalg.norm(deplacements[(s, k + 1)] - deplacements[(s, k)]))
                inc_rot.append(np.degrees((rotations[(s, k + 1)] * rotations[(s, k)].inv()).magnitude()))
            for lag in produits:
                if k + lag in coupes:
                    produits[lag].append(rel_par_k[k] * rel_par_k[k + lag] / variance)  # produit normalisé, par composante
    p = lambda v, q: float(np.percentile(v, q)) if len(v) else float("nan")  # noqa: E731
    out.update(
        inc_dep_med=p(inc_dep, 50), inc_dep_p90=p(inc_dep, 90), part_sauts=100 * float(np.mean(np.array(inc_dep) > 3.0)) if inc_dep else float("nan"),
        inc_rot_med=p(inc_rot, 50), inc_rot_p90=p(inc_rot, 90), n_incr=len(inc_dep),
        amp_dep_med=p(amp_dep, 50), amp_dep_p90=p(amp_dep, 90), amp_rot_med=p(amp_rot, 50), amp_rot_p90=p(amp_rot, 90),
    )
    for lag, v in produits.items():
        out[f"autocorr_{lag}"] = float(np.mean(v)) if v else float("nan")
    return out


def afficher(etiquette, s):
    print(
        f"{etiquette:34} | {s['n_coupes']:3d} coupes | incr. dépl. méd {s['inc_dep_med']:5.2f} p90 {s['inc_dep_p90']:5.2f} mm, sauts>3mm {s['part_sauts']:4.0f} % "
        f"| incr. rot. méd {s['inc_rot_med']:4.2f} p90 {s['inc_rot_p90']:4.2f}° | ampl. dépl. méd {s['amp_dep_med']:5.2f} p90 {s['amp_dep_p90']:5.2f} mm, rot. méd {s['amp_rot_med']:4.2f} p90 {s['amp_rot_p90']:4.2f}° "
        f"| autocorr. lag 1/2/3/5 : {s['autocorr_1']:5.2f} {s['autocorr_2']:5.2f} {s['autocorr_3']:5.2f} {s['autocorr_5']:5.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", required=True)
    parser.add_argument("--stacks", nargs="+", required=True)
    parser.add_argument("--etiquette", default="")
    args = parser.parse_args()
    afficher(args.etiquette or args.poses, caracteriser(lire_poses(args.poses), args.stacks))
