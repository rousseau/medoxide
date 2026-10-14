"""Évalue, AVEC vérité connue, ce que `medx svr recon` retrouve du mouvement sur des jeux simulés (`make_pyrecon_benchmark.py`).

Pour chaque jeu (trois stacks, mouvement vrai dans `poses.tsv`), lance `medx svr recon` (coupe-coupe `s2s` ou coupe-volume `s2v`) et mesure l'ERREUR DE POSE : pour chaque coupe recalée, écart quadratique moyen (mm) entre les
positions de pixels de masque placées avec la pose estimée et avec la vraie pose, après retrait du mouvement rigide COMMUN (Kabsch sur l'ensemble des coupes : le repère de sortie d'une reconstruction est libre). Rend la médiane, le
90ᵉ centile et la part des coupes à moins de 1,5 mm. La ligne « en-tête » donne l'erreur sans correction (poses identité), qui mesure l'amplitude du mouvement à corriger.

Usage : python scripts/evaluate_simulated_motion.py --jeux data/atlas/sim_pyrecon/STA31_lisse_mvt8_b5 ... [--methodes s2s s2v] [--sortie dossier] [--alpha 0.1] [--etiquette _c8 --extra "--s2s-coarse-sigma 8"]
"""

import argparse
import os
import shlex
import subprocess

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--jeux", nargs="+", required=True)
parser.add_argument("--methodes", nargs="+", default=["s2s", "s2v"])
parser.add_argument("--sortie", default="data/atlas/results/evaluation_mouvement")
parser.add_argument("--alpha", default="0.1")
parser.add_argument("--medx", default="target/release/medx")
parser.add_argument("--etiquette", default="", help="suffixe des fichiers de sortie (une variante de réglages par étiquette)")
parser.add_argument("--extra", default="", help="options supplémentaires passées à `medx svr recon`, une seule chaîne, par exemple --extra \"--s2s-coarse-sigma 8\"")
args = parser.parse_args()
os.makedirs(args.sortie, exist_ok=True)


def lire_poses(f):
    return {(int(l.split("\t")[0]), int(l.split("\t")[1])): np.array([float(v) for v in l.split("\t")[2:]]).reshape(4, 4, order="F") for l in open(f).read().strip().split("\n")}


def kabsch(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mq - R @ mp


def poser(M, X):
    return (M[:3, :3] @ X.T).T + M[:3, 3]


def erreurs(jeu, estimees):
    vraies = lire_poses(f"{jeu}/poses.tsv")
    rng = np.random.default_rng(0)
    pts = {}
    for s in range(3):
        A = nib.load(f"{jeu}/stack{s}.nii.gz").affine
        m = np.asarray(nib.load(f"{jeu}/stack{s}_mask.nii.gz").dataobj) > 0
        for k in range(m.shape[2]):
            ij = np.argwhere(m[:, :, k])
            if len(ij) < 500:
                continue
            sel = ij[rng.choice(len(ij), min(250, len(ij)), replace=False)]
            pts[(s, k)] = (A @ np.c_[sel, np.full(len(sel), k), np.ones(len(sel))].T).T[:, :3]
    cles = list(pts)
    Xe = np.vstack([poser(estimees[k], pts[k]) for k in cles])
    Xv = np.vstack([poser(vraies[k], pts[k]) for k in cles])
    R, t = kabsch(Xe, Xv)
    return np.array([np.sqrt((((R @ poser(estimees[k], pts[k]).T).T + t - poser(vraies[k], pts[k])) ** 2).sum(1).mean()) for k in cles]), cles


def ligne(nom, e):
    return f"{nom:28} médiane {np.median(e):5.2f} mm   p90 {np.percentile(e, 90):5.2f} mm   ≤ 1,5 mm : {100 * (e <= 1.5).mean():3.0f} %"


for jeu in args.jeux:
    nom_jeu = os.path.basename(jeu.rstrip("/"))
    print(f"\n=== {nom_jeu}", flush=True)
    identite = {k: np.eye(4) for k in lire_poses(f"{jeu}/poses.tsv")}
    print(ligne("en-tête (sans correction)", erreurs(jeu, identite)[0]), flush=True)
    for m in args.methodes:
        volume, poses = f"{args.sortie}/{nom_jeu}_{m}{args.etiquette}.nii.gz", f"{args.sortie}/{nom_jeu}_{m}{args.etiquette}.tsv"
        qualite = f"{args.sortie}/{nom_jeu}_{m}{args.etiquette}.qualite.tsv"
        if not os.path.exists(poses) or not os.path.exists(qualite):
            subprocess.run([args.medx, "svr", "recon", "--motion", m, "--device", "cpu", "--alpha", args.alpha, "--input", *[f"{jeu}/stack{s}.nii.gz" for s in range(3)],
                            "--mask", *[f"{jeu}/stack{s}_mask.nii.gz" for s in range(3)], "--output", volume, "--poses", poses, "--quality", qualite, *shlex.split(args.extra)], check=True, capture_output=True)
        e, cles = erreurs(jeu, lire_poses(poses))
        print(ligne(m, e), flush=True)
        # erreur vraie par coupe, à côté de l'indicateur de qualité (même clé stack, coupe) pour `analyse_qualite_coupes.py`
        open(f"{args.sortie}/{nom_jeu}_{m}{args.etiquette}.erreurs.tsv", "w").write("stack\tcoupe\terreur_mm\n" + "\n".join(f"{k[0]}\t{k[1]}\t{v:.4f}" for k, v in zip(cles, e)) + "\n")
