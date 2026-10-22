"""Accord entre les poses estimées par plusieurs méthodes, sur des acquisitions réelles (sans vérité terrain).

Pour chaque sujet, on place des pixels de masque (250 par coupe) avec la pose de chaque méthode ; le repère de sortie d'une méthode étant libre, on aligne rigidement (Kabsch) les deux jeux de points avant de comparer, et on rapporte l'écart quadratique par coupe. Deux sections :
  1. amplitude estimée (écart à l'en-tête) et désaccord entre chaque paire de méthodes (médiane, 90ᵉ centile, part des coupes à plus de 3 mm) ;
  2. décomposition du mouvement estimé de chaque méthode : décalage rigide de chaque stack (entre stacks) et mouvement résiduel par coupe (à l'intérieur d'un stack).
Seules les coupes recalées par TOUTES les méthodes comparées sont utilisées.

Aucun identifiant n'est écrit dans le code : racine des données, dossier de résultats et sujets sont passés en arguments. Chaque méthode est un fichier de poses `nom=fichier.tsv` (format de `medx svr recon --poses`, relatif au dossier du sujet dans `--resultats`) ; les poses de NiftyMIC viennent de `read_niftymic_poses.py`, celles de SVRTK de `read_svrtk_poses.py`.

Usage : python scripts/analyse_accord_poses.py --racine data/svr/jeu_reel_tru_haste --resultats data/svr/results --sujets sub-XXXX sub-YYYY --methodes s2s=poses_s2s_tous.tsv s2v=poses_loop_tous_loso.tsv SVRTK=poses_svrtk.tsv NiftyMIC=poses_niftymic.tsv
"""

import argparse
import itertools
import os
import re

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--racine", required=True)
parser.add_argument("--resultats", required=True)
parser.add_argument("--sujets", nargs="+", required=True)
parser.add_argument("--methodes", nargs="+", required=True, help="nom=fichier.tsv")
parser.add_argument("--pixels-par-coupe", type=int, default=250)
args = parser.parse_args()
METH = dict(m.split("=", 1) for m in args.methodes)


def stacks(sujet):
    """(stack, masque) dans le même ordre que `benchmark_4stacks.py` : run 1 (axial, coronal, sagittal) d'abord, puis les autres runs."""
    session = sorted(d for d in os.listdir(f"{args.racine}/{sujet}") if d.startswith("ses-"))[0]
    dossier = f"{args.racine}/{sujet}/{session}/anat"
    masques = f"{args.racine}/derivatives/medx-fetalbet/{sujet}/{session}/anat"
    trouves = []
    for f in os.listdir(dossier):
        m = re.match(rf"{sujet}_{session}_acq-(trufiax|truficor|trufisag)_run-(\d+)_T2w\.nii\.gz$", f)
        if m:
            trouves.append((int(int(m.group(2)) != 1), int(m.group(2)) * 10 + ["trufiax", "truficor", "trufisag"].index(m.group(1)), f"{sujet}_{session}_acq-{m.group(1)}_run-{m.group(2)}"))
    trouves.sort()
    return [(f"{dossier}/{n}_T2w.nii.gz", f"{masques}/{n}_desc-brain_mask.nii.gz") for _, _, n in trouves]


def lire_poses(f):
    """Poses d'un fichier TSV : `stack coupe` puis 16 valeurs en ordre colonne. Le stack est un indice (anciens fichiers, sans en-tête) ou un nom (en-tête `stack slice m00 …`) : un nom prend l'indice de son ordre d'apparition."""
    noms, poses = {}, {}
    for l in open(f).read().strip().split("\n"):
        c = l.split("\t")
        if c[0] == "stack":
            continue
        s = int(c[0]) if c[0].isdigit() else noms.setdefault(c[0], len(noms))
        poses[(s, int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    return poses


def kabsch(P, Q):
    """Rotation R et translation t minimisant |R P + t - Q|² (P, Q : points en lignes)."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, mq - R @ mp


def points_par_coupe(sujet, n_par, seed=0):
    rng = np.random.default_rng(seed)
    sortie = {}
    for s, (fs, fm) in enumerate(stacks(sujet)):
        A = nib.load(fs).affine
        m = np.asarray(nib.load(fm).dataobj) > 0
        for k in range(m.shape[2]):
            ij = np.argwhere(m[:, :, k])
            if len(ij) < 500:
                continue
            sel = ij[rng.choice(len(ij), min(n_par, len(ij)), replace=False)]
            sortie[(s, k)] = (A @ np.c_[sel, np.full(len(sel), k), np.ones(len(sel))].T).T[:, :3]
    return sortie


def poser(M, X):
    return (M[:3, :3] @ X.T).T + M[:3, 3]


for numero, sujet in enumerate(args.sujets, 1):
    pts = points_par_coupe(sujet, args.pixels_par_coupe)
    P = {m: lire_poses(f"{args.resultats}/{sujet}/{f}") for m, f in METH.items() if os.path.exists(f"{args.resultats}/{sujet}/{f}")}
    cles = [k for k in pts if all(k in p for p in P.values())]
    pos = {m: {k: poser(P[m][k], pts[k]) for k in cles} for m in P}
    pos["en-tête"] = {k: pts[k] for k in cles}
    print(f"\nsujet {numero} : {len(cles)} coupes comparées ; méthodes : {', '.join(P)}")

    def residus(a, b):
        R, t = kabsch(np.vstack([pos[a][k] for k in cles]), np.vstack([pos[b][k] for k in cles]))
        return np.array([np.sqrt((((R @ pos[a][k].T).T + t - pos[b][k]) ** 2).sum(1).mean()) for k in cles])

    amp = {m: residus(m, "en-tête") for m in P}
    print("  amplitude estimée (écart à l'en-tête, mm) médiane / p90 : " + " ; ".join(f"{m} {np.median(a):.1f}/{np.percentile(a, 90):.1f}" for m, a in amp.items()))
    for a, b in itertools.combinations(P, 2):
        r = residus(a, b)
        print(f"  désaccord {a}–{b} (mm) médiane {np.median(r):5.1f}  p90 {np.percentile(r, 90):5.1f}  part > 3 mm {100 * (r > 3).mean():3.0f} %")

    print("  décomposition : décalage rigide de chaque stack (mm) ; mouvement résiduel par coupe, médiane par stack (mm)")
    ns = sorted(set(k[0] for k in cles))
    for m in P:
        # on aligne d'abord l'ensemble des coupes sur l'en-tête (Kabsch global), puis on décompose par stack
        Xa = np.vstack([pos[m][k] for k in cles])
        Xh = np.vstack([pts[k] for k in cles])
        R, t = kabsch(Xa, Xh)
        glob = {k: (R @ pos[m][k].T).T + t for k in cles}
        inter, intra = [], []
        for s in ns:
            ks = [k for k in cles if k[0] == s]
            Hs = np.vstack([pts[k] for k in ks])
            Rs, ts = kabsch(Hs, np.vstack([glob[k] for k in ks]))  # mouvement rigide du stack entier
            inter.append(np.sqrt(((((Rs @ Hs.T).T + ts) - Hs) ** 2).sum(1).mean()))
            intra.append(np.median([np.sqrt(((glob[k] - ((Rs @ pts[k].T).T + ts)) ** 2).sum(1).mean()) for k in ks]))
        print(f"  {m:10}: {' / '.join(f'{x:4.1f}' for x in inter)} ; {' / '.join(f'{x:4.1f}' for x in intra)}")
