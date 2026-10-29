"""Reproductibilité de la segmentation entre graines et entre replis (docs/ROADMAP.md, § F2.6).

Entrées : un dossier contenant `full.nii.gz` (volume complet de référence du sujet) et, pour chaque variante `v` et chaque repli `h`, `<v>_h<h>.nii.gz` (reconstruction avec un stack mis de côté, voir `benchmark_variantes_s2s.py`).
Le volume de référence est orienté sur l'atlas (recherche dense d'orientations, `atlas_orientation_dense.py`, premier des trois meilleurs candidats qui passe le contrôle anatomique), puis chaque volume de repli est recalé rigidement sur lui et placé avec le même alignement (`atlas_aligne_sur_reference.py`) et segmenté par BOUNTI (protocole partagé de `evaluation_alignement.py`). Les fichiers présents ne sont pas recalculés.

Sorties (dans `<dossier>/atlas/`) : `s2s.nii.gz`, `ref_vers_atlas.tfm` (la référence), `<v>_h<h>.nii.gz` et `seg_<v>_h<h>.nii.gz` ; `<dossier>/reproductibilite.tsv`.
  M1 « bruit de graine » : Dice moyen (étiquettes de Q3) entre deux variantes de GRAINE (`--graines`) au même repli, moyenné sur les replis et les paires.
  M2 « entre replis » (le Q3) : Dice moyen entre les segmentations de deux replis de la même variante, moyenné sur les paires de replis ; pour les graines, moyenne des variantes de graine.

Usage : python scripts/analyse_reproductibilite_segmentation.py --dossier data/svr/results/sub-XXXX/f26 --variantes gs1 gs2 gs3 jac --graines gs1 gs2 gs3
"""

import argparse
import glob
import itertools
import os
import re
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dossier", required=True)
parser.add_argument("--variantes", nargs="+", required=True)
parser.add_argument("--graines", nargs="*", default=[], help="variantes qui ne diffèrent que par la graine (M1)")
parser.add_argument("--bin", default="data/svr/eval/bin")
args = parser.parse_args()
scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts)
from controle_anatomique import controle  # noqa: E402

D = os.path.abspath(args.dossier)
A = f"{D}/atlas"
os.makedirs(A, exist_ok=True)
medx = os.path.abspath(f"{args.bin}/medx")
LABELS = [3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17]  # ceux de Q3 (evaluation_analyse.py)


def lancer(commande, journal):
    with open(journal, "w") as j:
        return subprocess.run(commande, stdout=j, stderr=subprocess.STDOUT).returncode


def segmente(volume, seg):
    if not os.path.exists(seg):
        lancer([medx, "bounti", "--input", volume, "--output", seg], f"{seg}.log")


# 1. référence orientée sur l'atlas
if not os.path.exists(f"{A}/s2s.nii.gz"):
    masque = f"{A}/masque.nii.gz"
    if not os.path.exists(masque):
        lancer([medx, "bounti", "--input", f"{D}/full.nii.gz", "--output", f"{A}/seg_brut.nii.gz", "--brain-mask", masque], f"{A}/bet.log")
    if not os.path.exists(f"{A}/cand_0.nii.gz"):
        lancer([sys.executable, "-I", f"{scripts}/atlas_orientation_dense.py", f"{D}/full.nii.gz", masque, A], f"{A}/dense.log")
    choisi = 0
    for k in range(3):
        segmente(f"{A}/cand_{k}.nii.gz", f"{A}/seg_cand_{k}.nii.gz")
        if controle(f"{A}/seg_cand_{k}.nii.gz")[0]:
            choisi = k
            break
    else:
        print("AUCUN candidat ne passe le contrôle anatomique : candidat 0 gardé", flush=True)
    shutil.copy(f"{A}/cand_{choisi}.nii.gz", f"{A}/s2s.nii.gz")
    shutil.copy(f"{A}/cand_{choisi}.tfm", f"{A}/ref_vers_atlas.tfm")

# 2. volumes de repli placés et segmentés
segs = {}
for v in args.variantes:
    for f in sorted(glob.glob(f"{D}/{v}_h[0-9].nii.gz")):
        h = int(re.search(r"_h(\d+)\.nii\.gz$", f).group(1))
        sortie = f"{A}/{v}_h{h}.nii.gz"
        if not os.path.exists(sortie):
            lancer([sys.executable, "-I", f"{scripts}/atlas_aligne_sur_reference.py", A, f, sortie], f"{A}/{v}_h{h}.reg.txt")
        seg = f"{A}/seg_{v}_h{h}.nii.gz"
        segmente(sortie, seg)
        segs[(v, h)] = np.asarray(nib.load(seg).dataobj).astype(np.int16)
        print(f"{v} repli {h} : placé et segmenté", flush=True)


def dice(a, b):
    return float(np.mean([2 * ((a == k) & (b == k)).sum() / max((a == k).sum() + (b == k).sum(), 1) for k in LABELS]))


lignes = ["mesure\tvariante\tvaleur"]
m1 = []
for h in sorted({h for (_, h) in segs}):
    for a, b in itertools.combinations([g for g in args.graines if (g, h) in segs], 2):
        m1.append(dice(segs[(a, h)], segs[(b, h)]))
        print(f"M1 repli {h} {a}/{b} : Dice {m1[-1]:.3f}")
if m1:
    lignes.append(f"M1_bruit_de_graine\t{'+'.join(args.graines)}\t{np.mean(m1):.4f}")
    print(f"M1 bruit de graine : Dice moyen {np.mean(m1):.3f} (min {np.min(m1):.3f}, {len(m1)} comparaisons)")
m2 = {}
for v in args.variantes:
    hs = sorted(h for (vv, h) in segs if vv == v)
    paires = [dice(segs[(v, a)], segs[(v, b)]) for a, b in itertools.combinations(hs, 2)]
    if paires:
        m2[v] = float(np.mean(paires))
        lignes.append(f"M2_entre_replis\t{v}\t{m2[v]:.4f}")
        print(f"M2 {v} : Dice entre replis {m2[v]:.3f} ({len(paires)} paires)")
if args.graines:
    g = [m2[v] for v in args.graines if v in m2]
    if g:
        lignes.append(f"M2_entre_replis\tmoyenne_graines\t{np.mean(g):.4f}")
        print(f"M2 moyenne des graines : {np.mean(g):.3f}")
open(f"{D}/reproductibilite.tsv", "w").write("\n".join(lignes) + "\n")
