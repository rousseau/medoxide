"""Évaluation à plus de deux stacks, SANS vérité terrain, par prédiction hors échantillon : un sujet avec N >= 4 stacks TRUFI, chaque stack mis de côté tour à tour, reconstruction avec les N-1 autres.

Le protocole à deux stacks de `benchmark_real.py` laisse très peu de contraintes à un fœtus très agité ; avec trois stacks pour reconstruire, on est plus proche de l'usage réel. Pour chaque stack mis de côté `h` :
  - « sans correction » (poses d'en-tête) à plusieurs alpha (test d'évaluation réel `reconstruct_a_real_subject`, 0 cycle) ;
  - notre coupe-coupe (`medx svr recon --motion s2s`) à plusieurs alpha ; notre coupe-volume (`--motion s2v`) à alpha = 0,1 ; couverture minimale 1 (le juge lit le volume aux pixels des coupes testées : un domaine réduit y met des zéros) ;
  - SVRTK (réglages par défaut, Docker) et NeSVoR (réglages par défaut, sans SVoRT, CPU) ;
  - le JUGE (test `judge_a_volume_with_a_held_out_stack`) et un indicateur de netteté `S` (gradient moyen du volume lissé d'un voxel, dans le masque érodé, divisé par l'intensité médiane du tissu ; validé par dose-réponse sur des volumes floutés).
La comparaison se fait par FRONTIÈRE juge x netteté : le juge favorise le lissage, donc on compare deux méthodes à netteté égale. Les fichiers déjà présents sont réutilisés (reprise).

Aucun identifiant n'est écrit dans le code : le sujet et la racine sont passés en arguments. Les résultats restent dans `--sortie` (ignoré par Git).

Usage : MEDOXIDE_TEST_BIN=<binaire de test> python scripts/benchmark_4stacks.py --racine data/svr/jeu_reel_tru_haste --sujet sub-XXXX --sortie dossier [--alphas 0.03 0.1 0.3]
"""

import argparse
import os
import re
import subprocess
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("--racine", required=True)
parser.add_argument("--sujet", required=True)
parser.add_argument("--sortie", required=True)
parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.03, 0.3])
parser.add_argument("--medx", default="target/release/medx")
parser.add_argument("--sans-nesvor", action="store_true")
args = parser.parse_args()

racine = os.path.abspath(args.racine)
sortie = os.path.abspath(args.sortie)
os.makedirs(sortie, exist_ok=True)
binaire = os.environ["MEDOXIDE_TEST_BIN"]
resultats_sujet = os.path.abspath(f"data/svr/results/{args.sujet}")
session = sorted(d for d in os.listdir(f"{racine}/{args.sujet}") if d.startswith("ses-"))[0]
dossier = f"{racine}/{args.sujet}/{session}/anat"
masques = f"{racine}/derivatives/medx-fetalbet/{args.sujet}/{session}/anat"

# mêmes stacks et même ordre que `stacks_reels()` des tests : run 1 (axial, coronal, sagittal) d'abord, puis les autres runs par ordre de run et de plan
trouves = []
for f in os.listdir(dossier):
    m = re.match(rf"{args.sujet}_{session}_acq-(trufiax|truficor|trufisag)_run-(\d+)_T2w\.nii\.gz$", f)
    if m:
        plan, run = m.group(1), int(m.group(2))
        trouves.append((int(run != 1), run * 10 + ["trufiax", "truficor", "trufisag"].index(plan), f"{args.sujet}_{session}_acq-{plan}_run-{run}"))
trouves.sort()
noms = [t[2] for t in trouves]
N = len(noms)
assert N >= 4, f"il faut au moins 4 stacks TRUFI, {N} trouvés"
chemin_stack = lambda i: f"{dossier}/{noms[i]}_T2w.nii.gz"
chemin_masque = lambda i: f"{masques}/{noms[i]}_desc-brain_mask.nii.gz"


def nettete(chemin):
    v = np.asarray(nib.load(chemin).dataobj, dtype=np.float64)
    tissu = v > 0.05 * np.percentile(v[v > 0], 99.5)
    interieur = binary_erosion(tissu, iterations=3)
    g = np.sqrt(sum(np.gradient(gaussian_filter(v, 1.0), axis=a) ** 2 for a in range(3)))
    return float(g[interieur].mean() / np.median(v[interieur]))


def juge(h, volume):
    e = {**os.environ, "MEDOXIDE_DONNEES_REELLES": racine, "MEDOXIDE_SUJET": args.sujet, "MEDOXIDE_STACK_TEST": str(h), "MEDOXIDE_VOLUME": volume}
    r = subprocess.run([binaire, "judge_a_volume_with_a_held_out_stack", "--ignored", "--nocapture"], capture_output=True, text=True, env=e)
    m = re.search(r"moyenne ([\d.]+)", r.stdout + r.stderr)
    return float(m.group(1)) if m else float("nan")


def sans_correction(h, gardes, alpha):
    etiquette = f"_n{alpha}h{h}"
    volume = f"{resultats_sujet}/ours_header{etiquette}.nii.gz"
    if not os.path.exists(volume):
        e = {**os.environ, "MEDOXIDE_DONNEES_REELLES": racine, "MEDOXIDE_SUJET": args.sujet, "MEDOXIDE_STACKS_RETENUS": ",".join(map(str, gardes)), "MEDOXIDE_CYCLES": "0", "MEDOXIDE_LOSO_CYCLES": "0",
             "MEDOXIDE_LOO_CYCLES": "0", "MEDOXIDE_ETIQUETTE": etiquette, "MEDOXIDE_ALPHA": str(alpha)}
        subprocess.run([binaire, "reconstruct_a_real_subject", "--ignored", "--nocapture"], capture_output=True, text=True, env=e)
    return volume


def notre_methode(mouvement, h, gardes, alpha):
    volume = f"{sortie}/{mouvement}_a{alpha}_h{h}.nii.gz"
    if not os.path.exists(volume):
        subprocess.run([args.medx, "svr", "recon", "--motion", mouvement, "--device", "cpu", "--alpha", str(alpha), "--min-stack-coverage", "1", "--input", *[chemin_stack(i) for i in gardes],
                        "--mask", *[chemin_masque(i) for i in gardes], "--output", volume], check=True, capture_output=True)
    return volume


def svrtk(h, gardes):
    volume = f"{sortie}/svrtk_h{h}.nii.gz"
    if not os.path.exists(volume):
        subprocess.run(["python3", "scripts/run_svrtk.py", "--stacks", *[chemin_stack(i) for i in gardes], "--mask", chemin_masque(gardes[0]), "--sortie", sortie, "--nom", f"svrtk_h{h}.nii.gz"],
                       check=True, capture_output=True)
    return volume


def nesvor(h, gardes):
    dossier_h = f"{sortie}/nesvor_h{h}"
    volume = f"{dossier_h}/nesvor.nii.gz"
    if not os.path.exists(volume):
        subprocess.run(["python3", "scripts/run_nesvor.py", "--stacks", *[chemin_stack(i) for i in gardes], "--masques", *[chemin_masque(i) for i in gardes], "--sortie", dossier_h], check=True, capture_output=True)
    return volume


lignes = ["methode\talpha\tstack_teste\tncc_moyenne\tnettete"]


def enregistrer(nom, alpha, h, volume):
    t = time.time()
    j, s = juge(h, volume), nettete(volume)
    lignes.append(f"{nom}\t{alpha}\t{h}\t{j:.4f}\t{s:.4f}")
    print(lignes[-1], flush=True)
    open(f"{sortie}/resultats.tsv", "w").write("\n".join(lignes) + "\n")


principal, extras = args.alphas[0], args.alphas[1:]
for h in range(N):
    gardes = [i for i in range(N) if i != h]
    print(f"=== stack {h} mis de côté ; reconstruction avec {gardes}", flush=True)
    enregistrer("sans_correction", principal, h, sans_correction(h, gardes, principal))
    enregistrer("s2s", principal, h, notre_methode("s2s", h, gardes, principal))
    enregistrer("s2v", principal, h, notre_methode("s2v", h, gardes, principal))
    enregistrer("SVRTK", "defaut", h, svrtk(h, gardes))
    if not args.sans_nesvor:
        enregistrer("NeSVoR", "defaut", h, nesvor(h, gardes))
for a in extras:
    for h in range(N):
        gardes = [i for i in range(N) if i != h]
        enregistrer("sans_correction", a, h, sans_correction(h, gardes, a))
        enregistrer("s2s", a, h, notre_methode("s2s", h, gardes, a))
open(f"{sortie}/done", "w").write("ok")
