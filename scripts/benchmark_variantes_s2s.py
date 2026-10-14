"""Compare des variantes de notre coupe-coupe (`medx svr recon --motion s2s`) par prédiction hors échantillon, pour un sujet à N >= 3 stacks TRUFI (N - 1 stacks pour reconstruire, un mis de côté tour à tour).

Pour chaque stack mis de côté `h` et chaque variante : reconstruction avec les N - 1 autres stacks (couverture minimale 1 : le juge lit le volume aux pixels des coupes testées, un domaine réduit y met des zéros), puis
  - le JUGE (test `judge_a_volume_with_a_held_out_stack`) : NCC moyenne après recalage du stack mis de côté contre le volume ;
  - la netteté `S` (gradient moyen du volume lissé d'un voxel, dans le masque érodé, divisé par l'intensité médiane du tissu) ;
  - le TEMPS CPU de la reconstruction (utilisateur + système des processus enfants : le temps mural n'a pas de sens sur une machine partagée). L'ORDRE des variantes alterne d'un repli à l'autre (comparaison alternée).
Le juge favorisant le lissage, on compare par frontière juge x netteté ; les fichiers déjà présents sont réutilisés (reprise, sans temps).

Une variante s'écrit `nom=options` (options de `medx svr recon`, entre guillemets) ; `base=` est la variante par défaut.

Aucun identifiant n'est écrit dans le code : sujet et racine sont des arguments. Les résultats restent dans `--sortie` (ignoré par Git).

Usage : MEDOXIDE_TEST_BIN=<binaire de test> python scripts/benchmark_variantes_s2s.py --racine data/svr/jeu_reel_tru_haste --sujet sub-XXXX --sortie dossier --variantes base= "bloc1=--s2s-block-levels 1"
"""

import argparse
import os
import re
import resource
import shlex
import subprocess

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("--racine", required=True)
parser.add_argument("--sujet", required=True)
parser.add_argument("--sortie", required=True)
parser.add_argument("--variantes", nargs="+", required=True)
parser.add_argument("--medx", default="target/release/medx")
args = parser.parse_args()

racine = os.path.abspath(args.racine)
sortie = os.path.abspath(args.sortie)
os.makedirs(sortie, exist_ok=True)
binaire = os.environ["MEDOXIDE_TEST_BIN"]
VARIANTES = [tuple(v.split("=", 1)) for v in args.variantes]
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
assert N >= 3, f"il faut au moins 3 stacks TRUFI, {N} trouvés"
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


def reconstruire(nom, options, h, gardes):
    """Volume de la variante pour le repli `h` et son temps CPU (secondes ; vide si le fichier existait déjà)."""
    volume = f"{sortie}/{nom}_h{h}.nii.gz"
    if os.path.exists(volume):
        return volume, ""
    avant = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run([args.medx, "svr", "recon", "--motion", "s2s", "--device", "cpu", "--min-stack-coverage", "1", "--input", *[chemin_stack(i) for i in gardes], "--mask", *[chemin_masque(i) for i in gardes],
                    "--output", volume, *shlex.split(options)], check=True, capture_output=True)
    apres = resource.getrusage(resource.RUSAGE_CHILDREN)
    return volume, f"{apres.ru_utime + apres.ru_stime - avant.ru_utime - avant.ru_stime:.0f}"


lignes = ["variante\tstack_teste\tncc_moyenne\tnettete\ttemps_cpu_s"]
for h in range(N):
    gardes = [i for i in range(N) if i != h]
    ordre = VARIANTES if h % 2 == 0 else VARIANTES[::-1]  # comparaison alternée
    print(f"=== stack {h} mis de côté ; reconstruction avec {gardes}", flush=True)
    for nom, options in ordre:
        volume, temps = reconstruire(nom, options, h, gardes)
        lignes.append(f"{nom}\t{h}\t{juge(h, volume):.4f}\t{nettete(volume):.4f}\t{temps}")
        print(lignes[-1], flush=True)
        open(f"{sortie}/resultats.tsv", "w").write("\n".join(lignes) + "\n")
open(f"{sortie}/done", "w").write("ok")
