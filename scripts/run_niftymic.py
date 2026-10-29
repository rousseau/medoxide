"""Lance la reconstruction NiftyMIC (`niftymic_reconstruct_volume`, Docker amd64) sur des stacks NIfTI et leurs masques cérébraux.

Défauts de NiftyMIC, SAUF `--use-masks-srr 1` : le même masque cérébral sert à toutes les méthodes comparées. Le premier stack est la cible par défaut. Les stacks et masques sont copiés dans `<sortie>/entree/` (`stack<i>.nii.gz`, `stack<i>_mask.nii.gz`), monté en lecture seule ; le volume est écrit dans `<sortie>/srr.nii.gz`. Un volume déjà présent n'est pas recalculé.

Image : renbem/niftymic:latest (à télécharger au préalable, `docker pull`). Sous macOS arm64 l'image tourne en émulation : compter de une à quelques heures par sujet.

Usage : python scripts/run_niftymic.py --stacks s0.nii.gz s1.nii.gz --masques m0.nii.gz m1.nii.gz --sortie dossier/ [--delai-h 4]
"""

import argparse
import os
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--stacks", nargs="+", required=True)
parser.add_argument("--masques", nargs="+", required=True)
parser.add_argument("--sortie", required=True)
parser.add_argument("--delai-h", type=float, default=4.0, help="durée maximale (heures)")
args = parser.parse_args()
assert len(args.stacks) == len(args.masques)

sortie = os.path.abspath(args.sortie)
volume = f"{sortie}/srr.nii.gz"
if os.path.exists(volume):
    print("NiftyMIC : volume déjà présent", volume)
    raise SystemExit(0)
entree = f"{sortie}/entree"
os.makedirs(entree, exist_ok=True)
for i, (s, m) in enumerate(zip(args.stacks, args.masques)):
    shutil.copy(s, f"{entree}/stack{i}.nii.gz")
    shutil.copy(m, f"{entree}/stack{i}_mask.nii.gz")
n = len(args.stacks)
commande = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{entree}:/in:ro", "-v", f"{sortie}:/out", "renbem/niftymic:latest", "niftymic_reconstruct_volume",
            "--filenames", *[f"/in/stack{i}.nii.gz" for i in range(n)], "--filenames-masks", *[f"/in/stack{i}_mask.nii.gz" for i in range(n)], "--output", "/out/srr.nii.gz", "--use-masks-srr", "1"]
debut = time.time()
with open(f"{sortie}/niftymic.log", "w") as j:
    r = subprocess.run(commande, stdout=j, stderr=subprocess.STDOUT, timeout=args.delai_h * 3600)
print(f"NiftyMIC : code de retour {r.returncode}, {time.time() - debut:.0f} s")
raise SystemExit(r.returncode)
