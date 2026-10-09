"""Lance la reconstruction SVRTK (`mirtk reconstruct`, réglages par défaut) sur des stacks NIfTI, dans Docker.

Les stacks et le masque du stack template (le premier) sont copiés dans un dossier de travail local, monté en lecture seule dans le conteneur ; la reconstruction est écrite dans
le dossier de sortie. L'épaisseur de coupe de chaque stack est lue dans l'en-tête (3e pas du voxel), la résolution de sortie est celle donnée (0,8 mm par défaut) ; tout le reste
est laissé aux valeurs par défaut de SVRTK (décision du projet : on ne règle pas SVRTK).

Image : fetalsvrtk/svrtk:general_auto_arm (arm64) ou general_auto_amd (amd64), à télécharger au préalable (`docker pull`).

Usage : python scripts/run_svrtk.py --stacks s0.nii.gz s1.nii.gz s2.nii.gz --mask masque_du_premier_stack.nii.gz --sortie dossier/ --nom svrtk.nii.gz
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time

import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument("--stacks", nargs="+", required=True)
parser.add_argument("--mask", required=True, help="masque du premier stack (le template)")
parser.add_argument("--sortie", required=True)
parser.add_argument("--nom", default="svrtk.nii.gz")
parser.add_argument("--resolution", type=float, default=0.8)
parser.add_argument("--image", default="fetalsvrtk/svrtk:general_auto_arm")
parser.add_argument("--plateforme", default="linux/arm64")
args = parser.parse_args()

os.makedirs(args.sortie, exist_ok=True)
with tempfile.TemporaryDirectory(dir=args.sortie, prefix="entree_") as entree:
    noms = []
    for n, chemin in enumerate(args.stacks):
        shutil.copy(chemin, f"{entree}/stack{n}.nii.gz")
        noms.append(f"/in/stack{n}.nii.gz")
    shutil.copy(args.mask, f"{entree}/masque.nii.gz")
    epaisseurs = [str(round(float(nib.load(c).header.get_zooms()[2]), 4)) for c in args.stacks]
    commande = [
        "docker", "run", "--rm", "--platform", args.plateforme,
        "-v", f"{os.path.abspath(entree)}:/in:ro", "-v", f"{os.path.abspath(args.sortie)}:/out",
        args.image, "mirtk", "reconstruct", f"/out/{args.nom}", str(len(noms)), *noms,
        "--mask", "/in/masque.nii.gz", "--thickness", *epaisseurs, "--resolution", str(args.resolution),
    ]
    debut = time.time()
    resultat = subprocess.run(commande, capture_output=True, text=True)
    journal = [ligne for ligne in resultat.stdout.splitlines() if not ligne.startswith("Time for")]
    open(f"{args.sortie}/{args.nom}.log", "w").write("\n".join(journal))
    metriques = [ligne for ligne in journal if "global metrics" in ligne or "Number of slices" in ligne]
    print(f"SVRTK : code de retour {resultat.returncode}, {time.time() - debut:.0f} s ; " + " | ".join(m.strip() for m in metriques))
    if resultat.returncode != 0:
        print(resultat.stderr[-2000:])
    else:
        # la sortie de SVRTK n'a qu'un qform : on écrit aussi le sform (même affine), exigé par le lecteur de medoxide
        sortie_nifti = f"{args.sortie}/{args.nom}"
        image = nib.load(sortie_nifti)
        corrigee = nib.Nifti1Image(image.get_fdata(dtype="float32"), image.affine)
        corrigee.set_sform(image.affine, code=1)
        corrigee.set_qform(image.affine, code=1)
        nib.save(corrigee, sortie_nifti)
