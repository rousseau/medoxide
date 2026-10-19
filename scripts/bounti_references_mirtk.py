"""Produit les références MIRTK de la géométrie de BOUNTI pour des volumes donnés (voir `docs/BOUNTI.md`, sous-étape 5a).

Pour chaque `id=volume.nii.gz` : copie le volume dans `data/bounti/mirtk/<id>/entree.nii.gz` avec un `qform` cohérent avec son `sform` (MIRTK lit le `qform` : sans cela il place le volume avec son coin à l'origine), puis exécute dans l'image Docker de BOUNTI
(`fetalsvrtk/svrtk:general_auto_arm` ou `general_auto_amd`) le script `bounti_mirtk_preparer.sh`, qui laisse dans ce dossier chaque étape du prétraitement officiel. Ces fichiers ne sont pas versionnés (`data/` est ignoré) ; le test `the_geometry_matches_mirtk` de `medoxide-bounti` les compare au code Rust.

Usage : python scripts/bounti_references_mirtk.py calme1=chemin/volume1.nii.gz calme2=chemin/volume2.nii.gz [--image fetalsvrtk/svrtk:general_auto_arm]
"""

import argparse
import os
import shutil
import subprocess

import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument("volumes", nargs="+", help="id=fichier.nii.gz")
parser.add_argument("--image", default="fetalsvrtk/svrtk:general_auto_arm")
args = parser.parse_args()

racine = os.path.abspath("data/bounti/mirtk")
os.makedirs(racine, exist_ok=True)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bounti_mirtk_preparer.sh"), os.path.join(racine, "preparer.sh"))
os.chmod(os.path.join(racine, "preparer.sh"), 0o755)
for entree in args.volumes:
    ident, chemin = entree.split("=", 1)
    dossier = os.path.join(racine, ident)
    os.makedirs(dossier, exist_ok=True)
    image = nib.load(chemin)
    copie = nib.Nifti1Image(image.get_fdata(dtype="float32"), image.affine)
    copie.set_sform(image.affine, code=1)
    copie.set_qform(image.affine, code=1)  # qform cohérent avec le sform
    nib.save(copie, os.path.join(dossier, "entree.nii.gz"))
    subprocess.run(["docker", "run", "--rm", "-v", f"{racine}:/work", "--entrypoint", "bash", args.image, "/work/preparer.sh", ident], check=True, stdout=subprocess.DEVNULL)
    print(f"{ident} : références écrites dans {dossier}")
