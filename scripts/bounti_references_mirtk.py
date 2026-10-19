"""Produit les références MIRTK de la géométrie de BOUNTI pour des volumes donnés (voir `docs/BOUNTI.md`, sous-étape 5a).

Pour chaque `id=volume.nii.gz` : copie le volume dans `data/bounti/mirtk/<id>/entree.nii.gz` avec un `qform` cohérent avec son `sform` (MIRTK lit le `qform` : sans cela il place le volume avec son coin à l'origine), puis exécute dans l'image Docker de BOUNTI
(`fetalsvrtk/svrtk:general_auto_arm` ou `general_auto_amd`) le script `bounti_mirtk_preparer.sh` (prétraitement officiel) puis `bounti_mirtk_masque.sh` (extraction du cerveau par le réseau officiel et nettoyage du masque), qui laissent dans ce dossier chaque étape. Ces fichiers ne sont pas versionnés (`data/` est ignoré) ;
les tests `the_geometry_matches_mirtk` et `the_mask_cleaning_matches_mirtk` de `medoxide-bounti` les comparent au code Rust.

Troisième étape (`etiquettes`, sous-étape 5d-1) : des étiquettes de test (`data/bounti/reference/fusion_128_reel_labels.u8`, agrandies d'un facteur 2) sont posées sur la géométrie du cube de 256³ (`cube256.nii.gz`) puis ramenées dans la grille de l'entrée par `bounti_mirtk_etiquettes.sh`.

Quatrième et cinquième étapes (`reseau`, `finale`, sous-étape 5d-2) : le script officiel de segmentation des tissus (`data/bounti/src/run_monai_comb_red_atunet_unet_segmentation-2022-lr.py`, fusion à 256³) est exécuté SUR LE MAC, sur le cube `cube256m` laissé par `masque`, sous `garde_memoire.py` (la machine virtuelle de Docker Desktop, 7,7 Gio par défaut, ne suffit pas :
le processus y est tué) ; puis `bounti_mirtk_finale.sh` ramène ses étiquettes et le masque du cerveau dans la grille de l'entrée. Le script officiel complet n'est donc pas exécuté d'un seul tenant : chaque morceau est l'officiel, enchaîné ici.

Usage : python scripts/bounti_references_mirtk.py calme1=chemin/volume1.nii.gz calme2=chemin/volume2.nii.gz [--image fetalsvrtk/svrtk:general_auto_arm] [--etapes preparer,masque,etiquettes]
"""

import argparse
import os
import shutil
import subprocess

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("volumes", nargs="+", help="id=fichier.nii.gz")
parser.add_argument("--image", default="fetalsvrtk/svrtk:general_auto_arm")
parser.add_argument("--etapes", default="preparer,masque,etiquettes,reseau,finale", help="étapes à rejouer, séparées par des virgules (les suivantes lisent les fichiers laissés par les précédentes)")
args = parser.parse_args()

racine = os.path.abspath("data/bounti/mirtk")
os.makedirs(racine, exist_ok=True)
etapes = args.etapes.split(",")
for source, destination in [("bounti_mirtk_preparer.sh", "preparer.sh"), ("bounti_mirtk_masque.sh", "masque.sh"), ("bounti_mirtk_etiquettes.sh", "etiquettes.sh"), ("bounti_mirtk_finale.sh", "finale.sh")]:
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), source), os.path.join(racine, destination))
    os.chmod(os.path.join(racine, destination), 0o755)
for entree in args.volumes:
    ident, chemin = entree.split("=", 1)
    dossier = os.path.join(racine, ident)
    os.makedirs(dossier, exist_ok=True)
    if "preparer" in etapes:
        image = nib.load(chemin)
        copie = nib.Nifti1Image(image.get_fdata(dtype="float32"), image.affine)
        copie.set_sform(image.affine, code=1)
        copie.set_qform(image.affine, code=1)  # qform cohérent avec le sform
        nib.save(copie, os.path.join(dossier, "entree.nii.gz"))
    if "etiquettes" in etapes:
        etiquettes = np.fromfile("data/bounti/reference/fusion_128_reel_labels.u8", dtype=np.uint8).reshape(128, 128, 128)
        etiquettes = etiquettes.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        cube = nib.load(os.path.join(dossier, "cube256.nii.gz"))
        image = nib.Nifti1Image(etiquettes, cube.affine)
        image.set_sform(cube.affine, code=1)
        image.set_qform(cube.affine, code=1)
        nib.save(image, os.path.join(dossier, "etiquettes_cube256.nii.gz"))
    if "reseau" in etapes:
        donnees = os.path.abspath("data/bounti")
        sortie = os.path.join(dossier, "bounti_resultats")
        os.makedirs(sortie, exist_ok=True)
        modele = lambda nom: os.path.join(donnees, "models", nom) + "/"
        subprocess.run(
            ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "garde_memoire.py"), "19", "python3",
             os.path.join(donnees, "src/run_monai_comb_red_atunet_unet_segmentation-2022-lr.py"), dossier + "/",
             modele("monai-checkpoints-red-attunet-brain_bounti-19-lab"), modele("monai-checkpoints-unet-brain-bounti-19-lab"),
             "info256m.json", sortie, "256", "19"],
            check=True,
        )
    for script in ("preparer.sh", "masque.sh", "etiquettes.sh", "finale.sh"):
        if script.split(".")[0] in etapes:
            subprocess.run(["docker", "run", "--rm", "-v", f"{racine}:/work", "--entrypoint", "bash", args.image, f"/work/{script}", ident], check=True, stdout=subprocess.DEVNULL)
    print(f"{ident} : références écrites dans {dossier}")
