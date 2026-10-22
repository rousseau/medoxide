"""Rejoue la chaîne OFFICIELLE de Multi-BOUNTI fœtal (variante « general », sans N4) sur un volume à 0,5 mm, et garde chaque fichier intermédiaire : ce sont les références de test du portage (voir `docs/MULTIBOUNTI.md`).

L'image Docker de Multi-BOUNTI (AMD64, 38 Go) n'est pas utilisée ; la chaîne est enchaînée à partir de morceaux officiels, comme pour BOUNTI :
  - étapes MIRTK géométriques : l'image ARM `fetalsvrtk/svrtk:general_auto_arm` (outils `pad-3d`, `register`, `transform-image`, … ; `scripts/multibounti_mirtk_phases.sh`) ;
  - réseaux : le script Python officiel `run_monai_patch_atunet_segmentation_1case-2026-cpu.py` SUR LE MAC (CPU : MONAI n'a pas `ConvTranspose3d` sur MPS), avec les poids officiels ;
  - `guess-ga` et `thin-cortex-thick` (outils propres à l'image AMD) : binaires AMD64 extraits des couches de l'image, exécutés dans un conteneur AMD64 (`scripts/multibounti_outils_amd.Dockerfile`).
Le réseau 43 étiquettes lance 343 fenêtres de 128³ (recouvrement 0,8) : environ 70 minutes par volume sur le CPU. Chaque étape est sautée si son fichier de sortie existe déjà (reprise après interruption).

Prérequis (voir docs/MULTIBOUNTI.md) : `data/multibounti/image/` (scripts, gabarits, poids), `data/multibounti/amd/` (binaires et bibliothèques MIRTK AMD64, avec les liens `.so.0`), `data/multibounti/entrees/<id>.nii.gz` (`scripts/multibounti_entrees.py`).
Usage : python scripts/multibounti_reference.py <id> [--sans-reseau-43]      (sortie : data/multibounti/ref/<id>/, résultat final `sortie_officielle.nii.gz`)
`--sans-reseau-43` s'arrête après la phase C (avant le réseau de 343 fenêtres) : toutes les étapes géométriques et les deux petits réseaux sont alors disponibles en quelques minutes ; relancer sans l'option reprend à la suite.
"""

import os
import shutil
import subprocess
import sys
import time

RACINE = os.getcwd()
IMAGE_ARM = "fetalsvrtk/svrtk:general_auto_arm"
IMAGE_AMD = "multibounti-amd-outils"
OFFICIEL = f"{RACINE}/data/multibounti/image/home/perinatal-brain-mri-analysis"
MIRTK_AMD = f"{RACINE}/data/multibounti/amd/usr/bin/MIRTK"
POIDS_BET = f"{OFFICIEL}/models/atunet_bet_brain_153t_svr_2026_128_best_metric_model.pth"
POIDS_LR = f"{OFFICIEL}/models/atunet_lr_brain_fetal_2lab_best_metric_model.pth"
POIDS_MULTI = f"{OFFICIEL}/models/patch_atunet_new_multi_43_brain_neo_fetal_aff_fix_022026_best_metric_model.pth"
RESEAU = f"{OFFICIEL}/src/run_monai_patch_atunet_segmentation_1case-2026-cpu.py"
LIMITE_MEMOIRE_GO = "18"


def lancer(commande, **options):
    debut = time.time()
    subprocess.run(commande, check=True, **options)
    return time.time() - debut


def phase_mirtk(dossier, phase, environnement=None):
    """Une phase des commandes MIRTK du script officiel, dans l'image ARM."""
    env = [] if not environnement else [x for k, v in environnement.items() for x in ("-e", f"{k}={v}")]
    commande = ["docker", "run", "--rm", "--platform", "linux/arm64", "-v", f"{dossier}:/work", "-v", f"{OFFICIEL}/templates:/templates:ro",
                "-v", f"{RACINE}/scripts/multibounti_mirtk_phases.sh:/phases.sh:ro", *env, "--entrypoint", "bash", IMAGE_ARM, "/phases.sh", phase]
    return lancer(commande, stdout=subprocess.DEVNULL)


def outil_amd(dossier, *arguments):
    """Un outil MIRTK propre à l'image AMD (conteneur AMD64, bibliothèques montées) ; renvoie sa sortie standard."""
    commande = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{MIRTK_AMD}:/mirtk:ro", "-v", f"{dossier}:/work", IMAGE_AMD, "sh", "-c",
                "export LD_LIBRARY_PATH=/mirtk/build/lib; cd /work; " + " ".join(arguments)]
    return subprocess.run(commande, check=True, capture_output=True, text=True).stdout


def reseau(dossier, taille, etiquettes, poids, entree, sortie):
    """Le script d'inférence officiel (CPU), sous le garde-fou mémoire ; `cwd` = dossier de travail."""
    commande = ["python3", f"{RACINE}/scripts/garde_memoire.py", LIMITE_MEMOIRE_GO, "python3", RESEAU, str(taille), str(etiquettes), poids, entree, sortie]
    return lancer(commande, cwd=dossier, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def etape(dossier, nom, fichier_de_sortie, action):
    chemin = f"{dossier}/{fichier_de_sortie}"
    if os.path.exists(chemin):
        print(f"[{nom}] déjà fait ({fichier_de_sortie})", flush=True)
        return
    duree = action()
    print(f"[{nom}] fait en {duree:.0f} s", flush=True)


def main():
    ident = sys.argv[1]
    dossier = f"{RACINE}/data/multibounti/ref/{ident}"
    os.makedirs(dossier, exist_ok=True)
    entree = f"{RACINE}/data/multibounti/entrees/{ident}.nii.gz"
    if not os.path.exists(f"{dossier}/entree.nii.gz"):
        shutil.copy(entree, f"{dossier}/entree.nii.gz")

    etape(dossier, "A prétraitement MIRTK", "pad-crop-tr-t2-128.nii.gz", lambda: phase_mirtk(dossier, "A"))
    dz = float(open(f"{dossier}/dz.txt").read().split()[0])
    if not 0.49 <= dz <= 0.51:
        sys.exit(f"la résolution z doit valoir 0,5 mm (±0,01) : {dz}")
    etape(dossier, "réseau : extraction du cerveau", "bet-lab-pad-crop-t2-128.nii.gz",
          lambda: reseau(dossier, 128, 1, POIDS_BET, "pad-crop-tr-t2-128.nii.gz", "bet-lab-pad-crop-t2-128.nii.gz"))
    # la phase B réécrit `bet-lab-pad-crop-t2-128.nii.gz` (érosion, composante, dilatation) : on garde la sortie brute du réseau à part
    if not os.path.exists(f"{dossier}/bet-reseau-brut.nii.gz"):
        shutil.copy(f"{dossier}/bet-lab-pad-crop-t2-128.nii.gz", f"{dossier}/bet-reseau-brut.nii.gz")
    etape(dossier, "B nettoyage du masque", "pad-masked-n4-crop-tr-t2-128.nii.gz", lambda: phase_mirtk(dossier, "B"))
    etape(dossier, "réseau : gauche / droite", "lr-lab-pad-masked-crop-t2-128.nii.gz",
          lambda: reseau(dossier, 128, 2, POIDS_LR, "pad-masked-n4-crop-tr-t2-128.nii.gz", "lr-lab-pad-masked-crop-t2-128.nii.gz"))
    volume_ml = int(outil_amd(dossier, "/mirtk/build/lib/tools/guess-ga", "bet-lab-pad-crop-t2-128.nii.gz").split()[0])
    open(f"{dossier}/guess-ga.txt", "w").write(f"{volume_ml}\n")
    print(f"[guess-ga] {volume_ml} mL -> {'i.dof' if volume_ml > 170 else 'i-very-early-ga.dof'}", flush=True)
    etape(dossier, "C recalage affine et réorientation", "pad-reo-n4-masked-t2-256.nii.gz", lambda: phase_mirtk(dossier, "C", {"GUESS_GA": volume_ml}))
    if "--sans-reseau-43" in sys.argv:
        print("arrêt avant le réseau 43 étiquettes (--sans-reseau-43)")
        return
    etape(dossier, "réseau : 43 étiquettes", "multi-lab-reo-t2.nii.gz",
          lambda: reseau(dossier, 128, 43, POIDS_MULTI, "pad-reo-n4-masked-t2-256.nii.gz", "multi-lab-reo-t2.nii.gz"))
    etape(dossier, "D retour dans la grille d'origine", "tr-multi-lab-reo-t2.nii.gz", lambda: phase_mirtk(dossier, "D"))
    def amincir():
        debut = time.time()
        outil_amd(dossier, "/mirtk/build/lib/tools/thin-cortex-thick", "tr-multi-lab-reo-t2.nii.gz", "sortie_officielle.nii.gz")
        return time.time() - debut

    etape(dossier, "thin-cortex-thick", "sortie_officielle.nii.gz", amincir)
    print("terminé :", f"{dossier}/sortie_officielle.nii.gz")


if __name__ == "__main__":
    main()
