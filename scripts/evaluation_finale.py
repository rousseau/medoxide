"""Évaluation finale d'un sujet : exécute une méthode sur tous les stacks (volume complet) et, pour nos variantes, NeSVoR et SVRTK, à chaque repli (un stack mis de côté tour à tour) ; juge chaque volume de repli.

Protocole figé dans docs/ROADMAP.md (§ F1-éval). Tout est rangé dans `<sortie>/<sujet>/<méthode>/` (`full*` : tous les stacks ; `folds/` ou `h<k>/` : le stack d'indice k est mis de côté) ; les fichiers déjà présents sont réutilisés (reprise).

Méthodes :
  ours      notre coupe-coupe (s2s) aux défauts (α = 0,05, F2.4 : échelle par coupe), puis variantes α = 0,03 et 0,1 (frontière juge × netteté) et deux bases aux repliements : « en-tête » (poses de l'en-tête, aucune correction) et « grossier » (coupe-coupe grossier seul, ancien « sans correction ») ; volume complet, volume complet ancré sur le stack 0 (Q2 non circulaire), volumes complets en-tête et grossier.
  ours_s2v  notre coupe-volume (s2v) aux défauts (départ par stack avec biais, a priori temporel 0,01, échelle par coupe) : volume complet (poses : Q2 est non circulaire, le coupe-volume n'utilise pas les intersections) et repliements α = 0,03 / 0,05 / 0,1, rangés dans `ours_s2v/`.
  nesvor    NeSVoR, défauts, sans SVoRT, CPU (scripts/run_nesvor.py) : complet et repliements.
  svrtk     SVRTK (`mirtk reconstruct`, défauts, Docker) : complet (avec --debug : poses par coupe) et repliements. Masque du premier stack conservé comme template.
  niftymic  NiftyMIC (`niftymic_reconstruct_volume`, Docker amd64), défauts SAUF `--use-masks-srr 1` : même masque cérébral pour toutes les méthodes. Volume complet seulement (coûteux ; échoue parfois).

Les stacks sont ceux de `stacks_reels()` : run 1 axial, coronal, sagittal, puis les stacks des autres runs. Repliements : pour N stacks, N repliements, N − 1 stacks pour reconstruire. Aucun identifiant n'est écrit dans le code : sujet et racine sont des arguments.

Variables d'environnement : MEDOXIDE_TEST_BIN (binaire de test figé, pour le juge) ; `--bin` : dossier du `medx` figé.

Usage : python scripts/evaluation_finale.py --racine data/svr/jeu_reel_tru_haste --sujet sub-XXXX --methode ours|ours_s2v|nesvor|svrtk|niftymic [--sortie data/svr/eval] [--bin data/svr/eval/bin]
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--racine", required=True)
parser.add_argument("--sujet", required=True)
parser.add_argument("--methode", required=True, choices=["ours", "ours_s2v", "nesvor", "svrtk", "niftymic"])
parser.add_argument("--sortie", default="data/svr/eval")
parser.add_argument("--bin", default="data/svr/eval/bin")
parser.add_argument("--delai-niftymic-h", type=float, default=4.0, help="durée maximale (heures) d'un volume NiftyMIC")
args = parser.parse_args()

racine = os.path.abspath(args.racine)
sortie = os.path.abspath(f"{args.sortie}/{args.sujet}")
dossier_bin = os.path.abspath(args.bin)
medx = f"{dossier_bin}/medx"
binaire_test = os.environ.get("MEDOXIDE_TEST_BIN", f"{dossier_bin}/medoxide_svr_test")
scripts = os.path.dirname(os.path.abspath(__file__))
env = {**os.environ, "PATH": os.environ["PATH"] + ":/usr/local/bin:/opt/homebrew/bin:/Applications/Docker.app/Contents/Resources/bin"}

session = sorted(d for d in os.listdir(f"{racine}/{args.sujet}") if d.startswith("ses-"))[0]
dossier = f"{racine}/{args.sujet}/{session}/anat"
masques = f"{racine}/derivatives/medx-fetalbet/{args.sujet}/{session}/anat"
trouves = []
for f in os.listdir(dossier):
    m = re.match(rf"{args.sujet}_{session}_acq-(trufiax|truficor|trufisag)_run-(\d+)_T2w\.nii\.gz$", f)
    if m:
        trouves.append((int(int(m.group(2)) != 1), int(m.group(2)) * 10 + ["trufiax", "truficor", "trufisag"].index(m.group(1)), f"{args.sujet}_{session}_acq-{m.group(1)}_run-{m.group(2)}"))
trouves.sort()
noms = [t[2] for t in trouves]
N = len(noms)
assert N >= 3, f"il faut au moins 3 stacks TRUFI, {N} trouvés"
stack = lambda i: f"{dossier}/{noms[i]}_T2w.nii.gz"
masque = lambda i: f"{masques}/{noms[i]}_desc-brain_mask.nii.gz"
tous = list(range(N))
repli = lambda h: [i for i in tous if i != h]


def lancer(commande, journal, delai=None):
    """Exécute `commande`, écrit sa sortie dans `journal`, lève une erreur si le code de retour n'est pas nul."""
    os.makedirs(os.path.dirname(journal), exist_ok=True)
    with open(journal, "w") as j:
        r = subprocess.run(commande, stdout=j, stderr=subprocess.STDOUT, env=env, timeout=delai)
    if r.returncode != 0:
        raise RuntimeError(f"échec ({r.returncode}) : {' '.join(map(str, commande))[:200]} ; voir {journal}")


def juger(h, volume):
    """Juge hors échantillon du stack `h` contre `volume` (écrit `<volume>.juge.tsv`)."""
    if os.path.exists(f"{volume}.juge.tsv"):
        return
    e = {**env, "MEDOXIDE_DONNEES_REELLES": racine, "MEDOXIDE_SUJET": args.sujet, "MEDOXIDE_STACK_TEST": str(h), "MEDOXIDE_VOLUME": volume}
    subprocess.run([binaire_test, "judge_a_volume_with_a_held_out_stack", "--ignored", "--nocapture"], capture_output=True, text=True, env=e)


def etape(nom, fonction):
    debut = time.time()
    try:
        fonction()
        print(f"{args.sujet} {args.methode} {nom} : ok ({time.time() - debut:.0f} s)", flush=True)
    except Exception as e:  # noqa: BLE001 : on continue avec les autres étapes
        print(f"{args.sujet} {args.methode} {nom} : ÉCHEC {e}", flush=True)


# bases sans modèle de mouvement : « grossier » = coupe-coupe grossier seul (aucun balayage fin, ni biais, ni alignement par stack, ni échelle) ; « en-tête » = poses de l'en-tête (en plus, aucun balayage grossier)
GROSSIER = ["--s2s-sweeps", "0", "--s2s-bias-rounds", "0", "--s2s-block-levels", "0", "--slice-scale-rounds", "0"]
ENTETE = [*GROSSIER, "--s2s-coarse-sweeps", "0"]


def ours():
    d = f"{sortie}/ours"
    os.makedirs(d, exist_ok=True)
    entrees = ["--input", *[stack(i) for i in tous], "--mask", *[masque(i) for i in tous]]
    for nom, options in (("full_s2s", []), ("full_s2s_ancre0", ["--s2s-anchor-stack", "0"]), ("full_entete", ENTETE), ("full_grossier", GROSSIER)):
        if not os.path.exists(f"{d}/{nom}.nii.gz"):
            etape(nom, lambda: lancer([medx, "svr", "recon", "--motion", "s2s", "--device", "cpu", *entrees, "--output", f"{d}/{nom}.nii.gz", "--poses", f"{d}/poses_{nom}.tsv", *options], f"{d}/log_{nom}.txt"))
    # repliements : la variante par défaut (d005) et celles de la frontière juge × netteté ; juge et netteté par le script de variantes
    e = {**env, "MEDOXIDE_TEST_BIN": binaire_test}
    commande = [sys.executable, "-I", f"{scripts}/benchmark_variantes_s2s.py", "--racine", racine, "--sujet", args.sujet, "--sortie", f"{d}/folds", "--medx", medx, "--variantes",
                "d005=", "d003=--alpha 0.03", "d01=--alpha 0.1", f"en=--alpha 0.05 {' '.join(ENTETE)}", f"gr=--alpha 0.05 {' '.join(GROSSIER)}"]
    os.makedirs(f"{d}/folds", exist_ok=True)
    with open(f"{d}/log_folds.txt", "w") as j:
        r = subprocess.run(commande, stdout=j, stderr=subprocess.STDOUT, env=e)
    print(f"{args.sujet} ours repliements : code {r.returncode}", flush=True)


def ours_s2v():
    d = f"{sortie}/ours_s2v"
    os.makedirs(d, exist_ok=True)
    entrees = ["--input", *[stack(i) for i in tous], "--mask", *[masque(i) for i in tous]]
    if not os.path.exists(f"{d}/full_s2v.nii.gz"):
        etape("full_s2v", lambda: lancer([medx, "svr", "recon", "--motion", "s2v", "--device", "cpu", *entrees, "--output", f"{d}/full_s2v.nii.gz", "--poses", f"{d}/poses_full_s2v.tsv"], f"{d}/log_full_s2v.txt"))
    e = {**env, "MEDOXIDE_TEST_BIN": binaire_test}
    commande = [sys.executable, "-I", f"{scripts}/benchmark_variantes_s2s.py", "--racine", racine, "--sujet", args.sujet, "--sortie", f"{d}/folds", "--medx", medx, "--variantes",
                "v005=--motion s2v", "v003=--motion s2v --alpha 0.03", "v01=--motion s2v --alpha 0.1"]
    os.makedirs(f"{d}/folds", exist_ok=True)
    with open(f"{d}/log_folds.txt", "w") as j:
        r = subprocess.run(commande, stdout=j, stderr=subprocess.STDOUT, env=e)
    print(f"{args.sujet} ours_s2v repliements : code {r.returncode}", flush=True)


def nesvor():
    d = f"{sortie}/nesvor"
    for nom, gardes, h in [("full", tous, None)] + [(f"h{h}", repli(h), h) for h in tous]:
        volume = f"{d}/{nom}/nesvor.nii.gz"
        if not os.path.exists(volume):
            etape(nom, lambda: lancer([sys.executable, "-I", f"{scripts}/run_nesvor.py", "--stacks", *[stack(i) for i in gardes], "--masques", *[masque(i) for i in gardes], "--sortie", f"{d}/{nom}"], f"{d}/{nom}.log"))
        if h is not None and os.path.exists(volume):
            juger(h, volume)


def svrtk():
    d = f"{sortie}/svrtk"
    for nom, gardes, h in [("full", tous, None)] + [(f"h{h}", repli(h), h) for h in tous]:
        volume = f"{d}/{nom}/svrtk.nii.gz"
        if not os.path.exists(volume):
            options = ["--debug"] if h is None else []
            etape(nom, lambda: lancer([sys.executable, "-I", f"{scripts}/run_svrtk.py", "--stacks", *[stack(i) for i in gardes], "--mask", masque(gardes[0]), "--sortie", f"{d}/{nom}", "--nom", "svrtk.nii.gz", *options], f"{d}/{nom}.log"))
        if h is not None and os.path.exists(volume):
            juger(h, volume)


def niftymic():
    d = f"{sortie}/niftymic"
    volume = f"{d}/srr.nii.gz"
    if os.path.exists(volume):
        return
    os.makedirs(d, exist_ok=True)
    entree = f"{d}/entree"
    os.makedirs(entree, exist_ok=True)
    for i in tous:
        shutil.copy(stack(i), f"{entree}/stack{i}.nii.gz")
        shutil.copy(masque(i), f"{entree}/stack{i}_mask.nii.gz")
    # défauts de niftymic_reconstruct_volume, sauf --use-masks-srr 1 (même masque cérébral pour toutes les méthodes) ; le premier stack est la cible par défaut
    commande = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{entree}:/in:ro", "-v", f"{d}:/out", "renbem/niftymic:latest", "niftymic_reconstruct_volume",
                "--filenames", *[f"/in/stack{i}.nii.gz" for i in tous], "--filenames-masks", *[f"/in/stack{i}_mask.nii.gz" for i in tous], "--output", "/out/srr.nii.gz", "--use-masks-srr", "1"]
    etape("full", lambda: lancer(commande, f"{d}/niftymic.log", delai=args.delai_niftymic_h * 3600))


{"ours": ours, "ours_s2v": ours_s2v, "nesvor": nesvor, "svrtk": svrtk, "niftymic": niftymic}[args.methode]()
print(f"{args.sujet} {args.methode} : terminé", flush=True)
