"""Évaluation finale, étape d'alignement : met tous les volumes d'un sujet dans l'espace de l'atlas et les segmente avec BOUNTI (protocole figé dans docs/ROADMAP.md, § F1 et F1-éval).

Deux protocoles, toujours rapportés ensemble :
  PARTAGÉ      la référence (notre coupe-coupe, volume complet) est orientée sur l'atlas (recherche dense d'orientations, `atlas_orientation_dense.py`) ; les trois meilleurs candidats sont segmentés et le premier (corrélation maximale) qui passe le contrôle anatomique est retenu ; tous les autres volumes (complets et de repliement) sont recalés rigidement sur la référence puis placés avec le même alignement (`atlas_aligne_sur_reference.py`).
  INDÉPENDANT  chaque volume complet de chaque méthode est orienté seul sur l'atlas (recherche dense, masque BET de BOUNTI sur son propre volume) ; les trois meilleurs candidats sont segmentés.
Contrôle anatomique (la symétrie gauche-droite ne prouve pas l'orientation) : cervelet ≥ 2 mL, centre du cervelet en arrière et en bas du centre du cerveau (y < 0 et z < 0 dans le RAS de l'atlas), tronc cérébral en bas (z < 0).

Sorties : <sortie>/<sujet>/atlas/ (partagé : s2s.nii.gz = référence (notre coupe-coupe, volume complet), s2v, entete, grossier, svrtk, nesvor, niftymic et s2s_h<k>, s2v_h<k>, svrtk_h<k>, nesvor_h<k> = repliements, ref_vers_atlas.tfm, <étiquette>.nii.gz, seg_<étiquette>.nii.gz, alignement.tsv avec la corrélation de chaque recalage) et atlas_indep/<méthode>/ (cand_k.nii.gz, seg_cand_k.nii.gz, dense.log). Reprise : les fichiers présents ne sont pas recalculés.

Usage : python scripts/evaluation_alignement.py --sujet sub-XXXX [--sortie data/svr/eval] [--bin data/svr/eval/bin] [--etapes ref,partage,indep]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--sujet", required=True)
parser.add_argument("--sortie", default="data/svr/eval")
parser.add_argument("--bin", default="data/svr/eval/bin")
parser.add_argument("--etapes", default="ref,partage,indep")
args = parser.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controle_anatomique import controle  # noqa: E402

E = os.path.abspath(f"{args.sortie}/{args.sujet}")
medx = os.path.abspath(f"{args.bin}/medx")
scripts = os.path.dirname(os.path.abspath(__file__))
A = f"{E}/atlas"
os.makedirs(A, exist_ok=True)
etapes = set(args.etapes.split(","))


def cmd(commande, journal):
    with open(journal, "w") as j:
        r = subprocess.run(commande, stdout=j, stderr=subprocess.STDOUT)
    return r.returncode


def segmente(volume, seg, masque=None):
    """BOUNTI sur un volume déjà orienté ; sans effet si la sortie existe."""
    if not os.path.exists(seg):
        c = [medx, "bounti", "--input", volume, "--output", seg] + (["--brain-mask", masque] if masque else [])
        cmd(c, f"{seg}.log")


def masque_bet(volume, dossier):
    """Masque cérébral de BOUNTI sur un volume NON orienté (le réseau d'extraction y suffit pour guider le recalage)."""
    m = f"{dossier}/masque.nii.gz"
    if not os.path.exists(m):
        os.makedirs(dossier, exist_ok=True)
        cmd([medx, "bounti", "--input", volume, "--output", f"{dossier}/seg_brut.nii.gz", "--brain-mask", m], f"{dossier}/bet.log")
    return m


def dense(volume, dossier):
    """Recherche dense d'orientations ; renvoie [(corrélation, k)] des candidats raffinés."""
    m = masque_bet(volume, dossier)
    if not os.path.exists(f"{dossier}/cand_0.nii.gz"):
        cmd([sys.executable, "-I", f"{scripts}/atlas_orientation_dense.py", volume, m, dossier], f"{dossier}/dense.log")
    return [(float(c), int(k)) for k, c in re.findall(r"cand (\d+) corr ([\d.]+)", open(f"{dossier}/dense.log").read())]


# volumes à placer : étiquette -> chemin (complets puis repliements)
complets = {"entete": f"{E}/ours/full_entete.nii.gz", "grossier": f"{E}/ours/full_grossier.nii.gz", "s2v": f"{E}/ours_s2v/full_s2v.nii.gz", "svrtk": f"{E}/svrtk/full/svrtk.nii.gz", "nesvor": f"{E}/nesvor/full/nesvor.nii.gz", "niftymic": f"{E}/niftymic/srr.nii.gz",
            "ancre0": f"{E}/ours/full_s2s_ancre0.nii.gz"}
replis = {}
for f in sorted(glob.glob(f"{E}/ours/folds/d005_h*.nii.gz")):
    if f.endswith(".juge.tsv"):
        continue
    h = int(re.search(r"_h(\d+)\.nii\.gz$", f).group(1))
    replis[f"s2s_h{h}"] = f
    replis[f"s2v_h{h}"] = f"{E}/ours_s2v/folds/v005_h{h}.nii.gz"
    replis[f"svrtk_h{h}"] = f"{E}/svrtk/h{h}/svrtk.nii.gz"
    replis[f"nesvor_h{h}"] = f"{E}/nesvor/h{h}/nesvor.nii.gz"

if "ref" in etapes:
    ref = f"{E}/ours/full_s2s.nii.gz"
    cands = dense(ref, A)
    choisi = None
    for k, (corr, _) in enumerate(cands[:3]):
        seg = f"{A}/seg_cand_{k}.nii.gz"
        segmente(f"{A}/cand_{k}.nii.gz", seg)
        ok, cb = controle(seg)
        print(f"{args.sujet} référence candidat {k} : corrélation {corr:.3f}, cervelet {cb:.1f} mL, contrôle {'OK' if ok else 'ÉCHEC'}", flush=True)
        if ok and choisi is None:
            choisi = k
    if choisi is None:
        print(f"{args.sujet} référence : AUCUN candidat ne passe le contrôle anatomique", flush=True)
        choisi = 0
    shutil.copy(f"{A}/cand_{choisi}.nii.gz", f"{A}/s2s.nii.gz")
    shutil.copy(f"{A}/cand_{choisi}.tfm", f"{A}/ref_vers_atlas.tfm")
    shutil.copy(f"{A}/seg_cand_{choisi}.nii.gz", f"{A}/seg_s2s.nii.gz") if os.path.exists(f"{A}/seg_cand_{choisi}.nii.gz") else None
    open(f"{A}/reference.txt", "w").write(f"candidat retenu : {choisi} (corrélation {cands[choisi][0] if choisi < len(cands) else float('nan'):.3f})\n")

if "partage" in etapes:
    lignes = ["etiquette\tcorrelation_vers_reference\tcontrole\tcervelet_ml"]
    for etiquette, chemin in {**complets, **replis}.items():
        if not os.path.exists(chemin):
            lignes.append(f"{etiquette}\tabsent\t\t")
            continue
        sortie = f"{A}/{etiquette}.nii.gz"
        if not os.path.exists(sortie):
            journal = f"{A}/{etiquette}.reg.txt"
            cmd([sys.executable, "-I", f"{scripts}/atlas_aligne_sur_reference.py", A, chemin, sortie], journal)
        reg = open(f"{A}/{etiquette}.reg.txt").read().strip().splitlines() if os.path.exists(f"{A}/{etiquette}.reg.txt") else []
        corr = reg[-1].split()[-1] if reg else "?"
        seg = f"{A}/seg_{etiquette}.nii.gz"
        segmente(sortie, seg)
        ok, cb = controle(seg)
        lignes.append(f"{etiquette}\t{corr}\t{'OK' if ok else 'ECHEC'}\t{cb:.1f}")
        print(f"{args.sujet} partagé {etiquette} : corrélation {corr}, contrôle {'OK' if ok else 'ÉCHEC'} (cervelet {cb:.1f} mL)", flush=True)
    open(f"{A}/alignement.tsv", "w").write("\n".join(lignes) + "\n")

if "indep" in etapes:
    for nom, chemin in {"entete": complets["entete"], "s2v": complets["s2v"], "svrtk": complets["svrtk"], "nesvor": complets["nesvor"], "niftymic": complets["niftymic"]}.items():
        if not os.path.exists(chemin):
            continue
        d = f"{E}/atlas_indep/{nom}"
        cands = dense(chemin, d)
        res = []
        for k in range(min(3, len(cands))):
            segmente(f"{d}/cand_{k}.nii.gz", f"{d}/seg_cand_{k}.nii.gz")
            ok, cb = controle(f"{d}/seg_cand_{k}.nii.gz")
            res.append(f"c{k} {'OK' if ok else 'ÉCHEC'} ({cb:.1f})")
        print(f"{args.sujet} indépendant {nom} : corrélation {cands[0][0] if cands else float('nan'):.3f} ; " + " ; ".join(res), flush=True)
print(f"{args.sujet} alignement : terminé", flush=True)
