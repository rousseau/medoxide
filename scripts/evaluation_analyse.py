"""Évaluation finale, analyse : calcule les cinq familles de critères (Q1 à Q5) de docs/ROADMAP.md (§ F1 et F1-éval) pour notre coupe-coupe contre SVRTK, NeSVoR et NiftyMIC, puis applique la règle de décision.

Entrées : les sorties de evaluation_finale.py et evaluation_alignement.py dans <sortie>/<sujet>/. Les sujets agités sont ceux dont la concordance d'en-tête est < 0,45 (--calmes : liste des sujets calmes ; les autres sont agités).

  Q1  juge hors échantillon (3 repliements ou plus, moyenne par sujet) À NETTETÉ ÉGALE : frontière juge = f(netteté) de notre méthode (α = 0,03 / 0,05 / 0,1), interpolée à la netteté de chaque concurrent. Tolérance −0,005.
  Q2  concordance aux intersections sur les paires SANS le stack 0, que notre coupe-coupe ancré sur le stack 0 n'a pas utilisées (mêmes coupes pour tous ; NiftyMIC sur les coupes qu'il garde). Tolérance −0,01.
  Q3  reproductibilité de la segmentation : Dice moyen entre repliements (étiquettes cortex, matière blanche, ventricules, cervelet, noyaux gris, thalamus). Tolérance −0,01. Les recalages de corrélation < 0,9 sont comptés.
  Q4  trous : fraction de voxels nuls dans le cerveau segmenté (volumes complets, repère de l'atlas). Tolérance +0,1 point.
  Q5  complétude anatomique sous les DEUX protocoles d'alignement (partagé et indépendant, candidat 0) : réussi/échec par sujet.

Règle (par concurrent) : une famille est GAGNÉE si notre méthode est au moins égale (tolérance) sur ≥ 80 % des sujets évaluables ET strictement meilleure en moyenne ; NETTEMENT PERDUE si elle dépasse la tolérance sur ≥ 2 sujets. « Mieux » = ≥ 3 familles gagnées dont Q5, aucune nettement perdue. Pour NiftyMIC, Q1 et Q3 sont non évaluables (pas de repliements) : la variante limitée (≥ 2 gagnées dont Q5, aucune nettement perdue) est aussi donnée.

Usage : python scripts/evaluation_analyse.py --sujets sub-A sub-B ... --calmes sub-A [--sortie data/svr/eval] [--racine data/svr/jeu_reel_tru_haste]
"""

import argparse
import csv
import glob
import itertools
import os
import re
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controle_anatomique import controle  # noqa: E402
from intersection_consistency import Jeu, concordance, lire_poses  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--sujets", nargs="+", required=True)
parser.add_argument("--calmes", nargs="*", default=[])
parser.add_argument("--sortie", default="data/svr/eval")
parser.add_argument("--racine", default="data/svr/jeu_reel_tru_haste")
args = parser.parse_args()
scripts = os.path.dirname(os.path.abspath(__file__))
CONCURRENTS = ["SVRTK", "NeSVoR", "NiftyMIC"]
LABELS = [3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17]
E = lambda s: os.path.abspath(f"{args.sortie}/{s}")


def stacks_du_sujet(sujet):
    racine = os.path.abspath(args.racine)
    session = sorted(d for d in os.listdir(f"{racine}/{sujet}") if d.startswith("ses-"))[0]
    dossier = f"{racine}/{sujet}/{session}/anat"
    masques = f"{racine}/derivatives/medx-fetalbet/{sujet}/{session}/anat"
    t = []
    for f in os.listdir(dossier):
        m = re.match(rf"{sujet}_{session}_acq-(trufiax|truficor|trufisag)_run-(\d+)_T2w\.nii\.gz$", f)
        if m:
            t.append((int(int(m.group(2)) != 1), int(m.group(2)) * 10 + ["trufiax", "truficor", "trufisag"].index(m.group(1)), f"{sujet}_{session}_acq-{m.group(1)}_run-{m.group(2)}"))
    t.sort()
    return [f"{dossier}/{x[2]}_T2w.nii.gz" for x in t], [f"{masques}/{x[2]}_desc-brain_mask.nii.gz" for x in t]


def nettete(chemin):
    v = np.asarray(nib.load(chemin).dataobj, dtype=np.float64)
    tissu = v > 0.05 * np.percentile(v[v > 0], 99.5)
    interieur = binary_erosion(tissu, iterations=3)
    g = np.sqrt(sum(np.gradient(gaussian_filter(v, 1.0), axis=a) ** 2 for a in range(3)))
    return float(g[interieur].mean() / np.median(v[interieur]))


def juge_moyen(volume):
    f = f"{volume}.juge.tsv"
    if not os.path.exists(f):
        return None
    v = [float(r["ncc"]) for r in csv.DictReader(open(f), delimiter="\t")]
    return float(np.mean(v)) if v else None


# ------------------------------------------------------------------ Q1
def q1(s):
    """{concurrent: Δ} (Δ = juge de notre méthode à la netteté du concurrent − juge du concurrent) ; None si non défini."""
    lignes = list(csv.DictReader(open(f"{E(s)}/ours/folds/resultats.tsv"), delimiter="\t"))
    pts = []
    for v in ("d003", "d005", "d01"):
        l = [r for r in lignes if r["variante"] == v]
        if l:
            pts.append((np.mean([float(r["nettete"]) for r in l]), np.mean([float(r["ncc_moyenne"]) for r in l])))
    pts.sort()
    S, J = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    out = {}
    for nom, dossier, fichier in (("SVRTK", "svrtk", "svrtk.nii.gz"), ("NeSVoR", "nesvor", "nesvor.nii.gz")):
        cs, cj = [], []
        for h in range(len({r["stack_teste"] for r in lignes})):
            vol = f"{E(s)}/{dossier}/h{h}/{fichier}"
            j = juge_moyen(vol) if os.path.exists(vol) else None
            if j is not None:
                cs.append(nettete(vol)); cj.append(j)
        if not cs:
            out[nom] = None
            continue
        cs, cj = float(np.mean(cs)), float(np.mean(cj))
        out[nom] = float(np.interp(cs, S, J)) - cj if S[0] <= cs <= S[-1] else None
    out["NiftyMIC"] = "n/a"
    return out


# ------------------------------------------------------------------ Q2
def poses_concurrents(s):
    """Écrit (une fois) les poses de SVRTK et NiftyMIC au format medoxide dans <sujet>/poses/ et renvoie les chemins."""
    d = f"{E(s)}/poses"
    os.makedirs(d, exist_ok=True)
    st, mk = stacks_du_sujet(s)
    if not os.path.exists(f"{d}/svrtk.tsv") and os.path.exists(f"{E(s)}/svrtk/full/svrtk.nii.gz"):
        subprocess.run([sys.executable, "-I", f"{scripts}/read_svrtk_poses.py", "--debug-dir", f"{E(s)}/svrtk/full", "--volume", f"{E(s)}/svrtk/full/svrtk.nii.gz", "--stacks", *st, "--masques", *mk, "--sortie", f"{d}/svrtk.tsv"],
                       capture_output=True, text=True)
    if not os.path.exists(f"{d}/niftymic.tsv") and os.path.exists(f"{E(s)}/niftymic/srr.nii.gz"):
        n = len(st)
        subprocess.run([sys.executable, "-I", f"{scripts}/read_niftymic_poses.py", "--dossier", f"{E(s)}/niftymic", "--stacks", *[f"{E(s)}/niftymic/entree/stack{i}.nii.gz" for i in range(n)],
                        "--masques", *[f"{E(s)}/niftymic/entree/stack{i}_mask.nii.gz" for i in range(n)], "--sortie", f"{d}/niftymic.tsv"], capture_output=True, text=True)
    return {"SVRTK": f"{d}/svrtk.tsv", "NeSVoR": f"{E(s)}/nesvor/full/poses_nesvor.tsv", "NiftyMIC": f"{d}/niftymic.tsv", "ancré": f"{E(s)}/ours/poses_full_s2s_ancre0.tsv", "s2s": f"{E(s)}/ours/poses_full_s2s.tsv"}


def q2(s):
    """{méthode: concordance moyenne sur les paires sans le stack 0} sur les coupes communes (NiftyMIC : sur ses coupes gardées)."""
    chemins = poses_concurrents(s)
    st, mk = stacks_du_sujet(s)
    jeu = Jeu(st, mk)
    ident = lambda m: np.allclose(m, np.eye(4), atol=1e-12)
    poses = {}
    for nom, f in chemins.items():
        if os.path.exists(f):
            poses[nom] = {k: m for k, m in lire_poses(f).items() if not ident(m)}
    complets = [n for n in poses if n != "NiftyMIC"]
    coupes = sorted(set.intersection(*[set(poses[n]) for n in complets]))
    out, nb = {}, {}
    for nom in poses:
        utilisees = [k for k in coupes if k in poses[nom]] if nom == "NiftyMIC" else coupes
        c = concordance(jeu, poses[nom], utilisees, hors_stack=0)
        out[nom] = float(c.mean()) if len(c) else None
        nb[nom] = len(utilisees)
    return out, nb


# ------------------------------------------------------------------ Q3
def dice(a, b, k):
    x, y = a == k, b == k
    return 2 * (x & y).sum() / max(x.sum() + y.sum(), 1)


def q3(s):
    """{méthode: (Dice moyen entre repliements, nombre de recalages de corrélation < 0,9)}."""
    A = f"{E(s)}/atlas"
    corr = {}
    if os.path.exists(f"{A}/alignement.tsv"):
        for r in csv.DictReader(open(f"{A}/alignement.tsv"), delimiter="\t"):
            try:
                corr[r["etiquette"]] = float(r["correlation_vers_reference"])
            except ValueError:
                pass
    out = {}
    for nom, pref in (("ours", "s2s"), ("SVRTK", "svrtk"), ("NeSVoR", "nesvor")):
        fs = sorted(glob.glob(f"{A}/seg_{pref}_h*.nii.gz"), key=lambda f: int(re.search(r"_h(\d+)\.nii\.gz", f).group(1)))
        segs = [np.asarray(nib.load(f).dataobj).astype(np.int16) for f in fs if not f.endswith(".log")]
        if len(segs) < 2:
            out[nom] = None
            continue
        d = [np.mean([dice(a, b, k) for k in LABELS]) for a, b in itertools.combinations(segs, 2)]
        faibles = sum(1 for k, v in corr.items() if k.startswith(f"{pref}_h") and v < 0.9)
        out[nom] = (float(np.mean(d)), faibles)
    return out


# ------------------------------------------------------------------ Q4
def q4(s):
    """{méthode: fraction de trous (%)} dans le cerveau segmenté, volumes complets dans le repère de l'atlas (même alignement que la référence)."""
    A = f"{E(s)}/atlas"
    out = {}
    for nom, etiquette in (("ours", "s2s"), ("SVRTK", "svrtk"), ("NeSVoR", "nesvor"), ("NiftyMIC", "niftymic")):
        fi, fs = f"{A}/{etiquette}.nii.gz", f"{A}/seg_{etiquette}.nii.gz"
        if not (os.path.exists(fi) and os.path.exists(fs)):
            out[nom] = None
            continue
        zone = binary_closing(np.asarray(nib.load(fs).dataobj) > 0, iterations=4)
        im = np.asarray(nib.load(fi).dataobj)
        out[nom] = float(100 * ((im <= 0) & zone).sum() / max(zone.sum(), 1))
    return out


# ------------------------------------------------------------------ Q5
def q5(s):
    """{méthode: (réussi sous le protocole partagé, réussi sous l'indépendant)} ; notre méthode : la référence (les deux protocoles coïncident)."""
    A = f"{E(s)}/atlas"
    partage = {}
    if os.path.exists(f"{A}/alignement.tsv"):
        for r in csv.DictReader(open(f"{A}/alignement.tsv"), delimiter="\t"):
            partage[r["etiquette"]] = r["controle"] == "OK"
    out = {}
    ref_ok = os.path.exists(f"{A}/seg_s2s.nii.gz") and open(f"{A}/reference.txt").read() != ""
    out["ours"] = (ref_ok, ref_ok)
    for nom, etiquette in (("SVRTK", "svrtk"), ("NeSVoR", "nesvor"), ("NiftyMIC", "niftymic")):
        existe = os.path.exists(f"{E(s)}/{'niftymic/srr' if etiquette == 'niftymic' else etiquette + '/full/' + etiquette}.nii.gz")
        if not existe:
            out[nom] = (False, False)  # échec de la méthode : perdu pour ce sujet
            continue
        p = partage.get(etiquette, False)
        # protocole indépendant : candidat 0 (corrélation maximale)
        seg = f"{E(s)}/atlas_indep/{etiquette}/seg_cand_0.nii.gz"
        i = False
        if os.path.exists(seg):
            i = controle(seg)[0]
        out[nom] = (p, i)
    return out


def main():
    classe = {s: ("calme" if s in args.calmes else "agité") for s in args.sujets}
    res = {s: {} for s in args.sujets}
    for s in args.sujets:
        res[s]["q1"] = q1(s) if os.path.exists(f"{E(s)}/ours/folds/resultats.tsv") else {}
        res[s]["q2"] = q2(s) if os.path.exists(f"{E(s)}/ours/poses_full_s2s_ancre0.tsv") else ({}, {})
        res[s]["q3"] = q3(s)
        res[s]["q4"] = q4(s)
        res[s]["q5"] = q5(s)
    # affichage par famille
    print("Q1 : Δ juge à netteté égale (notre méthode − concurrent)")
    for s in args.sujets:
        print(f"  {s} ({classe[s]}) :", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[s]["q1"].items()})
    print("Q2 : concordance sans stack 0 :")
    for s in args.sujets:
        print(f"  {s} ({classe[s]}) :", {k: (round(v, 3) if v is not None else None) for k, v in res[s]["q2"][0].items()}, "coupes", res[s]["q2"][1])
    print("Q3 : Dice entre repliements (recalages faibles) :")
    for s in args.sujets:
        print(f"  {s} ({classe[s]}) :", {k: (round(v[0], 3), v[1]) if v else None for k, v in res[s]["q3"].items()})
    print("Q4 : trous (%) :")
    for s in args.sujets:
        print(f"  {s} ({classe[s]}) :", {k: (round(v, 3) if v is not None else None) for k, v in res[s]["q4"].items()})
    print("Q5 : (partagé, indépendant) :")
    for s in args.sujets:
        print(f"  {s} ({classe[s]}) :", res[s]["q5"])
    # décision
    print("\nDÉCISION")
    for c in CONCURRENTS:
        familles = {}
        # Q1
        d = [res[s]["q1"].get(c) for s in args.sujets]
        d = [x for x in d if isinstance(x, float)]
        familles["Q1"] = ("n/a", "") if not d else (bilan(d, -0.005), f"{len(d)} sujets, moyenne {np.mean(d):+.3f}")
        # Q2
        d = []
        for s in args.sujets:
            q, _ = res[s]["q2"]
            if q.get("ancré") is not None and q.get(c) is not None:
                d.append(q["ancré"] - q[c])
        familles["Q2"] = ("n/a", "") if not d else (bilan(d, -0.01), f"{len(d)} sujets, moyenne {np.mean(d):+.3f}")
        # Q3
        d = [res[s]["q3"]["ours"][0] - res[s]["q3"][c][0] for s in args.sujets if res[s]["q3"].get("ours") and res[s]["q3"].get(c)] if c != "NiftyMIC" else []
        familles["Q3"] = ("n/a", "") if not d else (bilan(d, -0.01), f"{len(d)} sujets, moyenne {np.mean(d):+.3f}")
        # Q4 : trous, Δ = concurrent − notre méthode (positif : nous avons moins de trous), tolérance −0,1 point
        d = [res[s]["q4"][c] - res[s]["q4"]["ours"] for s in args.sujets if res[s]["q4"].get(c) is not None and res[s]["q4"].get("ours") is not None]
        familles["Q4"] = ("n/a", "") if not d else (bilan(d, -0.1), f"{len(d)} sujets, moyenne {np.mean(d):+.3f} point")
        # Q5 : 1 si réussi sous les deux protocoles
        d = [float(all(res[s]["q5"]["ours"])) - float(all(res[s]["q5"][c])) for s in args.sujets]
        familles["Q5"] = (bilan(d, 0.0), f"réussis sous les deux protocoles : nous {sum(all(res[s]['q5']['ours']) for s in args.sujets)}/{len(args.sujets)}, {c} {sum(all(res[s]['q5'][c]) for s in args.sujets)}/{len(args.sujets)}")
        gagnees = [q for q, (v, _) in familles.items() if v == "GAGNÉE"]
        perdues = [q for q, (v, _) in familles.items() if v == "NETTEMENT PERDUE"]
        evaluables = [q for q, (v, _) in familles.items() if v != "n/a"]
        strict = len(gagnees) >= 3 and "Q5" in gagnees and not perdues
        limite = len(gagnees) >= 2 and "Q5" in gagnees and not perdues
        print(f"\n contre {c} :")
        for q, (v, detail) in familles.items():
            print(f"   {q} : {v} {detail}")
        print(f"   → gagnées {gagnees}, nettement perdues {perdues} ; évaluables {evaluables} ; règle stricte : {'ATTEINTE' if strict else 'non atteinte'} ; règle limitée aux familles évaluables : {'ATTEINTE' if limite else 'non atteinte'}")


def bilan(d, tolerance):
    """GAGNÉE, NETTEMENT PERDUE, ou ÉGALE/PERDUE selon la règle (écart d = nous − concurrent par sujet ; au moins égale si d ≥ tolérance)."""
    d = np.array(d)
    egal = (d >= tolerance).mean() >= 0.8
    if egal and d.mean() > 0:
        return "GAGNÉE"
    if (d < tolerance).sum() >= 2:
        return "NETTEMENT PERDUE"
    return "non gagnée"


main()
