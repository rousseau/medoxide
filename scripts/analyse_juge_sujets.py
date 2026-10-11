"""Critère A1 : qualité de reconstruction sur plusieurs sujets réels, avec le SUJET comme unité statistique (les repliements et les coupes d'un même sujet ne sont pas indépendants).

Pour chaque sujet, le juge hors échantillon (`benchmark_real.py`, `judge_a_volume_with_a_held_out_stack`) a recalé les coupes de chaque stack mis de côté contre le volume reconstruit avec les deux autres, pour
chaque méthode. Ici : NCC moyenne par coupe sur les trois repliements (coupes communes à toutes les méthodes) -> UNE valeur par sujet et par méthode ; puis, pour chaque paire de méthodes, la
différence moyenne et le nombre de sujets où la première gagne. Les identifiants de sujets sont passés en arguments (aucun dans le code). Fichiers attendus dans `data/svr/results/<sujet>/` :
`ours_loop_loso_sans{h}.nii.gz.juge.tsv` (sans la coupe), `ours_loop_sans{h}.nii.gz.juge.tsv` (boucle actuelle), `ours_header_sans{h}.nii.gz.juge.tsv` (sans correction), `svrtk_sans{h}.nii.gz.juge.tsv`.
Option --substitut sujet:h=fichier : remplacer le fichier « sans la coupe » d'un repliement (cas d'un fichier nommé autrement).

Usage : python scripts/analyse_juge_sujets.py --sujets sub-A sub-B ... [--substitut sub-X:2=ours_loop_diag01_S]
"""

import argparse
import csv

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sujets", nargs="+", required=True)
parser.add_argument("--substitut", action="append", default=[])
args = parser.parse_args()
subst = {}
for s in args.substitut:
    cle, f = s.split("=", 1)
    sujet, h = cle.split(":")
    subst[(sujet, int(h))] = f

methodes = {"sans_la_coupe": "ours_loop_loso_sans{h}", "boucle_actuelle": "ours_loop_sans{h}", "sans_correction": "ours_header_sans{h}", "SVRTK": "svrtk_sans{h}"}


def lire(sujet, nom, h):
    f = subst.get((sujet, h), methodes[nom].format(h=h)) if nom == "sans_la_coupe" else methodes[nom].format(h=h)
    return {int(r["coupe"]): float(r["ncc"]) for r in csv.DictReader(open(f"data/svr/results/{sujet}/{f}.nii.gz.juge.tsv"), delimiter="\t")}


par_sujet = {}
for sujet in args.sujets:
    v = {m: [] for m in methodes}
    for h in range(3):
        d = {m: lire(sujet, m, h) for m in methodes}
        for k in sorted(set.intersection(*[set(x) for x in d.values()])):
            for m in methodes:
                v[m].append(d[m][k])
    par_sujet[sujet] = {m: float(np.mean(x)) for m, x in v.items()} | {"n": len(v["SVRTK"])}

print(f"{'sujet':10} {'coupes':>6} " + " ".join(f"{m:>16}" for m in methodes))
for sujet, r in par_sujet.items():
    print(f"{sujet:10} {r['n']:6d} " + " ".join(f"{r[m]:16.4f}" for m in methodes))
moy = {m: np.mean([r[m] for r in par_sujet.values()]) for m in methodes}
print(f"{'moyenne':10} {'':6} " + " ".join(f"{moy[m]:16.4f}" for m in methodes))
print("\nPar sujet (unité statistique) :")
n = len(par_sujet)
for a, b in (("sans_la_coupe", "SVRTK"), ("sans_la_coupe", "sans_correction"), ("sans_la_coupe", "boucle_actuelle"), ("boucle_actuelle", "SVRTK"), ("boucle_actuelle", "sans_correction"), ("SVRTK", "sans_correction")):
    d = np.array([r[a] - r[b] for r in par_sujet.values()])
    print(f"  {a} − {b:16}: différence moyenne {d.mean():+.4f} (min {d.min():+.4f}, max {d.max():+.4f}) ; {a} gagne chez {int((d > 0).sum())} sujet(s) sur {n}")
