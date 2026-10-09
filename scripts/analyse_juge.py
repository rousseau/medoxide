"""Analyse appariée du juge hors échantillon (`benchmark_real.py`) : différences de NCC coupe par coupe entre méthodes, intervalle de confiance bootstrap à 95 %.

Pour chaque stack mis de côté, le juge a recalé ses coupes contre trois volumes (medoxide boucle, medoxide sans correction, SVRTK) : les NCC sont appariées par coupe. Une différence positive
« A − B » signifie que A prédit mieux les coupes inconnues que B. Bootstrap sur les coupes (rééchantillonnage avec remise, 10 000 tirages) ; les coupes d'un même stack sont corrélées, donc
l'intervalle est indicatif. Aucun identifiant n'est écrit dans le code.

Usage : python scripts/analyse_juge.py --sujet sub-XXXX
"""

import argparse
import csv

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sujet", required=True)
args = parser.parse_args()
dossier = f"data/svr/results/{args.sujet}"
methodes = {"boucle": "ours_loop_sans{h}", "sans_correction": "ours_header_sans{h}", "SVRTK": "svrtk_sans{h}"}


def lire(h, nom):
    lignes = list(csv.DictReader(open(f"{dossier}/{methodes[nom].format(h=h)}.nii.gz.juge.tsv"), delimiter="\t"))
    return {int(r["coupe"]): float(r["ncc"]) for r in lignes}


donnees = {h: {m: lire(h, m) for m in methodes} for h in range(3)}
rng = np.random.default_rng(0)


def ic(d):
    d = np.asarray(d)
    moyennes = [rng.choice(d, d.size).mean() for _ in range(10000)]
    return d.mean(), np.percentile(moyennes, 2.5), np.percentile(moyennes, 97.5)


print(f"NCC moyenne par coupe (hors échantillon), par stack mis de côté :")
print(f"  {'stack':>6} {'coupes':>7} " + " ".join(f"{m:>16}" for m in methodes))
tout = {m: [] for m in methodes}
for h in range(3):
    communes = sorted(set.intersection(*[set(donnees[h][m]) for m in methodes]))
    vals = {m: np.array([donnees[h][m][k] for k in communes]) for m in methodes}
    for m in methodes:
        tout[m].extend(vals[m])
    print(f"  {h:>6} {len(communes):>7} " + " ".join(f"{vals[m].mean():16.4f}" for m in methodes))
tout = {m: np.array(v) for m, v in tout.items()}
print(f"  {'tous':>6} {len(tout['boucle']):>7} " + " ".join(f"{tout[m].mean():16.4f}" for m in methodes))
print("\nDifférences appariées (NCC, moyenne [IC 95 % bootstrap]) sur toutes les coupes :")
for a, b in (("boucle", "sans_correction"), ("boucle", "SVRTK"), ("SVRTK", "sans_correction")):
    m, lo, hi = ic(tout[a] - tout[b])
    part = 100 * np.mean(tout[a] > tout[b])
    print(f"  {a} − {b:16}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] ; {a} meilleure sur {part:.0f} % des coupes")
