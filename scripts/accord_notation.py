#!/usr/bin/env python3
"""Critère (iv) de F4.0.5 : accord entre le score FetMRQC_SR et la notation humaine en aveugle (docs/ROADMAP.md).

Entrées : `--grille` (grille.tsv remplie), `--correspondance` (correspondance_kit.tsv : identifiant, position, sujet, methode),
`--scores` (scores.tsv : sujet, methode, graine, score ; la ligne de graine 1 sert aux `s2s`, la seule ligne aux autres méthodes).
Pour chaque sujet : corrélation de Spearman (rangs moyens en cas d'égalité) entre le score et `qualite_globale` sur les méthodes du sujet ;
`rho_moyen` = moyenne des sujets où elle est définie. `--exclure id1,id2` : refait le calcul sans ces identifiants (analyse de sensibilité).
Sortie : un TSV par sujet sur la sortie standard, puis la moyenne et la notation moyenne par méthode. Aucune dépendance hors de la bibliothèque standard.
"""

import argparse
import csv
import math
import sys


def rangs(valeurs):
    """Rangs de 1 à n ; les égalités reçoivent la moyenne de leurs rangs."""
    ordre = sorted(range(len(valeurs)), key=lambda i: valeurs[i])
    r = [0.0] * len(valeurs)
    i = 0
    while i < len(ordre):
        j = i
        while j + 1 < len(ordre) and valeurs[ordre[j + 1]] == valeurs[ordre[i]]:
            j += 1
        for k in range(i, j + 1):
            r[ordre[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    """Spearman = Pearson des rangs ; NaN si l'une des séries est constante ou si n < 3."""
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    a, b = rangs(x), rangs(y)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sab = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    saa = sum((u - ma) ** 2 for u in a)
    sbb = sum((v - mb) ** 2 for v in b)
    if saa == 0 or sbb == 0:
        return float("nan")
    return sab / math.sqrt(saa * sbb)


def lire_tsv(chemin):
    with open(chemin, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def score_de(scores, sujet, methode):
    lignes = [r for r in scores if r["sujet"] == sujet and r["methode"] == methode]
    if methode == "s2s":
        lignes = [r for r in lignes if r["graine"] == "1"]
    if len(lignes) != 1:
        raise SystemExit(f"{sujet} {methode} : {len(lignes)} lignes de score (attendu : 1)")
    return float(lignes[0]["score"])


def accord(grille, corr, scores, exclure=()):
    """Retourne ({sujet: (n, rho)}, rho_moyen, nombre de sujets définis, {méthode: note moyenne})."""
    notes = {r["identifiant"]: int(r["qualite_globale"]) for r in grille}
    par_sujet = {}
    for c in corr:
        if c["identifiant"] in exclure:
            continue
        par_sujet.setdefault(c["sujet"], []).append((c["methode"], notes[c["identifiant"]], score_de(scores, c["sujet"], c["methode"])))
    res = {}
    for s, lst in sorted(par_sujet.items()):
        res[s] = (len(lst), spearman([x[2] for x in lst], [float(x[1]) for x in lst]))
    definis = [v[1] for v in res.values() if not math.isnan(v[1])]
    moyen = sum(definis) / len(definis) if definis else float("nan")
    par_methode = {}
    for lst in par_sujet.values():
        for m, n, _ in lst:
            par_methode.setdefault(m, []).append(n)
    return res, moyen, len(definis), {m: sum(v) / len(v) for m, v in sorted(par_methode.items())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grille", required=True)
    p.add_argument("--correspondance", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--exclure", default="")
    a = p.parse_args(argv)
    grille, corr, scores = lire_tsv(a.grille), lire_tsv(a.correspondance), lire_tsv(a.scores)
    if sorted(r["identifiant"] for r in grille) != sorted(r["identifiant"] for r in corr):
        raise SystemExit("identifiants de la grille et de la correspondance différents")
    exclure = {x for x in a.exclure.split(",") if x}
    res, moyen, nd, par_m = accord(grille, corr, scores, exclure)
    print("sujet\tn\trho")
    for s, (n, rho) in res.items():
        print(f"{s}\t{n}\t{rho:.3f}")
    print(f"rho_moyen\t{nd}\t{moyen:.3f}")
    print("methode\tnote_moyenne")
    for m, v in par_m.items():
        print(f"{m}\t{v:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
