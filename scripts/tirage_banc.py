#!/usr/bin/env python3
"""Tirage des 12 sujets du banc de vérité terrain (docs/ROADMAP.md, F4.0.1).

Règle fixée AVANT le tirage (aucun identifiant n'est écrit ici ; ils restent dans `data/` et dans le ROADMAP) :

1. éligibles : sessions de la cohorte CRL-2025 dont l'âge gestationnel est dans [22, 39] semaines,
   sauf les sessions listées dans `--exclure` (contrôle de segmentation échoué) ;
2. un sujet = une session : si le sujet en a plusieurs, une est tirée au hasard (sujets pris dans
   l'ordre alphabétique) ;
3. les sujets sont triés par âge de la session tirée (égalités : identifiant), coupés en 12 strates
   d'effectifs voisins (`numpy.array_split`), un sujet est tiré par strate (strates prises dans l'ordre) ;
4. générateur : `numpy.random.default_rng(graine)`, graine 20261107 ;
5. strates impaires (1, 3, … en comptant à partir de 1) : « dev » ; paires : « test ».

Sortie : un TSV (sujet, session, âge, strate, jeu). Le fichier reste sous `data/`, jamais dans git.
"""

import argparse
import csv
import sys

import numpy as np

GA_MIN, GA_MAX = 22.0, 39.0
N_STRATES = 12


def eligibles(lignes, exclus):
    """Sessions éligibles : âge dans [GA_MIN, GA_MAX] et hors de la liste d'exclusion (`sujet:session`)."""
    return [
        r
        for r in lignes
        if GA_MIN <= float(r["ga_weeks"]) <= GA_MAX and f"{r['subject_id']}:{r['session_id']}" not in exclus
    ]


def tirer(lignes, exclus, graine, n_strates=N_STRATES):
    """Applique la règle ci-dessus ; renvoie la liste de dictionnaires triée par strate."""
    rng = np.random.default_rng(graine)
    par_sujet = {}
    for r in eligibles(lignes, exclus):
        par_sujet.setdefault(r["subject_id"], []).append(r)
    choisis = []
    for sujet in sorted(par_sujet):
        sessions = sorted(par_sujet[sujet], key=lambda r: r["session_id"])
        choisis.append(sessions[int(rng.integers(len(sessions)))])
    choisis.sort(key=lambda r: (float(r["ga_weeks"]), r["subject_id"]))
    sortie = []
    for k, strate in enumerate(np.array_split(np.arange(len(choisis)), n_strates), start=1):
        r = choisis[int(strate[int(rng.integers(len(strate)))])]
        sortie.append(
            {
                "sujet": r["subject_id"],
                "session": r["session_id"],
                "age": r["ga_weeks"],
                "strate": k,
                "jeu": "dev" if k % 2 == 1 else "test",
            }
        )
    return sortie


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sujets", required=True, help="subjects.csv de la cohorte CRL")
    p.add_argument("--exclure", default="", help="sessions à exclure, `sujet:session`, séparées par des virgules")
    p.add_argument("--graine", type=int, default=20261107)
    p.add_argument("--sortie", required=True, help="TSV de sortie (sous data/)")
    a = p.parse_args()
    with open(a.sujets, newline="") as f:
        lignes = list(csv.DictReader(f))
    exclus = {x for x in a.exclure.split(",") if x}
    res = tirer(lignes, exclus, a.graine)
    with open(a.sortie, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sujet", "session", "age", "strate", "jeu"], delimiter="\t")
        w.writeheader()
        w.writerows(res)
    print(f"{len(res)} sujets écrits dans {a.sortie}", file=sys.stderr)


if __name__ == "__main__":
    main()
