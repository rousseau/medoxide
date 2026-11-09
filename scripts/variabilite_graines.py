#!/usr/bin/env python3
"""Variabilité entre graines (docs/ROADMAP.md, F4.0.3) : aides communes à `benchmark_variantes_s2s.py` et `evaluation_finale.py`, et tableau final.

`--graines g1 g2 …` des deux scripts : sans l'option, comportement et arborescence inchangés ; avec l'option (même une seule valeur), les sorties vont
dans `<sortie>/graine-<g>/` et la graine est transmise par `--s2s-seed <g>` aux reconstructions.

Tableau final (ligne de commande) : pour chaque sujet (une base par sujet, qui contient les sous-dossiers `graine-<g>`), variante et mesure, la moyenne, le minimum,
le maximum et l'étendue (max − min) sur les graines ; puis, par variante et mesure, l'étendue moyenne sur les sujets et l'étendue maximale (sujet le plus variable).
Une mesure par sujet et graine = moyenne des lignes du fichier (par exemple des replis) pour la variante. Aucun identifiant n'est écrit dans le code.

Usage : python3 scripts/variabilite_graines.py --bases dossier_sujet1 dossier_sujet2 … --graines 1 2 3 --fichier resultats.tsv [--cle variante] [--sortie tableau.tsv]
"""

import argparse
import csv
import os


def valider_graines(graines):
    """Liste de graines entières positives ou nulles, sans doublon (deux graines égales écriraient dans le même dossier) ; ValueError sinon."""
    graines = list(graines)
    if not graines:
        raise ValueError("--graines : au moins une graine")
    if any((not isinstance(g, int)) or g < 0 for g in graines):
        raise ValueError(f"--graines : entiers positifs ou nuls attendus, reçu {graines}")
    if len(set(graines)) != len(graines):
        raise ValueError(f"--graines : doublon dans {graines}")
    return graines


def dossier_de_graine(sortie, graine):
    """`sortie` inchangé sans graine (`None`), sinon `<sortie>/graine-<graine>`."""
    return sortie if graine is None else f"{sortie}/graine-{graine}"


def options_avec_graine(options, graine):
    """Options (texte) suivies de `--s2s-seed <graine>` ; ValueError si elles contiennent déjà une graine (`--s2s-seed` seul ou `--s2s-seed=…`)."""
    for mot in options.split():
        if mot == "--s2s-seed" or mot.startswith("--s2s-seed="):
            raise ValueError(f"les options contiennent déjà --s2s-seed : {options!r}")
    return f"{options} --s2s-seed {graine}".strip()


def agreger(valeurs):
    """Moyenne, minimum, maximum et étendue d'une liste de valeurs."""
    return {"moyenne": sum(valeurs) / len(valeurs), "minimum": min(valeurs), "maximum": max(valeurs), "etendue": max(valeurs) - min(valeurs)}


def lire_mesures(chemin, cle):
    """`{valeur de la colonne cle : {colonne numérique : moyenne des lignes}}` ; les colonnes qui ont une cellule vide ou non numérique sont ignorées."""
    with open(chemin, newline="") as f:
        lignes = list(csv.DictReader(f, delimiter="\t"))
    sortie = {}
    for k in sorted({l[cle] for l in lignes}):
        groupe = [l for l in lignes if l[cle] == k]
        mesures = {}
        for colonne in groupe[0]:
            if colonne == cle:
                continue
            try:
                valeurs = [float(l[colonne]) for l in groupe]
            except ValueError:
                continue
            if all(v == v for v in valeurs):  # exclut les NaN
                mesures[colonne] = sum(valeurs) / len(valeurs)
        sortie[k] = mesures
    return sortie


def tableau(bases, graines, fichier, cle="variante"):
    """Lignes `(sujet, variante, mesure, moyenne, minimum, maximum, etendue)` par sujet, puis `("RÉSUMÉ", variante, mesure, étendue moyenne, étendue minimale, étendue maximale, nombre de sujets)`."""
    par_sujet = {}
    for base in bases:
        sujet = os.path.basename(os.path.normpath(base))
        par_graine = [lire_mesures(f"{dossier_de_graine(base, g)}/{fichier}", cle) for g in graines]
        for k in par_graine[0]:
            for mesure in par_graine[0][k]:
                valeurs = [m[k][mesure] for m in par_graine]
                par_sujet[(sujet, k, mesure)] = agreger(valeurs)
    lignes = [(s, k, m, a["moyenne"], a["minimum"], a["maximum"], a["etendue"]) for (s, k, m), a in sorted(par_sujet.items())]
    for k, m in sorted({(k, m) for (_, k, m) in par_sujet}):
        etendues = [a["etendue"] for (s, kk, mm), a in par_sujet.items() if kk == k and mm == m]
        lignes.append(("RÉSUMÉ", k, m, sum(etendues) / len(etendues), min(etendues), max(etendues), len(etendues)))
    return lignes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bases", nargs="+", required=True, help="un dossier par sujet, contenant les `graine-<g>`")
    p.add_argument("--graines", nargs="+", type=int, required=True)
    p.add_argument("--fichier", required=True, help="chemin du TSV dans chaque dossier de graine")
    p.add_argument("--cle", default="variante")
    p.add_argument("--sortie")
    a = p.parse_args()
    lignes = tableau(a.bases, valider_graines(a.graines), a.fichier, a.cle)
    texte = "sujet\tvariante\tmesure\tmoyenne\tminimum\tmaximum\tetendue\n" + "".join("\t".join(str(round(x, 6)) if isinstance(x, float) else str(x) for x in l) + "\n" for l in lignes)
    if a.sortie:
        with open(a.sortie, "w") as f:
            f.write(texte)
    print(texte, end="")


if __name__ == "__main__":
    main()
