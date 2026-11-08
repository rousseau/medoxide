#!/usr/bin/env python3
"""Pilotage de la mesure de référence de la variabilité entre graines (docs/ROADMAP.md, F4.0.3, point 0.3c). Aucune mesure nouvelle : lance, range, appelle les scripts existants.

Rangement sous `<base>/` : `s2s/<sujet>/graine-<g>/` (repliements : `benchmark_variantes_s2s.py --graines`), `s2s/<sujet>/complet_g<g>.nii.gz` et `poses_g<g>.tsv` (volume complet), de même pour `s2v/<sujet>/` (une seule exécution, graine 1,
car s2v ne dépend pas de la graine : 0.3b).
  complets : reconstruction complète avec poses (s2s : graines données ; s2v : graine 1).
  liens    : prépare `s2s/<sujet>/bounti/` pour `analyse_reproductibilite_segmentation.py` : `full.nii.gz` (s2s graine 1) et `gs<g>_h<h>.nii.gz` (volumes de repli de la variante par défaut).
  q2       : concordance aux intersections (`analyse_concordance_sujets.py`, sans modification) pour s2s à chaque graine et s2v, puis l'étendue entre graines de s2s par sujet et lecture.

Aucun identifiant n'est écrit dans le code : racine et sujets sont des arguments.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variabilite_graines import agreger, dossier_de_graine, valider_graines  # noqa: E402

VARIANTE_S2S = "d005"


def chemins_complets(base, methode, sujet, graine):
    """`(volume, poses)` de la reconstruction complète."""
    d = f"{base}/{methode}/{sujet}"
    return f"{d}/complet_g{graine}.nii.gz", f"{d}/poses_g{graine}.tsv"


def liens_bounti(base, sujet, graines, replis):
    """Liste `(lien, cible relative au dossier du lien)` de `s2s/<sujet>/bounti/` : `full.nii.gz` (graine la plus petite) et `gs<g>_h<h>.nii.gz` pour chaque graine et chaque repli."""
    graines = valider_graines(graines)
    liens = [("full.nii.gz", f"../complet_g{min(graines)}.nii.gz")]
    for g in graines:
        for h in replis:
            liens.append((f"gs{g}_h{h}.nii.gz", f"../graine-{g}/{VARIANTE_S2S}_h{h}.nii.gz"))
    return liens


def replis_presents(base, sujet, graines):
    """Indices de repli présents pour TOUTES les graines (volume de la variante par défaut)."""
    par_graine = []
    for g in graines:
        fichiers = glob.glob(f"{dossier_de_graine(f'{base}/s2s/{sujet}', g)}/{VARIANTE_S2S}_h[0-9]*.nii.gz")
        par_graine.append({int(re.search(r"_h(\d+)\.nii\.gz$", f).group(1)) for f in fichiers})
    return sorted(set.intersection(*par_graine)) if par_graine else []


def etendues_q2(chemin_tsv, graines):
    """`{(sujet, mesure): agrégat des colonnes s2s_g<g>}` d'après le TSV de `analyse_concordance_sujets.py` (lignes `sujet`, `mesure`, une colonne par méthode) ; les lignes `coupes` sont ignorées."""
    with open(chemin_tsv, newline="") as f:
        lignes = list(csv.reader(f, delimiter="\t"))
    entete = lignes[0]
    colonnes = [entete.index(f"s2s_g{g}") for g in graines]
    sortie = {}
    for l in lignes[1:]:
        if l[1] in ("toutes", "hors_stack_0"):
            sortie[(l[0], l[1])] = agreger([float(l[c]) for c in colonnes])
    return sortie


def cmd_complets(a):
    from dependance_graine import piles  # mêmes stacks et masques que 0.3b (run 1 ax, cor, sag)

    entrees, masques = piles(a.racine, a.sujet)
    for methode, graines in (("s2s", valider_graines(a.graines)), ("s2v", [1])):
        for g in graines:
            volume, poses = chemins_complets(a.base, methode, a.sujet, g)
            if os.path.exists(volume) and os.path.exists(poses):
                continue
            os.makedirs(os.path.dirname(volume), exist_ok=True)
            r = subprocess.run([a.medx, "svr", "recon", "--motion", methode, "--device", "cpu", "--s2s-seed", str(g), "--input", *entrees, "--mask", *masques, "--output", volume, "--poses", poses],
                               capture_output=True, text=True)
            print(f"{a.sujet} {methode} graine {g} : {'ok' if r.returncode == 0 else 'ÉCHEC ' + r.stderr[-300:]}", flush=True)


def cmd_liens(a):
    graines = valider_graines(a.graines)
    dossier = f"{a.base}/s2s/{a.sujet}/bounti"
    os.makedirs(dossier, exist_ok=True)
    for lien, cible in liens_bounti(a.base, a.sujet, graines, replis_presents(a.base, a.sujet, graines)):
        chemin = f"{dossier}/{lien}"
        if not os.path.lexists(chemin):
            os.symlink(cible, chemin)
    print(f"{dossier} : {len(os.listdir(dossier))} liens")


def cmd_q2(a):
    graines = valider_graines(a.graines)
    # `analyse_concordance_sujets.py` lit `data/svr/results/<sujet>/<fichier>` : on lui donne un chemin relatif qui en sort vers la base
    vers_base = os.path.relpath(os.path.abspath(a.base), os.path.abspath("data/svr/results/x"))
    methodes = ["sans correction=-"] + [f"s2s_g{g}={vers_base}/s2s/{{sujet}}/poses_g{g}.tsv" for g in graines] + [f"s2v={vers_base}/s2v/{{sujet}}/poses_g1.tsv"]
    sorties = []
    for sujet, etiquette in zip(a.sujets, a.etiquettes):
        sortie = f"{a.base}/q2_{sujet}.tsv"
        commande = [sys.executable, "-I", f"{os.path.dirname(os.path.abspath(__file__))}/analyse_concordance_sujets.py", "--racine", a.racine, "--sujets", sujet, "--etiquettes", etiquette, "--sortie", sortie,
                    "--methodes", *[m.replace("{sujet}", sujet) for m in methodes]]
        r = subprocess.run(commande, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{sujet} : ÉCHEC de la concordance :\n{r.stderr[-400:]}")
            continue
        sorties.append((sujet, etiquette, sortie))
    toutes = collecter_q2(sorties, graines)
    for (sujet, mesure), ag in toutes.items():
        etiquette = next(e for s, e, _ in sorties if s == sujet)
        print(f"{sujet} {etiquette} {mesure} : moyenne {ag['moyenne']:.4f} min {ag['minimum']:.4f} max {ag['maximum']:.4f} étendue {ag['etendue']:.4f}")
    for mesure, (moyenne, maximum, n) in resumer_q2(toutes).items():
        print(f"Q2 {mesure} : étendue moyenne {moyenne:.4f} ; maximale {maximum:.4f} ({n} sujets)")


def collecter_q2(sorties, graines):
    """`{(sujet, mesure): agrégat}` pour tous les sujets de `sorties` (liste de `(sujet, étiquette, TSV)`) : la clé porte le sujet, pas l'étiquette, sinon les sujets de même étiquette s'écrasent."""
    toutes = {}
    for sujet, _, sortie in sorties:
        for (_, mesure), ag in etendues_q2(sortie, graines).items():
            toutes[(sujet, mesure)] = ag
    return toutes


def resumer_q2(toutes):
    """`{mesure: (étendue moyenne, étendue maximale, nombre de sujets)}` d'après `{(sujet, mesure): agrégat}` : un sujet compte une fois par mesure."""
    sortie = {}
    for mesure in ("toutes", "hors_stack_0"):
        e = [ag["etendue"] for (_, m), ag in toutes.items() if m == mesure]
        if e:
            sortie[mesure] = (sum(e) / len(e), max(e), len(e))
    return sortie


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    for nom, f in (("complets", cmd_complets), ("liens", cmd_liens), ("q2", cmd_q2)):
        q = s.add_parser(nom)
        q.add_argument("--base", required=True)
        q.add_argument("--graines", nargs="+", type=int, default=[1, 2, 3])
        q.set_defaults(f=f)
        if nom == "q2":
            q.add_argument("--racine", required=True)
            q.add_argument("--sujets", nargs="+", required=True)
            q.add_argument("--etiquettes", nargs="+", required=True)
        else:
            q.add_argument("--sujet", required=True)
        if nom == "complets":
            q.add_argument("--racine", required=True)
            q.add_argument("--medx", required=True)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
