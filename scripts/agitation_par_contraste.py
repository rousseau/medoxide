#!/usr/bin/env python3
"""Critère d'agitation par contraste (docs/ROADMAP.md, F4.0.4, point 0.4b) : concordance d'en-tête des stacks d'un sujet, sans reconstruction.

La concordance d'un sujet est la moyenne de `score_before` de `medx svr qc` sur les coupes qui ont un score et au moins 500 pixels de masque.
Calibrage d'un seuil sur des sujets aux étiquettes connues (règle écrite d'avance) : si les scores de TOUS les calmes sont strictement supérieurs
à ceux de TOUS les agités, le seuil est le milieu entre le plus faible des calmes et le plus fort des agités ; sinon aucun seuil n'est défini.

Aucun identifiant n'est écrit dans le code : racine, sujets et étiquettes sont des arguments. Les sorties restent sous `--travail` (ignoré par Git).
"""

import argparse
import csv
import os
import re
import subprocess
import sys

MIN_PIXELS = 500


def concordance_moyenne(chemin_tsv):
    """Moyenne de `score_before` sur les coupes avec score et `mask_pixels` ≥ MIN_PIXELS ; lève ValueError s'il n'y en a aucune."""
    valeurs = []
    with open(chemin_tsv, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["score_before"] != "n/a" and int(r["mask_pixels"]) >= MIN_PIXELS:
                valeurs.append(float(r["score_before"]))
    if not valeurs:
        raise ValueError(f"aucune coupe avec score dans {chemin_tsv}")
    return sum(valeurs) / len(valeurs)


def seuil_de_separation(calmes, agites):
    """Milieu entre le plus faible des calmes et le plus fort des agités si les deux groupes sont strictement séparés (calmes au-dessus), sinon `None`."""
    bas, haut = min(calmes), max(agites)
    return (bas + haut) / 2.0 if bas > haut else None


def piles(racine, sujet, contraste):
    """Stacks et masques du run 1 d'un sujet pour un contraste (`trufi` ou `haste`), dans l'ordre axial, coronal, sagittal.

    Le nom de la séquence varie d'un sujet à l'autre (`haste`, `hastepaceauto`, `hastete183`…) : pour chaque plan on cherche l'unique pile
    `acq-<contraste><séquence><plan>_run-1` ; zéro ou plusieurs candidates : erreur (on ne devine pas)."""
    session = sorted(d for d in os.listdir(f"{racine}/{sujet}") if d.startswith("ses-"))[0]
    dossier, masques = f"{racine}/{sujet}/{session}/anat", f"{racine}/derivatives/medx-fetalbet/{sujet}/{session}/anat"
    entrees, sorties = [], []
    for plan in ("ax", "cor", "sag"):
        motif = re.compile(rf"^{re.escape(sujet)}_{re.escape(session)}_acq-({contraste}[a-z0-9]*{plan})_run-1_T2w\.nii\.gz$")
        trouves = [m.group(1) for f in sorted(os.listdir(dossier)) if (m := motif.match(f))]
        if len(trouves) != 1:
            raise ValueError(f"{sujet} : {len(trouves)} pile(s) {contraste} pour le plan {plan} ({trouves})")
        nom = f"{sujet}_{session}_acq-{trouves[0]}_run-1"
        entrees.append(f"{dossier}/{nom}_T2w.nii.gz")
        sorties.append(f"{masques}/{nom}_desc-brain_mask.nii.gz")
    return entrees, sorties


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--racine", required=True)
    p.add_argument("--contraste", choices=["trufi", "haste"], required=True)
    p.add_argument("--calmes", nargs="+", required=True)
    p.add_argument("--agites", nargs="+", required=True)
    p.add_argument("--travail", required=True)
    p.add_argument("--medx", default="target/release/medx")
    a = p.parse_args()
    os.makedirs(a.travail, exist_ok=True)
    scores = {}
    for sujet in a.calmes + a.agites:
        entrees, masques = piles(a.racine, sujet, a.contraste)
        sortie = f"{a.travail}/{sujet}_{a.contraste}.tsv"
        r = subprocess.run([a.medx, "svr", "qc", "--input", *entrees, "--mask", *masques, "--output", sortie], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"échec de medx svr qc pour {sujet} :\n{r.stderr[-800:]}")
        scores[sujet] = concordance_moyenne(sortie)
        print(f"{sujet} ({'calme' if sujet in a.calmes else 'agité'}) : concordance {scores[sujet]:.3f}", flush=True)
    seuil = seuil_de_separation([scores[s] for s in a.calmes], [scores[s] for s in a.agites])
    print(f"SEUIL {a.contraste} :", "aucun (groupes non séparés)" if seuil is None else f"{seuil:.3f}")


if __name__ == "__main__":
    main()
