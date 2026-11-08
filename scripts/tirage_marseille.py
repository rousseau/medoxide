#!/usr/bin/env python3
"""Tirage séquentiel des jeux neufs (docs/ROADMAP.md, F4.0.4, point 0.4d).

1. `liste` : sujets éligibles d'un bucket BIDS à partir des seuls JSON (lecture seule) : une session avec, pour chacun des deux
   contrastes (trufi, haste), une pile par plan (ax, cor, sag) ; on retire tout sujet dont une pile a la même heure d'acquisition (`AcquisitionTime`)
   qu'une pile déjà utilisée. La session retenue est la première session éligible.
2. `tirer` : mélange de la liste par une graine écrite d'avance, puis, sujet par sujet dans cet ordre : téléchargement des trois piles TRUFI,
   masques `medx fetalbet`, concordance `medx svr qc`, étiquette calme/agité (seuil), affectation au test 2 puis au développement tant que
   les quotas ne sont pas pleins. S'arrête quand tous les quotas sont pleins. Reprenable : les sujets déjà dans le fichier de sortie ne sont pas refaits.

Aucun identifiant n'est écrit dans le code : sujets, bucket, racines sont des arguments ou des fichiers sous `--travail` (ignoré par Git).
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agitation_par_contraste as ac

PLANS = ("ax", "cor", "sag")
CONTRASTES = ("trufi", "haste")
SEUIL = 0.45
QUOTAS = {"test2": {"calme": 4, "agite": 4}, "dev": {"calme": 3, "agite": 3}}


def lire_jsons(racine):
    """Liste (nom de fichier, contenu) de tous les `*_T2w.json` sous `racine`."""
    sortie = []
    for f in sorted(glob.glob(f"{racine}/**/*_T2w.json", recursive=True)):
        with open(f) as fh:
            sortie.append((os.path.basename(f), json.load(fh)))
    return sortie


def piles_du_plan(noms, contraste, plan):
    """Fichiers (sans `_T2w.json`) du contraste et du plan parmi `noms` : `{(acq, run)}` trié ; run vaut 1 s'il n'y a pas d'entité `run`."""
    trouves = []
    for nom in noms:
        m = re.match(rf"^(?:sub-[a-z0-9]+)_(?:ses-[a-z0-9]+)_acq-({contraste}[a-z0-9]*{plan})(?:_run-([0-9]+))?_T2w\.json$", nom)
        if m:
            trouves.append((m.group(1), int(m.group(2) or 1)))
    return sorted(trouves)


def eligibles(racine_json, racine_utilisees):
    """`(sujet -> première session éligible, sujets retirés par recoupement)`.

    Éligible : une session où chaque (contraste, plan) a au moins une pile. Recoupement : une pile du sujet (toute session) a l'heure
    d'acquisition d'une pile de `racine_utilisees`."""
    utilisees = {j["AcquisitionTime"] for _, j in lire_jsons(racine_utilisees) if "AcquisitionTime" in j}
    sessions, retires = {}, set()
    for nom, j in lire_jsons(racine_json):
        m = re.match(r"^(sub-[a-z0-9]+)_(ses-[a-z0-9]+)_", nom)
        sessions.setdefault((m.group(1), m.group(2)), []).append((nom, j))
        if j.get("AcquisitionTime") in utilisees:
            retires.add(m.group(1))
    choisis = {}
    for (sujet, session), fichiers in sorted(sessions.items()):
        noms = [n for n, _ in fichiers]
        if sujet in retires or sujet in choisis:
            continue
        if all(piles_du_plan(noms, c, p) for c in CONTRASTES for p in PLANS):
            choisis[sujet] = session
    return choisis, sorted(retires)


def melanger(liste, graine):
    """Copie de `liste` mélangée par `random.Random(graine).shuffle` (la liste d'entrée n'est pas modifiée)."""
    copie = list(liste)
    random.Random(graine).shuffle(copie)
    return copie


def etiquette(concordance, seuil=SEUIL):
    """`calme` si la concordance est au moins le seuil, `agite` sinon."""
    return "calme" if concordance >= seuil else "agite"


def effectifs_vides():
    return {jeu: {e: 0 for e in q} for jeu, q in QUOTAS.items()}


def complet(effectifs):
    """Vrai quand tous les quotas sont atteints."""
    return all(effectifs[j][e] >= QUOTAS[j][e] for j in QUOTAS for e in QUOTAS[j])


def affecter(etiq, effectifs):
    """Destination (`test2`, `dev` ou `None`) d'un sujet d'étiquette `etiq` ; met `effectifs` à jour. Le test 2 se remplit avant le développement."""
    for jeu in ("test2", "dev"):
        if effectifs[jeu][etiq] < QUOTAS[jeu][etiq]:
            effectifs[jeu][etiq] += 1
            return jeu
    return None


def pile_a_telecharger(noms, contraste, plan):
    """Nom de base de la pile retenue : le run 1 (ou l'unique fichier sans entité `run`) ; ValueError s'il n'y a pas de run 1."""
    for acq, run in piles_du_plan(noms, contraste, plan):
        if run == 1:
            return acq
    raise ValueError(f"pas de run 1 pour {contraste} {plan}")


def lancer(commande):
    r = subprocess.run(commande, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"échec de {' '.join(commande[:4])} … :\n{r.stderr[-800:]}")
    return r.stdout


def preparer(a, sujet, session, contraste):
    """Télécharge les trois piles d'un contraste (images et JSON) sous `<racine>/<sujet>/<session>/anat/` avec l'entité `_run-1`."""
    noms = [n for n, _ in lire_jsons(f"{a.json}/{sujet}/{session}")]
    dossier = f"{a.racine}/{sujet}/{session}/anat"
    os.makedirs(dossier, exist_ok=True)
    for plan in PLANS:
        acq = pile_a_telecharger(noms, contraste, plan)
        sources = [n for n in noms if re.match(rf"^{sujet}_{session}_acq-{acq}(_run-1)?_T2w\.json$", n)]
        if len(sources) != 1:
            sys.exit(f"{sujet} : {len(sources)} fichier(s) pour {acq} (run 1)")
        base = sources[0][: -len(".json")]
        for ext in (".nii.gz", ".json"):
            cible = f"{dossier}/{sujet}_{session}_acq-{acq}_run-1_T2w{ext}"
            if not os.path.exists(cible):
                lancer(["rclone", "copyto", f"{a.remote}/sourcedata/{sujet}/{session}/anat/{base}{ext}", cible])


def classer(a, sujet, session):
    """Télécharge, masque, calcule la concordance TRUFI d'un sujet ; renvoie la concordance."""
    preparer(a, sujet, session, "trufi")
    lancer([a.medx, "fetalbet", "--bids", a.racine, "--participant", sujet[4:], "--session", session[4:], "--output-dir", a.racine])
    entrees, masques = ac.piles(a.racine, sujet, "trufi")
    sortie = f"{a.travail}/{sujet}_trufi.tsv"
    lancer([a.medx, "svr", "qc", "--input", *entrees, "--mask", *masques, "--output", sortie])
    return ac.concordance_moyenne(sortie)


def cmd_completer(a):
    """0.4e : télécharge les piles HASTE de chaque sujet affecté (test 2 ou développement) et les masque pile par pile
    (`fetalbet --input/--output`, pour ne pas réécrire les masques TRUFI) ; reprenable."""
    with open(f"{a.travail}/tirage.tsv", newline="") as f:
        sujets = [r for r in csv.DictReader(f, delimiter="\t") if r["jeu"] in ("test2", "dev")]
    for r in sujets:
        sujet, session = r["sujet"], r["session"]
        preparer(a, sujet, session, "haste")
        entrees, masques = ac.piles(a.racine, sujet, "haste")
        for entree, masque in zip(entrees, masques):
            if os.path.exists(masque):
                continue
            os.makedirs(os.path.dirname(masque), exist_ok=True)
            lancer([a.medx, "fetalbet", "--input", entree, "--output", masque])
        print(f"{sujet} : HASTE téléchargé et masqué", flush=True)
    print("COMPLETER FINI")


def cmd_liste(a):
    choisis, retires = eligibles(a.json, a.utilisees)
    os.makedirs(a.travail, exist_ok=True)
    with open(f"{a.travail}/eligibles.tsv", "w") as f:
        for s, ses in choisis.items():
            f.write(f"{s}\t{ses}\n")
    print(f"{len(choisis)} sujets éligibles ; retirés par recoupement : {retires}")


def cmd_tirer(a):
    os.makedirs(a.travail, exist_ok=True)
    with open(f"{a.travail}/eligibles.tsv") as f:
        elig = dict(l.rstrip("\n").split("\t") for l in f if l.strip())
    ordre = melanger(sorted(elig), a.graine)
    chemin = f"{a.travail}/tirage.tsv"
    faits = []
    if os.path.exists(chemin):
        with open(chemin, newline="") as f:
            faits = list(csv.DictReader(f, delimiter="\t"))
    ef = effectifs_vides()
    for r in faits:
        ef[r["jeu"]][r["etiquette"]] += 1 if r["jeu"] in ef else 0
    print("ordre du mélange :", " ".join(ordre), flush=True)
    for sujet in ordre:
        if complet(ef):
            break
        if any(r["sujet"] == sujet for r in faits):
            continue
        c = classer(a, sujet, elig[sujet])
        e = etiquette(c)
        jeu = affecter(e, ef)
        ligne = {"sujet": sujet, "session": elig[sujet], "concordance": f"{c:.4f}", "etiquette": e, "jeu": jeu or "ignore"}
        faits.append(ligne)
        with open(chemin, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ligne), delimiter="\t")
            w.writeheader()
            w.writerows(faits)
        print(f"{sujet} : concordance {c:.3f} → {e} → {ligne['jeu']} ; effectifs {ef}", flush=True)
    print("QUOTAS PLEINS" if complet(ef) else "LISTE ÉPUISÉE AVANT LES QUOTAS", ef)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    for nom, f in (("liste", cmd_liste), ("tirer", cmd_tirer), ("completer", cmd_completer)):
        q = s.add_parser(nom)
        q.add_argument("--travail", required=True)
        q.add_argument("--json", help="dossier des `*_T2w.json` du bucket (copie locale)")
        q.set_defaults(f=f)
        if nom == "liste":
            q.add_argument("--utilisees", required=True, help="dossier des JSON des piles déjà utilisées")
        else:
            if nom == "tirer":
                q.add_argument("--graine", type=int, required=True)
            q.add_argument("--racine", required=True, help="racine BIDS locale des piles téléchargées")
            q.add_argument("--remote", required=True, help="remote rclone du bucket, par exemple `garage:<bucket>`")
            q.add_argument("--medx", required=True)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
