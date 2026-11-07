#!/usr/bin/env python3
"""F4.0.2 (docs/ROADMAP.md) : le juge hors échantillon contre la vérité du banc.

Sous-commandes :
  preparer : pour les sujets « dev » du tirage, récupère la source CRL (Garage), simule 3 piles (condition `lisse`, réglages figés des amendements 1 à 3)
             et construit une racine à la forme d'un jeu réel (`sub-bNN/ses-1/anat/…_acq-trufi{ax,cor,sag}_run-1_T2w.nii.gz`, masques sous
             `derivatives/medx-fetalbet/`) que `benchmark_variantes_s2s.py` lit sans modification ; écrit `correspondances.tsv` (code, sujet, session).
  lancer   : `benchmark_variantes_s2s.py` pour un code, avec les 5 variantes (en-tête, α = 0,03 ; 0,05 ; 0,1 ; 0,3 en s2s), sur les 3 replis (ou ceux demandés).
  analyser : pour chaque code, recale chaque volume de repli sur la source, mesure PSNR/SSIM/NCC dans le masque de la source restreint aux voxels couverts
             par TOUTES les variantes du repli, puis calcule ρ̄ et ρ̄_α (`banc_juge_analyse.py`).

Aucun identifiant n'est écrit dans le code : le tirage et les chemins sont des arguments ; les codes `sub-bNN` sont anonymes, la correspondance reste sous `--racine`
(ignoré par Git). Les fichiers déjà présents sont réutilisés (reprise).
"""

import argparse
import csv
import os
import subprocess
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_juge_analyse as bj
import banc_mesures as bm
import banc_simule as bs

#: Réglages de simulation figés (ROADMAP F4.0.1, amendements 1 à 3). L'échelle de coupe vient de l'amendement 1 (A.1) ; le texte de F4.0.2 ne la liste
#: pas : précision écrite au ROADMAP avant toute mesure.
REGLAGES = {"bruit_pct": 1.0093461841900417, "biais_pct": 0.0, "echelle_pct": 5.0, "flou_mm": 2.0, "profil": 4.0, "graine": 1, "condition": "lisse"}
PLANS = ["trufiax", "truficor", "trufisag"]
SESSION = "ses-1"
#: Variantes : nom -> options de `medx svr recon` (s2s par défaut ; `--motion` n'est donc pas donné).
OPTIONS = {
    "entete": "--s2s-sweeps 0 --s2s-coarse-sweeps 0 --s2s-bias-rounds 0 --s2s-block-levels 0",
    "a003": "--alpha 0.03",
    "a005": "",
    "a010": "--alpha 0.1",
    "a030": "--alpha 0.3",
}
assert list(OPTIONS) == bj.VARIANTES


def code_du_sujet(numero):
    """Code anonyme du sujet n° `numero` (à partir de 1) : `sub-b01`."""
    return f"sub-b{numero:02d}"


def commande_de_simulation(medx, source, masque, gabarits, dossier, r=REGLAGES):
    """Ligne de commande de `medx svr simulate` aux réglages figés `r`."""
    motion, amplitude = bs.CONDITIONS[r["condition"]]
    return [
        medx, "svr", "simulate", "--source", source, "--source-mask", masque, "--template", *gabarits, "--output-dir", dossier, "--name", "sim",
        "--motion", motion, "--amplitude", str(amplitude), "--interleaved", "--noise-pct", str(r["bruit_pct"]), "--bias-pct", str(r["biais_pct"]),
        "--slice-scale-pct", str(r["echelle_pct"]), "--inplane-blur-mm", str(r["flou_mm"]), "--slice-profile-factor", str(r["profil"]),
        "--seed", str(r["graine"]), "--forward-model", "independent",
    ]


def construire_racine(simule, racine, code):
    """Racine de forme « jeu réel » pour un code : liens vers les piles simulées (`sim_acq-<n>`) renommées en `acq-trufi…_run-1`, leurs JSON et leurs masques."""
    dossier = f"{racine}/jeu/{code}/{SESSION}/anat"
    masques = f"{racine}/jeu/derivatives/medx-fetalbet/{code}/{SESSION}/anat"
    os.makedirs(dossier, exist_ok=True)
    os.makedirs(masques, exist_ok=True)
    for n, plan in enumerate(PLANS, start=1):
        nom = f"{code}_{SESSION}_acq-{plan}_run-1"
        for source, cible in (
            (f"{simule}/sim_acq-{n}_T2w.nii.gz", f"{dossier}/{nom}_T2w.nii.gz"),
            (f"{simule}/sim_acq-{n}_T2w.json", f"{dossier}/{nom}_T2w.json"),
            (f"{simule}/sim_acq-{n}_mask.nii.gz", f"{masques}/{nom}_desc-brain_mask.nii.gz"),
        ):
            if os.path.lexists(cible):
                os.remove(cible)
            os.symlink(os.path.abspath(source), cible)
    return f"{racine}/jeu"


def preparer(a):
    if not a.remote:
        sys.exit("--remote (ou la variable d'environnement exportée OXIDE_REMOTE) est requis")
    sujets = bs.lire_tirage(a.tirage, "dev")
    lignes = []
    for numero, r in enumerate(sujets, start=1):
        code = code_du_sujet(numero)
        source, masque, _ = bs.recuperer_source(a.remote, r["sujet"], r["session"], f"{a.racine}/sources/{r['sujet']}")
        simule = f"{a.racine}/simule/{code}"
        if not os.path.exists(f"{simule}/sim_poses_vraies.tsv"):
            bs.executer(commande_de_simulation(a.medx, source, masque, a.gabarits, simule))
        construire_racine(simule, a.racine, code)
        lignes.append({"code": code, "sujet": r["sujet"], "session": r["session"], "source": source, "masque": masque})
        print(f"{code} prêt", flush=True)
    with open(f"{a.racine}/correspondances.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "sujet", "session", "source", "masque"], delimiter="\t")
        w.writeheader()
        w.writerows(lignes)


def commande_de_juge(a, code):
    """Ligne de commande de `benchmark_variantes_s2s.py` pour `code` (variantes `nom=options`, replis demandés)."""
    cmd = [sys.executable, "-B", os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_variantes_s2s.py"), "--racine", f"{a.racine}/jeu", "--sujet", code,
           "--sortie", f"{a.racine}/juge/{code}", "--medx", a.medx, "--variantes", *[f"{n}={o}" for n, o in OPTIONS.items()]]
    if a.replis:
        cmd += ["--replis", *map(str, a.replis)]
    return cmd


def lancer(a):
    for code in a.codes:
        env = {**os.environ, "MEDOXIDE_TEST_BIN": a.bin_test}
        subprocess.run(commande_de_juge(a, code), check=True, env=env)


def lire_correspondances(racine):
    with open(f"{racine}/correspondances.tsv", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def analyser(a):
    lignes_sortie, tableau = [], {}
    for c in lire_correspondances(a.racine):
        code = c["code"]
        juge_dir, analyse = f"{a.racine}/juge/{code}", f"{a.racine}/analyse/{code}"
        os.makedirs(analyse, exist_ok=True)
        with open(f"{juge_dir}/resultats.tsv", newline="") as f:
            juges = {(r["variante"], int(r["stack_teste"])): r for r in csv.DictReader(f, delimiter="\t")}
        replis = sorted({h for _, h in juges})
        ref, masque_source = bs.lire(c["source"]), bs.lire(c["masque"]) > 0
        for h in replis:
            if not all((v, h) in juges for v in bj.VARIANTES):
                continue  # repli incomplet : ignoré (la reprise le complétera)
            alignes = {}
            for v in bj.VARIANTES:
                sortie = f"{analyse}/{v}_h{h}.aligne.nii.gz"
                if not os.path.exists(sortie):
                    bs.recaler(c["source"], f"{juge_dir}/{v}_h{h}.nii.gz", c["masque"], sortie)
                alignes[v] = bs.lire(sortie)
            couvert = masque_source.copy()
            for x in alignes.values():
                couvert &= x > 0
            mesures = {}
            for v in bj.VARIANTES:
                m = bm.mesures_volume(alignes[v], ref, couvert)
                j = float(juges[(v, h)]["ncc_moyenne"])
                mesures[v] = (j, m["psnr"])
                lignes_sortie.append({"code": code, "repli": h, "variante": v, "juge": j, "nettete": juges[(v, h)]["nettete"], "psnr": m["psnr"],
                                      "ssim": m["ssim"], "ncc": m["ncc"], "voxels": int(couvert.sum())})
            tableau[(code, h)] = mesures
    os.makedirs(f"{a.racine}/analyse", exist_ok=True)
    with open(f"{a.racine}/analyse/resultats.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes_sortie[0]), delimiter="\t")
        w.writeheader()
        w.writerows(lignes_sortie)
    rapport(tableau)


def rapport(tableau):
    """Imprime ρ par sujet, ρ̄ (5 variantes), ρ̄_α (4 variantes α), l'α qui maximise le juge et celui qui maximise le PSNR, et la lecture pré-déclarée."""
    par5, parA = bj.rho_par_sujet(tableau, bj.VARIANTES), bj.rho_par_sujet(tableau, bj.ALPHAS)
    for s in par5:
        print(f"{s} : rho (5 variantes) {par5[s]:+.3f} ; rho_alpha (4) {parA[s]:+.3f} ; meilleur alpha : juge {bj.meilleur_alpha(tableau, s, 0)}, PSNR {bj.meilleur_alpha(tableau, s, 1)}")
    r5, rA = bj.rho_moyen(tableau, bj.VARIANTES), bj.rho_moyen(tableau, bj.ALPHAS)
    print(f"RHO MOYEN (5 variantes) {r5:+.3f} ; RHO MOYEN ALPHA (4 variantes) {rA:+.3f} ({len(par5)} sujets, {len(tableau)} replis)")
    print("LECTURE :", bj.lecture(r5, rA))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="action", required=True)
    for nom in ("preparer", "lancer", "analyser"):
        s = sp.add_parser(nom)
        s.add_argument("--racine", required=True, help="dossier de travail (sous data/, ignoré par Git)")
        s.add_argument("--medx", default="target/release/medx")
        if nom == "preparer":
            s.add_argument("--tirage", required=True)
            s.add_argument("--gabarits", nargs=3, required=True)
            s.add_argument("--remote", default=os.environ.get("OXIDE_REMOTE"))
        if nom == "lancer":
            s.add_argument("--codes", nargs="+", required=True)
            s.add_argument("--replis", nargs="+", type=int)
            s.add_argument("--bin-test", default=os.environ.get("MEDOXIDE_TEST_BIN"))
    a = p.parse_args()
    {"preparer": preparer, "lancer": lancer, "analyser": analyser}[a.action](a)


if __name__ == "__main__":
    main()
