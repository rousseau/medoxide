#!/usr/bin/env python3
"""Calibrage et vérification du simulateur sur des piles réelles (docs/ROADMAP.md, F4.0.1, critère (v) et amendement 1).

Protocole figé dans le ROADMAP avant toute mesure. Pour chaque sujet et chaque pile K (0 axial, 1 coronal, 2 sagittal, run 1), la source est le volume
SVRTK reconstruit SANS la pile K (`<résultats>/sub-X/svrtk_sansK.nii.gz`) et la pile K est simulée à la géométrie réelle (`--no-recenter`, calme, une graine).

Sous-commandes :
  calibrer  : sujets de calibrage -> bruit (proportionnalité), facteur de profil (grille), biais (grille) ; écrit les valeurs choisies.
  verifier  : sujets de vérification, aux valeurs données -> trois rapports simulé/réel par sujet et verdict contre [0,8 ; 1,25].

Aucun identifiant n'est écrit dans le code : sujets et chemins sont des arguments. Tout reste sous `--travail` (ignoré par Git).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_plausibilite as bp

PLANS = ["trufiax", "truficor", "trufisag"]
BORNES = (0.8, 1.25)
GRILLE_PROFIL = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2]
GRILLE_BIAIS = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
BRUIT_ESSAI = 1.0


def piles_du_sujet(racine, sujet):
    """(piles, masques) du run 1 axial, coronal, sagittal du sujet, dans cet ordre (indices 0, 1, 2)."""
    session = sorted(d for d in os.listdir(f"{racine}/{sujet}") if d.startswith("ses-"))[0]
    dossier = f"{racine}/{sujet}/{session}/anat"
    masques = f"{racine}/derivatives/medx-fetalbet/{sujet}/{session}/anat"
    piles = [f"{dossier}/{sujet}_{session}_acq-{p}_run-1_T2w.nii.gz" for p in PLANS]
    ms = [f"{masques}/{sujet}_{session}_acq-{p}_run-1_desc-brain_mask.nii.gz" for p in PLANS]
    for f in piles + ms:
        if not os.path.exists(f):
            raise FileNotFoundError(f)
    return piles, ms


def lire(chemin):
    return np.asarray(nib.load(chemin).dataobj, dtype=np.float64)


def simuler_pile(medx, source, gabarits, dossier, k, bruit, biais, profil):
    """`medx svr simulate --no-recenter` (repris si la pile existe) ; renvoie le chemin de la pile K simulée."""
    pile = f"{dossier}/sim_acq-{k + 1}_T2w.nii.gz"
    if not os.path.exists(pile):
        r = subprocess.run(
            [medx, "svr", "simulate", "--source", source, "--template", *gabarits, "--output-dir", dossier, "--name", "sim", "--no-recenter", "--motion", "none",
             "--noise-pct", str(bruit), "--bias-pct", str(biais), "--slice-profile-factor", str(profil), "--seed", "1"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"échec de simulate :\n{r.stderr[-1500:]}")
    return pile


def plage_de_biais(binaire_test, piles, masques):
    """Plages (max − min du log du champ estimé) de chaque pile, par le test de diagnostic `diagnose_the_estimated_bias_range`."""
    env = {**os.environ, "MEDOXIDE_PILES": ",".join(piles), "MEDOXIDE_MASQUES": ",".join(masques)}
    r = subprocess.run([binaire_test, "diagnose_the_estimated_bias_range", "--ignored", "--nocapture"], capture_output=True, text=True, env=env)
    plages = bp.lire_plage_biais(r.stdout)
    if len(plages) != len(piles):
        raise RuntimeError(f"diagnostic du biais : {len(plages)} plages pour {len(piles)} piles\n{r.stdout[-800:]}{r.stderr[-800:]}")
    return plages


def mesures_reelles(racine, sujet):
    """Quantités de référence sur les piles réelles : carrés d'air, emplacements de bord, rapport de bruit, largeurs et plage médiane du biais, par pile."""
    piles, masques = piles_du_sujet(racine, sujet)
    sortie = []
    for k in range(3):
        s, m = lire(piles[k]), lire(masques[k]) > 0
        mu = bp.moyenne_tissu(s, m)
        carres = bp.choisir_carres(s, mu)
        emp = bp.emplacements_de_bord(s, m, mu)
        esp = float(nib.load(piles[k]).header.get_zooms()[2])
        sortie.append({"carres": carres, "emplacements": emp, "espacement": esp,
                       "largeur": float(np.median(bp.largeur_de_bord(s, emp, esp))), "mu": mu})
    donnees = [lire(f) for f in piles]
    ms = [lire(f) > 0 for f in masques]
    bruit = bp.rapport_bruit_sujet(donnees, ms, [r["carres"] for r in sortie])  # lève ValueError sous 20 carrés pour le sujet
    for r in sortie:
        r["bruit_sujet"] = bruit
    return sortie


def mesures_simulees(reel, masques, piles_sim):
    """Mêmes quantités sur les piles simulées (une par K), aux mêmes carrés et emplacements que le réel."""
    sim = [lire(f) for f in piles_sim]
    ms = [lire(f) > 0 for f in masques]
    bruit = bp.rapport_bruit_sujet(sim, ms, [r["carres"] for r in reel])
    return [{"bruit_sujet": bruit,
             "largeur": float(np.median(bp.largeur_de_bord(sim[k], reel[k]["emplacements"], reel[k]["espacement"])))} for k in range(3)]


def simuler_sujet(medx, resultats, racine, travail, sujet, bruit, biais, profil, etiquette):
    """Simule les trois piles du sujet (pile K depuis `svrtk_sansK`) aux réglages donnés ; renvoie les chemins des trois piles simulées."""
    piles, _ = piles_du_sujet(racine, sujet)
    sim = []
    for k in range(3):
        source = f"{resultats}/{sujet}/svrtk_sans{k}.nii.gz"
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        sim.append(simuler_pile(medx, source, piles, f"{travail}/{sujet}/{etiquette}/K{k}", k, bruit, biais, profil))
    return sim


def rapports(reel, mes, plages_reelles, plages_sim):
    """Trois rapports simulé/réel (médiane sur les trois piles) : bruit, largeur de bord, plage du biais."""
    r_bruit = float(np.median([m["bruit_sujet"] / r["bruit_sujet"] for m, r in zip(mes, reel)]))
    r_larg = float(np.median([m["largeur"] / r["largeur"] for m, r in zip(mes, reel)]))
    r_biais = float(np.median(plages_sim) / np.median(plages_reelles))
    return {"bruit": r_bruit, "largeur": r_larg, "biais": r_biais}


def dans_les_bornes(rapport):
    return BORNES[0] <= rapport <= BORNES[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["calibrer", "verifier"])
    p.add_argument("--racine", required=True, help="jeu réel (piles et masques de fetal-bet)")
    p.add_argument("--resultats", required=True, help="dossier des volumes `sub-X/svrtk_sansK.nii.gz`")
    p.add_argument("--sujets", nargs="+", required=True)
    p.add_argument("--travail", required=True)
    p.add_argument("--medx", default="target/release/medx")
    p.add_argument("--bin-test", default=os.environ.get("MEDOXIDE_TEST_BIN"), help="binaire de test de medoxide-svr (diagnostic du biais)")
    p.add_argument("--bruit", type=float, help="verifier : bruit_pct calibré")
    p.add_argument("--profil", type=float, help="verifier : facteur de profil calibré")
    p.add_argument("--biais", type=float, help="verifier : bias_pct calibré")
    a = p.parse_args()
    os.makedirs(a.travail, exist_ok=True)
    reel = {s: mesures_reelles(a.racine, s) for s in a.sujets}
    masques = {s: piles_du_sujet(a.racine, s)[1] for s in a.sujets}
    plages_reelles = {s: plage_de_biais(a.bin_test, *piles_du_sujet(a.racine, s)) for s in a.sujets}
    for s in a.sujets:
        print(f"{s} réel : bruit (sujet) {reel[s][0]['bruit_sujet']:.5f} ; largeur {[round(r['largeur'], 2) for r in reel[s]]} mm ; plage du biais {[round(x, 3) for x in plages_reelles[s]]}", flush=True)

    def evaluer(s, bruit, biais, profil, etiquette):
        sim = simuler_sujet(a.medx, a.resultats, a.racine, a.travail, s, bruit, biais, profil, etiquette)
        mes = mesures_simulees(reel[s], masques[s], sim)
        pb = plage_de_biais(a.bin_test, sim, masques[s]) if biais is not None else None
        return mes, pb

    if a.action == "calibrer":
        # bruit : proportionnalité à partir d'un essai à BRUIT_ESSAI %
        essais = {s: evaluer(s, BRUIT_ESSAI, 0.0, 1.0, "bruit") for s in a.sujets}
        facteur = float(np.median([essais[s][0][0]["bruit_sujet"] / reel[s][0]["bruit_sujet"] for s in a.sujets]))
        bruit = BRUIT_ESSAI / facteur
        print(f"bruit_pct calibré = {bruit:.4f} (rapport simulé/réel à {BRUIT_ESSAI} % : {facteur:.3f})", flush=True)
        # profil : grille, au bruit calibré, sans biais
        larg = {f: np.median([evaluer(s, bruit, 0.0, f, f"profil{f}")[0][k]["largeur"] / reel[s][k]["largeur"] for s in a.sujets for k in range(3)]) for f in GRILLE_PROFIL}
        for f, v in larg.items():
            print(f"  profil {f}: largeur simulé/réel = {v:.3f}", flush=True)
        profil = bp.meilleur_parametre(GRILLE_PROFIL, [larg[f] for f in GRILLE_PROFIL], 1.0)
        # biais : grille, au bruit et au profil calibrés
        plages = {}
        for b in GRILLE_BIAIS:
            plages[b] = np.median([np.median(evaluer(s, bruit, b, profil, f"biais{b}")[1]) / np.median(plages_reelles[s]) for s in a.sujets])
            print(f"  biais {b} %: plage simulé/réel = {plages[b]:.3f}", flush=True)
        biais = bp.meilleur_parametre(GRILLE_BIAIS, [max(plages[b], 1e-9) for b in GRILLE_BIAIS], 1.0)
        res = {"bruit_pct": bruit, "facteur_profil": profil, "bias_pct": biais, "sujets": a.sujets}
        print("CALIBRAGE", json.dumps(res))
        with open(f"{a.travail}/calibrage.json", "w") as f:
            json.dump(res, f, indent=1)
    else:
        if None in (a.bruit, a.profil, a.biais):
            p.error("verifier demande --bruit, --profil et --biais")
        tout = True
        for s in a.sujets:
            mes, pb = evaluer(s, a.bruit, a.biais, a.profil, "verif")
            r = rapports(reel[s], mes, plages_reelles[s], pb)
            ok = all(dans_les_bornes(v) for v in r.values())
            tout &= ok
            print(f"VERIF {s} : bruit {r['bruit']:.3f} ; largeur {r['largeur']:.3f} ; biais {r['biais']:.3f} -> {'TENU' if ok else 'NON TENU'}")
        print("VERDICT (v) :", "TENU" if tout else "NON TENU")


if __name__ == "__main__":
    main()
