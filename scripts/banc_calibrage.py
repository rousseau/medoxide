#!/usr/bin/env python3
"""Calibrage et vérification du simulateur sur des piles réelles (docs/ROADMAP.md, F4.0.1, critère (v), amendements 1 et 2).

Protocole figé dans le ROADMAP avant toute mesure. Pour chaque sujet et chaque pile K (0 axial, 1 coronal, 2 sagittal, run 1), la source est le volume
SVRTK reconstruit SANS la pile K (`<résultats>/sub-X/svrtk_sansK.nii.gz`) et la pile K est simulée à la géométrie réelle (`--no-recenter`, calme, une graine).

Sous-commandes :
  calibrer  : sujets de calibrage, au bruit et au biais DÉJÀ figés (`--bruit`, `--biais`) -> flou dans le plan (grille), puis facteur de profil (grille) ;
              écrit les valeurs choisies (amendement 2, C).
  verifier  : sujets de vérification, aux valeurs données -> quatre rapports simulé/réel par sujet et verdict contre [0,8 ; 1,25] (amendement 2, B et D).

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
GRILLE_FLOU_PLAN = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
GRILLE_PROFIL = [0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
PROFIL_NON_PHYSIQUE = 2.0  # au-delà : publié, marqué non physique (amendement 2, C)


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


def simuler_pile(medx, source, gabarits, dossier, k, bruit, biais, profil, flou):
    """`medx svr simulate --no-recenter` (repris si la pile existe) ; renvoie le chemin de la pile K simulée."""
    pile = f"{dossier}/sim_acq-{k + 1}_T2w.nii.gz"
    if not os.path.exists(pile):
        r = subprocess.run(
            [medx, "svr", "simulate", "--source", source, "--template", *gabarits, "--output-dir", dossier, "--name", "sim", "--no-recenter", "--motion", "none",
             "--noise-pct", str(bruit), "--bias-pct", str(biais), "--slice-profile-factor", str(profil),
             "--inplane-blur-mm", str(flou), "--seed", "1"],
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
        zooms = [float(z) for z in nib.load(piles[k]).header.get_zooms()[:2]]
        emp_plan = [bp.emplacements_de_bord_axe(s, m, mu, axe) for axe in (0, 1)]
        sortie.append({"carres": carres, "emplacements": emp, "espacement": esp,
                       "largeur": float(np.median(bp.largeur_de_bord(s, emp, esp))), "mu": mu,
                       "emp_plan": emp_plan, "pas_plan": zooms,
                       "largeur_plan": [float(np.median(bp.largeur_de_bord_axe(s, emp_plan[axe], zooms[axe], axe))) for axe in (0, 1)]})
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
             "largeur": float(np.median(bp.largeur_de_bord(sim[k], reel[k]["emplacements"], reel[k]["espacement"]))),
             "largeur_plan": [float(np.median(bp.largeur_de_bord_axe(sim[k], reel[k]["emp_plan"][axe], reel[k]["pas_plan"][axe], axe))) for axe in (0, 1)]}
            for k in range(3)]


def simuler_sujet(medx, resultats, racine, travail, sujet, bruit, biais, profil, flou, etiquette):
    """Simule les trois piles du sujet (pile K depuis `svrtk_sansK`) aux réglages donnés ; renvoie les chemins des trois piles simulées."""
    piles, _ = piles_du_sujet(racine, sujet)
    sim = []
    for k in range(3):
        source = f"{resultats}/{sujet}/svrtk_sans{k}.nii.gz"
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        sim.append(simuler_pile(medx, source, piles, f"{travail}/{sujet}/{etiquette}/K{k}", k, bruit, biais, profil, flou))
    return sim


def rapports(reel, mes, plages_reelles, plages_sim):
    """Quatre rapports simulé/réel (médiane sur les trois piles) : bruit, largeur entre coupes, plage du biais, largeur dans le plan.

    Largeur dans le plan (amendement 2, B) : un rapport par direction (lignes, colonnes), médiane sur les trois piles ; le rapport retenu `plan` est le
    PIRE des deux (le plus éloigné de 1 en échelle logarithmique) ; `plan_x` et `plan_y` sont aussi rendus."""
    r_bruit = float(np.median([m["bruit_sujet"] / r["bruit_sujet"] for m, r in zip(mes, reel)]))
    r_larg = float(np.median([m["largeur"] / r["largeur"] for m, r in zip(mes, reel)]))
    r_biais = float(np.median(plages_sim) / np.median(plages_reelles))
    plan = [float(np.median([m["largeur_plan"][d] / r["largeur_plan"][d] for m, r in zip(mes, reel)])) for d in (0, 1)]
    pire = max(plan, key=lambda v: abs(np.log(v)))
    return {"bruit": r_bruit, "largeur": r_larg, "biais": r_biais, "plan": pire, "plan_x": plan[0], "plan_y": plan[1]}


def rapport_plan_pour_calibrage(reels, mesures):
    """Rapport simulé/réel de largeur de bord DANS le plan pour le calibrage : médiane groupée sur sujets × piles × les deux directions (amendement 2, C)."""
    return float(np.median([m["largeur_plan"][d] / r["largeur_plan"][d] for reel, mes in zip(reels, mesures) for r, m in zip(reel, mes) for d in (0, 1)]))


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
    p.add_argument("--bruit", type=float, help="bruit_pct figé (amendement 1)")
    p.add_argument("--biais", type=float, help="bias_pct figé (amendement 1)")
    p.add_argument("--profil", type=float, help="verifier : facteur de profil calibré")
    p.add_argument("--flou", type=float, help="verifier : flou dans le plan calibré, en mm")
    a = p.parse_args()
    os.makedirs(a.travail, exist_ok=True)
    reel = {s: mesures_reelles(a.racine, s) for s in a.sujets}
    masques = {s: piles_du_sujet(a.racine, s)[1] for s in a.sujets}
    plages_reelles = {s: plage_de_biais(a.bin_test, *piles_du_sujet(a.racine, s)) for s in a.sujets}
    for s in a.sujets:
        print(f"{s} réel : bruit (sujet) {reel[s][0]['bruit_sujet']:.5f} ; largeur entre coupes {[round(r['largeur'], 2) for r in reel[s]]} mm ; dans le plan (x, y) {[[round(v, 2) for v in r['largeur_plan']] for r in reel[s]]} mm ; plage du biais {[round(x, 3) for x in plages_reelles[s]]}", flush=True)

    def evaluer(s, bruit, biais, profil, flou, etiquette, avec_plage=False):
        sim = simuler_sujet(a.medx, a.resultats, a.racine, a.travail, s, bruit, biais, profil, flou, etiquette)
        mes = mesures_simulees(reel[s], masques[s], sim)
        pb = plage_de_biais(a.bin_test, sim, masques[s]) if avec_plage else None
        return mes, pb

    if None in (a.bruit, a.biais):
        p.error("--bruit et --biais (valeurs figées) sont requis")
    if a.action == "calibrer":
        # (1) flou dans le plan : grille, profil 1, au bruit et au biais figés
        plan = {}
        for f in GRILLE_FLOU_PLAN:
            mes = [evaluer(s, a.bruit, a.biais, 1.0, f, f"flou{f}_profil1.0")[0] for s in a.sujets]
            plan[f] = rapport_plan_pour_calibrage([reel[s] for s in a.sujets], mes)
            print(f"  flou {f} mm: largeur dans le plan simulé/réel = {plan[f]:.3f}", flush=True)
        flou = bp.meilleur_parametre(GRILLE_FLOU_PLAN, [plan[f] for f in GRILLE_FLOU_PLAN], 1.0)
        # (2) facteur de profil : grille étendue, au flou choisi
        larg = {}
        for f in GRILLE_PROFIL:
            larg[f] = float(np.median([evaluer(s, a.bruit, a.biais, f, flou, f"flou{flou}_profil{f}")[0][k]["largeur"] / reel[s][k]["largeur"]
                                       for s in a.sujets for k in range(3)]))
            print(f"  profil {f}: largeur entre coupes simulé/réel = {larg[f]:.3f}", flush=True)
        profil = bp.meilleur_parametre(GRILLE_PROFIL, [larg[f] for f in GRILLE_PROFIL], 1.0)
        res = {"bruit_pct": a.bruit, "bias_pct": a.biais, "flou_plan_mm": flou, "facteur_profil": profil,
               "profil_non_physique": profil > PROFIL_NON_PHYSIQUE, "sujets": a.sujets}
        print("CALIBRAGE", json.dumps(res))
        with open(f"{a.travail}/calibrage.json", "w") as f:
            json.dump(res, f, indent=1)
    else:
        if None in (a.profil, a.flou):
            p.error("verifier demande --profil et --flou")
        tout = True
        for s in a.sujets:
            mes, pb = evaluer(s, a.bruit, a.biais, a.profil, a.flou, "verif", avec_plage=True)
            r = rapports(reel[s], mes, plages_reelles[s], pb)
            ok = all(dans_les_bornes(r[c]) for c in ("bruit", "largeur", "biais", "plan"))
            tout &= ok
            print(f"VERIF {s} : bruit {r['bruit']:.3f} ; largeur entre coupes {r['largeur']:.3f} ; biais {r['biais']:.3f} ; "
                  f"largeur dans le plan {r['plan']:.3f} (x {r['plan_x']:.3f}, y {r['plan_y']:.3f}) -> {'TENU' if ok else 'NON TENU'}")
        print("VERDICT (v) :", "TENU" if tout else "NON TENU")


if __name__ == "__main__":
    main()
