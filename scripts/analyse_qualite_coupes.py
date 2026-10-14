"""Peut-on estimer, SANS vérité terrain, combien de coupes sont mal recalées ?

Entrées : les résultats de `evaluate_simulated_motion.py --methodes s2s` : pour chaque jeu simulé, `<jeu>_s2s.qualite.tsv` (indicateurs par coupe écrits par `medx svr recon --quality`) et `<jeu>_s2s.erreurs.tsv` (erreur de pose vraie, en mm, après retrait du mouvement commun). Une coupe est « mal recalée » si son erreur dépasse 1,5 mm (seuil des évaluations précédentes).

Indicateurs candidats, par coupe recalée : `score_apres` (concordance des profils d'intersection après recalage, plus grand = mieux), `score_avant`, `gain = score_apres - score_avant`, `correction_rms_mm` (déplacement appliqué), `nb_recalages`.

CRITÈRES, fixés AVANT de regarder les résultats :
  1. Détection : l'AUC (probabilité qu'une coupe mal recalée ait un moins bon indicateur qu'une coupe bien recalée), regroupée sur tous les jeux, doit être >= 0,8 pour qu'un indicateur soit jugé utile.
  2. Comptage : avec le seuil de l'indicateur choisi sur les AUTRES jeux (validation « un jeu laissé de côté »), le nombre de coupes mal recalées prédit sur le jeu laissé de côté doit être à moins de 25 % (ou 3 coupes) du nombre vrai, SUR CHAQUE JEU. Sinon : le nombre n'est pas estimable de façon fiable par cet indicateur.
Les seuils sont cherchés sur une grille de quantiles de l'indicateur, en minimisant l'écart absolu entre nombre prédit et nombre vrai sommé sur les jeux d'apprentissage.

Usage : python scripts/analyse_qualite_coupes.py [--dossier data/atlas/results/qualite] [--seuil-mm 1.5]
"""

import argparse
import glob
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dossier", default="data/atlas/results/qualite")
parser.add_argument("--seuil-mm", type=float, default=1.5)
parser.add_argument("--methode", default="s2s")
args = parser.parse_args()


def lire(chemin):
    lignes = [l.rstrip("\n").split("\t") for l in open(chemin)]
    return [dict(zip(lignes[0], l)) for l in lignes[1:]]


jeux = {}
for f in sorted(glob.glob(os.path.join(args.dossier, f"*_{args.methode}.erreurs.tsv"))):
    nom = os.path.basename(f).removesuffix(f"_{args.methode}.erreurs.tsv")
    q = lire(f.replace(".erreurs.tsv", ".qualite.tsv"))
    erreurs = {(int(r["stack"]), int(r["coupe"])): float(r["erreur_mm"]) for r in lire(f)}
    lignes = []
    for r in q:
        cle = (int(r["stack"]), int(r["coupe"]))
        if r["recalee"] == "1" and cle in erreurs:
            avant, apres = float(r["score_avant"]), float(r["score_apres"])
            lignes.append((erreurs[cle], apres, avant, apres - avant, float(r["correction_rms_mm"]), int(r["nb_recalages"])))
    jeux[nom] = np.array(lignes)
INDICATEURS = {"score_apres": (1, +1), "score_avant": (2, +1), "gain": (3, +1), "correction_rms_mm": (4, -1), "nb_recalages": (5, -1)}
# signe : +1 = plus grand veut dire mieux recalé ; -1 = plus grand veut dire plus suspect


def auc(mal, valeur):
    """P(valeur d'une coupe mal recalée < valeur d'une bonne), en comptant les ex aequo pour 1/2 ; `valeur` orientée pour que petit = suspect."""
    a, b = valeur[mal], valeur[~mal]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    cmp = a[:, None] - b[None, :]
    return float(((cmp < 0).sum() + 0.5 * (cmp == 0).sum()) / cmp.size)


print(f"{len(jeux)} jeux ; seuil d'erreur {args.seuil_mm} mm\n")
print(f"{'jeu':32} {'coupes':>6} {'mal recalées':>12} {'part':>5}")
for nom, t in jeux.items():
    n_mal = int((t[:, 0] > args.seuil_mm).sum())
    print(f"{nom:32} {len(t):6d} {n_mal:12d} {100 * n_mal / len(t):4.0f}%")
tout = np.vstack(list(jeux.values()))
mal_tout = tout[:, 0] > args.seuil_mm

print("\nAUC (regroupé / par jeu : min–max) :")
meilleur, meilleur_auc = None, 0
for nom_i, (col, signe) in INDICATEURS.items():
    orientee = signe * tout[:, col]
    a = auc(mal_tout, orientee)
    par_jeu = [auc(t[:, 0] > args.seuil_mm, signe * t[:, col]) for t in jeux.values()]
    par_jeu = [v for v in par_jeu if not np.isnan(v)]
    print(f"  {nom_i:18} {a:.3f}   ({min(par_jeu):.2f}–{max(par_jeu):.2f})")
    if a > meilleur_auc:
        meilleur, meilleur_auc = nom_i, a
print(f"\nmeilleur indicateur : {meilleur} (AUC {meilleur_auc:.3f}) ; critère 1 (AUC >= 0,8) : {'atteint' if meilleur_auc >= 0.8 else 'NON atteint'}")

print("\nComptage, validation « un jeu laissé de côté » :")
for nom_i, (col, signe) in INDICATEURS.items():
    print(f"  indicateur {nom_i} :")
    tous_ok = True
    for laisse in jeux:
        appr = np.vstack([t for n, t in jeux.items() if n != laisse])
        v_appr = signe * appr[:, col]
        grille = np.unique(np.quantile(v_appr, np.linspace(0.005, 0.995, 199)))

        def ecart(seuil):
            """Somme, sur les jeux d'apprentissage, de |nombre prédit - nombre vrai| (prédit : indicateur orienté < seuil)."""
            return sum(abs(int((signe * t[:, col] < seuil).sum()) - int((t[:, 0] > args.seuil_mm).sum())) for n, t in jeux.items() if n != laisse)

        seuil = min(grille, key=ecart)
        t = jeux[laisse]
        vrai, predit = int((t[:, 0] > args.seuil_mm).sum()), int((signe * t[:, col] < seuil).sum())
        ok = abs(predit - vrai) <= max(3, 0.25 * vrai)
        tous_ok &= ok
        print(f"    {laisse:32} vrai {vrai:3d}  prédit {predit:3d}  {'ok' if ok else 'ÉCHEC'}")
    print(f"    -> critère 2 : {'atteint' if tous_ok else 'NON atteint'}")
