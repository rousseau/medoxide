#!/usr/bin/env python3
"""F4.0.6, étape E43 : lecture des tableaux du test ignoré `diagnostic_des_poses_divergentes` (crates/medoxide-svr/src/diagnostic_pose.rs).

Usage : gradient_pose_enquete.py <dossier d'un sujet> [--seuil 0.1] [--suspect 0.9]

Le dossier contient `cycles.tsv` (toutes les coupes, tous les cycles), `coupes.tsv` (coupes à écart isolé > seuil : diagnostic à part),
`essais.tsv` (7 paires de recalages par coupe divergente) et `finale.tsv` (écart final des deux trajectoires). Aucun identifiant de sujet
n'est écrit : le nom du dossier n'est pas repris dans la sortie.

Classement de chaque coupe à plus de `seuil` mm à la fin (trajectoires complètes) :
  - « propre »      : son recalage isolé (même départ, même référence) diverge dans un cycle, donc l'écart naît du recalage lui-même ;
  - « seuil »       : dans un cycle de la trajectoire, la coupe est relancée dans une trajectoire et pas dans l'autre (NCC du premier recalage
                      de part et d'autre du seuil) ;
  - « voisine »     : une coupe voisine du même stack (k ± 1) est propre ou « seuil » : l'écart arrive par l'a priori temporel et la référence ;
  - « référence »   : aucun des trois : l'écart vient des références, devenues différentes, des autres coupes.
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def lire(chemin):
    with open(chemin, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dossier", type=Path)
    ap.add_argument("--seuil", type=float, default=0.1)
    ap.add_argument("--suspect", type=float, default=0.9)
    a = ap.parse_args()
    cycles = lire(a.dossier / "cycles.tsv")
    coupes = lire(a.dossier / "coupes.tsv")
    essais = lire(a.dossier / "essais.tsv")
    finale = lire(a.dossier / "finale.tsv")

    f = {(int(r["stack"]), int(r["coupe"])): float(r["depl_final_mm"]) for r in finale}
    n = len(f)
    sup = sorted(((k, d) for k, d in f.items() if d > a.seuil), key=lambda x: -x[1])
    print(f"## Trajectoires complètes : {n} coupes ; > {a.seuil} mm : {len(sup)} ({100*len(sup)/n:.1f} %) ; > 1 mm : "
          f"{sum(1 for _, d in sup if d > 1)} ({100*sum(1 for _, d in sup if d > 1)/n:.1f} %)\n")

    iso = defaultdict(list)  # (stack, coupe) -> cycles où l'écart isolé dépasse le seuil
    seuil_flip = defaultdict(list)  # (stack, coupe) -> cycles où relancée d'un côté seulement (trajectoires complètes)
    for r in cycles:
        k = (int(r["stack"]), int(r["coupe"]))
        c = int(r["cycle"])
        if float(r["depl_iso_mm"]) > a.seuil:
            iso[k].append(c)
        if "relances_N" in r and int(r["relances_A"]) != int(r["relances_N"]):
            seuil_flip[k].append(c)
    print("| stack/coupe | écart final (mm) | classe | cycles « propre » | cycles « seuil » |")
    print("|---|---|---|---|---|")
    classes = defaultdict(int)
    for (s, k), d in sup:
        if (s, k) in iso:
            cl = "propre"
        elif (s, k) in seuil_flip:
            cl = "seuil"
        elif any(((s, v) in iso or (s, v) in seuil_flip) for v in (k - 1, k + 1)):
            cl = "voisine"
        else:
            cl = "référence"
        classes[cl] += 1
        print(f"| {s}/{k} | {d:.3f} | {cl} | {iso.get((s, k), '')} | {seuil_flip.get((s, k), '')} |")
    print(f"\nClasses : {dict(classes)}\n")

    # recalages isolés : même état de départ pour les deux gradients
    total = len(cycles)
    nbr = sum(1 for r in cycles if float(r["depl_iso_mm"]) > a.seuil)
    print(f"## Recalages isolés (un cycle depuis les mêmes poses) : {total} recalages de coupe ; > {a.seuil} mm : {nbr} "
          f"({100*nbr/total:.2f} %) ; > 1 mm : {sum(1 for r in cycles if float(r['depl_iso_mm']) > 1)}\n")
    memes = sum(1 for r in cycles if int(r["relances_A"]) == int(r["relances_I"]))
    print(f"Nombre de recalages (relances) identique dans les deux variantes : {memes} / {total}.\n")
    dc = [abs(float(r["ncc_first_A"]) - float(r["ncc_first_I"])) for r in cycles if int(r["pixels"]) > 0]
    print(f"Écart de NCC du premier recalage, |A − I| : médiane {statistics.median(dc):.2e}, max {max(dc):.2e}.\n")
    proches = [r for r in cycles if abs(float(r["ncc_first_A"]) - a.suspect) < 0.01]
    print(f"Coupes dont la NCC du premier recalage est à moins de 0,01 du seuil {a.suspect} : {len(proches)} / {total}.\n")

    if not coupes:
        return
    print(f"## Les {len(coupes)} recalages isolés à plus de {a.seuil} mm\n")
    print("| cycle | s/k | pixels | écart iso (mm) | NCC 1er A / I | relances A / I | premiers recalages (mm) | meilleur (perte f64) | "
          "Δperte (I−A) | err. gradient 0 / A / I |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    px = {(int(r["cycle"]), int(r["stack"]), int(r["coupe"])): r["pixels"] for r in cycles}
    verdicts = defaultdict(int)
    for r in coupes:
        key = (int(r["cycle"]), int(r["stack"]), int(r["coupe"]))
        dperte = float(r["perte_meilleure_I"]) - float(r["perte_meilleure_A"])
        verdicts[r["meilleur"]] += 1
        print(f"| {key[0]} | {key[1]}/{key[2]} | {px[key]} | {float(r['depl_iso_mm']):.3f} | {float(r['ncc_first_A']):.3f} / "
              f"{float(r['ncc_first_I']):.3f} | {r['relances_A']} / {r['relances_I']} | {float(r['premiers_recalages_mm']):.3f} | "
              f"{r['meilleur']} | {dperte:+.2e} | {r['err_grad_zero']} / {r['err_grad_A']} / {r['err_grad_I']} |")
    print(f"\nMeilleur minimum (perte `f64` indépendante, parmi les essais réellement faits) : {dict(verdicts)}.\n")
    errs = [float(r[c]) for r in coupes for c in ("err_grad_zero", "err_grad_A", "err_grad_I")]
    print(f"Erreur relative du gradient analytique contre différences finies `f64` sur ces coupes (3 poses chacune) : "
          f"médiane {statistics.median(errs):.2e}, max {max(errs):.2e}.\n")

    # essais deux à deux : à départ égal, les deux gradients retombent-ils au même endroit ?
    par_depart = [(float(r["depl_A_I_mm"]), float(r["perte_f64_I"]) - float(r["perte_f64_A"])) for r in essais]
    for nom, cond in (("écart < 0,01 mm", lambda d: d < 0.01), ("0,01 à 0,1 mm", lambda d: 0.01 <= d < 0.1),
                      ("0,1 à 1 mm", lambda d: 0.1 <= d < 1), ("≥ 1 mm", lambda d: d >= 1)):
        ens = [x for x in par_depart if cond(x[0])]
        if ens:
            print(f"Essais à départ commun, {nom} : {len(ens)} sur {len(par_depart)} ; |Δperte| médiane "
                  f"{statistics.median(abs(x[1]) for x in ens):.2e}")
    print()
    # amplitude de la plage de pertes atteinte par les essais d'une même coupe
    plage = defaultdict(list)
    for r in essais:
        plage[(r["cycle"], r["stack"], r["coupe"])].extend([float(r["perte_f64_A"]), float(r["perte_f64_I"])])
    p = [max(v) - min(v) for v in plage.values()]
    print(f"Plage de perte (max − min sur les 14 recalages d'une coupe) : médiane {statistics.median(p):.2e}, max {max(p):.2e}.")


if __name__ == "__main__":
    sys.exit(main())
