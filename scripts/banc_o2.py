#!/usr/bin/env python3
"""F4.0.1, option O2 du bilan de la porte G0 (docs/ROADMAP.md) : critères (i) et (ii) du banc sur six reconstructions calmes.

Assemblage seulement : aucune mesure n'est nouvelle. Pour chaque sujet « dev » de `correspondances.tsv` (écrit par `banc_juge.py preparer`) :
  1. `medx svr simulate` aux réglages figés de `banc_juge.REGLAGES`, condition `calme` ;
  2. `medx svr recon` aux défauts, `--s2s-seed` = graine ;
  3. pour σ = 0 ; 0,5 ; 1 ; 2 voxels : `banc_simule.flouter` puis `banc_simule.lignes_de_mesure` (recalage rigide sur la source, normalisation affine, PSNR, SSIM, NCC,
     netteté, dans le masque de la source ; ni Dice ni erreur de pose).
Passes : `passe1` pour tous les sujets ; `passe2` (chaîne entière refaite, répertoire distinct) pour les sujets choisis, pour le critère (i).

  (i)  deux passes : piles simulées, volume reconstruit et volumes flous identiques bit à bit (SHA-256), toutes les mesures à écart absolu 0 ;
  (ii) pour chaque sujet : PSNR et SSIM strictement décroissants quand σ passe de 0 à 0,5 à 1 à 2.

Usage :
  python3 scripts/banc_o2.py executer --racine data/svr/banc/o2 --correspondances data/svr/banc/juge/correspondances.tsv --gabarits P1 P2 P3 \\
      --medx data/svr/banc/bin/medx_a03a39f --passes sub-b01:1 sub-b02:1 … sub-b01:2 sub-b06:2 [--jobs 2]
  python3 scripts/banc_o2.py verdict --racine data/svr/banc/o2 [--doubles sub-b01 sub-b06]

Les fichiers déjà présents sont réutilisés (reprise). Les résultats restent sous `--racine` (ignoré par Git).
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_juge as bjg
import banc_simule as bs

#: Flous du critère (ii), en voxels de la grille reconstruite ; 0 = volume non flouté.
SIGMAS = [0.0, 0.5, 1.0, 2.0]
#: Mesures comparées par (i) et (ii).
MESURES = ["psnr", "ssim", "ncc"]
REGLAGES_CALME = {**bjg.REGLAGES, "condition": "calme"}


def commande_de_simulation(medx, source, masque, gabarits, dossier):
    """Simulation aux réglages figés de F4.0.1 (amendements 1 à 3), condition `calme`."""
    return bjg.commande_de_simulation(medx, source, masque, gabarits, dossier, REGLAGES_CALME)


def somme(chemin):
    """SHA-256 d'un fichier."""
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def strictement_decroissant(valeurs):
    """Vrai si la suite est strictement décroissante (au moins deux valeurs finies)."""
    return len(valeurs) >= 2 and all(b < a for a, b in zip(valeurs, valeurs[1:]))


def verdict_ii(mesures):
    """`mesures` : {sigma: {"psnr": …, "ssim": …}} pour un sujet. Renvoie (tenu, détail) : PSNR et SSIM strictement décroissants sur SIGMAS."""
    detail = {m: [mesures[s][m] for s in SIGMAS] for m in ("psnr", "ssim")}
    return all(strictement_decroissant(v) for v in detail.values()), detail


def verdict_i(passe1, passe2):
    """Compare deux passes d'un sujet : `{"sommes": {nom: sha256}, "mesures": {sigma: {…}}}`. Renvoie (tenu, différences)."""
    diff = [f"somme {k}" for k in sorted(set(passe1["sommes"]) | set(passe2["sommes"])) if passe1["sommes"].get(k) != passe2["sommes"].get(k)]
    for s in SIGMAS:
        for m in MESURES:
            a, b = passe1["mesures"][str(s)][m], passe2["mesures"][str(s)][m]
            if a != b:
                diff.append(f"mesure {m} σ={s} : {a!r} contre {b!r}")
    return not diff, diff


def passe(a, code, numero, correspondance):
    """Une passe complète pour `code` ; écrit `<racine>/<code>/passe<n>/mesures.json` et le renvoie."""
    dossier = os.path.abspath(f"{a.racine}/{code}/passe{numero}")
    sortie = f"{dossier}/mesures.json"
    if os.path.exists(sortie):
        return json.load(open(sortie))
    simule = f"{dossier}/simule"
    if not os.path.exists(f"{simule}/sim_poses_vraies.tsv"):
        bs.executer(commande_de_simulation(a.medx, correspondance["source"], correspondance["masque"], a.gabarits, simule))
    piles, masques = lister(simule)
    volume, _, _ = bs.reconstruire(a.medx, piles, masques, [], REGLAGES_CALME["graine"], f"{dossier}/recon")
    sommes = {f"pile{i}": somme(p) for i, p in enumerate(piles)}
    sommes["volume"] = somme(volume)
    mesures = {}
    for sigma in SIGMAS:
        flou = f"{dossier}/flou_{sigma}.nii.gz"
        bs.flouter(volume, sigma, flou)
        sommes[f"flou_{sigma}"] = somme(flou)
        mesures[str(sigma)] = bs.lignes_de_mesure(correspondance["source"], correspondance["masque"], None, flou, f"{dossier}/mesure_{sigma}", a.medx, False, None, None, None)
    resultat = {"code": code, "passe": numero, "sommes": sommes, "mesures": mesures}
    json.dump(resultat, open(sortie, "w"), indent=1)
    return resultat


def lister(simule):
    """Piles simulées (ordre alphabétique de `sim_acq-<n>_T2w.nii.gz`) et masques associés."""
    piles = sorted(f"{simule}/{f}" for f in os.listdir(simule) if f.endswith("_T2w.nii.gz"))
    return piles, [p.replace("_T2w.nii.gz", "_mask.nii.gz") for p in piles]


def lire_passes(liste):
    """`["sub-b01:1", …]` -> `[("sub-b01", 1), …]`."""
    sortie = []
    for x in liste:
        code, _, n = x.partition(":")
        if not code or not n.isdigit():
            raise ValueError(f"passe invalide : {x!r} (attendu code:numéro)")
        sortie.append((code, int(n)))
    return sortie


def executer(a):
    with open(a.correspondances, newline="") as f:
        corr = {r["code"]: r for r in csv.DictReader(f, delimiter="\t")}
    taches = lire_passes(a.passes)
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futurs = {ex.submit(passe, a, code, n, corr[code]): (code, n) for code, n in taches}
        for f in cf.as_completed(futurs):
            code, n = futurs[f]
            f.result()
            print(f"{code} passe {n} faite", flush=True)


def verdict(a):
    passes = {}
    for code in sorted(os.listdir(a.racine)):
        for n in (1, 2):
            chemin = f"{a.racine}/{code}/passe{n}/mesures.json"
            if os.path.exists(chemin):
                passes[(code, n)] = json.load(open(chemin))
    lignes, tenu_ii, sujets = [], True, sorted({c for c, n in passes if n == 1})
    for c in sujets:
        ok_s, detail = verdict_ii({s: passes[(c, 1)]["mesures"][str(s)] for s in SIGMAS})
        tenu_ii &= ok_s
        for s in SIGMAS:
            m = passes[(c, 1)]["mesures"][str(s)]
            lignes.append({"code": c, "passe": 1, "sigma": s, **{k: m[k] for k in MESURES}})
        print(f"(ii) {c} : PSNR {['%.3f' % v for v in detail['psnr']]} ; SSIM {['%.4f' % v for v in detail['ssim']]} -> {'tenu' if ok_s else 'NON TENU'}")
    print(f"(ii) sur {len(sujets)} sujets : {'TENU' if tenu_ii and sujets else 'NON TENU'}")
    doubles = [c for c in sujets if (c, 2) in passes]
    tenu_i = bool(doubles)
    for c in doubles:
        ok, diff = verdict_i(passes[(c, 1)], passes[(c, 2)])
        tenu_i &= ok
        print(f"(i) {c} : {'identique' if ok else 'DIFFÉRENCES : ' + '; '.join(diff)}")
    print(f"(i) sur {len(doubles)} sujets : {'TENU' if tenu_i else 'NON TENU'}")
    if lignes:
        with open(f"{a.racine}/resultats.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lignes[0]), delimiter="\t")
            w.writeheader()
            w.writerows(lignes)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="action", required=True)
    e = sp.add_parser("executer")
    e.add_argument("--racine", required=True)
    e.add_argument("--correspondances", required=True)
    e.add_argument("--gabarits", nargs=3, required=True)
    e.add_argument("--medx", required=True)
    e.add_argument("--passes", nargs="+", required=True)
    e.add_argument("--jobs", type=int, default=2)
    v = sp.add_parser("verdict")
    v.add_argument("--racine", required=True)
    a = p.parse_args()
    {"executer": executer, "verdict": verdict}[a.action](a)


if __name__ == "__main__":
    main()
