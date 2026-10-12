"""Concordance aux intersections de toutes les méthodes sur plusieurs sujets réels, sur toutes les paires et sur les paires NON utilisées par le coupe-coupe ancré.

Pour chaque sujet : tous les stacks TRUFI de la session (axial, coronal, sagittal du run 1, puis les stacks d'autres runs), leurs masques, et les poses de chaque méthode lues dans
`data/svr/results/<sujet>/` : poses d'en-tête (identité), SVRTK (`poses_svrtk.tsv`), NeSVoR (`poses_nesvor.tsv`), coupe-volume « sans la coupe » (`poses_loop_tous_loso.tsv`), coupe-coupe
(`poses_s2s_tous.tsv`, toutes paires) et coupe-coupe ancré sur le stack 0 (`poses_s2s_tous_ancre.tsv`). Mêmes coupes pour toutes les méthodes (intersection des coupes présentes).
Deux mesures (`intersection_consistency.concordance`) : toutes les paires (circulaire pour le coupe-coupe non ancré, qui l'optimise), et les paires qui n'impliquent pas le stack 0 (jamais
utilisées par le coupe-coupe ancré). Unité statistique : le sujet. Les identifiants de sujets sont passés en arguments (aucun dans le code) ; les résultats chiffrés sont écrits dans un TSV local.

Usage : python scripts/analyse_concordance_sujets.py --racine data/svr/jeu_reel_tru_haste --sujets sub-A sub-B ... --etiquettes "calme 1" "agité 1" ... [--sortie docs/svr/figures/concordance_sujets.tsv]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intersection_consistency import Jeu, concordance, lire_poses  # noqa: E402

PLANS = ["trufiax", "truficor", "trufisag"]
METHODES = [
    ("sans correction", None),
    ("SVRTK", "poses_svrtk.tsv"),
    ("NeSVoR", "poses_nesvor.tsv"),
    ("coupe-volume", "poses_loop_tous_loso.tsv"),
    ("coupe-coupe", "poses_s2s_tous.tsv"),
    ("coupe-coupe ancré", "poses_s2s_tous_ancre.tsv"),
]


def stacks_du_sujet(racine, sujet):
    """Chemins des stacks TRUFI et de leurs masques, dans l'ordre des poses (run 1 ax, cor, sag, puis les autres runs)."""
    ses = sorted(d for d in os.listdir(f"{racine}/{sujet}") if d.startswith("ses-"))[0]
    anat = f"{racine}/{sujet}/{ses}/anat"
    noms = []
    for f in os.listdir(anat):
        if not f.endswith("_T2w.nii.gz"):
            continue
        reste = f.removeprefix(f"{sujet}_{ses}_acq-").removesuffix("_T2w.nii.gz")
        if reste == f or "_run-" not in reste:
            continue
        plan, run = reste.split("_run-")
        if plan in PLANS:
            noms.append((int(run) != 1, int(run) * 10 + PLANS.index(plan), f"{sujet}_{ses}_acq-{plan}_run-{run}"))
    noms.sort()
    return ([f"{anat}/{n}_T2w.nii.gz" for _, _, n in noms], [f"{racine}/derivatives/medx-fetalbet/{sujet}/{ses}/anat/{n}_desc-brain_mask.nii.gz" for _, _, n in noms])


def mesurer(racine, sujet):
    """{méthode: (moyenne toutes paires, moyenne paires hors stack 0, nb paires, nb paires hors stack 0)} et le nombre de coupes communes."""
    stacks, masques = stacks_du_sujet(racine, sujet)
    jeu = Jeu(stacks, masques)
    poses = {nom: lire_poses(f"data/svr/results/{sujet}/{f}") for nom, f in METHODES if f}
    # une pose égale à l'identité est une coupe non recalée : non utilisée
    poses = {nom: {k: m for k, m in p.items() if not np.allclose(m, np.eye(4), atol=1e-12)} for nom, p in poses.items()}
    coupes = sorted(set.intersection(*[set(p) for p in poses.values()]))
    poses["sans correction"] = {k: np.eye(4) for k in coupes}
    out = {}
    for nom, _ in METHODES:
        tout = concordance(jeu, poses[nom], coupes)
        hors = concordance(jeu, poses[nom], coupes, hors_stack=0)
        out[nom] = (float(tout.mean()), float(hors.mean()), len(tout), len(hors))
    return out, len(coupes), len(stacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--racine", required=True)
    parser.add_argument("--sujets", nargs="+", required=True)
    parser.add_argument("--etiquettes", nargs="+", required=True)
    parser.add_argument("--sortie", default="docs/svr/figures/concordance_sujets.tsv")
    args = parser.parse_args()
    assert len(args.sujets) == len(args.etiquettes)
    noms = [m for m, _ in METHODES]
    resultats = {}
    for sujet, etiquette in zip(args.sujets, args.etiquettes):
        resultats[etiquette], n_coupes, n_stacks = mesurer(args.racine, sujet)
        print(f"{etiquette} : {n_stacks} stacks, {n_coupes} coupes communes", flush=True)
    lignes = ["sujet\tmesure\t" + "\t".join(noms)]
    for mesure, idx, titre in (("toutes", 0, "TOUTES LES PAIRES (circulaire pour coupe-coupe non ancré)"), ("hors_stack_0", 1, "PAIRES SANS LE STACK 0 (non utilisées par coupe-coupe ancré)")):
        print(f"\n{titre}\n{'':12}" + "".join(f"{n:>19}" for n in noms))
        for etiquette in args.etiquettes:
            v = [resultats[etiquette][n][idx] for n in noms]
            print(f"{etiquette:12}" + "".join(f"{x:19.4f}" for x in v))
            lignes.append(f"{etiquette}\t{mesure}\t" + "\t".join(f"{x:.5f}" for x in v))
        moyennes = [np.mean([resultats[e][n][idx] for e in args.etiquettes]) for n in noms]
        print(f"{'moyenne':12}" + "".join(f"{x:19.4f}" for x in moyennes))
        ancre = noms.index("coupe-coupe ancré")
        for autre in ("sans correction", "SVRTK", "NeSVoR", "coupe-volume", "coupe-coupe"):
            d = np.array([resultats[e]["coupe-coupe ancré"][idx] - resultats[e][autre][idx] for e in args.etiquettes])
            print(f"  ancré − {autre:16}: moyenne {d.mean():+.4f} ; gagne chez {int((d > 0).sum())}/{len(d)} ; min {d.min():+.4f} ; ≥ +0,05 chez {int((d >= 0.05).sum())}/{len(d)}")
    os.makedirs(os.path.dirname(args.sortie) or ".", exist_ok=True)
    open(args.sortie, "w").write("\n".join(lignes) + "\n")
    print("écrit :", args.sortie)
