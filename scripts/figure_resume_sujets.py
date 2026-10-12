"""Figure de synthèse du juge hors échantillon sur plusieurs sujets réels (sans vérité terrain).

Pour chaque sujet : NCC moyenne par coupe, sur les trois replis (coupes communes à toutes les méthodes), de chaque méthode, lue dans `data/svr/results/<sujet>/<volume>.juge.tsv` (voir `benchmark_real.py`,
`analyse_juge_sujets.py`). Panneau A : une valeur par sujet et par méthode (points), avec l'étendue des trois replis (trait fin). Panneau B : différence à « sans correction », par sujet (au-dessus de 0,
la méthode prédit mieux les coupes inconnues que l'absence de correction), et moyenne sur les sujets. Les identifiants de sujets sont passés en arguments : aucun dans le code. Les données sont celles de
patientes : la figure est écrite dans docs/svr/figures/ (ignoré par Git) et ne doit pas être publiée.

Usage : python scripts/figure_resume_sujets.py --sujets sub-A sub-B ... --etiquettes "calme 1" "agité 1" ... [--substitut sub-X:2=fichier] [--sortie docs/svr/figures/resume_sujets.png]
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sujets", nargs="+", required=True)
parser.add_argument("--etiquettes", nargs="+", required=True)
parser.add_argument("--substitut", action="append", default=[])
parser.add_argument("--sortie", default="docs/svr/figures/resume_sujets.png")
args = parser.parse_args()
assert len(args.sujets) == len(args.etiquettes)
subst = {}
for s in args.substitut:
    cle, f = s.split("=", 1)
    sujet, h = cle.split(":")
    subst[(sujet, int(h))] = f

# nom affiché : (fichier par repli, couleur, marqueur)
METHODES = {
    "sans correction": ("ours_header_sans{h}", "#7f7f7f", "s"),
    "SVRTK (défauts)": ("svrtk_sans{h}", "#e08214", "D"),
    "NeSVoR (défauts, sans SVoRT)": ("nesvor_sans{h}", "#7b3294", "v"),
    "notre coupe-volume (sans la coupe)": ("ours_loop_loso_sans{h}", "#2166ac", "o"),
    "notre coupe-coupe": ("ours_s2s_sans{h}", "#1a9850", "^"),
}


def lire(sujet, modele, h):
    f = subst.get((sujet, h), modele.format(h=h)) if "loso" in modele else modele.format(h=h)
    return {int(r["coupe"]): float(r["ncc"]) for r in csv.DictReader(open(f"data/svr/results/{sujet}/{f}.nii.gz.juge.tsv"), delimiter="\t")}


moy, etendue = {}, {}
for sujet in args.sujets:
    par_repli = {m: [[] for _ in range(3)] for m in METHODES}
    for h in range(3):
        d = {m: lire(sujet, METHODES[m][0], h) for m in METHODES}
        for k in sorted(set.intersection(*[set(x) for x in d.values()])):
            for m in METHODES:
                par_repli[m][h].append(d[m][k])
    for m in METHODES:
        tout = [v for r in par_repli[m] for v in r]
        moy[(sujet, m)] = float(np.mean(tout))
        etendue[(sujet, m)] = (min(np.mean(r) for r in par_repli[m]), max(np.mean(r) for r in par_repli[m]))

x = np.arange(len(args.sujets))
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [1.1, 1]}, sharex=True)
decalage = np.linspace(-0.27, 0.27, len(METHODES))
for (m, (_, couleur, marqueur)), dx in zip(METHODES.items(), decalage):
    ax_a.vlines(x + dx, [etendue[(s, m)][0] for s in args.sujets], [etendue[(s, m)][1] for s in args.sujets], color=couleur, lw=1.2, alpha=0.6)
    ax_a.scatter(x + dx, [moy[(s, m)] for s in args.sujets], color=couleur, marker=marqueur, s=46, label=m, zorder=3)
ax_a.set_ylabel("NCC hors échantillon (juge)")
ax_a.set_title("A. Prédiction des coupes d'un stack mis de côté (plus haut = mieux ; trait = étendue des 3 replis)", fontsize=10.5, loc="left")
ax_a.grid(axis="y", alpha=0.3)
ax_a.legend(fontsize=8.5, loc="lower left", ncol=2, frameon=True)

reference = "sans correction"
autres = [m for m in METHODES if m != reference]
decalage_b = np.linspace(-0.22, 0.22, len(autres))
moyennes_b = {}
for m, dx in zip(autres, decalage_b):
    _, couleur, marqueur = METHODES[m]
    ax_b.scatter(x + dx, [moy[(s, m)] - moy[(s, reference)] for s in args.sujets], color=couleur, marker=marqueur, s=50, zorder=3)
    moyenne = np.mean([moy[(s, m)] - moy[(s, reference)] for s in args.sujets])
    ax_b.hlines(moyenne, len(args.sujets) - 0.45, len(args.sujets) + 0.45, color=couleur, lw=2.5)
    moyennes_b[m] = moyenne
# étiquettes des moyennes : écartées d'au moins 0,016 pour rester lisibles
ordre = sorted(moyennes_b, key=moyennes_b.get)
positions, dernier = {}, -9.0
for m in ordre:
    dernier = max(moyennes_b[m], dernier + 0.016)
    positions[m] = dernier
for m in ordre:
    ax_b.text(len(args.sujets) + 0.5, positions[m], f"{moyennes_b[m]:+.3f}", color=METHODES[m][1], va="center", fontsize=9)
ax_b.axhline(0, color="k", lw=1)
ax_b.axvline(len(args.sujets) - 0.5, color="#bbbbbb", lw=1, ls="--")
ax_b.set_xticks(list(x) + [len(args.sujets)])
ax_b.set_xticklabels(args.etiquettes + ["moyenne"], rotation=0, fontsize=9)
ax_b.set_xlim(-0.6, len(args.sujets) + 1.0)
ax_b.set_ylabel("Différence à « sans correction »")
ax_b.set_title("B. Gain sur l'absence de correction (au-dessus de 0 : la méthode aide)", fontsize=10.5, loc="left")
ax_b.grid(axis="y", alpha=0.3)
fig.suptitle("Juge hors échantillon sur six sujets réels (TRUFI, reconstruction à 2 stacks, 3 replis)", fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig(args.sortie, dpi=150)
print("écrit :", args.sortie)
for s, e in zip(args.sujets, args.etiquettes):
    print(f"{e:12}", " ".join(f"{m.split(' (')[0][:14]:>14} {moy[(s, m)]:.4f}" for m in METHODES))
