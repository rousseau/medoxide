"""Figure qualitative à jour : les reconstructions de chaque méthode avec TOUS les stacks TRUFI, côte à côte, pour plusieurs sujets réels.

Pour chaque sujet : sans correction, SVRTK (défauts), NeSVoR (défauts, sans SVoRT), notre coupe-volume (« sans la coupe ») et notre coupe-coupe, reconstruits avec tous les stacks (3, ou 4 pour un
sujet : voir `analyse_concordance_sujets.py` pour les fichiers). Chaque volume est recalé rigidement (SimpleITK, corrélation) sur la reconstruction coupe-coupe du même sujet (le repère de sortie n'est
défini qu'à un mouvement rigide près), puis on affiche deux coupes passant par le centre du volume reconstruit, perpendiculaires aux axes z et y de la grille de reconstruction (repère du scanner : ce ne
sont pas des plans anatomiques), même emplacement et même fenêtre d'intensité (0 à 99,5e centile de chaque volume) pour toutes les méthodes, choisis sans regarder les images. Chaque cellule indique la
concordance aux intersections (toutes paires) de la méthode, lue dans le TSV de `analyse_concordance_sujets.py` (circulaire pour le coupe-coupe). Les identifiants de sujets sont passés en
arguments (aucun dans le code). Données de patientes : la figure est écrite dans docs/svr/figures/ (ignoré par Git), à ne pas publier.

Usage : python scripts/figure_reconstructions_tous.py --sujets sub-A sub-B ... --etiquettes "calme 1" ... [--tsv docs/svr/figures/concordance_sujets.tsv]
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument("--sujets", nargs="+", required=True)
parser.add_argument("--etiquettes", nargs="+", required=True)
parser.add_argument("--tsv", default="docs/svr/figures/concordance_sujets.tsv")
parser.add_argument("--sortie", default="docs/svr/figures/reconstructions_tous.png")
args = parser.parse_args()
assert len(args.sujets) == len(args.etiquettes)

METHODES = [
    ("sans correction", "ours_header_tous_loso.nii.gz"),
    ("SVRTK (défauts)", "svrtk_debug/svrtk.nii.gz"),
    ("NeSVoR (défauts)", "nesvor/nesvor.nii.gz"),
    ("notre coupe-volume", "ours_loop_tous_loso.nii.gz"),
    ("notre coupe-coupe", "ours_s2s_tous.nii.gz"),
]
REFERENCE = "ours_s2s_tous.nii.gz"
CONC = {}
for ligne in csv.DictReader(open(args.tsv), delimiter="\t"):
    if ligne["mesure"] == "toutes":
        CONC[ligne["sujet"]] = {"sans correction": ligne["sans correction"], "SVRTK (défauts)": ligne["SVRTK"], "NeSVoR (défauts)": ligne["NeSVoR"], "notre coupe-volume": ligne["coupe-volume"], "notre coupe-coupe": ligne["coupe-coupe"]}


def lire(chemin):
    return sitk.Cast(sitk.ReadImage(chemin), sitk.sitkFloat32)


def rigide_vers(fixe, mobile):
    init = sitk.CenteredTransformInitializer(fixe, mobile, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsCorrelation()
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.25, 42)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.5)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    r.SetInitialTransform(init, inPlace=False)
    return r.Execute(fixe, mobile)


def volumes(sujet):
    """Les volumes du sujet sur la grille de la reconstruction coupe-coupe."""
    dossier = f"data/svr/results/{sujet}"
    fixe = lire(f"{dossier}/{REFERENCE}")
    out = []
    for nom, f in METHODES:
        if f == REFERENCE:
            out.append(sitk.GetArrayFromImage(fixe).astype(np.float64))
        else:
            mobile = lire(f"{dossier}/{f}")
            out.append(sitk.GetArrayFromImage(sitk.Resample(mobile, fixe, rigide_vers(fixe, mobile), sitk.sitkLinear, 0.0)).astype(np.float64))
    return out


# plans de la GRILLE de reconstruction (repère du scanner), pas des plans anatomiques : la tête du fœtus a une orientation quelconque
VUES = [("plan z de la grille", lambda im, c: im[c[0]], (1, 2)), ("plan y de la grille", lambda im, c: im[:, c[1], :], (0, 2))]
n = len(args.sujets)
fig, axes = plt.subplots(n, 2 * len(METHODES), figsize=(1.7 * 2 * len(METHODES), 2.15 * n + 0.9))
for ligne, (sujet, etiquette) in enumerate(zip(args.sujets, args.etiquettes)):
    print("sujet", etiquette, flush=True)
    ims = volumes(sujet)
    ref = ims[len(METHODES) - 1]
    masque = ref > 0
    centre = np.argwhere(masque).mean(axis=0).round().astype(int)  # (z, y, x)
    for v, (nom_vue, extraire, axes_boite) in enumerate(VUES):
        # boîte englobante du volume reconstruit dans ce plan (même recadrage pour toutes les méthodes)
        coupe_ref = extraire(masque, centre)
        pts = np.argwhere(coupe_ref)
        (a0, b0), (a1, b1) = pts.min(axis=0), pts.max(axis=0)
        marge = 4
        for m, ((nom, _), im) in enumerate(zip(METHODES, ims)):
            ax = axes[ligne, v * len(METHODES) + m]
            haut = np.percentile(im[masque], 99.5)
            ax.imshow(extraire(im, centre)[max(a0 - marge, 0):a1 + marge, max(b0 - marge, 0):b1 + marge], cmap="gray", vmin=0, vmax=haut, origin="lower", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.03, f"conc. {float(CONC[etiquette][nom]):.3f}", transform=ax.transAxes, color="yellow", fontsize=8, va="bottom")
            if ligne == 0:
                ax.set_title(nom, fontsize=9.5)
            if v == 0 and m == 0:
                ax.set_ylabel(etiquette, fontsize=10)
for v, (nom_vue, _, _) in enumerate(VUES):
    fig.text((v * len(METHODES) + len(METHODES) / 2) / (2 * len(METHODES)) * 0.97 + 0.02, 0.985, nom_vue, ha="center", fontsize=11.5, fontweight="bold")
fig.suptitle("Reconstructions avec tous les stacks TRUFI ; conc. = concordance aux intersections (toutes paires ; circulaire pour le coupe-coupe)", fontsize=9.5, y=0.012, va="bottom")
fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=0.4, w_pad=0.25)
fig.savefig(args.sortie, dpi=110)
print("écrit :", args.sortie)
