"""Figure qualitative : les reconstructions de chaque méthode, côte à côte, pour plusieurs sujets réels.

Pour chaque sujet, au repli `--repli` (un stack mis de côté, reconstruction avec les deux autres : les volumes sont ceux que le juge a évalués, voir `benchmark_real.py`), les cinq volumes : sans correction,
SVRTK (défauts), NeSVoR (défauts, sans SVoRT, CPU), notre coupe-volume (« sans la coupe ») et notre coupe-coupe. Chaque volume est recalé rigidement (SimpleITK, corrélation) sur la reconstruction coupe-coupe du même sujet (le repère de sortie
d'une reconstruction n'est défini qu'à un mouvement rigide près), puis on affiche deux coupes passant par le centre du volume reconstruit, perpendiculaires aux axes z et y de la grille de reconstruction (repère du scanner : ce ne sont pas des plans anatomiques) (même emplacement pour toutes les méthodes,
choisi sans regarder les images), avec la même fenêtre d'intensité (0 à 99,5e centile) pour une méthode donnée. Chaque cellule indique la NCC moyenne du juge de ce repli. Le repli par défaut (1) est fixé
avant d'examiner les images. Les identifiants de sujets sont passés en arguments (aucun dans le code). Données de patientes : la figure est écrite dans docs/svr/figures/ (ignoré par Git), à ne pas publier.

Usage : python scripts/figure_qualitative_sujets.py --sujets sub-A sub-B ... --etiquettes "calme 1" ... [--repli 1] [--substitut sub-X:2=fichier]
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
parser.add_argument("--repli", type=int, default=1)
parser.add_argument("--substitut", action="append", default=[])
parser.add_argument("--sortie", default="docs/svr/figures/qualitatif_sujets.png")
args = parser.parse_args()
assert len(args.sujets) == len(args.etiquettes)
subst = {}
for s in args.substitut:
    cle, f = s.split("=", 1)
    sujet, h = cle.split(":")
    subst[(sujet, int(h))] = f

H = args.repli
METHODES = [
    ("sans correction", "ours_header_sans{h}"),
    ("SVRTK (défauts)", "svrtk_sans{h}"),
    ("NeSVoR (défauts)", "nesvor_sans{h}"),
    ("notre coupe-volume", "ours_loop_loso_sans{h}"),
    ("notre coupe-coupe", "ours_s2s_sans{h}"),
]
REFERENCE = "ours_s2s_sans{h}"


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


def ncc_juge(sujet, modele):
    f = subst.get((sujet, H), modele.format(h=H)) if "loso" in modele else modele.format(h=H)
    v = [float(r["ncc"]) for r in csv.DictReader(open(f"data/svr/results/{sujet}/{f}.nii.gz.juge.tsv"), delimiter="\t")]
    return float(np.mean(v))


def volumes(sujet):
    """Les cinq volumes du sujet sur la grille de la reconstruction coupe-coupe."""
    dossier = f"data/svr/results/{sujet}"
    fixe = lire(f"{dossier}/{REFERENCE.format(h=H)}.nii.gz")
    out = []
    for nom, modele in METHODES:
        f = subst.get((sujet, H), modele.format(h=H)) if "loso" in modele else modele.format(h=H)
        if modele == REFERENCE:
            out.append(sitk.GetArrayFromImage(fixe).astype(np.float64))
        else:
            mobile = lire(f"{dossier}/{f}.nii.gz")
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
        for m, ((nom, modele), im) in enumerate(zip(METHODES, ims)):
            ax = axes[ligne, v * len(METHODES) + m]
            haut = np.percentile(im[masque], 99.5)
            ax.imshow(extraire(im, centre)[max(a0 - marge, 0):a1 + marge, max(b0 - marge, 0):b1 + marge], cmap="gray", vmin=0, vmax=haut, origin="lower", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.03, f"NCC {ncc_juge(sujet, modele):.3f}", transform=ax.transAxes, color="yellow", fontsize=8, va="bottom")
            if ligne == 0:
                ax.set_title(nom, fontsize=9.5)
            if v == 0 and m == 0:
                ax.set_ylabel(etiquette, fontsize=10)
for v, (nom_vue, _, _) in enumerate(VUES):
    fig.text((v * len(METHODES) + len(METHODES) / 2) / (2 * len(METHODES)) * 0.97 + 0.02, 0.985, nom_vue, ha="center", fontsize=11.5, fontweight="bold")
fig.suptitle(f"Reconstructions à 2 stacks, repli {H} (un stack mis de côté) ; NCC = prédiction du stack mis de côté, comme dans la figure quantitative", fontsize=9.5, y=0.012, va="bottom")
fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=0.4, w_pad=0.25)
fig.savefig(args.sortie, dpi=110)
print("écrit :", args.sortie)
