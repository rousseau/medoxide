"""Figure qualitative à jour : les reconstructions de chaque méthode avec TOUS les stacks TRUFI, côte à côte, pour plusieurs sujets réels.

Pour chaque sujet : sans correction, SVRTK (défauts), NeSVoR (défauts, sans SVoRT), NiftyMIC (défauts sauf : résolution 0,8 mm, sans correction du champ de biais, masques utilisés aussi pour la reconstruction `--use-masks-srr 1` ; `read_niftymic_poses.py`), notre coupe-volume (« sans la coupe ») et notre coupe-coupe, reconstruits avec tous les stacks (3, ou 4 pour un
sujet) et nos réglages par défaut actuels (α = 0,1, passe-haut des profils, couverture minimale 2 ; pour le coupe-coupe, alignement par stack, correction du champ de biais et régularisation temporelle λ = 0,01 ; la colonne « sans correction » n'a ni mouvement estimé, ni blocs, ni correction de biais) ; fichiers `*_defaut` de `data/svr/results/<sujet>/`, produits par `medx svr recon` (`--s2s-sweeps 0 --s2s-bias-rounds 0 --s2s-block-levels 0` pour « sans correction »). Quand NiftyMIC n'a pas écrit de volume (échec), la cellule le dit. Chaque volume est recalé rigidement (SimpleITK, corrélation) sur la reconstruction coupe-coupe du même sujet (le repère de sortie n'est
défini qu'à un mouvement rigide près), puis on affiche deux coupes passant par le centre du volume reconstruit, perpendiculaires aux axes z et y de la grille de reconstruction (repère du scanner : ce ne
sont pas des plans anatomiques), même emplacement et même fenêtre d'intensité (0 à 99,5e centile de chaque volume) pour toutes les méthodes, choisis sans regarder les images. Chaque cellule indique la
concordance aux intersections (toutes paires) de la méthode, lue dans le TSV de `analyse_concordance_sujets.py` (circulaire pour le coupe-coupe) ; pour NiftyMIC, qui rejette des coupes, elle est mesurée sur les seules coupes qu'il a gardées, dont le nombre est indiqué (k/n) ; tous les volumes sont limités au même masque (le domaine de la reconstruction coupe-coupe, issu des masques fetalBET : voxels atteints par au moins deux stacks) et recalés rigidement avec une métrique évaluée dans ce masque ; NiftyMIC est lancé avec `--use-masks-srr 1` (par défaut il n'utilise les masques que pour le recalage et reconstruit à partir des images entières) ; SVRTK reçoit le masque du premier stack (région d'intérêt), NeSVoR les masques de tous les stacks. Les identifiants de sujets sont passés en
arguments (aucun dans le code). Données de patientes : la figure est écrite dans docs/svr/figures/ (ignoré par Git), à ne pas publier.

Usage : python scripts/figure_reconstructions_tous.py --sujets sub-A sub-B ... --etiquettes "calme 1" ... [--tsv docs/svr/figures/concordance_sujets_defaut.tsv]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument("--sujets", nargs="+", required=True)
parser.add_argument("--etiquettes", nargs="+", required=True)
parser.add_argument("--tsv", default="docs/svr/figures/concordance_sujets_defaut.tsv")
parser.add_argument("--sortie", default="docs/svr/figures/reconstructions_tous.png")
args = parser.parse_args()
assert len(args.sujets) == len(args.etiquettes)

METHODES = [
    ("sans correction", "ours_header_defaut.nii.gz"),
    ("SVRTK (défauts)", "svrtk_debug/svrtk.nii.gz"),
    ("NeSVoR (défauts)", "nesvor/nesvor.nii.gz"),
    ("NiftyMIC (défauts)", "niftymic/srr.nii.gz"),
    ("notre coupe-volume", "ours_s2v_defaut.nii.gz"),
    ("notre coupe-coupe", "ours_s2s_defaut.nii.gz"),
]
COLONNES = {"sans correction": "sans correction", "SVRTK (défauts)": "SVRTK", "NeSVoR (défauts)": "NeSVoR", "NiftyMIC (défauts)": "NiftyMIC", "notre coupe-volume": "coupe-volume", "notre coupe-coupe": "coupe-coupe"}
REFERENCE = "ours_s2s_defaut.nii.gz"
CONC, NB = {}, {}
for ligne in csv.DictReader(open(args.tsv), delimiter="\t"):
    cible = CONC if ligne["mesure"] == "toutes" else NB if ligne["mesure"] == "coupes" else None
    if cible is not None:
        cible[ligne["sujet"]] = {nom: ligne[col] for nom, col in COLONNES.items()}


def lire(chemin):
    return sitk.Cast(sitk.ReadImage(chemin), sitk.sitkFloat32)


def rigide_vers(fixe, mobile, masque_fixe):
    """Transformation rigide recalant `mobile` sur `fixe`, la métrique n'étant évaluée que dans le masque commun (domaine de la reconstruction coupe-coupe, issu des masques fetalBET) : le tissu hors du cerveau ne guide pas le recalage."""
    init = sitk.CenteredTransformInitializer(fixe, mobile, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricFixedMask(masque_fixe)
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
    """Les volumes du sujet sur la grille de la reconstruction coupe-coupe, TOUS multipliés par le même masque : le domaine de cette reconstruction (voxels de masque fetalBET atteints par au moins deux stacks)."""
    dossier = f"data/svr/results/{sujet}"
    fixe = lire(f"{dossier}/{REFERENCE}")
    domaine = sitk.GetArrayFromImage(fixe) > 0
    masque_fixe = sitk.Cast(fixe > 0, sitk.sitkUInt8)
    out = []
    for nom, f in METHODES:
        if f == REFERENCE:
            out.append(sitk.GetArrayFromImage(fixe).astype(np.float64))
        elif not os.path.exists(f"{dossier}/{f}"):
            out.append(None)  # méthode sans volume (échec)
        else:
            mobile = lire(f"{dossier}/{f}")
            im = sitk.GetArrayFromImage(sitk.Resample(mobile, fixe, rigide_vers(fixe, mobile, masque_fixe), sitk.sitkLinear, 0.0)).astype(np.float64)
            out.append(im * domaine)
    return out


# plans de la GRILLE de reconstruction (repère du scanner), pas des plans anatomiques : la tête du fœtus a une orientation quelconque
VUES = [("plan z de la grille", lambda im, c: im[c[0]], (1, 2)), ("plan y de la grille", lambda im, c: im[:, c[1], :], (0, 2))]
n = len(args.sujets)
fig, axes = plt.subplots(n, 2 * len(METHODES), figsize=(1.55 * 2 * len(METHODES), 2.15 * n + 0.9))
for ligne, (sujet, etiquette) in enumerate(zip(args.sujets, args.etiquettes)):
    print("sujet", etiquette, flush=True)
    ims = volumes(sujet)
    ref = ims[len(METHODES) - 1]  # notre coupe-coupe
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
            if ligne == 0:
                ax.set_title(nom, fontsize=9.5)
            if v == 0 and m == 0:
                ax.set_ylabel(etiquette, fontsize=10)
            if im is None:
                ax.imshow(np.zeros_like(extraire(ims[-1], centre)[max(a0 - marge, 0):a1 + marge, max(b0 - marge, 0):b1 + marge]), cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "pas de volume\n(échec)", transform=ax.transAxes, color="red", fontsize=8, ha="center", va="center")
                continue
            haut = np.percentile(im[masque], 99.5)
            ax.imshow(extraire(im, centre)[max(a0 - marge, 0):a1 + marge, max(b0 - marge, 0):b1 + marge], cmap="gray", vmin=0, vmax=haut, origin="lower", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            partiel = nom.startswith("NiftyMIC")
            valeur = float(CONC[etiquette][nom])
            ax.text(0.03, 0.03, ("conc. n/a" if np.isnan(valeur) else f"conc. {valeur:.3f}") + (f"\n{NB[etiquette][nom]}/{NB[etiquette]['notre coupe-coupe']} coupes" if partiel else ""), transform=ax.transAxes, color="yellow", fontsize=8, va="bottom")
for v, (nom_vue, _, _) in enumerate(VUES):
    fig.text((v * len(METHODES) + len(METHODES) / 2) / (2 * len(METHODES)) * 0.97 + 0.02, 0.985, nom_vue, ha="center", fontsize=11.5, fontweight="bold")
fig.suptitle("Reconstructions avec tous les stacks TRUFI ; conc. = concordance aux intersections (toutes paires ; circulaire pour le coupe-coupe ; NiftyMIC : sur les coupes qu'il a gardées ; tous les volumes limités au même masque fetalBET)", fontsize=9.5, y=0.012, va="bottom")
fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=0.4, w_pad=0.25)
fig.savefig(args.sortie, dpi=110)
print("écrit :", args.sortie)
