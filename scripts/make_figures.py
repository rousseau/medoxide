"""Figures de résultats : jeux synthétiques (atlas, pyrecon) et sujets réels (sans vérité terrain).

Synthétique : pour chaque condition de mouvement, une coupe axiale de la vérité (atlas), de la reconstruction sans correction, de la boucle de medoxide, de SVRTK (réglages par défaut) et de
la reconstruction aux vraies poses (borne haute). Chaque reconstruction est d'abord calée rigidement sur l'atlas (SimpleITK, corrélation : une reconstruction n'est définie qu'à un mouvement
rigide global près) puis rééchantillonnée sur sa grille ; le PSNR (tout le tissu de l'atlas lisible dans les 3 stacks, non nul pour toutes les méthodes, dynamique = 99e centile de l'atlas) est
le même calcul que `compare_reconstructions.py --align` : les valeurs doivent concorder avec le tableau de l'étude.

Réel : les trois coupes orthogonales de la reconstruction sans correction, de medoxide et de SVRTK, calées rigidement sur celle de medoxide, et le déplacement du centre de chaque coupe estimé
par medoxide. Les données réelles sont celles de patientes : les figures sont écrites dans docs/svr/figures/ (ignoré par Git) et ne doivent pas être publiées.

Usage : python scripts/make_figures.py synthetique
        python scripts/make_figures.py reel --sujet sub-XXXX --racine data/svr/jeu_reel_tru_haste
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

SORTIE = "docs/svr/figures"
os.makedirs(SORTIE, exist_ok=True)
IDENTITE = sitk.Transform(3, sitk.sitkIdentity)


def lire(chemin):
    return sitk.Cast(sitk.ReadImage(chemin), sitk.sitkFloat32)


def rigide_vers(fixe, mobile):
    """Transformation rigide qui recale `mobile` sur `fixe` (corrélation, 3 niveaux)."""
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


def sur_grille(fixe, chemin):
    mobile = lire(chemin)
    return sitk.GetArrayFromImage(sitk.Resample(mobile, fixe, rigide_vers(fixe, mobile), sitk.sitkLinear, 0.0)).astype(np.float64)


def synthetique():
    atlas = lire("data/atlas/gholipour/STA31.nii.gz")
    verite = sitk.GetArrayFromImage(atlas).astype(np.float64)
    conditions = [
        ("indépendant\n±4°/±4 mm", "STA31_mvt4_b5", "pyrecon_mvt4", "ours_loop_baseline.nii.gz"),
        ("lisse avec sauts\n±4°/±4 mm", "STA31_lisse_mvt4_b5", "pyrecon_lisse_mvt4", "ours_loop_baseline.nii.gz"),
        ("indépendant\n±6°/±6 mm", "STA31_mvt6_b5", "pyrecon_mvt6", "ours_loop_baseline.nii.gz"),
    ]
    colonnes = [("Sans correction", "ours_no_correction.nii.gz"), ("medoxide (boucle)", None), ("SVRTK (défaut)", "svrtk.nii.gz"), ("Vraies poses (borne haute)", "ours_true_poses.nii.gz")]
    z = verite.shape[0] // 2  # tableau numpy (z, y, x) : coupe axiale du milieu
    fig, axes = plt.subplots(len(conditions), 1 + len(colonnes), figsize=(3.1 * (1 + len(colonnes)), 3.2 * len(conditions)))
    vmax = np.percentile(verite[verite > 150], 99.5)
    for ligne, (titre, jeu, res, boucle) in enumerate(conditions):
        dossier = f"data/atlas/results/{res}"
        stacks = [lire(f"data/atlas/sim_pyrecon/{jeu}/stack{n}.nii.gz") for n in range(3)]
        dans = np.ones(verite.shape, bool)
        for s in stacks:
            dans &= sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(s * 0 + 1, sitk.sitkUInt8), atlas, IDENTITE, sitk.sitkNearestNeighbor, 0)) > 0
        images = {}
        for nom, fichier in colonnes:
            images[nom] = sur_grille(atlas, f"{dossier}/{boucle if fichier is None else fichier}")
        commun = (verite > 150) & dans
        for im in images.values():
            commun &= im != 0
        plage = np.percentile(verite[commun], 99)
        axes[ligne, 0].imshow(verite[z], cmap="gray", vmin=0, vmax=vmax)
        axes[ligne, 0].set_title("Vérité (atlas)", fontsize=10)
        axes[ligne, 0].set_ylabel(titre, fontsize=10)
        for col, (nom, _) in enumerate(colonnes, start=1):
            im = images[nom]
            psnr = 20 * np.log10(plage / np.sqrt(np.mean((im[commun] - verite[commun]) ** 2)))
            ncc = np.corrcoef(im[commun], verite[commun])[0, 1]
            axes[ligne, col].imshow(im[z], cmap="gray", vmin=0, vmax=vmax)
            axes[ligne, col].set_title(f"{nom}\nPSNR {psnr:.1f} dB, NCC {ncc:.3f}", fontsize=10)
            print(f"{titre.replace(chr(10), ' '):28} | {nom:28} | NCC {ncc:.3f} / PSNR {psnr:.2f} dB")
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Jeux synthétiques (atlas STA31, 3 stacks simulés par pyrecon, bruit 5 %) : coupe axiale, reconstructions calées sur l'atlas", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{SORTIE}/figure_synthetique.png", dpi=130)
    print("écrit", f"{SORTIE}/figure_synthetique.png")


def deplacements_par_coupe(sujet, racine, plans, poses_tsv):
    """Déplacement (mm) du centre de chaque coupe sous la pose estimée par medoxide, par stack."""
    poses = {}
    for ligne in open(poses_tsv):
        c = ligne.split("\t")
        poses[(int(c[0]), int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    ses = sorted(d for d in os.listdir(f"{racine}/{sujet}") if d.startswith("ses-"))[0]
    sorties = []
    for n, plan in enumerate(plans):
        img = nib.load(f"{racine}/{sujet}/{ses}/anat/{sujet}_{ses}_acq-{plan}_run-1_T2w.nii.gz")
        nx, ny, nz = img.shape[:3]
        d = []
        for k in range(nz):
            if (n, k) not in poses:
                d.append(np.nan)
                continue
            centre = img.affine @ np.array([(nx - 1) / 2, (ny - 1) / 2, k, 1.0])
            deplacement = np.linalg.norm((poses[(n, k)] @ centre - centre)[:3])
            d.append(deplacement if deplacement > 1e-9 else np.nan)  # pose identité exacte = coupe non recalée (masque trop petit) : pas d'estimation, on ne la trace pas
        sorties.append(np.array(d))
    return sorties


def reel(sujet, racine):
    dossier = f"data/svr/results/{sujet}"
    ours = lire(f"{dossier}/ours_loop.nii.gz")
    sources = [("Sans correction (medoxide, poses d'en-tête)", f"{dossier}/ours_header.nii.gz"), ("medoxide (boucle, 6 cycles)", f"{dossier}/ours_loop.nii.gz"), ("SVRTK (défaut)", f"{dossier}/svrtk.nii.gz")]
    images = []
    for nom, chemin in sources:
        images.append(sitk.GetArrayFromImage(ours) if chemin.endswith("ours_loop.nii.gz") else sur_grille(ours, chemin))
    ref = images[1]
    masque = ref > 0
    coupes = np.argwhere(masque).mean(axis=0).round().astype(int)  # centre du volume reconstruit (z, y, x)
    fig = plt.figure(figsize=(3.4 * 3, 3.4 * 3 + 2.6))
    grille = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8])
    vues = [("axiale", lambda im: im[coupes[0]]), ("coronale", lambda im: im[:, coupes[1], :]), ("sagittale", lambda im: im[:, :, coupes[2]])]
    for ligne, (nom_vue, extraire) in enumerate(vues):
        for col, ((nom, _), im) in enumerate(zip(sources, images)):
            ax = fig.add_subplot(grille[ligne, col])
            coupe = extraire(im)
            haut = np.percentile(im[masque], 99.5)
            ax.imshow(coupe, cmap="gray", vmin=0, vmax=haut, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if ligne == 0:
                ax.set_title(nom, fontsize=10)
            if col == 0:
                ax.set_ylabel(nom_vue, fontsize=10)
    ax = fig.add_subplot(grille[3, :])
    plans = ["trufiax", "truficor", "trufisag"]
    try:
        for n, d in enumerate(deplacements_par_coupe(sujet, racine, plans, f"{dossier}/poses_loop.tsv")):
            ax.plot(d, marker="o", ms=3, label=["axial", "coronal", "sagittal"][n])
        ax.set_xlabel("indice de coupe dans le stack")
        ax.set_ylabel("déplacement estimé du centre (mm)")
        ax.set_title("Mouvement estimé par medoxide (déplacement du centre de chaque coupe ; coupes non recalées omises)", fontsize=10)
        ax.legend(fontsize=8)
    except FileNotFoundError:
        ax.axis("off")
    fig.suptitle("Sujet réel (3 stacks TRUFI de 0,81 × 0,81 × 3,5 mm) : coupes orthogonales du centre, reconstructions calées sur celle de medoxide", fontsize=11)
    fig.tight_layout()
    chemin = f"{SORTIE}/figure_reel_{sujet}.png"
    fig.savefig(chemin, dpi=130)
    print("écrit", chemin)


parser = argparse.ArgumentParser()
parser.add_argument("quoi", choices=["synthetique", "reel"])
parser.add_argument("--sujet")
parser.add_argument("--racine")
args = parser.parse_args()
if args.quoi == "synthetique":
    synthetique()
else:
    reel(args.sujet, args.racine)
