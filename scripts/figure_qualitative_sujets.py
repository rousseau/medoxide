"""Figure qualitative : les reconstructions de chaque méthode, côte à côte, pour plusieurs sujets réels.

Pour chaque sujet, au repli `--repli` (un stack mis de côté, reconstruction avec les deux autres : les volumes sont ceux que le juge a évalués), les six volumes : en-tête (poses de l'en-tête, aucune correction),
SVRTK (défauts), NeSVoR (défauts, sans SVoRT, CPU), NiftyMIC (défauts sauf masques, `run_niftymic.py`), notre coupe-volume et notre coupe-coupe, ces deux derniers aux défauts figés de F2.4 (binaire 0d1b10c). Chaque volume est recalé rigidement (SimpleITK, corrélation) sur la reconstruction coupe-coupe du même sujet (le repère de sortie
d'une reconstruction n'est défini qu'à un mouvement rigide près), puis on affiche deux coupes passant par le centre du volume reconstruit, perpendiculaires aux axes z et y de la grille de reconstruction (repère du scanner : ce ne sont pas des plans anatomiques) (même emplacement pour toutes les méthodes,
choisi sans regarder les images), avec la même fenêtre d'intensité (0 à 99,5e centile) pour une méthode donnée. Chaque cellule indique la NCC moyenne du juge de ce repli. Le repli par défaut (1) est fixé
avant d'examiner les images. Les identifiants de sujets sont passés en arguments (aucun dans le code). Données de patientes : la figure est écrite dans docs/svr/figures/ (ignoré par Git), à ne pas publier.

Fichiers lus dans data/svr/results/<sujet>/ : f21/entete_h<h>, svrtk_sans<h>, nesvor_sans<h>, niftymic_sans<h>, fig/s2v_h<h>, f21/sc1_h<h> (chacun `.nii.gz` et `.nii.gz.juge.tsv`).
Mise à jour 2026-11-02 : `fig/s2v_h1` (6 sujets de développement) vient de la mesure F3.1 (`f31/juge/ncc_h1`, binaire de 12f7d86, défauts actuels, départ `stacks-bias`) ; les volumes d'avant (binaire 0d1b10c, dont le défaut CLI était `--s2v-init header` ; les options passées n'étaient pas enregistrées) sont dans `fig/archive_avant_f31/`. `f21/sc1_h1` est inchangé : recalcul aux défauts actuels (un sujet calme, repli 1) identique bit à bit.

Colonne « variante » (facultative) : `--variante MODELE --nom-variante "α = 0,03"` ajoute à droite des défauts le volume d'un réglage en essai. MODELE est un chemin sans extension, avec `{sujet}` et `{h}`
(par exemple `data/svr/variantes/essai/{sujet}/d003_h{h}`, ce que produit `benchmark_variantes_s2s.py --sortie data/svr/variantes/essai/<sujet>`) ; le volume et son `.nii.gz.juge.tsv` doivent exister pour chaque sujet.
Il est recalé comme les autres sur la coupe-coupe par défaut.

Tableau : à chaque figure, `<sortie sans .png>_tableau.tsv` (et l'écran) donne, par sujet et par méthode, le juge (NCC moyenne du repli) et la netteté S (gradient moyen du volume lissé d'un voxel, dans le masque
érodé, divisé par l'intensité médiane du tissu : même définition que `benchmark_variantes_s2s.py`), avec la moyenne sur les sujets. Le juge favorise le lissage : toujours lire les deux ensemble.

Liste des sujets : soit `--sujets ... --etiquettes ...`, soit `--sujets-fichier` (TSV `sujet<TAB>etiquette`, une ligne par sujet, `#` pour un commentaire : celui du dépôt privé oxide, `svr/sujets.tsv`).
`--lister` écrit les chemins (relatifs à `--racine-resultats`) que la figure lira, un par ligne, puis s'arrête : `scripts/recuperer_figure_svr.sh` s'en sert pour les rapatrier depuis Garage.

Usage : python scripts/figure_qualitative_sujets.py --sujets-fichier ~/Exp/oxide/svr/sujets.tsv [--repli 1] [--variante MODELE --nom-variante NOM]
        python scripts/figure_qualitative_sujets.py --sujets sub-A sub-B ... --etiquettes "calme 1" ... [--repli 1]
"""

import argparse
import csv

import os

parser = argparse.ArgumentParser()
parser.add_argument("--sujets", nargs="+")
parser.add_argument("--etiquettes", nargs="+")
parser.add_argument("--sujets-fichier", help="TSV `sujet<TAB>etiquette` (une ligne par sujet, `#` : commentaire), à la place de --sujets et --etiquettes")
parser.add_argument("--repli", type=int, default=1)
parser.add_argument("--racine-resultats", default="data/svr/results", help="dossier des résultats : <racine>/<sujet>/<volume>.nii.gz")
parser.add_argument("--variante", help="modèle de chemin (sans extension, avec {sujet} et {h}) du volume d'un réglage en essai : ajoute une colonne")
parser.add_argument("--nom-variante", default="variante")
parser.add_argument("--sortie", default="docs/svr/figures/qualitatif_sujets.png")
parser.add_argument("--lister", action="store_true", help="écrit les fichiers lus (relatifs à --racine-resultats), puis s'arrête")
args = parser.parse_args()
if args.sujets_fichier:
    assert not args.sujets and not args.etiquettes, "--sujets-fichier exclut --sujets et --etiquettes"
    lignes = [l.rstrip("\n").split("\t") for l in open(args.sujets_fichier) if l.strip() and not l.lstrip().startswith("#")]
    assert all(len(l) == 2 for l in lignes), f"{args.sujets_fichier} : chaque ligne doit être `sujet<TAB>etiquette`"
    args.sujets, args.etiquettes = [l[0] for l in lignes], [l[1] for l in lignes]
assert args.sujets and args.etiquettes and len(args.sujets) == len(args.etiquettes), "il faut autant de sujets que d'étiquettes"
H = args.repli
# nos méthodes aux défauts figés (α = 0,05, F2.4) ; SVRTK, NeSVoR et NiftyMIC à leurs défauts ; chemins relatifs à data/svr/results/<sujet>/ (sans extension)
METHODES = [
    ("en-tête", "f21/entete_h{h}"),
    ("SVRTK (défauts)", "svrtk_sans{h}"),
    ("NeSVoR (défauts)", "nesvor_sans{h}"),
    ("NiftyMIC (défauts)", "niftymic_sans{h}"),
    ("notre coupe-volume", "fig/s2v_h{h}"),
    ("notre coupe-coupe", "f21/sc1_h{h}"),
]
REFERENCE = "f21/sc1_h{h}"
INDICE_REFERENCE = [m for _, m in METHODES].index(REFERENCE)
if args.variante:
    # la variante n'est pas rangée sous <racine-resultats>/<sujet>/ : son modèle de chemin est complet ; on le distingue par un préfixe « @ »
    METHODES.append((args.nom_variante, "@" + args.variante))


def chemin(sujet, modele):
    """Chemin (sans extension) du volume d'une méthode pour un sujet, au repli choisi."""
    if modele.startswith("@"):
        return modele[1:].format(sujet=sujet, h=H)
    return f"{args.racine_resultats}/{sujet}/{modele.format(h=H)}"


if args.lister:
    for sujet in args.sujets:
        for _, modele in METHODES:
            if not modele.startswith("@"):
                for ext in (".nii.gz", ".nii.gz.juge.tsv"):
                    print(f"{sujet}/{modele.format(h=H)}{ext}")
    raise SystemExit(0)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, gaussian_filter


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
    v = [float(r["ncc"]) for r in csv.DictReader(open(f"{chemin(sujet, modele)}.nii.gz.juge.tsv"), delimiter="\t")]
    return float(np.mean(v))


def nettete(fichier):
    """Netteté S, définie comme dans benchmark_variantes_s2s.py (calculée sur le volume tel qu'écrit, avant tout recalage)."""
    v = np.asarray(nib.load(fichier).dataobj, dtype=np.float64)
    tissu = v > 0.05 * np.percentile(v[v > 0], 99.5)
    interieur = binary_erosion(tissu, iterations=3)
    g = np.sqrt(sum(np.gradient(gaussian_filter(v, 1.0), axis=a) ** 2 for a in range(3)))
    return float(g[interieur].mean() / np.median(v[interieur]))


def volumes(sujet):
    """Les volumes du sujet (une entrée par méthode) sur la grille de la reconstruction coupe-coupe."""
    fixe = lire(f"{chemin(sujet, REFERENCE)}.nii.gz")
    out = []
    for nom, modele in METHODES:
        if modele == REFERENCE:
            out.append(sitk.GetArrayFromImage(fixe).astype(np.float64))
        else:
            mobile = lire(f"{chemin(sujet, modele)}.nii.gz")
            out.append(sitk.GetArrayFromImage(sitk.Resample(mobile, fixe, rigide_vers(fixe, mobile), sitk.sitkLinear, 0.0)).astype(np.float64))
    return out


# plans de la GRILLE de reconstruction (repère du scanner), pas des plans anatomiques : la tête du fœtus a une orientation quelconque
VUES = [("plan z de la grille", lambda im, c: im[c[0]], (1, 2)), ("plan y de la grille", lambda im, c: im[:, c[1], :], (0, 2))]
n = len(args.sujets)
resultats = []  # (sujet, étiquette, méthode, juge, netteté)
fig, axes = plt.subplots(n, 2 * len(METHODES), figsize=(1.45 * 2 * len(METHODES), 2.0 * n + 0.9))
for ligne, (sujet, etiquette) in enumerate(zip(args.sujets, args.etiquettes)):
    print("sujet", etiquette, flush=True)
    ims = volumes(sujet)
    for nom, modele in METHODES:
        resultats.append((sujet, etiquette, nom, ncc_juge(sujet, modele), nettete(f"{chemin(sujet, modele)}.nii.gz")))
    ref = ims[INDICE_REFERENCE]
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
fig.suptitle(f"Reconstructions à 2 stacks, repli {H} (un stack mis de côté), nos méthodes aux défauts figés (α = 0,05), couverture 1 (volumes jugés ; les « languettes » visibles sont supprimées par défaut avec --min-stack-coverage 2)" + (f" ; dernière colonne : variante en essai ({args.nom_variante})" if args.variante else "") + " ; NCC = prédiction du stack mis de côté (le juge favorise le lissage)", fontsize=8.2, y=0.012, va="bottom")
fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=0.4, w_pad=0.25)
os.makedirs(os.path.dirname(os.path.abspath(args.sortie)), exist_ok=True)
fig.savefig(args.sortie, dpi=110)
print("écrit :", args.sortie)

# tableau du juge et de la netteté S, par sujet et par méthode, avec la moyenne sur les sujets
noms = [nom for nom, _ in METHODES]
cases = {(etiquette, nom): (juge, S) for _, etiquette, nom, juge, S in resultats}
fichier_tableau = os.path.splitext(args.sortie)[0] + "_tableau.tsv"
lignes = ["sujet\tmethode\tjuge_ncc\tnettete_S"] + [f"{e}\t{nom}\t{cases[(e, nom)][0]:.4f}\t{cases[(e, nom)][1]:.4f}" for e in args.etiquettes for nom in noms]
lignes += [f"moyenne\t{nom}\t{np.mean([cases[(e, nom)][0] for e in args.etiquettes]):.4f}\t{np.mean([cases[(e, nom)][1] for e in args.etiquettes]):.4f}" for nom in noms]
open(fichier_tableau, "w").write("\n".join(lignes) + "\n")
largeur = max(len(nom) for nom in noms) + 2
for indice, titre in ((0, f"Juge (NCC moyenne du repli {H}, plus haut = mieux)"), (1, "Netteté S (plus haut = plus net)")):
    print(f"\n{titre}\n{'':<10}" + "".join(f"{nom:>{largeur}}" for nom in noms))
    for e in [*args.etiquettes, "moyenne"]:
        valeurs = [np.mean([cases[(x, nom)][indice] for x in args.etiquettes]) if e == "moyenne" else cases[(e, nom)][indice] for nom in noms]
        print(f"{e:<10}" + "".join(f"{v:>{largeur}.3f}" for v in valeurs))
print("écrit :", fichier_tableau)
