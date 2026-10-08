"""Compare des reconstructions 3D (medoxide, SVRTK, ...) à l'atlas HR dont elles ont été simulées, sur un ensemble de voxels COMMUN à toutes les méthodes.

Chaque reconstruction (NIfTI, n'importe quelle grille) est rééchantillonnée (SimpleITK, trilinéaire) sur la grille de l'atlas. L'ensemble commun est
l'intersection de : voxels de tissu de l'atlas (> 150), lisibles dans chacun des stacks (champ de vue), et où chaque reconstruction est non nulle (SVRTK
met à 0 hors de son masque, medoxide hors de son domaine). Même définition que les tests de medoxide (`construire_comparaison`), appliquée à l'atlas
natif plutôt qu'à la grille de reconstruction : les chiffres de medoxide doivent concorder à quelques centièmes (recoupement à faire).

Mesures : NCC ; PSNR brut (dynamique = 99e centile de l'atlas sur l'ensemble) ; PSNR après ajustement affine d'intensité y ≈ a·x + b (SVRTK normalise ses
intensités : le PSNR brut dépend de cette échelle, la NCC non). Avec --align, chaque reconstruction est d'abord recalée rigidement sur l'atlas
(SimpleITK, corrélation) : sépare l'erreur de repère de l'erreur d'image.

Usage : python scripts/compare_reconstructions.py --atlas data/atlas/gholipour/STA31.nii.gz --stacks s0.nii.gz s1.nii.gz s2.nii.gz \
          --recon medoxide=ours_loop.nii.gz --recon svrtk=svrtk.nii.gz [--align]
"""

import argparse

import numpy as np
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument("--atlas", required=True)
parser.add_argument("--stacks", nargs="+", required=True, help="les stacks d'entrée (leur champ de vue limite l'ensemble commun)")
parser.add_argument("--recon", action="append", required=True, help="nom=chemin.nii.gz (répétable)")
parser.add_argument("--align", action="store_true", help="recaler aussi chaque reconstruction sur l'atlas (rigide, corrélation)")
parser.add_argument("--interieur-mm", type=float, default=0.0, help="ne garder que les voxels à plus de cette distance (mm) du bord du tissu de l'atlas : écarte les effets de masque et de bord (choisi après coup, à rapporter comme tel)")
parser.add_argument("--grille-ref", help="évaluer sur la grille de cette image (au lieu de celle de l'atlas) : l'atlas est alors rééchantillonné sur elle")
parser.add_argument("--ensemble", help="NIfTI 0/1 (sur la grille de --grille-ref) : ensemble de voxels imposé (recoupement avec medoxide : comparison_set.nii.gz)")
args = parser.parse_args()

atlas_natif = sitk.Cast(sitk.ReadImage(args.atlas), sitk.sitkFloat32)
identite = sitk.Transform(3, sitk.sitkIdentity)
# grille d'évaluation : celle de l'atlas par défaut ; sinon celle de --grille-ref, où l'atlas est rééchantillonné (trilinéaire)
atlas = atlas_natif if not args.grille_ref else sitk.Resample(atlas_natif, sitk.ReadImage(args.grille_ref), identite, sitk.sitkLinear, 0.0)
verite = sitk.GetArrayFromImage(atlas).astype(np.float64)

# champ de vue de chaque stack, sur la grille de l'atlas
dans_stacks = np.ones(verite.shape, bool)
for chemin in args.stacks:
    s = sitk.ReadImage(chemin)
    uns = sitk.Cast(s * 0 + 1, sitk.sitkUInt8)
    dans_stacks &= sitk.GetArrayFromImage(sitk.Resample(uns, atlas, identite, sitk.sitkNearestNeighbor, 0)) > 0


def rigide_vers_atlas(mobile):
    """Transformation rigide (6 paramètres) qui recale `mobile` sur l'atlas, par corrélation."""
    fixe = atlas
    init = sitk.CenteredTransformInitializer(fixe, sitk.Cast(mobile, sitk.sitkFloat32), sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
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
    return r.Execute(fixe, sitk.Cast(mobile, sitk.sitkFloat32))


def sur_grille_atlas(chemin, aligner):
    mobile = sitk.Cast(sitk.ReadImage(chemin), sitk.sitkFloat32)
    tx = rigide_vers_atlas(mobile) if aligner else identite
    return sitk.GetArrayFromImage(sitk.Resample(mobile, atlas, tx, sitk.sitkLinear, 0.0)).astype(np.float64), tx


def mesures(x, y):
    ncc = float(np.corrcoef(x, y)[0, 1])
    plage = float(np.percentile(y, 99))
    psnr = lambda v: 20 * np.log10(plage / np.sqrt(np.mean((v - y) ** 2)))  # noqa: E731
    a, b = np.polyfit(x, y, 1)
    return ncc, psnr(x), psnr(a * x + b), a, b


variantes = [False, True] if args.align else [False]
for aligner in variantes:
    images = {}
    for entree in args.recon:
        nom, chemin = entree.split("=", 1)
        images[nom], tx = sur_grille_atlas(chemin, aligner)
        if aligner:
            p = tx.GetParameters() if hasattr(tx, "GetParameters") else ()
            print(f"  [{nom}] transformation rigide trouvée (rotations rad, translations mm) : {np.round(p, 3)}")
    if args.ensemble:
        commun = sitk.GetArrayFromImage(sitk.ReadImage(args.ensemble)) > 0.5
    else:
        commun = (verite > 150) & dans_stacks
        for im in images.values():
            commun &= im != 0
    if args.interieur_mm > 0:
        from scipy import ndimage

        pas = np.array(atlas.GetSpacing())[::-1]  # tableau numpy : (z, y, x)
        commun &= ndimage.distance_transform_edt(verite > 150, sampling=pas) > args.interieur_mm
    y = verite[commun]
    print(f"\n{'après calage rigide sur l atlas' if aligner else 'sans calage (repère de sortie tel quel)'}{f', voxels à plus de {args.interieur_mm:g} mm du bord' if args.interieur_mm > 0 else ''} : {int(commun.sum())} voxels communs (plage {np.percentile(y, 99):.0f})")
    print(f"  {'méthode':28} {'NCC':>7} {'PSNR brut':>10} {'PSNR après gain':>16}   (gain a, décalage b)")
    for nom, im in images.items():
        ncc, p_brut, p_gain, a, b = mesures(im[commun], y)
        print(f"  {nom:28} {ncc:7.4f} {p_brut:9.2f}  {p_gain:15.2f}   (a = {a:.3f}, b = {b:.1f})")
