"""Porte « oracle » de la pondération des coupes (docs/RECHERCHE_MULTISTART.md § 24) : compare, à l'atlas qui a servi à simuler les piles, la reconstruction de toutes les coupes (`base`) à celle qui exclut les coupes dont l'erreur de pose VRAIE dépasse 1,5 mm (`oracle`).

Entrée : `<dossier>/CRL<âge>/<modèle>_<amplitude>_<graine>_{base,oracle}.nii.gz`, écrits par `diagnose_the_semi_synthetic_benchmark` (MEDOXIDE_ATLAS=CRL<âge>, MEDOXIDE_ORACLE_SORTIE), et l'atlas centré `data/atlas/gholipour/CRL<âge>.nii.gz` (scripts/preparer_atlas_crl.py).

Mesures (§ 24) : le volume `base` est recalé rigidement sur l'atlas (SimpleITK, corrélation ; l'oracle a les mêmes poses, donc le même repère : on lui applique la même transformation), puis
`E = RMSE(x, atlas) / RMS(atlas)` dans le domaine reconstruit commun aux deux variantes ; netteté S de `benchmark_variantes_s2s.py`. Gain `G = (E_base − E_oracle) / E_base`, moyenné sur les graines puis sur les
scénarios pour chaque âge. Critères, écrits avant la mesure : (a) G moyen sur les âges ≥ 5 % ; (b) variation moyenne de S ≥ −2 % ; (c) aucun âge avec un G moyen < −2 %. La couverture (voxels reconstruits) est affichée pour interpréter.

Usage : python scripts/analyse_oracle_ponderation.py --dossier data/svr/oracle --atlas data/atlas/gholipour --ages 23 26 29 32 35 37 --scenarios lisse:6 sauts:4 --graines 7 11 13
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("--dossier", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("--ages", nargs="+", type=int, required=True)
parser.add_argument("--scenarios", nargs="+", required=True, help="modele:amplitude, par exemple lisse:6 sauts:4")
parser.add_argument("--graines", nargs="+", type=int, required=True)
parser.add_argument("--seuil-gain", type=float, default=0.05)
args = parser.parse_args()


def lire(chemin):
    return sitk.Cast(sitk.ReadImage(chemin), sitk.sitkFloat32)


def rigide_vers(fixe, mobile):
    """Transformation rigide qui amène `mobile` sur `fixe` (même réglage que scripts/figure_qualitative_sujets.py)."""
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


def nettete(fichier):
    """Netteté S, même définition que benchmark_variantes_s2s.py."""
    v = np.asarray(nib.load(fichier).dataobj, dtype=np.float64)
    tissu = v > 0.05 * np.percentile(v[v > 0], 99.5)
    interieur = binary_erosion(tissu, iterations=3)
    g = np.sqrt(sum(np.gradient(gaussian_filter(v, 1.0), axis=a) ** 2 for a in range(3)))
    return float(g[interieur].mean() / np.median(v[interieur]))


lignes = []  # (âge, scénario, graine, E_base, E_oracle, S_base, S_oracle, couverture_base, couverture_oracle)
for ga in args.ages:
    atlas = lire(f"{args.atlas}/CRL{ga}.nii.gz")
    a = sitk.GetArrayFromImage(atlas).astype(np.float64)
    for sc in args.scenarios:
        modele, amplitude = sc.split(":")
        for graine in args.graines:
            base = f"{args.dossier}/CRL{ga}/{modele}_{amplitude}_{graine}_base.nii.gz"
            oracle = base.replace("_base.", "_oracle.")
            if not (os.path.exists(base) and os.path.exists(oracle)):
                print(f"manque : CRL{ga} {sc} graine {graine}", file=sys.stderr)
                continue
            xb, xo = lire(base), lire(oracle)
            vo = sitk.GetArrayFromImage(xo)
            if not np.isfinite(vo).all() or (vo > 0).sum() < 0.1 * (sitk.GetArrayFromImage(xb) > 0).sum():
                # reconstruction de l'oracle dégénérée (NaN ou presque vide) : écartée de la moyenne, listée à part
                print(f"DÉGÉNÉRÉ : CRL{ga} {sc} graine {graine} (oracle non fini ou < 10 % des voxels de base)", flush=True)
                continue
            t = rigide_vers(atlas, xb)
            rb = sitk.GetArrayFromImage(sitk.Resample(xb, atlas, t, sitk.sitkLinear, 0.0)).astype(np.float64)
            ro = sitk.GetArrayFromImage(sitk.Resample(xo, atlas, t, sitk.sitkLinear, 0.0)).astype(np.float64)
            domaine = (rb > 0) & (ro > 0) & (a > 0)
            rms = np.sqrt(np.mean(a[domaine] ** 2))
            e = lambda r: float(np.sqrt(np.mean((r[domaine] - a[domaine]) ** 2)) / rms)
            lignes.append((ga, sc, graine, e(rb), e(ro), nettete(base), nettete(oracle), float((rb > 0).sum()), float((ro > 0).sum())))
            print(f"CRL{ga} {sc} graine {graine} : E base {lignes[-1][3]:.4f} oracle {lignes[-1][4]:.4f} ; S {lignes[-1][5]:.4f} / {lignes[-1][6]:.4f}", flush=True)

if not lignes:
    sys.exit("aucun résultat")
os.makedirs(args.dossier, exist_ok=True)
with open(f"{args.dossier}/oracle_resultats.tsv", "w") as f:
    f.write("age\tscenario\tgraine\tE_base\tE_oracle\tS_base\tS_oracle\tvoxels_base\tvoxels_oracle\n")
    for l in lignes:
        f.write("\t".join(str(x) for x in l) + "\n")

print("\nPar âge (moyenne des graines puis des scénarios) :")
print(f"{'âge':>4} {'E base':>8} {'E oracle':>9} {'gain G':>8} {'ΔS':>7} {'couverture oracle/base':>24}")
gains, dS = {}, []
for ga in args.ages:
    par_sc = []
    for sc in args.scenarios:
        l = [x for x in lignes if x[0] == ga and x[1] == sc]
        if l:
            par_sc.append((np.mean([(x[3] - x[4]) / x[3] for x in l]), np.mean([x[6] / x[5] - 1 for x in l]), np.mean([x[3] for x in l]), np.mean([x[4] for x in l]), np.mean([x[8] / x[7] for x in l])))
    if par_sc:
        g, s, eb, eo, cov = (float(np.mean([p[i] for p in par_sc])) for i in range(5))
        gains[ga] = g
        dS.append(s)
        print(f"{ga:>4} {eb:>8.4f} {eo:>9.4f} {100 * g:>7.1f}% {100 * s:>6.1f}% {cov:>24.2f}")
G = float(np.mean(list(gains.values())))
S = float(np.mean(dS))
pire = min(gains.values())
print(f"\nG moyen {100 * G:.1f} % (critère a : ≥ {100 * args.seuil_gain:.0f} %)  → {'tenu' if G >= args.seuil_gain else 'NON tenu'}")
print(f"ΔS moyen {100 * S:.1f} % (critère b : ≥ −2 %)  → {'tenu' if S >= -0.02 else 'NON tenu'}")
print(f"pire âge {100 * pire:.1f} % (critère c : ≥ −2 %)  → {'tenu' if pire >= -0.02 else 'NON tenu'}")
ok = G >= args.seuil_gain and S >= -0.02 and pire >= -0.02
print("\nVERDICT : l'oracle " + ("PASSE : la piste est poursuivie." if ok else "ÉCHOUE : la pondération des coupes est abandonnée (§ 24)."))
