"""Met un volume sur l'atlas avec le MÊME alignement que la référence du sujet, après l'avoir recalé rigidement sur elle (le repère de sortie d'une reconstruction n'est défini qu'à un mouvement rigide près).

<dossier atlas du sujet> contient s2s.nii.gz (la référence, déjà dans l'atlas) et ref_vers_atlas.tfm (sa transformation), produits par atlas_orientation_dense.py. Départs identité et moments d'inertie, puis raffinement à pleine résolution (sans lui, des recalages restent dans un optimum local et faussent la reproductibilité mesurée entre repliements). La corrélation finale est affichée : sous 0,9, l'alignement est suspect.

Usage : python scripts/atlas_aligne_sur_reference.py <dossier atlas du sujet> <volume> <sortie>"""
import os
import sys

import numpy as np
import SimpleITK as sitk
dossier, vol, sortie = sys.argv[1:4]
atlas = sitk.Cast(sitk.ReadImage(os.environ.get("MEDOXIDE_ATLAS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "atlas", "gholipour", "STA31.nii.gz"))), sitk.sitkFloat32)
t_atlas = sitk.Euler3DTransform(sitk.ReadTransform(f"{dossier}/ref_vers_atlas.tfm"))
ref_full = sitk.Cast(sitk.ReadImage(f"{dossier}/s2s.nii.gz"), sitk.sitkFloat32)  # déjà dans l'atlas
img = sitk.Cast(sitk.ReadImage(vol), sitk.sitkFloat32)
# volume dans l'atlas avec t_atlas seul, puis recalage rigide sur la référence (dans l'atlas)
dans_atlas = sitk.Resample(img, atlas, t_atlas, sitk.sitkLinear, 0.0)
best = None
for init in ("identite", "moments"):
    t = sitk.Euler3DTransform() if init == "identite" else sitk.Euler3DTransform(sitk.CenteredTransformInitializer(ref_full, dans_atlas, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS))
    r = sitk.ImageRegistrationMethod(); r.SetMetricAsCorrelation(); r.SetMetricSamplingStrategy(r.NONE); r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 100, relaxationFactor=0.6); r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([2]); r.SetSmoothingSigmasPerLevel([1.0]); r.SetInitialTransform(t, inPlace=True)
    r.Execute(ref_full, dans_atlas)
    if best is None or r.GetMetricValue() < best[0]:
        best = (r.GetMetricValue(), sitk.Euler3DTransform(t))
    if -r.GetMetricValue() > 0.9:
        break
# raffinement final à pleine résolution
t = best[1]
r = sitk.ImageRegistrationMethod(); r.SetMetricAsCorrelation(); r.SetMetricSamplingStrategy(r.NONE); r.SetInterpolator(sitk.sitkLinear)
r.SetOptimizerAsRegularStepGradientDescent(0.5, 1e-5, 150, relaxationFactor=0.6); r.SetOptimizerScalesFromPhysicalShift()
r.SetShrinkFactorsPerLevel([1]); r.SetSmoothingSigmasPerLevel([0.5]); r.SetInitialTransform(t, inPlace=True); r.Execute(ref_full, dans_atlas)
if r.GetMetricValue() < best[0]:
    best = (r.GetMetricValue(), sitk.Euler3DTransform(t))
print("corrélation avec la référence", round(-best[0], 3))
sitk.WriteImage(sitk.Resample(dans_atlas, atlas, best[1], sitk.sitkLinear, 0.0), sortie)
