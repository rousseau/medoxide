"""Recalage rigide de la référence s2s d'un sujet sur l'atlas STA31 avec une recherche DENSE d'orientations (24 rotations cubiques + 1200 rotations tirées uniformément).
Étapes : métrique évaluée sans optimisation à chaque départ (grossier) ; les 40 meilleurs sont optimisés (shrink 4) ; les 6 meilleurs raffinés (shrink 2) ; volumes écrits dans l'atlas (cand_j.nii.gz) ; corrélations listées.
Un volume de reconstruction n'a pas l'orientation de l'atlas, et BOUNTI exige un volume orienté comme l'atlas : cette étape trouve la rotation rigide qui l'y amène. L'initialisation envoie le centre de l'atlas sur le centre de masse du masque (T(p) = R(p - centre_atlas) + centre_masse).
L'atlas est STA31 (CRL), data/atlas/gholipour/STA31.nii.gz ou la variable MEDOXIDE_ATLAS. La symétrie gauche-droite d'une segmentation ne prouve PAS que l'orientation est bonne : contrôle anatomique à appliquer ensuite (cervelet en arrière et en bas du cerveau, tronc cérébral en bas).

Usage : python scripts/atlas_orientation_dense.py <volume> <masque cérébral> <dossier de sortie>"""
import itertools
import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation
vol_path, mask_path, sortie = sys.argv[1:4]
atlas = sitk.Cast(sitk.ReadImage(os.environ.get("MEDOXIDE_ATLAS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "atlas", "gholipour", "STA31.nii.gz"))), sitk.sitkFloat32)
ref_full = sitk.Cast(sitk.ReadImage(vol_path), sitk.sitkFloat32)
mask = sitk.Cast(sitk.ReadImage(mask_path) > 0, sitk.sitkUInt8)
ref = sitk.Mask(ref_full, mask)
centre_atlas = np.array(atlas.TransformContinuousIndexToPhysicalPoint([s / 2 for s in atlas.GetSize()]))
idx = np.mean(np.nonzero(sitk.GetArrayFromImage(mask))[::-1], axis=1)
com = np.array(mask.TransformContinuousIndexToPhysicalPoint([float(x) for x in idx]))

def metrique(shrink, it, tx, optimise=True):
    r = sitk.ImageRegistrationMethod(); r.SetMetricAsCorrelation(); r.SetMetricSamplingStrategy(r.NONE); r.SetInterpolator(sitk.sitkLinear)
    if optimise:
        r.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, it, relaxationFactor=0.6, maximumStepSizeInPhysicalUnits=4.0)
    else:
        r.SetOptimizerAsGradientDescent(learningRate=0.0, numberOfIterations=1)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([shrink]); r.SetSmoothingSigmasPerLevel([shrink / 2]); r.SetInitialTransform(tx, inPlace=True)
    r.Execute(atlas, ref)
    return -r.GetMetricValue()

mats = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product([1, -1], repeat=3):
        m = np.zeros((3, 3))
        for i, (p, s) in enumerate(zip(perm, signs)): m[i, p] = s
        if np.isclose(np.linalg.det(m), 1.0): mats.append(m)
mats += list(Rotation.random(1200, random_state=3).as_matrix())
def depart(rot):
    tx = sitk.Euler3DTransform(); tx.SetMatrix(np.asarray(rot).flatten().tolist()); tx.SetCenter(tuple(centre_atlas)); tx.SetTranslation(tuple(com - centre_atlas)); return tx
grossier = []
for k, rot in enumerate(mats):
    tx = depart(rot)
    grossier.append((metrique(4, 1, tx, optimise=False), k))
grossier.sort(reverse=True)
print("meilleurs départs (corr, k):", [(round(c, 3), k) for c, k in grossier[:5]], flush=True)
opt = []
for c, k in grossier[:40]:
    tx = depart(mats[k]); v = metrique(4, 60, tx); opt.append((v, k, sitk.Euler3DTransform(tx)))
opt.sort(key=lambda o: -o[0])
fin = []
for v, k, tx in opt[:6]:
    v2 = metrique(2, 80, tx); fin.append((v2, k, tx))
fin.sort(key=lambda o: -o[0])
for j, (v, k, tx) in enumerate(fin):
    sitk.WriteImage(sitk.Resample(ref_full, atlas, tx, sitk.sitkLinear, 0.0), f"{sortie}/cand_{j}.nii.gz")
    sitk.WriteTransform(tx, f"{sortie}/cand_{j}.tfm")
    print("cand", j, "corr", round(v, 4), "départ", k, flush=True)
