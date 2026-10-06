"""Références de l'opérateur d'acquisition coupe <- volume, calculées avec NeSVoR (étape 2d du SVR).

Pour quelques paires de stacks d'un même sujet, construit un volume isotrope aligné sur les axes du monde (RAS+) à
partir du stack axial TRUFI, puis simule des coupes du stack coronal TRUFI avec `slice_acquisition_torch` de NeSVoR
(MIT, https://github.com/daviddmc/NeSVoR : le code est EXÉCUTÉ depuis un clone local, jamais copié).

Conventions de NeSVoR rétablies par lecture de son code (nesvor/slice_acquisition/slice_acq_torch.py, nesvor/image/image.py,
nesvor/transform/transform.py) :
  - volume : tenseur (D, H, W) = (z, y, x), voxels isotropes de résolution r ; unités = voxels de sortie, repère centré ;
  - coupe : pixel (i, j) à la position locale ((i - (nx-1)/2) rho, (j - (ny-1)/2) rho, 0), rho = pixel / r (isotrope) ;
  - transformation (R | t) appliquée comme R (x + t), donc position = R (local + t) ; R colonnes = axes de la coupe ;
  - PSF : get_PSF(res_ratio = (rho, rho, épaisseur / r)), grille entière en voxels, sigma = 1,2067 FWHM dans le plan,
    épaisseur hors plan ; arrondi au voxel le plus proche ; normalisation par la somme des poids (seuil 1e-2).
Pour limiter la mémoire, chaque coupe est réduite à un bloc central de BLOC x BLOC pixels.

Écrit dans data/reference/acquisition/ : <nom>_volume.nii.gz (volume isotrope, affine RAS+ diagonale), <nom>.f32 (par coupe
retenue, valeurs puis poids de NeSVoR, tableaux (BLOC, BLOC) indexés [i, j]) et index.tsv.

Usage : python scripts/make_reference_acquisition.py --nesvor ~/chemin/vers/NeSVoR
"""

import argparse
import collections
import glob
import os
import re
import sys
import time

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

RACINE = "data/svr/jeu_reel_tru_haste"
SORTIE = "data/reference/acquisition"
N_SUJETS, PAS_COUPE, BLOC, R_SORTIE, MARGE_MM = 4, 6, 160, 0.8, 8.0

parser = argparse.ArgumentParser()
parser.add_argument("--nesvor", required=True, help="clone local de NeSVoR (ajouté au chemin d'import)")
sys.path.insert(0, os.path.expanduser(parser.parse_args().nesvor))
import nesvor.slice_acquisition.slice_acq_torch as acq  # noqa: E402
from nesvor.utils.psf import get_PSF  # noqa: E402

acq.BATCH_SIZE = 1
os.makedirs(SORTIE, exist_ok=True)


def boite_monde(img):
    """Boîte englobante monde (RAS+) des centres de voxel d'une image nibabel."""
    n = np.array(img.shape[:3]) - 1
    coins = np.array([[i, j, k, 1] for i in (0, n[0]) for j in (0, n[1]) for k in (0, n[2])]).T
    pts = (img.affine @ coins)[:3]
    return pts.min(1), pts.max(1)


par_sujet = collections.defaultdict(dict)
for f in sorted(glob.glob(f"{RACINE}/sub-*/ses-*/anat/*_T2w.nii.gz")):
    m = re.search(r"(sub-\d+)_ses-\d+_acq-(trufiax|truficor)_run-1_T2w", f)
    if m:
        par_sujet[m.group(1)][m.group(2)] = f
paires = [(v["trufiax"], v["truficor"]) for _, v in sorted(par_sujet.items()) if len(v) == 2][:N_SUJETS]

index = []
for vol_path, coupes_path in paires:
    axial, coronal = nib.load(vol_path), nib.load(coupes_path)
    nom = os.path.basename(coupes_path).removesuffix("_T2w.nii.gz")
    # --- volume isotrope aligné RAS+ : région commune aux deux stacks, plus une marge pour la PSF
    lo = np.maximum(boite_monde(axial)[0], boite_monde(coronal)[0]) - MARGE_MM
    hi = np.minimum(boite_monde(axial)[1], boite_monde(coronal)[1]) + MARGE_MM
    forme = np.ceil((hi - lo) / R_SORTIE).astype(int) + 1
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in forme], indexing="ij")
    monde = np.stack([ii, jj, kk], 0).reshape(3, -1) * R_SORTIE + lo[:, None]
    idx = (np.linalg.inv(axial.affine) @ np.vstack([monde, np.ones(monde.shape[1])]))[:3]
    dedans = np.all((idx >= 0) & (idx <= np.array(axial.shape[:3])[:, None] - 1), axis=0)
    vals = ndimage.map_coordinates(axial.get_fdata(dtype=np.float32), idx, order=1, mode="nearest")
    vol = np.where(dedans, vals, 0).astype(np.float32).reshape(forme)
    affine_v = np.diag([R_SORTIE, R_SORTIE, R_SORTIE, 1.0])
    affine_v[:3, 3] = lo
    img = nib.Nifti1Image(vol, affine_v)
    img.set_sform(affine_v, code=1)
    img.set_qform(affine_v, code=1)
    chemin_vol = f"{SORTIE}/{nom}_volume.nii.gz"
    nib.save(img, chemin_vol)
    centre_v = lo + (forme - 1) / 2 * R_SORTIE  # position monde du centre du volume
    vol_t = torch.from_numpy(np.ascontiguousarray(vol.transpose(2, 1, 0)))[None, None]  # (1, 1, D, H, W)

    # --- coupes
    A = coronal.affine
    pix = np.linalg.norm(A[:3, :3], axis=0)
    assert abs(pix[0] - pix[1]) < 1e-3, "NeSVoR suppose des pixels isotropes dans le plan"
    nx, ny, nz = coronal.shape[:3]
    rho = pix[0] / R_SORTIE
    psf = get_PSF(res_ratio=(rho, rho, pix[2] / R_SORTIE), device=torch.device("cpu"), psf_type="gaussian")
    i0, j0 = (nx - BLOC) // 2, (ny - BLOC) // 2
    ks = list(range(2, nz, PAS_COUPE))
    sorties, temps = [], []
    for k in ks:
        Ak = A.copy()
        Ak[:3, 3] += k * A[:3, 2]
        R = Ak[:3, :3] / pix  # colonnes = axes unitaires de la coupe (u, v, normale)
        pos0 = (Ak @ np.array([i0, j0, 0, 1.0]))[:3]  # pixel (i0, j0), monde, mm
        pos0 = (pos0 - centre_v) / R_SORTIE  # en voxels de sortie, repère centré
        c_s = np.array([(BLOC - 1) / 2 * rho, (BLOC - 1) / 2 * rho, 0.0])
        t = np.linalg.solve(R, pos0 + R @ c_s)  # R (local + t) = pos0 + R (i rho, j rho, 0)
        transform = torch.from_numpy(np.concatenate([R, t[:, None]], 1).astype(np.float32))[None]
        debut = time.time()
        valeurs, poids = acq.slice_acquisition_torch(transform, vol_t, None, None, psf, (BLOC, BLOC), float(rho), True)
        temps.append(time.time() - debut)
        # tenseurs (1, 1, H=j, W=i) -> tableaux [i, j]
        sorties.append(np.stack([valeurs[0, 0].numpy().T, poids[0, 0].numpy().T]))
    np.stack(sorties).astype("<f4").tofile(f"{SORTIE}/{nom}.f32")
    index.append("\t".join([chemin_vol, coupes_path, ",".join(map(str, ks)), str(i0), str(j0), str(BLOC), f"{SORTIE}/{nom}.f32"]))
    print(f"{nom} : volume {tuple(forme)}, PSF {tuple(psf.shape)}, {len(ks)} coupes, {np.mean(temps):.1f} s par coupe (NeSVoR, CPU)")
open(f"{SORTIE}/index.tsv", "w").write("\n".join(index) + "\n")
