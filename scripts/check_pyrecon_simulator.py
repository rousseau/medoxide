"""Vérification indépendante du simulateur de mouvement de pyrecon (ROSI), avant de l'utiliser comme générateur de données de benchmark.

Le simulateur (`rosi.simulation.simul3Ddata.simulateMvt`) est EXÉCUTÉ depuis un clone local, jamais copié. Les critères sont fixés avant le calcul et
reposent sur des images analytiques dont la valeur attendue se calcule exactement :

  C1  rampe linéaire, sans mouvement, affine oblique, 3 orientations : chaque pixel vaut la rampe au centre du pixel (monde), à 1e-6 de la dynamique.
      (Une PSF symétrique et une interpolation spline cubique reproduisent exactement une fonction affine.)
  C2  rampe linéaire, avec mouvement : chaque pixel vaut la rampe en M_k · (centre nominal du pixel), M_k = `theorical_transformations[k]` rendue par
      le simulateur, à 1e-6 : cette matrice est bien le mouvement appliqué (sens et pivot compris).
  C3  image quadratique I = x² + y² + z² (indices HR, affine identité, sans mouvement) : l'interpolation cubique avec `prefilter=False` traite les voxels comme
      des coefficients de spline, ce qui ajoute un flou de variance 1/3 voxel² par axe (variance de la B-spline cubique), et la PSF ajoute sa variance
      discrète v_psf selon z. Valeur attendue : I(centre) + (1/2) f'' (1/3 + 1/3 + 1/3 + v_psf) avec f'' = 2 ; écart toléré 1e-6.
  C4  `rigidMatrix` : orthonormale de déterminant +1, translation = paramètres 3..5, rotation égale à celle de scipy pour une convention d'Euler (ordre, type et
      signe des angles cherchés).
  C5  reproductibilité : même graine -> mêmes paramètres.
  C6  descriptif : bornes des paramètres tirés (--motion -R R) et autocorrélation entre coupes voisines (mouvement indépendant par coupe ?).

Usage : python scripts/check_pyrecon_simulator.py --pyrecon ~/chemin/vers/pyrecon/ROSI
"""

import argparse
import itertools
import os
import random
import sys

import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--pyrecon", required=True, help="dossier ROSI du clone local de pyrecon (ajouté au chemin d'import)")
sys.path.insert(0, os.path.expanduser(parser.parse_args().pyrecon))
from rosi.registration.transformation import rigidMatrix  # noqa: E402
from rosi.simulation.simul3Ddata import psf, simulateMvt  # noqa: E402

echecs = []


def verdict(nom, ok, detail):
    print(f"{'OK    ' if ok else 'ÉCHEC '} {nom} : {detail}")
    if not ok:
        echecs.append(nom)


def affine_oblique():
    """Affine HR anisotrope et oblique (rotation quelconque), pour que les 3 orientations soient distinguables."""
    r = Rotation.from_euler("xyz", [17, -23, 31], degrees=True).as_matrix()
    a = np.eye(4)
    a[:3, :3] = r @ np.diag([1.1, 0.9, 1.3])
    a[:3, 3] = [-20.0, 15.0, 8.0]
    return a


def hr_image(valeur_monde, affine, forme):
    """Image HR dont chaque voxel vaut `valeur_monde(position monde)`."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in forme], indexing="ij"), 0).reshape(3, -1)
    monde = (affine @ np.vstack([ijk, np.ones(ijk.shape[1])]))[:3]
    return valeur_monde(monde).reshape(forme)


RAMPE = lambda p: 3.0 + 0.7 * p[0] - 0.4 * p[1] + 1.1 * p[2]  # noqa: E731
FORME = (48, 48, 48)
A_HR = affine_oblique()
HR = hr_image(RAMPE, A_HR, FORME)
IMG = nib.Nifti1Image(HR, A_HR)
MASQUE = np.ones(FORME)
DYN = float(HR.max() - HR.min())

# ------------------------------------------------------------------ C1 et C2
for avec_mouvement in (False, True):
    for orientation in ("axial", "coronal", "sagittal"):
        random.seed(12345)
        lr, _, _, transfos = simulateMvt(IMG, [-3, 3], [-3, 3], 3, orientation, MASQUE, motion=avec_mouvement)
        data, aff = lr.get_fdata(), lr.affine
        nx, ny, nz = data.shape
        ijk = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), 0).reshape(3, -1)
        nominal = aff @ np.vstack([ijk, np.ones(ijk.shape[1])])  # centres nominaux (monde), un par pixel
        attendu = np.empty(ijk.shape[1])
        pos = np.empty((3, ijk.shape[1]))
        for k in range(nz):
            sel = ijk[2] == k
            pos[:, sel] = (transfos[k] @ nominal[:, sel])[:3]
        attendu = RAMPE(pos).reshape(data.shape)
        # pixels dont les positions (±1,5 voxel HR de PSF verticale) restent loin du bord du volume HR
        inv = np.linalg.inv(A_HR)
        idx = (inv @ np.vstack([pos, np.ones(pos.shape[1])]))[:3]
        loin = np.all((idx > 4) & (idx < np.array(FORME)[:, None] - 5), axis=0).reshape(data.shape)
        erreur = np.abs(data - attendu)[loin].max() / DYN
        nom = f"C{2 if avec_mouvement else 1} {'avec' if avec_mouvement else 'sans'} mouvement, {orientation:8}"
        verdict(nom, loin.sum() > 1000 and erreur < 1e-6, f"{int(loin.sum())} pixels, écart relatif max {erreur:.2e}")

# ------------------------------------------------------------------ C3
Q = lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2  # noqa: E731
identite = np.eye(4)
HRQ = hr_image(Q, identite, FORME)
random.seed(1)
lr, _, _, _ = simulateMvt(nib.Nifti1Image(HRQ, identite), [-3, 3], [-3, 3], 1, "axial", MASQUE, motion=False)
data = lr.get_fdata()
idx_psf = np.linspace(-0.5, 0.5, 5)
poids = psf(0, idx_psf)
poids = poids / poids.sum()
v_psf = float((poids * idx_psf**2).sum())  # variance discrète de la PSF selon z (unités voxel)
i, j, k = np.meshgrid(*[np.arange(n) for n in FORME], indexing="ij")
centre = (i**2 + j**2 + k**2).astype(float)
# f'' = 2 par axe ; variance ajoutée par axe : 1/3 (spline cubique sans préfiltrage), plus v_psf selon z
attendu = centre + 0.5 * 2 * (1 / 3 + 1 / 3 + 1 / 3 + v_psf)
zone = (slice(4, -5),) * 3
ecart = np.abs(data - attendu)[zone].max()
verdict("C3 flou de l'interpolation sans préfiltrage", ecart < 1e-6, f"écart max {ecart:.2e} (attendu +{0.5 * 2 * (1 + v_psf):.4f} ; variance de PSF z {v_psf:.4f} voxel², spline 1/3 par axe)")
print(f"       -> flou ajouté par axe : sigma = {np.sqrt(1 / 3):.3f} voxel HR (FWHM {2.3548 * np.sqrt(1 / 3):.2f} voxel) ; PSF z : sigma = {np.sqrt(v_psf):.3f} voxel LR")

# ------------------------------------------------------------------ C4
rng = np.random.default_rng(0)
ok_ortho, ok_trans, ecarts = True, True, {}
for _ in range(50):
    p = np.concatenate([rng.uniform(-30, 30, 3), rng.uniform(-10, 10, 3)])
    m = rigidMatrix(p)
    r = m[:3, :3]
    ok_ortho &= bool(np.allclose(r @ r.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(r) - 1) < 1e-12)
    ok_trans &= bool(np.allclose(m[:3, 3], p[3:6]) and np.allclose(m[3], [0, 0, 0, 1]))
    # recherche élargie : ordre des axes, extrinsèque / intrinsèque, signe des angles (une première version ne testait que le signe +, et ne trouvait rien :
    # la matrice de pyrecon est la transposée de la convention trigonométrique usuelle, c'est-à-dire des angles de signe opposé)
    for ordre in ("".join(o) for o in itertools.permutations("xyz")):
        for ext in (True, False):
            for signe in (1.0, -1.0):
                seq = ordre if ext else ordre.upper()
                ang = [signe * p[{"x": 0, "y": 1, "z": 2}[a]] for a in ordre]
                e = np.abs(Rotation.from_euler(seq, ang, degrees=True).as_matrix() - r).max()
                ecarts.setdefault((seq, signe), []).append(e)
trouve = [(s, sg) for (s, sg), e in ecarts.items() if max(e) < 1e-12]
verdict("C4 rigidMatrix orthonormale, det +1, translation", ok_ortho and ok_trans, "50 tirages")
# 'XYZ' intrinsèque (angles a, b, c) et 'zyx' extrinsèque (angles c, b, a) sont deux écritures de la MÊME rotation : le critère n'est donc pas « une seule
# séquence » (première formulation, erronée) mais « au moins une, et seulement des écritures équivalentes » ; on retient R = Rx(-θ1) Ry(-θ2) Rz(-θ3).
verdict("C4 convention d'Euler identifiée", len(trouve) >= 1 and set(trouve) <= {("XYZ", -1.0), ("zyx", -1.0)}, f"(séquence scipy, signe) : {trouve} -> R = Rx(-θ1) · Ry(-θ2) · Rz(-θ3), angles de signe opposé à la convention trigonométrique usuelle")

# ------------------------------------------------------------------ C5, C6
def tirage(graine):
    random.seed(graine)
    return simulateMvt(IMG, [-4, 4], [-4, 4], 3, "axial", MASQUE, motion=True)[2]


a, b = tirage(7), tirage(7)
verdict("C5 reproductibilité (même graine)", np.array_equal(a, b), f"{a.shape[0]} coupes")
verdict("C5 graines différentes -> tirages différents", not np.array_equal(a, tirage(8)), "")
p = tirage(99)
lim = np.abs(p).max(axis=0)
ac = [np.corrcoef(p[:-1, c], p[1:, c])[0, 1] for c in range(6)]
print(f"C6 descriptif : paramètres tirés avec --motion -4 4, bornes observées (|max| par paramètre) {np.round(lim, 2)} ; autocorrélation lag 1 {np.round(ac, 2)} (coupes : {p.shape[0]})")
verdict("C6 bornes = ±(max-min)/2 = ±4", bool(np.all(lim <= 4.0 + 1e-9)), "")

print("\nRÉSULTAT :", "tous les critères sont satisfaits" if not echecs else f"ÉCHECS : {echecs}")
sys.exit(1 if echecs else 0)
