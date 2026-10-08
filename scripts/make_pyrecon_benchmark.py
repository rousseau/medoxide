"""Jeux de benchmark pour comparer la reconstruction avec correction de mouvement (medoxide, SVRTK, ...) : trois stacks (axial, coronal, sagittal)
simulés par le simulateur de pyrecon (ROSI, `rosi.simulation.simul3Ddata.simulateMvt`, vérifié par `scripts/check_pyrecon_simulator.py`) depuis un atlas
HR, avec un mouvement rigide indépendant par coupe, un bruit gaussien et un masque cérébral. Le code de pyrecon est EXÉCUTÉ depuis un clone local, jamais copié.

Paramètres alignés sur les jeux simulés par medoxide (`data/atlas/sim/`) pour que seuls le modèle direct et le mouvement changent :
  - pixels du plan = pixels de l'atlas (0,8 mm), épaisseur 3,5 mm (sous-échantillonnage 3,5 / 0,8 = 4,375 voxels HR) ;
  - mouvement : angles et translations uniformes dans ± amplitude (degrés et mm), un tirage par coupe, graine fixée par (amplitude, stack) ;
  - masque : atlas > 150 (rééchantillonné au plus proche voisin par le simulateur) ;
  - bruit : gaussien additif, écart type 5 % de l'intensité moyenne des pixels de masque, graine fixée.

Écrit dans <sortie>/STA31_mvt<amplitude>_b<bruit>/ : stack<n>.nii.gz, stack<n>_mask.nii.gz (float32 / uint8, sform et qform égaux) et poses.tsv (même format que
medoxide : stack, coupe, puis les 16 valeurs de M_k, matrice monde 4x4 en ordre colonne : le mouvement absolu appliqué, composé après l'affine d'en-tête).
Amplitude 0 : sans mouvement (matrices identité) — mesure l'effet du seul modèle direct.

Modèles de mouvement (--modele) :
  independant  tirage indépendant par coupe (comportement d'origine du simulateur) ;
  lisse        « dérive lisse avec sauts » : pour chaque paramètre, un processus gaussien AR(1) d'autocorrélation --rho entre coupes successives (ordre d'acquisition
               séquentiel), remis à zéro avec la probabilité --saut par coupe (un saut), transformé en uniforme par la fonction de répartition gaussienne : la loi
               MARGINALE de chaque paramètre reste exactement uniforme sur ± amplitude, comme dans le cas indépendant ; seule la corrélation entre coupes change ;
  lisseint     idem, mais l'acquisition est entrelacée en deux paquets (coupes paires puis impaires) : le processus est lisse dans le TEMPS d'acquisition, donc moins
               entre coupes voisines. Les paramètres sont injectés dans le simulateur de pyrecon en remplaçant son générateur aléatoire (`rd`) par une suite
               préparée (6 tirages par coupe, dans l'ordre a1, a2, a3, t1, t2, t3) : on réutilise le simulateur sans copier son code, et on vérifie que les paramètres
               qu'il rend sont bien ceux planifiés.

Usage : python scripts/make_pyrecon_benchmark.py --pyrecon ~/chemin/pyrecon/ROSI --atlas data/atlas/gholipour/STA31.nii.gz --amplitudes 0 2 4 6
"""

import argparse
import contextlib
import io
import os
import random
import sys

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pyrecon", required=True, help="dossier ROSI du clone local de pyrecon")
parser.add_argument("--atlas", required=True, help="atlas HR (NIfTI)")
parser.add_argument("--amplitudes", type=float, nargs="+", default=[0, 2, 4, 6])
parser.add_argument("--bruit-pct", type=int, default=5)
parser.add_argument("--sortie", default="data/atlas/sim_pyrecon")
parser.add_argument("--epaisseur-mm", type=float, default=3.5)
parser.add_argument("--modele", choices=["independant", "lisse", "lisseint"], default="independant")
parser.add_argument("--rho", type=float, default=0.9, help="autocorrélation du processus gaussien entre coupes successives (modèles lisses)")
parser.add_argument("--saut", type=float, default=0.04, help="probabilité de saut par coupe (modèles lisses)")
args = parser.parse_args()
sys.path.insert(0, os.path.expanduser(args.pyrecon))
import rosi.simulation.simul3Ddata as simulateur  # noqa: E402
from rosi.simulation.simul3Ddata import simulateMvt  # noqa: E402
from scipy.stats import norm  # noqa: E402


def uniformes_lisses(n_coupes, rho, saut, ordre_temps, rng):
    """6 suites de `n_coupes` uniformes sur [0, 1], lisses dans le temps d'acquisition avec des sauts.

    Chaque suite : z_t AR(1) de variance 1 et d'autocorrélation rho, remis à zéro (redessiné) avec la probabilité `saut` ; u = Phi(z) est uniforme ;
    la coupe k lit l'instant ordre_temps[k]. Rend un tableau (n_coupes, 6).
    """
    u = np.empty((n_coupes, 6))
    for c in range(6):
        z = np.empty(n_coupes)
        z[0] = rng.normal()
        for t in range(1, n_coupes):
            z[t] = rng.normal() if rng.random() < saut else rho * z[t - 1] + np.sqrt(1 - rho**2) * rng.normal()
        u[:, c] = norm.cdf(z[ordre_temps])
    return u


def ordre_temps(n_coupes, entrelace):
    """Instant d'acquisition de chaque coupe : séquentiel, ou deux paquets (coupes paires puis impaires)."""
    if not entrelace:
        return np.arange(n_coupes)
    rang = np.empty(n_coupes, dtype=int)
    rang[np.concatenate([np.arange(0, n_coupes, 2), np.arange(1, n_coupes, 2)])] = np.arange(n_coupes)
    return rang


class SuiteDeTirages:
    """Remplace `random` dans le simulateur : rend, dans l'ordre, les uniformes planifiés (6 par coupe : a1, a2, a3, t1, t2, t3)."""

    def __init__(self, uniformes):
        self.suite = iter(uniformes.reshape(-1))

    def random(self):
        return next(self.suite)

atlas = nib.load(args.atlas)
pas = float(np.mean(atlas.header.get_zooms()))
sous_echantillonnage = args.epaisseur_mm / pas
masque_hr = (atlas.get_fdata() > 150).astype(np.float64)
nom_atlas = os.path.basename(args.atlas).split(".")[0]

for amplitude in args.amplitudes:
    prefixe = "" if args.modele == "independant" else f"_{args.modele}"
    dossier = f"{args.sortie}/{nom_atlas}{prefixe}_mvt{int(amplitude)}_b{args.bruit_pct}"
    os.makedirs(dossier, exist_ok=True)
    lignes = []
    for n, orientation in enumerate(("axial", "coronal", "sagittal")):
        random.seed(int(2000 + 10 * amplitude + n))
        simulateur.rd = random  # générateur d'origine (modèle indépendant)
        planifies = None
        if args.modele != "independant" and amplitude > 0:
            axe_coupes = {"axial": 2, "coronal": 1, "sagittal": 0}[orientation]
            n_coupes = int(atlas.shape[axe_coupes] // sous_echantillonnage)
            rng = np.random.default_rng(int(7000 + 10 * amplitude + n))
            u = uniformes_lisses(n_coupes, args.rho, args.saut, ordre_temps(n_coupes, args.modele == "lisseint"), rng)
            simulateur.rd = SuiteDeTirages(u)
            planifies = (2 * u - 1) * amplitude  # paramètres attendus : (2u - 1) * amplitude, dans [-amplitude, amplitude]
        with contextlib.redirect_stdout(io.StringIO()):  # le simulateur est bavard
            lr, lr_masque, parametres, transfos = simulateMvt(
                atlas, [-amplitude, amplitude], [-amplitude, amplitude], sous_echantillonnage, orientation, masque_hr, motion=amplitude > 0
            )
        if planifies is not None:
            assert parametres.shape == planifies.shape and np.abs(parametres - planifies).max() < 1e-12, "paramètres rendus != planifiés"
        data = lr.get_fdata().astype(np.float32)
        masque = lr_masque.get_fdata() > 0.5
        if args.bruit_pct > 0:
            sigma = args.bruit_pct / 100.0 * float(data[masque].mean())
            data = data + np.random.default_rng(0xC0FFEE + 7919 * (n + 1) + int(amplitude)).normal(0.0, sigma, data.shape).astype(np.float32)
        for nom, tableau, dtype in ((f"stack{n}", data, np.float32), (f"stack{n}_mask", masque, np.uint8)):
            img = nib.Nifti1Image(tableau.astype(dtype), lr.affine)
            img.set_sform(lr.affine, code=1)
            img.set_qform(lr.affine, code=1)
            nib.save(img, f"{dossier}/{nom}.nii.gz")
        for k, m in enumerate(transfos):
            lignes.append(f"{n}\t{k}\t" + "\t".join(repr(float(v)) for v in m.flatten(order="F")))
        print(f"mvt ±{amplitude:g} {orientation:8}: {data.shape}, {int(masque.sum())} voxels de masque, sigma de bruit {sigma if args.bruit_pct else 0:.1f}")
    open(f"{dossier}/poses.tsv", "w").write("\n".join(lignes))
