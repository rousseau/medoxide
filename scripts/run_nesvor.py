"""Lance NeSVoR (`nesvor svr`, paramètres par défaut, SANS SVoRT) sur des stacks NIfTI et convertit les coupes recalées en poses au format de medoxide.

Décision du projet : NeSVoR aux réglages par défaut, mais `--registration stack` (recalage stack-à-stack) et non `svort` : SVoRT replace les coupes dans un repère d'atlas, ce qui
rendrait les poses incomparables à celles des autres méthodes. Seuls changements : sortie 0,8 mm (comme SVRTK et medoxide), CPU (`--device -1`), graine fixée, export des coupes.

Conversion (`--valider` la vérifie sans recalage) : NeSVoR n'écrit que les coupes à masque non vide, dans l'ordre des stacks puis des coupes, chacune avec sa propre affine ; son axe x est
inversé par rapport à l'en-tête (colonne 0 de la rotation de signe opposé). Pour chaque coupe, centre et rotation normalisée sont comparés à ceux de l'en-tête : M = [R_n·diag(−1,1,1)·R_h⁻¹ | c_n − R·c_h],
transformation monde appliquée APRÈS l'affine d'en-tête (même convention que `poses_*.tsv`). Format : `stack, coupe, 16 valeurs de M` (ordre colonne). Le repère de sortie de NeSVoR peut différer de celui de
medoxide : on ne compare que des quantités indépendantes du repère (concordance aux intersections).

Usage : python scripts/run_nesvor.py --stacks s0 s1 s2 --masques m0 m1 m2 --sortie dossier/ [--nesvor ~/Sync-Exp/NeSVoR-0.5.0] [--poses poses_nesvor.tsv]
        python scripts/run_nesvor.py --valider --stacks ... --masques ... --coupes dossier_de_coupes_sans_recalage
"""

import argparse
import os
import subprocess
import time

import nibabel as nib
import numpy as np

SYMETRIE_X = np.diag([-1.0, 1.0, 1.0])


def centre_et_rotation(affine, forme, k):
    """Centre monde de la coupe `k` et rotation (colonnes normalisées) d'une affine."""
    nx, ny = forme[:2]
    centre = (affine @ np.array([(nx - 1) / 2, (ny - 1) / 2, k, 1.0]))[:3]
    return centre, affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)


def poses_depuis_coupes(chemins_stacks, chemins_masques, dossier_coupes):
    """Poses (stack, coupe) -> M 4x4 à partir des coupes écrites par NeSVoR ; refuse un nombre de coupes inattendu."""
    poses = {}
    n = 0
    for s, (cs, cm) in enumerate(zip(chemins_stacks, chemins_masques)):
        stack = nib.load(cs)
        masque = nib.load(cm).get_fdata() > 0
        for k in range(stack.shape[2]):
            if not masque[:, :, k].any():
                continue
            fichier = os.path.join(dossier_coupes, f"{n}.nii.gz")
            if not os.path.exists(fichier):
                raise SystemExit(f"coupe {n} absente : NeSVoR a écarté des coupes que le masque ne laisse pas prévoir")
            coupe = nib.load(fichier)
            ch, rh = centre_et_rotation(stack.affine, stack.shape, k)
            cn, rn = centre_et_rotation(coupe.affine, coupe.shape, 0)
            r = rn @ SYMETRIE_X @ np.linalg.inv(rh)
            m = np.eye(4)
            m[:3, :3] = r
            m[:3, 3] = cn - r @ ch
            poses[(s, k)] = m
            n += 1
    sup = [f for f in os.listdir(dossier_coupes) if f.endswith(".nii.gz") and not f.startswith("mask_")]
    if len(sup) != n:
        raise SystemExit(f"{len(sup)} coupes écrites par NeSVoR, {n} attendues")
    return poses


def deviation(m):
    """Translation (mm) et angle (°) d'une matrice rigide."""
    return float(np.linalg.norm(m[:3, 3])), float(np.degrees(np.arccos(np.clip((np.trace(m[:3, :3]) - 1) / 2, -1, 1))))


def ecrire_poses(poses, chemin):
    with open(chemin, "w") as f:
        f.write("\n".join(f"{s}\t{k}\t" + "\t".join(repr(float(v)) for v in m.flatten(order="F")) for (s, k), m in sorted(poses.items())))


parser = argparse.ArgumentParser()
parser.add_argument("--stacks", nargs="+", required=True)
parser.add_argument("--masques", nargs="+", required=True)
parser.add_argument("--sortie", help="dossier de sortie (volume nesvor.nii.gz, coupes, journal)")
parser.add_argument("--nesvor", default=os.path.expanduser("~/Sync-Exp/NeSVoR-0.5.0"), help="dépôt NeSVoR (lancé depuis les sources)")
parser.add_argument("--poses", help="fichier de poses à écrire (défaut : <sortie>/poses_nesvor.tsv)")
parser.add_argument("--valider", action="store_true", help="vérifie la conversion sur des coupes écrites SANS recalage (`nesvor register --registration none`) : M doit être l'identité")
parser.add_argument("--coupes", help="avec --valider : dossier de coupes écrites sans recalage")
args = parser.parse_args()
assert len(args.stacks) == len(args.masques)
stacks = [os.path.abspath(c) for c in args.stacks]
masques = [os.path.abspath(c) for c in args.masques]

if args.valider:
    poses = poses_depuis_coupes(stacks, masques, args.coupes)
    ecarts = np.array([deviation(m) for m in poses.values()])
    print(f"{len(poses)} coupes ; écart maximal à l'identité : {ecarts[:, 0].max():.2e} mm, {ecarts[:, 1].max():.2e} °")
    raise SystemExit(0 if ecarts[:, 0].max() < 1e-3 and ecarts[:, 1].max() < 1e-3 else 1)

sortie = os.path.abspath(args.sortie)
os.makedirs(os.path.join(sortie, "coupes"), exist_ok=True)
commande = [
    "python3", "-m", "nesvor.cli.main", "svr",
    "--input-stacks", *stacks, "--stack-masks", *masques,
    "--registration", "stack", "--output-resolution", "0.8", "--device", "-1", "--seed", "1",
    "--output-volume", os.path.join(sortie, "nesvor.nii.gz"), "--output-slices", os.path.join(sortie, "coupes"),
]
debut = time.time()
resultat = subprocess.run(commande, cwd=os.path.abspath(args.nesvor), env={**os.environ, "PYTHONPATH": os.path.abspath(args.nesvor)}, capture_output=True, text=True)
duree = time.time() - debut
open(os.path.join(sortie, "nesvor.log"), "w").write(resultat.stdout + "\n" + resultat.stderr)
print(f"NeSVoR : code de retour {resultat.returncode}, {duree:.0f} s")
if resultat.returncode != 0:
    print(resultat.stderr[-2000:])
    raise SystemExit(1)
poses = poses_depuis_coupes(stacks, masques, os.path.join(sortie, "coupes"))
ecrire_poses(poses, args.poses or os.path.join(sortie, "poses_nesvor.tsv"))
print(f"{len(poses)} poses écrites")
