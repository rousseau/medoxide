"""Lit les transformations par coupe d'une reconstruction SVRTK lancée avec `--debug`, les met dans la convention de medoxide et valide cette convention numériquement.

Entrées : le dossier de sortie d'une reconstruction `mirtk reconstruct ... --debug` (fichiers `transformationN.dof`, `sliceN.nii.gz`, `weightsN.nii.gz`, et le volume reconstruit) et les stacks d'entrée.
  1. `transformationN.dof` -> matrice 4x4 monde par `mirtk convert-dof -output-format aladin` (dans Docker) ; fichier `matrices.txt` écrit à côté du dossier, réutilisé s'il existe.
  2. Chaque `sliceN` (une coupe recadrée sur le masque, avec sa propre affine) est identifiée à (stack, coupe) : l'origine de son pixel (0,0) tombe sur des indices ENTIERS d'un seul stack.
  3. Convention, par mesure : pour chaque coupe, la NCC entre ses pixels de masque et le volume de SVRTK échantillonné (trilinéaire) aux positions T(x), pour T = identité, M et M⁻¹ ; la bonne
     convention est celle de NCC maximale. Rapportée sur toutes les coupes.
  4. Poses au format de medoxide (`stack, coupe, 16 valeurs en ordre colonne`, mouvement monde appliqué après l'affine d'en-tête, direction coupe -> volume de SVRTK).

Le repère de sortie est celui du volume de SVRTK (stack template), différent de celui de medoxide : on ne compare que des quantités indépendantes du repère (ou après calage rigide de Kabsch).

Usage : python scripts/read_svrtk_poses.py --debug-dir dossier/out --volume dossier/out/svrtk.nii.gz --stacks s0.nii.gz s1.nii.gz s2.nii.gz --masques m0.nii.gz m1.nii.gz m2.nii.gz --sortie poses_svrtk.tsv
"""

import argparse
import glob
import os
import subprocess

import nibabel as nib
import numpy as np
from scipy import ndimage


def convertir_dofs(dossier, n):
    """Matrices 4x4 (convention « aladin ») des n transformations `transformationN.dof` du dossier, via Docker ; lues dans `matrices.txt` si présent."""
    fichier = os.path.join(os.path.dirname(os.path.abspath(dossier)), "matrices.txt")
    if not os.path.exists(fichier):
        commande = f"cd /out && for i in $(seq 0 {n - 1}); do mirtk convert-dof transformation$i.dof /tmp/t.mat -output-format aladin >/dev/null 2>&1; echo \"SLICE $i\"; cat /tmp/t.mat; done"
        sortie = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/arm64", "-v", f"{os.path.abspath(dossier)}:/out", "fetalsvrtk/svrtk:general_auto_arm", "bash", "-lc", commande],
            capture_output=True, text=True, check=True,
        ).stdout
        open(fichier, "w").write(sortie)
    mats, courant = {}, []
    for ligne in open(fichier):
        if ligne.startswith("SLICE"):
            if courant:
                mats[idx] = np.array(courant, float)
            idx, courant = int(ligne.split()[1]), []
        elif ligne.strip():
            courant.append([float(v) for v in ligne.split()])
    mats[idx] = np.array(courant, float)
    return mats


def identifier_coupes(dossier, stacks):
    """{indice SVRTK: (stack, coupe, i0, j0)} : l'origine du pixel (0,0) de la coupe recadrée doit tomber sur des indices entiers (à 1e-2 près) d'un seul stack."""
    stacks_nib = [nib.load(s) for s in stacks]
    inverses = [np.linalg.inv(s.affine) for s in stacks_nib]
    sortie = {}
    n = len(glob.glob(os.path.join(dossier, "slice[0-9]*.nii.gz")))
    for i in range(n):
        im = nib.load(os.path.join(dossier, f"slice{i}.nii.gz"))
        origine = im.affine @ np.array([0, 0, 0, 1.0])
        trouves = []
        for s, inv in enumerate(inverses):
            idx = (inv @ origine)[:3]
            nz = stacks_nib[s].shape[2]
            if np.all(np.abs(idx - np.round(idx)) < 1e-2) and 0 <= round(idx[2]) < nz:
                # même orientation que le stack : colonnes de l'affine égales
                if np.allclose(inv[:3, :3] @ im.affine[:3, :3], np.eye(3), atol=1e-3):
                    trouves.append((s, int(round(idx[2])), int(round(idx[0])), int(round(idx[1]))))
        sortie[i] = trouves[0] if len(trouves) == 1 else None
    return sortie


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def valider(dossier, volume, stacks, masques, mats, ident):
    vol = nib.load(volume)
    donnees, inv_vol = vol.get_fdata(dtype=np.float32), np.linalg.inv(vol.affine)
    stacks_nib = [nib.load(s) for s in stacks]
    masques_nib = [nib.load(m).get_fdata() > 0 for m in masques]
    resultats = {"identité": [], "M": [], "M⁻¹": []}
    for i, info in ident.items():
        if info is None:
            continue
        s, k, i0, j0 = info
        im = nib.load(os.path.join(dossier, f"slice{i}.nii.gz"))
        nx, ny = im.shape[:2]
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        monde = im.affine @ np.vstack([ii.ravel(), jj.ravel(), np.zeros(ii.size), np.ones(ii.size)])
        masque = masques_nib[s][i0:i0 + nx, j0:j0 + ny, k].ravel()
        pixels = im.get_fdata(dtype=np.float32)[:, :, 0].ravel()
        if masque.sum() < 200:
            continue
        for nom, T in (("identité", np.eye(4)), ("M", mats[i]), ("M⁻¹", np.linalg.inv(mats[i]))):
            x = inv_vol @ (T @ monde)
            v = ndimage.map_coordinates(donnees, x[:3], order=1, mode="constant", cval=0.0)
            dans = masque & (v != 0)
            if dans.sum() >= 200:
                resultats[nom].append(ncc(pixels[dans], v[dans]))
    return resultats


def ecrire_poses(chemin, mats, ident, direction):
    lignes = []
    for i, info in sorted(ident.items()):
        if info is None:
            continue
        m = mats[i] if direction == "M" else np.linalg.inv(mats[i])
        lignes.append(f"{info[0]}\t{info[1]}\t" + "\t".join(repr(float(v)) for v in m.flatten(order="F")))
    open(chemin, "w").write("\n".join(lignes))
    return len(lignes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--stacks", nargs="+", required=True)
    parser.add_argument("--masques", nargs="+", required=True)
    parser.add_argument("--sortie", required=True)
    args = parser.parse_args()
    n = len(glob.glob(os.path.join(args.debug_dir, "transformation[0-9]*.dof")))
    if n == 0:
        raise SystemExit(f"aucune transformation dans {args.debug_dir} : SVRTK a-t-il été lancé avec --debug (scripts/run_svrtk.py --debug) ?")
    mats = convertir_dofs(args.debug_dir, n)
    ident = identifier_coupes(args.debug_dir, args.stacks)
    non_id = [i for i, v in ident.items() if v is None]
    print(f"{n} coupes SVRTK ; {n - len(non_id)} identifiées à (stack, coupe) ; non identifiées : {non_id}")
    res = valider(args.debug_dir, args.volume, args.stacks, args.masques, mats, ident)
    for nom, v in res.items():
        print(f"  convention {nom:9} : NCC moyenne {np.mean(v):.4f}, médiane {np.median(v):.4f} ({len(v)} coupes)")
    meilleure = max(res, key=lambda k: np.mean(res[k]))
    print(f"convention retenue : {meilleure}")
    if meilleure in ("M", "M⁻¹"):
        print(f"{ecrire_poses(args.sortie, mats, ident, meilleure)} poses écrites dans {args.sortie}")
