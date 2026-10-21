"""Dispersion des recalages de `scripts/multibounti_repetabilite.py` : écart moyen, en mm, entre deux transformations, mesuré sur les voxels non nuls du gabarit gauche/droite
(`tmp-lr-ref`) : moyenne de `‖A₁ p − A₂ p‖`. Affiche, pour chaque volume, l'écart de chaque essai à la transformation officielle et la dispersion entre essais.

Usage : python scripts/multibounti_repetabilite_mesure.py <id>...
"""

import itertools
import os
import sys

import nibabel as nib
import numpy as np

RACINE = os.getcwd()


def points_du_gabarit(ref):
    image = nib.load(f"{ref}/tmp-lr-ref.nii.gz")
    voxels = np.argwhere(np.asanyarray(image.dataobj) > 0)
    return (image.affine @ np.c_[voxels, np.ones(len(voxels))].T).T[:, :3]


def lire_matrice(chemin):
    return np.loadtxt(chemin)


def ecart(a, b, points):
    p = np.c_[points, np.ones(len(points))].T
    return float(np.linalg.norm(((a - b) @ p)[:3], axis=0).mean())


def main():
    for ident in sys.argv[1:]:
        ref = f"{RACINE}/data/multibounti/ref/{ident}"
        dossier = f"{RACINE}/data/multibounti/essais/repet/{ident}"
        points = points_du_gabarit(ref)
        sujet = nib.load(f"{ref}/lr-lab-pad-masked-crop-t2-128.nii.gz")
        donnees = np.asanyarray(sujet.dataobj) > 0
        inverse_sujet = np.linalg.inv(sujet.affine)
        officielle = lire_matrice(f"{dossier}/officiel.mat")

        def recouvrement(m):
            p = (m @ np.c_[points, np.ones(len(points))].T)
            v = np.rint((inverse_sujet @ p)[:3]).astype(int)
            dedans = np.all((v >= 0) & (v < np.array(donnees.shape)[:, None]), axis=0)
            return float(donnees[tuple(v[:, dedans])].sum() / len(points))

        sens = "directe" if recouvrement(officielle) >= recouvrement(np.linalg.inv(officielle)) else "inverse"
        print(f"{ident} : matrice aladin {sens} (recouvrement du gabarit transformé avec le sujet : {recouvrement(officielle):.2f} / {recouvrement(np.linalg.inv(officielle)):.2f})")
        mats = [lire_matrice(f"{dossier}/essai-{i}.mat") for i in range(len([f for f in os.listdir(dossier) if f.startswith("essai-") and f.endswith(".mat")]))]
        if sens == "inverse":
            mats = [np.linalg.inv(m) for m in mats]
            officielle = np.linalg.inv(officielle)
        a_officiel = [ecart(m, officielle, points) for m in mats]
        entre = [ecart(a, b, points) for a, b in itertools.combinations(mats, 2)]
        print(f"  écart à l'officiel (mm) : {' '.join(f'{e:.2f}' for e in a_officiel)}")
        print(f"  médiane {np.median(a_officiel):.2f}, max {max(a_officiel):.2f} ; entre essais : médiane {np.median(entre):.2f}, max {max(entre):.2f}")


if __name__ == "__main__":
    main()
