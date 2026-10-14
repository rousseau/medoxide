"""Lit les transformations par coupe d'une reconstruction NiftyMIC (`--write-motion-correction 1`), les met dans la convention de medoxide et valide cette convention numériquement.

NiftyMIC (Ebner et al. 2020, NeuroImage 206:116324) écrit, dans le sous-dossier de correction de mouvement, un fichier ITK `.tfm` par stack (recalage volume-vers-volume : le stack de référence est l'identité) et un par coupe (`<stack>_slice<k>.tfm`, `k` = indice de coupe du stack d'origine ; les coupes dont le masque est vide ne sont pas écrites). Chaque `.tfm` est une `AffineTransform_double_3_3` : 9 coefficients de matrice (ligne par ligne) puis 3 de translation, en coordonnées LPS (ITK).

Convention, par mesure (comme `read_svrtk_poses.py`) : pour chaque coupe, la NCC entre ses pixels de masque et le volume de NiftyMIC échantillonné (trilinéaire) aux positions T(x), pour plusieurs façons de composer le recalage de la coupe et celui du stack (et leurs inverses) ; la bonne est celle de NCC maximale, rapportée sur toutes les coupes. Poses au format de medoxide (`stack, coupe, 16 valeurs en ordre colonne`, mouvement monde RAS appliqué après l'affine d'en-tête, direction coupe -> volume de NiftyMIC).

Le repère de sortie est celui du volume de NiftyMIC : on ne compare que des quantités indépendantes du repère (ou après calage rigide de Kabsch).

Usage : python scripts/read_niftymic_poses.py --dossier sortie_niftymic --stacks s0.nii.gz s1.nii.gz s2.nii.gz --masques m0.nii.gz m1.nii.gz m2.nii.gz --sortie poses_niftymic.tsv
Les stacks et masques sont donnés dans l'ordre voulu pour les indices de stack de la sortie ; leurs noms de fichier doivent contenir le nom NiftyMIC (`<nom>_T2w`) : le fichier `.tfm` est cherché par le nom du stack sans `.nii.gz`.
"""

import argparse
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def lire_tfm(chemin):
    """Matrice 4x4 (LPS) d'un fichier ITK `AffineTransform_double_3_3` : p' = A p + t (centre fixe supposé nul)."""
    for ligne in open(chemin):
        if ligne.startswith("Parameters:"):
            p = [float(v) for v in ligne.split()[1:]]
        if ligne.startswith("FixedParameters:"):
            assert all(abs(float(v)) < 1e-9 for v in ligne.split()[1:]), "centre de rotation non nul : non géré"
    m = np.eye(4)
    m[:3, :3] = np.array(p[:9]).reshape(3, 3)
    m[:3, 3] = p[9:12]
    return m


def ras(m_lps):
    """Même transformation exprimée en coordonnées RAS (conjugaison par le changement de signe des axes x et y)."""
    return LPS_RAS @ m_lps @ LPS_RAS


def candidats(ts, tk):
    """Façons de composer le recalage de la coupe `ts` et celui du stack `tk` (en LPS), avec leurs inverses."""
    base = {"coupe∘stack": ts @ tk, "stack∘coupe": tk @ ts, "coupe": ts, "stack": tk}
    return {**base, **{f"{n}⁻¹": np.linalg.inv(m) for n, m in base.items()}}


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dossier", required=True, help="dossier de sortie de niftymic_reconstruct_volume")
    parser.add_argument("--volume", default=None, help="volume reconstruit (défaut : <dossier>/srr.nii.gz)")
    parser.add_argument("--sous-dossier", default="motion_correction")
    parser.add_argument("--stacks", nargs="+", required=True)
    parser.add_argument("--masques", nargs="+", required=True)
    parser.add_argument("--sortie", required=True)
    parser.add_argument("--convention", default=None, help="impose une convention au lieu de la meilleure (voir la liste imprimée)")
    args = parser.parse_args()

    chemin_volume = args.volume or os.path.join(args.dossier, "srr.nii.gz")
    sans_volume = not os.path.exists(chemin_volume)
    if sans_volume:
        # NiftyMIC peut échouer avant d'écrire le volume (sujets très agités) tout en ayant écrit les transformations : pas de validation possible, la convention doit être imposée
        assert args.convention, f"{chemin_volume} absent : donner --convention (validée sur un autre sujet)"
    else:
        vol = nib.load(chemin_volume)
        donnees, inv_vol = vol.get_fdata(dtype=np.float32), np.linalg.inv(vol.affine)
    dossier_tfm = os.path.join(args.dossier, args.sous_dossier)

    coupes = []  # (stack, coupe, pixels RAS (n,3), intensités, ts, tk)
    for s, (fs, fm) in enumerate(zip(args.stacks, args.masques)):
        nom = os.path.basename(fs).removesuffix(".nii.gz")
        chemin_stack = os.path.join(dossier_tfm, f"{nom}.tfm")
        if not os.path.exists(chemin_stack):
            print(f"stack {s} ({nom}) : aucune transformation (toutes ses coupes rejetées par NiftyMIC) ; ignoré")
            continue
        tk = lire_tfm(chemin_stack)
        img, masque = nib.load(fs), nib.load(fm).get_fdata() > 0
        d = img.get_fdata(dtype=np.float32)
        for k in range(img.shape[2]):
            chemin = os.path.join(dossier_tfm, f"{nom}_slice{k}.tfm")
            if not os.path.exists(chemin):
                continue
            ij = np.argwhere(masque[:, :, k])
            if len(ij) < 100:
                continue
            h = np.c_[ij, np.full(len(ij), k), np.ones(len(ij))]
            coupes.append((s, k, (img.affine @ h.T).T[:, :3], d[ij[:, 0], ij[:, 1], k], lire_tfm(chemin), tk))
    print(f"{len(coupes)} coupes avec transformation (sur {sum(nib.load(f).shape[2] for f in args.stacks)})")

    def echantillonner(m_ras, x):
        p = (m_ras[:3, :3] @ x.T).T + m_ras[:3, 3]
        idx = (inv_vol[:3, :3] @ p.T).T + inv_vol[:3, 3]
        return ndimage.map_coordinates(donnees, idx.T, order=1, mode="constant", cval=np.nan)

    noms = list(candidats(np.eye(4), np.eye(4)))
    scores = {n: [] for n in noms + ["identité"]}
    for _, _, x, y, ts, tk in ([] if sans_volume else coupes):
        for n, m in candidats(ts, tk).items():
            v = echantillonner(ras(m), x)
            ok = ~np.isnan(v)
            scores[n].append(ncc(y[ok], v[ok]) if ok.sum() > 50 else np.nan)
        v = echantillonner(np.eye(4), x)
        ok = ~np.isnan(v)
        scores["identité"].append(ncc(y[ok], v[ok]) if ok.sum() > 50 else np.nan)
    if not sans_volume:
        print("NCC moyenne coupe-volume selon la convention :")
        for n, s in scores.items():
            print(f"  {n:14} {np.nanmean(s):.4f}")
    meilleure = args.convention or max(noms, key=lambda n: np.nanmean(scores[n]))
    print(f"convention retenue : {meilleure}")

    with open(args.sortie, "w") as f:
        for s, k, _, _, ts, tk in coupes:
            m = ras(candidats(ts, tk)[meilleure])
            f.write(f"{s}\t{k}\t" + "\t".join(repr(float(v)) for v in m.flatten(order="F")) + "\n")
    print(f"poses écrites dans {args.sortie}")


if __name__ == "__main__":
    main()
