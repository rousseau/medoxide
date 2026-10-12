"""Concordance aux intersections : mesure de la qualité d'un jeu de poses par coupe SANS reconstruction (critère de pyrecon/ROSI).

Deux coupes de stacks différents, aux poses données, se croisent le long d'une droite (l'intersection de leurs plans). Si les poses sont justes, les intensités de chaque coupe le long de cette droite
concordent (même tissu au même endroit). Pour chaque paire de coupes de stacks différents dont les plans se croisent avec un angle d'au moins `--angle-min` degrés : échantillonnage de la droite tous
les 0,5 mm, bilinéaire dans chaque coupe, en ne gardant que les points dans les deux coupes ET dans le masque cérébral des deux ; la paire compte si elle a au moins 20 points (10 mm) ; sa mesure est la
corrélation normalisée (NCC) des deux profils. Résultat : NCC moyenne et médiane sur les paires, pour chaque jeu de poses.

La mesure est neutre vis-à-vis des méthodes : elle n'utilise ni volume reconstruit, ni la NCC à un volume, ni l'information mutuelle de SVRTK. Les paires sont restreintes aux coupes présentes dans TOUS
les jeux de poses comparés (même population pour chacun). Les poses sont au format medoxide : `stack, coupe, 16 valeurs` (matrice monde 4x4 en ordre colonne, mouvement appliqué après l'affine d'en-tête) ;
une coupe absente du fichier n'est pas utilisée.

Contrôle de la mesure (`--valider`) : sur des poses vraies connues, la concordance doit baisser de façon monotone quand on les perturbe de plus en plus (dose-réponse).

Usage : python scripts/intersection_consistency.py --stacks s0 s1 s2 --masques m0 m1 m2 --poses nom=fichier.tsv [--poses nom2=fichier2.tsv ...] [--identite]
        python scripts/intersection_consistency.py --stacks ... --masques ... --valider vraies.tsv
"""

import argparse

import nibabel as nib
import numpy as np
from scipy import ndimage


def lire_poses(chemin):
    poses = {}
    for ligne in open(chemin):
        c = ligne.strip().split("\t")
        if len(c) == 18:
            poses[(int(c[0]), int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    return poses


class Jeu:
    """Stacks, masques et affines."""

    def __init__(self, chemins_stacks, chemins_masques):
        self.img = [nib.load(c) for c in chemins_stacks]
        self.data = [i.get_fdata(dtype=np.float32) for i in self.img]
        self.mask = [nib.load(c).get_fdata() > 0 for c in chemins_masques]
        self.aff = [i.affine for i in self.img]


def preparer(jeu, poses, coupes):
    """Pour chaque coupe : (matrice monde MA, inverse, centre, normale unitaire, h = n·c)."""
    out = {}
    for (s, k) in coupes:
        MA = poses[(s, k)] @ jeu.aff[s]
        nx, ny = jeu.data[s].shape[:2]
        c = (MA @ np.array([(nx - 1) / 2, (ny - 1) / 2, k, 1.0]))[:3]
        n = MA[:3, 2] / np.linalg.norm(MA[:3, 2])
        out[(s, k)] = (MA, np.linalg.inv(MA), c, n, float(n @ c))
    return out


def concordance(jeu, poses, coupes, angle_min_deg=30.0, pas_mm=0.5, portee_mm=150.0, min_points=20, hors_stack=None):
    """Liste des NCC de chaque paire de coupes de stacks différents (coupes dans `coupes`). Si `hors_stack` est donné, les paires qui impliquent ce stack sont ignorées."""
    prep = preparer(jeu, poses, coupes)
    cles = sorted(coupes)
    t = np.arange(-portee_mm, portee_mm + pas_mm, pas_mm)
    sin_min = np.sin(np.radians(angle_min_deg))
    ncc = []
    for a in range(len(cles)):
        for b in range(a + 1, len(cles)):
            (s1, k1), (s2, k2) = cles[a], cles[b]
            if s1 == s2 or hors_stack in (s1, s2):
                continue
            _, inv1, c1, n1, h1 = prep[(s1, k1)]
            _, inv2, c2, n2, h2 = prep[(s2, k2)]
            d = np.cross(n1, n2)
            nd = np.linalg.norm(d)
            if nd < sin_min:
                continue
            x0 = (h1 * np.cross(n2, d) + h2 * np.cross(d, n1)) / (nd * nd)
            pts = x0[None, :] + t[:, None] * (d / nd)[None, :]
            ph = np.hstack([pts, np.ones((len(t), 1))]).T
            vals, ok = [], np.ones(len(t), bool)
            for (s, k, inv) in ((s1, k1, inv1), (s2, k2, inv2)):
                idx = inv @ ph
                nx, ny = jeu.data[s].shape[:2]
                i, j = idx[0], idx[1]
                dedans = (i >= 0) & (i <= nx - 1) & (j >= 0) & (j <= ny - 1)
                ii, jj = np.clip(np.round(i).astype(int), 0, nx - 1), np.clip(np.round(j).astype(int), 0, ny - 1)
                ok &= dedans & jeu.mask[s][ii, jj, k]
                vals.append(ndimage.map_coordinates(jeu.data[s][:, :, k], [i, j], order=1, mode="nearest"))
            if ok.sum() >= min_points:
                p, q = vals[0][ok], vals[1][ok]
                p, q = p - p.mean(), q - q.mean()
                den = np.sqrt((p * p).sum() * (q * q).sum())
                if den > 1e-9:
                    ncc.append(float((p * q).sum() / den))
    return np.array(ncc)


def resume(nom, ncc):
    return f"{nom:28} {len(ncc):5d} paires | NCC moyenne {ncc.mean():.4f}, médiane {np.median(ncc):.4f}, 10e centile {np.percentile(ncc, 10):.4f}"


def perturber(poses, jeu, amplitude_mm, graine):
    """Compose à gauche de chaque pose un mouvement rigide aléatoire de déplacement quadratique moyen ≈ `amplitude_mm` autour du centre de la coupe (rotation et translation)."""
    rng = np.random.default_rng(graine)
    from scipy.spatial.transform import Rotation

    out = {}
    for (s, k), m in poses.items():
        nx, ny = jeu.data[s].shape[:2]
        c = (m @ jeu.aff[s] @ np.array([(nx - 1) / 2, (ny - 1) / 2, k, 1.0]))[:3]
        t = rng.normal(size=3) * amplitude_mm / np.sqrt(3) / np.sqrt(2)
        theta = rng.normal(size=3) * (amplitude_mm / 40.0) / np.sqrt(3) / np.sqrt(2)  # ≈ amplitude/2 mm à 40 mm du centre
        R = Rotation.from_rotvec(theta).as_matrix()
        D = np.eye(4)
        D[:3, :3] = R
        D[:3, 3] = c + t - R @ c
        out[(s, k)] = D @ m
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacks", nargs="+", required=True)
    parser.add_argument("--masques", nargs="+", required=True)
    parser.add_argument("--poses", action="append", default=[], help="nom=fichier.tsv")
    parser.add_argument("--identite", action="store_true", help="ajouter les poses d'en-tête (identité)")
    parser.add_argument("--valider", help="fichier de poses vraies : dose-réponse sous perturbation")
    parser.add_argument("--angle-min", type=float, default=30.0)
    args = parser.parse_args()
    jeu = Jeu(args.stacks, args.masques)
    if args.valider:
        vraies = lire_poses(args.valider)
        coupes = sorted(vraies)
        print(resume("poses vraies", concordance(jeu, vraies, coupes, args.angle_min)))
        for amp in (0.5, 1.0, 2.0, 4.0, 8.0):
            print(resume(f"vraies + perturbation {amp:g} mm", concordance(jeu, perturber(vraies, jeu, amp, 7), coupes, args.angle_min)))
        ident = {k: np.eye(4) for k in coupes}
        print(resume("identité (en-tête)", concordance(jeu, ident, coupes, args.angle_min)))
    else:
        jeux = {nom: lire_poses(f) for nom, f in (p.split("=", 1) for p in args.poses)}
        # une pose exactement égale à l'identité est une coupe non recalée (masque trop petit) : elle n'est pas utilisée
        jeux = {nom: {k: m for k, m in p.items() if not np.allclose(m, np.eye(4), atol=1e-12)} for nom, p in jeux.items()}
        coupes = sorted(set.intersection(*[set(p) for p in jeux.values()]))
        if args.identite:
            jeux["identité (en-tête)"] = {k: np.eye(4) for k in coupes}
        print(f"{len(coupes)} coupes communes à tous les jeux de poses")
        for nom, p in jeux.items():
            print(resume(nom, concordance(jeu, p, coupes, args.angle_min)))
