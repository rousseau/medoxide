"""Mesures du banc de vérité terrain (docs/ROADMAP.md, F4.0.1) : fonctions pures sur tableaux numpy.

Toutes les mesures se font dans un masque booléen (celui du volume source) et sur des volumes DÉJÀ recalés
sur la source. Les choix suivants sont figés par le ROADMAP et ne changent pas après une mesure :

- normalisation affine `y = a·x + b` par moindres carrés dans le masque, une fois, avant PSNR et SSIM ;
- PSNR : pic = maximum de la source dans le masque ;
- SSIM 3D : fenêtre uniforme 7×7×7, K1 = 0,01, K2 = 0,03, étendue = max − min de la source dans le masque,
  variance et covariance de POPULATION (écart assumé avec le réglage par défaut de scikit-image, qui prend la
  variance d'échantillon), bords en réflexion, moyenne de la carte sur les voxels du masque ;
- netteté S : définition de `benchmark_variantes_s2s.py`, reprise à l'identique ;
- Dice par groupe : 7 groupes de tissus de BOUNTI-19 (table `GROUPES`).
"""

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, uniform_filter

#: Groupes de tissus : nom -> numéros d'étiquettes de BOUNTI-19 (table `crates/medoxide-bounti/src/table.rs`).
#: Le cavum du septum pellucidum (9) et les 3e et 4e ventricules (18, 19) comptent avec les ventricules ;
#: le vermis (13) avec le cervelet ; thalamus (16, 17) et noyaux de la base (14, 15) forment les noyaux gris.
GROUPES = {
    "lcr_externe": (1, 2),
    "cortex": (3, 4),
    "matiere_blanche": (5, 6),
    "ventricules": (7, 8, 9, 18, 19),
    "cervelet": (11, 12, 13),
    "noyaux_gris": (14, 15, 16, 17),
    "tronc": (10,),
}

K1, K2, FENETRE_SSIM = 0.01, 0.03, 7


def regrouper(etiquettes):
    """Image d'étiquettes BOUNTI-19 -> image de numéros de groupe (1 à 7, dans l'ordre de `GROUPES`), 0 pour le reste."""
    sortie = np.zeros(etiquettes.shape, dtype=np.uint8)
    for numero, (_, labels) in enumerate(GROUPES.items(), start=1):
        sortie[np.isin(etiquettes, labels)] = numero
    return sortie


def normalisation_affine(x, ref, masque):
    """Renvoie `(a, b)` minimisant la somme des `(a·x + b − ref)²` dans le masque (moindres carrés)."""
    A = np.stack([x[masque].ravel(), np.ones(int(masque.sum()))], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, ref[masque].ravel(), rcond=None)
    return float(a), float(b)


def psnr(x, ref, masque):
    """PSNR en dB dans le masque ; `x` est supposé déjà normalisé. Pic = maximum de `ref` dans le masque."""
    eqm = float(np.mean((x[masque] - ref[masque]) ** 2))
    if eqm == 0.0:
        return float("inf")
    return float(20.0 * np.log10(ref[masque].max() / np.sqrt(eqm)))


def carte_ssim(x, ref, masque, fenetre=FENETRE_SSIM):
    """Carte de SSIM 3D (fenêtre uniforme `fenetre`³), étendue de la source lue dans le masque."""
    etendue = float(ref[masque].max() - ref[masque].min())
    c1, c2 = (K1 * etendue) ** 2, (K2 * etendue) ** 2
    moy = lambda v: uniform_filter(v, size=fenetre, mode="reflect")
    mx, my = moy(x), moy(ref)
    vx, vy = moy(x * x) - mx * mx, moy(ref * ref) - my * my
    cxy = moy(x * ref) - mx * my
    return ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def ssim(x, ref, masque):
    """SSIM moyen sur les voxels du masque."""
    return float(carte_ssim(x, ref, masque)[masque].mean())


def ncc(x, ref, masque):
    """Corrélation croisée normalisée (Pearson) dans le masque."""
    a, b = x[masque] - x[masque].mean(), ref[masque] - ref[masque].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def nettete(v):
    """Netteté S de `benchmark_variantes_s2s.py` : gradient moyen du volume lissé d'un voxel, dans le tissu érodé de 3
    voxels, divisé par l'intensité médiane de ce tissu."""
    v = np.asarray(v, dtype=np.float64)
    tissu = v > 0.05 * np.percentile(v[v > 0], 99.5)
    interieur = binary_erosion(tissu, iterations=3)
    g = np.sqrt(sum(np.gradient(gaussian_filter(v, 1.0), axis=a) ** 2 for a in range(3)))
    return float(g[interieur].mean() / np.median(v[interieur]))


def dice_par_groupe(seg, ref_seg):
    """Dice de chaque groupe entre deux images d'étiquettes BOUNTI-19 ; `nan` si le groupe est absent des deux."""
    a, b = regrouper(seg), regrouper(ref_seg)
    sortie = {}
    for numero, nom in enumerate(GROUPES, start=1):
        ma, mb = a == numero, b == numero
        total = int(ma.sum()) + int(mb.sum())
        sortie[nom] = float("nan") if total == 0 else 2.0 * int((ma & mb).sum()) / total
    return sortie


def mesures_volume(x, ref, masque):
    """Normalise `x` sur `ref` dans le masque puis renvoie `dict(a, b, psnr, ssim, ncc, nettete)`.

    La netteté est celle de `x` NON normalisé (invariante à l'échelle) ; PSNR et SSIM sont ceux du volume normalisé."""
    a, b = normalisation_affine(x, ref, masque)
    xn = a * x + b
    return {
        "a": a,
        "b": b,
        "psnr": psnr(xn, ref, masque),
        "ssim": ssim(xn, ref, masque),
        "ncc": ncc(x, ref, masque),
        "nettete": nettete(x),
    }


def lire_poses(chemin):
    """Poses d'un TSV de `medx svr recon --poses` ou `medx svr simulate` : dict `(indice de pile, coupe) -> matrice 4×4`.

    Le fichier donne 16 valeurs en ordre colonne ; la pile est désignée par son nom, numérotée dans l'ordre d'apparition."""
    noms, poses = {}, {}
    with open(chemin) as f:
        for ligne in f.read().strip().split("\n"):
            c = ligne.split("\t")
            if c[0] == "stack":
                continue
            s = noms.setdefault(c[0], len(noms))
            poses[(s, int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    return poses


def kabsch(P, Q):
    """Rotation `R` et translation `t` minimisant `|R P + t − Q|²` (points en lignes), sans réflexion."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, mq - R @ mp


def _poser(M, X):
    return (M[:3, :3] @ X.T).T + M[:3, 3]


def erreur_de_pose(points, poses_vraies, poses_estimees):
    """Erreur de pose par coupe, en mm, après retrait du mouvement COMMUN.

    `points[(pile, coupe)]` : points (n × 3, monde) de la coupe à sa pose d'en-tête. Les poses vraies et estimées sont composées après l'affine
    d'en-tête ; une reconstruction est définie à un mouvement rigide d'ensemble près : on ajuste (Kabsch) le déplacement `G` qui amène au mieux les points
    posés par les poses vraies sur ceux posés par les poses estimées, sur TOUTES les coupes, puis l'erreur d'une coupe est l'écart quadratique moyen
    de ses points `|G·(Mv X) − Me X|`. Renvoie `dict (pile, coupe) -> erreur_mm`."""
    cles = [k for k in points if k in poses_vraies and k in poses_estimees]
    vrais = {k: _poser(poses_vraies[k], points[k]) for k in cles}
    estimes = {k: _poser(poses_estimees[k], points[k]) for k in cles}
    R, t = kabsch(np.vstack([vrais[k] for k in cles]), np.vstack([estimes[k] for k in cles]))
    return {k: float(np.sqrt((((R @ vrais[k].T).T + t - estimes[k]) ** 2).sum(1).mean())) for k in cles}
