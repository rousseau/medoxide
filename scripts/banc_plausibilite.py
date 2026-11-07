"""Mesures de plausibilité du simulateur face à des piles réelles (docs/ROADMAP.md, F4.0.1, critère (v) et amendement 1).

Trois rapports simulé/réel, définis AVANT toute mesure (voir le protocole du ROADMAP) :
- bruit : `σ_air / μ_tissu`, σ_air sur des carrés de 12 × 12 pixels d'air (coins de chaque coupe, maximum < 5 % de μ_tissu) choisis sur la pile réelle ;
- largeur de bord : largeur 10–90 % (mm) de la transition le long de la coupe, aux 500 emplacements de plus fort gradient de la pile réelle ;
- biais : plage médiane du logarithme du champ de biais estimé (`diagnose_the_estimated_bias_range`).
Fonctions pures sur tableaux numpy (axes `[i, j, k]` avec `k` l'indice de coupe) ; aucune entrée-sortie ici.
"""

import re

import numpy as np
from scipy.ndimage import binary_dilation

CARRE = 12
SEUIL_AIR = 0.05
N_EMPLACEMENTS = 500
CONTRASTE_MIN = 0.2
MIN_CARRES = 20


def moyenne_tissu(pile, masque):
    """Intensité moyenne de la pile dans le masque de cerveau."""
    return float(pile[masque].mean())


def choisir_carres(pile, mu):
    """Carrés d'air : pour chaque coupe `k`, chacun des quatre coins `CARRE × CARRE` dont le maximum est inférieur à `SEUIL_AIR · mu`.

    Renvoie une liste de `(k, i0, j0)` (coin supérieur gauche du carré). Moins de `MIN_CARRES` carrés : `ValueError` (pas de mesure)."""
    ni, nj, nk = pile.shape
    carres = []
    for k in range(nk):
        for i0 in (0, ni - CARRE):
            for j0 in (0, nj - CARRE):
                if pile[i0 : i0 + CARRE, j0 : j0 + CARRE, k].max() < SEUIL_AIR * mu:
                    carres.append((k, i0, j0))
    if len(carres) < MIN_CARRES:
        raise ValueError(f"{len(carres)} carrés d'air seulement (minimum {MIN_CARRES}) : pas de mesure du bruit")
    return carres


def sigma_air(pile, carres):
    """Écart type des pixels de tous les carrés d'air, regroupés."""
    valeurs = np.concatenate([pile[i0 : i0 + CARRE, j0 : j0 + CARRE, k].ravel() for k, i0, j0 in carres])
    return float(valeurs.std())


def rapport_bruit(pile, masque, carres):
    """`σ_air / μ_tissu` de la pile, les carrés d'air étant ceux choisis sur la pile réelle."""
    return sigma_air(pile, carres) / moyenne_tissu(pile, masque)


def emplacements_de_bord(pile, masque, mu, n_max=N_EMPLACEMENTS):
    """Emplacements `(i, j, k)` des bords les plus nets le long de la coupe, choisis sur la pile (réelle).

    Dans le masque dilaté de 2 voxels et pour `3 ≤ k ≤ nz − 4` : maxima locaux, le long de `k`, du gradient `|S[k+1] − S[k−1]| / 2` (strictement supérieur
    au précédent, au moins égal au suivant) ; on garde les `n_max` plus forts dont le contraste (max − min sur `S[k−3 … k+3]`) dépasse `CONTRASTE_MIN · mu`."""
    nk = pile.shape[2]
    g = np.zeros_like(pile, dtype=np.float64)
    g[:, :, 1:-1] = np.abs(pile[:, :, 2:] - pile[:, :, :-2]) / 2.0
    maxi = np.zeros(pile.shape, dtype=bool)
    maxi[:, :, 1:-1] = (g[:, :, 1:-1] > g[:, :, :-2]) & (g[:, :, 1:-1] >= g[:, :, 2:])
    valide = maxi & binary_dilation(masque, iterations=2)
    valide[:, :, :3] = False
    valide[:, :, nk - 3 :] = False
    candidats = np.argwhere(valide)
    gradients = g[valide]
    ordre = np.argsort(-gradients, kind="stable")
    gardes = []
    for idx in ordre:
        i, j, k = candidats[idx]
        fen = pile[i, j, k - 3 : k + 4]
        if fen.max() - fen.min() > CONTRASTE_MIN * mu:
            gardes.append((i, j, k))
            if len(gardes) == n_max:
                break
    return np.array(gardes, dtype=int).reshape(-1, 3)


def largeur_de_bord(pile, emplacements, espacement_mm):
    """Largeur 10–90 % (mm) de la transition à chaque emplacement : profil normalisé `q = (p − min) / (max − min)` sur `S[k−3 … k+3]`, orienté croissant ;
    positions (interpolation linéaire, en coupes) du premier franchissement de 0,1 puis du premier franchissement de 0,9 qui suit ; écart × espacement."""
    largeurs = []
    for i, j, k in emplacements:
        p = pile[i, j, k - 3 : k + 4].astype(np.float64)
        q = (p - p.min()) / (p.max() - p.min())
        if q[0] > q[-1]:
            q = q[::-1]

        def position(niveau, depuis):
            idx = next(n for n in range(depuis, len(q)) if q[n] >= niveau)
            if idx == 0:
                return 0.0, 0
            return (idx - 1) + (niveau - q[idx - 1]) / (q[idx] - q[idx - 1]), idx

        p10, i10 = position(0.1, 0)
        p90, _ = position(0.9, i10)
        largeurs.append((p90 - p10) * espacement_mm)
    return np.array(largeurs)


def lire_plage_biais(texte):
    """Plages (`max − min` du log du champ) de chaque pile, lues dans la sortie de `diagnose_the_estimated_bias_range` (lignes `BIAIS pile min max etendue reduction`)."""
    return [float(m.group(1)) for m in re.finditer(r"^BIAIS \d+ \S+ \S+ (\S+) \S+$", texte, re.M)]


def meilleur_parametre(grille, mesures, cible):
    """Valeur de `grille` dont la mesure (même ordre dans `mesures`) minimise `|ln(mesure / cible)|`."""
    ecarts = [abs(np.log(m / cible)) for m in mesures]
    return grille[int(np.argmin(ecarts))]
