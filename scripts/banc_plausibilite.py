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
PART_DANS_LA_DISPERSION = 0.8  # (v′), amendement 3 : au moins 80 % des valeurs simulées dans [min ; max] du réel


def moyenne_tissu(pile, masque):
    """Intensité moyenne de la pile dans le masque de cerveau."""
    return float(pile[masque].mean())


def choisir_carres(pile, mu):
    """Carrés d'air : pour chaque coupe `k`, chacun des quatre coins `CARRE × CARRE` dont le maximum est inférieur à `SEUIL_AIR · mu`.

    Renvoie une liste de `(k, i0, j0)` (coin supérieur gauche du carré), éventuellement vide : le seuil de `MIN_CARRES` s'applique au SUJET (voir `exiger_carres`)."""
    ni, nj, nk = pile.shape
    carres = []
    for k in range(nk):
        for i0 in (0, ni - CARRE):
            for j0 in (0, nj - CARRE):
                if pile[i0 : i0 + CARRE, j0 : j0 + CARRE, k].max() < SEUIL_AIR * mu:
                    carres.append((k, i0, j0))
    return carres


def exiger_carres(carres_par_pile):
    """Vérifie que le sujet (toutes ses piles) a au moins `MIN_CARRES` carrés d'air ; sinon `ValueError` (pas de mesure du bruit : arrêt prévu par le protocole)."""
    total = sum(len(c) for c in carres_par_pile)
    if total < MIN_CARRES:
        raise ValueError(f"{total} carrés d'air seulement pour le sujet (minimum {MIN_CARRES}) : pas de mesure du bruit")


def sigma_air(pile, carres):
    """Écart type des pixels de tous les carrés d'air, regroupés."""
    valeurs = np.concatenate([pile[i0 : i0 + CARRE, j0 : j0 + CARRE, k].ravel() for k, i0, j0 in carres])
    return float(valeurs.std())


def rapport_bruit(pile, masque, carres):
    """`σ_air / μ_tissu` d'UNE pile, les carrés d'air étant ceux choisis sur la pile réelle (la mesure du protocole est `rapport_bruit_sujet`)."""
    return sigma_air(pile, carres) / moyenne_tissu(pile, masque)


def rapport_bruit_sujet(piles, masques, carres_par_pile):
    """`R_bruit` d'un sujet : écart type regroupé des pixels des carrés d'air de ses piles, chaque pixel divisé par `μ_tissu` de SA pile (moyenne dans le masque
    de cette pile). `carres_par_pile[n]` : carrés choisis sur la pile réelle `n`, réutilisés sur la pile simulée `n`."""
    exiger_carres(carres_par_pile)
    valeurs = []
    for pile, masque, carres in zip(piles, masques, carres_par_pile):
        mu = moyenne_tissu(pile, masque)
        valeurs.extend(pile[i0 : i0 + CARRE, j0 : j0 + CARRE, k].ravel() / mu for k, i0, j0 in carres)
    return float(np.concatenate(valeurs).std())


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


def _autres_axes(axe):
    """Les deux autres axes d'un tableau 3D, dans l'ordre croissant."""
    return [a for a in range(3) if a != axe]


def emplacements_de_bord_axe(pile, masque, mu, axe, n_max=N_EMPLACEMENTS):
    """Comme `emplacements_de_bord`, mais le long de l'axe `axe` du tableau (0 ou 1 : DANS le plan de la coupe ; 2 : entre coupes). Mêmes règles,
    mêmes seuils ; les emplacements sont rendus en indices du tableau d'origine."""
    deplace = np.moveaxis(pile, axe, 2)
    emp = emplacements_de_bord(deplace, np.moveaxis(masque, axe, 2), mu, n_max)
    sortie = np.empty_like(emp)
    sortie[:, _autres_axes(axe)] = emp[:, :2]
    sortie[:, axe] = emp[:, 2]
    return sortie


def largeur_de_bord_axe(pile, emplacements, pas_mm, axe):
    """Comme `largeur_de_bord`, le long de l'axe `axe` (emplacements en indices du tableau d'origine, `pas_mm` : pas des voxels sur cet axe)."""
    emp = np.asarray(emplacements, dtype=int).reshape(-1, 3)
    return largeur_de_bord(np.moveaxis(pile, axe, 2), emp[:, _autres_axes(axe) + [axe]], pas_mm)


def lire_plage_biais(texte):
    """Plages (`max − min` du log du champ) de chaque pile, lues dans la sortie de `diagnose_the_estimated_bias_range` (lignes `BIAIS pile min max etendue reduction`)."""
    return [float(m.group(1)) for m in re.finditer(r"^BIAIS \d+ \S+ \S+ (\S+) \S+$", texte, re.M)]


def meilleur_parametre(grille, mesures, cible):
    """Valeur de `grille` dont la mesure (même ordre dans `mesures`) minimise `|ln(mesure / cible)|`."""
    ecarts = [abs(np.log(m / cible)) for m in mesures]
    return grille[int(np.argmin(ecarts))]


def seuil_de_dispersion(n):
    """Nombre minimal de valeurs simulées dans l'intervalle réel pour `n` valeurs : 80 % arrondis VERS LE HAUT (calcul entier : `⌈4n/5⌉`)."""
    return -(-4 * n // 5)


def grandeurs_du_sujet(mesures, plages):
    """Valeurs des quatre grandeurs de (v′) pour un sujet, à partir des trois piles (`mesures[k]` : `bruit_sujet`, `largeur`, `largeur_plan` [lignes, colonnes])
    et des plages du biais estimé par pile : bruit (une valeur par sujet), largeur entre coupes (par pile), plage du biais (par pile), largeur dans le plan
    (par pile et par direction, les deux directions comptées ensemble)."""
    return {
        "bruit": [float(mesures[0]["bruit_sujet"])],
        "largeur": [float(m["largeur"]) for m in mesures],
        "biais": [float(x) for x in plages],
        "plan": [float(v) for m in mesures for v in m["largeur_plan"]],
    }


def verdict_dispersion(simules, reels):
    """(v′) : `simules` et `reels` sont des listes de `grandeurs_du_sujet` (un élément par sujet). Pour chaque grandeur, compte les valeurs simulées dans
    l'intervalle fermé [min ; max] des valeurs RÉELLES (tous sujets de `reels` réunis) ; tenu si ce nombre atteint `seuil_de_dispersion(n)`.
    Renvoie `(par_grandeur, tenu)`, `par_grandeur[nom] = dict(dans, n, seuil, min, max, tenu)`."""
    sortie = {}
    for nom in ("bruit", "largeur", "biais", "plan"):
        sim = [v for g in simules for v in g[nom]]
        ref = [v for g in reels for v in g[nom]]
        bas, haut = min(ref), max(ref)
        dans = sum(1 for v in sim if bas <= v <= haut)
        seuil = seuil_de_dispersion(len(sim))
        sortie[nom] = {"dans": dans, "n": len(sim), "seuil": seuil, "min": bas, "max": haut, "tenu": dans >= seuil}
    return sortie, all(v["tenu"] for v in sortie.values())
