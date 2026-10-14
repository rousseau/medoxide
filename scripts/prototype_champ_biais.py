"""Référence Python de l'estimation d'un champ de biais par stack, SANS volume, à partir des intersections de coupes (étape 1 de la brique « biais explicite »).

Modèle : l'image du stack s est le tissu multiplié par un champ lisse b_s, fixe dans le repère du SCANNER : il dépend de la position NOMINALE (en-tête) du pixel, pas de l'anatomie qui a bougé. Pour deux coupes de stacks différents qui se croisent (aux poses données), un point de leur droite
d'intersection est le même tissu ; ses deux pixels ont des positions nominales x1 et x2 différentes, donc
        log y1 − log y2 = φ(x1)·c_s1 − φ(x2)·c_s2           (φ : base polynomiale de degré 3 des coordonnées nominales normalisées ; c_s : les coefficients du stack s).
Toutes les paires donnent un système LINÉAIRE aux moindres carrés (avec une petite régularisation) en les coefficients des champs ; le terme constant du stack 0 est fixé à 0 (seuls les rapports de gain sont identifiables). Aucun volume intermédiaire.

Ce script mesure l'estimateur sur le simulateur (`make_pyrecon_benchmark.py --biais-pct`), dont le champ vrai est recalculable (graines fixées), avec les poses VRAIES (pour isoler l'estimation du biais de celle du mouvement) :
  - corrélation, par stack, entre le log du champ estimé et celui du vrai champ aux positions nominales des pixels de masque (le terme constant est sans effet sur une corrélation) ;
  - rapports de gain entre stacks (estimés et vrais) ;
  - sur un jeu SANS biais : écart type du log du champ estimé (doit être proche de 0).

Usage : python scripts/prototype_champ_biais.py --jeu data/atlas/sim_pyrecon/STA31_mvt4_b5_biais30 --biais-pct 30   (poses vraies : `poses.tsv` du jeu)
"""

import argparse
import itertools

import nibabel as nib
import numpy as np
from scipy import ndimage

import intersection_consistency as ic

DEGRE = 3
ECHELLE_MM = 60.0


def exposants(degre=DEGRE):
    """Exposants (a, b, c) des monômes u^a v^b w^c de degré total <= `degre` (le premier est la constante)."""
    return sorted([e for e in itertools.product(range(degre + 1), repeat=3) if sum(e) <= degre], key=lambda e: (sum(e), e))


EXPOSANTS = exposants()


def base(x, centre):
    """φ(x) pour des positions nominales `x` (n, 3) : matrice (n, nb de monômes)."""
    u = (x - centre) / ECHELLE_MM
    return np.stack([u[:, 0] ** a * u[:, 1] ** b * u[:, 2] ** c for a, b, c in EXPOSANTS], axis=1)


def echantillons(jeu, poses, pas_mm=0.5, portee_mm=150.0, angle_min_deg=30.0, par_paire=40, seuils=None, graine=0):
    """Pour chaque paire de coupes qui se croisent : jusqu'à `par_paire` points de la droite d'intersection, dans les deux masques et au-dessus des seuils d'intensité. Rend une liste de (s1, x1_nominale, y1, s2, x2_nominale, y2) par paire."""
    rng = np.random.default_rng(graine)
    coupes = sorted(poses)
    prep = ic.preparer(jeu, poses, coupes)
    t = np.arange(-portee_mm, portee_mm + pas_mm, pas_mm)
    sin_min = np.sin(np.radians(angle_min_deg))
    sortie = []
    for a in range(len(coupes)):
        for b in range(a + 1, len(coupes)):
            (s1, k1), (s2, k2) = coupes[a], coupes[b]
            if s1 == s2:
                continue
            _, inv1, _, n1, h1 = prep[(s1, k1)]
            _, inv2, _, n2, h2 = prep[(s2, k2)]
            d = np.cross(n1, n2)
            nd = np.linalg.norm(d)
            if nd < sin_min:
                continue
            x0 = (h1 * np.cross(n2, d) + h2 * np.cross(d, n1)) / (nd * nd)
            ph = np.hstack([x0[None, :] + t[:, None] * (d / nd)[None, :], np.ones((len(t), 1))]).T
            ok = np.ones(len(t), bool)
            vals, nominales = [], []
            for (s, k, inv) in ((s1, k1, inv1), (s2, k2, inv2)):
                idx = inv @ ph
                nx, ny = jeu.data[s].shape[:2]
                i, j = idx[0], idx[1]
                dedans = (i >= 0) & (i <= nx - 1) & (j >= 0) & (j <= ny - 1)
                ii, jj = np.clip(np.round(i).astype(int), 0, nx - 1), np.clip(np.round(j).astype(int), 0, ny - 1)
                ok &= dedans & jeu.mask[s][ii, jj, k]
                y = ndimage.map_coordinates(jeu.data[s][:, :, k], [i, j], order=1, mode="nearest")
                ok &= y > seuils[s]
                vals.append(y)
                nominales.append((jeu.aff[s] @ np.stack([i, j, np.full(len(t), k), np.ones(len(t))]))[:3].T)  # position NOMINALE (en-tête) des pixels échantillonnés
            if ok.sum() >= 20:
                choix = np.flatnonzero(ok)
                choix = rng.choice(choix, min(par_paire, len(choix)), replace=False)
                sortie.append((s1, nominales[0][choix], vals[0][choix], s2, nominales[1][choix], vals[1][choix]))
    return sortie


def estimer_champs(jeu, poses, lam=1e-3, **options):
    """Coefficients (n_stacks, nb de monômes) des logs de champs de biais, terme constant du stack 0 nul ; et le centre de normalisation des positions."""
    ns = len(jeu.data)
    seuils = [0.1 * np.percentile(jeu.data[s][jeu.mask[s]], 99) for s in range(ns)]
    pts = np.vstack([(jeu.aff[s] @ np.vstack([np.argwhere(jeu.mask[s]).T, np.ones(int(jeu.mask[s].sum()))]))[:3].T for s in range(ns)])
    centre = pts.mean(axis=0)
    paires = echantillons(jeu, poses, seuils=seuils, **options)
    nb = len(EXPOSANTS)
    lignes, cibles = [], []
    for s1, x1, y1, s2, x2, y2 in paires:
        a = np.zeros((len(y1), ns * nb))
        a[:, s1 * nb:(s1 + 1) * nb] += base(x1, centre)
        a[:, s2 * nb:(s2 + 1) * nb] -= base(x2, centre)
        lignes.append(a)
        cibles.append(np.log(y1) - np.log(y2))
    A, y = np.vstack(lignes), np.concatenate(cibles)
    libres = [c for c in range(ns * nb) if c != 0]  # le terme constant du stack 0 est fixé à 0
    reg = lam * np.sqrt(len(y)) * np.ones(len(libres))
    reg[[i for i, c in enumerate(libres) if c % nb == 0]] = 0.0  # pas de régularisation des termes constants
    sol = np.linalg.lstsq(np.vstack([A[:, libres], np.diag(reg)]), np.concatenate([y, np.zeros(len(libres))]), rcond=None)[0]
    coef = np.zeros(ns * nb)
    coef[libres] = sol
    return coef.reshape(ns, nb), centre, len(paires), len(y)


def log_champ_estime(coef, centre, jeu, s):
    """Log du champ estimé aux positions nominales des pixels de masque du stack s (et ces positions)."""
    x = (jeu.aff[s] @ np.vstack([np.argwhere(jeu.mask[s]).T, np.ones(int(jeu.mask[s].sum()))]))[:3].T
    return base(x, centre) @ coef[s], x


def log_champ_vrai(x, n, biais_pct):
    """Log du champ du simulateur (mêmes graines et mêmes tirages que `make_pyrecon_benchmark.py`) aux positions nominales `x` du stack n."""
    rb = np.random.default_rng(31000 + 17 * n)
    champ = np.ones(len(x))
    for _ in range(4):
        direction = rb.normal(size=3)
        direction /= np.linalg.norm(direction)
        champ += (biais_pct / 100.0) / 2.0 * np.cos(2 * np.pi * (x @ direction) / rb.uniform(100.0, 200.0) + rb.uniform(0, 2 * np.pi))
    gain = 1.0 + (biais_pct / 100.0) * rb.uniform(-1.0, 1.0)
    return np.log(gain * champ)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jeu", required=True)
    parser.add_argument("--biais-pct", type=int, default=0)
    args = parser.parse_args()
    ns = 3
    jeu = ic.Jeu([f"{args.jeu}/stack{n}.nii.gz" for n in range(ns)], [f"{args.jeu}/stack{n}_mask.nii.gz" for n in range(ns)])
    poses = ic.lire_poses(f"{args.jeu}/poses.tsv")
    coef, centre, n_paires, n_lignes = estimer_champs(jeu, poses)
    print(f"{n_paires} paires, {n_lignes} équations, {coef.size - 1} inconnues")
    gains_estimes, gains_vrais = [], []
    for s in range(ns):
        est, x = log_champ_estime(coef, centre, jeu, s)
        if args.biais_pct:
            vrai = log_champ_vrai(x, s, args.biais_pct)
            corr = float(np.corrcoef(est, vrai)[0, 1])
            erreur = float(np.std((est - est.mean()) - (vrai - vrai.mean())))
            print(f"stack {s} : corrélation du log du champ avec le vrai {corr:.3f} ; écart type du log (estimé {est.std():.3f}, vrai {vrai.std():.3f}) ; écart type de l'erreur {erreur:.3f}")
            gains_estimes.append(est.mean())
            gains_vrais.append(vrai.mean())
        else:
            print(f"stack {s} : écart type du log du champ estimé {est.std():.4f} (moyenne {est.mean():+.4f})")
            gains_estimes.append(est.mean())
    print("rapports de gain par rapport au stack 0 (log) : estimés", np.round(np.array(gains_estimes) - gains_estimes[0], 3), end="")
    print(" ; vrais", np.round(np.array(gains_vrais) - gains_vrais[0], 3) if gains_vrais else "(sans biais : 0 attendu)")
