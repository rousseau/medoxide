"""Analyse de F4.0.2 (docs/ROADMAP.md) : le juge hors échantillon classe-t-il les variantes comme la vérité du banc ?

Fonctions pures, sans entrées-sorties. Un « tableau » est un dict `{(sujet, repli): {variante: (juge, psnr)}}`.

- `spearman(a, b)` : corrélation de rang de Spearman (rangs moyens pour les égalités) ;
- `rho_sujet_repli` : Spearman juge/PSNR sur les variantes d'un (sujet, repli) ;
- `rho_par_sujet` : moyenne des replis d'un sujet ; `rho_moyen` : moyenne des sujets (unité statistique : le sujet) ;
- `meilleur_alpha` : variante α maximisant la grandeur choisie (juge ou PSNR), parmi les variantes α.
"""

import numpy as np
from scipy.stats import rankdata

ENTETE = "entete"
ALPHAS = ["a003", "a005", "a010", "a030"]  # α = 0,03 ; 0,05 (le défaut) ; 0,1 ; 0,3
VARIANTES = [ENTETE] + ALPHAS


def spearman(a, b):
    """Corrélation de Spearman = Pearson des rangs (rangs moyens pour les égalités) ; `nan` si l'une des deux listes est constante."""
    ra, rb = rankdata(a), rankdata(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rho_sujet_repli(mesures, variantes):
    """Spearman entre juge et PSNR sur `variantes` pour un (sujet, repli) : `mesures[variante] = (juge, psnr)`."""
    return spearman([mesures[v][0] for v in variantes], [mesures[v][1] for v in variantes])


def rho_par_sujet(tableau, variantes):
    """`{sujet: moyenne sur ses replis de rho_sujet_repli}` (un repli de Spearman indéfini fait échouer le calcul : on ne masque rien)."""
    par = {}
    for (sujet, _), mesures in tableau.items():
        par.setdefault(sujet, []).append(rho_sujet_repli(mesures, variantes))
    if any(np.isnan(r) for rs in par.values() for r in rs):
        raise ValueError("Spearman indéfini (juge ou PSNR constant sur un repli)")
    return {s: float(np.mean(rs)) for s, rs in par.items()}


def rho_moyen(tableau, variantes):
    """Moyenne sur les sujets de `rho_par_sujet` (chaque sujet compte pour un, quel que soit son nombre de replis)."""
    return float(np.mean(list(rho_par_sujet(tableau, variantes).values())))


def meilleur_alpha(tableau, sujet, colonne):
    """Variante α qui maximise `colonne` (0 : juge, 1 : PSNR) pour `sujet`, moyenne des replis ; en cas d'égalité, la première de `ALPHAS`."""
    moyennes = {a: np.mean([m[a][colonne] for (s, _), m in tableau.items() if s == sujet]) for a in ALPHAS}
    return max(ALPHAS, key=lambda a: moyennes[a])


def lecture(rho_5, rho_alpha):
    """Niveau de décision de Q1 d'après `ρ̄` (5 variantes) et `ρ̄_α` (4 variantes α), règle écrite d'avance (F4.0.2) : renvoie le texte de la lecture."""
    if rho_5 >= 0.8:
        if rho_alpha < 0.5:
            return "Q1 ne distingue que les écarts grossiers (en-tête contre s2s) : il ne décide pas seul entre variantes proches"
        return "Q1 garde son rôle"
    if rho_5 >= 0.5:
        return "Q1 ne décide plus seul : il faut aussi la vérité terrain (Q6)"
    return "Q1 sort de la règle de décision"
