"""Tests de `banc_juge_analyse.py` : oracles calculés à la main. Lancer : `python3 -B -m unittest scripts/test_banc_juge_analyse.py -v`."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_juge_analyse as bj

V = bj.VARIANTES  # entete, a003, a005, a010, a030


def repli(juges, psnrs):
    return {v: (j, p) for v, j, p in zip(V, juges, psnrs)}


class TestSpearman(unittest.TestCase):
    def test_classement_identique_et_inverse(self):
        self.assertAlmostEqual(bj.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(bj.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_valeur_calculee_a_la_main(self):
        # rangs de a : 1 2 3 4 5 ; rangs de b : 2 1 4 3 5 -> d = (−1, 1, −1, 1, 0), Σd² = 4, ρ = 1 − 6·4 / (5·24) = 0,8
        self.assertAlmostEqual(bj.spearman([1, 2, 3, 4, 5], [20, 10, 40, 30, 50]), 0.8)

    def test_egalites_par_rangs_moyens(self):
        # a = (1, 2, 2, 3) -> rangs (1 ; 2,5 ; 2,5 ; 4) ; b = (1, 2, 3, 4) -> rangs (1, 2, 3, 4) ; Pearson : covariance 2,25... = 0,9487
        self.assertAlmostEqual(bj.spearman([1, 2, 2, 3], [1, 2, 3, 4]), 0.9486832980505138)

    def test_seul_le_rang_compte(self):
        """Transformation monotone non linéaire : le classement est identique, donc ρ = 1 (la corrélation de Pearson des valeurs brutes vaudrait ≈ 0,74)."""
        self.assertAlmostEqual(bj.spearman([1, 2, 3, 4], [1, 2, 3, 100]), 1.0)
        self.assertAlmostEqual(bj.spearman([0.001, 0.002, 0.9, 0.95], [5, 6, 7, 8]), 1.0)

    def test_constante_donne_nan(self):
        self.assertTrue(math.isnan(bj.spearman([1, 1, 1], [1, 2, 3])))


class TestAgregation(unittest.TestCase):
    def tableau(self):
        # sujet A : replis 0 (ρ = 1) et 1 (ρ = −1) -> 0 ; sujet B : un repli (ρ = 1) -> 1 ; ρ̄ = (0 + 1) / 2 = 0,5 (et NON (1 − 1 + 1) / 3 = 1/3)
        croissant = repli([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        decroissant = repli([5, 4, 3, 2, 1], [10, 20, 30, 40, 50])
        return {("A", 0): croissant, ("A", 1): decroissant, ("B", 0): croissant}

    def test_moyenne_des_replis_puis_des_sujets(self):
        par = bj.rho_par_sujet(self.tableau(), V)
        self.assertAlmostEqual(par["A"], 0.0)
        self.assertAlmostEqual(par["B"], 1.0)
        self.assertAlmostEqual(bj.rho_moyen(self.tableau(), V), 0.5)

    def test_variantes_alpha_seules(self):
        # entête très bas pour le juge ET le PSNR (gonfle ρ sur 5 variantes) ; les 4 α : juge inversé du PSNR
        t = {("A", 0): repli([0.1, 0.9, 0.8, 0.7, 0.6], [10, 40, 30, 20, 25])}
        self.assertAlmostEqual(bj.rho_moyen(t, V), bj.spearman([0.1, 0.9, 0.8, 0.7, 0.6], [10, 40, 30, 20, 25]))
        self.assertAlmostEqual(bj.rho_moyen(t, bj.ALPHAS), bj.spearman([0.9, 0.8, 0.7, 0.6], [40, 30, 20, 25]))
        self.assertGreater(bj.rho_moyen(t, V), bj.rho_moyen(t, bj.ALPHAS))

    def test_spearman_indefini_leve_une_erreur(self):
        with self.assertRaises(ValueError):
            bj.rho_par_sujet({("A", 0): repli([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])}, V)

    def test_meilleur_alpha_par_colonne(self):
        # juge maximal pour a010, PSNR maximal pour a003 ; l'en-tête n'est jamais candidat
        t = {("A", 0): repli([9, 0.1, 0.2, 0.9, 0.3], [9, 30, 20, 10, 15])}
        self.assertEqual(bj.meilleur_alpha(t, "A", 0), "a010")
        self.assertEqual(bj.meilleur_alpha(t, "A", 1), "a003")


class TestLecture(unittest.TestCase):
    def test_seuils_de_la_lecture(self):
        self.assertEqual(bj.lecture(0.8, 0.5), "Q1 garde son rôle")
        self.assertIn("grossiers", bj.lecture(0.85, 0.49))
        self.assertIn("Q6", bj.lecture(0.79, 0.9))
        self.assertIn("Q6", bj.lecture(0.5, 0.0))
        self.assertIn("sort", bj.lecture(0.49, 0.9))


if __name__ == "__main__":
    unittest.main()
