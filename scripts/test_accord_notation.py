"""Test de `accord_notation.py` : oracles calculés à la main (script d'analyse ponctuel : pas de mutations). Lancer : `python3 -B -m unittest scripts/test_accord_notation.py -v`."""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accord_notation as a  # noqa: E402


class TestSpearman(unittest.TestCase):
    def test_rangs_egalites(self):
        self.assertEqual(a.rangs([3, 1, 3, 2]), [3.5, 1.0, 3.5, 2.0])

    def test_monotone_et_inverse(self):
        self.assertAlmostEqual(a.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(a.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_valeur_a_la_main(self):
        # rangs (1,2,3,4,5) et (2,1,4,3,5) : somme des d^2 = 1+1+1+1+0 = 4 ; rho = 1 - 6*4/(5*24) = 0.8
        self.assertAlmostEqual(a.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8)

    def test_constante_donne_nan(self):
        self.assertTrue(math.isnan(a.spearman([1, 2, 3], [4, 4, 4])))


class TestCablage(unittest.TestCase):
    def test_bout_en_bout(self):
        with tempfile.TemporaryDirectory() as d:
            def ecrire(nom, texte):
                with open(os.path.join(d, nom), "w", encoding="utf-8") as f:
                    f.write(texte)
                return os.path.join(d, nom)
            corr = ecrire("c.tsv", "identifiant\tposition\tsujet\tmethode\n1\t0\tS1\ts2s\n2\t1\tS1\tm2\n3\t2\tS1\tm3\n")
            grille = ecrire("g.tsv", "identifiant\tqualite_globale\n1\t4\n2\t2\n3\t1\n")
            scores = ecrire("s.tsv", "sujet\tmethode\tgraine\tscore\nS1\ts2s\t1\t0.9\nS1\ts2s\t2\t0.1\nS1\tm2\t1\t0.5\nS1\tm3\t1\t0.2\n")
            res, moyen, nd, par_m = a.accord(a.lire_tsv(grille), a.lire_tsv(corr), a.lire_tsv(scores))
            self.assertEqual(res["S1"][0], 3)
            self.assertAlmostEqual(res["S1"][1], 1.0)  # la graine 2 de s2s (0.1) ne compte pas
            self.assertEqual(nd, 1)
            self.assertEqual(par_m["s2s"], 4.0)
            res2, _, _, _ = a.accord(a.lire_tsv(grille), a.lire_tsv(corr), a.lire_tsv(scores), exclure={"3"})
            self.assertEqual(res2["S1"][0], 2)
            self.assertTrue(math.isnan(res2["S1"][1]))  # n < 3


if __name__ == "__main__":
    unittest.main()
