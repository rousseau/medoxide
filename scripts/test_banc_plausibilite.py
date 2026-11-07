"""Tests de `banc_plausibilite.py` : oracles calculés à la main sur des piles synthétiques. Lancer : `python3 -B -m unittest scripts/test_banc_plausibilite.py -v`."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_plausibilite as bp


def pile_synthetique(graine=0, nz=10):
    """Pile 60 × 60 × nz : tissu (400) au centre, tissu maternel (300) dans les coins haut-gauche, bruit de |N(0, 3)| dans les trois autres coins et le reste de l'air."""
    rng = np.random.default_rng(graine)
    s = np.abs(rng.normal(0, 3.0, (60, 60, nz)))
    s[0:20, 0:20, :] = 300.0
    s[15:45, 15:45, :] = 400.0
    masque = np.zeros(s.shape, dtype=bool)
    masque[15:45, 15:45, :] = True
    return s, masque


class TestBruit(unittest.TestCase):
    def test_les_carres_de_tissu_maternel_sont_exclus(self):
        s, m = pile_synthetique()
        carres = bp.choisir_carres(s, bp.moyenne_tissu(s, m))
        self.assertEqual(len(carres), 3 * 10)  # trois coins d'air par coupe, dix coupes
        self.assertTrue(all((i0, j0) != (0, 0) for _, i0, j0 in carres))

    def test_rapport_contre_calcul_direct(self):
        s, m = pile_synthetique(1)
        carres = bp.choisir_carres(s, 400.0)
        valeurs = []
        for k in range(10):
            for (i0, j0) in ((0, 48), (48, 0), (48, 48)):
                valeurs.extend(s[i0 : i0 + 12, j0 : j0 + 12, k].ravel().tolist())
        attendu = np.std(np.array(valeurs)) / 400.0
        self.assertAlmostEqual(bp.rapport_bruit(s, m, carres), attendu, places=12)
        # |N(0, σ)| : écart type σ √(1 − 2/π)
        self.assertAlmostEqual(bp.rapport_bruit(s, m, carres) * 400.0, 3.0 * np.sqrt(1 - 2 / np.pi), delta=0.1)

    def test_pas_assez_de_carres_d_air(self):
        s, m = pile_synthetique()
        s[:, :, :] = np.maximum(s, 100.0)  # plus d'air nulle part
        with self.assertRaises(ValueError):
            bp.choisir_carres(s, 400.0)


class TestBord(unittest.TestCase):
    def rampe(self, largeur, debut=4, nz=14):
        """Marche en rampe linéaire de 0 à 1000 sur `largeur` coupes à partir de la coupe `debut` (valeurs exactement représentables : pas d'égalité douteuse) ; masque plein."""
        p = np.zeros((6, 6, nz))
        for k in range(nz):
            p[:, :, k] = 1000.0 * np.clip((k - debut) / largeur, 0.0, 1.0)
        return p, np.ones(p.shape, dtype=bool)

    def test_largeur_d_une_rampe_connue(self):
        p, m = self.rampe(4)
        emp = bp.emplacements_de_bord(p, m, 500.0, n_max=10)
        self.assertGreater(len(emp), 0)
        self.assertTrue(all(k == 5 for _, _, k in emp))  # seul maximum local strict : le premier point de la rampe où le gradient est maximal
        # largeur 10–90 % d'une rampe de 4 coupes : 0,8 × 4 = 3,2 coupes, soit 11,2 mm à 3,5 mm
        self.assertTrue(np.allclose(bp.largeur_de_bord(p, emp, 3.5), 11.2, atol=1e-9))

    def test_une_rampe_deux_fois_plus_etroite_donne_une_largeur_moitie(self):
        p, m = self.rampe(2)
        emp = bp.emplacements_de_bord(p, m, 500.0, n_max=10)
        self.assertTrue(all(k == 5 for _, _, k in emp))
        w = bp.largeur_de_bord(p, emp, 1.0)
        self.assertTrue(np.allclose(w, 1.6, atol=1e-9), w[:3])  # 0,8 × 2 coupes

    def test_orientation_indifferente(self):
        p, m = self.rampe(4)
        inverse = p[:, :, ::-1].copy()
        emp = bp.emplacements_de_bord(inverse, m, 500.0, n_max=5)
        self.assertGreater(len(emp), 0)
        self.assertTrue(np.allclose(bp.largeur_de_bord(inverse, emp, 3.5), 11.2))

    def test_contraste_insuffisant_exclu(self):
        p, m = self.rampe(4)
        self.assertEqual(len(bp.emplacements_de_bord(p, m, 1e6, n_max=10)), 0)


class TestDivers(unittest.TestCase):
    def test_lecture_de_la_plage_du_biais(self):
        texte = "bruit\nBIAIS 0 -0.1000 0.2000 0.3000 0.0500\nBIAIS 1 -0.2 0.4 0.6 0.1\nautre\n"
        self.assertEqual(bp.lire_plage_biais(texte), [0.3, 0.6])

    def test_meilleur_parametre(self):
        self.assertEqual(bp.meilleur_parametre([0.5, 1.0, 2.0], [0.4, 1.1, 5.0], 1.0), 1.0)
        # |ln| : 0,5 contre 2,0 pour une cible de 1 sont à égale distance ; la mesure 0,9 est plus proche que 1,3
        self.assertEqual(bp.meilleur_parametre([1, 2], [0.9, 1.3], 1.0), 1)


class TestCalibrage(unittest.TestCase):
    def test_bornes_du_critere(self):
        import banc_calibrage as bc

        self.assertTrue(bc.dans_les_bornes(0.8) and bc.dans_les_bornes(1.25) and bc.dans_les_bornes(1.0))
        self.assertFalse(bc.dans_les_bornes(0.79) or bc.dans_les_bornes(1.26))

    def test_rapports_mediane_sur_les_trois_piles(self):
        import banc_calibrage as bc

        reel = [{"bruit": 0.01, "largeur": 10.0}] * 3
        mes = [{"bruit": 0.02, "largeur": 5.0}, {"bruit": 0.01, "largeur": 10.0}, {"bruit": 0.03, "largeur": 20.0}]
        r = bc.rapports(reel, mes, [0.2, 0.2, 0.2], [0.1, 0.3, 0.4])
        self.assertAlmostEqual(r["bruit"], 2.0)  # médiane de (2, 1, 3)
        self.assertAlmostEqual(r["largeur"], 1.0)  # médiane de (0,5 ; 1 ; 2)
        self.assertAlmostEqual(r["biais"], 0.3 / 0.2)


if __name__ == "__main__":
    unittest.main()
