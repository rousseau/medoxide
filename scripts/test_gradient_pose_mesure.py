"""Tests de `gradient_pose_mesure.py` : python3 -B -m unittest scripts/test_gradient_pose_mesure.py -v (depuis la racine du dépôt)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradient_pose_mesure as m  # noqa: E402


class TestMesure(unittest.TestCase):
    def test_dice_identique_et_disjoint(self):
        a = np.zeros((6, 6, 6), dtype=np.int16)
        a[:3, :, :] = 3
        a[3:, :, :] = 4
        d = m.dice_par_etiquette(a, a, labels=[3, 4])
        self.assertEqual(d, {3: 1.0, 4: 1.0})
        b = np.where(a == 3, 4, 3).astype(np.int16)
        self.assertEqual(m.dice_par_etiquette(a, b, labels=[3, 4]), {3: 0.0, 4: 0.0})

    def test_dice_calcule_a_la_main(self):
        a = np.zeros((4, 1, 1), dtype=np.int16)
        b = np.zeros((4, 1, 1), dtype=np.int16)
        a[:3] = 3  # 3 voxels
        b[1:3] = 3  # 2 voxels, tous dans a : Dice = 2·2 / (3 + 2) = 0,8
        self.assertAlmostEqual(m.dice_par_etiquette(a, b, labels=[3])[3], 0.8)

    def test_etiquette_vide_partout_ecartee_vide_d_un_cote_comptee(self):
        a = np.zeros((4, 1, 1), dtype=np.int16)
        b = np.zeros((4, 1, 1), dtype=np.int16)
        a[:2] = 5
        d = m.dice_par_etiquette(a, b, labels=[5, 6])  # 5 : vide dans b seulement ; 6 : vide dans les deux
        self.assertEqual(d, {5: 0.0})

    def test_nettete_croit_avec_le_contraste_et_est_invariante_a_l_echelle(self):
        rng = np.random.default_rng(0)
        x = np.indices((24, 24, 24)).sum(axis=0) % 7 + 10 * rng.random((24, 24, 24)) + 50
        s = m.nettete(x)
        self.assertAlmostEqual(m.nettete(3.0 * x), s, places=9)
        from scipy.ndimage import gaussian_filter

        self.assertLess(m.nettete(gaussian_filter(x, 2.0)), s)  # un volume plus flou est moins net

    def test_verdicts_aux_seuils(self):
        c1 = (100, 0.0, 0.0, 0.05, 0.005, True)
        v = m.verdicts(c1, 1.0, 1.0299, (0.8, 0.8), (0.7901, 0.8), 0.975)
        self.assertTrue(all(t for t, _ in v.values()), v)
        self.assertFalse(m.verdicts(c1, 1.0, 1.0301, (0.8, 0.8), (0.8, 0.8), 0.99)["c2"][0])
        self.assertFalse(m.verdicts(c1, 1.0, 1.0, (0.8, 0.8), (0.7899, 0.8), 0.99)["c3"][0])
        self.assertFalse(m.verdicts(c1, 1.0, 1.0, (0.8, 0.8), (0.8, 0.7899), 0.99)["c3"][0])
        self.assertFalse(m.verdicts(c1, 1.0, 1.0, (0.8, 0.8), (0.8, 0.8), 0.9749)["c4"][0])
        self.assertFalse(m.verdicts((100, 0, 0, 0.06, 0, False), 1.0, 1.0, (0.8, 0.8), (0.8, 0.8), 0.99)["c1"][0])


if __name__ == "__main__":
    unittest.main()
