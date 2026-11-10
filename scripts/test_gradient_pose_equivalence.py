"""Tests de `gradient_pose_equivalence.py` : python3 -B -m unittest scripts/test_gradient_pose_equivalence.py -v (depuis la racine du dépôt)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradient_pose_equivalence as g  # noqa: E402


def translation(x, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def rotation_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


class TestEquivalence(unittest.TestCase):
    def test_translation_connue(self):
        pix = {("s", 0): np.random.default_rng(0).normal(size=(50, 3)) * 30}
        e = g.deplacements({("s", 0): np.eye(4)}, {("s", 0): translation(0.3, 0.4)}, pix)
        self.assertAlmostEqual(e[("s", 0)], 0.5, places=12)

    def test_rotation_connue(self):
        # points sur l'axe x à 10 mm : une rotation de 1° autour de z déplace chacun de 10·2·sin(0,5°)
        pix = {("s", 0): np.array([[10.0, 0.0, 0.0]] * 4)}
        e = g.deplacements({("s", 0): np.eye(4)}, {("s", 0): rotation_z(1.0)}, pix)
        self.assertAlmostEqual(e[("s", 0)], 20 * np.sin(np.radians(0.5)), places=9)

    def test_deplacement_ne_depend_pas_du_sens(self):
        pix = {("s", 0): np.random.default_rng(1).normal(size=(30, 3)) * 20}
        a, b = {("s", 0): rotation_z(2.0)}, {("s", 0): translation(1.0, 2.0)}
        self.assertAlmostEqual(g.deplacements(a, b, pix)[("s", 0)], g.deplacements(b, a, pix)[("s", 0)], places=12)

    def test_coupe_absente_ou_sans_pixel_ignoree(self):
        pix = {("s", 0): np.zeros((3, 3)), ("s", 1): np.zeros((0, 3))}
        e = g.deplacements({("s", 0): np.eye(4), ("s", 1): np.eye(4), ("s", 2): np.eye(4)}, {("s", 0): np.eye(4), ("s", 1): np.eye(4)}, pix)
        self.assertEqual(set(e), {("s", 0)})

    def test_bilan_seuils(self):
        ok = g.bilan([0.01] * 95 + [0.2] * 5)  # 5 % à plus de 0,1 mm : au seuil, tenu
        self.assertTrue(ok[5])
        trop = g.bilan([0.01] * 94 + [0.2] * 6)
        self.assertFalse(trop[5])
        gros = g.bilan([0.01] * 99 + [2.0])  # 1 % à plus de 1 mm : au-dessus de 0,5 %
        self.assertFalse(gros[5])
        self.assertAlmostEqual(g.bilan([0.0] * 199 + [2.0])[4], 0.005)
        self.assertTrue(g.bilan([0.0] * 199 + [2.0])[5])

    def test_bilan_vide(self):
        with self.assertRaises(ValueError):
            g.bilan([])

    def test_lecture_des_poses_en_ordre_colonne(self):
        import tempfile

        m = translation(1.0, 2.0, 3.0)
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
            f.write("stack\tslice\t" + "\t".join(f"m{i}{j}" for j in range(4) for i in range(4)) + "\n")
            f.write("a\t3\t" + "\t".join(str(v) for v in m.flatten(order="F")) + "\n")
        try:
            lu = g.lire_poses(f.name)
        finally:
            os.unlink(f.name)
        np.testing.assert_allclose(lu[("a", 3)], m)


if __name__ == "__main__":
    unittest.main()
