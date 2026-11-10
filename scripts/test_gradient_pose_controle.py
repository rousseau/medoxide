"""Tests de `gradient_pose_controle.py` : python3 -B -m unittest scripts/test_gradient_pose_controle.py -v (depuis la racine du dépôt)."""

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradient_pose_controle as c  # noqa: E402


def bilan(med, fin, gros):
    return (60, med, 1.0, fin, gros, True)


class TestPerturber(unittest.TestCase):
    def test_facteur_float32_et_geometrie_conservee(self):
        with tempfile.TemporaryDirectory() as t:
            data = np.arange(24, dtype=np.int16).reshape(2, 3, 4) * 100
            aff = np.diag([0.8, 0.8, 3.5, 1.0])
            nib.save(nib.Nifti1Image(data, aff), f"{t}/a.nii.gz")
            c.perturber(f"{t}/a.nii.gz", f"{t}/b.nii.gz", 7e-7)
            b = nib.load(f"{t}/b.nii.gz")
            self.assertEqual(b.get_data_dtype(), np.float32)
            np.testing.assert_allclose(b.affine, aff)
            y = np.asarray(b.dataobj)
            np.testing.assert_allclose(y, data * (1 + 7e-7), rtol=1e-7)
            self.assertTrue(np.any(y != data))  # le facteur est bien représenté (pas absorbé par l'arrondi)
            self.assertAlmostEqual(float((y[1, 2, 3] / data[1, 2, 3]) - 1), 7e-7, delta=1e-7)


class TestVerdicts(unittest.TestCase):
    def test_c1_a_la_main(self):
        controles = [bilan(0.010, 0.05, 0.01), bilan(0.020, 0.10, 0.02), bilan(0.015, 0.08, 0.0)]  # plus grandes parts 0,10 et 0,02 ; médiane max 0,020
        self.assertTrue(c.verdict_c1(bilan(0.030, 0.10, 0.02), controles)[0])  # à égalité : tenu ; médiane 0,030 = 1,5 × 0,020 : tenu
        self.assertFalse(c.verdict_c1(bilan(0.030, 0.101, 0.02), controles)[0])  # part > 0,1 mm au-dessus
        self.assertFalse(c.verdict_c1(bilan(0.030, 0.10, 0.021), controles)[0])  # part > 1 mm au-dessus
        self.assertFalse(c.verdict_c1(bilan(0.0301, 0.10, 0.02), controles)[0])  # médiane au-dessus de 1,5 fois
        self.assertTrue(c.verdict_c1(bilan(0.0, 0.0, 0.0), controles)[0])
        with self.assertRaises(ValueError):
            c.verdict_c1(bilan(0, 0, 0), [])

    def test_c4_a_la_main(self):
        self.assertTrue(c.verdict_c4(0.940, [0.95, 0.97, 0.99])[0])  # seuil 0,94 : à égalité, tenu
        self.assertFalse(c.verdict_c4(0.939, [0.95, 0.97, 0.99])[0])
        self.assertTrue(c.verdict_c4(0.99, [0.95]) [0])
        with self.assertRaises(ValueError):
            c.verdict_c4(0.9, [])


if __name__ == "__main__":
    unittest.main()
