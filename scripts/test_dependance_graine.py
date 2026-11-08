"""Tests de `dependance_graine.py` : oracles calculés à la main. Lancer : `python3 -B -m unittest scripts/test_dependance_graine.py -v`."""

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dependance_graine as dg


def volume(d, nom, valeurs):
    chemin = f"{d}/{nom}.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(valeurs, dtype=np.float32), np.eye(4)), chemin)
    return chemin


def poses(d, nom, lignes):
    chemin = f"{d}/{nom}.tsv"
    with open(chemin, "w") as f:
        f.write("stack\tslice\ttx\tty\n")
        for l in lignes:
            f.write("\t".join(map(str, l)) + "\n")
    return chemin


class TestEcarts(unittest.TestCase):
    def test_ecart_volumes_est_le_maximum_en_valeur_absolue(self):
        d = tempfile.mkdtemp()
        a = volume(d, "a", [[[1.0, 2.0], [3.0, 4.0]]])
        b = volume(d, "b", [[[1.0, 2.5], [2.0, 4.0]]])  # écarts 0, 0,5, 1, 0
        self.assertEqual(dg.ecart_volumes(a, b), 1.0)
        self.assertEqual(dg.ecart_volumes(a, a), 0.0)
        with self.assertRaises(ValueError):
            dg.ecart_volumes(a, volume(d, "c", [[[1.0, 2.0]]]))

    def test_ecart_poses_numerique_et_texte(self):
        d = tempfile.mkdtemp()
        a = poses(d, "a", [("s0", 0, 0.5, -1.0), ("s0", 1, 2.0, 3.0)])
        b = poses(d, "b", [("s0", 0, 0.5, -1.25), ("s0", 1, 2.0, 3.0)])
        self.assertEqual(dg.ecart_poses(a, b), 0.25)
        self.assertEqual(dg.ecart_poses(b, a), 0.25)  # symétrique : la valeur absolue compte
        self.assertEqual(dg.ecart_poses(a, a), 0.0)
        with self.assertRaises(ValueError):
            dg.ecart_poses(a, poses(d, "c", [("s1", 0, 0.5, -1.0), ("s0", 1, 2.0, 3.0)]))  # texte différent
        with self.assertRaises(ValueError):
            dg.ecart_poses(a, poses(d, "d", [("s0", 0, 0.5, -1.0)]))  # forme différente

    def test_classer_lecture_pre_declaree(self):
        self.assertEqual(dg.classer(0.0, 0.0), "ne dépend pas de la graine")
        self.assertEqual(dg.classer(0.0, 1e-9), "dépend de la graine")
        self.assertEqual(dg.classer(1e-9, 0.0), "non déterministe à graine fixe")
        self.assertEqual(dg.classer(1e-9, 5.0), "non déterministe à graine fixe")


if __name__ == "__main__":
    unittest.main()
