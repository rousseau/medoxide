"""Tests de `banc_simule.py` (F4.0.1) : lecture des variantes et du tirage, flou, recalage rigide sur un décalage connu.
Lancer : `python3 -B -m unittest scripts/test_banc_simule.py -v`."""

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, shift

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_mesures as bm
import banc_simule as bs


def blob(graine=0, n=64):
    """Volume 64³ à structure lisse non symétrique (somme de gaussiennes) : un recalage y est bien posé."""
    rng = np.random.default_rng(graine)
    x, y, z = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    v = np.zeros((n, n, n))
    for _ in range(6):
        c = rng.uniform(22, 42, 3)
        s = rng.uniform(4, 8)
        v += rng.uniform(0.5, 1.5) * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / (2 * s * s))
    return 1000 * v


class TestOutils(unittest.TestCase):
    def test_variantes(self):
        v = bs.parser_variantes(["defaut=", "entete=--s2s-sweeps 0 --s2s-block-levels 0", 'x=--a "b c"'])
        self.assertEqual(v[0], ("defaut", []))
        self.assertEqual(v[1], ("entete", ["--s2s-sweeps", "0", "--s2s-block-levels", "0"]))
        self.assertEqual(v[2], ("x", ["--a", "b c"]))
        with self.assertRaises(ValueError):
            bs.parser_variantes(["a=", "a=--b"])

    def test_tirage_filtre_par_jeu(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv") as f:
            f.write("sujet\tsession\tage\tstrate\tjeu\ns1\tses-01\t24\t1\tdev\ns2\tses-01\t26\t2\ttest\ns3\tses-02\t28\t3\tdev\n")
            f.flush()
            self.assertEqual([r["sujet"] for r in bs.lire_tirage(f.name, "dev")], ["s1", "s3"])
            self.assertEqual([r["sujet"] for r in bs.lire_tirage(f.name, "test")], ["s2"])
            self.assertEqual(len(bs.lire_tirage(f.name, "tous")), 3)

    def test_flou_ecrit_le_volume_floute(self):
        with tempfile.TemporaryDirectory() as d:
            v = blob()
            nib.save(nib.Nifti1Image(v.astype(np.float32), np.eye(4)), f"{d}/v.nii.gz")
            bs.flouter(f"{d}/v.nii.gz", 2.0, f"{d}/f.nii.gz")
            attendu = gaussian_filter(v.astype(np.float32), 2.0)
            self.assertLess(np.abs(bs.lire(f"{d}/f.nii.gz") - attendu).max(), 1e-3)
            bs.flouter(f"{d}/v.nii.gz", 0.0, f"{d}/c.nii.gz")
            self.assertTrue(np.array_equal(bs.lire(f"{d}/c.nii.gz"), v.astype(np.float32)))


class TestRecalage(unittest.TestCase):
    def test_un_decalage_connu_est_retrouve(self):
        with tempfile.TemporaryDirectory() as d:
            ref = blob(3)
            masque = (ref > 50).astype(np.uint8)
            deplace = shift(ref, (3.0, -2.0, 1.5), order=1)
            for nom, v in (("ref", ref), ("mob", deplace)):
                nib.save(nib.Nifti1Image(v.astype(np.float32), np.eye(4)), f"{d}/{nom}.nii.gz")
            nib.save(nib.Nifti1Image(masque, np.eye(4)), f"{d}/m.nii.gz")
            m = masque > 0
            avant = bm.ncc(deplace, ref, m)
            bs.recaler(f"{d}/ref.nii.gz", f"{d}/mob.nii.gz", f"{d}/m.nii.gz", f"{d}/al.nii.gz")
            apres = bm.ncc(bs.lire(f"{d}/al.nii.gz"), ref, m)
            self.assertGreater(apres, 0.995)
            self.assertGreater(apres, avant + 0.02)

    def test_reproductible_bit_a_bit(self):
        with tempfile.TemporaryDirectory() as d:
            ref = blob(4)
            nib.save(nib.Nifti1Image(ref.astype(np.float32), np.eye(4)), f"{d}/ref.nii.gz")
            nib.save(nib.Nifti1Image(shift(ref, (2.0, 1.0, -1.0), order=1).astype(np.float32), np.eye(4)), f"{d}/mob.nii.gz")
            nib.save(nib.Nifti1Image((ref > 50).astype(np.uint8), np.eye(4)), f"{d}/m.nii.gz")
            bs.recaler(f"{d}/ref.nii.gz", f"{d}/mob.nii.gz", f"{d}/m.nii.gz", f"{d}/a.nii.gz")
            bs.recaler(f"{d}/ref.nii.gz", f"{d}/mob.nii.gz", f"{d}/m.nii.gz", f"{d}/b.nii.gz")
            self.assertTrue(np.array_equal(bs.lire(f"{d}/a.nii.gz"), bs.lire(f"{d}/b.nii.gz")))


if __name__ == "__main__":
    unittest.main()
