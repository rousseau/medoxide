"""Tests de `banc_o2.py` (câblage et verdicts, sans reconstruction). Lancer : `python3 -B -m unittest scripts/test_banc_o2.py -v`."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_o2 as o2


def mesures(psnr, ssim=None):
    ssim = ssim or [0.9, 0.8, 0.7, 0.6]
    return {s: {"psnr": p, "ssim": q, "ncc": 0.9} for s, p, q in zip(o2.SIGMAS, psnr, ssim)}


class TestCablage(unittest.TestCase):
    def test_simulation_calme_aux_reglages_figes(self):
        c = o2.commande_de_simulation("medx", "s.nii", "m.nii", ["a", "b", "c"], "dos")
        def valeur(option):
            return c[c.index(option) + 1]
        self.assertEqual((valeur("--motion"), valeur("--amplitude")), ("none", "0.0"))
        self.assertEqual(valeur("--noise-pct"), "1.0093461841900417")
        self.assertEqual((valeur("--bias-pct"), valeur("--slice-scale-pct"), valeur("--seed")), ("0.0", "5.0", "1"))
        self.assertEqual((valeur("--inplane-blur-mm"), valeur("--slice-profile-factor")), ("2.0", "4.0"))
        self.assertEqual(valeur("--forward-model"), "independent")

    def test_sigmas_du_critere_ii(self):
        self.assertEqual(o2.SIGMAS, [0.0, 0.5, 1.0, 2.0])

    def test_lire_passes(self):
        self.assertEqual(o2.lire_passes(["sub-b01:1", "sub-b06:2"]), [("sub-b01", 1), ("sub-b06", 2)])
        with self.assertRaises(ValueError):
            o2.lire_passes(["sub-b01"])


class TestVerdicts(unittest.TestCase):
    def test_strictement_decroissant(self):
        self.assertTrue(o2.strictement_decroissant([4, 3, 2, 1]))
        self.assertFalse(o2.strictement_decroissant([4, 3, 3, 1]))  # égalité : non
        self.assertFalse(o2.strictement_decroissant([4, 3, 3.5, 1]))
        self.assertFalse(o2.strictement_decroissant([1]))

    def test_ii_exige_psnr_et_ssim(self):
        ok, _ = o2.verdict_ii(mesures([30, 28, 25, 20]))
        self.assertTrue(ok)
        ok, _ = o2.verdict_ii(mesures([30, 28, 25, 26]))  # PSNR remonte en σ = 2
        self.assertFalse(ok)
        ok, _ = o2.verdict_ii(mesures([30, 28, 25, 20], ssim=[0.9, 0.8, 0.85, 0.6]))  # SSIM remonte en σ = 1
        self.assertFalse(ok)

    def test_i_ecart_nul_exige(self):
        p = {"sommes": {"volume": "aa"}, "mesures": {str(s): {"psnr": 30.0, "ssim": 0.9, "ncc": 0.9} for s in o2.SIGMAS}}
        q = {"sommes": {"volume": "aa"}, "mesures": {str(s): {"psnr": 30.0, "ssim": 0.9, "ncc": 0.9} for s in o2.SIGMAS}}
        self.assertEqual(o2.verdict_i(p, q), (True, []))
        q["mesures"]["1.0"]["psnr"] = 30.0000000001  # le moindre écart compte
        ok, diff = o2.verdict_i(p, q)
        self.assertFalse(ok)
        self.assertTrue(any("psnr" in d for d in diff))
        q["mesures"]["1.0"]["psnr"] = 30.0
        q["sommes"]["volume"] = "bb"
        self.assertFalse(o2.verdict_i(p, q)[0])

    def test_somme(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"abc")
            f.flush()
            self.assertEqual(o2.somme(f.name), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
