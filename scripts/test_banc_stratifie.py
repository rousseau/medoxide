"""Tests de `banc_stratifie.py` : oracles calculés à la main. Lancer : `python3 -B -m unittest scripts/test_banc_stratifie.py -v`."""

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banc_stratifie as b  # noqa: E402


def pile(nx=9, ny=9, nz=40):
    """Fond de 1 et une impulsion de 100 au centre de chaque coupe, masque plein."""
    d = np.ones((nx, ny, nz), dtype=np.float32)
    d[4, 4, :] = 100.0
    return d, np.ones((nx, ny, nz), dtype=np.uint8)


class TestTirage(unittest.TestCase):
    def test_effectifs_sans_remise_reproductible(self):
        t = b.tirer_coupes(40, 0.3, np.random.default_rng(5))
        self.assertEqual(len(t), 12)
        self.assertEqual(len(set(t)), 12)
        self.assertEqual(t, sorted(t))
        self.assertEqual(t, b.tirer_coupes(40, 0.3, np.random.default_rng(5)))
        self.assertNotEqual(t, b.tirer_coupes(40, 0.3, np.random.default_rng(6)))
        self.assertEqual(len(b.tirer_coupes(40, 0.1, np.random.default_rng(5))), 4)
        self.assertEqual(b.tirer_coupes(40, 0.0, np.random.default_rng(5)), [])


class TestArtefacts(unittest.TestCase):
    def comptes(self, d, m):
        creuses = int(sum(d[0, 0, k] < 0.5 for k in range(d.shape[2])))  # fond de 1 (0,3 si creux ; un flou ajoute moins de 0,01)
        manquantes = int(sum(not m[:, :, k].any() for k in range(m.shape[2])))
        floues = int(sum(abs(d[4, 4, k] - 100.0) > 1e-3 and abs(d[4, 4, k] - 30.0) > 1e-3 for k in range(d.shape[2])))
        return creuses, manquantes, floues

    def test_good_et_single_ne_changent_rien(self):
        for niveau in ("good", "single"):
            d0, m0 = pile()
            d, m = b.injecter_artefacts(d0, m0, niveau, 3, 0)
            self.assertTrue(np.array_equal(d, d0) and np.array_equal(m, m0))

    def test_med_et_bad(self):
        d0, m0 = pile()
        self.assertEqual(self.comptes(*b.injecter_artefacts(d0, m0, "med", 3, 0)), (4, 4, 0))
        d, m = b.injecter_artefacts(d0, m0, "bad", 3, 0)
        self.assertEqual(self.comptes(d, m), (12, 12, 4))
        # une impulsion floutée par σ = 1,5 : 1/(2π σ²) du total au centre, calculé à la main sur un noyau continu (≈ 0,0707), donc loin de 100 ; le fond reste ≈ 1
        floues = [k for k in range(40) if abs(d[4, 4, k] - 100.0) > 1e-3 and abs(d[4, 4, k] - 30.0) > 1e-3]
        for k in floues:
            self.assertLess(d[4, 4, k], 40.0)
        self.assertEqual(self.comptes(*b.injecter_artefacts(d0, m0, "bad", 3, 0)), (12, 12, 4))
        self.assertTrue(np.array_equal(d, b.injecter_artefacts(d0, m0, "bad", 3, 0)[0]))  # reproductible
        self.assertFalse(np.array_equal(m, b.injecter_artefacts(d0, m0, "bad", 4, 0)[1]))  # graine
        self.assertFalse(np.array_equal(m, b.injecter_artefacts(d0, m0, "bad", 3, 1)[1]))  # numéro de stack
        self.assertEqual(float(d0[4, 4, 0]), 100.0)  # l'entrée n'est pas modifiée

    def test_deriver_piles_garde_les_stacks_du_niveau(self):
        with tempfile.TemporaryDirectory() as t:
            for n in (1, 2, 3):
                d, m = pile()
                nib.save(nib.Nifti1Image(d, np.eye(4)), f"{t}/sim_acq-{n}_T2w.nii.gz")
                nib.save(nib.Nifti1Image(m, np.eye(4)), f"{t}/sim_acq-{n}_mask.nii.gz")
            attendu = {"good": 3, "med": 3, "bad": 2, "single": 1}
            for niveau, nb in attendu.items():
                piles, masques = b.deriver_piles(t, f"{t}/{niveau}", niveau, 1)
                self.assertEqual((len(piles), len(masques)), (nb, nb))
                self.assertTrue(all(os.path.exists(p) for p in piles + masques))
            lu = np.asarray(nib.load(f"{t}/bad/sim_acq-1_T2w.nii.gz").dataobj)
            self.assertEqual(sum(lu[0, 0, k] < 0.5 for k in range(40)), 12)


class TestAnalyse(unittest.TestCase):
    def test_lecture_du_journal_et_grandeurs(self):
        texte = "\n".join([
            "SEMISYN lisse 4 1 entete 0.20 3.14 4.00",
            "SLICE lisse 4 1 fin 0 0 1.0 1.0 100 0.9",
            "SLICE lisse 4 1 fin 0 1 1.0 5.0 100 0.9",
            "SLICE lisse 4 1 fin 0 2 1.0 NaN 100 0.9",
            "SLICE lisse 4 1 fin 0 3 1.0 7.0 100 0.9",
            "SLICE lisse 4 1 entete 0 3 1.0 9.0 100 0.9",
            "SLICE lisse 6 2 fin 1 0 1.0 2.0 100 0.9",
        ])
        erreurs, lignes = b.lire_journal_e2(texte)
        self.assertEqual(erreurs, {(4.0, 1): [1.0, 5.0, 7.0], (6.0, 2): [2.0]})
        self.assertEqual(len(lignes), 7)
        g = b.grandeurs_e2(erreurs[(4.0, 1)])
        self.assertAlmostEqual(g["part_sup_4mm"], 2 / 3)
        self.assertAlmostEqual(g["part_sup_6mm"], 1 / 3)
        self.assertEqual(g["mediane_mm"], 5.0)

    def valeurs(self, good, med, bad):
        mk = lambda x: {f"s{i}": [v - 0.1, v, v + 0.1] for i, v in enumerate(x)}
        return {"good": mk(good), "med": mk(med), "bad": mk(bad)}

    def test_verdict_monotonie_et_separation(self):
        # erreur : 6 sujets, 5 monotones sur 6 (le dernier : med < good) ; étendue entre graines 0,2 ; écart Good − Bad = 1,0 > 0,6
        v = b.verdict_famille(self.valeurs([1, 1, 1, 1, 1, 2], [1.5, 1.5, 1.5, 1.5, 1.5, 1.5], [2, 2, 2, 2, 2, 2]), +1)
        self.assertEqual(v["monotones"], "5/6")
        self.assertTrue(v["ii"])
        self.assertAlmostEqual(v["etendue"], 0.2)
        self.assertAlmostEqual(v["ecart"], 2 - (5 * 1 + 2) / 6)  # 2 − 7/6
        self.assertTrue(v["iii"])
        # 4 sujets sur 6 : (ii) non tenu
        v = b.verdict_famille(self.valeurs([1, 1, 1, 1, 2, 2], [1.5, 1.5, 1.5, 1.5, 1.5, 1.5], [2, 2, 2, 2, 2, 2]), +1)
        self.assertEqual(v["monotones"], "4/6")
        self.assertFalse(v["ii"])
        # écart trop petit devant l'étendue : (iii) non tenu (écart 0,5 < 3 × 0,2)
        v = b.verdict_famille(self.valeurs([1] * 6, [1.2] * 6, [1.5] * 6), +1)
        self.assertTrue(v["ii"])
        self.assertFalse(v["iii"])

    def test_sens_decroissant_pour_le_psnr(self):
        v = b.verdict_famille(self.valeurs([30] * 6, [28] * 6, [24] * 6), -1)
        self.assertEqual(v["monotones"], "6/6")
        self.assertAlmostEqual(v["ecart"], 6.0)
        self.assertTrue(v["iii"])
        v = b.verdict_famille(self.valeurs([24] * 6, [28] * 6, [30] * 6), -1)
        self.assertEqual(v["monotones"], "0/6")


if __name__ == "__main__":
    unittest.main()
