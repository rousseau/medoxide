"""Tests de `banc_juge.py` (câblage, sans lancer de reconstruction). Lancer : `python3 -B -m unittest scripts/test_banc_juge.py -v`."""

import os
import re
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_juge as bjg
import banc_juge_analyse as bj


class TestCablage(unittest.TestCase):
    def test_variantes_et_options(self):
        self.assertEqual(list(bjg.OPTIONS), ["entete", "a003", "a005", "a010", "a030"])
        self.assertEqual(bjg.OPTIONS["a005"], "")  # le défaut : aucune option
        self.assertIn("--s2s-sweeps 0", bjg.OPTIONS["entete"])
        self.assertEqual([bjg.OPTIONS[v].split()[-1] for v in ("a003", "a010", "a030")], ["0.03", "0.1", "0.3"])

    def test_commande_de_simulation_aux_reglages_figes(self):
        c = bjg.commande_de_simulation("medx", "s.nii", "m.nii", ["a", "b", "c"], "dos")
        def valeur(option):
            return c[c.index(option) + 1]
        self.assertEqual(valeur("--noise-pct"), "1.0093461841900417")
        self.assertEqual((valeur("--bias-pct"), valeur("--slice-scale-pct")), ("0.0", "5.0"))
        self.assertEqual((valeur("--inplane-blur-mm"), valeur("--slice-profile-factor")), ("2.0", "4.0"))
        self.assertEqual((valeur("--motion"), valeur("--amplitude"), valeur("--seed")), ("smooth", "6.0", "1"))
        self.assertIn("--interleaved", c)
        self.assertEqual(c[c.index("--template") + 1 : c.index("--template") + 4], ["a", "b", "c"])
        self.assertNotIn("--no-recenter", c)

    def test_commande_de_juge(self):
        a = types.SimpleNamespace(racine="R", medx="medx", replis=[1])
        c = bjg.commande_de_juge(a, "sub-b01")
        self.assertEqual(c[c.index("--racine") + 1], "R/jeu")
        self.assertEqual(c[c.index("--sujet") + 1], "sub-b01")
        self.assertEqual(c[c.index("--sortie") + 1], "R/juge/sub-b01")
        self.assertEqual(c[c.index("--replis") + 1 :], ["1"])
        variantes = c[c.index("--variantes") + 1 : c.index("--replis")]
        self.assertEqual([v.split("=")[0] for v in variantes], bj.VARIANTES)
        self.assertEqual(variantes[2], "a005=")
        a.replis = None
        self.assertNotIn("--replis", bjg.commande_de_juge(a, "sub-b01"))

    def test_codes_anonymes(self):
        self.assertEqual([bjg.code_du_sujet(n) for n in (1, 6, 12)], ["sub-b01", "sub-b06", "sub-b12"])

    def test_racine_a_la_forme_d_un_jeu_reel(self):
        """Les noms produits sont exactement ceux que lisent `benchmark_variantes_s2s.py` (expression régulière copiée du script) et le test de juge."""
        d = tempfile.mkdtemp()
        simule = f"{d}/simule"
        os.makedirs(simule)
        for n in (1, 2, 3):
            for suffixe in ("T2w.nii.gz", "T2w.json", "mask.nii.gz"):
                open(f"{simule}/sim_acq-{n}_{suffixe}", "w").close()
        racine = bjg.construire_racine(simule, d, "sub-b01")
        dossier = f"{racine}/sub-b01/ses-1/anat"
        motif = re.compile(r"sub-b01_ses-1_acq-(trufiax|truficor|trufisag)_run-(\d+)_T2w\.nii\.gz$")
        piles = sorted(f for f in os.listdir(dossier) if motif.match(f))
        self.assertEqual([motif.match(f).group(1) for f in piles], ["trufiax", "truficor", "trufisag"])
        for plan in ("trufiax", "truficor", "trufisag"):
            self.assertTrue(os.path.exists(f"{racine}/derivatives/medx-fetalbet/sub-b01/ses-1/anat/sub-b01_ses-1_acq-{plan}_run-1_desc-brain_mask.nii.gz"))
        # le plan axial est la pile 1 de la simulation (ordre des gabarits), le sagittal la pile 3
        self.assertTrue(os.readlink(f"{dossier}/sub-b01_ses-1_acq-trufiax_run-1_T2w.nii.gz").endswith("sim_acq-1_T2w.nii.gz"))
        self.assertTrue(os.readlink(f"{dossier}/sub-b01_ses-1_acq-trufisag_run-1_T2w.nii.gz").endswith("sim_acq-3_T2w.nii.gz"))


if __name__ == "__main__":
    unittest.main()
