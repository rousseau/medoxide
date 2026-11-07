"""Tests de `agitation_par_contraste.py` : oracles calculés à la main. Lancer : `python3 -B -m unittest scripts/test_agitation_par_contraste.py -v`."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agitation_par_contraste as ac

ENTETE = "stack\tslice\tregistered\tscore_before\tscore_after\tn_registrations\tcorrection_rms_mm\texcluded\treason\tmask_pixels\tmean_signal\tn_partners\n"


class TestConcordance(unittest.TestCase):
    def ecrire(self, lignes):
        d = tempfile.mkdtemp()
        chemin = f"{d}/qc.tsv"
        with open(chemin, "w") as f:
            f.write(ENTETE)
            for score, pixels in lignes:
                f.write(f"s\t0\t1\t{score}\t0.9\t1\t0.1\t0\t\t{pixels}\t100\t2\n")
        return chemin

    def test_moyenne_des_coupes_avec_score_et_masque_suffisant(self):
        # retenues : 0,8 et 0,4 (moyenne 0,6) ; exclues : « n/a » et la coupe à 499 pixels (score 0,1)
        chemin = self.ecrire([("0.8", 500), ("0.4", 900), ("n/a", 900), ("0.1", 499)])
        self.assertAlmostEqual(ac.concordance_moyenne(chemin), 0.6)

    def test_aucune_coupe_valide(self):
        with self.assertRaises(ValueError):
            ac.concordance_moyenne(self.ecrire([("n/a", 900), ("0.5", 10)]))


class TestSeuil(unittest.TestCase):
    def test_milieu_si_separes(self):
        # plus faible calme 0,488 ; plus fort agité 0,410 : milieu 0,449
        self.assertAlmostEqual(ac.seuil_de_separation([0.589, 0.574, 0.488], [0.410, 0.296, 0.344]), 0.449)

    def test_aucun_seuil_si_recouvrement_ou_egalite(self):
        self.assertIsNone(ac.seuil_de_separation([0.5, 0.3], [0.4, 0.2]))
        self.assertIsNone(ac.seuil_de_separation([0.5, 0.4], [0.4, 0.2]))  # égalité : pas strictement séparés


class TestPiles(unittest.TestCase):
    def racine(self, fichiers):
        d = tempfile.mkdtemp()
        os.makedirs(f"{d}/sub-X/ses-1/anat")
        for f in fichiers:
            open(f"{d}/sub-X/ses-1/anat/sub-X_ses-1_{f}_T2w.nii.gz", "w").close()
        return d

    def test_noms_de_sequence_variables(self):
        d = self.racine(["acq-hastepaceautoax_run-1", "acq-hastepaceautocor_run-1", "acq-hastepaceautosag_run-1", "acq-hastesag_run-2"])
        e, m = ac.piles(d, "sub-X", "haste")
        self.assertTrue(e[0].endswith("sub-X/ses-1/anat/sub-X_ses-1_acq-hastepaceautoax_run-1_T2w.nii.gz"))
        self.assertTrue(m[2].endswith("derivatives/medx-fetalbet/sub-X/ses-1/anat/sub-X_ses-1_acq-hastepaceautosag_run-1_desc-brain_mask.nii.gz"))
        d = self.racine(["acq-hastete183ax_run-1", "acq-hastete183cor_run-1", "acq-hastete183sag_run-1"])
        self.assertTrue(ac.piles(d, "sub-X", "haste")[0][1].endswith("acq-hastete183cor_run-1_T2w.nii.gz"))

    def test_trufi_et_haste_ne_se_melangent_pas(self):
        d = self.racine(["acq-trufiax_run-1", "acq-truficor_run-1", "acq-trufisag_run-1", "acq-hasteax_run-1", "acq-hastecor_run-1", "acq-hastesag_run-1"])
        self.assertTrue(ac.piles(d, "sub-X", "trufi")[0][1].endswith("acq-truficor_run-1_T2w.nii.gz"))
        self.assertTrue(ac.piles(d, "sub-X", "haste")[0][1].endswith("acq-hastecor_run-1_T2w.nii.gz"))

    def test_absente_ou_ambigue_leve_une_erreur(self):
        with self.assertRaises(ValueError):
            ac.piles(self.racine(["acq-hasteax_run-1", "acq-hastecor_run-1"]), "sub-X", "haste")
        with self.assertRaises(ValueError):
            ac.piles(self.racine(["acq-hasteax_run-1", "acq-hastepaceautoax_run-1", "acq-hastecor_run-1", "acq-hastesag_run-1"]), "sub-X", "haste")


if __name__ == "__main__":
    unittest.main()
