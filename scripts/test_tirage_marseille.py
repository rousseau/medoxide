"""Tests de `tirage_marseille.py` : oracles calculés à la main. Lancer : `python3 -B -m unittest scripts/test_tirage_marseille.py -v`."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tirage_marseille as tm


def ecrire(racine, sujet, session, acq, heure, run=None):
    dossier = f"{racine}/{sujet}/{session}/anat"
    os.makedirs(dossier, exist_ok=True)
    entite = f"_run-{run}" if run else ""
    with open(f"{dossier}/{sujet}_{session}_acq-{acq}{entite}_T2w.json", "w") as f:
        json.dump({"AcquisitionTime": heure}, f)


def session_complete(racine, sujet, session, heure0, contrastes=("trufi", "haste")):
    n = 0
    for c in contrastes:
        for p in ("ax", "cor", "sag"):
            ecrire(racine, sujet, session, f"{c}{p}", f"10:00:{heure0 + n:02d}.000000")
            n += 1


class TestEligibles(unittest.TestCase):
    def test_liste_par_session_et_recoupement(self):
        d = tempfile.mkdtemp()
        mar, util = f"{d}/mar", f"{d}/util"
        session_complete(mar, "sub-a", "ses-01", 0)                       # éligible
        session_complete(mar, "sub-b", "ses-01", 10, contrastes=("trufi",))  # pas de HASTE : non éligible
        session_complete(mar, "sub-c", "ses-01", 20, contrastes=("haste",))  # pas de TRUFI : non éligible
        session_complete(mar, "sub-d", "ses-01", 30)                       # recoupe un sujet déjà utilisé
        ecrire(util, "sub-x", "ses-01", "trufiax", "10:00:30.000000")      # même heure que la 1re pile de sub-d
        session_complete(mar, "sub-e", "ses-01", 40, contrastes=("trufi",))
        session_complete(mar, "sub-e", "ses-02", 40)                       # 2e session éligible : retenue (la 1re ne l'est pas)
        choisis, retires = tm.eligibles(mar, util)
        self.assertEqual(choisis, {"sub-a": "ses-01", "sub-e": "ses-02"})
        self.assertEqual(retires, ["sub-d"])

    def test_la_premiere_session_eligible_est_retenue(self):
        d = tempfile.mkdtemp()
        mar, util = f"{d}/mar", f"{d}/util"
        os.makedirs(util)
        session_complete(mar, "sub-a", "ses-01", 0)
        session_complete(mar, "sub-a", "ses-02", 20)
        self.assertEqual(tm.eligibles(mar, util)[0], {"sub-a": "ses-01"})

    def test_plusieurs_runs_ne_comptent_pas_double(self):
        noms = [
            "sub-a_ses-01_acq-trufiax_run-2_T2w.json",
            "sub-a_ses-01_acq-trufiax_run-1_T2w.json",
            "sub-a_ses-01_acq-trufiax_run-3_T2w.json",
            "sub-a_ses-01_acq-truficor_T2w.json",
        ]
        self.assertEqual(tm.piles_du_plan(noms, "trufi", "ax"), [("trufiax", 1), ("trufiax", 2), ("trufiax", 3)])
        self.assertEqual(tm.piles_du_plan(noms, "trufi", "cor"), [("truficor", 1)])
        self.assertEqual(tm.pile_a_telecharger(noms, "trufi", "ax"), "trufiax")
        with self.assertRaises(ValueError):
            tm.pile_a_telecharger(["sub-a_ses-01_acq-trufiax_run-2_T2w.json"], "trufi", "ax")

    def test_les_autres_sequences_du_meme_plan_sont_distinguees(self):
        noms = ["sub-a_ses-01_acq-hastepaceautoax_run-1_T2w.json", "sub-a_ses-01_acq-trufiax_run-1_T2w.json"]
        self.assertEqual(tm.piles_du_plan(noms, "haste", "ax"), [("hastepaceautoax", 1)])


class TestTirage(unittest.TestCase):
    def test_melange_est_une_permutation_reproductible_sans_effet_de_bord(self):
        liste = [f"s{i}" for i in range(20)]
        copie = list(liste)
        m1, m2 = tm.melanger(liste, 20261107), tm.melanger(liste, 20261107)
        self.assertEqual(liste, copie)
        self.assertEqual(m1, m2)
        self.assertEqual(sorted(m1), sorted(liste))
        self.assertNotEqual(m1, liste)
        self.assertNotEqual(m1, tm.melanger(liste, 1))

    def test_etiquette_aux_limites(self):
        self.assertEqual(tm.etiquette(0.45), "calme")
        self.assertEqual(tm.etiquette(0.4499), "agite")
        self.assertEqual(tm.etiquette(0.9), "calme")
        self.assertEqual(tm.etiquette(0.1), "agite")

    def test_affectation_sequentielle_oracle(self):
        ef = tm.effectifs_vides()
        etiquettes = ["agite"] * 9 + ["calme"] * 9
        dest = [tm.affecter(e, ef) for e in etiquettes]
        # 4 agités au test 2, 3 au développement, les 2 suivants ignorés ; idem pour les calmes (4 puis 3, puis 2 ignorés)
        self.assertEqual(dest, ["test2"] * 4 + ["dev"] * 3 + [None] * 2 + ["test2"] * 4 + ["dev"] * 3 + [None] * 2)
        self.assertEqual(ef, {"test2": {"calme": 4, "agite": 4}, "dev": {"calme": 3, "agite": 3}})
        self.assertTrue(tm.complet(ef))

    def test_complet_exige_les_quatre_quotas(self):
        ef = tm.effectifs_vides()
        for e in ["agite"] * 7 + ["calme"] * 6:
            tm.affecter(e, ef)
        self.assertFalse(tm.complet(ef))
        tm.affecter("calme", ef)
        self.assertTrue(tm.complet(ef))

    def test_l_ordre_decide_de_l_affectation(self):
        ef = tm.effectifs_vides()
        self.assertEqual(tm.affecter("calme", ef), "test2")
        self.assertEqual(ef["dev"]["calme"], 0)


if __name__ == "__main__":
    unittest.main()
