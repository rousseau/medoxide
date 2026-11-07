"""Tests des scripts du banc de vérité terrain (F4.0.1). Lancer : `python3 -m unittest scripts/test_banc.py -v`
depuis la racine du dépôt, ou `python3 -m unittest test_banc -v` depuis `scripts/`. Bibliothèque standard seulement."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tirage_banc


def cohorte_fictive():
    """40 sujets `s00`…`s39`, âges de 21 à 40 semaines ; les sujets multiples de 5 ont une seconde session."""
    lignes = []
    for i in range(40):
        age = 21.0 + i * 0.5
        lignes.append({"subject_id": f"s{i:02d}", "session_id": "ses-01", "ga_weeks": f"{age:.2f}"})
        if i % 5 == 0:
            lignes.append({"subject_id": f"s{i:02d}", "session_id": "ses-02", "ga_weeks": f"{age + 1:.2f}"})
    return lignes


class TestTirage(unittest.TestCase):
    def test_determinisme(self):
        a = tirage_banc.tirer(cohorte_fictive(), set(), 7)
        b = tirage_banc.tirer(cohorte_fictive(), set(), 7)
        self.assertEqual(a, b)
        self.assertNotEqual(a, tirage_banc.tirer(cohorte_fictive(), set(), 8))

    def test_un_sujet_par_strate_et_alternance(self):
        res = tirage_banc.tirer(cohorte_fictive(), set(), 7, n_strates=6)
        self.assertEqual([r["strate"] for r in res], [1, 2, 3, 4, 5, 6])
        self.assertEqual([r["jeu"] for r in res], ["dev", "test"] * 3)
        self.assertEqual(len({r["sujet"] for r in res}), 6)

    def test_ages_croissants_par_strate(self):
        res = tirage_banc.tirer(cohorte_fictive(), set(), 7, n_strates=6)
        ages = [float(r["age"]) for r in res]
        self.assertEqual(ages, sorted(ages))

    def test_filtre_d_age_et_exclusion(self):
        elig = tirage_banc.eligibles(cohorte_fictive(), {"s10:ses-01"})
        self.assertTrue(all(22.0 <= float(r["ga_weeks"]) <= 39.0 for r in elig))
        self.assertNotIn(("s10", "ses-01"), {(r["subject_id"], r["session_id"]) for r in elig})


if __name__ == "__main__":
    unittest.main()
