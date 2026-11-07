"""Tests de `banc_plausibilite.py` : oracles calculés à la main sur des piles synthétiques. Lancer : `python3 -B -m unittest scripts/test_banc_plausibilite.py -v`."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_plausibilite as bp


def pile_synthetique(graine=0, nz=10):
    """Pile 60 × 60 × nz : tissu (400) au centre, tissu maternel (300) dans les coins haut-gauche, bruit de |N(0, 3)| dans les trois autres coins et le reste de l'air."""
    rng = np.random.default_rng(graine)
    s = np.abs(rng.normal(0, 3.0, (60, 60, nz)))
    s[0:20, 0:20, :] = 300.0
    s[15:45, 15:45, :] = 400.0
    masque = np.zeros(s.shape, dtype=bool)
    masque[15:45, 15:45, :] = True
    return s, masque


class TestBruit(unittest.TestCase):
    def test_les_carres_de_tissu_maternel_sont_exclus(self):
        s, m = pile_synthetique()
        carres = bp.choisir_carres(s, bp.moyenne_tissu(s, m))
        self.assertEqual(len(carres), 3 * 10)  # trois coins d'air par coupe, dix coupes
        self.assertTrue(all((i0, j0) != (0, 0) for _, i0, j0 in carres))

    def test_rapport_contre_calcul_direct(self):
        s, m = pile_synthetique(1)
        carres = bp.choisir_carres(s, 400.0)
        valeurs = []
        for k in range(10):
            for (i0, j0) in ((0, 48), (48, 0), (48, 48)):
                valeurs.extend(s[i0 : i0 + 12, j0 : j0 + 12, k].ravel().tolist())
        attendu = np.std(np.array(valeurs)) / 400.0
        self.assertAlmostEqual(bp.rapport_bruit(s, m, carres), attendu, places=12)
        # |N(0, σ)| : écart type σ √(1 − 2/π)
        self.assertAlmostEqual(bp.rapport_bruit(s, m, carres) * 400.0, 3.0 * np.sqrt(1 - 2 / np.pi), delta=0.1)

    def test_pas_assez_de_carres_d_air_pour_le_sujet(self):
        s, m = pile_synthetique()
        s[:, :, :] = np.maximum(s, 100.0)  # plus d'air nulle part
        carres = bp.choisir_carres(s, 400.0)
        self.assertEqual(carres, [])
        with self.assertRaises(ValueError):
            bp.exiger_carres([carres, carres, carres])
        s2, m2 = pile_synthetique(2)
        bp.exiger_carres([bp.choisir_carres(s2, 400.0), [], []])  # 30 carrés : suffisant
        dix = bp.choisir_carres(s2, 400.0)[:10]
        bp.exiger_carres([dix, dix, dix])  # 30 carrés au total, aucune pile n'atteint 20 : le seuil est bien par sujet
        with self.assertRaises(ValueError):
            bp.exiger_carres([dix[:5], dix[:5], dix[:5]])  # 15 carrés

    def test_bruit_du_sujet_regroupe_les_piles_normalisees_par_leur_propre_tissu(self):
        # deux piles : la seconde est la première multipliée par 2 (tissu 800, bruit doublé) : après normalisation par μ_tissu de chaque pile, mêmes valeurs
        s1, m1 = pile_synthetique(3)
        s2 = 2.0 * s1
        carres = [bp.choisir_carres(s1, 400.0), bp.choisir_carres(s2, 800.0)]
        r = bp.rapport_bruit_sujet([s1, s2], [m1, m1], carres)
        un = bp.rapport_bruit_sujet([s1], [m1], [carres[0]])
        self.assertAlmostEqual(r, un, places=12)
        # contre un calcul direct : toutes les valeurs des carrés, divisées par 400 (pile 1) ou 800 (pile 2)
        brutes = []
        for pile, mu in ((s1, 400.0), (s2, 800.0)):
            for k in range(10):
                for (i0, j0) in ((0, 48), (48, 0), (48, 48)):
                    brutes.extend((pile[i0 : i0 + 12, j0 : j0 + 12, k] / mu).ravel().tolist())
        self.assertAlmostEqual(r, float(np.std(brutes)), places=12)


class TestBord(unittest.TestCase):
    def rampe(self, largeur, debut=4, nz=14):
        """Marche en rampe linéaire de 0 à 1000 sur `largeur` coupes à partir de la coupe `debut` (valeurs exactement représentables : pas d'égalité douteuse) ; masque plein."""
        p = np.zeros((6, 6, nz))
        for k in range(nz):
            p[:, :, k] = 1000.0 * np.clip((k - debut) / largeur, 0.0, 1.0)
        return p, np.ones(p.shape, dtype=bool)

    def test_largeur_d_une_rampe_connue(self):
        p, m = self.rampe(4)
        emp = bp.emplacements_de_bord(p, m, 500.0, n_max=10)
        self.assertGreater(len(emp), 0)
        self.assertTrue(all(k == 5 for _, _, k in emp))  # seul maximum local strict : le premier point de la rampe où le gradient est maximal
        # largeur 10–90 % d'une rampe de 4 coupes : 0,8 × 4 = 3,2 coupes, soit 11,2 mm à 3,5 mm
        self.assertTrue(np.allclose(bp.largeur_de_bord(p, emp, 3.5), 11.2, atol=1e-9))

    def test_une_rampe_deux_fois_plus_etroite_donne_une_largeur_moitie(self):
        p, m = self.rampe(2)
        emp = bp.emplacements_de_bord(p, m, 500.0, n_max=10)
        self.assertTrue(all(k == 5 for _, _, k in emp))
        w = bp.largeur_de_bord(p, emp, 1.0)
        self.assertTrue(np.allclose(w, 1.6, atol=1e-9), w[:3])  # 0,8 × 2 coupes

    def test_orientation_indifferente(self):
        p, m = self.rampe(4)
        inverse = p[:, :, ::-1].copy()
        emp = bp.emplacements_de_bord(inverse, m, 500.0, n_max=5)
        self.assertGreater(len(emp), 0)
        self.assertTrue(np.allclose(bp.largeur_de_bord(inverse, emp, 3.5), 11.2))

    def test_contraste_insuffisant_exclu(self):
        p, m = self.rampe(4)
        self.assertEqual(len(bp.emplacements_de_bord(p, m, 1e6, n_max=10)), 0)


class TestBordDansLePlan(unittest.TestCase):
    def test_une_rampe_dans_chaque_axe_donne_la_meme_largeur(self):
        """Une rampe de 4 pixels (0,8 × 4 = 3,2 pixels de 10 à 90 %) le long de l'axe 0, puis 1, puis 2 : même largeur en mm, emplacements sur l'axe demandé."""
        for axe in (0, 1, 2):
            n = [6, 6, 6]
            n[axe] = 14
            p = np.zeros(n)
            idx = np.arange(14)
            valeurs = 1000.0 * np.clip((idx - 4) / 4.0, 0.0, 1.0)
            p = np.moveaxis(np.broadcast_to(valeurs, (6, 6, 14)).copy(), 2, axe)
            m = np.ones(p.shape, dtype=bool)
            emp = bp.emplacements_de_bord_axe(p, m, 500.0, axe, n_max=10)
            self.assertGreater(len(emp), 0, axe)
            self.assertTrue(all(e[axe] == 5 for e in emp), (axe, emp[:3]))
            self.assertTrue(np.allclose(bp.largeur_de_bord_axe(p, emp, 0.7, axe), 3.2 * 0.7), axe)

    def test_axe_2_identique_a_la_version_entre_coupes(self):
        rng = np.random.default_rng(1)
        p = np.cumsum(rng.random((8, 8, 16)) * 100.0, axis=2)
        m = np.ones(p.shape, dtype=bool)
        e1 = bp.emplacements_de_bord(p, m, 300.0, n_max=20)
        e2 = bp.emplacements_de_bord_axe(p, m, 300.0, 2, n_max=20)
        self.assertTrue(np.array_equal(e1, e2))
        self.assertTrue(np.array_equal(bp.largeur_de_bord(p, e1, 2.0), bp.largeur_de_bord_axe(p, e2, 2.0, 2)))

    def test_deux_axes_ne_se_confondent_pas(self):
        """Bord seulement le long de x : aucun emplacement le long de y (le gradient en y est nul), un seul axe a des bords."""
        p = np.zeros((14, 14, 14))
        p[:, :, :] = (1000.0 * np.clip((np.arange(14) - 4) / 4.0, 0.0, 1.0))[:, None, None]
        m = np.ones(p.shape, dtype=bool)
        self.assertGreater(len(bp.emplacements_de_bord_axe(p, m, 500.0, 0)), 0)
        self.assertEqual(len(bp.emplacements_de_bord_axe(p, m, 500.0, 1)), 0)


class TestDispersion(unittest.TestCase):
    def sujet(self, bruit, largeurs, plages, plan):
        return {"bruit": [bruit], "largeur": largeurs, "biais": plages, "plan": plan}

    def test_seuil_80_pour_cent_arrondi_vers_le_haut(self):
        # n : 2 -> 2 (1,6) ; 4 -> 4 (3,2) ; 5 -> 4 (4,0 exact) ; 6 -> 5 (4,8) ; 10 -> 8 (8,0 exact) ; 12 -> 10 (9,6)
        self.assertEqual([bp.seuil_de_dispersion(n) for n in (1, 2, 4, 5, 6, 10, 12)], [1, 2, 4, 4, 5, 8, 10])

    def test_intervalle_ferme_et_comptage(self):
        reel = [self.sujet(1.0, [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]), self.sujet(3.0, [3.0], [3.0], [3.0])]
        # bornes incluses : 1,0 et 3,0 sont dans l'intervalle ; 0,99 et 3,01 non
        sim = [self.sujet(1.0, [1.0, 3.0, 0.99], [3.0, 3.01, 2.0], [2.0] * 6)]
        par, tenu = bp.verdict_dispersion(sim, reel)
        self.assertEqual((par["bruit"]["dans"], par["bruit"]["n"]), (1, 1))
        self.assertEqual((par["largeur"]["dans"], par["largeur"]["n"], par["largeur"]["seuil"]), (2, 3, 3))
        self.assertFalse(par["largeur"]["tenu"])  # 2 sur 3 : sous les 80 %
        self.assertEqual(par["biais"]["dans"], 2)
        self.assertFalse(tenu)

    def test_tenu_quand_chaque_grandeur_atteint_son_seuil(self):
        reel = [self.sujet(0.01, [10.0, 12.0], [0.1, 0.2], [3.0, 3.2]), self.sujet(0.012, [10.5], [0.15], [3.1])]
        sim = [self.sujet(0.011, [11.0] * 3, [0.15] * 3, [3.1] * 6)]
        par, tenu = bp.verdict_dispersion(sim, reel)
        self.assertTrue(tenu, par)
        self.assertEqual(par["plan"]["n"], 6)

    def test_grandeurs_du_sujet(self):
        mes = [{"bruit_sujet": 0.01, "largeur": 1.0, "largeur_plan": [2.0, 3.0]},
               {"bruit_sujet": 0.01, "largeur": 4.0, "largeur_plan": [5.0, 6.0]},
               {"bruit_sujet": 0.01, "largeur": 7.0, "largeur_plan": [8.0, 9.0]}]
        g = bp.grandeurs_du_sujet(mes, [0.1, 0.2, 0.3])
        self.assertEqual(g["bruit"], [0.01])
        self.assertEqual(g["largeur"], [1.0, 4.0, 7.0])
        self.assertEqual(g["biais"], [0.1, 0.2, 0.3])
        self.assertEqual(g["plan"], [2.0, 3.0, 5.0, 6.0, 8.0, 9.0])  # les deux directions comptées ensemble


class TestDivers(unittest.TestCase):
    def test_lecture_de_la_plage_du_biais(self):
        texte = "bruit\nBIAIS 0 -0.1000 0.2000 0.3000 0.0500\nBIAIS 1 -0.2 0.4 0.6 0.1\nautre\n"
        self.assertEqual(bp.lire_plage_biais(texte), [0.3, 0.6])

    def test_meilleur_parametre(self):
        self.assertEqual(bp.meilleur_parametre([0.5, 1.0, 2.0], [0.4, 1.1, 5.0], 1.0), 1.0)
        # |ln| : 0,5 contre 2,0 pour une cible de 1 sont à égale distance ; la mesure 0,9 est plus proche que 1,3
        self.assertEqual(bp.meilleur_parametre([1, 2], [0.9, 1.3], 1.0), 1)


class TestCalibrage(unittest.TestCase):
    def test_bornes_du_critere(self):
        import banc_calibrage as bc

        self.assertTrue(bc.dans_les_bornes(0.8) and bc.dans_les_bornes(1.25) and bc.dans_les_bornes(1.0))
        self.assertFalse(bc.dans_les_bornes(0.79) or bc.dans_les_bornes(1.26))

    def test_rapports_mediane_sur_les_trois_piles(self):
        import banc_calibrage as bc

        reel = [{"bruit_sujet": 0.01, "largeur": 10.0, "largeur_plan": [3.0, 3.0]}] * 3
        mes = [{"bruit_sujet": 0.02, "largeur": 5.0, "largeur_plan": [3.0, 3.0]}] * 3
        mes = [dict(mes[0], largeur=5.0), dict(mes[0], largeur=10.0), dict(mes[0], largeur=20.0)]
        r = bc.rapports(reel, mes, [0.2, 0.2, 0.2], [0.1, 0.3, 0.4])
        self.assertAlmostEqual(r["bruit"], 2.0)  # le bruit est celui du sujet, identique pour les trois piles
        self.assertAlmostEqual(r["largeur"], 1.0)  # médiane de (0,5 ; 1 ; 2)
        self.assertAlmostEqual(r["biais"], 0.3 / 0.2)
        self.assertAlmostEqual(r["plan"], 1.0)

    def test_le_rapport_dans_le_plan_est_le_pire_des_deux_directions(self):
        import banc_calibrage as bc

        reel = [{"bruit_sujet": 0.01, "largeur": 10.0, "largeur_plan": [3.0, 3.0]}] * 3
        # x : 0,9 (écart de ln = 0,105) ; y : 0,5 (écart de ln = 0,693) -> retenu : y
        mes = [{"bruit_sujet": 0.01, "largeur": 10.0, "largeur_plan": [2.7, 1.5]}] * 3
        r = bc.rapports(reel, mes, [0.2] * 3, [0.2] * 3)
        self.assertAlmostEqual(r["plan_x"], 0.9)
        self.assertAlmostEqual(r["plan_y"], 0.5)
        self.assertAlmostEqual(r["plan"], 0.5)
        # direction trop large contre trop étroite : 1,6 (ln = 0,47) est pire que 0,8 (ln = 0,22)
        mes = [{"bruit_sujet": 0.01, "largeur": 10.0, "largeur_plan": [4.8, 2.4]}] * 3
        self.assertAlmostEqual(bc.rapports(reel, mes, [0.2] * 3, [0.2] * 3)["plan"], 1.6)

    def test_rapport_de_calibrage_median_groupe(self):
        import banc_calibrage as bc

        reel = [[{"largeur_plan": [2.0, 2.0]}] * 3] * 2
        mes = [[{"largeur_plan": [1.0, 4.0]}] * 3, [{"largeur_plan": [2.0, 2.0]}] * 3]
        # rapports groupés : 12 valeurs = 6 × {0,5 ; 2} + 6 × {1 ; 1} -> triées : 0,5 ×3, 1 ×6, 2 ×3 -> médiane 1
        self.assertAlmostEqual(bc.rapport_plan_pour_calibrage(reel, mes), 1.0)


if __name__ == "__main__":
    unittest.main()
