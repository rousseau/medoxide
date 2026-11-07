"""Tests de `banc_mesures.py` (F4.0.1). Références indépendantes du code testé : boucles directes, cas calculés à la main,
table des étiquettes relue depuis le code Rust. Lancer : `python3 -B -m unittest scripts/test_banc_mesures.py -v`."""

import math
import os
import re
import sys
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_mesures as bm

TABLE_RUST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "medoxide-bounti", "src", "table.rs")


def volume_aleatoire(graine=0, n=14):
    rng = np.random.default_rng(graine)
    ref = 1.0 + rng.random((n, n, n))
    return ref, np.ones((n, n, n), dtype=bool), rng


class TestSsim(unittest.TestCase):
    def test_images_identiques_donnent_un(self):
        ref, masque, _ = volume_aleatoire()
        self.assertAlmostEqual(bm.ssim(ref, ref, masque), 1.0, places=12)

    def test_valeur_au_centre_contre_boucle_directe(self):
        # Référence écrite à part : on prend la fenêtre 7³ autour d'un voxel intérieur et on applique la formule à la main.
        ref, masque, rng = volume_aleatoire(1)
        x = ref + 0.2 * rng.standard_normal(ref.shape)
        i = (7, 6, 7)
        fx = x[i[0] - 3 : i[0] + 4, i[1] - 3 : i[1] + 4, i[2] - 3 : i[2] + 4]
        fy = ref[i[0] - 3 : i[0] + 4, i[1] - 3 : i[1] + 4, i[2] - 3 : i[2] + 4]
        mx, my = fx.mean(), fy.mean()
        vx, vy = ((fx - mx) ** 2).mean(), ((fy - my) ** 2).mean()
        cxy = ((fx - mx) * (fy - my)).mean()
        etendue = ref.max() - ref.min()
        c1, c2 = (0.01 * etendue) ** 2, (0.03 * etendue) ** 2
        attendu = (2 * mx * my + c1) * (2 * cxy + c2) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
        self.assertAlmostEqual(float(bm.carte_ssim(x, ref, masque)[i]), attendu, places=10)

    def test_decalage_constant_formule_a_la_main(self):
        # x = ref + d : variances et covariance égales, donc le second facteur vaut 1 et SSIM = (2 mx my + C1) / (mx² + my² + C1).
        ref, masque, _ = volume_aleatoire(2)
        d = 0.5
        i = (7, 7, 7)
        fy = ref[4:11, 4:11, 4:11]
        my = fy.mean()
        mx = my + d
        c1 = (0.01 * (ref.max() - ref.min())) ** 2
        attendu = (2 * mx * my + c1) / (mx**2 + my**2 + c1)
        self.assertAlmostEqual(float(bm.carte_ssim(ref + d, ref, masque)[i]), attendu, places=10)


class TestPsnrNccNormalisation(unittest.TestCase):
    def test_psnr_cas_calcule_a_la_main(self):
        ref, masque, _ = volume_aleatoire(3)
        # écart constant 0,1 : EQM = 0,01, PSNR = 20 log10(max(ref) / 0,1)
        attendu = 20 * math.log10(ref.max() / 0.1)
        self.assertAlmostEqual(bm.psnr(ref + 0.1, ref, masque), attendu, places=9)
        self.assertEqual(bm.psnr(ref, ref, masque), float("inf"))

    def test_psnr_ignore_ce_qui_est_hors_masque(self):
        ref, masque, _ = volume_aleatoire(4)
        masque[:5] = False
        x = ref.copy()
        x[:5] += 100.0
        self.assertEqual(bm.psnr(x, ref, masque), float("inf"))

    def test_normalisation_retrouve_la_pente_et_le_decalage(self):
        ref, masque, rng = volume_aleatoire(5)
        masque[:4] = False
        x = (ref - 3.0) / 2.0  # donc ref = 2 x + 3
        x[:4] = 1e6 * rng.random(x[:4].shape)  # hors masque : ne doit rien changer
        a, b = bm.normalisation_affine(x, ref, masque)
        self.assertAlmostEqual(a, 2.0, places=9)
        self.assertAlmostEqual(b, 3.0, places=9)

    def test_ncc(self):
        ref, masque, _ = volume_aleatoire(6)
        self.assertAlmostEqual(bm.ncc(2 * ref + 5, ref, masque), 1.0, places=12)
        self.assertAlmostEqual(bm.ncc(-ref, ref, masque), -1.0, places=12)

    def test_mesures_volume_ramene_une_image_mise_a_l_echelle_a_la_source(self):
        ref, masque, _ = volume_aleatoire(7)
        m = bm.mesures_volume(0.5 * ref + 3.0, ref, masque)
        self.assertAlmostEqual(m["ssim"], 1.0, places=9)
        self.assertGreater(m["psnr"], 150.0)
        self.assertAlmostEqual(m["ncc"], 1.0, places=9)


class TestNettete(unittest.TestCase):
    def volume_texture(self):
        rng = np.random.default_rng(8)
        v = gaussian_filter(rng.random((40, 40, 40)), 1.5)
        return 50.0 + 100.0 * (v - v.min()) / (v.max() - v.min())

    def test_invariante_a_l_echelle(self):
        v = self.volume_texture()
        self.assertAlmostEqual(bm.nettete(v), bm.nettete(7.0 * v), places=10)

    def test_diminue_quand_on_floute(self):
        v = self.volume_texture()
        self.assertLess(bm.nettete(gaussian_filter(v, 1.5)), bm.nettete(v))


class TestGroupes(unittest.TestCase):
    def test_les_19_etiquettes_sont_dans_un_seul_groupe(self):
        vus = [e for labels in bm.GROUPES.values() for e in labels]
        self.assertEqual(sorted(vus), list(range(1, 20)))

    def test_groupes_contre_les_abreviations_du_code_rust(self):
        # Oracle : la table de BOUNTI en Rust. Le groupe se déduit de l'abréviation, par une règle écrite à part.
        if not os.path.exists(TABLE_RUST):
            self.skipTest("table.rs introuvable")
        with open(TABLE_RUST, encoding="utf-8") as f:
            texte = f.read()
        paires = re.findall(r'e\((\d+), "[^"]*", "([^"]*)"', texte)
        self.assertEqual(len(paires), 20)
        regle = [
            (r"^eCSF", "lcr_externe"),
            (r"^cGM", "cortex"),
            (r"^fWM", "matiere_blanche"),
            (r"^(LV|CSP|V3|V4)", "ventricules"),
            (r"^(Cb|Vermis)", "cervelet"),
            (r"^(BG_|Thal)", "noyaux_gris"),
            (r"^BS$", "tronc"),
        ]
        for index, abbr in paires:
            index = int(index)
            if index == 0:
                continue
            attendus = [nom for motif, nom in regle if re.match(motif, abbr)]
            self.assertEqual(len(attendus), 1, abbr)
            self.assertIn(index, bm.GROUPES[attendus[0]], f"{abbr} ({index}) devrait être dans {attendus[0]}")

    def test_regrouper(self):
        et = np.array([0, 1, 2, 3, 5, 9, 10, 13, 16, 19])
        self.assertEqual(bm.regrouper(et).tolist(), [0, 1, 1, 2, 3, 4, 7, 5, 6, 4])


class TestDice(unittest.TestCase):
    def test_cas_calcules_a_la_main(self):
        a = np.zeros((4, 4, 4), dtype=np.uint8)
        b = np.zeros_like(a)
        a[:2] = 5  # matière blanche : 32 voxels
        b[:1] = 6  # matière blanche (autre étiquette du même groupe) : 16 voxels, tous inclus dans a
        a[3] = 10  # tronc seulement dans a
        d = bm.dice_par_groupe(a, b)
        self.assertAlmostEqual(d["matiere_blanche"], 2 * 16 / (32 + 16))
        self.assertEqual(d["tronc"], 0.0)
        self.assertTrue(math.isnan(d["cortex"]))

    def test_identiques(self):
        rng = np.random.default_rng(9)
        a = rng.integers(0, 20, (6, 6, 6)).astype(np.uint8)
        for nom, v in bm.dice_par_groupe(a, a).items():
            self.assertTrue(v == 1.0 or math.isnan(v), nom)


def pose(rx=0.0, ry=0.0, rz=0.0, t=(0.0, 0.0, 0.0)):
    """Matrice 4×4 : rotation (rad, axes x puis y puis z) et translation t."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry @ Rx
    M[:3, 3] = t
    return M


class TestErreurDePose(unittest.TestCase):
    def points(self):
        rng = np.random.default_rng(10)
        return {(s, k): rng.normal(size=(30, 3)) * 30 + [s * 5, k, 0] for s in range(2) for k in range(12)}

    def poses_aleatoires(self, graine):
        rng = np.random.default_rng(graine)
        return {(s, k): pose(*rng.normal(size=3) * 0.05, t=rng.normal(size=3) * 2) for s in range(2) for k in range(12)}

    def test_poses_identiques_donnent_zero(self):
        v = self.poses_aleatoires(1)
        e = bm.erreur_de_pose(self.points(), v, v)
        self.assertEqual(len(e), 24)
        self.assertLess(max(e.values()), 1e-9)

    def test_un_mouvement_commun_est_retire(self):
        v = self.poses_aleatoires(2)
        G = pose(0.3, -0.2, 0.5, t=(10.0, -4.0, 7.0))
        estimees = {k: G @ m for k, m in v.items()}
        self.assertLess(max(bm.erreur_de_pose(self.points(), v, estimees).values()), 1e-9)

    def test_une_coupe_decalee_de_2_mm_est_mesuree(self):
        v = self.poses_aleatoires(3)
        estimees = dict(v)
        estimees[(1, 5)] = pose(t=(2.0, 0.0, 0.0)) @ v[(1, 5)]
        e = bm.erreur_de_pose(self.points(), v, estimees)
        self.assertAlmostEqual(e[(1, 5)], 2.0, delta=0.15)
        self.assertLess(np.median([x for k, x in e.items() if k != (1, 5)]), 0.15)

    def test_lecture_du_format_de_medx(self):
        import tempfile

        entete = "stack\tslice\t" + "\t".join(f"m{l}{c}" for c in range(4) for l in range(4))
        M = pose(0.1, 0.2, 0.3, t=(1.0, 2.0, 3.0))
        lignes = [entete] + [f"{nom}\t{k}\t" + "\t".join(repr(float(v)) for v in M.reshape(-1, order="F")) for nom in ("a", "b") for k in range(3)]
        with tempfile.NamedTemporaryFile("w", suffix=".tsv") as f:
            f.write("\n".join(lignes) + "\n")
            f.flush()
            poses = bm.lire_poses(f.name)
        self.assertEqual(sorted(poses), [(s, k) for s in range(2) for k in range(3)])
        self.assertTrue(np.array_equal(poses[(1, 2)], M))


if __name__ == "__main__":
    unittest.main()
