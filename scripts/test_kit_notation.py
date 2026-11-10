"""Tests de `kit_notation.py` : oracles calculés à la main. Lancer : `python3 -B -m unittest scripts/test_kit_notation.py -v`."""

import csv
import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kit_notation as k  # noqa: E402

SUJETS = ("SUJETSECRET1", "SUJETSECRET2")
METHODES = ("methodeAlpha", "methodeBeta", "methodeGamma")


def ecrire(chemin, data, affine=None):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine), chemin)


class TestPermutation(unittest.TestCase):
    def test_reproductible_bijective_ids_distincts(self):
        p1, i1 = k.permutation_et_identifiants(29)
        p2, i2 = k.permutation_et_identifiants(29)
        self.assertEqual((p1, i1), (p2, i2))
        self.assertEqual(sorted(p1), list(range(29)))
        self.assertEqual(len(set(i1)), 29)
        self.assertTrue(all(1000 <= i <= 9999 for i in i1))

    def test_pas_l_identite_et_depend_de_la_graine(self):
        p1, _ = k.permutation_et_identifiants(29)
        self.assertNotEqual(p1, list(range(29)))
        self.assertNotEqual(p1, k.permutation_et_identifiants(29, graine=1)[0])


class TestGeometrie(unittest.TestCase):
    def test_indices_medians_a_la_main(self):
        m = np.zeros((10, 12, 14))
        m[2:7, 3:10, 4:13] = 1  # bornes i 2..6, j 3..9, k 4..12
        self.assertEqual(k.indices_medians(m), ((2 + 6) // 2, (3 + 9) // 2, (4 + 12) // 2))
        with self.assertRaises(ValueError):
            k.indices_medians(np.zeros((3, 3, 3)))

    def test_fenetrage(self):
        c = np.array([[0.0, 5.0, 10.0, 20.0, -3.0]])
        self.assertEqual(k.fenetrer(c, 0.0, 10.0).tolist(), [[0, 128, 255, 255, 0]])  # 127,5 arrondi à 128 (arrondi pair de numpy : 128)
        self.assertEqual(k.fenetrer(c, 4.0, 4.0).max(), 0)

    def test_reechantillonnage_identite_et_translation(self):
        rng = np.random.default_rng(1)
        d = rng.normal(size=(6, 7, 8))
        with tempfile.TemporaryDirectory() as t:
            a = os.path.join(t, "a.nii.gz")
            ecrire(a, d)
            img = nib.load(a)
            np.testing.assert_allclose(k.reechantillonner(img, img), d, atol=1e-5)
            # grille décalée de 1 voxel en i : la valeur en (i, j, k) de la grille est celle de la source en (i+1, j, k)
            dec = np.eye(4)
            dec[0, 3] = 1.0
            ecrire(os.path.join(t, "g.nii.gz"), d, dec)
            y = k.reechantillonner(img, nib.load(os.path.join(t, "g.nii.gz")))
            np.testing.assert_allclose(y[:-1], d[1:], atol=1e-5)
            self.assertTrue(np.all(y[-1] == 0))  # hors du champ de vue


class TestKit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.data = os.path.join(t, "data", "notation")
        os.makedirs(self.data)
        rng = np.random.default_rng(0)
        self.lignes = []
        for s in SUJETS:
            m = np.zeros((14, 16, 18))
            m[3:11, 4:12, 5:13] = 1
            ref = os.path.join(t, f"{s}_ref.nii.gz")
            ecrire(ref, rng.uniform(size=m.shape))
            mp = os.path.join(t, f"{s}_mask.nii.gz")
            ecrire(mp, m)
            for mt in METHODES:
                v = os.path.join(t, f"{s}_{mt}.nii.gz")
                ecrire(v, rng.uniform(size=m.shape) * (1 + len(mt)))
                self.lignes.append({"sujet": s, "methode": mt, "chemin": v, "masque": mp, "reference": ref})
        self.kit = os.path.join(self.data, "kit")
        self.corr = os.path.join(self.data, "correspondance_kit.tsv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_kit_complet_et_aveugle(self):
        ordre = k.construire(self.lignes, self.kit, self.corr)
        self.assertEqual(len(ordre), 6)
        fichiers = sorted(os.listdir(self.kit))
        self.assertEqual(fichiers, sorted(["grille.tsv", "LISEZMOI.txt"] + [f"{i}.png" for i, *_ in ordre]))
        # la grille : un identifiant par ligne, dans l'ordre de présentation, notes vides
        with open(os.path.join(self.kit, "grille.tsv"), newline="") as f:
            g = list(csv.reader(f, delimiter="\t"))
        self.assertEqual(g[0], ["identifiant", *k.NOTES])
        self.assertEqual([int(r[0]) for r in g[1:]], [o[0] for o in ordre])
        self.assertTrue(all(all(c == "" for c in r[1:]) for r in g[1:]))
        # la table est cohérente avec ce qui a été fabriqué : chaque (sujet, méthode) exactement une fois
        with open(self.corr, newline="") as f:
            c = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(sorted((r["sujet"], r["methode"]) for r in c), sorted((l["sujet"], l["methode"]) for l in self.lignes))
        self.assertEqual([int(r["position"]) for r in c], list(range(6)))
        # aveugle : aucun nom de sujet ni de méthode dans le kit (noms, octets des PNG, métadonnées, grille, LISEZMOI)
        for nom in fichiers:
            contenu = open(os.path.join(self.kit, nom), "rb").read()
            for mot in SUJETS + METHODES:
                self.assertNotIn(mot.encode(), contenu, (nom, mot))
                self.assertNotIn(mot, nom)
        for nom in fichiers:
            if nom.endswith(".png"):
                im = Image.open(os.path.join(self.kit, nom))
                self.assertEqual(im.info, {})  # aucune métadonnée texte
                self.assertEqual(im.mode, "L")

    def test_png_a_la_taille_des_trois_coupes(self):
        ordre = k.construire(self.lignes, self.kit, self.corr)
        im = Image.open(os.path.join(self.kit, f"{ordre[0][0]}.png"))
        # coupes de (14,16,18) : axiale (14, 16), coronale (14, 18), sagittale (16, 18) ; hauteur = max des deuxièmes dimensions = 18 ; largeur = 14 + 14 + 16
        self.assertEqual(im.size, ((14 + 14 + 16) * k.AGRANDISSEMENT, 18 * k.AGRANDISSEMENT))

    def test_memes_coupes_pour_tous_les_volumes_d_un_sujet(self):
        # deux méthodes = le même volume : leurs images sont identiques à l'octet près
        s = SUJETS[0]
        v0 = next(l for l in self.lignes if l["sujet"] == s)
        copie = dict(v0, methode="copie")
        lignes = [v0, copie]
        ordre = k.construire(lignes, self.kit, self.corr)
        a, b = (open(os.path.join(self.kit, f"{o[0]}.png"), "rb").read() for o in ordre)
        self.assertEqual(a, b)

    def test_correspondance_seulement_sous_data_et_hors_du_kit(self):
        with self.assertRaises(ValueError):
            k.construire(self.lignes, self.kit, os.path.join(self.tmp.name, "correspondance.tsv"))  # pas sous data/
        with self.assertRaises(ValueError):
            k.construire(self.lignes, self.kit, os.path.join(self.kit, "correspondance.tsv"))  # dans le kit
        self.assertFalse(os.path.exists(self.kit))

    def test_colonne_manquante(self):
        p = os.path.join(self.tmp.name, "c.tsv")
        open(p, "w").write("sujet\tmethode\nA\tm\n")
        with self.assertRaises(ValueError):
            k.lire_couples(p)


if __name__ == "__main__":
    unittest.main()
