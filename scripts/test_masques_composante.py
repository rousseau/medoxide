"""Tests de `masques_composante.py` : oracles calculés à la main. Lancer : `python3 -B -m unittest scripts/test_masques_composante.py -v`."""

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import masques_composante as mc


class TestComposante(unittest.TestCase):
    def test_deux_blocs_on_garde_le_plus_grand(self):
        m = np.zeros((10, 10, 10), dtype=np.uint8)
        m[1:4, 1:4, 1:4] = 1   # 27 voxels
        m[7:9, 7:9, 7] = 1     # 4 voxels, isolé
        n, k, gardes, ecartes = mc.plus_grande_composante(m)
        self.assertEqual((k, gardes, ecartes), (2, 27, 4))
        self.assertEqual(int(n.sum()), 27)
        self.assertTrue(n[2, 2, 2] and not n[7, 7, 7])

    def test_connexite_26_relie_les_voisins_par_un_coin(self):
        m = np.zeros((4, 4, 4), dtype=np.uint8)
        m[0, 0, 0] = 1
        m[1, 1, 1] = 1   # voisin par un coin : même composante à 26 voisins, deux à 6
        m[3, 3, 3] = 1
        _, k, gardes, ecartes = mc.plus_grande_composante(m)
        self.assertEqual((k, gardes, ecartes), (2, 2, 1))

    def test_masque_vide_et_valeurs_non_binaires(self):
        _, k, g, e = mc.plus_grande_composante(np.zeros((3, 3, 3)))
        self.assertEqual((k, g, e), (0, 0, 0))
        m = np.zeros((3, 3, 3)); m[0, 0, 0] = 255
        self.assertEqual(mc.plus_grande_composante(m)[1:], (1, 1, 0))

    def test_axe_des_coupes(self):
        self.assertEqual(mc.axe_des_coupes((1.0, 1.0, 3.5)), 2)
        self.assertEqual(mc.axe_des_coupes((3.5, 1.0, 1.0, 1.0)), 0)
        self.assertEqual(mc.axe_des_coupes((1.0, 4.0, 1.0)), 1)

    def test_cinq_coupes_regulieres_dans_l_etendue(self):
        m = np.zeros((4, 4, 40), dtype=np.uint8)
        m[1, 1, 10] = 1
        m[2, 2, 30] = 1
        self.assertEqual(mc.cinq_coupes(m, 2), [10, 15, 20, 25, 30])
        with self.assertRaises(ValueError):
            mc.cinq_coupes(np.zeros((3, 3, 3)), 2)

    def test_chemins(self):
        brut = "/r/derivatives/medx-fetalbet/sub-x/ses-1/anat/sub-x_ses-1_acq-a_run-1_desc-brain_mask.nii.gz"
        self.assertEqual(mc.chemin_nettoye(brut), brut.replace("medx-fetalbet", "medx-fetalbet-cc"))
        self.assertEqual(mc.chemin_pile(brut), "/r/sub-x/ses-1/anat/sub-x_ses-1_acq-a_run-1_T2w.nii.gz")

    def test_cli_nettoie_sans_toucher_l_original_et_rapporte(self):
        d = tempfile.mkdtemp()
        dossier = f"{d}/derivatives/medx-fetalbet/sub-x/ses-1/anat"
        os.makedirs(dossier)
        m = np.zeros((10, 10, 10), dtype=np.uint8)
        m[1:4, 1:4, 1:4] = 1
        m[7:9, 7:9, 7] = 1
        brut = f"{dossier}/sub-x_ses-1_acq-a_run-1_desc-brain_mask.nii.gz"
        nib.save(nib.Nifti1Image(m, np.eye(4)), brut)
        class A: racine = d; sujets = ["sub-x"]; rapport = f"{d}/rapport.tsv"
        mc.cmd_nettoyer(A)
        self.assertEqual(int(np.asarray(nib.load(brut).dataobj).sum()), 31)  # original intact
        self.assertEqual(int(np.asarray(nib.load(mc.chemin_nettoye(brut)).dataobj).sum()), 27)
        lignes = open(A.rapport).read().splitlines()
        self.assertIn("\t2\t27\t4\t0.8710\tnon", lignes[1])  # 27/31 = 0,871 ≥ 0,8 : pas d'alerte

    def test_alerte_sous_80_pour_cent(self):
        d = tempfile.mkdtemp()
        dossier = f"{d}/derivatives/medx-fetalbet/sub-x/ses-1/anat"
        os.makedirs(dossier)
        m = np.zeros((10, 10, 10), dtype=np.uint8)
        m[1:3, 1:3, 1:3] = 1   # 8 voxels
        m[6:8, 6:8, 6:9] = 1   # 12 voxels : la plus grande, 12/20 = 0,6
        nib.save(nib.Nifti1Image(m, np.eye(4)), f"{dossier}/sub-x_ses-1_acq-a_run-1_desc-brain_mask.nii.gz")
        class A: racine = d; sujets = ["sub-x"]; rapport = f"{d}/rapport.tsv"
        mc.cmd_nettoyer(A)
        self.assertIn("\t0.6000\toui", open(A.rapport).read())


if __name__ == "__main__":
    unittest.main()
