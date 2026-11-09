"""Tests de `variabilite_reference.py` : oracles calculés à la main. Lancer : `python3 -B -m unittest scripts/test_variabilite_reference.py -v`."""

import argparse
import os
import stat
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import variabilite_reference as vr


class TestRangement(unittest.TestCase):
    def test_chemins_complets(self):
        self.assertEqual(vr.chemins_complets("/b", "s2s", "sub-x", 2), ("/b/s2s/sub-x/complet_g2.nii.gz", "/b/s2s/sub-x/poses_g2.tsv"))

    def test_liens_bounti(self):
        liens = vr.liens_bounti("/b", "sub-x", [2, 1], [0, 1])
        self.assertEqual(liens[0], ("full.nii.gz", "../complet_g1.nii.gz"))  # graine la plus petite
        self.assertIn(("gs2_h1.nii.gz", "../graine-2/d005_h1.nii.gz"), liens)
        self.assertIn(("gs1_h0.nii.gz", "../graine-1/d005_h0.nii.gz"), liens)
        self.assertEqual(len(liens), 1 + 2 * 2)

    def test_replis_presents_pour_toutes_les_graines(self):
        d = tempfile.mkdtemp()
        for g, hs in ((1, (0, 1, 2)), (2, (0, 2)), (3, (0, 2, 3))):
            dossier = f"{d}/s2s/sub-x/graine-{g}"
            os.makedirs(dossier)
            for h in hs:
                open(f"{dossier}/d005_h{h}.nii.gz", "w").close()
            open(f"{dossier}/autre_h1.nii.gz", "w").close()  # autre variante : ignorée
        self.assertEqual(vr.replis_presents(d, "sub-x", [1, 2, 3]), [0, 2])

    def test_etendues_q2(self):
        d = tempfile.mkdtemp()
        chemin = f"{d}/q.tsv"
        with open(chemin, "w") as f:
            f.write("sujet\tmesure\tsans correction\ts2s_g1\ts2s_g2\ts2s_g3\ts2v\n")
            f.write("calme 1\ttoutes\t0.5\t0.80\t0.90\t0.85\t0.7\n")
            f.write("calme 1\thors_stack_0\t0.5\t0.60\t0.60\t0.66\t0.7\n")
            f.write("calme 1\tcoupes\t100\t100\t100\t100\t100\n")
        r = vr.etendues_q2(chemin, [1, 2, 3])
        self.assertEqual(set(r), {("calme 1", "toutes"), ("calme 1", "hors_stack_0")})
        self.assertAlmostEqual(r[("calme 1", "toutes")]["etendue"], 0.10)
        self.assertAlmostEqual(r[("calme 1", "toutes")]["moyenne"], 0.85)
        self.assertAlmostEqual(r[("calme 1", "hors_stack_0")]["etendue"], 0.06)

    def test_resume_q2_compte_chaque_sujet(self):
        d = tempfile.mkdtemp()
        sorties = []
        for i, (sujet, etiquette, ecart) in enumerate((("a", "calme", 0.02), ("b", "calme", 0.04), ("c", "agité", 0.10))):
            chemin = f"{d}/q{i}.tsv"
            with open(chemin, "w") as f:
                f.write("sujet\tmesure\ts2s_g1\ts2s_g2\ts2s_g3\n")
                f.write(f"{etiquette} 1\ttoutes\t0.5\t{0.5 + ecart}\t0.5\n")
            sorties.append((sujet, etiquette, chemin))
        moyenne, maximum, n = vr.resumer_q2(vr.collecter_q2(sorties, [1, 2, 3]))["toutes"]
        self.assertEqual(n, 3)  # trois sujets, deux étiquettes : aucun n'est écrasé
        self.assertAlmostEqual(moyenne, (0.02 + 0.04 + 0.10) / 3)
        self.assertAlmostEqual(maximum, 0.10)


class TestCablage(unittest.TestCase):
    def test_complets_s2s_a_chaque_graine_s2v_une_seule_fois(self):
        d = tempfile.mkdtemp()
        sujet, ses = "sub-t", "ses-1"
        img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4))
        for plan in ("trufiax", "truficor", "trufisag"):
            nom = f"{sujet}_{ses}_acq-{plan}_run-1"
            os.makedirs(f"{d}/r/{sujet}/{ses}/anat", exist_ok=True)
            os.makedirs(f"{d}/r/derivatives/medx-fetalbet/{sujet}/{ses}/anat", exist_ok=True)
            nib.save(img, f"{d}/r/{sujet}/{ses}/anat/{nom}_T2w.nii.gz")
            nib.save(img, f"{d}/r/derivatives/medx-fetalbet/{sujet}/{ses}/anat/{nom}_desc-brain_mask.nii.gz")
        medx = f"{d}/medx"
        with open(medx, "w") as f:
            f.write(f'#!/bin/sh\necho "$@" >> "{d}/appels.log"\nprev=""\nfor a in "$@"; do [ "$prev" = "--output" ] && : > "$a"; [ "$prev" = "--poses" ] && : > "$a"; prev="$a"; done\n')
        os.chmod(medx, os.stat(medx).st_mode | stat.S_IEXEC)
        a = argparse.Namespace(base=f"{d}/b", graines=[1, 2, 3], sujet=sujet, racine=f"{d}/r", medx=medx)
        vr.cmd_complets(a)
        with open(f"{d}/appels.log") as f:
            appels = f.read().splitlines()
        s2s = [l for l in appels if "--motion s2s" in l]
        s2v = [l for l in appels if "--motion s2v" in l]
        self.assertEqual([("--s2s-seed %d" % g) in l for g, l in zip((1, 2, 3), s2s)], [True] * 3)
        self.assertEqual(len(s2s), 3)
        self.assertEqual(len(s2v), 1)
        self.assertIn("--s2s-seed 1", s2v[0])
        for g in (1, 2, 3):
            self.assertTrue(os.path.exists(f"{d}/b/s2s/{sujet}/complet_g{g}.nii.gz"))
            self.assertTrue(os.path.exists(f"{d}/b/s2s/{sujet}/poses_g{g}.tsv"))
        self.assertTrue(os.path.exists(f"{d}/b/s2v/{sujet}/poses_g1.tsv"))
        vr.cmd_complets(a)  # reprise : rien n'est relancé
        with open(f"{d}/appels.log") as f:
            self.assertEqual(len(f.read().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
