"""Tests de `variabilite_graines.py` et du câblage `--graines` des deux scripts d'évaluation (oracles calculés à la main, exécutables stubbés).
Lancer : `python3 -B -m unittest scripts/test_variabilite_graines.py -v`."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

import variabilite_graines as vg


class TestAides(unittest.TestCase):
    def test_dossier_de_graine(self):
        self.assertEqual(vg.dossier_de_graine("/s", None), "/s")
        self.assertEqual(vg.dossier_de_graine("/s", 3), "/s/graine-3")

    def test_options_avec_graine(self):
        self.assertEqual(vg.options_avec_graine("", 2), "--s2s-seed 2")
        self.assertEqual(vg.options_avec_graine("--alpha 0.1", 7), "--alpha 0.1 --s2s-seed 7")
        for deja in ("--s2s-seed 4", "--alpha 0.1 --s2s-seed=4"):
            with self.assertRaises(ValueError):
                vg.options_avec_graine(deja, 1)

    def test_graines_valides(self):
        self.assertEqual(vg.valider_graines([1, 2, 3]), [1, 2, 3])
        self.assertEqual(vg.valider_graines([0]), [0])
        for mauvais in ([], [1, 1], [-1], [1.5]):
            with self.assertRaises(ValueError):
                vg.valider_graines(mauvais)

    def test_agreger(self):
        a = vg.agreger([0.80, 0.90, 0.85])
        self.assertAlmostEqual(a["moyenne"], 0.85)
        self.assertEqual((a["minimum"], a["maximum"]), (0.80, 0.90))
        self.assertAlmostEqual(a["etendue"], 0.10)


def ecrire_resultats(base, graine, lignes):
    d = vg.dossier_de_graine(base, graine)
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/resultats.tsv", "w") as f:
        f.write("variante\tstack_teste\tncc_moyenne\tnettete\ttemps_cpu_s\n")
        for variante, h, ncc, net in lignes:
            f.write(f"{variante}\t{h}\t{ncc}\t{net}\t\n")  # temps vide : colonne ignorée


class TestTableau(unittest.TestCase):
    def test_tableau_oracle_deux_sujets_trois_graines(self):
        d = tempfile.mkdtemp()
        # sujet a : juge moyen par graine (replis 0 et 1) 0.80, 0.90, 0.85 ; sujet b : 0.70, 0.71, 0.72
        for g, (ja, jb) in zip((1, 2, 3), ((0.80, 0.70), (0.90, 0.71), (0.85, 0.72))):
            ecrire_resultats(f"{d}/a", g, [("v", 0, ja - 0.01, 0.1), ("v", 1, ja + 0.01, 0.1)])
            ecrire_resultats(f"{d}/b", g, [("v", 0, jb - 0.01, 0.2), ("v", 1, jb + 0.01, 0.2)])
        lignes = vg.tableau([f"{d}/a", f"{d}/b"], [1, 2, 3], "resultats.tsv")
        par = {(l[0], l[1], l[2]): l[3:] for l in lignes}
        moy, mini, maxi, ete = par[("a", "v", "ncc_moyenne")]
        self.assertAlmostEqual(moy, 0.85)
        self.assertAlmostEqual(ete, 0.10)
        self.assertAlmostEqual(par[("b", "v", "ncc_moyenne")][3], 0.02)
        self.assertAlmostEqual(par[("a", "v", "nettete")][3], 0.0)
        self.assertNotIn(("a", "v", "temps_cpu_s"), par)
        # résumé : étendue moyenne (0,10 + 0,02) / 2 = 0,06 ; minimale 0,02 ; maximale 0,10 ; 2 sujets
        resume = par[("RÉSUMÉ", "v", "ncc_moyenne")]
        self.assertAlmostEqual(resume[0], 0.06)
        self.assertAlmostEqual(resume[1], 0.02)
        self.assertAlmostEqual(resume[2], 0.10)
        self.assertEqual(resume[3], 2)


def stub(chemin, contenu):
    with open(chemin, "w") as f:
        f.write(contenu)
    os.chmod(chemin, os.stat(chemin).st_mode | stat.S_IEXEC)


def preparer_jeu():
    """Racine avec 3 stacks et masques minuscules, `medx` et binaire de test factices (le premier copie un volume et note ses arguments)."""
    d = tempfile.mkdtemp()
    sujet, ses = "sub-t", "ses-1"
    rng = np.random.default_rng(0)
    img = lambda a: nib.Nifti1Image(a.astype(np.float32), np.diag([1.0, 1.0, 3.0, 1.0]))
    for plan in ("trufiax", "truficor", "trufisag"):
        nom = f"{sujet}_{ses}_acq-{plan}_run-1"
        os.makedirs(f"{d}/racine/{sujet}/{ses}/anat", exist_ok=True)
        os.makedirs(f"{d}/racine/derivatives/medx-fetalbet/{sujet}/{ses}/anat", exist_ok=True)
        nib.save(img(rng.random((12, 12, 6)) + 1), f"{d}/racine/{sujet}/{ses}/anat/{nom}_T2w.nii.gz")
        nib.save(img(np.ones((12, 12, 6))), f"{d}/racine/derivatives/medx-fetalbet/{sujet}/{ses}/anat/{nom}_desc-brain_mask.nii.gz")
    nib.save(img(rng.random((12, 12, 12)) + 1), f"{d}/volume.nii.gz")
    os.makedirs(f"{d}/bin")
    stub(f"{d}/bin/medx", f'#!/bin/sh\necho "$@" >> "{d}/appels.log"\nout=""; prev=""\nfor a in "$@"; do [ "$prev" = "--output" ] && out="$a"; prev="$a"; done\ncp "{d}/volume.nii.gz" "$out"\n')
    stub(f"{d}/bin/medoxide_svr_test", '#!/bin/sh\necho "moyenne 0.9000"\n')
    return d, sujet


def lancer_script(script, args, d):
    env = {**os.environ, "MEDOXIDE_TEST_BIN": f"{d}/bin/medoxide_svr_test"}
    return subprocess.run([sys.executable, "-I", f"{ICI}/{script}", *args], capture_output=True, text=True, env=env, cwd=d)


class TestCablage(unittest.TestCase):
    def appels(self, d):
        with open(f"{d}/appels.log") as f:
            return f.read().splitlines()

    def test_variantes_sans_graines_comportement_inchange(self):
        d, sujet = preparer_jeu()
        r = lancer_script("benchmark_variantes_s2s.py", ["--racine", f"{d}/racine", "--sujet", sujet, "--sortie", f"{d}/s", "--medx", f"{d}/bin/medx", "--replis", "0", "--variantes", "base="], d)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(f"{d}/s/resultats.tsv"))
        self.assertFalse(os.path.exists(f"{d}/s/graine-1"))
        self.assertFalse(any("--s2s-seed" in a for a in self.appels(d)))

    def test_variantes_avec_graines_sous_dossiers_et_option_transmise(self):
        d, sujet = preparer_jeu()
        r = lancer_script("benchmark_variantes_s2s.py", ["--racine", f"{d}/racine", "--sujet", sujet, "--sortie", f"{d}/s", "--medx", f"{d}/bin/medx", "--replis", "0", "--graines", "1", "2", "--variantes", "base=", "a=--alpha 0.1"], d)
        self.assertEqual(r.returncode, 0, r.stderr)
        for g in (1, 2):
            self.assertTrue(os.path.exists(f"{d}/s/graine-{g}/resultats.tsv"))
            self.assertTrue(os.path.exists(f"{d}/s/graine-{g}/base_h0.nii.gz"))
        appels = self.appels(d)
        self.assertEqual(len(appels), 4)  # 2 variantes x 2 graines
        self.assertEqual(sum("--s2s-seed 1" in a for a in appels), 2)
        self.assertEqual(sum("--s2s-seed 2" in a for a in appels), 2)
        self.assertTrue(all(a.count("--s2s-seed") == 1 for a in appels))
        self.assertIn("--alpha 0.1 --s2s-seed 1", "\n".join(appels))

    def test_variantes_refus_d_une_graine_deja_dans_les_options(self):
        d, sujet = preparer_jeu()
        r = lancer_script("benchmark_variantes_s2s.py", ["--racine", f"{d}/racine", "--sujet", sujet, "--sortie", f"{d}/s", "--medx", f"{d}/bin/medx", "--replis", "0", "--graines", "1", "--variantes", "x=--s2s-seed 9"], d)
        self.assertNotEqual(r.returncode, 0)

    def test_evaluation_finale_avec_graines(self):
        d, sujet = preparer_jeu()
        base = ["--racine", f"{d}/racine", "--sujet", sujet, "--methode", "ours", "--sortie", f"{d}/e", "--bin", f"{d}/bin"]
        r = lancer_script("evaluation_finale.py", [*base, "--graines", "1", "2"], d)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for g in (1, 2):
            self.assertTrue(os.path.exists(f"{d}/e/graine-{g}/{sujet}/ours/full_s2s.nii.gz"))
            self.assertTrue(os.path.exists(f"{d}/e/graine-{g}/{sujet}/ours/folds/resultats.tsv"))
        appels = self.appels(d)
        recon = [a for a in appels if a.startswith("svr recon")]
        self.assertEqual(len(recon), 2 * (4 + 3 * 5))  # par graine : 4 volumes complets + 3 replis x 5 variantes
        self.assertTrue(all(a.count("--s2s-seed") == 1 for a in recon))
        self.assertEqual(sum("--s2s-seed 1" in a for a in recon), 4 + 15)
        self.assertEqual(sum("--s2s-seed 2" in a for a in recon), 4 + 15)

    def test_evaluation_finale_sans_graines_inchangee(self):
        d, sujet = preparer_jeu()
        base = ["--racine", f"{d}/racine", "--sujet", sujet, "--methode", "ours", "--sortie", f"{d}/e", "--bin", f"{d}/bin"]
        r = lancer_script("evaluation_finale.py", base, d)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(f"{d}/e/{sujet}/ours/full_s2s.nii.gz"))
        self.assertFalse(any("--s2s-seed" in a for a in self.appels(d)))

    def test_evaluation_finale_refuse_graines_pour_une_methode_externe(self):
        d, sujet = preparer_jeu()
        r = lancer_script("evaluation_finale.py", ["--racine", f"{d}/racine", "--sujet", sujet, "--methode", "svrtk", "--sortie", f"{d}/e", "--bin", f"{d}/bin", "--graines", "1"], d)
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
