"""Tests de `run_fetmrqc_sr.py` : oracles calculés à la main, exécutables factices pour le câblage. Lancer : `python3 -B -m unittest scripts/test_run_fetmrqc_sr.py -v`."""

import csv
import os
import stat
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_fetmrqc_sr as r  # noqa: E402


def ecrire_volume(chemin, data):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), chemin)


def executable(chemin, corps):
    with open(chemin, "w") as f:
        f.write("#!/bin/sh\n" + corps)
    os.chmod(chemin, os.stat(chemin).st_mode | stat.S_IXUSR)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.vols = []
        for i, (sujet, methode, graine) in enumerate([("A", "m1", "1"), ("A", "m2", "-"), ("B", "m1", "1")]):
            p = os.path.join(self.d, f"v{i}.nii.gz")
            ecrire_volume(p, np.arange(8).reshape(2, 2, 2) + i)
            self.vols.append((sujet, methode, graine, p))
        self.liste = os.path.join(self.d, "liste.tsv")
        r.ecrire_tsv(
            self.liste, r.COLONNES_LISTE, [dict(zip(r.COLONNES_LISTE, v)) for v in self.vols]
        )
        self.dossier = os.path.join(self.d, "qualite")

    def tearDown(self):
        self.tmp.cleanup()


class TestPreparer(Base):
    def test_arborescence_et_correspondance(self):
        table = r.preparer(self.liste, self.dossier)
        self.assertEqual([t["numero"] for t in table], ["001", "002", "003"])
        for t, v in zip(table, self.vols):
            c = r.chemins(self.dossier, t["numero"])
            self.assertTrue(os.path.islink(c["im"]))
            self.assertEqual(os.path.realpath(c["im"]), os.path.realpath(v[3]))
        lu = r.lire_tsv(os.path.join(self.dossier, "correspondance.tsv"), ("numero",) + r.COLONNES_LISTE)
        self.assertEqual([(x["sujet"], x["methode"], x["graine"]) for x in lu], [v[:3] for v in self.vols])

    def test_nom_du_fichier_sans_sujet_ni_methode(self):
        r.preparer(self.liste, self.dossier)
        for racine, _, fichiers in os.walk(os.path.join(self.dossier, "bids")):
            for nom in fichiers + [racine]:
                for mot in ("m1", "m2"):
                    self.assertNotIn(mot, os.path.basename(nom))

    def test_volume_absent(self):
        r.ecrire_tsv(self.liste, r.COLONNES_LISTE, [dict(zip(r.COLONNES_LISTE, ("A", "m", "1", "/inexistant.nii.gz")))])
        with self.assertRaises(FileNotFoundError):
            r.preparer(self.liste, self.dossier)

    def test_colonne_manquante(self):
        with open(self.liste, "w") as f:
            f.write("sujet\tmethode\n" "A\tm\n")
        with self.assertRaises(ValueError):
            r.preparer(self.liste, self.dossier)


class TestEcreter(Base):
    def test_ecretage_au_centile_et_original_intact(self):
        src = os.path.join(self.d, "e.nii.gz")
        data = np.zeros((10, 10, 10))
        data.flat[:] = np.arange(1000, dtype=float)  # centile 99,5 de 0..999 = 994,005
        data.flat[0] = -5.0
        ecrire_volume(src, data)
        out = os.path.join(self.d, "e_ecrete.nii.gz")
        r.ecreter(src, out)
        y = np.asarray(nib.load(out).dataobj)
        self.assertAlmostEqual(float(y.max()), np.percentile(np.arange(1000.0), 99.5), places=2)
        self.assertEqual(float(y.min()), 0.0)  # le négatif est ramené à 0
        self.assertEqual(float(np.asarray(nib.load(src).dataobj).min()), -5.0)


class TestFlouter(Base):
    def test_sigma_zero_original_et_flou_decroissant_en_variance(self):
        rng = np.random.default_rng(0)
        p = os.path.join(self.d, "bruit.nii.gz")
        ecrire_volume(p, rng.normal(size=(16, 16, 16)))
        r.ecrire_tsv(self.liste, r.COLONNES_LISTE, [dict(zip(r.COLONNES_LISTE, ("A", "s2s", "1", p))), dict(zip(r.COLONNES_LISTE, ("A", "s2s", "2", p))), dict(zip(r.COLONNES_LISTE, ("B", "autre", "1", p)))])
        sortie = r.flouter(self.liste, os.path.join(self.d, "flou"), [0.0, 0.5, 1.0, 2.0], "s2s", "1")
        self.assertEqual([s["methode"] for s in sortie], ["s2s_flou0", "s2s_flou0.5", "s2s_flou1", "s2s_flou2"])  # une seule ligne source retenue
        self.assertEqual(sortie[0]["chemin"], os.path.realpath(p))
        variances = [float(np.var(np.asarray(nib.load(s["chemin"]).dataobj))) for s in sortie]
        self.assertTrue(all(a > b for a, b in zip(variances, variances[1:])), variances)
        # oracle : un Dirac flouté par σ = 1 garde une somme de 1 et son maximum vaut (2π)^(-3/2) ≈ 0,0635 (à la troncature près)
        d = np.zeros((21, 21, 21))
        d[10, 10, 10] = 1.0
        q = os.path.join(self.d, "dirac.nii.gz")
        ecrire_volume(q, d)
        r.ecrire_tsv(self.liste, r.COLONNES_LISTE, [dict(zip(r.COLONNES_LISTE, ("A", "s2s", "1", q)))])
        y = np.asarray(nib.load(r.flouter(self.liste, os.path.join(self.d, "flou2"), [1.0], "s2s", "1")[0]["chemin"]).dataobj)
        self.assertAlmostEqual(float(y.sum()), 1.0, places=4)
        self.assertAlmostEqual(float(y.max()), (2 * np.pi) ** -1.5, places=3)


class TestSegmentation(Base):
    def test_cablage_et_noms_reconnus_par_srqc(self):
        r.preparer(self.liste, self.dossier)
        medx = os.path.join(self.d, "medx_factice")
        journal = os.path.join(self.d, "appels.txt")
        # le faux `medx` écrit un fichier à l'endroit de --output et note ses arguments
        executable(medx, f'echo "$@" >> {journal}\nwhile [ $# -gt 0 ]; do [ "$1" = --output ] && touch "$2"; shift; done\n')
        lignes = r.segmenter(self.dossier, medx)
        appels = open(journal).read().splitlines()
        self.assertEqual(len(appels), 6)  # fetalbet + bounti par volume
        self.assertTrue(all(a.split()[0] in ("fetalbet", "bounti") for a in appels))
        self.assertEqual(sum(a.startswith("fetalbet") for a in appels), 3)
        for l in lignes:
            self.assertTrue(l["mask"].endswith("_mask.nii.gz") and l["seg"].endswith("_dseg.nii.gz"))
            self.assertTrue(os.path.exists(l["mask"]) and os.path.exists(l["seg"]))
        # l'entrée de medx est la copie écrêtée, pas l'original
        self.assertTrue(all("ecrete_" in a.split()[2] for a in appels))
        # en parallèle : même résultat
        self.assertEqual(len(r.segmenter(self.dossier, medx, jobs=3)), 3)
        # reprise : rien n'est relancé
        r.segmenter(self.dossier, medx)
        self.assertEqual(len(open(journal).read().splitlines()), 6)
        # le CSV d'entrée de srqc a les colonnes attendues, `ses` vide
        with open(os.path.join(self.dossier, "bids.csv"), newline="") as f:
            lu = list(csv.DictReader(f))
        self.assertEqual(list(lu[0]), list(r.COLONNES_BIDS))
        self.assertEqual([x["name"] for x in lu], ["sub-001", "sub-002", "sub-003"])
        self.assertTrue(all(x["ses"] == "" for x in lu))


class TestScore(Base):
    def prediction(self, scores):
        with open(os.path.join(self.dossier, "prediction.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "fetmrqc_prob_good", "fetmrqc_pred_good", "sub"])
            for n, s in scores:
                w.writerow([n, s, s > 0.7, n])

    def test_conversion(self):
        r.preparer(self.liste, self.dossier)
        self.prediction([("sub-003", 0.25), ("sub-001", 0.9), ("sub-002", 0.5)])  # ordre du fichier différent
        sortie = r.convertir(self.dossier)
        self.assertEqual(
            [(s["sujet"], s["methode"], s["graine"], float(s["score"])) for s in sortie],
            [("A", "m1", "1", 0.9), ("A", "m2", "-", 0.5), ("B", "m1", "1", 0.25)],
        )
        lu = r.lire_tsv(os.path.join(self.dossier, "scores.tsv"), r.COLONNES_SCORES)
        self.assertEqual([float(x["score"]) for x in lu], [0.9, 0.5, 0.25])
        self.assertEqual(r.valider(self.dossier), [])

    def test_score_manquant_ou_nan(self):
        r.preparer(self.liste, self.dossier)
        self.prediction([("sub-001", 0.9), ("sub-002", 0.5)])
        with self.assertRaises(ValueError):
            r.convertir(self.dossier)
        self.prediction([("sub-001", 0.9), ("sub-002", float("nan")), ("sub-003", 0.1)])
        with self.assertRaises(ValueError):
            r.convertir(self.dossier)

    def test_valider_detecte_un_ordre_permute(self):
        r.preparer(self.liste, self.dossier)
        self.prediction([("sub-001", 0.9), ("sub-002", 0.5), ("sub-003", 0.1)])
        r.convertir(self.dossier)
        p = os.path.join(self.dossier, "scores.tsv")
        lignes = r.lire_tsv(p, r.COLONNES_SCORES)
        lignes[0], lignes[1] = lignes[1], lignes[0]
        r.ecrire_tsv(p, r.COLONNES_SCORES, lignes)
        self.assertEqual(len(r.valider(self.dossier)), 1)

    def test_cablage_ligne_de_commande(self):
        env = os.path.join(self.d, "env")
        os.makedirs(os.path.join(env, "bin"))
        journal = os.path.join(self.d, "appels.txt")
        executable(os.path.join(env, "bin", "srqc_compute_iqms"), f'echo "iqms $@" >> {journal}\nwhile [ $# -gt 0 ]; do [ "$1" = --out_csv ] && echo x > "$2"; shift; done\n')
        executable(
            os.path.join(env, "bin", "srqc_inference"),
            f'echo "inference $@" >> {journal}\nwhile [ $# -gt 0 ]; do [ "$1" = --out_csv ] && out="$2"; shift; done\n'
            'printf "name,fetmrqc_prob_good,fetmrqc_pred_good\\nsub-001,0.9,True\\nsub-002,0.5,False\\nsub-003,0.1,False\\n" > "$out"\n',
        )
        r.preparer(self.liste, self.dossier)
        ancien = os.path.join(self.dossier, "iqms.csv")
        open(ancien, "w").write("vieux")  # doit être effacé : deux exécutions indépendantes
        open(os.path.join(self.dossier, "bids.csv"), "w").write("name\n")
        sortie = r.scorer(self.dossier, env, "/modele.joblib", 3)
        appels = open(journal).read().splitlines()
        self.assertEqual(len(appels), 2)
        self.assertIn("--bids_csv " + os.path.join(self.dossier, "bids.csv"), appels[0])
        self.assertIn("--nworkers 3", appels[0])
        self.assertIn("--model_path /modele.joblib", appels[1])
        self.assertIn("--dataset " + os.path.join(self.dossier, "iqms_finis.csv"), appels[1])
        self.assertNotEqual(open(ancien).read(), "vieux")
        self.assertEqual([float(s["score"]) for s in sortie], [0.9, 0.5, 0.1])

    def test_assainir_remplace_les_infinis_seulement(self):
        src, dst = os.path.join(self.d, "i.csv"), os.path.join(self.d, "f.csv")
        open(src, "w").write("name,sub,a,b,c\nsub-001,001,1.5,inf,True\nsub-002,002,-inf,2.0,False\nsub-003,003,nan,0.0,True\n")
        journal = os.path.join(self.d, "j.log")
        remplaces = r.assainir(src, dst, journal)
        self.assertEqual(remplaces, [("sub-001", "b"), ("sub-002", "a")])
        lignes = list(csv.DictReader(open(dst)))
        self.assertEqual([(l["a"], l["b"], l["c"]) for l in lignes], [("1.5", "0.0", "True"), ("0.0", "2.0", "False"), ("nan", "0.0", "True")])
        self.assertEqual(open(journal).read().count("remplacée"), 2)
        self.assertIn("inf", open(src).read())  # le fichier d'origine n'est pas touché


if __name__ == "__main__":
    unittest.main()
