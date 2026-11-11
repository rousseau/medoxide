#!/usr/bin/env python3
"""Score de qualité FetMRQC_SR de volumes reconstruits (docs/ROADMAP.md, F4.0.5, point 0.5).

FetMRQC_SR (Sanchez et al. 2025, Apache-2.0) est installé hors dépôt, dans un environnement Python 3.9 à commits épinglés (voir F4.0.5) ; ce
script le lance et convertit sa sortie en un TSV `sujet, methode, graine, score` (score `qcglobal` = probabilité continue de « bon »).
Segmentation (O3) : masque et étiquettes viennent de `medx fetalbet` et `medx bounti`, rangés sous le nom que `srqc_segmentation` reconnaît
comme déjà fait (`sub-NNN_mask.nii.gz`, `sub-NNN_dseg.nii.gz`) : l'image Docker d'origine n'est pas lancée.

Étapes (sous-commandes), toutes reprenables (ce qui existe est réutilisé) :
  preparer  <liste.tsv> <dossier>   un faux sujet BIDS par volume (`sub-NNN`), `correspondance.tsv` (numéro -> sujet, methode, graine, chemin)
  segmenter <dossier>               masque (fetalbet) et étiquettes (bounti) de chaque volume, sur une copie écrêtée au centile 99,5
                                    (le prétraitement « robuste » de `srqc_segmentation`) ; puis `bids.csv`
  flouter   <liste.tsv> <dossier>   (critère (ii)) copies floutées (gaussien isotrope, σ en voxels) des volumes de la liste ; écrit `<dossier>/liste.tsv`
  scorer    <dossier>               `srqc_compute_iqms` puis `srqc_inference` : `iqms.csv`, `prediction.csv`, `scores.tsv`
  convertir <dossier>               seulement la conversion de `prediction.csv` en `scores.tsv` (`--valider` : vérifie, sans calcul, qu'elle est
                                    cohérente avec `correspondance.tsv`)

`liste.tsv` : colonnes `sujet`, `methode`, `graine`, `chemin` (en-tête obligatoire). Aucun identifiant de sujet n'est écrit dans le code.
"""

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

COLONNES_LISTE = ("sujet", "methode", "graine", "chemin")
COLONNES_BIDS = ("name", "sub", "ses", "im", "mask", "seg")
COLONNES_SCORES = ("sujet", "methode", "graine", "score")
CENTILE_ECRETAGE = 99.5  # même valeur que `srqc_segmentation --robust_prepro`


def numero(i):
    """Identifiant à trois chiffres du i-ième volume (à partir de 1) : `001`."""
    return f"{i:03d}"


def lire_tsv(chemin, colonnes):
    """Lit un TSV à en-tête ; ValueError si une colonne attendue manque."""
    with open(chemin, newline="") as f:
        lignes = list(csv.DictReader(f, delimiter="\t"))
    if lignes and not set(colonnes) <= set(lignes[0]):
        raise ValueError(f"{chemin} : colonnes attendues {colonnes}, trouvées {tuple(lignes[0])}")
    return lignes


def ecrire_tsv(chemin, colonnes, lignes):
    with open(chemin, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(colonnes)
        w.writerows([[ligne[c] for c in colonnes] for ligne in lignes])


def chemins(dossier, n):
    """Chemins d'un volume numéroté n : dossier BIDS, volume, masque, étiquettes (noms reconnus par `srqc_segmentation`)."""
    base = os.path.join(dossier, "bids", f"sub-{n}", "anat")
    return {
        "anat": base,
        "im": os.path.join(base, f"sub-{n}_T2w.nii.gz"),
        "mask": os.path.join(base, f"sub-{n}_mask.nii.gz"),
        "seg": os.path.join(base, f"sub-{n}_dseg.nii.gz"),
    }


def preparer(liste, dossier):
    """Crée l'arborescence BIDS (liens vers les volumes, tels que produits) et `correspondance.tsv`. Retourne la table."""
    table = []
    for i, ligne in enumerate(lire_tsv(liste, COLONNES_LISTE), start=1):
        n = numero(i)
        c = chemins(dossier, n)
        os.makedirs(c["anat"], exist_ok=True)
        source = os.path.realpath(ligne["chemin"])
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        if os.path.lexists(c["im"]):
            os.remove(c["im"])
        os.symlink(source, c["im"])
        table.append({"numero": n, **{k: ligne[k] for k in COLONNES_LISTE}})
    ecrire_tsv(os.path.join(dossier, "correspondance.tsv"), ("numero",) + COLONNES_LISTE, table)
    return table


def ecreter(source, sortie, centile=CENTILE_ECRETAGE):
    """Copie `source` écrêtée entre 0 et son centile (comme `srqc_segmentation --robust_prepro`) ; l'original n'est pas modifié."""
    im = nib.load(source)
    data = np.asarray(im.dataobj, dtype=np.float64)
    haut = np.percentile(data, centile)
    nib.save(nib.Nifti1Image(data.clip(0, haut).astype(np.float32), im.affine), sortie)


def flouter(liste, dossier, sigmas, methode, graine):
    """Pour chaque ligne de `liste` de la méthode et de la graine données, une copie floutée par σ (voxels) ; écrit `<dossier>/liste.tsv`.

    La méthode des copies est `<methode>_flou<σ>` (σ = 0 : l'original, tel quel). Retourne les lignes écrites."""
    os.makedirs(dossier, exist_ok=True)
    sortie = []
    for ligne in lire_tsv(liste, COLONNES_LISTE):
        if ligne["methode"] != methode or ligne["graine"] != graine:
            continue
        im = nib.load(ligne["chemin"])
        data = np.asarray(im.dataobj, dtype=np.float32)
        for sigma in sigmas:
            nom = f"{ligne['sujet']}_{methode}_flou{sigma:g}.nii.gz"
            chemin = os.path.join(dossier, nom)
            if sigma == 0:
                chemin = os.path.realpath(ligne["chemin"])
            elif not os.path.exists(chemin):
                nib.save(nib.Nifti1Image(gaussian_filter(data, sigma).astype(np.float32), im.affine), chemin)
            sortie.append({"sujet": ligne["sujet"], "methode": f"{methode}_flou{sigma:g}", "graine": graine, "chemin": chemin})
    ecrire_tsv(os.path.join(dossier, "liste.tsv"), COLONNES_LISTE, sortie)
    return sortie


def commande_fetalbet(medx, entree, sortie):
    return [medx, "fetalbet", "--input", entree, "--output", sortie]


def commande_bounti(medx, entree, sortie, device):
    return [medx, "bounti", "--input", entree, "--output", sortie, "--device", device]


def lancer(cmd, journal):
    """Lance `cmd`, ajoute sa sortie au journal ; CalledProcessError si le code de retour n'est pas 0."""
    with open(journal, "a") as j:
        j.write("$ " + " ".join(cmd) + "\n")
        j.flush()
        subprocess.run(cmd, stdout=j, stderr=subprocess.STDOUT, check=True)


def segmenter_un(dossier, n, medx, device, journal):
    """Masque et étiquettes du volume n (ce qui existe est gardé)."""
    c = chemins(dossier, n)
    if os.path.exists(c["mask"]) and os.path.exists(c["seg"]):
        return
    ecrete = os.path.join(dossier, "journal", f"ecrete_{n}.nii.gz")
    ecreter(c["im"], ecrete)
    if not os.path.exists(c["mask"]):
        lancer(commande_fetalbet(medx, ecrete, c["mask"]), journal)
    if not os.path.exists(c["seg"]):
        lancer(commande_bounti(medx, ecrete, c["seg"], device), journal)
    os.remove(ecrete)


def segmenter(dossier, medx, device="cpu", jobs=1):
    """Masque et étiquettes de chaque volume de `correspondance.tsv`, `jobs` volumes à la fois, puis `bids.csv`."""
    table = lire_tsv(os.path.join(dossier, "correspondance.tsv"), ("numero",) + COLONNES_LISTE)
    os.makedirs(os.path.join(dossier, "journal"), exist_ok=True)
    journal = os.path.join(dossier, "journal", "segmentation.log")
    with ThreadPoolExecutor(jobs) as pool:
        for f in [pool.submit(segmenter_un, dossier, t["numero"], medx, device, journal) for t in table]:
            f.result()  # relève l'erreur du premier échec
    lignes = []
    for t in table:
        c = chemins(dossier, t["numero"])
        lignes.append({"name": f"sub-{t['numero']}", "sub": t["numero"], "ses": "", "im": c["im"], "mask": c["mask"], "seg": c["seg"]})
    ecrire_csv_bids(os.path.join(dossier, "bids.csv"), lignes)
    return lignes


def ecrire_csv_bids(chemin, lignes):
    """CSV d'entrée de `srqc_compute_iqms` : colonnes `name, sub, ses, im, mask, seg` ; `ses` vide (pas de session)."""
    with open(chemin, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES_BIDS, lineterminator="\n")
        w.writeheader()
        w.writerows(lignes)


def commande_iqms(env, bids_csv, iqms_csv, nworkers):
    return [os.path.join(env, "bin", "srqc_compute_iqms"), "--bids_csv", bids_csv, "--out_csv", iqms_csv, "--nworkers", str(nworkers)]


def commande_inference(env, modele, iqms_csv, prediction_csv):
    return [
        os.path.join(env, "bin", "srqc_inference"),
        "--dataset", iqms_csv,
        "--model_path", modele,
        "--out_csv", prediction_csv,
    ]


def assainir(iqms_csv, finis_csv, journal=None):
    """Copie `iqms_csv` en `finis_csv` en remplaçant chaque valeur infinie par 0.0 ; retourne la liste des `(nom, colonne)` remplacés.

    Pourquoi : `nrmse` et `nmae` divisent par une norme de la référence, nulle si une coupe voisine l'est sur la fenêtre ; la moyenne vaut alors
    `inf`, et la forêt aléatoire (scikit-learn) refuse l'entrée. FetMRQC_SR traite déjà un NaN ainsi (`metrics_sr.py` : valeur 0.0, colonne
    `*_nan` à vrai) ; on étend cette convention à `inf` (précision 17 de F4.0.5). Le fichier d'IQMs d'origine n'est pas modifié ; les
    remplacements sont écrits au journal.
    """
    with open(iqms_csv, newline="") as f:
        lecteur = csv.DictReader(f)
        champs = lecteur.fieldnames
        lignes = list(lecteur)
    remplaces = []
    for ligne in lignes:
        for c in champs:
            try:
                infini = np.isinf(float(ligne[c]))
            except ValueError:
                continue
            if infini:
                ligne[c] = "0.0"
                remplaces.append((ligne["name"], c))
    with open(finis_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=champs, lineterminator="\n")
        w.writeheader()
        w.writerows(lignes)
    if journal and remplaces:
        with open(journal, "a") as j:
            for nom, c in remplaces:
                j.write(f"valeur infinie remplacée par 0.0 : {nom} {c}\n")
    return remplaces


def convertir(dossier, prediction="prediction.csv", scores="scores.tsv"):
    """`prediction.csv` (sortie de `srqc_inference`) -> TSV `sujet, methode, graine, score`, dans l'ordre de `correspondance.tsv`.

    ValueError si un volume n'a pas de score fini (NaN ou absent) : un score manquant est une erreur, pas une ligne vide.
    """
    table = lire_tsv(os.path.join(dossier, "correspondance.tsv"), ("numero",) + COLONNES_LISTE)
    with open(os.path.join(dossier, prediction), newline="") as f:
        par_nom = {r["name"]: r for r in csv.DictReader(f)}
    sortie = []
    for t in table:
        r = par_nom.get(f"sub-{t['numero']}")
        if r is None:
            raise ValueError(f"pas de score pour sub-{t['numero']}")
        s = float(r["fetmrqc_prob_good"])
        if not np.isfinite(s):
            raise ValueError(f"score non fini pour sub-{t['numero']} : {r['fetmrqc_prob_good']}")
        sortie.append({"sujet": t["sujet"], "methode": t["methode"], "graine": t["graine"], "score": repr(s)})
    ecrire_tsv(os.path.join(dossier, scores), COLONNES_SCORES, sortie)
    return sortie


def scorer(dossier, env, modele, nworkers, prediction="prediction.csv", iqms="iqms.csv", scores="scores.tsv"):
    """IQMs, inférence, conversion. Le calcul des IQMs repart de zéro (le CSV d'IQMs existant est effacé) pour que deux exécutions soient indépendantes."""
    os.makedirs(os.path.join(dossier, "journal"), exist_ok=True)
    journal = os.path.join(dossier, "journal", "score.log")
    iqms_csv = os.path.join(dossier, iqms)
    if os.path.exists(iqms_csv):
        os.remove(iqms_csv)
    lancer(commande_iqms(env, os.path.join(dossier, "bids.csv"), iqms_csv, nworkers), journal)
    finis_csv = os.path.splitext(iqms_csv)[0] + "_finis.csv"
    assainir(iqms_csv, finis_csv, journal)
    lancer(commande_inference(env, modele, finis_csv, os.path.join(dossier, prediction)), journal)
    return convertir(dossier, prediction, scores)


def valider(dossier):
    """Vérifie la cohérence `correspondance.tsv` / `bids.csv` / `scores.tsv` sans rien calculer ; retourne la liste des problèmes."""
    problemes = []
    table = lire_tsv(os.path.join(dossier, "correspondance.tsv"), ("numero",) + COLONNES_LISTE)
    numeros = [t["numero"] for t in table]
    if len(set(numeros)) != len(numeros):
        problemes.append("numéros dupliqués dans correspondance.tsv")
    for fichier in ("bids.csv",):
        p = os.path.join(dossier, fichier)
        if os.path.exists(p):
            with open(p, newline="") as f:
                noms = [r["name"] for r in csv.DictReader(f)]
            if noms != [f"sub-{n}" for n in numeros]:
                problemes.append(f"{fichier} : noms différents de correspondance.tsv")
    p = os.path.join(dossier, "scores.tsv")
    if os.path.exists(p):
        scores = lire_tsv(p, COLONNES_SCORES)
        if len(scores) != len(table):
            problemes.append(f"scores.tsv : {len(scores)} lignes pour {len(table)} volumes")
        for s in scores:
            if not np.isfinite(float(s["score"])):
                problemes.append(f"score non fini : {s}")
        if [(s["sujet"], s["methode"], s["graine"]) for s in scores] != [(t["sujet"], t["methode"], t["graine"]) for t in table]:
            problemes.append("scores.tsv : l'ordre ou les colonnes sujet/méthode/graine diffèrent de correspondance.tsv")
    return problemes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("preparer")
    a.add_argument("liste")
    a.add_argument("dossier")
    a = sp.add_parser("segmenter")
    a.add_argument("dossier")
    a.add_argument("--medx", required=True, help="binaire `medx`")
    a.add_argument("--device", default="cpu")
    a.add_argument("--jobs", type=int, default=1, help="volumes segmentés en parallèle")
    a = sp.add_parser("flouter")
    a.add_argument("liste")
    a.add_argument("dossier")
    a.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    a.add_argument("--methode", default="s2s")
    a.add_argument("--graine", default="1")
    a = sp.add_parser("scorer")
    a.add_argument("dossier")
    a.add_argument("--env", required=True, help="environnement Python de FetMRQC_SR (contient bin/srqc_*)")
    a.add_argument("--modele", required=True, help="modèle .joblib (somme SHA-256 vérifiée par l'appelant avant de lancer : c'est un pickle)")
    a.add_argument("--nworkers", type=int, default=5)
    a.add_argument("--prediction", default="prediction.csv")
    a.add_argument("--iqms", default="iqms.csv")
    a.add_argument("--scores", default="scores.tsv")
    a = sp.add_parser("convertir")
    a.add_argument("dossier")
    a.add_argument("--valider", action="store_true")
    args = p.parse_args(argv)
    # chemins réels : les dossiers de travail peuvent être des liens vers un autre clone (worktree), qui peut disparaître
    for k in ("dossier", "liste"):
        if hasattr(args, k):
            setattr(args, k, os.path.realpath(getattr(args, k)))
    if args.cmd == "preparer":
        print(f"{len(preparer(args.liste, args.dossier))} volumes")
    elif args.cmd == "segmenter":
        print(f"{len(segmenter(args.dossier, args.medx, args.device, args.jobs))} volumes segmentés")
    elif args.cmd == "flouter":
        print(f"{len(flouter(args.liste, args.dossier, args.sigmas, args.methode, args.graine))} volumes")
    elif args.cmd == "scorer":
        print(f"{len(scorer(args.dossier, args.env, args.modele, args.nworkers, args.prediction, args.iqms, args.scores))} scores")
    elif args.valider:
        pb = valider(args.dossier)
        print("\n".join(pb) if pb else "cohérent")
        return 1 if pb else 0
    else:
        print(f"{len(convertir(args.dossier))} scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
