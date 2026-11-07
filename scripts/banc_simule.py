#!/usr/bin/env python3
"""Banc de vérité terrain à simulateur interne (docs/ROADMAP.md, F4.0.1) : simule, reconstruit, mesure.

Pour chaque sujet du tirage (`scripts/tirage_banc.py`), chaque condition, chaque variante et chaque graine :
  1. `medx svr simulate` depuis la reconstruction CRL du sujet (récupérée sur Garage) avec un mouvement vrai connu ;
  2. `medx svr recon` avec les options de la variante (`--s2s-seed` = graine) ;
  3. recalage rigide du volume sur la source (corrélation dans le masque de la source, SimpleITK), ré-échantillonné sur la grille de la source ;
  4. mesures dans le masque de la source (`banc_mesures.py` : PSNR et SSIM après normalisation affine, NCC, netteté) ;
  5. avec `--bounti` : Dice de PRÉSERVATION par groupe de tissus, `medx bounti` sur le volume recalé contre la segmentation BOUNTI-19 de la source
     (ce n'est pas une justesse : il n'y a pas de segmentation manuelle) ;
  6. erreur de pose contre les poses vraies, après retrait du mouvement commun.
Dose-réponse : `--flous 0.5 1 2` ajoute, pour la première variante et la première graine, les mêmes mesures sur le volume reconstruit flouté
(σ en voxels de la grille reconstruite) avant recalage.

Les fichiers déjà présents sont réutilisés (reprise). Aucun identifiant n'est écrit dans le code : le tirage et les chemins sont des arguments, les
résultats restent sous `--racine` (ignoré par Git). Une variante s'écrit `nom=options` (options de `medx svr recon`, entre guillemets) ; `defaut=` est la
variante par défaut.

Usage : python3 scripts/banc_simule.py --tirage data/svr/banc/tirage.tsv --jeu dev --racine data/svr/banc \\
    --gabarits pile1.nii.gz pile2.nii.gz pile3.nii.gz --conditions calme lisse sauts \\
    --variantes defaut= "entete=--s2s-sweeps 0 --s2s-coarse-sweeps 0 --s2s-bias-rounds 0 --s2s-block-levels 0" --graines 1 2 3
"""

import argparse
import csv
import os
import resource
import shlex
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_mesures as bm

#: Conditions de séries : nom -> (modèle de mouvement de `medx svr simulate`, amplitude en degrés et mm). Amplitudes de E2 (ROADMAP F4.0.1).
CONDITIONS = {"calme": ("none", 0.0), "lisse": ("smooth", 6.0), "sauts": ("jumps", 4.0)}

SOURCE_T2W = "{sujet}_{session}_res-0p5mm_desc-resampled_T2w.nii.gz"
SOURCE_MASQUE = "{sujet}_{session}_res-0p5mm_desc-brain_mask.nii.gz"
SOURCE_BOUNTI = "{sujet}_{session}_res-0p5mm_desc-bounti19_dseg.nii.gz"


def parser_variantes(liste):
    """`["nom=options …", …]` -> `[(nom, [mots …]), …]` ; les options sont découpées comme par un shell."""
    sortie = []
    for v in liste:
        nom, _, options = v.partition("=")
        if not nom or nom in dict(sortie):
            raise ValueError(f"variante invalide ou répétée : {v!r}")
        sortie.append((nom, shlex.split(options)))
    return sortie


def lire_tirage(chemin, jeu):
    """Lignes du tirage (`sujet`, `session`, `age`, `strate`, `jeu`), filtrées par jeu (`dev`, `test` ou `tous`)."""
    with open(chemin, newline="") as f:
        lignes = list(csv.DictReader(f, delimiter="\t"))
    return [r for r in lignes if jeu == "tous" or r["jeu"] == jeu]


def executer(commande, **kw):
    """Lance une commande, en s'arrêtant avec sa sortie d'erreur si elle échoue."""
    r = subprocess.run(commande, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"échec de {' '.join(map(str, commande))} :\n{r.stderr[-2000:]}")
    return r


def recuperer_source(remote, sujet, session, dossier):
    """Copie depuis Garage la reconstruction à 0,5 mm, le masque du cerveau et la segmentation BOUNTI-19 de la session ; renvoie les trois chemins."""
    os.makedirs(dossier, exist_ok=True)
    noms = [m.format(sujet=sujet, session=session) for m in (SOURCE_T2W, SOURCE_MASQUE, SOURCE_BOUNTI)]
    if not all(os.path.exists(f"{dossier}/{n}") for n in noms):
        filtres = [x for n in noms for x in ("--include", n)]
        executer(["rclone", "copy", f"{remote}/derivatives/fetal-folding-geometry/{sujet}/{session}/anat", dossier, *filtres])
    return [f"{dossier}/{n}" for n in noms]


def simuler(medx, source, masque, gabarits, dossier, condition, bruit_pct, biais_pct, echelle_pct, graine, moteur="independent"):
    """`medx svr simulate` (repris si les poses vraies existent) ; renvoie (piles, masques, poses vraies)."""
    motion, amplitude = CONDITIONS[condition]
    poses = f"{dossier}/sim_poses_vraies.tsv"
    if not os.path.exists(poses):
        executer(
            [medx, "svr", "simulate", "--source", source, "--source-mask", masque, "--template", *gabarits, "--output-dir", dossier, "--name", "sim",
             "--motion", motion, "--amplitude", str(amplitude), "--interleaved", "--noise-pct", str(bruit_pct), "--bias-pct", str(biais_pct),
             "--slice-scale-pct", str(echelle_pct), "--seed", str(graine), "--forward-model", moteur]
        )
    piles = sorted(f"{dossier}/{f}" for f in os.listdir(dossier) if f.endswith("_T2w.nii.gz"))
    return piles, [p.replace("_T2w.nii.gz", "_mask.nii.gz") for p in piles], poses


def reconstruire(medx, piles, masques, options, graine, dossier):
    """`medx svr recon` (repris si le volume existe) ; renvoie (volume, poses estimées, temps CPU en s, ou nan si repris)."""
    os.makedirs(dossier, exist_ok=True)
    volume, poses = f"{dossier}/volume.nii.gz", f"{dossier}/poses.tsv"
    if os.path.exists(volume) and os.path.exists(poses):
        return volume, poses, float("nan")
    avant = resource.getrusage(resource.RUSAGE_CHILDREN)
    executer([medx, "svr", "recon", "--input", *piles, "--mask", *masques, "--output", volume, "--poses", poses, "--s2s-seed", str(graine), *options])
    apres = resource.getrusage(resource.RUSAGE_CHILDREN)
    return volume, poses, (apres.ru_utime - avant.ru_utime) + (apres.ru_stime - avant.ru_stime)


def flouter(volume, sigma, sortie):
    """Écrit `volume` flouté par une gaussienne de `sigma` voxels (`sigma = 0` : copie)."""
    img = nib.load(volume)
    donnees = np.asarray(img.dataobj, dtype=np.float32)
    if sigma > 0:
        donnees = gaussian_filter(donnees, sigma).astype(np.float32)
    nib.save(nib.Nifti1Image(donnees, img.affine), sortie)
    return sortie


def recaler(source, volume, masque, sortie):
    """Recalage rigide de `volume` sur `source` (corrélation dans le masque de la source, trois niveaux, un seul fil pour la reproductibilité) puis
    ré-échantillonnage linéaire sur la grille de la source, écrit dans `sortie`. Renvoie la transformation (SimpleITK)."""
    import SimpleITK as sitk

    fixe = sitk.ReadImage(source, sitk.sitkFloat32)
    mobile = sitk.ReadImage(volume, sitk.sitkFloat32)
    m = sitk.ReadImage(masque, sitk.sitkUInt8)
    depart = sitk.CenteredTransformInitializer(fixe, mobile, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    r = sitk.ImageRegistrationMethod()
    r.SetNumberOfThreads(1)
    r.SetMetricAsCorrelation()
    r.SetMetricFixedMask(m)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.1, 12345)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300, gradientMagnitudeTolerance=1e-8)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(depart, inPlace=False)
    t = r.Execute(fixe, mobile)
    sitk.WriteImage(sitk.Resample(mobile, fixe, t, sitk.sitkLinear, 0.0), sortie)
    return t


def lire(chemin):
    return np.asarray(nib.load(chemin).dataobj, dtype=np.float64)


def points_par_coupe(masques, n_par=250, graine=0):
    """Points (monde, à la pose d'en-tête) de pixels de masque tirés dans chaque coupe d'au moins 500 pixels de masque ; `masques` dans l'ordre des piles."""
    rng = np.random.default_rng(graine)
    sortie = {}
    for s, chemin in enumerate(masques):
        img = nib.load(chemin)
        m = np.asarray(img.dataobj) > 0
        for k in range(m.shape[2]):
            ij = np.argwhere(m[:, :, k])
            if len(ij) < 500:
                continue
            sel = ij[rng.choice(len(ij), min(n_par, len(ij)), replace=False)]
            sortie[(s, k)] = (img.affine @ np.c_[sel, np.full(len(sel), k), np.ones(len(sel))].T).T[:, :3]
    return sortie


def lignes_de_mesure(source, masque, bounti_source, volume, dossier, medx, avec_bounti, points, poses_vraies, poses_estimees):
    """Recale `volume` sur `source` et renvoie le dictionnaire des mesures (y compris Dice par groupe si demandé et erreur de pose)."""
    os.makedirs(dossier, exist_ok=True)
    aligne = f"{dossier}/aligne.nii.gz"
    recaler(source, volume, masque, aligne)
    ref, m, x = lire(source), lire(masque) > 0, lire(aligne)
    r = bm.mesures_volume(x, ref, m)
    if avec_bounti:
        seg = f"{dossier}/bounti.nii.gz"
        if not os.path.exists(seg):
            executer([medx, "bounti", "--input", aligne, "--output", seg])
        r.update({f"dice_{g}": d for g, d in bm.dice_par_groupe(np.asarray(nib.load(seg).dataobj), np.asarray(nib.load(bounti_source).dataobj)).items()})
    if poses_vraies and poses_estimees:
        e = np.array(list(bm.erreur_de_pose(points, bm.lire_poses(poses_vraies), bm.lire_poses(poses_estimees)).values()))
        r.update({"pose_part_1_5mm": float((e <= 1.5).mean()), "pose_mediane_mm": float(np.median(e)), "pose_p90_mm": float(np.percentile(e, 90))})
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tirage", required=True)
    p.add_argument("--jeu", choices=["dev", "test", "tous"], default="dev")
    p.add_argument("--sujets", nargs="+", default=None, help="restreint aux sujets listés (identifiants du tirage)")
    p.add_argument("--racine", required=True)
    p.add_argument("--gabarits", nargs="+", required=True)
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    p.add_argument("--variantes", nargs="+", required=True)
    p.add_argument("--graines", nargs="+", type=int, default=[1])
    p.add_argument("--flous", nargs="*", type=float, default=[])
    p.add_argument("--bruit-pct", type=float, default=5.0)
    p.add_argument("--biais-pct", type=float, default=0.0)
    p.add_argument("--echelle-pct", type=float, default=0.0)
    p.add_argument("--graine-simulation", type=int, default=7)
    p.add_argument("--moteur", choices=["independent", "reconstruction"], default="independent")
    p.add_argument("--bounti", action="store_true")
    p.add_argument("--medx", default="target/release/medx")
    p.add_argument("--remote", default=os.environ.get("OXIDE_REMOTE", "garage:fetal-crl2025"))
    p.add_argument("--sortie", required=True, help="dossier des TSV (un par variante)")
    a = p.parse_args()

    variantes = parser_variantes(a.variantes)
    racine, sortie = os.path.abspath(a.racine), os.path.abspath(a.sortie)
    os.makedirs(sortie, exist_ok=True)
    lignes = lire_tirage(a.tirage, a.jeu)
    if a.sujets:
        lignes = [r for r in lignes if r["sujet"] in a.sujets]
    resultats = {nom: [] for nom, _ in variantes}
    for r in lignes:
        sujet, session = r["sujet"], r["session"]
        source, masque, bounti_source = recuperer_source(a.remote, sujet, session, f"{racine}/sources/{sujet}_{session}")
        for condition in a.conditions:
            d_sim = f"{racine}/simul/{sujet}_{session}/{condition}"
            piles, masques_piles, poses_vraies = simuler(a.medx, source, masque, a.gabarits, d_sim, condition, a.bruit_pct, a.biais_pct, a.echelle_pct, a.graine_simulation, a.moteur)
            points = points_par_coupe(masques_piles)
            for vi, (nom, options) in enumerate(variantes):
                for gi, graine in enumerate(a.graines):
                    d_rec = f"{racine}/recon/{sujet}_{session}/{condition}/{nom}_g{graine}"
                    volume, poses_est, cpu = reconstruire(a.medx, piles, masques_piles, options, graine, d_rec)
                    flous = [0.0] + (a.flous if (vi == 0 and gi == 0) else [])
                    for sigma in flous:
                        d_mes = f"{d_rec}/mesure_flou{sigma:g}"
                        v = volume if sigma == 0 else flouter(volume, sigma, f"{d_rec}/volume_flou{sigma:g}.nii.gz")
                        m = lignes_de_mesure(source, masque, bounti_source, v, d_mes, a.medx, a.bounti and sigma == 0, points, poses_vraies, poses_est if sigma == 0 else None)
                        ligne = {"sujet": sujet, "strate": r["strate"], "jeu": r["jeu"], "condition": condition, "variante": nom, "graine": graine, "flou": sigma,
                                 "temps_cpu_s": cpu, **m}
                        resultats[nom].append(ligne)
                        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in ligne.items() if k in ("sujet", "condition", "variante", "graine", "flou", "psnr", "ssim")), flush=True)
    for nom, rows in resultats.items():
        if not rows:
            continue
        cles = list(dict.fromkeys(k for row in rows for k in row))
        with open(f"{sortie}/{nom}.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cles, delimiter="\t")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
