#!/usr/bin/env python3
"""F4.0.7 : banc stratifié par mal-posé (Good, Med, Bad, Single) ; docs/ROADMAP.md, section F4.0.7 (validée le 2026-11-17) et ses précisions de mise en œuvre.

Deux bancs, sur les six sujets « banc-dév » de `correspondances.tsv` (écrit par `banc_juge.py preparer`) :
  * `e2` : banc semi-synthétique E2 (mouvement), test Rust `diagnose_the_semi_synthetic_benchmark` d'un binaire figé, aux variables `MEDOXIDE_NIVEAU` et
    `MEDOXIDE_GRAINE_ARTEFACTS` (artefacts injectés par le test) ; un journal par (passe, sujet, niveau, modèle de mouvement) ;
  * `interne` : banc interne (fidélité, usage A de la décision O1) : piles simulées « calmes » de `banc_o2.py` (`<o2>/<code>/passe1/simule`), stacks conservés et artefacts
    injectés ici (mêmes définitions : creux = coupe × 0,3 ; manquante = masque vidé ; flou = gaussienne de 1,5 pixel dans le plan, dans le masque), `medx svr recon` aux défauts
    (`--min-stack-coverage 1` pour Single seulement), recalage rigide sur la source, PSNR et SSIM dans le masque de la source restreint aux voxels couverts par Good, Med et Bad.
  * `analyser` : tableaux et verdict de (i) à (iii) par famille, tels que pré-déclarés.
Les fichiers déjà présents sont réutilisés (reprise). Les résultats restent sous `--racine` (ignoré par Git).
Mention de l'amendement 3 de F4.0.1 pour tout chiffre du banc interne : banc non validé pour la résolution absolue ; réglages de simulation non physiques.
"""

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banc_mesures as bm
import banc_simule as bs

NIVEAUX = ["good", "med", "bad", "single"]
# `medx svr recon --input` exige au moins deux piles : le niveau Single n'existe pas pour le banc interne (précision 10 de F4.0.7).
NIVEAUX_INTERNE = ["good", "med", "bad"]
#: Stacks conservés par niveau (indices dans l'ordre des piles simulées : axial, coronal, sagittal) : un préfixe, comme dans le test Rust.
STACKS = {"good": [0, 1, 2], "med": [0, 1, 2], "bad": [0, 1], "single": [0]}
#: Fractions de coupes (creux de signal, manquantes, floues), tirées indépendamment sans remise.
FRACTIONS = {"good": (0.0, 0.0, 0.0), "med": (0.1, 0.1, 0.0), "bad": (0.3, 0.3, 0.1), "single": (0.0, 0.0, 0.0)}
FACTEUR_CREUX = 0.3
SIGMA_FLOU_PLAN = 1.5
MODELES = ["lisse", "sauts"]
AMPLITUDES = [4, 6]
GRAINES = [1, 2, 3]
FAMILLES_MESURES = ["part_sup_4mm", "mediane_mm"]


def tirer_coupes(n, fraction, rng):
    """`round(fraction · n)` indices de `0..n`, sans remise, croissants."""
    return sorted(int(k) for k in rng.permutation(n)[: int(round(fraction * n))])


def injecter_artefacts(data, masque, niveau, graine, n_stack):
    """Copies de `data` et `masque` (coupes selon l'axe 2) avec les artefacts du niveau : flou dans le plan (normalisé dans le masque), creux de signal, coupes manquantes.

    Les trois tirages sont indépendants, sans remise ; leur graine est (`graine`, `n_stack`)."""
    creux, manquantes, floues = FRACTIONS[niveau]
    n = data.shape[2]
    rng = np.random.default_rng([graine, n_stack])
    c, m, f = tirer_coupes(n, creux, rng), tirer_coupes(n, manquantes, rng), tirer_coupes(n, floues, rng)
    d, mk = data.astype(np.float32).copy(), masque.copy()
    for k in f:
        zone = (mk[:, :, k] > 0).astype(np.float64)
        num = gaussian_filter(d[:, :, k].astype(np.float64) * zone, SIGMA_FLOU_PLAN, mode="nearest")
        den = gaussian_filter(zone, SIGMA_FLOU_PLAN, mode="nearest")
        flou = np.where((zone > 0) & (den > 1e-12), num / np.maximum(den, 1e-12), d[:, :, k])
        d[:, :, k] = flou.astype(np.float32)
    for k in c:
        d[:, :, k] *= np.float32(FACTEUR_CREUX)
    for k in m:
        mk[:, :, k] = 0
    return d, mk


def deriver_piles(simule, sortie, niveau, graine):
    """Écrit dans `sortie` les piles du niveau (stacks conservés, artefacts de graine `graine`) à partir de `sim_acq-<n>_T2w.nii.gz` de `simule` ; renvoie (piles, masques)."""
    os.makedirs(sortie, exist_ok=True)
    piles, masques = [], []
    for n in STACKS[niveau]:
        base = f"sim_acq-{n + 1}"
        img, msk = nib.load(f"{simule}/{base}_T2w.nii.gz"), nib.load(f"{simule}/{base}_mask.nii.gz")
        d, m = injecter_artefacts(np.asarray(img.dataobj, dtype=np.float32), np.asarray(msk.dataobj).astype(np.uint8), niveau, graine, n)
        p, q = f"{sortie}/{base}_T2w.nii.gz", f"{sortie}/{base}_mask.nii.gz"
        nib.save(nib.Nifti1Image(d, img.affine), p)
        nib.save(nib.Nifti1Image(m.astype(np.uint8), msk.affine), q)
        if os.path.exists(f"{simule}/{base}_T2w.json"):
            with open(f"{simule}/{base}_T2w.json") as f, open(f"{sortie}/{base}_T2w.json", "w") as g:
                g.write(f.read())
        piles.append(p)
        masques.append(q)
    return piles, masques


def somme(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


# ---------------------------------------------------------------- banc interne

def tache_interne(a, code, corr, graine, niveau):
    """Reconstruction et recalage d'un (sujet, graine, niveau) ; renvoie le chemin du volume recalé."""
    d = os.path.abspath(f"{a.racine}/p{a.passe}/interne/{code}/g{graine}/{niveau}")
    aligne = f"{d}/aligne.nii.gz"
    if os.path.exists(aligne):
        return aligne
    simule = os.path.abspath(f"{a.o2}/{code}/passe1/simule")
    piles, masques = deriver_piles(simule, f"{d}/piles", niveau, graine)
    options = []
    volume, _, _ = bs.reconstruire(a.medx, piles, masques, options, graine, f"{d}/recon")
    bs.recaler(corr["source"], volume, corr["masque"], aligne)
    return aligne


def mesurer_interne(a, code, corr, graine):
    """Lignes (niveau, psnr, ssim, voxels) d'un (sujet, graine), dans le masque commun de Good, Med et Bad."""
    chemins = {n: os.path.abspath(f"{a.racine}/p{a.passe}/interne/{code}/g{graine}/{n}/aligne.nii.gz") for n in NIVEAUX_INTERNE}
    ref, masque = bs.lire(corr["source"]), bs.lire(corr["masque"]) > 0
    vols = {n: bs.lire(p) for n, p in chemins.items()}
    commun = masque.copy()
    for n in ("good", "med", "bad"):
        commun &= vols[n] > 0
    lignes = []
    for n in NIVEAUX_INTERNE:
        m = bm.mesures_volume(vols[n], ref, commun)
        lignes.append({"code": code, "graine": graine, "niveau": n, "psnr": m["psnr"], "ssim": m["ssim"], "voxels": int(commun.sum()), "volume": somme(f"{a.racine}/p{a.passe}/interne/{code}/g{graine}/{n}/recon/volume.nii.gz")})
    return lignes


def lire_corr(chemin):
    with open(chemin, newline="") as f:
        return {r["code"]: r for r in csv.DictReader(f, delimiter="\t")}


def interne(a):
    corr = lire_corr(a.correspondances)
    taches = [(c, g, n) for c in a.codes for g in a.graines for n in NIVEAUX_INTERNE]
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(tache_interne, a, c, corr[c], g, n): (c, g, n) for c, g, n in taches}
        for f in cf.as_completed(futs):
            f.result()
            print("interne", *futs[f], "fait", flush=True)
    lignes = []
    for c in a.codes:
        for g in a.graines:
            lignes += mesurer_interne(a, c, corr[c], g)
    sortie = f"{a.racine}/p{a.passe}/interne.tsv"
    with open(sortie, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "graine", "niveau", "psnr", "ssim", "voxels", "volume"], delimiter="\t")
        w.writeheader()
        w.writerows(lignes)
    print("écrit", sortie)


# ---------------------------------------------------------------- banc E2

def journal_e2(a, code, niveau, modele):
    return os.path.abspath(f"{a.racine}/p{a.passe}/e2/{code}_{niveau}_{modele}.log")


def lignes_attendues():
    """Une ligne `entete` et une ligne `fin` par (amplitude, graine)."""
    return 2 * len(AMPLITUDES) * len(GRAINES)


def e2_complet(chemin):
    if not os.path.exists(chemin):
        return False
    texte = open(chemin).read()
    return len(re.findall(r"^SEMISYN ", texte, re.M)) == lignes_attendues() and "test result: ok" in texte


def tache_e2(a, code, corr, niveau, modele):
    log = journal_e2(a, code, niveau, modele)
    if e2_complet(log):
        return log
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = dict(os.environ, MEDOXIDE_DONNEES_REELLES=os.path.abspath(a.jeu), MEDOXIDE_SUJET=code, MEDOXIDE_VOLUME=os.path.abspath(corr["source"]), MEDOXIDE_MODELE=modele,
               MEDOXIDE_AMPLITUDES=",".join(map(str, AMPLITUDES)), MEDOXIDE_GRAINES=",".join(map(str, GRAINES)), MEDOXIDE_NIVEAU=niveau, MEDOXIDE_ENTRELACE="1", MEDOXIDE_DETAIL="1")
    with open(log, "w") as f:
        subprocess.run([os.path.abspath(a.binaire), "diagnose_the_semi_synthetic_benchmark", "--ignored", "--nocapture"], env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    return log


def e2(a):
    corr = lire_corr(a.correspondances)
    taches = [(c, n, m) for c in a.codes for n in NIVEAUX for m in MODELES]
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(tache_e2, a, c, corr[c], n, m): (c, n, m) for c, n, m in taches}
        for f in cf.as_completed(futs):
            log = f.result()
            print("e2", *futs[f], "complet" if e2_complet(log) else "INCOMPLET", flush=True)


# ---------------------------------------------------------------- analyse

def lire_journal_e2(texte):
    """`{(amplitude, graine): erreurs finies de la variante fin}` et lignes numériques (SEMISYN, SLICE) du journal, pour la comparaison exacte de (i)."""
    erreurs, lignes = {}, []
    for l in texte.splitlines():
        if l.startswith(("SEMISYN ", "SLICE ")):
            lignes.append(l)
        if l.startswith("SLICE "):
            c = l.split()
            # SLICE modèle amplitude graine variante stack coupe avant après surface ncc
            if c[4] == "fin":
                e = float(c[8])
                if np.isfinite(e):
                    erreurs.setdefault((float(c[2]), int(c[3])), []).append(e)
    return erreurs, lignes


def grandeurs_e2(erreurs):
    e = np.asarray(erreurs)
    return {"part_sup_4mm": float((e > 4).mean()), "part_sup_6mm": float((e > 6).mean()), "mediane_mm": float(np.median(e))}


def monotone_croissant(good, med, bad):
    return good <= med <= bad


def etendue_moyenne(par_sujet):
    """`par_sujet` : {sujet: [valeur par graine]} -> moyenne sur sujets de (max − min des graines)."""
    return float(np.mean([max(v) - min(v) for v in par_sujet.values()]))


def verdict_famille(valeurs, sens):
    """`valeurs[niveau][sujet] = [une valeur par graine]` pour Good, Med, Bad. `sens` = +1 si la grandeur doit croître de Good à Bad (erreur), −1 si elle doit décroître (PSNR).

    (ii) part des sujets (moyenne des graines) monotones de Good à Med à Bad (non strict), tenu si ≥ 80 % ; (iii) écart Good − Bad de la moyenne sur sujets > 3 × la plus grande des étendues
    moyennes entre graines de Good et de Bad. Renvoie un dictionnaire."""
    sujets = sorted(valeurs["good"])
    moy = {n: {s: float(np.mean(valeurs[n][s])) for s in sujets} for n in ("good", "med", "bad")}
    ok = [monotone_croissant(*[sens * moy[n][s] for n in ("good", "med", "bad")]) for s in sujets]
    part = sum(ok) / len(ok)
    ecart = sens * (np.mean(list(moy["bad"].values())) - np.mean(list(moy["good"].values())))
    etendue = max(etendue_moyenne(valeurs["good"]), etendue_moyenne(valeurs["bad"]))
    return {"monotones": f"{sum(ok)}/{len(ok)}", "part": part, "ii": part >= 0.8, "ecart": float(ecart), "etendue": etendue, "iii": bool(ecart > 3 * etendue),
            "moyennes": {n: float(np.mean(list(moy[n].values()))) for n in ("good", "med", "bad")}}


def analyser(a):
    sortie = []
    # --- E2
    tables = {}  # (modele, amplitude, grandeur) -> niveau -> sujet -> [par graine]
    for c in a.codes:
        for n in NIVEAUX:
            for m in MODELES:
                log = journal_e2(a, c, n, m)
                if not e2_complet(log):
                    sortie.append(f"MANQUE journal {log}")
                    continue
                erreurs, _ = lire_journal_e2(open(log).read())
                for (amp, g), e in sorted(erreurs.items()):
                    for nom, v in grandeurs_e2(e).items():
                        tables.setdefault((m, int(amp), nom), {}).setdefault(n, {}).setdefault(c, []).append(v)
    for (m, amp, nom), t in sorted(tables.items()):
        sortie.append(f"E2 {m} ±{amp} {nom} : " + " ; ".join(f"{n} {np.mean([np.mean(v) for v in t[n].values()]):.3f}" for n in NIVEAUX if n in t))
        if all(n in t and len(t[n]) == len(a.codes) for n in ("good", "med", "bad")) and nom in FAMILLES_MESURES:
            sortie.append("   verdict " + json.dumps(verdict_famille(t, +1), ensure_ascii=False))
    # --- banc interne
    chemin = f"{a.racine}/p1/interne.tsv"
    if os.path.exists(chemin):
        lignes = list(csv.DictReader(open(chemin), delimiter="\t"))
        for grandeur in ("psnr", "ssim"):
            t = {}
            for l in lignes:
                t.setdefault(l["niveau"], {}).setdefault(l["code"], []).append(float(l[grandeur]))
            sortie.append(f"INTERNE {grandeur} : " + " ; ".join(f"{n} {np.mean([np.mean(v) for v in t[n].values()]):.3f}" for n in NIVEAUX if n in t))
            if all(n in t for n in ("good", "med", "bad")):
                sortie.append("   verdict " + json.dumps(verdict_famille(t, -1), ensure_ascii=False))
    # --- (i) déterminisme : deux passes identiques
    for c in a.doubles:
        for n in NIVEAUX:
            for m in MODELES:
                l1, l2 = journal_e2(argparse.Namespace(racine=a.racine, passe=1), c, n, m), journal_e2(argparse.Namespace(racine=a.racine, passe=2), c, n, m)
                if os.path.exists(l1) and os.path.exists(l2):
                    sortie.append(f"(i) E2 {c} {n} {m} : " + ("identique" if lire_journal_e2(open(l1).read())[1] == lire_journal_e2(open(l2).read())[1] else "DIFFÉRENT"))
    ch2 = f"{a.racine}/p2/interne.tsv"
    if os.path.exists(chemin) and os.path.exists(ch2):
        l2 = {(l["code"], l["graine"], l["niveau"]): l for l in csv.DictReader(open(ch2), delimiter="\t")}
        for l in csv.DictReader(open(chemin), delimiter="\t"):
            k = (l["code"], l["graine"], l["niveau"])
            if k in l2:
                ecart = max(abs(float(l[x]) - float(l2[k][x])) for x in ("psnr", "ssim"))
                sortie.append(f"(i) interne {k} : volume " + ("identique" if l["volume"] == l2[k]["volume"] else "DIFFÉRENT") + f", écart max PSNR/SSIM {ecart:.2e}")
    print("\n".join(sortie))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    for nom in ("e2", "interne", "analyser"):
        s = sp.add_parser(nom)
        s.add_argument("--racine", required=True)
        s.add_argument("--codes", nargs="+", default=[f"sub-b0{i}" for i in range(1, 7)])
        s.add_argument("--passe", type=int, default=1)
        if nom != "analyser":
            s.add_argument("--correspondances", required=True)
            s.add_argument("--jobs", type=int, default=2)
        if nom == "e2":
            s.add_argument("--binaire", required=True)
            s.add_argument("--jeu", required=True, help="racine BIDS des piles simulées (data/svr/banc/juge/jeu)")
        if nom == "interne":
            s.add_argument("--medx", required=True)
            s.add_argument("--o2", required=True, help="racine de banc_o2.py (piles simulées calmes)")
            s.add_argument("--graines", nargs="+", type=int, default=GRAINES)
        if nom == "analyser":
            s.add_argument("--doubles", nargs="*", default=["sub-b01", "sub-b06"])
    a = p.parse_args()
    {"e2": e2, "interne": interne, "analyser": analyser}[a.cmd](a)


if __name__ == "__main__":
    main()
