"""Valide `medx svr qc` (contrôle qualité des acquisitions avant reconstruction) selon des critères fixés avant de coder.

C1  la prédiction « coupe non recalée » du contrôle (hors_masque, masque_insuffisant, sans_partenaire) égale `registered = 0` du SVR pour >= 99 % des coupes ;
C2  rappel de `signal_faible` >= 90 % sur des coupes assombries (x0,3) ou vidées (0) injectées dans un jeu simulé ;
C3  faux positifs de `signal_faible` <= 2 % des coupes sur les jeux sans défaut ;
C4  AUC de `score_before` pour détecter une coupe à plus de 1,5 mm de sa vraie pose d'en-tête >= 0,8 (moyenne des jeux simulés avec mouvement).

Usage : python scripts/valider_qc_acquisition.py --medx <binaire medx> --sortie <dossier> --jeux STA31_lisse_mvt4_b5 ... [--reel <dossier de 3 piles et masques : stackN.nii.gz et stackN_mask.nii.gz>]
Les jeux simulés sont ceux de `scripts/make_pyrecon_benchmark.py` (data/atlas/sim_pyrecon, avec la vraie pose de chaque coupe dans poses.tsv).
"""
import argparse, os, subprocess
import numpy as np, nibabel as nib

p = argparse.ArgumentParser()
p.add_argument("--medx", required=True)
p.add_argument("--sortie", required=True)
p.add_argument("--jeux", nargs="+", required=True)
p.add_argument("--reel", help="dossier de piles réelles (stack0..2.nii.gz et stack0..2_mask.nii.gz), sans vérité")
p.add_argument("--graine", type=int, default=0)
a = p.parse_args()
SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "atlas", "sim_pyrecon")
os.makedirs(a.sortie, exist_ok=True)
rng = np.random.default_rng(a.graine)


def lire_tsv(f):
    l = [x.rstrip("\n").split("\t") for x in open(f) if x.strip()]
    return [dict(zip(l[0], r)) for r in l[1:]]


def qc(dossier, nom):
    sortie = f"{a.sortie}/{nom}.tsv"
    subprocess.run([a.medx, "svr", "qc", "--input", *[f"{dossier}/stack{s}.nii.gz" for s in range(3)], "--mask", *[f"{dossier}/stack{s}_mask.nii.gz" for s in range(3)], "--output", sortie], check=True, capture_output=True)
    return lire_tsv(sortie)


def svr(dossier, nom):
    q = f"{a.sortie}/{nom}.svr.tsv"
    subprocess.run([a.medx, "svr", "recon", "--motion", "s2s", "--device", "cpu", "--s2s-update", "jacobi", "--s2s-sweeps", "1", "--s2s-coarse-sweeps", "0", "--s2s-block-levels", "0", "--s2s-bias-rounds", "0",
                    "--input", *[f"{dossier}/stack{s}.nii.gz" for s in range(3)], "--mask", *[f"{dossier}/stack{s}_mask.nii.gz" for s in range(3)], "--output", f"{a.sortie}/{nom}.svr.nii.gz", "--quality", q], check=True, capture_output=True)
    return lire_tsv(q)


def structurel(l):
    return l["reason"] in ("hors_masque", "masque_insuffisant", "sans_partenaire")


def accord_svr(lignes, rapport):
    n = len(lignes)
    d = sum(structurel(l) == (r["registered"] == "1") for l, r in zip(lignes, rapport))  # égal = désaccord
    return 100.0 * (n - d) / n, d


def fp_signal(lignes, injectees=()):
    cles = {(l["slice"], l["stack"]) for l in injectees}
    return sum(l["reason"] == "signal_faible" and (l["slice"], l["stack"]) not in cles for l in lignes)


def auc(scores_bas_mauvais, mauvais):
    """AUC : probabilité qu'une coupe mauvaise ait un score plus bas qu'une bonne (Mann-Whitney, égalités comptées 1/2)."""
    m = np.array([s for s, y in zip(scores_bas_mauvais, mauvais) if y]); b = np.array([s for s, y in zip(scores_bas_mauvais, mauvais) if not y])
    if len(m) == 0 or len(b) == 0:
        return float("nan")
    return float(((m[:, None] < b[None, :]).sum() + 0.5 * (m[:, None] == b[None, :]).sum()) / (len(m) * len(b)))


def erreurs_en_tete(jeu):
    poser = lambda M, X: (M[:3, :3] @ X.T).T + M[:3, 3]
    vraies = {}
    for l in open(f"{SIM}/{jeu}/poses.tsv"):
        c = l.split("\t")
        if c[0] != "stack":
            vraies[(int(c[0]), int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    err = {}
    for s in range(3):
        A = nib.load(f"{SIM}/{jeu}/stack{s}.nii.gz").affine
        m = np.asarray(nib.load(f"{SIM}/{jeu}/stack{s}_mask.nii.gz").dataobj) > 0
        for k in range(m.shape[2]):
            ij = np.argwhere(m[:, :, k])
            if len(ij) < 500:
                continue
            sel = ij[np.random.default_rng(0).choice(len(ij), min(250, len(ij)), replace=False)]
            X = (A @ np.c_[sel, np.full(len(sel), k), np.ones(len(sel))].T).T[:, :3]
            # écart entre le placement d'en-tête (identité) et le placement vrai, mouvement commun non retiré (c'est ce que le contrôle voit)
            err[(s, k)] = float(np.sqrt(((poser(vraies[(s, k)], X) - X) ** 2).sum(1).mean()))
    return err


def injecter(jeu):
    """Copie du jeu avec, par pile, une coupe assombrie (x0,3) et une vidée, prises au milieu (masque >= 500 pixels, voisines aussi, deux défauts non adjacents)."""
    d = f"{a.sortie}/{jeu}-defauts"; os.makedirs(d, exist_ok=True)
    verite = []
    for s in range(3):
        img = nib.load(f"{SIM}/{jeu}/stack{s}.nii.gz"); data = np.asarray(img.dataobj).astype(np.float32)
        masque = nib.load(f"{SIM}/{jeu}/stack{s}_mask.nii.gz")
        pix = (np.asarray(masque.dataobj) > 0).sum((0, 1))
        bonnes = [k for k in range(2, data.shape[2] - 2) if all(pix[j] >= 500 for j in (k - 1, k, k + 1))]
        choix = []
        for k in rng.permutation(bonnes):
            if all(abs(k - j) >= 4 for j in choix):
                choix.append(int(k))
            if len(choix) == 2:
                break
        for k, f in zip(choix, (0.3, 0.0)):
            data[:, :, k] *= f
            verite.append({"stack": f"stack{s}", "slice": str(k), "facteur": f})
        nib.save(nib.Nifti1Image(data, img.affine, img.header), f"{d}/stack{s}.nii.gz")
        nib.save(masque, f"{d}/stack{s}_mask.nii.gz")
    return d, verite


res = {"c1": [], "c2": [], "c3": [], "c4": []}
for jeu in a.jeux:
    print(f"\n=== {jeu}")
    d = f"{SIM}/{jeu}"
    propre = qc(d, f"{jeu}-propre")
    rapport = svr(d, jeu)
    # noms de pile : le contrôle écrit le nom de fichier sans extension (stack0...)
    pct, desaccords = accord_svr(propre, rapport)
    fp = fp_signal(propre)
    print(f"C1 accord avec le SVR : {pct:.1f} % ({desaccords} désaccord(s) sur {len(propre)})   C3 faux positifs de signal_faible : {fp}/{len(propre)} = {100*fp/len(propre):.1f} %")
    res["c1"].append(pct); res["c3"].append(100 * fp / len(propre))
    dd, verite = injecter(jeu)
    abimees = qc(dd, f"{jeu}-defauts")
    trouvees = sum(any(l["stack"] == v["stack"] and l["slice"] == v["slice"] and l["reason"] == "signal_faible" for l in abimees) for v in verite)
    fausses = fp_signal(abimees, [{"stack": v["stack"], "slice": v["slice"]} for v in verite])
    print(f"C2 défauts injectés : {trouvees}/{len(verite)} retrouvés ; signal_faible en plus sur les piles abîmées : {fausses}")
    res["c2"].append(100 * trouvees / len(verite))
    if jeu.find("mvt0") < 0:
        err = erreurs_en_tete(jeu)
        sc = [(float(l["score_before"]), err[(int(l["stack"][5:]), int(l["slice"]))] > 1.5) for l in propre if l["score_before"] != "n/a" and (int(l["stack"][5:]), int(l["slice"])) in err]
        v = auc([x for x, _ in sc], [y for _, y in sc])
        print(f"C4 AUC de score_before pour une coupe à > 1,5 mm de sa vraie pose : {v:.2f} ({sum(y for _, y in sc)} mauvaises sur {len(sc)})")
        res["c4"].append(v)
if a.reel:
    print("\n=== réel (sans vérité)")
    propre = qc(a.reel, "reel-propre"); rapport = svr(a.reel, "reel")
    pct, desaccords = accord_svr(propre, rapport); fp = fp_signal(propre)
    print(f"C1 accord avec le SVR : {pct:.1f} % ({desaccords} désaccord(s) sur {len(propre)})   C3 signal_faible : {fp}/{len(propre)}")
    res["c1"].append(pct); res["c3"].append(100 * fp / len(propre))

print("\n=== critères")
print(f"C1 (>= 99 % sur chaque jeu) : min {min(res['c1']):.1f} % -> {'atteint' if min(res['c1']) >= 99 else 'NON atteint'}")
print(f"C2 (rappel >= 90 %) : moyenne {np.mean(res['c2']):.0f} %, min {min(res['c2']):.0f} % -> {'atteint' if np.mean(res['c2']) >= 90 else 'NON atteint'}")
print(f"C3 (<= 2 % de faux positifs) : max {max(res['c3']):.1f} % -> {'atteint' if max(res['c3']) <= 2 else 'NON atteint'}")
if res["c4"]:
    print(f"C4 (AUC moyenne >= 0,8) : {np.nanmean(res['c4']):.2f} -> {'atteint' if np.nanmean(res['c4']) >= 0.8 else 'NON atteint'}")
