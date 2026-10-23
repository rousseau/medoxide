"""Compare le gradient des blocs (graphe unique séquentiel, ou coupe par coupe en parallèle : suffixe `par`) du recalage coupe-coupe : erreur de pose (vérité connue sur les jeux simulés) après l'alignement par blocs puis après chaque balayage net, durée de chaque phase, concordance finale (mesure commune).

Usage : python scripts/analyse_jacobi_blocs.py <dossier> <jeu> [<jeu> ...] ; le dossier contient jacb/<jeu>-<GS-GS|GS-GSpar|J-GS|J-GSpar>/{stderr.txt,traces/} (lancements avec MEDX_S2S_TRACE=1 et MEDX_S2S_TRACE_DIR ;
la variante est <mise à jour des coupes>-<mise à jour des blocs>) ; `real3` désigne un jeu réel sans vérité.
"""
import glob, os, re, sys
import numpy as np, nibabel as nib

D = sys.argv[1]
SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "atlas", "sim_pyrecon")
VARIANTES = ["GS-GS", "GS-GSpar", "J-GS", "J-GSpar"]


def lire_poses(f):
    d = {}
    for l in open(f).read().strip().split("\n"):
        c = l.split("\t")
        if c[0] != "stack":
            d[(int(c[0]), int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    return d


def poser(M, X):
    return (M[:3, :3] @ X.T).T + M[:3, 3]


def kabsch(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, mq - R @ mp


def points(jeu):
    rng = np.random.default_rng(0); pts = {}
    for s in range(3):
        A = nib.load(f"{jeu}/stack{s}.nii.gz").affine
        m = np.asarray(nib.load(f"{jeu}/stack{s}_mask.nii.gz").dataobj) > 0
        for k in range(m.shape[2]):
            ij = np.argwhere(m[:, :, k])
            if len(ij) >= 500:
                sel = ij[rng.choice(len(ij), min(250, len(ij)), replace=False)]
                pts[(s, k)] = (A @ np.c_[sel, np.full(len(sel), k), np.ones(len(sel))].T).T[:, :3]
    return pts


def erreurs(pts, vraies, est):
    cles = list(pts)
    Xe = np.vstack([poser(est[k], pts[k]) for k in cles]); Xv = np.vstack([poser(vraies[k], pts[k]) for k in cles])
    R, t = kabsch(Xe, Xv)
    return np.array([np.sqrt((((R @ poser(est[k], pts[k]).T).T + t - poser(vraies[k], pts[k])) ** 2).sum(1).mean()) for k in cles])


def phases(d):
    t = open(f"{d}/stderr.txt").read()
    g = lambda nom: next((float(m[1]) for m in re.finditer(rf"phase {re.escape(nom)} : ([\d.]+) s", t)), float("nan"))
    interne = re.search(r" en (\d+) s", open(f"{d}/stdout.txt").read())
    conc = re.search(r"mesure commune\) ([\d.]+)", t)
    return g("alignement par blocs"), g("recalage coupe-coupe (total)"), g("reconstruction du volume"), int(interne[1]), float(conc[1])


for jeu in sys.argv[2:]:
    print(f"\n=== {jeu}")
    sim = jeu != "real3"
    if sim:
        vraies = lire_poses(f"{SIM}/{jeu}/poses.tsv"); pts = points(f"{SIM}/{jeu}")
        e0 = erreurs(pts, vraies, {k: np.eye(4) for k in vraies})
        print(f"en-tête (sans correction) : médiane {np.median(e0):.2f} mm")
    print(f"{'variante':8} {'blocs':>7} {'coupe-coupe':>12} {'recon':>6} {'total':>7} {'concordance':>12}" + (f" {'err. après blocs':>17} {'err. après 3 balayages':>23} {'≤1,5 mm':>8}" if sim else ""))
    for v in VARIANTES:
        d = f"{D}/jacb/{jeu}-{v}"
        if not os.path.exists(f"{d}/stdout.txt"):
            print(v, "absent"); continue
        blocs, s2s, recon, total, conc = phases(d)
        ligne = f"{v:8} {blocs:6.0f}s {s2s:11.0f}s {recon:5.0f}s {total:6.0f}s {conc:12.4f}"
        if sim:
            fb = f"{d}/traces/blocs.tsv"  # absent des lancements faits sans cette trace
            eb = np.median(erreurs(pts, vraies, lire_poses(fb))) if os.path.exists(fb) else float("nan")
            fich = sorted(glob.glob(f"{d}/traces/sweep-*.tsv"))
            ef = erreurs(pts, vraies, lire_poses(fich[-1]))
            ligne += f" {eb:15.2f} mm {np.median(ef):21.2f} mm {100*(ef<=1.5).mean():7.0f}%"
        print(ligne)
