"""Compare Gauss-Seidel et Jacobi du recalage coupe-coupe : temps, balayages jusqu'à convergence, erreur de pose (vérité connue sur les jeux simulés), concordance finale (mesure commune)."""
import glob, os, re, sys
import numpy as np, nibabel as nib

S = sys.argv[1]
SIM = "/Users/rousseau/Exp/medoxide/data/atlas/sim_pyrecon"

def lire_poses(f):
    d = {}
    for l in open(f).read().strip().split("\n"):
        c = l.split("\t")
        if c[0] == "stack":
            continue
        d[(int(c[0]), int(c[1]))] = np.array([float(v) for v in c[2:]]).reshape(4, 4, order="F")
    return d

def kabsch(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, mq - R @ mp

def poser(M, X):
    return (M[:3, :3] @ X.T).T + M[:3, 3]

def points(jeu):
    rng = np.random.default_rng(0); pts = {}
    for s in range(3):
        A = nib.load(f"{jeu}/stack{s}.nii.gz").affine
        m = np.asarray(nib.load(f"{jeu}/stack{s}_mask.nii.gz").dataobj) > 0
        for k in range(m.shape[2]):
            ij = np.argwhere(m[:, :, k])
            if len(ij) < 500:
                continue
            sel = ij[rng.choice(len(ij), min(250, len(ij)), replace=False)]
            pts[(s, k)] = (A @ np.c_[sel, np.full(len(sel), k), np.ones(len(sel))].T).T[:, :3]
    return pts

def erreurs(pts, vraies, est):
    cles = list(pts)
    Xe = np.vstack([poser(est[k], pts[k]) for k in cles]); Xv = np.vstack([poser(vraies[k], pts[k]) for k in cles])
    R, t = kabsch(Xe, Xv)
    return np.array([np.sqrt((((R @ poser(est[k], pts[k]).T).T + t - poser(vraies[k], pts[k])) ** 2).sum(1).mean()) for k in cles])

def trace(variante_dir):
    lignes = []
    for l in open(f"{variante_dir}/stderr.txt"):
        m = re.match(r"s2s balayage (\d+) sigma ([\d.]+) net (\w+) : (\d+) mises à jour, concordance ([\d.]+) -> ([\d.]+), déplacement moyen ([\d.]+) mm, max ([\d.]+) mm, ([\d.]+) s", l)
        if m:
            lignes.append(dict(i=int(m[1]), sigma=float(m[2]), net=m[3] == "true", n=int(m[4]), avant=float(m[5]), apres=float(m[6]), dep=float(m[7]), depmax=float(m[8]), t=float(m[9])))
    return lignes

def final(variante_dir):
    for l in open(f"{variante_dir}/stderr.txt"):
        m = re.search(r"mesure commune\) ([\d.]+)", l)
        if m:
            return float(m[1])

def temps(variante_dir):
    t = open(f"{variante_dir}/time.txt").read()
    real = float(re.search(r"([\d.]+) real", t)[1]); rss = int(re.search(r"(\d+)\s+maximum", t)[1]) / 1e9
    u = open(f"{variante_dir}/stderr.txt").read()
    user = float(re.search(r"([\d.]+) user", u)[1]); sys_ = float(re.search(r"([\d.]+) sys", u)[1])
    return real, user + sys_, rss

for jeu in sys.argv[2:]:
    print(f"\n=== {jeu}")
    sim = jeu != "real3"
    if sim:
        vraies = lire_poses(f"{SIM}/{jeu}/poses.tsv"); pts = points(f"{SIM}/{jeu}")
        e0 = erreurs(pts, vraies, {k: np.eye(4) for k in vraies})
        print(f"en-tête (sans correction) : médiane {np.median(e0):.2f} mm, ≤1,5 mm {100*(e0<=1.5).mean():.0f} %")
    for var in ["gauss-seidel", "jacobi"]:
        d = f"{S}/jac/{jeu}-{var}"
        if not os.path.exists(f"{d}/time.txt"):
            print(var, "absent"); continue
        real, cpu, rss = temps(d)
        tr = trace(d)
        nets = [t for t in tr if t["net"]]
        tfin = sum(t["t"] for t in nets); tgro = sum(t["t"] for t in tr if not t["net"])
        print(f"-- {var}: durée totale {real:.0f} s (CPU {cpu:.0f} s, RSS {rss:.1f} Go) ; balayages grossiers {tgro:.0f} s, nets {tfin:.0f} s ({tfin/len(nets):.0f} s/balayage) ; concordance finale (mesure commune) {final(d):.4f}")
        # erreur par balayage net (vérité) ; fichiers sweep-XX : 0..5 grossiers, 6..11 nets
        fich = sorted(glob.glob(f"{d}/traces/sweep-*.tsv"))
        err = []
        if sim:
            for f in fich:
                e = erreurs(pts, vraies, lire_poses(f))
                err.append(e)
        ligne_med, ligne_p, ligne_dep = [], [], []
        for j, t in enumerate(tr):
            if not t["net"]:
                continue
            idx = t["i"]
            if sim and idx < len(err):
                ligne_med.append(f"{np.median(err[idx]):.2f}"); ligne_p.append(f"{100*(err[idx]<=1.5).mean():.0f}")
            ligne_dep.append(f"{t['dep']:.2f}")
        print("   balayage net n°          " + " ".join(f"{n+1:>5}" for n in range(len(nets))))
        if sim:
            print("   erreur médiane (mm)      " + " ".join(f"{v:>5}" for v in ligne_med))
            print("   coupes ≤ 1,5 mm (%)      " + " ".join(f"{v:>5}" for v in ligne_p))
            if err:
                med = [np.median(err[t["i"]]) for t in nets]
                conv = next((n + 1 for n in range(1, len(med)) if abs(med[n] - med[n - 1]) < 0.05), len(med))
                print(f"   convergence (variation de l'erreur médiane < 0,05 mm entre deux balayages nets) : balayage {conv}")
        print("   déplacement moyen (mm)   " + " ".join(f"{v:>5}" for v in ligne_dep))
        depl = [t["dep"] for t in nets]
        convd = next((n + 1 for n, v in enumerate(depl) if v < 0.1), None)
        print(f"   convergence (déplacement moyen < 0,1 mm) : {'balayage ' + str(convd) if convd else 'non atteinte en ' + str(len(nets)) + ' balayages'}")
