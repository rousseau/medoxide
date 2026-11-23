"""Cas de test des pas d'optimisation de l'estimation jointe de medoxide-ulf (docs/ULF.md, P4 bis, critère 4), en HDF5
lisible par Rust.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # utils_cpr.py, inr.py et compare_methods.py de fmi (cb7174b)
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_jointe.py synthetique <medoxide>/crates/medoxide-ulf/tests/donnees/jointe_synthetique.h5
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_jointe.py p1 <inputs.npz> <sortie.h5>   # un plan de P1
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_jointe.py entrees <inputs.npz> <sortie.h5>   # entrées seules (critère 5)
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_jointe.py methode_b synthetique|<inputs.npz> <sortie.h5>   # P5, critère 2

Reprend `compare_methods.estimate_joint` (sans RED) en float64 sur CPU, réseaux initialisés par fmi (graine 0, tables
dans ± 1e-4) :
- N_PREFIT pas de `utils_cpr.prefit` pour le champ (cible b0_init) puis pour l'image (cible `averaged_corrected`) ;
- depuis les mêmes poids initiaux, le démarrage complet (300 pas par réseau), puis N_JOINT pas de la boucle jointe
  (Adam à groupes, décroissance cosinus sur N_ITER_JOINT = 200). Les pas joints partent de là et non de l'état après
  N_PREFIT pas : |u| y est presque constant (écart-type relatif 1,5e-5 sur le cas synthétique), et la variance locale
  de la LNCC du terme image y perd une dizaine de chiffres (docs/ULF.md, critère 4).

Contenu :
- entrées : `k_reg`, `k_inv` (bruts), `temps`, `espacement_mm`, `b0_init`, `bornes_b0` ; attributs `n_prefit`,
  `n_joint`, `n_iter_joint` ; `champ_config` et `image_config` (attributs de la grille et du MLP, comme cas_inr.py) ;
- `cible_image` : magnitude cible du démarrage à chaud de l'image, à l'échelle des données normalisées ;
- `etat_<e>/{champ,image}/niveau_<l>/table`, `.../couche_<c>/{poids,biais}` (forme PyTorch (sorties, entrées)) et
  `etat_<e>/champ/{s,o}` : paramètres à l'état e = 0 (initial), 1 (après N_PREFIT pas de démarrage du champ, champ
  seul), 2 (après N_PREFIT pas de démarrage de l'image, image seule), 3 (après le démarrage complet, 300 pas par réseau
  depuis l'état 0 : point de départ des pas joints), puis 4, 5, … après chaque pas joint (les deux réseaux) ;
- `pertes_jointes/<terme>` : valeurs non pondérées des termes à chaque pas joint (avant le pas), `totale` pondérée ;
- `plancher_pytorch/etat_<e>/<réseau>/<paramètre>` (e = 1, 2) : écart relatif de chaque paramètre entre le calcul de
  PyTorch à un thread et au nombre de threads par défaut, sur les démarrages (mode `plancher`, ajouté à un fichier
  existant : `cas_jointe.py plancher <cas.h5>`). À l'initialisation, la sortie du réseau du champ est presque constante
  et son centrage perd des chiffres ; seuil amendé le 2026-11-22 (docs/ULF.md, critère 4).
"""

import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_methods as cm  # noqa: E402
import utils_cpr  # noqa: E402
from cas_inr import P1, SYNTHETIQUE  # noqa: E402
from cas_pertes import checkpointer  # noqa: E402

N_PREFIT = 2
N_JOINT = 2


def ecrire_reseau(groupe, net):
    for l, niveau in enumerate(net.encoding.levels):
        groupe.create_dataset(f"niveau_{l}/table", data=niveau.embedding.weight.detach().numpy())
    for c, m in enumerate(m for m in net.mlp.net if isinstance(m, torch.nn.Linear)):
        groupe.create_dataset(f"couche_{c}/poids", data=m.weight.detach().numpy())
        groupe.create_dataset(f"couche_{c}/biais", data=m.bias.detach().numpy())


def ecrire_etat(f, e, champ=None, image=None):
    if champ is not None:
        g = f.create_group(f"etat_{e}/champ")
        ecrire_reseau(g, champ.net)
        g.create_dataset("s", data=champ.scale_param.item())
        g.create_dataset("o", data=champ.offset_param.item())
    if image is not None:
        ecrire_reseau(f.create_group(f"etat_{e}/image"), image.net)


def ecrire(chemin, cas, k_reg, k_inv, temps, espacement, b0_init, bornes):
    dims = tuple(k_reg.shape)
    k_reg, k_inv = torch.tensor(k_reg), torch.tensor(k_inv)
    t = torch.tensor(np.broadcast_to(np.asarray(temps)[:, None, None], dims).copy())
    b0_init = torch.tensor(b0_init, dtype=torch.float64)
    sigma = [cm.LNCC_SIGMA_MM / h for h in espacement]
    data_scale = float(torch.maximum(utils_cpr.ifft(k_reg, normalize=True).abs().max(),
                                     utils_cpr.ifft(k_inv, normalize=True).abs().max()))
    k_reg_n, k_inv_n = k_reg / data_scale, k_inv / data_scale
    def reseaux():
        return (utils_cpr.INRField(**cas["champ"], b0_min_hz=bornes[0], b0_max_hz=bornes[1], seed=0),
                utils_cpr.INRImage(**cas["image"], seed=0))

    field, image = reseaux()
    support = utils_cpr.object_support_mask(k_reg, k_inv)

    with h5py.File(chemin, "w") as f:
        for nom, v in dict(k_reg=k_reg.numpy(), k_inv=k_inv.numpy(), temps=np.asarray(temps, dtype=np.float64),
                           espacement_mm=np.asarray(espacement, dtype=np.float64), b0_init=b0_init.numpy(),
                           bornes_b0=np.asarray(bornes, dtype=np.float64)).items():
            f.create_dataset(nom, data=v)
        f.attrs["n_prefit"], f.attrs["n_joint"], f.attrs["n_iter_joint"] = N_PREFIT, N_JOINT, cm.N_ITER_JOINT
        for nom in ("champ", "image"):
            g = f.create_group(f"{nom}_config")
            for k, v in cas[nom].items():
                g.attrs[k] = v
        ecrire_etat(f, 0, field, image)

        # démarrage à chaud (compare_methods.estimate_joint, JOINT_WARM_START), N_PREFIT premiers pas
        utils_cpr.prefit(lambda: field.b0(dims), field.parameters(), b0_init, n_iter=N_PREFIT)
        ecrire_etat(f, 1, champ=field)
        u_mag = cm.averaged_corrected(k_reg, k_inv, t, b0_init)
        u_target = (u_mag / data_scale).to(torch.complex128)
        f.create_dataset("cible_image", data=u_target.real.numpy())
        utils_cpr.prefit(lambda: image.image(dims), image.parameters(), u_target, n_iter=N_PREFIT)
        ecrire_etat(f, 2, image=image)

        # démarrage à chaud complet (300 pas par réseau, comme la méthode), depuis les mêmes poids initiaux : point de
        # départ des pas joints
        field, image = reseaux()
        utils_cpr.prefit(lambda: field.b0(dims), field.parameters(), b0_init)
        utils_cpr.prefit(lambda: image.image(dims), image.parameters(), u_target)
        ecrire_etat(f, 3, field, image)

        # boucle jointe, N_JOINT premiers pas
        opt = torch.optim.Adam(image.net.param_groups(lr=cm.LR) + field.param_groups(lr=cm.LR, scalar_lr=cm.SCALAR_LR))
        sched = cm.cosine_schedule(opt, cm.N_ITER_JOINT)
        pertes = {n: [] for n in ("fidelity", "image", "b0", "smooth", "support", "scalaires", "totale")}
        for it in range(N_JOINT):
            opt.zero_grad()
            u = image.image(dims)
            b0 = field.b0(dims)
            terms = utils_cpr.joint_losses(u, b0, k_reg_n, k_inv_n, t, mode=cm.JOINT_LOSS_MODE,
                                           b0_mode=cm.B0_LOSS_MODE, lncc_sigma_vox=sigma,
                                           image_mode=cm.JOINT_IMAGE_MODE,
                                           image_term_updates_b0=cm.JOINT_IMAGE_UPDATES_B0)
            smooth = utils_cpr.bending_penalty(b0, espacement)
            out_of_support = u.abs().square()[~support].mean()
            loss = sum(cm.W_JOINT[name] * term for name, term in terms.items())
            loss = (loss + cm.W_SMOOTH_JOINT * smooth + cm.W_SUPPORT * out_of_support
                    + cm.W_SCALAR_PENALTY * field.scalar_penalty())
            loss.backward()
            opt.step()
            sched.step()
            for nom, v in {**terms, "smooth": smooth, "support": out_of_support,
                           "scalaires": field.scalar_penalty(), "totale": loss}.items():
                pertes[nom].append(v.item())
            print(f"pas joint {it} : perte {loss.item():.6e}")
            ecrire_etat(f, 4 + it, field, image)
        for nom, v in pertes.items():
            f.create_dataset(f"pertes_jointes/{nom}", data=np.asarray(v))
    print(f"écrit : {chemin}")
    plancher(chemin)


def parametres(net, champ=None):
    """Paramètres aplatis, nommés comme dans le fichier."""
    p = {f"niveau_{l}/table": n.embedding.weight for l, n in enumerate(net.encoding.levels)}
    for c, m in enumerate(m for m in net.mlp.net if isinstance(m, torch.nn.Linear)):
        p[f"couche_{c}/poids"], p[f"couche_{c}/biais"] = m.weight, m.bias
    if champ is not None:
        p["s"], p["o"] = champ.scale_param, champ.offset_param
    return {k: v.detach().clone().reshape(-1) for k, v in p.items()}


def plancher(chemin):
    """Plancher de reproductibilité de PyTorch lui-même sur les deux démarrages à chaud (états 1 et 2) : écart relatif
    de chaque paramètre entre le calcul à un thread et le calcul au nombre de threads par défaut (seul l'ordre des
    sommes change). Écrit dans `plancher_pytorch/etat_<e>/<réseau>/<paramètre>` ; amendement du 2026-11-22
    (docs/ULF.md, critère 4)."""
    with h5py.File(chemin, "a") as f:
        cas = {n: {k: (v.item() if hasattr(v, "item") else v) for k, v in f[f"{n}_config"].attrs.items()}
               for n in ("champ", "image")}
        b0_init = torch.tensor(f["b0_init"][()])
        bornes = f["bornes_b0"][()]
        dims = tuple(b0_init.shape)
        n_prefit = int(f.attrs["n_prefit"])
        u_target = torch.tensor(f["cible_image"][()]).to(torch.complex128)
        defaut = torch.get_num_threads()

        def demarrages(threads):
            torch.set_num_threads(threads)
            field = utils_cpr.INRField(**cas["champ"], b0_min_hz=bornes[0], b0_max_hz=bornes[1], seed=0)
            image = utils_cpr.INRImage(**cas["image"], seed=0)
            utils_cpr.prefit(lambda: field.b0(dims), field.parameters(), b0_init, n_iter=n_prefit)
            utils_cpr.prefit(lambda: image.image(dims), image.parameters(), u_target, n_iter=n_prefit)
            return parametres(field.net, field), parametres(image.net)

        a, b = demarrages(defaut), demarrages(1)
        torch.set_num_threads(defaut)
        if "plancher_pytorch" in f:
            del f["plancher_pytorch"]
        for (e, nom), pa, pb in zip(((1, "champ"), (2, "image")), a, b):
            for k in pa:
                ref = np.asarray(f[f"etat_{e}/{nom}/{k}"][()], dtype=np.float64).reshape(-1)
                assert np.array_equal(pa[k].numpy(), ref), f"{k} : le calcul par défaut ne redonne pas le fichier"
                v = float((pa[k] - pb[k]).norm() / pa[k].norm())
                f.create_dataset(f"plancher_pytorch/etat_{e}/{nom}/{k}", data=v)
        print(f"plancher écrit ({defaut} threads contre 1) : {chemin}")


def donnees_synthetiques():
    rng = np.random.default_rng(20261123)
    dims = (32, 24, 16)
    espacement = (2.0, 2.0, 3.0)
    temps = -1.0e-3 + 6.25e-5 * np.arange(dims[0])
    x = [np.linspace(-1, 1, n) for n in dims]
    X = np.stack(np.meshgrid(*x, indexing="ij"))
    objet = np.exp(-((X[0] / 0.35) ** 2 + (X[1] / 0.3) ** 2 + (X[2] / 0.3) ** 4) * 3)
    u_vrai = objet * (1 + 0.3 * np.cos(4 * X[0]) * np.sin(3 * X[1])) * np.exp(0.4j * X[2])
    b0_vrai = 80 * X[0] + 40 * X[1] ** 2 - 30 * X[0] * X[2]
    t = np.broadcast_to(temps[:, None, None], dims).copy()
    bruit = lambda: 0.02 * (rng.standard_normal(dims) + 1j * rng.standard_normal(dims))  # noqa: E731
    from fmi.reconstruction.operators import cpr as cprmod
    k_reg = cprmod.time_segmented_cpr_adjoint(u_vrai, t, b0_vrai, 0.0, 0) + bruit()
    k_inv = cprmod.time_segmented_cpr_adjoint(u_vrai, t, -b0_vrai, 0.0, 0) + bruit()
    b0_init = 0.8 * b0_vrai + 5 * rng.standard_normal(dims)
    return k_reg, k_inv, temps, espacement, b0_init, (12.5, 108.0)


def synthetique(sortie):
    ecrire(sortie, SYNTHETIQUE, *donnees_synthetiques())


def p1(entrees, sortie):
    checkpointer()
    i = np.load(entrees)
    t = i["time_array"]
    temps = t[:, 0, 0]
    assert np.allclose(t, temps[:, None, None]), "temps non constant le long des axes de phase"
    ecrire(sortie, P1, i["k_reg"], i["k_inv"], temps, i["spacing_mm"].tolist(), i["b0_warm_start"].astype(np.float64),
           i["b0_bounds"].tolist())


def entrees(entrees_npz, sortie):
    """Entrées d'un plan de P1 pour l'exemple `jointe_p1` de medoxide-ulf (critère 5) : données seules, sans pas."""
    i = np.load(entrees_npz)
    t = i["time_array"]
    temps = t[:, 0, 0]
    assert np.allclose(t, temps[:, None, None]), "temps non constant le long des axes de phase"
    with h5py.File(sortie, "w") as f:
        for nom, v in dict(k_reg=i["k_reg"], k_inv=i["k_inv"], temps=temps.astype(np.float64),
                           espacement_mm=i["spacing_mm"].astype(np.float64),
                           b0_init=i["b0_warm_start"].astype(np.float64),
                           bornes_b0=i["b0_bounds"].astype(np.float64)).items():
            f.create_dataset(nom, data=v)
    print(f"écrit : {sortie}")


N_PAS_B = 2


def methode_b(source, sortie):
    """Méthode B (`compare_methods.estimate_inr`, docs/ULF.md, P5, critère 2), float64 sur CPU, champ initialisé par fmi
    (graine 0) : `source` vaut `synthetique` (cas versionné, grille du champ de cas_inr.SYNTHETIQUE) ou un `inputs.npz`
    de P1 (grille de la méthode B, cas_inr.P1_B).

    Contenu : entrées comme `p1` (sans `b0_init`) ; `champ_config` ; `etat_<e>/champ/...` après e pas (e = 0 … N_PAS_B) ;
    `pertes_b/{b0,smooth,scalaires,totale}` avant chaque pas ; `grad_b0` : gradient de b0 + W_SMOOTH · lissage par
    rapport à b0 (Hz) à l'état 0 ; `plancher_pytorch/etat_<e>/champ/<paramètre>` : écart entre le calcul par défaut et le
    calcul à un thread."""
    from cas_inr import P1_B
    if source == "synthetique":
        k_reg, k_inv, temps, espacement, _, bornes = donnees_synthetiques()
        config = SYNTHETIQUE["champ"]
    else:
        checkpointer()
        i = np.load(source)
        t = i["time_array"]
        temps = t[:, 0, 0]
        assert np.allclose(t, temps[:, None, None]), "temps non constant le long des axes de phase"
        k_reg, k_inv, espacement, bornes = i["k_reg"], i["k_inv"], i["spacing_mm"].tolist(), i["b0_bounds"].tolist()
        config = P1_B["champ"]
    dims = tuple(k_reg.shape)
    k_reg_t, k_inv_t = torch.tensor(k_reg), torch.tensor(k_inv)
    t = torch.tensor(np.broadcast_to(np.asarray(temps)[:, None, None], dims).copy())
    sigma = [cm.LNCC_SIGMA_MM / h for h in espacement]

    def pertes(b0):
        terme = utils_cpr.self_consistency_loss(b0, k_reg_t, k_inv_t, t, mode=cm.B0_LOSS_MODE, lncc_sigma_vox=sigma)
        return terme, utils_cpr.bending_penalty(b0, espacement)

    def pas(n_pas):
        field = utils_cpr.INRField(**config, b0_min_hz=bornes[0], b0_max_hz=bornes[1], seed=0)
        opt = torch.optim.Adam(field.param_groups(lr=cm.LR, scalar_lr=cm.SCALAR_LR))
        sched = cm.cosine_schedule(opt, cm.N_ITER_INR)
        etats, bilans = [parametres(field.net, field)], []
        for _ in range(n_pas):
            opt.zero_grad()
            b0 = field.b0(dims)
            terme, smooth = pertes(b0)
            loss = terme + cm.W_SMOOTH * smooth + cm.W_SCALAR_PENALTY * field.scalar_penalty()
            loss.backward()
            opt.step()
            sched.step()
            bilans.append((terme.item(), smooth.item(), field.scalar_penalty().item(), loss.item()))
            etats.append(parametres(field.net, field))
        return field, etats, bilans

    with h5py.File(sortie, "w") as f:
        for nom, v in dict(k_reg=np.asarray(k_reg), k_inv=np.asarray(k_inv), temps=np.asarray(temps, dtype=np.float64),
                           espacement_mm=np.asarray(espacement, dtype=np.float64),
                           bornes_b0=np.asarray(bornes, dtype=np.float64)).items():
            f.create_dataset(nom, data=v)
        g = f.create_group("champ_config")
        for k, v in config.items():
            g.attrs[k] = v
        f.attrs["n_pas"], f.attrs["n_iter"] = N_PAS_B, cm.N_ITER_INR
        f.attrs["w_smooth"], f.attrs["w_scalaires"] = cm.W_SMOOTH, cm.W_SCALAR_PENALTY

        field = utils_cpr.INRField(**config, b0_min_hz=bornes[0], b0_max_hz=bornes[1], seed=0)
        ecrire_etat(f, 0, champ=field)
        b0 = field.b0(dims).detach().requires_grad_()
        terme, smooth = pertes(b0)
        (terme + cm.W_SMOOTH * smooth).backward()
        f.create_dataset("b0_etat_0", data=b0.detach().numpy())
        f.create_dataset("grad_b0", data=b0.grad.numpy())

        defaut = torch.get_num_threads()
        _, etats, bilans = pas(N_PAS_B)
        for e in range(1, N_PAS_B + 1):
            g = f.create_group(f"etat_{e}/champ")
            for k, v in etats[e].items():
                forme = f[f"etat_0/champ/{k}"].shape
                g.create_dataset(k, data=v.numpy().reshape(forme))
        for j, nom in enumerate(("b0", "smooth", "scalaires", "totale")):
            f.create_dataset(f"pertes_b/{nom}", data=np.asarray([b[j] for b in bilans]))
        torch.set_num_threads(1)
        _, etats1, _ = pas(N_PAS_B)
        torch.set_num_threads(defaut)
        for e in range(1, N_PAS_B + 1):
            for k in etats[e]:
                v = float((etats[e][k] - etats1[e][k]).norm() / etats[e][k].norm())
                f.create_dataset(f"plancher_pytorch/etat_{e}/champ/{k}", data=v)
        print(f"pertes avant chaque pas : {bilans}")
    print(f"écrit : {sortie}")


if __name__ == "__main__":
    {"synthetique": synthetique, "p1": p1, "plancher": plancher, "entrees": entrees,
     "methode_b": methode_b}[sys.argv[1]](*sys.argv[2:])
