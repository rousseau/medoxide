"""Cas de test des pas d'optimisation de l'estimation jointe de medoxide-ulf (docs/ULF.md, P4 bis, critère 4), en HDF5
lisible par Rust.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # utils_cpr.py, inr.py et compare_methods.py de fmi (cb7174b)
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_jointe.py synthetique <medoxide>/crates/medoxide-ulf/tests/donnees/jointe_synthetique.h5
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_jointe.py p1 <inputs.npz> <sortie.h5>   # un plan de P1

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
- `pertes_jointes/<terme>` : valeurs non pondérées des termes à chaque pas joint (avant le pas), `totale` pondérée.
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


def synthetique(sortie):
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
    ecrire(sortie, SYNTHETIQUE, k_reg, k_inv, temps, espacement, b0_init, (12.5, 108.0))


def p1(entrees, sortie):
    checkpointer()
    i = np.load(entrees)
    t = i["time_array"]
    temps = t[:, 0, 0]
    assert np.allclose(t, temps[:, None, None]), "temps non constant le long des axes de phase"
    ecrire(sortie, P1, i["k_reg"], i["k_inv"], temps, i["spacing_mm"].tolist(), i["b0_warm_start"].astype(np.float64),
           i["b0_bounds"].tolist())


if __name__ == "__main__":
    {"synthetique": synthetique, "p1": p1}[sys.argv[1]](*sys.argv[2:])
