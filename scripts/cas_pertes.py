"""Cas de test des pertes de l'estimation jointe de medoxide-ulf (docs/ULF.md, P4 bis, critère 3), en HDF5 lisible par Rust.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # utils_cpr.py et compare_methods.py de fmi (référence cb7174b)
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_pertes.py synthetique <medoxide>/crates/medoxide-ulf/tests/donnees/pertes_synthetique.h5
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_pertes.py p1 <inputs.npz> <outputs.npz> <sortie.h5>   # un plan de P1

Termes de `utils_cpr.joint_losses` avec les réglages de `compare_methods.estimate_joint` (mode `magnitude`, terme image
`lncc` sans gradient vers le champ, terme B0 `lncc`, fenêtre LNCC de 4 mm), plus le lissage (`bending_penalty`) et la
pénalité hors support (moyenne de |u|² hors de `object_support_mask`), tous non pondérés, en float64, au point (u, b0) :
- cas p1 : sorties de l'étape 1 de P1 (u ramenée à l'échelle des données normalisées, comme dans la méthode) ;
- cas synthétique (32 × 24 × 16, versionné) : objet lisse simulé par le modèle direct, avec bruit, et un point (u, b0)
  écarté de la vérité.

Contenu : entrées (`k_reg`, `k_inv`, `temps`, `espacement_mm`, `sigma_lncc_vox`, `u`, `b0`), `echelle` (data_scale),
`masque_support` (uint8), et pour chaque terme `terme_<nom>/valeur`, `terme_<nom>/grad_u` (complexe, convention de PyTorch
∂L/∂Re + i ∂L/∂Im) et `terme_<nom>/grad_b0`. `ecart_pytorch_f32/terme_<nom>/{valeur,grad_u,grad_b0}` : écart relatif au
float64 du même calcul fait par PyTorch en complex64 / float32, plancher de conditionnement auquel se compare le calcul
`f32` de Rust (amendement du 2026-11-22, docs/ULF.md, même règle qu'au critère 1).
"""

import sys

import h5py
import numpy as np
import torch

import compare_methods as cm
import utils_cpr
from fmi.reconstruction.operators import cpr as cprmod

TERMES = ("fidelity", "image", "b0", "smooth", "support")


def checkpointer():
    """Recalcul à la volée des opérateurs CPR traversés par un gradient (mémoire : P1 en complex128 sur CPU)."""
    from torch.utils.checkpoint import checkpoint

    def envelopper(f):
        def g(x, t, d_b0, t_shift, readout_axis):
            if torch.is_grad_enabled() and isinstance(x, torch.Tensor) and (x.requires_grad or d_b0.requires_grad):
                return checkpoint(f, x, t, d_b0, t_shift, readout_axis, use_reentrant=False)
            return f(x, t, d_b0, t_shift, readout_axis)
        return g

    cprmod.time_segmented_cpr = envelopper(cprmod.time_segmented_cpr)
    cprmod.time_segmented_cpr_adjoint = envelopper(cprmod.time_segmented_cpr_adjoint)


def ecrire(chemin, k_reg, k_inv, temps, espacement, u_donnees, b0):
    """u_donnees : image à l'échelle des données (comme la sortie de estimate_joint)."""
    k_reg, k_inv = torch.tensor(k_reg), torch.tensor(k_inv)
    dims = tuple(k_reg.shape)
    t = torch.tensor(np.broadcast_to(np.asarray(temps)[:, None, None], dims).copy())
    sigma = [cm.LNCC_SIGMA_MM / h for h in espacement]
    echelle = float(torch.maximum(utils_cpr.ifft(k_reg, normalize=True).abs().max(),
                                  utils_cpr.ifft(k_inv, normalize=True).abs().max()))
    support = utils_cpr.object_support_mask(k_reg, k_inv)
    u = (torch.tensor(u_donnees) / echelle).to(torch.complex128).requires_grad_()
    b0t = torch.tensor(b0, dtype=torch.float64).requires_grad_()
    termes = utils_cpr.joint_losses(u, b0t, k_reg / echelle, k_inv / echelle, t, mode=cm.JOINT_LOSS_MODE,
                                    b0_mode=cm.B0_LOSS_MODE, lncc_sigma_vox=sigma, image_mode=cm.JOINT_IMAGE_MODE,
                                    image_term_updates_b0=cm.JOINT_IMAGE_UPDATES_B0)
    termes["smooth"] = utils_cpr.bending_penalty(b0t, espacement)
    termes["support"] = u.abs().square()[~support].mean()
    with h5py.File(chemin, "w") as f:
        for nom, v in dict(k_reg=k_reg.numpy(), k_inv=k_inv.numpy(), temps=np.asarray(temps, dtype=np.float64),
                           espacement_mm=np.asarray(espacement, dtype=np.float64), sigma_lncc_vox=np.asarray(sigma),
                           u=u.detach().numpy(), b0=b0t.detach().numpy(), echelle=echelle,
                           masque_support=support.numpy().astype(np.uint8)).items():
            f.create_dataset(nom, data=v)
        for nom in TERMES:
            gu, gb = torch.autograd.grad(termes[nom], [u, b0t], retain_graph=True, allow_unused=True)
            f.create_dataset(f"terme_{nom}/valeur", data=termes[nom].item())
            f.create_dataset(f"terme_{nom}/grad_u", data=(torch.zeros_like(u) if gu is None else gu).numpy())
            f.create_dataset(f"terme_{nom}/grad_b0", data=(torch.zeros_like(b0t) if gb is None else gb).numpy())
            print(f"{nom:<9} {termes[nom].item(): .6e}")
        # même calcul en float32 / complex64 (masque et échelle du float64, comme le fait Rust)
        u32 = u.detach().to(torch.complex64).requires_grad_()
        b032 = b0t.detach().float().requires_grad_()
        t32 = t.float()
        termes32 = utils_cpr.joint_losses(u32, b032, (k_reg / echelle).to(torch.complex64),
                                          (k_inv / echelle).to(torch.complex64), t32, mode=cm.JOINT_LOSS_MODE,
                                          b0_mode=cm.B0_LOSS_MODE, lncc_sigma_vox=sigma, image_mode=cm.JOINT_IMAGE_MODE,
                                          image_term_updates_b0=cm.JOINT_IMAGE_UPDATES_B0)
        termes32["smooth"] = utils_cpr.bending_penalty(b032, espacement)
        termes32["support"] = u32.abs().square()[~support].mean()

        def ecart(a, r):
            r = np.asarray(r)
            a = np.zeros_like(r) if a is None else a.detach().numpy().astype(r.dtype)
            n = np.linalg.norm(r)
            return float(np.linalg.norm(a - r) / n) if n > 0 else float(np.linalg.norm(a))

        for nom in TERMES:
            gu, gb = torch.autograd.grad(termes32[nom], [u32, b032], retain_graph=True, allow_unused=True)
            g = f"ecart_pytorch_f32/terme_{nom}"
            f.create_dataset(f"{g}/valeur", data=abs(termes32[nom].item() - f[f"terme_{nom}/valeur"][()])
                             / abs(f[f"terme_{nom}/valeur"][()]))
            f.create_dataset(f"{g}/grad_u", data=ecart(gu, f[f"terme_{nom}/grad_u"][()]))
            f.create_dataset(f"{g}/grad_b0", data=ecart(gb, f[f"terme_{nom}/grad_b0"][()]))
        print(f"support : {support.double().mean().item():.1%} du volume, échelle {echelle:.4e}")


def synthetique(sortie):
    rng = np.random.default_rng(20261122)
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
    k_reg = cprmod.time_segmented_cpr_adjoint(u_vrai, t, b0_vrai, 0.0, 0) + bruit()
    k_inv = cprmod.time_segmented_cpr_adjoint(u_vrai, t, -b0_vrai, 0.0, 0) + bruit()
    u = 0.9 * u_vrai + bruit()
    b0 = 0.8 * b0_vrai + 5 * rng.standard_normal(dims)
    ecrire(sortie, k_reg, k_inv, temps, espacement, u, b0)


def p1(entrees, sorties, sortie):
    checkpointer()
    i, o = np.load(entrees), np.load(sorties)
    t = i["time_array"]
    temps = t[:, 0, 0]
    assert np.allclose(t, temps[:, None, None]), "temps non constant le long des axes de phase"
    ecrire(sortie, i["k_reg"], i["k_inv"], temps, i["spacing_mm"].tolist(), o["u"].astype(np.complex128),
           o["b0_hz"].astype(np.float64))


if __name__ == "__main__":
    {"synthetique": synthetique, "p1": p1}[sys.argv[1]](*sys.argv[2:])
