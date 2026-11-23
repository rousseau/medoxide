"""Cas de test de l'INR à grille de hachage de medoxide-ulf (docs/ULF.md, P4 bis, critère 1), en HDF5 lisible par Rust.

    cd ~/Exp/fridge-magnet-imaging/notebooks/francois   # inr.py et utils_cpr.py de fmi (référence cb7174b)
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_inr.py synthetique <medoxide>/crates/medoxide-ulf/tests/donnees/inr_synthetique.h5
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_inr.py p1 <sortie.h5>   # réglages de P1 (finest32) sur 115 × 115 × 45
    PYTHONPATH=. uv run python <medoxide>/scripts/cas_inr.py p1_b <sortie.h5>   # champ de la méthode B (finest 16), seul

Pour chaque réseau (`champ` : INRField, `image` : INRImage), en float64 :
- `config` (attributs) : niveaux, caractéristiques par niveau, log2 de la table, résolution de base, facteur entre
  niveaux, neurones, couches cachées, sorties ;
- `niveau_<l>/table` (poids de la table), `niveau_<l>/indices` (indices de table lus à l'entrée de `nn.Embedding`,
  forme (points, coins)) et `niveau_<l>/encodage` (sortie du niveau, forme (points, caractéristiques)) ;
- `couche_<c>/poids` (forme PyTorch (sorties, entrées)) et `couche_<c>/biais` ;
- `sortie` : sortie brute du réseau (points, sorties) ; pour le champ, `b0` (Hz) avec les scalaires `s` et `o` ;
  pour l'image, `re` et `im` ;
- `r` : tableau aléatoire de la forme de la sortie finale (b0, ou re et im empilés) et `grad/...` : gradient de
  L = Σ r · sortie par rapport à chaque paramètre (mêmes noms).

`ecart_pytorch_f32/<nom>` : écart relatif L2 au float64 du même calcul fait par PyTorch en float32 (réseaux en float32,
coordonnées en float32, comme l'étude f32 de b0_bases_lisses.py), pour chaque sortie et chaque gradient : c'est le
plancher de conditionnement auquel se compare le calcul `f32` de Rust (amendement du 2026-11-22, docs/ULF.md).

Les tables sont tirées uniformément dans [−1, 1] (l'initialisation de PyTorch, ± 1e-4, rendrait la sortie presque
constante et le test peu sensible) ; les scalaires du champ valent s = 0,3, o = −0,2. Le cas synthétique (grille
9 × 6 × 5, petites tables pour avoir des niveaux denses et hachés) est versionné ; le cas p1 (réglages de P1, environ
300 Mo) ne l'est pas, mais ne contient aucune donnée de sujet.
"""

import copy
import sys

import h5py
import numpy as np
import torch

from inr import grid_to_image, make_coordinate_grid
from utils_cpr import INRField, INRImage, _eval_net

SYNTHETIQUE = {
    "dims": (9, 6, 5),
    "champ": dict(n_levels=4, n_features_per_level=2, log2_hashmap_size=6, base_resolution=2, per_level_scale=2.0,
                  n_neurons=8, n_hidden_layers=2),
    "image": dict(n_levels=3, n_features_per_level=2, log2_hashmap_size=8, base_resolution=4, per_level_scale=2.0,
                  n_neurons=8, n_hidden_layers=2),
}
P1 = {  # compare_methods.B0_INR_CONFIG avec finest 32 (b0_sensitivity.overrides) et U_INR_CONFIG
    "dims": (115, 115, 45),
    "champ": dict(n_levels=8, n_features_per_level=2, log2_hashmap_size=19, base_resolution=2,
                  per_level_scale=(32 / 2) ** (1 / 7), n_neurons=64, n_hidden_layers=2),
    "image": dict(n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, per_level_scale=2.0,
                  n_neurons=64, n_hidden_layers=2),
}
P1_B = {  # méthode B : compare_methods.B0_INR_CONFIG par défaut (finest 16), champ seul (docs/ULF.md, P5)
    "dims": (115, 115, 45),
    "champ": dict(P1["champ"], per_level_scale=(16 / 2) ** (1 / 7)),
}
B0_MIN, B0_MAX = 12.5, 108.0  # bornes de l'amplitude du champ (Hz), de l'ordre de celles de P1


def ecrire_reseau(groupe, net, config, n_sorties, capture):
    for k, v in config.items():
        groupe.attrs[k] = v
    groupe.attrs["n_output_dims"] = n_sorties
    for l, niveau in enumerate(net.encoding.levels):
        g = groupe.create_group(f"niveau_{l}")
        g.create_dataset("table", data=niveau.embedding.weight.detach().numpy())
        g.create_dataset("indices", data=capture["indices"][l])
        g.create_dataset("encodage", data=capture["encodage"][l])
        g.attrs["echelle"], g.attrs["resolution"] = niveau.scale, niveau.resolution
        g.attrs["taille_table"], g.attrs["hachee"] = niveau.hashmap_size, niveau.use_hash
    lineaires = [m for m in net.mlp.net if isinstance(m, torch.nn.Linear)]
    for c, m in enumerate(lineaires):
        groupe.create_dataset(f"couche_{c}/poids", data=m.weight.detach().numpy())
        groupe.create_dataset(f"couche_{c}/biais", data=m.bias.detach().numpy())
    return lineaires


def ecrire_gradients(groupe, net, lineaires):
    for l, niveau in enumerate(net.encoding.levels):
        groupe.create_dataset(f"grad/niveau_{l}/table", data=niveau.embedding.weight.grad.numpy())
    for c, m in enumerate(lineaires):
        groupe.create_dataset(f"grad/couche_{c}/poids", data=m.weight.grad.numpy())
        groupe.create_dataset(f"grad/couche_{c}/biais", data=m.bias.grad.numpy())


def preparer(net, gen):
    """Tables uniformes dans [−1, 1] et crochets qui capturent les indices et l'encodage de chaque niveau."""
    capture = {"indices": {}, "encodage": {}}
    for l, niveau in enumerate(net.encoding.levels):
        with torch.no_grad():
            niveau.embedding.weight.uniform_(-1.0, 1.0, generator=gen)
        niveau.embedding.register_forward_hook(
            lambda m, entree, sortie, l=l: capture["indices"].__setitem__(l, entree[0].reshape(-1, 8).numpy().copy()))
        niveau.register_forward_hook(
            lambda m, entree, sortie, l=l: capture["encodage"].__setitem__(l, sortie.detach().numpy().copy()))
    return capture


def ecart(a, r):
    a, r = a.detach().double().reshape(-1), r.detach().double().reshape(-1)
    return float((a - r).norm() / r.norm())


def gradients(net):
    g = {f"grad/niveau_{l}/table": n.embedding.weight.grad for l, n in enumerate(net.encoding.levels)}
    for c, m in enumerate(m for m in net.mlp.net if isinstance(m, torch.nn.Linear)):
        g[f"grad/couche_{c}/poids"], g[f"grad/couche_{c}/biais"] = m.weight.grad, m.bias.grad
    return g


def ecarts_f32(groupe, net64, dims, sorties64, sorties_fn, r):
    """Même calcul en float32 (copie du réseau, coordonnées float32) ; écarts au float64 dans ecart_pytorch_f32/."""
    net = copy.deepcopy(net64).float()
    net.zero_grad()
    brut = grid_to_image(net(make_coordinate_grid(dims, dtype=torch.float32)), dims)
    sorties, perte, extra = sorties_fn(brut, r.float())
    perte.backward()
    for nom, v in sorties.items():
        groupe.create_dataset(f"ecart_pytorch_f32/{nom}", data=ecart(v, sorties64[nom]))
    grads64 = gradients(net64)
    for nom, g in {**gradients(net), **extra()}.items():
        ref = grads64[nom] if nom in grads64 else torch.tensor(groupe[nom][()], dtype=torch.float64)
        groupe.create_dataset(f"ecart_pytorch_f32/{nom}", data=ecart(g, ref))


def ecrire(chemin, cas):
    dims = cas["dims"]
    gen = torch.Generator().manual_seed(20261122)
    with h5py.File(chemin, "w") as f:
        f.attrs["dims"] = dims
        f.create_dataset("coordonnees", data=make_coordinate_grid(dims, dtype=torch.float64).numpy())

        champ = INRField(**cas["champ"], b0_min_hz=B0_MIN, b0_max_hz=B0_MAX, seed=0)
        capture = preparer(champ.net, gen)
        with torch.no_grad():
            champ.scale_param.fill_(0.3)
            champ.offset_param.fill_(-0.2)
        g = f.create_group("champ")
        g.attrs["b0_min_hz"], g.attrs["b0_max_hz"] = B0_MIN, B0_MAX
        b0 = champ.b0(dims)
        sortie = _eval_net(champ.net, dims)
        lineaires = ecrire_reseau(g, champ.net, cas["champ"], 1, capture)
        g.create_dataset("sortie", data=sortie.detach().reshape(1, -1).T.numpy())
        g.create_dataset("b0", data=b0.detach().numpy())
        g.create_dataset("s", data=champ.scale_param.item())
        g.create_dataset("o", data=champ.offset_param.item())
        r = torch.randn(dims, generator=gen, dtype=torch.float64)
        g.create_dataset("r", data=r.numpy())
        (r * b0).sum().backward()
        ecrire_gradients(g, champ.net, lineaires)
        g.create_dataset("grad/s", data=champ.scale_param.grad.item())
        g.create_dataset("grad/o", data=champ.offset_param.grad.item())
        s32 = torch.tensor(0.3, dtype=torch.float32, requires_grad=True)
        o32 = torch.tensor(-0.2, dtype=torch.float32, requires_grad=True)

        def champ_f32(brut, r32):
            forme = torch.tanh(brut[0])
            forme = forme - forme.mean()
            forme = forme / torch.sqrt(forme.square().mean().detach() + 1e-6)
            amp = B0_MIN * torch.exp(np.log(B0_MAX / B0_MIN) * torch.sigmoid(s32))
            b0_32 = B0_MAX * torch.tanh(o32) + amp * forme
            return ({"b0": b0_32, "sortie": brut.reshape(1, -1).T}, (r32 * b0_32).sum(),
                    lambda: {"grad/s": s32.grad, "grad/o": o32.grad})

        ecarts_f32(g, champ.net, dims, {"b0": b0, "sortie": sortie.reshape(1, -1).T}, champ_f32, r)

        if "image" not in cas:
            return
        image = INRImage(**cas["image"], seed=0)
        capture = preparer(image.net, gen)
        g = f.create_group("image")
        u = image.image(dims)
        sortie = _eval_net(image.net, dims)
        lineaires = ecrire_reseau(g, image.net, cas["image"], 2, capture)
        g.create_dataset("sortie", data=sortie.detach().reshape(2, -1).T.numpy())
        g.create_dataset("re", data=u.real.detach().numpy())
        g.create_dataset("im", data=u.imag.detach().numpy())
        r = torch.randn((2, *dims), generator=gen, dtype=torch.float64)
        g.create_dataset("r", data=r.numpy())
        (r[0] * u.real + r[1] * u.imag).sum().backward()
        ecrire_gradients(g, image.net, lineaires)

        def image_f32(brut, r32):
            return ({"re": brut[0], "im": brut[1], "sortie": brut.reshape(2, -1).T},
                    (r32[0] * brut[0] + r32[1] * brut[1]).sum(), lambda: {})

        ecarts_f32(g, image.net, dims, {"re": u.real, "im": u.imag, "sortie": sortie.reshape(2, -1).T}, image_f32, r)


if __name__ == "__main__":
    mode, sortie = sys.argv[1], sys.argv[2]
    ecrire(sortie, {"synthetique": SYNTHETIQUE, "p1": P1, "p1_b": P1_B}[mode])
    print(f"écrit : {sortie}")
