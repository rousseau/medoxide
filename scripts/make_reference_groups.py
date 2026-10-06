"""Références du regroupement des stacks par sujet pour tester le crate medoxide-svr (étape 1e).

Pour chaque sujet : barycentre 3D de chaque masque nettoyé (relu dans data/reference/pivots/masques.tsv, produit par
make_reference_mask_pivots.py) converti en monde par l'affine du stack ; distances entre stacks ; groupes par
chaînage (deux stacks à moins de GAP mm sont dans le même groupe, transitivement). Les groupes sont classés du plus
grand au plus petit (égalité : plus petit indice d'abord), les indices croissent dans chaque groupe.
Écrit data/reference/groups/groups.tsv (sujet, indice, stack, rang du groupe) et distances.tsv (sujet, i, j, mm).

Usage : python scripts/make_reference_groups.py
"""

import collections
import itertools
import os

import nibabel as nib
import numpy as np

GAP = 18.0
SORTIE = "data/reference/groups"
os.makedirs(SORTIE, exist_ok=True)
par_sujet = collections.defaultdict(list)
for ligne in open("data/reference/pivots/masques.tsv").read().splitlines():
    c = ligne.split("\t")
    stack, bi, bj, bk = c[0], float(c[4]), float(c[5]), float(c[6])
    A = nib.load(stack).affine
    par_sujet[stack.split("/")[3]].append((stack, (A @ np.array([bi, bj, bk, 1.0]))[:3]))
groupes, distances = [], []
for sub, items in sorted(par_sujet.items()):
    items.sort(key=lambda x: x[0])
    n = len(items)
    d = np.array([[np.linalg.norm(a[1] - b[1]) for b in items] for a in items])
    for i, j in itertools.combinations(range(n), 2):
        distances.append(f"{sub}\t{i}\t{j}\t{float(d[i, j])!r}")
    vu, comps = [False] * n, []
    for s in range(n):
        if vu[s]:
            continue
        comp, file = [], [s]
        vu[s] = True
        while file:
            x = file.pop(0)
            comp.append(x)
            for y in range(n):
                if not vu[y] and d[x, y] <= GAP:
                    vu[y] = True
                    file.append(y)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0]))
    for rang, comp in enumerate(comps):
        for i in comp:
            groupes.append(f"{sub}\t{i}\t{items[i][0]}\t{rang}")
open(f"{SORTIE}/groups.tsv", "w").write("\n".join(groupes) + "\n")
open(f"{SORTIE}/distances.tsv", "w").write("\n".join(distances) + "\n")
sujets_multi = [sub for sub in par_sujet if len({x.split("\t")[3] for x in groupes if x.startswith(sub + "\t")}) > 1]
print(f"{len(par_sujet)} sujets, {len(groupes)} stacks, sujets à plusieurs groupes : {len(sujets_multi)}")
