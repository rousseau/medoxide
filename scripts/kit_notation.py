#!/usr/bin/env python3
"""Kit de notation en aveugle (docs/ROADMAP.md, F4.0.5, précisions 9 et 10).

Entrée : `couples.tsv`, une ligne par couple (sujet, méthode) : colonnes `sujet`, `methode`, `chemin` (volume reconstruit), `masque` (masque
cérébral du sujet, dans la grille de `reference`), `reference` (volume qui donne la grille commune du sujet : celle de `s2s`, 0,8 mm).
Sortie, dans `--sortie` (le « kit », ouvert par le noteur) :
  <id>.png       les trois coupes médianes (axiale, coronale, sagittale) côte à côte, nommé par un identifiant à quatre chiffres tiré au hasard ;
  grille.tsv     une ligne par identifiant, dans l'ordre mélangé de présentation, colonnes de notes vides ;
  LISEZMOI.txt   comment remplir la grille (sens des notes).
Et, **hors du kit**, `--correspondance` : identifiant -> sujet, méthode, position (jamais dans git : le chemin doit passer par un dossier `data/`).

Précautions d'aveugle : tous les volumes d'un sujet sont rééchantillonnés par la même interpolation (trilinéaire) sur la même grille, coupés aux
mêmes indices (milieu de la boîte englobante du masque) et fenêtrés par les mêmes centiles (1 et 99 dans le masque) ; les PNG n'ont aucune
métadonnée texte ni inscription ; l'ordre de présentation est une permutation de tous les couples (pas regroupé par sujet).

Aucun identifiant de sujet ni nom de méthode n'est écrit dans le code ; les noms viennent du TSV d'entrée.
"""

import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

GRAINE = 20261110
CENTILES = (1, 99)
AGRANDISSEMENT = 3
NOTES = ("qualite_globale", "usage_clinique", "flou", "bruit_texture", "dedoublement", "trous", "troncature", "commentaire")
COLONNES_COUPLES = ("sujet", "methode", "chemin", "masque", "reference")

LISEZMOI = """Notation en aveugle de volumes de cerveau fœtal reconstruits.

Chaque image <identifiant>.png montre trois coupes du même volume (axiale, coronale, sagittale). Les images sont présentées dans un ordre
mélangé ; remplissez grille.tsv (une ligne par identifiant) dans un éditeur de texte ou un tableur, en conservant les tabulations, puis enregistrez-le
au même endroit sous le même nom.

qualite_globale (entier de 1 à 4, plus grand = meilleur) :
  1 = inutilisable (structures illisibles ou absentes) ; 2 = médiocre (lisible, défauts gênants) ; 3 = bonne (défauts mineurs) ; 4 = excellente.
usage_clinique (probabilité qu'un clinicien s'en serve, entier de 1 à 4) :
  1 = très improbable ; 2 = peu probable ; 3 = probable ; 4 = très probable.
flou, bruit_texture, dedoublement, trous, troncature (artefacts) : 1 = présent, 0 = absent.
  flou = perte de netteté ; bruit_texture = grain ou texture parasite ; dedoublement = structures dédoublées ;
  trous = zones manquantes à l'intérieur ; troncature = cerveau coupé par le bord du champ.
commentaire : texte libre, facultatif (évitez d'y écrire une hypothèse sur l'origine de l'image).
"""


def permutation_et_identifiants(n, graine=GRAINE):
    """(position de chaque couple, identifiant de chaque couple), d'un même générateur : une permutation de 0..n-1, puis n identifiants à quatre
    chiffres tirés sans remise parmi 1000..9999."""
    rng = np.random.default_rng(graine)
    positions = rng.permutation(n)
    ids = rng.choice(np.arange(1000, 10000), size=n, replace=False)
    return [int(p) for p in positions], [int(i) for i in ids]


def reechantillonner(img, grille):
    """Volume de `img` (nibabel) sur la grille de `grille` (nibabel) : interpolation trilinéaire, 0 hors du champ de vue de `img`."""
    forme = grille.shape[:3]
    ijk = np.indices(forme).reshape(3, -1).astype(np.float64)
    monde = grille.affine[:3, :3] @ ijk + grille.affine[:3, 3:4]
    source = np.linalg.inv(img.affine)[:3, :3] @ monde + np.linalg.inv(img.affine)[:3, 3:4]
    data = np.asarray(img.dataobj, dtype=np.float64)
    return map_coordinates(data, source, order=1, mode="constant", cval=0.0).reshape(forme)


def indices_medians(masque):
    """Indices (i, j, k) du milieu de la boîte englobante du masque (arrondi à l'entier inférieur de la demi-somme des bornes)."""
    idx = np.argwhere(masque > 0)
    if len(idx) == 0:
        raise ValueError("masque vide")
    lo, hi = idx.min(axis=0), idx.max(axis=0)
    return tuple(int(v) for v in (lo + hi) // 2)


def fenetrer(coupe, bas, haut):
    """Coupe en niveaux de gris 0..255 entre les bornes `bas` et `haut` (valeurs hors bornes écrêtées)."""
    if haut <= bas:
        return np.zeros(coupe.shape, dtype=np.uint8)
    return np.round(255 * (np.clip(coupe, bas, haut) - bas) / (haut - bas)).astype(np.uint8)


def trois_coupes(vol, masque, indices):
    """Image 2D : axiale, coronale et sagittale (médianes) côte à côte, orientées de la même façon pour toutes les images d'un sujet."""
    bas, haut = np.percentile(vol[masque > 0], CENTILES)
    i, j, k = indices
    coupes = [vol[:, :, k], vol[:, j, :], vol[i, :, :]]
    hauteur = max(c.shape[1] for c in coupes)
    panneaux = []
    for c in coupes:
        g = fenetrer(c, bas, haut).T[::-1]  # axe de la grille en abscisse, orientation fixe
        p = np.zeros((hauteur, g.shape[1]), dtype=np.uint8)
        p[hauteur - g.shape[0] :, :] = g
        panneaux.append(p)
    return np.concatenate(panneaux, axis=1)


def ecrire_png(chemin, tableau, agrandissement=AGRANDISSEMENT):
    """PNG sans métadonnée texte (Pillow n'en ajoute pas sans `pnginfo`)."""
    im = Image.fromarray(tableau, mode="L")
    im = im.resize((im.width * agrandissement, im.height * agrandissement), Image.BICUBIC)
    im.save(chemin, format="PNG", optimize=True)


def sous_data(chemin):
    """Vrai si le chemin réel passe par un dossier nommé `data` (la table de correspondance n'est écrite que là, jamais dans git)."""
    return "data" in os.path.realpath(os.path.abspath(chemin)).split(os.sep)[:-1]


def lire_couples(chemin):
    with open(chemin, newline="") as f:
        lignes = list(csv.DictReader(f, delimiter="\t"))
    if not lignes or not set(COLONNES_COUPLES) <= set(lignes[0]):
        raise ValueError(f"{chemin} : colonnes attendues {COLONNES_COUPLES}")
    return lignes


def construire(couples, sortie, correspondance, graine=GRAINE):
    """Écrit le kit et la table de correspondance ; retourne la liste des (identifiant, position, sujet, méthode)."""
    if not sous_data(correspondance):
        raise ValueError(f"la table de correspondance doit être écrite sous un dossier data/ (hors git) : {correspondance}")
    if (os.path.realpath(os.path.dirname(os.path.abspath(correspondance))) + os.sep).startswith(os.path.realpath(sortie) + os.sep):
        raise ValueError("la table de correspondance ne doit pas être dans le kit")
    os.makedirs(sortie, exist_ok=True)
    n = len(couples)
    positions, ids = permutation_et_identifiants(n, graine)
    cache = {}  # sujet -> (grille, masque, indices)
    for c in couples:
        if c["sujet"] not in cache:
            grille = nib.load(c["reference"])
            m = reechantillonner(nib.load(c["masque"]), grille)
            cache[c["sujet"]] = (grille, m, indices_medians(m))
    table = []
    for k, c in enumerate(couples):
        grille, masque, idx = cache[c["sujet"]]
        vol = reechantillonner(nib.load(c["chemin"]), grille)
        ecrire_png(os.path.join(sortie, f"{ids[k]}.png"), trois_coupes(vol, masque, idx))
        table.append((ids[k], positions[k], c["sujet"], c["methode"]))
    ordre = sorted(table, key=lambda t: t[1])
    with open(os.path.join(sortie, "grille.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(("identifiant",) + NOTES)
        for t in ordre:
            w.writerow((t[0],) + ("",) * len(NOTES))
    with open(os.path.join(sortie, "LISEZMOI.txt"), "w") as f:
        f.write(LISEZMOI)
    os.makedirs(os.path.dirname(os.path.abspath(correspondance)), exist_ok=True)
    with open(correspondance, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(("identifiant", "position", "sujet", "methode"))
        w.writerows(ordre)
    return ordre


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("couples")
    p.add_argument("--sortie", required=True, help="dossier du kit")
    p.add_argument("--correspondance", required=True, help="TSV de correspondance, hors du kit, sous un dossier data/")
    p.add_argument("--graine", type=int, default=GRAINE)
    a = p.parse_args(argv)
    print(f"{len(construire(lire_couples(a.couples), a.sortie, a.correspondance, a.graine))} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
