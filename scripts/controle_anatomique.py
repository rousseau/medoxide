"""Contrôle anatomique d'une segmentation BOUNTI dans le repère de l'atlas (STA31, RAS : x vers la droite, y vers l'avant, z vers le haut).

La symétrie gauche-droite d'une segmentation ne prouve pas que le volume est bien orienté (BOUNTI étiquette selon le repère, pas selon l'anatomie) : on vérifie que le cervelet fait au moins 2 mL, que son centre est en arrière et en bas du centre du cerveau (y < 0 et z < 0) et que le tronc cérébral est en bas (z < 0).
"""

import os

import nibabel as nib
import numpy as np


def centre(etiquettes, k, affine):
    """Centre (mm) de l'étiquette `k`, NaN si elle est absente."""
    idx = np.array(np.nonzero(etiquettes == k)).T
    return nib.affines.apply_affine(affine, idx).mean(0) if len(idx) else np.full(3, np.nan)


def controle(seg):
    """(réussi, volume du cervelet en mL) pour le fichier de segmentation `seg` ; (False, 0) s'il n'existe pas."""
    if not os.path.exists(seg):
        return False, 0.0
    i = nib.load(seg)
    l = np.asarray(i.dataobj).astype(int)
    v = float(np.prod(i.header.get_zooms())) / 1000
    cbv = ((l == 11).sum() + (l == 12).sum()) * v
    if (l > 0).sum() == 0 or cbv < 2:
        return False, float(cbv)
    c0 = nib.affines.apply_affine(i.affine, np.array(np.nonzero(l > 0)).T).mean(0)
    cb = (centre(l, 11, i.affine) + centre(l, 12, i.affine)) / 2 - c0
    bs = centre(l, 10, i.affine) - c0
    return bool(cb[1] < 0 and cb[2] < 0 and bs[2] < 0), float(cbv)
