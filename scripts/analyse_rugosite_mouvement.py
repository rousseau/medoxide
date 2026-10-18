"""Rugosité du mouvement estimé sur acquisitions réelles : déplacement (mm) de 100 pixels de masque de la coupe k entre les poses de la coupe k et de la coupe k + 1 d'un même stack (médiane et 90e centile sur les coupes), par méthode.

Sert à situer le mouvement réel entre un mouvement lisse (pas de quelques dixièmes de mm) et un mouvement indépendant d'une coupe à l'autre (pas de plusieurs mm), c'est-à-dire à juger si l'a priori de régularité temporelle du coupe-coupe est susceptible de nuire. Aucun identifiant n'est écrit dans le code : sujets, étiquettes et méthodes sont passés en arguments ; chaque méthode est un fichier de poses `nom=fichier.tsv` (format de `medx svr recon --poses`), relatif au dossier du sujet dans `--resultats`.

Usage : python scripts/analyse_rugosite_mouvement.py --racine data/svr/jeu_reel_tru_haste --resultats data/svr/results --sujets sub-A sub-B --etiquettes "calme 1" "agité 1" --methodes s2s=poses_s2s_defaut.tsv SVRTK=poses_svrtk.tsv
"""
import argparse
import os
import re

import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--racine", required=True)
parser.add_argument("--resultats", required=True)
parser.add_argument("--sujets", nargs="+", required=True)
parser.add_argument("--etiquettes", nargs="+", required=True)
parser.add_argument("--methodes", nargs="+", required=True, help="nom=fichier.tsv")
args = parser.parse_args()
assert len(args.sujets) == len(args.etiquettes)
R, RES = args.racine, args.resultats
methodes = [tuple(m.split("=", 1)) for m in args.methodes]
sujets = list(zip(args.etiquettes, args.sujets))


def stacks(sujet):
    session=sorted(d for d in os.listdir(f'{R}/{sujet}') if d.startswith('ses-'))[0]
    dossier=f'{R}/{sujet}/{session}/anat'; masques=f'{R}/derivatives/medx-fetalbet/{sujet}/{session}/anat'
    trouves=[]
    for f in os.listdir(dossier):
        m=re.match(rf'{sujet}_{session}_acq-(trufiax|truficor|trufisag)_run-(\d+)_T2w\.nii\.gz$',f)
        if m: trouves.append((int(int(m.group(2))!=1),int(m.group(2))*10+['trufiax','truficor','trufisag'].index(m.group(1)),f'{sujet}_{session}_acq-{m.group(1)}_run-{m.group(2)}'))
    trouves.sort()
    return [(f'{dossier}/{n}_T2w.nii.gz',f'{masques}/{n}_desc-brain_mask.nii.gz') for _,_,n in trouves]
def lire(f): return {(int(l.split('\t')[0]),int(l.split('\t')[1])):np.array([float(v) for v in l.split('\t')[2:]]).reshape(4,4,order='F') for l in open(f).read().strip().split('\n')}
def poser(M,X): return (M[:3,:3]@X.T).T+M[:3,3]
print('rugosité = déplacement (mm) entre coupes consécutives d\'un stack ; médiane / p90')
print('sujet'.ljust(10)+''.join(m.ljust(16) for m,_ in methodes))
for et,s in sujets:
    pts={}
    rng=np.random.default_rng(0)
    for si,(fs,fm) in enumerate(stacks(s)):
        A=nib.load(fs).affine; m=np.asarray(nib.load(fm).dataobj)>0
        for k in range(m.shape[2]):
            ij=np.argwhere(m[:,:,k])
            if len(ij)<500: continue
            sel=ij[rng.choice(len(ij),100,replace=False)]
            pts[(si,k)]=(A@np.c_[sel,np.full(len(sel),k),np.ones(len(sel))].T).T[:,:3]
    ligne=et.ljust(10)
    for nom,f in methodes:
        fp=f'{RES}/{s}/{f}'
        if not os.path.exists(fp): ligne+='-'.ljust(16); continue
        P=lire(fp); d=[]
        for (si,k),X in pts.items():
            if (si,k+1) in pts and (si,k) in P and (si,k+1) in P:
                a=poser(P[(si,k)],X); b=poser(P[(si,k+1)],X)  # mêmes pixels de la coupe k, posés avec les deux poses
                d.append(np.sqrt(((a-b)**2).sum(1).mean()))
        ligne+=(f'{np.median(d):.2f} / {np.percentile(d,90):.2f}').ljust(16) if d else '-'.ljust(16)
    print(ligne)
