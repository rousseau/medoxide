"""Benchmark sur données réelles SANS vérité terrain, par prédiction hors échantillon (medoxide contre SVRTK).

Pour chacun des trois stacks (0 axial, 1 coronal, 2 sagittal) mis de côté tour à tour :
  1. medoxide reconstruit un volume avec les deux autres stacks (boucle recalage / reconstruction, 6 cycles) et une reconstruction sans correction (poses d'en-tête) ;
  2. SVRTK (réglages par défaut) reconstruit un volume avec les deux mêmes stacks ;
  3. le JUGE (test `judge_a_volume_with_a_held_out_stack`, même machinerie pour tous) recale les coupes du stack mis de côté contre chacun des trois volumes : plus la NCC finale est haute, mieux
     le volume prédit des coupes qu'il n'a jamais vues.
Aucun identifiant ni donnée de patiente n'est écrit dans le code : le sujet et la racine du jeu sont passés en arguments ; tous les fichiers restent dans data/svr/results/<sujet>/.

Usage : python scripts/benchmark_real.py --racine data/svr/jeu_reel_tru_haste --sujet sub-XXXX
"""

import argparse
import os
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--racine", required=True)
parser.add_argument("--sujet", required=True)
args = parser.parse_args()

racine = os.path.abspath(args.racine)
sortie = os.path.abspath(f"data/svr/results/{args.sujet}")
os.makedirs(sortie, exist_ok=True)
sessions = sorted(d for d in os.listdir(f"{racine}/{args.sujet}") if d.startswith("ses-"))
ses = sessions[0]
plans = ["trufiax", "truficor", "trufisag"]


def chemins(plan):
    base = f"{args.sujet}_{ses}_acq-{plan}_run-1"
    return (
        f"{racine}/{args.sujet}/{ses}/anat/{base}_T2w.nii.gz",
        f"{racine}/derivatives/medx-fetalbet/{args.sujet}/{ses}/anat/{base}_desc-brain_mask.nii.gz",
    )


def cargo(test, **env):
    e = {**os.environ, "MEDOXIDE_DONNEES_REELLES": racine, "MEDOXIDE_SUJET": args.sujet, **{k: str(v) for k, v in env.items()}}
    r = subprocess.run(["cargo", "test", "--release", "-p", "medoxide-svr", test, "--", "--ignored", "--nocapture"], capture_output=True, text=True, env=e)
    return r.stdout + r.stderr


lignes = ["stack_teste\tvolume\tcoupes\tncc_mediane\tncc_moyenne\tncc_p10\tcorrection_mm"]
for h in range(3):
    gardes = [i for i in range(3) if i != h]
    etiquette = f"_sans{h}"
    print(f"=== stack {h} mis de côté : reconstruction avec les stacks {gardes}", flush=True)
    sortie_ours = cargo("reconstruct_a_real_subject", MEDOXIDE_STACKS_RETENUS=",".join(map(str, gardes)), MEDOXIDE_ETIQUETTE=etiquette)
    print([l for l in sortie_ours.splitlines() if l.startswith("terminé") or "panicked" in l], flush=True)
    s0 = chemins(plans[gardes[0]])
    subprocess.run(
        ["python3", "scripts/run_svrtk.py", "--stacks", *[chemins(plans[i])[0] for i in gardes], "--mask", s0[1], "--sortie", sortie, "--nom", f"svrtk{etiquette}.nii.gz"],
        check=True, capture_output=True,
    )
    for nom, fichier in (("medoxide_boucle", f"ours_loop{etiquette}.nii.gz"), ("medoxide_sans_correction", f"ours_header{etiquette}.nii.gz"), ("SVRTK", f"svrtk{etiquette}.nii.gz")):
        sortie_juge = cargo("judge_a_volume_with_a_held_out_stack", MEDOXIDE_STACK_TEST=h, MEDOXIDE_VOLUME=f"{sortie}/{fichier}")
        m = re.search(r"JUGE stack \d+ : (\d+) coupes ; NCC finale médiane ([\d.]+), moyenne ([\d.]+), 10e centile ([\d.]+) ; correction moyenne ([\d.]+) mm", sortie_juge)
        if not m:
            print("échec du juge :", sortie_juge[-600:], flush=True)
            continue
        lignes.append(f"{h}\t{nom}\t" + "\t".join(m.groups()))
        print(f"  stack {h} testé, volume {nom:26}: {m.group(1)} coupes, NCC médiane {m.group(2)}, moyenne {m.group(3)}, p10 {m.group(4)}, correction {m.group(5)} mm", flush=True)
open(f"{sortie}/juge_hors_echantillon.tsv", "w").write("\n".join(lignes) + "\n")
print("terminé :", f"{sortie}/juge_hors_echantillon.tsv")
