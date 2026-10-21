"""Parcourt en flux les couches d'une image Docker du registre public (rien n'est stocké sur le disque, sauf ce qu'on extrait) et liste ou extrait les entrées dont le chemin correspond à une expression régulière.

Sert à lire l'image `fetalsvrtk/svrtk:perinatal_brain_mri_analysis_amd` de Multi-BOUNTI (13 Go compressés, environ 38 Go décompressés, AMD64 seulement) sans la tirer : voir `docs/MULTIBOUNTI.md`.
Les couches sont lues dans l'ordre donné ; une couche plus récente remplace un fichier d'une couche plus ancienne (donner les indices dans l'ordre croissant). Les liens symboliques ne sont pas extraits (`filter="data"` de `tarfile`).
Les fichiers extraits sont sous licence GPL v3 (voir `docs/MULTIBOUNTI.md`) : ne jamais les mettre dans le dépôt.

Usage : python scripts/multibounti_couches.py <dépôt> <étiquette> <liste|extraire> <expression régulière> <indices séparés par des virgules | petites> [dossier_de_sortie]
Exemple : python scripts/multibounti_couches.py fetalsvrtk/svrtk perinatal_brain_mri_analysis_amd liste 'home/perinatal-brain-mri-analysis/' 46,49,50
"""
import json, re, subprocess, sys, tarfile, urllib.request, os
repo, tag, action, expr, indices = sys.argv[1:6]
sortie = sys.argv[6] if len(sys.argv) > 6 else None
motif = re.compile(expr)
def jeton():
    return json.load(urllib.request.urlopen(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))["token"]
tok = jeton()
req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}", headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.docker.distribution.manifest.v2+json"})
m = json.load(urllib.request.urlopen(req))
couches = m["layers"]
if indices == "petites":
    sel = [i for i, c in enumerate(couches) if 1e5 < c["size"] < 4e8]
else:
    sel = [int(x) for x in indices.split(",")]
for i in sel:
    c = couches[i]
    proc = subprocess.Popen(["curl", "-sL", "-H", f"Authorization: Bearer {jeton()}", f"https://registry-1.docker.io/v2/{repo}/blobs/{c['digest']}"], stdout=subprocess.PIPE)
    n = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
            for membre in tar:
                if motif.search(membre.name):
                    n += 1
                    if action == "liste":
                        print(f"couche {i:2d} {membre.size:>12d} {membre.name}")
                    elif action == "extraire" and membre.isfile():
                        tar.extract(membre, path=sortie, filter="data")
                        print(f"extrait (couche {i}) {membre.name} {membre.size}")
    except Exception as e:
        print(f"couche {i} : erreur de lecture {type(e).__name__}: {e}", file=sys.stderr)
    proc.stdout.close(); proc.wait()
    print(f"-- couche {i} ({c['size']/1e6:.0f} Mo compressés) : {n} entrée(s)", file=sys.stderr)
