#!/bin/bash
# Prépare puis publie les poids convertis de Multi-BOUNTI (GPL v3) sur Hugging Face : <utilisateur>/medoxide-multibounti.
# Le jeton d'accès (droit d'écriture) est lu dans la variable d'environnement HF_TOKEN : il n'est jamais écrit dans un fichier.
# Usage : HF_TOKEN=... bash scripts/multibounti_publier_poids.sh [utilisateur]      (défaut : rousseau ; à lancer depuis la racine du dépôt)
set -e
UTILISATEUR="${1:-rousseau}"
DEPOT="$UTILISATEUR/medoxide-multibounti"
PREPARATION=data/multibounti/hf
: "${HF_TOKEN:?definir HF_TOKEN avec un jeton Hugging Face a droit decriture}"
rm -rf "$PREPARATION"
mkdir -p "$PREPARATION"
cp -R data/multibounti/burn/bet data/multibounti/burn/lr data/multibounti/burn/multi data/multibounti/burn/gabarit "$PREPARATION"/
rm -f "$PREPARATION"/*/*.rs
cp docs/huggingface/multibounti-model-card.md "$PREPARATION/README.md"
cp data/bounti/LICENSE-auto-proc-svrtk "$PREPARATION/LICENSE"
(cd "$PREPARATION" && shasum -a 256 bet/bet_64.bpk lr/lr_64.bpk multi/multi_64.bpk gabarit/lr-lab.nii.gz)
python3 - "$DEPOT" "$PREPARATION" <<'PYTHON'
import os, sys
from huggingface_hub import HfApi
depot, dossier = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(depot, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=depot, repo_type="model", folder_path=dossier, commit_message="Poids de Multi-BOUNTI fœtal convertis pour Burn")
print("publié :", f"https://huggingface.co/{depot}")
PYTHON
