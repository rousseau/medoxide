#!/bin/bash
# Produit les poids convertis de BOUNTI (trois fichiers .bpk) pour `medx bounti`, sans dépendre de Docker à l'usage : Docker ne sert qu'ici, une fois, pour extraire les poids des auteurs.
#
# ATTENTION, LICENCE : les poids de BOUNTI sont ceux des auteurs (SVRTK, King's College London), sous la licence GPL v3 du logiciel (hypothèse de ce projet, à confirmer auprès des auteurs, voir docs/BOUNTI.md).
# Les fichiers produits sont des œuvres dérivées sous GPL v3 : ils ne sont PAS distribués avec medoxide (licence MIT) et ne doivent pas être versionnés (data/ est ignoré par Git).
#
# Étapes (chacune s'arrête à la première erreur) :
#   1. extraction des trois checkpoints PyTorch de l'image Docker `fetalsvrtk/svrtk:general_auto_arm` (ou $BOUNTI_IMAGE) vers data/bounti/models/ ;
#   2. export ONNX de chaque réseau et vérification contre onnxruntime (scripts/export_bounti_onnx.py, critère C1 de docs/BOUNTI.md) ;
#   3. conversion par `onnx2burn` (burn-onnx 0.22.0-pre.4) en poids `burnpack` et en code Rust ;
#   4. contrôle que le code Rust généré est celui qui est compilé dans crates/medoxide-bounti/src (aux imports, à la mise en forme et aux accolades près) : sans cela les poids n'iraient pas avec le modèle ;
#   5. copie des .bpk dans le dossier de destination.
#
# Prérequis : docker, python3 avec torch, monai, onnx et onnxruntime, et `cargo install burn-onnx --version 0.22.0-pre.4 --bin onnx2burn`.
# Usage (depuis la racine du dépôt) : bash scripts/bounti_poids.sh [dossier_de_destination]
# Destination par défaut : $MEDOXIDE_BOUNTI_WEIGHTS, sinon ~/.cache/medoxide/bounti (là où `medx bounti` les cherche).
set -euo pipefail

IMAGE=${BOUNTI_IMAGE:-fetalsvrtk/svrtk:general_auto_arm}
DEST=${1:-${MEDOXIDE_BOUNTI_WEIGHTS:-$HOME/.cache/medoxide/bounti}}
RACINE=data/bounti
# dossier du checkpoint de chaque réseau dans l'image (pas de tableau associatif : le bash de macOS est en version 3.2)
dossier_de() {
  case "$1" in
    unet) echo monai-checkpoints-unet-brain-bounti-19-lab ;;
    attunet) echo monai-checkpoints-red-attunet-brain_bounti-19-lab ;;
    bet) echo monai-checkpoints-attunet-brain-bet-1-lab ;;
  esac
}
RESEAUX="unet attunet bet"

[ -f Cargo.toml ] && [ -d crates/medoxide-bounti ] || { echo "à lancer depuis la racine du dépôt medoxide" >&2; exit 1; }
command -v onnx2burn >/dev/null || { echo "onnx2burn introuvable : cargo install burn-onnx --version 0.22.0-pre.4 --bin onnx2burn" >&2; exit 1; }
onnx2burn --version | grep -q "0.22.0-pre.4" || echo "AVERTISSEMENT : onnx2burn n'est pas en 0.22.0-pre.4 ; le contrôle du code généré (étape 4) dira si c'est grave" >&2

# 1. extraction des checkpoints (une seule fois)
mkdir -p "$RACINE/models"
manquants=0
for n in $RESEAUX; do [ -f "$RACINE/models/$(dossier_de "$n")/best_metric_model.pth" ] || manquants=1; done
if [ "$manquants" = 1 ]; then
  conteneur=$(docker create --platform linux/arm64 "$IMAGE")
  trap 'docker rm -f "$conteneur" >/dev/null 2>&1 || true' EXIT
  for n in $RESEAUX; do
    docker cp "$conteneur:/home/auto-proc-svrtk/trained_models/$(dossier_de "$n")" "$RACINE/models/"
  done
  docker rm -f "$conteneur" >/dev/null; trap - EXIT
fi

# normalisation du code Rust pour la comparaison : sans commentaires, imports, attributs, espaces, virgules ni accolades, et sans le chemin du fichier de poids que le code généré écrit dans `Default`
normaliser() { grep -v -E '^\s*(//|use |#!\[)' "$1" | tr -d ' \n\t,{}' | sed -E 's/from_file\("[^"]*"/from_file(""/'; }

travail=$(mktemp -d)
trap 'rm -rf "$travail"' EXIT
for n in $RESEAUX; do
  # 2. export ONNX (vérifié contre onnxruntime par le script)
  python3 scripts/export_bounti_onnx.py "$n" 64
  # 3. conversion
  onnx2burn --no-development "$RACINE/onnx/${n}_64.onnx" "$travail/$n"
  # 4. le code généré est celui qui est compilé
  [ "$(normaliser "$travail/$n/${n}_64.rs" | shasum)" = "$(normaliser "crates/medoxide-bounti/src/$n.rs" | shasum)" ] \
    || { echo "ERREUR : le code généré pour $n diffère de crates/medoxide-bounti/src/$n.rs : les poids ne vont pas avec le modèle compilé" >&2; exit 1; }
  # 5. copie
  mkdir -p "$DEST/$n"
  cp "$travail/$n/${n}_64.bpk" "$DEST/$n/${n}_64.bpk"
  echo "$n : $DEST/$n/${n}_64.bpk"
done
echo "Poids écrits dans $DEST (licence GPL v3, ne pas redistribuer avec medoxide). Utilisation : medx bounti --weights $DEST ..."
