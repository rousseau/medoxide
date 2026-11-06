#!/usr/bin/env bash
# Rapatrie depuis Garage les fichiers que lit scripts/figure_qualitative_sujets.py, pour la régénérer sur n'importe quelle machine.
#
# Usage (depuis la racine de medoxide) :
#   scripts/recuperer_figure_svr.sh --sujets-fichier ~/Exp/oxide/svr/sujets.tsv [--repli 1]
#   scripts/recuperer_figure_svr.sh --deposer --sujets-fichier ... [--repli 1]     (depuis la machine qui a les résultats : les envoie)
#   scripts/recuperer_figure_svr.sh --jeu [--deposer] --sujets-fichier ...         (le jeu d'acquisitions : piles TRUFI + masques, pour régénérer des volumes et pour l'évaluation)
#
# Le TSV des sujets (`sujet<TAB>etiquette`) est dans le dépôt privé oxide : aucun identifiant ici.
# Les autres options (--repli, --variante...) sont passées telles quelles à la figure, qui connaît la liste des fichiers (`--lister`) : une seule source de vérité.
#
# Disposition sur Garage (bucket $OXIDE_REMOTE, lu dans ~/.config/garage/env) : $OXIDE_REMOTE/derivatives/medoxide-svr/results/<sujet>/<chemin>,
# le même arbre que data/svr/results/ en local (ignoré par Git). Rapatrier → figure → (les données restent dans data/, ignoré par Git ; les supprimer avec `rm -r data/svr/results`).
# Avec --jeu : data/svr/jeu_reel_tru_haste/ (arbre BIDS ; `--racine` de benchmark_variantes_s2s.py et evaluation_finale.py) ↔ $OXIDE_REMOTE/derivatives/medoxide-svr/jeu_reel_tru_haste/ ;
# seulement, par sujet du TSV : <sujet>/*/anat/*acq-trufi* (piles et sidecars .json, dont SliceTiming) et derivatives/medx-fetalbet/<sujet>/*/anat/*acq-trufi* (masques). Rien d'autre.
# Sort avec le code 1 et la liste des fichiers manquants si certains ne sont pas sur Garage. Données de patientes : ne rien publier de ce qui est rapatrié.

set -euo pipefail

SENS=pull
JEU=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --deposer) SENS=push ;;
    --jeu) JEU=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

ENV_GARAGE="$HOME/.config/garage/env"
[ -f "$ENV_GARAGE" ] && . "$ENV_GARAGE"
: "${OXIDE_REMOTE:?OXIDE_REMOTE absent (~/.config/garage/env)}"
[ -f scripts/figure_qualitative_sujets.py ] || { echo "À lancer depuis la racine de medoxide" >&2; exit 2; }

if [ "$JEU" = 1 ]; then
  DISTANT="$OXIDE_REMOTE/derivatives/medoxide-svr/jeu_reel_tru_haste"
  LOCAL="data/svr/jeu_reel_tru_haste"
  # sujets : première colonne du TSV passé à --sujets-fichier (lignes vides et commentaires ignorés)
  FICHIER=""
  for ((i = 0; i < ${#ARGS[@]}; i++)); do [ "${ARGS[$i]}" = "--sujets-fichier" ] && FICHIER="${ARGS[$((i + 1))]}"; done
  [ -f "$FICHIER" ] || { echo "--jeu demande --sujets-fichier <tsv>" >&2; exit 2; }
  FILTRES="$(mktemp)"
  trap 'rm -f "$FILTRES"' EXIT
  SUJETS=()
  while IFS=$'\t' read -r sujet _; do
    case "$sujet" in ""|\#*) continue ;; esac
    SUJETS+=("$sujet")
    printf '+ /%s/*/anat/*acq-trufi*\n+ /derivatives/medx-fetalbet/%s/*/anat/*acq-trufi*\n' "$sujet" "$sujet" >> "$FILTRES"
  done < "$FICHIER"
  echo "- *" >> "$FILTRES"
  if [ "$SENS" = push ]; then
    rclone copy --filter-from "$FILTRES" "$LOCAL" "$DISTANT" --progress
    ARRIVEE="$DISTANT"
  else
    mkdir -p "$LOCAL"
    rclone copy --filter-from "$FILTRES" "$DISTANT" "$LOCAL" --progress
    ARRIVEE="$LOCAL"
  fi
  # contrôle : au moins 3 piles par sujet, un masque par pile
  MAUVAIS=0
  for sujet in "${SUJETS[@]}"; do
    PILES=$(rclone lsf -R --files-only "$ARRIVEE/$sujet" --include "*acq-trufi*_T2w.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    MASQUES=$(rclone lsf -R --files-only "$ARRIVEE/derivatives/medx-fetalbet/$sujet" --include "*acq-trufi*_desc-brain_mask.nii.gz" 2>/dev/null | wc -l | tr -d ' ')
    echo "$sujet : $PILES piles, $MASQUES masques"
    if [ "$PILES" -lt 3 ] || [ "$PILES" != "$MASQUES" ]; then MAUVAIS=$((MAUVAIS + 1)); fi
  done
  [ "$MAUVAIS" = 0 ] && { echo "complet"; exit 0; }
  echo "$MAUVAIS sujet(s) incomplet(s) dans $ARRIVEE" >&2
  exit 1
fi

DISTANT="$OXIDE_REMOTE/derivatives/medoxide-svr/results"
LOCAL="data/svr/results"

LISTE="$(mktemp)"
trap 'rm -f "$LISTE"' EXIT
python3 -I scripts/figure_qualitative_sujets.py "${ARGS[@]}" --racine-resultats "$LOCAL" --lister > "$LISTE"
echo "$(wc -l < "$LISTE") fichiers attendus ($DISTANT)"

if [ "$SENS" = push ]; then
  rclone copy --files-from "$LISTE" "$LOCAL" "$DISTANT" --progress
else
  mkdir -p "$LOCAL"
  rclone copy --files-from "$LISTE" "$DISTANT" "$LOCAL" --progress
fi

# ce qui manque (localement après un pull, sur Garage après un dépôt)
MANQUANTS=0
while IFS= read -r f; do
  if [ "$SENS" = pull ]; then [ -s "$LOCAL/$f" ] || { echo "manque : $f"; MANQUANTS=$((MANQUANTS + 1)); }
  else rclone lsf "$DISTANT/$f" > /dev/null 2>&1 || { echo "manque : $f"; MANQUANTS=$((MANQUANTS + 1)); }; fi
done < "$LISTE"
if [ "$MANQUANTS" -gt 0 ]; then
  echo "$MANQUANTS fichier(s) manquant(s) sur $([ "$SENS" = pull ] && echo Garage || echo cette machine)" >&2
  exit 1
fi
echo "complet"
