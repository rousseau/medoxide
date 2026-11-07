#!/usr/bin/env bash
# Lance la porte « oracle » (docs/RECHERCHE_MULTISTART.md § 24) : un processus par âge d'atlas CRL2025, chacun les deux scénarios (lisse ±6 mm, sauts ±4 mm) et trois graines ; ~10 minutes par run sur la PGX.
# Prérequis : scripts/preparer_atlas_crl.py (atlas dans data/atlas/gholipour/CRL<âge>.nii.gz) et un binaire de test figé : MEDOXIDE_TEST_BIN=<binaire de medoxide-svr, cargo test --release --no-run>.
# Usage (racine de medoxide) : MEDOXIDE_TEST_BIN=... scripts/lancer_oracle_ponderation.sh [--dossier data/svr/oracle] 23 26 29 32 35 37 ; puis scripts/analyse_oracle_ponderation.py.
set -euo pipefail
DOSSIER=data/svr/oracle
[ "${1:-}" = "--dossier" ] && { DOSSIER="$2"; shift 2; }
: "${MEDOXIDE_TEST_BIN:?binaire de test absent}"
mkdir -p "$DOSSIER"
for age in "$@"; do
  (
    for sc in "lisse 6" "sauts 4"; do
      set -- $sc
      MEDOXIDE_ATLAS="CRL$age" MEDOXIDE_MODELE="$1" MEDOXIDE_AMPLITUDES="$2" MEDOXIDE_GRAINES=7,11,13 \
        MEDOXIDE_VARIANTES="b=1,12+t=0,01" MEDOXIDE_ORACLE_SORTIE="$DOSSIER/CRL$age" \
        "$MEDOXIDE_TEST_BIN" diagnose_the_semi_synthetic_benchmark --ignored --nocapture > "$DOSSIER/CRL$age-$1.log" 2>&1
    done
  ) &
done
wait
