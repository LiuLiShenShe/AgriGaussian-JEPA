#!/usr/bin/env bash
# Stage 4C end-to-end: cross-crop reconstruction -> semantics -> library audit.
#
# Two entry points in one script, because the two halves have very different
# run-times and failure modes:
#
#   bash agri/plant_gs/paa/stage4c/run_stage4c.sh reconstruct <CROP> <SAMPLE> [GPU]
#       one PAA sample through the Stage 4B-Fix runner (audit -> prepare ->
#       7k gate -> 30k -> export -> provenance -> JEPA)
#
#   bash agri/plant_gs/paa/stage4c/run_stage4c.sh semantics <WORKSPACE> [GPU]
#       label -> merge -> panel -> carve -> carve provenance -> quality
#
# Everything that turns those runs into a library is a separate, GPU-free step:
#
#   PY_GS=... python -m agri.plant_gs.paa.stage4c.assets
#   python -m agri.plant_gs.paa.stage4c.camera_summary
#   python -m agri.plant_gs.paa.stage4c.input_quality
#   python -m agri.plant_gs.paa.stage4c.annotation export
#
# Keeping the gather steps out of the shell script is deliberate: they must be
# re-runnable at any time, on any subset of the workspaces, without touching the
# training code.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PAA="$REPO/agri/plant_gs/paa"
PY_GS="${PY_GS:-python}"
PY_JEPA="${PY_JEPA:-$PY_GS}"

usage() {
  cat >&2 <<'EOF'
usage:
  run_stage4c.sh reconstruct <CROP> <SAMPLE> [GPU]
  run_stage4c.sh semantics   <WORKSPACE> [GPU]

shorthands for every stage-4C sample:
  run_stage4c.sh reconstruct-all [GPU0] [GPU1]
EOF
  exit 2
}

case "${1:-}" in
  reconstruct)
    [ $# -ge 3 ] || usage
    CROP="$2"; SAMPLE="$3"; GPU="${4:-0}"
    WS="$REPO/data/plant_gs/PAA/${CROP}_${SAMPLE}"
    echo "reconstruct $CROP/$SAMPLE -> $WS on GPU $GPU"
    cd "$REPO"
    GPU="$GPU" PY_GS="$PY_GS" PY_JEPA="$PY_JEPA" SEED="${SEED:-42}" \
      bash "$PAA/run_paa_pilot.sh" \
        "data/external/PAA-Plant-Dataset/$CROP/$SAMPLE" "$WS"
    ;;
  semantics)
    [ $# -ge 2 ] || usage
    WS="$2"; GPU="${3:-0}"
    cd "$REPO"
    PY_GS="$PY_GS" bash "$PAA/stage4c/run_semantics.sh" "$WS" "$GPU"
    ;;
  reconstruct-all)
    G0="${2:-0}"; G1="${3:-1}"
    # The four Stage 4C crops, as chosen from the Hub listing: every sample has
    # a complete 102-frame Camera/ + RGB_seg/masks/ pair. Two per GPU.
    pairs=(
      "Cabbage 20260328-9 $G0"
      "Tobacco 20260401-23 $G1"
      "Rice 1 $G0"
      "Sunflower 1 $G1"
    )
    for spec in "${pairs[@]}"; do
      set -- $spec
      bash "${BASH_SOURCE[0]}" reconstruct "$1" "$2" "$3" &
    done
    wait
    ;;
  *)
    usage
    ;;
esac
