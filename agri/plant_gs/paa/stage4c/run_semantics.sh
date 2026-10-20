#!/usr/bin/env bash
# Stage 4C semantic separation for one workspace: label -> merge -> panel ->
# carve -> provenance -> quality.
#
# The order is not arbitrary. The carve reads `semantic/labels_merged`, so the
# merge has to exist first; the provenance check reads the PLY the carve wrote,
# so it runs after; the mesh quality report needs the carve too. Each step is
# skipped when its output is already present, so a re-run after a failure
# continues instead of redoing 17 minutes of labelling.
#
#   bash agri/plant_gs/paa/stage4c/run_semantics.sh <WORKSPACE> [GPU]
#
# Environment
#   PY_GS   interpreter with gsplat / torch / the semantic models (required)
#   LABELS  label directory name to carve from (default labels_merged)
#   FORCE=1 redo every step even if its output exists
set -euo pipefail

WS="$(realpath "${1:?usage: run_semantics.sh WORKSPACE [GPU]}")"
GPU="${2:-${GPU:-0}}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PAA="$REPO/agri/plant_gs/paa"
PY_GS="${PY_GS:-python}"
LABELS="${LABELS:-labels_merged}"
FORCE="${FORCE:-0}"
LOG="$WS/logs"
mkdir -p "$LOG"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/agri/plant_gs:$PAA:$REPO/agri/plant_gs/foreground:${PYTHONPATH:-}"

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }
have() { [ "$FORCE" != "1" ] && [ -e "$1" ]; }

step "workspace $WS (gpu $GPU)"

# 1. per-view candidate labels from two independent backends.
if have "$WS/semantic/summary.json"; then
  echo "semantic labels already present; skipping"
else
  "$PY_GS" -u "$PAA/semantic.py" --dataset "$WS/dataset" \
    --out "$WS/semantic" --backend both 2>&1 | tee "$LOG/09_semantic.log"
fi

# 2. conservative merge: plant only where both backends agree.
if have "$WS/semantic/merge_summary.json"; then
  echo "merge already present; skipping"
else
  "$PY_GS" -u "$PAA/merge_labels.py" \
    --plant-dir "$WS/semantic/labels_segformer" \
    --other-dir "$WS/semantic/labels_sam_clip" \
    --out "$WS/semantic/labels_merged" 2>&1 | tee "$LOG/10_merge.log"
fi

# 3. the human-inspectable reference panel.
if have "$WS/semantic/plant_reference_panel/panel_summary.json"; then
  echo "reference panel already present; skipping"
else
  "$PY_GS" -u "$PAA/plant_reference_panel.py" --dataset "$WS/dataset" \
    --labels "$WS/semantic/$LABELS" --out "$WS/semantic/plant_reference_panel" \
    --views 12 2>&1 | tee "$LOG/11_panel.log"
fi

# 4. lift the labels into 3D and carve. Always from the exported PLY, never the
#    checkpoint: the PLY is the deliverable and the carve has to be a subset of
#    the file that gets used downstream.
if have "$WS/plant_only/plant_only_report.json" && [ "$FORCE" != "1" ]; then
  echo "plant-only carve already present; skipping"
else
  "$PY_GS" -u "$PAA/plant_only.py" --dataset "$WS/dataset" \
    --from-ply "$WS/final/point_cloud.ply" \
    --labels "$WS/semantic/$LABELS" --plant-source "$LABELS" \
    --out "$WS/plant_only" 2>&1 | tee "$LOG/12_plant_only.log"
fi

# 5. trace the carve back to its source rows.
"$PY_GS" -u -m agri.plant_gs.paa.stage4c.plant_only_provenance "$WS/plant_only" \
  2>&1 | tee "$LOG/13_plant_only_provenance.log"

# 6. how faithful the carve is to the labels: thin structure, white tissue,
#    residue. Runs on the gsplat interpreter because it renders.
"$PY_GS" -u -m agri.plant_gs.paa.stage4c.semantic_quality \
  --workspace "$WS" --labels "$WS/semantic/$LABELS" \
  --plant-dir "$WS/plant_only" --out "$WS/semantic_quality" \
  2>&1 | tee "$LOG/14_semantic_quality.log"

printf '\nSEMANTICS_COMPLETE %s\n' "$WS"
