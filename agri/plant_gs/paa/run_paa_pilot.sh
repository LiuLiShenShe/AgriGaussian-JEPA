#!/usr/bin/env bash
# End-to-end PAA single-plant run: prepare -> 7k gate -> 30k formal -> export -> JEPA check.
#
# Two stages, two independent training runs
# ----------------------------------------
# Stage 4B's records describe a 7k "trial" and a 30k "formal" run as if the
# second continued the first. They did not: they were two separate trainings
# with different step counts and different save cadences. Treating them as one
# run is what allowed a 7k model to be described with 30k provenance (and, in
# the Cotton/1 case, allowed two live processes to share one directory).
#
# This runner keeps them genuinely separate:
#
#   results/train_7k/    quality gate only. Own config, log, status, checkpoints.
#   results/train_30k/   the formal run. A NEW training, not a continuation.
#
# The 30k stage never resumes the 7k model. It is a fresh optimisation with its
# own seed-to-checkpoint chain, and its provenance says so.
#
# Usage (from the repository root):
#
#   bash agri/plant_gs/paa/run_paa_pilot.sh \
#       data/external/PAA-Plant-Dataset/Cotton/1 \
#       data/plant_gs/PAA/Cotton_1_repro_v1
#
#   $1  PAA sample directory
#   $2  workspace directory (created; must differ from $1)
#   $3  optional: 30k step count (default $STEPS_30K, i.e. 30000)
#
# Environment
#   PY_GS       interpreter with gsplat            (default: python on PATH)
#   PY_JEPA     interpreter with Gaussian-JEPA CUDA extensions
#   GPU         CUDA_VISIBLE_DEVICES              (default 0)
#   MAXSZ       long-edge cap for prepared frames (default 1600)
#   SEED        random seed                       (default 0)
#   STEPS_7K    gate-stage steps                  (default 7000)
#   STEPS_30K   formal-stage steps                (default 30000)
#   RESUME=1    continue: skip completed stages, resume an interrupted one
#               from its own resumable checkpoint. Never re-runs a stage.
#   SKIP_GATE=1 run the formal stage even if the 7k gate fails (records why)
#   REGISTRY    JSONL registry path (default: <workspace>/../registry.jsonl)
#   GATE_MIN_PSNR / GATE_MIN_IOU / GATE_MAX_BG_ALPHA   gate thresholds
#
# Reproduction
#   From scratch      : run the command above into a NEW workspace.
#   After a crash     : same command with RESUME=1. Completed stages are
#                       skipped; an interrupted stage resumes only if its
#                       ``splats_resume.pt`` carries optimiser + strategy + RNG
#                       state, and the script fails with a diagnosis otherwise.
#   Re-verify only    : SKIP_GATE=1 RESUME=1 with results/train_7k and
#                       results/train_30k already complete -- the training
#                       stages are skipped and export/validation still run.
set -euo pipefail

SAMPLE="$(realpath "${1:?usage: run_paa_pilot.sh SAMPLE_DIR WORKSPACE [STEPS_30K]}")"
SAMPLE_NAME="$(basename "$SAMPLE")"
CROP="$(basename "$(dirname "$SAMPLE")")"
WS="$(realpath -m "${2:?usage: run_paa_pilot.sh SAMPLE_DIR WORKSPACE [STEPS_30K]}")"
PY_GS="${PY_GS:-python}"
PY_JEPA="${PY_JEPA:-$PY_GS}"
GPU="${GPU:-0}"
MAXSZ="${MAXSZ:-1600}"
SEED="${SEED:-0}"
STEPS_7K="${STEPS_7K:-7000}"
STEPS_30K="${3:-${STEPS_30K:-30000}}"
RESUME="${RESUME:-0}"
SKIP_GATE="${SKIP_GATE:-0}"
GATE_MIN_PSNR="${GATE_MIN_PSNR:-18}"
GATE_MIN_IOU="${GATE_MIN_IOU:-0.85}"
GATE_MAX_BG_ALPHA="${GATE_MAX_BG_ALPHA:-0.05}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE="$REPO/agri/plant_gs"
PAA="$CODE/paa"
FG="$CODE/foreground"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$CODE:$PAA:$FG:${PYTHONPATH:-}"

mkdir -p "$WS/logs"
# stage_state / gate_check / stage_manifest live in stages.sh so the tests can
# exercise them without running a training pipeline.
# shellcheck source=agri/plant_gs/paa/stages.sh
source "$PAA/stages.sh"

# --------------------------------------------------------------------------- #
# run_stage NAME STEPS EVAL_EVERY SAVE_EVERY
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 0. Environment
# --------------------------------------------------------------------------- #
step "environment"
for PY in "$PY_GS" "$PY_JEPA"; do
  echo "--- interpreter: $PY"
  "$PY" - <<'EOF'
import platform, sys, torch
print("python", platform.python_version(), sys.executable)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(),
      "devices", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("  gpu", i, torch.cuda.get_device_name(i))
for mod in ("gsplat", "pointnet2_ops", "knn_cuda"):
    try:
        m = __import__(mod)
        print(f"{mod}", getattr(m, "__version__", "imported"))
    except Exception as exc:
        print(f"{mod} MISSING:", type(exc).__name__)
EOF
done

# --------------------------------------------------------------------------- #
# 1. COLMAP audit only (no writes) -- cheap, and it fails fast on bad samples
# --------------------------------------------------------------------------- #
step "1/9 COLMAP audit"
"$PY_GS" "$PAA/colmap_check.py" --sample-dir "$SAMPLE" \
  --json "$WS/colmap_audit.json" 2>&1 | tee "$WS/logs/01_colmap_audit.log"

# --------------------------------------------------------------------------- #
# 2. prepare frames, masks and foreground points
#
# The mask crop offsets are measured on THIS sample. A sample whose masks do
# not register fails here instead of inheriting another plant's offsets.
# --------------------------------------------------------------------------- #
step "2/9 prepare dataset"
PREPARE_FLAGS=(--max-size "$MAXSZ" --seed "$SEED")
[ "$RESUME" = "1" ] && PREPARE_FLAGS+=(--resume)
"$PY_GS" "$PAA/prepare_paa.py" --sample "$SAMPLE" --workspace "$WS" \
  "${PREPARE_FLAGS[@]}" 2>&1 | tee "$WS/logs/02_prepare.log"

# --------------------------------------------------------------------------- #
# 3. stage 1 -- 7k quality gate
# --------------------------------------------------------------------------- #
step "3/9 stage 1: 7k gate training"
run_stage train_7k "$STEPS_7K" 1000 2000

# --------------------------------------------------------------------------- #
# 4. gate decision
# --------------------------------------------------------------------------- #
step "4/9 quality gate on the 7k run"
if ! gate_check "$WS/results/train_7k/validation.json" "$GATE_MIN_PSNR" "$GATE_MIN_IOU" "$GATE_MAX_BG_ALPHA"; then
  if [ "$SKIP_GATE" = "1" ]; then
    echo "GATE OVERRIDDEN by SKIP_GATE=1; recording the override in final/report.json"
  else
    die "the 7k quality gate failed. Fix the data/reconstruction, or set
SKIP_GATE=1 to run the formal stage anyway (the override is recorded)."
  fi
fi

# --------------------------------------------------------------------------- #
# 5. stage 2 -- 30k formal training, a NEW run
# --------------------------------------------------------------------------- #
step "5/9 stage 2: 30k formal training (independent run, not a continuation)"
run_stage train_30k "$STEPS_30K" 3000 5000

BEST="$WS/results/train_30k/splats_best.pt"
[ -f "$BEST" ] || die "no $BEST. The formal run produced no selected checkpoint;
refusing to export a PLY with no source."
[ -f "$WS/results/train_30k/splats.pt" ] || die "no $WS/results/train_30k/splats.pt;
the formal run is incomplete."

# --------------------------------------------------------------------------- #
# 6. export, from the explicitly named 30k best checkpoint
#
# The exporter itself refuses if the run is unfinished or the checkpoint changes
# during export; this script only has to name the right file.
# --------------------------------------------------------------------------- #
step "6/9 export from the 30k best checkpoint"
"$PY_GS" -u "$FG/export.py" \
  --checkpoint "$BEST" --output "$WS/final" \
  2>&1 | tee "$WS/logs/04_export.log"

# --------------------------------------------------------------------------- #
# 7. provenance + PLY validation
# --------------------------------------------------------------------------- #
step "7/9 provenance verification + PLY validation"
"$PY_GS" "$FG/verify_provenance.py" "$WS/final" \
  --json "$WS/final/provenance_verification.json" 2>&1 | tee "$WS/logs/05_provenance.log"
"$PY_GS" "$CODE/validate_ply.py" "$WS/final/point_cloud.ply" \
  --json "$WS/final/ply_validation.json" 2>&1 | tee "$WS/logs/05_ply.log"

# --------------------------------------------------------------------------- #
# 8. Gaussian-JEPA reader replay, forward/backward, optimiser + EMA probe.
#
#    These run in the gaussian_jepa env, which is where the compiled
#    pointnet2_ops / knn_cuda extensions live. The structural checks above stay
#    on the gsplat interpreter, which is where plyfile and cv2 work.
# --------------------------------------------------------------------------- #
step "8/9 Gaussian-JEPA compatibility + forward/backward + optimiser probe"
"$PY_JEPA" "$CODE/verify_gsjepa_compatible.py" "$WS/final/point_cloud.ply" \
  --json "$WS/final/gsjepa_compatibility.json" 2>&1 | tee "$WS/logs/06_compat.log"
"$PY_JEPA" -u "$CODE/jepa/smoke_real_plant.py" \
  --ply "$WS/final/point_cloud.ply" \
  --output "$WS/jepa_smoke.json" --num-gaussians 1024 --seed "$SEED" \
  2>&1 | tee "$WS/logs/06_jepa_smoke.log"
"$PY_JEPA" -u "$CODE/jepa/optimizer_probe.py" \
  --ply "$WS/final/point_cloud.ply" \
  --output "$WS/jepa_optimizer_probe.json" --steps 5 --seed "$SEED" \
  2>&1 | tee "$WS/logs/06_jepa_probe.log"

# --------------------------------------------------------------------------- #
# 9. Gaussian statistics, then register the sample.
#
#    Registration is an upsert keyed on crop/acquisition. The record carries a
#    `provenance` block; an accepted row requires that chain to have verified.
# --------------------------------------------------------------------------- #
step "9/9 gaussian stats + registry"
"$PY_GS" "$PAA/gaussian_stats.py" "$WS/final/point_cloud.ply" \
  --json "$WS/final/gaussian_stats.json" 2>&1 | tee "$WS/logs/07_stats.log"
REGISTRY="${REGISTRY:-$WS/../registry.jsonl}"
"$PY_GS" "$PAA/registry.py" --workspace "$WS" --sample "$SAMPLE_NAME" \
  --crop "$CROP" --registry "$REGISTRY" 2>&1 | tee "$WS/logs/08_registry.log"

printf '\nPILOT_COMPLETE %s\n' "$WS"