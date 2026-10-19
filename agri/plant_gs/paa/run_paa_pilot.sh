#!/usr/bin/env bash
# End-to-end PAA single-plant run: prepare -> 3DGS -> export -> Gaussian-JEPA check.
#
# Every path is an argument or an environment variable; nothing is hard-coded to
# a machine. Run from the repository root:
#
#   bash agri/plant_gs/paa/run_paa_pilot.sh \
#       data/external/PAA-Plant-Dataset/Broccoli/20260327-10 \
#       data/plant_gs/PAA/Broccoli_20260327-10 7000
#
#   $1  PAA sample directory
#   $2  workspace directory (created; must differ from $1)
#   $3  optimisation steps (default 7000)
#
# Environment:
#   PY      python interpreter (default: python on PATH)
#   GPU     CUDA_VISIBLE_DEVICES for training (default 0)
#   MAXSZ   long-edge cap for prepared frames (default 1600)
#   SEED    random seed (default 0)
#   FORCE   set to 1 to rebuild an existing workspace
set -euo pipefail

SAMPLE="$(realpath "${1:?usage: run_paa_pilot.sh SAMPLE_DIR WORKSPACE [STEPS]}")"
SAMPLE_NAME="$(basename "$SAMPLE")"
CROP="$(basename "$(dirname "$SAMPLE")")"
WS="$(realpath -m "${2:?usage: run_paa_pilot.sh SAMPLE_DIR WORKSPACE [STEPS]}")"
STEPS="${3:-7000}"
PY="${PY:-python}"
GPU="${GPU:-0}"
MAXSZ="${MAXSZ:-1600}"
SEED="${SEED:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE="$REPO/agri/plant_gs"
PAA="$CODE/paa"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$CODE:$PAA:${PYTHONPATH:-}"

mkdir -p "$WS/logs"
step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }

step "environment"
"$PY" - <<'EOF'
import platform, sys, torch
print("python", platform.python_version(), sys.executable)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(),
      "devices", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("  gpu", i, torch.cuda.get_device_name(i))
try:
    import gsplat
    print("gsplat", gsplat.__version__)
except ImportError as exc:
    print("gsplat MISSING:", exc)
EOF

if [ "${FORCE:-0}" = "1" ] && [ -d "$WS" ]; then
  step "FORCE=1: removing existing workspace $WS"
  rm -rf "$WS"
  mkdir -p "$WS/logs"
fi

# --------------------------------------------------------------------------- #
# 1. COLMAP audit only (no writes) -- cheap, and it fails fast on bad samples
# --------------------------------------------------------------------------- #
step "1/7 COLMAP audit"
"$PY" "$PAA/colmap_check.py" --sample-dir "$SAMPLE" \
  --json "$WS/colmap_audit.json" 2>&1 | tee "$WS/logs/01_colmap_audit.log"

# --------------------------------------------------------------------------- #
# 2. prepare frames, masks and foreground points
# --------------------------------------------------------------------------- #
step "2/7 prepare dataset"
"$PY" "$PAA/prepare_paa.py" --sample "$SAMPLE" --workspace "$WS" \
  --max-size "$MAXSZ" --seed "$SEED" --force \
  2>&1 | tee "$WS/logs/02_prepare.log"

# --------------------------------------------------------------------------- #
# 3. 3DGS optimisation with the validated foreground trainer
# --------------------------------------------------------------------------- #
step "3/7 train 3DGS ($STEPS steps)"
if [ -f "$WS/results/train/splats.pt" ]; then
  echo "training already complete; skipping"
else
  "$PY" -u "$CODE/foreground/train.py" \
    --data "$WS/dataset" --result "$WS/results/train" \
    --steps "$STEPS" --eval-every 1000 --save-every 2000 --downscale 1 \
    --sh-degree 3 --foreground-alpha-weight 0.1 --background-alpha-weight 0.75 \
    --max-gaussians 300000 --seed "$SEED" 2>&1 | tee "$WS/logs/03_train.log"
fi

# --------------------------------------------------------------------------- #
# 4. clean the cloud down to the specimen and export a standard 3DGS PLY
# --------------------------------------------------------------------------- #
step "4/7 export"
"$PY" "$CODE/foreground/export.py" \
  --checkpoint "$WS/results/train/splats_best.pt" --output "$WS/final" \
  2>&1 | tee "$WS/logs/04_export.log"

# --------------------------------------------------------------------------- #
# 5. structural validation of the PLY + Gaussian-JEPA reader replay
# --------------------------------------------------------------------------- #
step "5/7 PLY validation"
"$PY" "$CODE/validate_ply.py" "$WS/final/point_cloud.ply" \
  --json "$WS/final/ply_validation.json" 2>&1 | tee "$WS/logs/05_ply.log"
"$PY" "$CODE/verify_gsjepa_compatible.py" "$WS/final/point_cloud.ply" \
  --json "$WS/final/gsjepa_compatibility.json" 2>&1 | tee "$WS/logs/05_compat.log"

# --------------------------------------------------------------------------- #
# 6. Gaussian-JEPA forward / backward, then optimiser + EMA plumbing.
#
# The forward/backward run must come first: if the real PLY cannot be read and
# grouped, there is nothing to optimise. The optimiser probe then checks the
# half of the contract a forward-only test cannot see -- that parameters move
# and the EMA teacher updates. Both are engineering checks; neither claims the
# model learned anything about plants.
# --------------------------------------------------------------------------- #
step "6/7 Gaussian-JEPA forward/backward + optimiser probe"
"$PY" -u "$CODE/jepa/smoke_real_plant.py" \
  --ply "$WS/final/point_cloud.ply" \
  --output "$WS/jepa_smoke.json" --num-gaussians 1024 --seed "$SEED" \
  2>&1 | tee "$WS/logs/06_jepa_smoke.log"
"$PY" -u "$CODE/jepa/optimizer_probe.py" \
  --ply "$WS/final/point_cloud.ply" \
  --output "$WS/jepa_optimizer_probe.json" --steps 5 --seed "$SEED" \
  2>&1 | tee "$WS/logs/06_jepa_probe.log"

# --------------------------------------------------------------------------- #
# 7. register the sample for batch accounting and split safety
# --------------------------------------------------------------------------- #
step "7/7 registry"
REGISTRY="${REGISTRY:-$WS/../registry.jsonl}"
"$PY" "$PAA/registry.py" --workspace "$WS" --sample "$SAMPLE_NAME" \
  --crop "$CROP" --registry "$REGISTRY" 2>&1 | tee "$WS/logs/07_registry.log"

printf '\nPILOT_COMPLETE %s\n' "$WS"