#!/usr/bin/env bash
# End-to-end PAA single-plant run: prepare -> 3DGS -> export -> Gaussian-JEPA check.
#
# Stage 4B note on environments: this script uses TWO interpreters on purpose.
# The 3DGS half needs gsplat (installed in the base env, `$PY_GS`), the
# Gaussian-JEPA half needs the compiled CUDA extensions of the gaussian_jepa env
# (`$PY_JEPA`). Stage 4A ran both with one interpreter, which worked only
# because that env happened to have gsplat; it does not, and running the JEPA
# steps without pointnet2_ops/knn_cuda would fail deep inside the model rather
# than at import time.
#
# Every path is an argument or an environment variable; nothing is hard-coded to
# a machine. Run from the repository root:
#
#   bash agri/plant_gs/paa/run_paa_pilot.sh \
#       data/external/PAA-Plant-Dataset/Broccoli/20260327-10 \
#       data/plant_gs/PAA/Broccoli_20260327-10 30000
#
#   $1  PAA sample directory
#   $2  workspace directory (created; must differ from $1)
#   $3  optimisation steps (default 7000)
#
# Environment:
#   PY_GS   python interpreter with gsplat (default: python on PATH)
#   PY_JEPA python interpreter with the Gaussian-JEPA CUDA extensions
#   GPU     CUDA_VISIBLE_DEVICES for training (default 0)
#   MAXSZ   long-edge cap for prepared frames (default 1600)
#   SEED    random seed (default 0)
#   RESUME  set to 1 to accept an already-complete dataset and skip to training
#   REGISTRY  JSONL registry path (default: <workspace>/../registry.jsonl)
set -euo pipefail

SAMPLE="$(realpath "${1:?usage: run_paa_pilot.sh SAMPLE_DIR WORKSPACE [STEPS]}")"
SAMPLE_NAME="$(basename "$SAMPLE")"
CROP="$(basename "$(dirname "$SAMPLE")")"
WS="$(realpath -m "${2:?usage: run_paa_pilot.sh SAMPLE_DIR WORKSPACE [STEPS]}")"
STEPS="${3:-7000}"
PY_GS="${PY_GS:-python}"
PY_JEPA="${PY_JEPA:-$PY_GS}"
GPU="${GPU:-0}"
MAXSZ="${MAXSZ:-1600}"
SEED="${SEED:-0}"
RESUME="${RESUME:-0}"
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
        print("  gpu", i, torch.get_device_name(i))
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
step "1/7 COLMAP audit"
"$PY_GS" "$PAA/colmap_check.py" --sample-dir "$SAMPLE" \
  --json "$WS/colmap_audit.json" 2>&1 | tee "$WS/logs/01_colmap_audit.log"

# --------------------------------------------------------------------------- #
# 2. prepare frames, masks and foreground points
#
# The mask crop offsets are measured on THIS sample. A sample whose masks do
# not register fails here instead of inheriting another plant's offsets.
# --------------------------------------------------------------------------- #
step "2/7 prepare dataset"
PREPARE_FLAGS=(--max-size "$MAXSZ" --seed "$SEED")
if [ "$RESUME" = "1" ]; then
  PREPARE_FLAGS+=(--resume)
fi
"$PY_GS" "$PAA/prepare_paa.py" --sample "$SAMPLE" --workspace "$WS" \
  "${PREPARE_FLAGS[@]}" 2>&1 | tee "$WS/logs/02_prepare.log"

# --------------------------------------------------------------------------- #
# 3. 3DGS optimisation with the validated foreground trainer
#
# `splats.pt` marks a finished run. When it exists but `splats_best.pt` does
# not (a crash after training, before best selection), the run is NOT skipped:
# the trainer owns best selection, so it is re-entered rather than guessed at.
# --------------------------------------------------------------------------- #
step "3/7 train 3DGS ($STEPS steps)"
if [ -f "$WS/results/train/splats.pt" ]; then
  echo "training already complete; skipping"
elif [ -f "$WS/results/train/splats_best.pt" ]; then
  echo "found splats_best.pt but no splats.pt: re-entering the trainer to finish"
  "$PY_GS" -u "$CODE/foreground/train.py" \
    --data "$WS/dataset" --result "$WS/results/train" \
    --steps "$STEPS" --eval-every 1000 --save-every 2000 --downscale 1 \
    --sh-degree 3 --foreground-alpha-weight 0.1 --background-alpha-weight 0.75 \
    --max-gaussians 300000 --seed "$SEED" 2>&1 | tee "$WS/logs/03_train.log"
else
  "$PY_GS" -u "$CODE/foreground/train.py" \
    --data "$WS/dataset" --result "$WS/results/train" \
    --steps "$STEPS" --eval-every 1000 --save-every 2000 --downscale 1 \
    --sh-degree 3 --foreground-alpha-weight 0.1 --background-alpha-weight 0.75 \
    --max-gaussians 300000 --seed "$SEED" 2>&1 | tee "$WS/logs/03_train.log"
fi

# --------------------------------------------------------------------------- #
# 4. clean the cloud down to the specimen and export a standard 3DGS PLY
# --------------------------------------------------------------------------- #
step "4/7 export"
if [ -f "$WS/final/point_cloud.ply" ] && [ "$WS/final/point_cloud.ply" -nt \
     "$WS/results/train/splats_best.pt" ]; then
  echo "export is newer than the checkpoint; skipping"
else
  "$PY_GS" "$CODE/foreground/export.py" \
    --checkpoint "$WS/results/train/splats_best.pt" --output "$WS/final" \
    2>&1 | tee "$WS/logs/04_export.log"
fi

# --------------------------------------------------------------------------- #
# 5. structural validation of the PLY
# --------------------------------------------------------------------------- #
step "5/7 PLY validation"
"$PY_GS" "$CODE/validate_ply.py" "$WS/final/point_cloud.ply" \
  --json "$WS/final/ply_validation.json" 2>&1 | tee "$WS/logs/05_ply.log"

# --------------------------------------------------------------------------- #
# 6. Gaussian-JEPA reader replay, forward/backward, optimiser + EMA probe.
#
#    These run in the gaussian_jepa env, which is where the compiled
#    pointnet2_ops / knn_cuda extensions live. The structural PLY checks in
#    step 5 stay on the gsplat interpreter, which is where plyfile and cv2 work.
# --------------------------------------------------------------------------- #
step "6/7 Gaussian-JEPA compatibility + forward/backward + optimiser probe"
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
# 7. Gaussian statistics, then register the sample for batch accounting and
#    split safety. Registration is an upsert keyed on crop/acquisition, so
#    re-running a sample updates its row instead of appending a second one.
# --------------------------------------------------------------------------- #
step "7/7 gaussian stats + registry"
"$PY_GS" "$PAA/gaussian_stats.py" "$WS/final/point_cloud.ply" \
  --json "$WS/final/gaussian_stats.json" 2>&1 | tee "$WS/logs/07_stats.log"
REGISTRY="${REGISTRY:-$WS/../registry.jsonl}"
"$PY_GS" "$PAA/registry.py" --workspace "$WS" --sample "$SAMPLE_NAME" \
  --crop "$CROP" --registry "$REGISTRY" 2>&1 | tee "$WS/logs/08_registry.log"

printf '\nPILOT_COMPLETE %s\n' "$WS"
