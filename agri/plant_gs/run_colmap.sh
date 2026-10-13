#!/usr/bin/env bash
# Stage 1: COLMAP structure-from-motion for the XianKeLai1 cyclamen capture.
#
# Capture is a HANDHELD ORBIT (background changes frame to frame, subject stays
# centred) -- not a turntable. That is the good case for SfM: the static room
# gives strong parallax, so poses are recovered from background features and the
# plant itself needs no masking at this stage.
#
# NOTE: this COLMAP build has no CUDA, so BOTH use_gpu flags must be 0. Leaving
# SiftMatching.use_gpu at its default makes the CPU-only matcher try to create a
# GL context, which aborts inside Qt ("could not connect to display").
#
# Usage: RESUME_FROM=match|mapper ./run_colmap.sh   to skip completed stages.
set -euo pipefail

WS=/fj/AgriGaussian-JEPA/data/plant_gs/plant_001
DB=$WS/colmap/database.db
SP=$WS/colmap/sparse
IMG=$WS/images
LOG=$WS/logs
RESUME_FROM="${RESUME_FROM:-}"

mkdir -p "$WS/colmap" "$SP" "$LOG"

STAGES=(database features match mapper)
should_run() {
  local want="$1" seen=0 s
  [[ -z "$RESUME_FROM" ]] && return 0
  for s in "${STAGES[@]}"; do
    [[ "$s" == "$RESUME_FROM" ]] && seen=1
    [[ "$s" == "$want" ]] && { [[ $seen == 1 ]] && return 0 || return 1; }
  done
  return 1
}

if should_run database; then
  echo "=== [1/4] database_creator ==="
  rm -f "$DB" "$DB-wal" "$DB-shm"
  # NB: in COLMAP 3.9.1 the camera model is an option of feature_extractor,
  # not of database_creator.
  colmap database_creator --database_path "$DB" 2>&1 | tee "$LOG/colmap_01_database.log"
fi

if should_run features; then
  echo "=== [2/4] feature_extractor ==="
  colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMG" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0 \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.num_threads 16 \
    2>&1 | tee "$LOG/colmap_02_features.log"
fi

if should_run match; then
  echo "=== [3/4] sequential_matcher ==="
  # Sequential matching fits an orbit video: adjacent frames are a few degrees
  # apart, so a temporal window of 30 covers a wide arc. quadratic_overlap
  # widens that window for temporally distant frames.
  colmap sequential_matcher \
    --database_path "$DB" \
    --SequentialMatching.overlap 30 \
    --SequentialMatching.quadratic_overlap 1 \
    --SequentialMatching.loop_detection 0 \
    --SiftMatching.use_gpu 0 \
    --SiftMatching.num_threads 16 \
    2>&1 | tee "$LOG/colmap_03_match.log"
fi

if should_run mapper; then
  echo "=== [4/4] mapper ==="
  rm -rf "$SP"; mkdir -p "$SP"
  colmap mapper \
    --database_path "$DB" \
    --image_path "$IMG" \
    --output_path "$SP" \
    --Mapper.ba_global_function_tolerance 1e-7 \
    2>&1 | tee "$LOG/colmap_04_mapper.log"
fi

echo "=== done ==="
ls -la "$SP"