#!/usr/bin/env bash
# Uses full RGB for camera geometry. No background Gaussians are initialized.
set -euo pipefail
WS="$(realpath "${1:?usage: run_sfm.sh WORKSPACE}")"
export QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python - "$WS" <<'PY'
import json,sys
from pathlib import Path
p=json.loads((Path(sys.argv[1])/'preparation.json').read_text())
assert p['pairing_accepted'], 'Unresolved/unaudited mask correspondences'
assert p['paired_mask_count'] >= 20
PY
DB="$WS/colmap/database.db"
mkdir -p "$WS/colmap/sparse" "$WS/logs"
if [ ! -f "$WS/colmap/features.done" ]; then
  colmap feature_extractor --database_path "$DB" --image_path "$WS/images" \
    --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0 --SiftExtraction.max_image_size 1920 \
    --SiftExtraction.max_num_features 8192 --SiftExtraction.num_threads 16 \
    > "$WS/logs/colmap_features.log" 2>&1
  touch "$WS/colmap/features.done"
fi
if [ ! -f "$WS/colmap/matching.done" ]; then
  colmap sequential_matcher --database_path "$DB" \
    --SequentialMatching.overlap 15 --SequentialMatching.quadratic_overlap 1 \
    --SequentialMatching.loop_detection 0 --SiftMatching.use_gpu 0 \
    --SiftMatching.num_threads 16 > "$WS/logs/colmap_matching.log" 2>&1
  touch "$WS/colmap/matching.done"
fi
if [ ! -f "$WS/colmap/mapper.done" ]; then
  colmap mapper --database_path "$DB" --image_path "$WS/images" \
    --output_path "$WS/colmap/sparse" --Mapper.num_threads 16 \
    --Mapper.ba_global_function_tolerance 1e-6 \
    > "$WS/logs/colmap_mapper.log" 2>&1
  touch "$WS/colmap/mapper.done"
fi
printf 'SFM_COMPLETE %s\n' "$WS"
