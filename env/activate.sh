# Source this file to use the project's two interpreters:
#
#   source env/activate.sh
#   "$AGRI_PY_GS"   -m pytest ...
#   "$AGRI_PY_JEPA" tools/smoke_test_release.py
#
# There is deliberately no `activate` shell function and no global Python on
# PATH. The project needs two *different* interpreters (Python 3.13 / torch
# 2.9.1+cu128 for 3DGS; Python 3.9 / torch 2.0.1+cu118 for Gaussian-JEPA), so a
# command names the one it wants instead of inheriting whichever environment
# happened to be activated.
#
# The CUDA-11.8 toolchain variables are NOT exported by this file. They belong to
# the JEPA build only, and exporting them globally breaks the 3DGS side, whose
# torch was built against CUDA 12.8: torch's own version check then fails with
# "detected CUDA version 11.8 mismatches 12.8". Source
# env/activate-jepa-build.sh for the one operation that needs them (building
# pointnet2_ops / knn_cuda, or forcing knn_cuda's JIT rebuild).

AGRI_ENVS_ROOT="${AGRI_ENVS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/envs}"

export AGRI_ENVS_ROOT
export AGRI_PY_GS="$AGRI_ENVS_ROOT/agri-gs/bin/python"
export AGRI_PY_JEPA="$AGRI_ENVS_ROOT/agri-jepa/bin/python"

# Scripts in this repo already read PY_GS / PY_JEPA.
export PY_GS="$AGRI_PY_GS"
export PY_JEPA="$AGRI_PY_JEPA"
