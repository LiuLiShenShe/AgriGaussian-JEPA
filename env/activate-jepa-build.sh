# Toolchain variables for the one operation that needs them: BUILDING the
# Gaussian-JEPA CUDA extensions (pointnet2_ops, knn_cuda).
#
#   source env/activate-jepa-build.sh
#   bash env/bootstrap.sh jepa
#
# Do NOT source this into a shell that also runs 3DGS work. CUDA 11.8 + gcc-11
# are what the extensions require, but the 3DGS environment's torch was built
# against CUDA 12.8 and torch refuses to build extensions when CUDA_HOME points
# elsewhere. bootstrap.sh exports these internally, so sourcing this file is only
# needed for a manual build.

AGRI_ENVS_ROOT="${AGRI_ENVS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/envs}"

export AGRI_ENVS_ROOT
export AGRI_PY_JEPA="$AGRI_ENVS_ROOT/agri-jepa/bin/python"
export PY_JEPA="$AGRI_PY_JEPA"

export CUDA_HOME="$AGRI_ENVS_ROOT/toolchain/cuda-11.8"
export CC="$AGRI_ENVS_ROOT/toolchain/gcc-11/usr/bin/gcc-11"
export CXX="$AGRI_ENVS_ROOT/toolchain/gcc-11/usr/bin/g++-11"
export CUDAHOSTCXX="$CXX"
export PATH="$AGRI_ENVS_ROOT/toolchain/gcc-11/usr/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9}"
export MAX_JOBS="${MAX_JOBS:-16}"
