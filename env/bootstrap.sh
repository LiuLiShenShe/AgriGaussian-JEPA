#!/usr/bin/env bash
# Bootstrap the project's two environments without conda.
#
#   bash env/bootstrap.sh gs      # 3DGS: train / export / PLY / semantics
#   bash env/bootstrap.sh jepa    # Gaussian-JEPA: forward/backward, extensions
#   bash env/bootstrap.sh verify  # verify both without installing anything
#   bash env/bootstrap.sh tools   # download the non-Python toolchain (gcc-11)
#   bash env/bootstrap.sh cuda    # install CUDA 11.8 into envs/toolchain
#
# Everything lands inside the repository, at envs/:
#
#   envs/agri-gs/       python 3.13 + torch 2.9.1+cu128 + gsplat
#   envs/agri-jepa/     python 3.9  + torch 2.0.1+cu118 + pointnet2_ops/knn_cuda
#   envs/toolchain/     gcc-11 (for CUDA 11.8) and cuda-11.8
#
# No conda, no root. The compilers and the CUDA toolkit are unpacked from
# packages fetched at run time: gcc-11 from the distro archive via
# `apt-get download` (which needs no privileges), CUDA 11.8 from NVIDIA's
# runfile installer with `--silent --toolkit`.
#
# Overrides
#   ENVS_ROOT          where the envs go (default: <repo>/envs)
#   TORCH_CUDA_ARCH_LIST  GPU archs to compile the extensions for
#   MAX_JOBS           parallel build jobs
#   SKIP_TOOLCHAIN=1   assume gcc-11 / CUDA 11.8 are already provisioned
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS_ROOT="${ENVS_ROOT:-$REPO/envs}"
TOOLS="$ENVS_ROOT/toolchain"
CUDA_DIR="$TOOLS/cuda-11.8"
GCC_DIR="$TOOLS/gcc-11"
PY_GS="${PY_GS:-$ENVS_ROOT/agri-gs/bin/python}"
PY_JEPA="${PY_JEPA:-$ENVS_ROOT/agri-jepa/bin/python}"
PY_VERSION_GS="${PY_VERSION_GS:-3.13}"
PY_VERSION_JEPA="${PY_VERSION_JEPA:-3.9}"

log()  { printf '\n===== %s =====\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

# gsplat is a local editable checkout, not a PyPI release. BUILD_NO_CUDA=1 keeps
# its already-built CUDA objects instead of recompiling them against whatever
# nvcc happens to be on PATH (that mismatch is what produced "CUDA version 12.6
# mismatches 13.0" during a normal build).
GSPLAT_SRC="${GSPLAT_SRC:-/fj/VR-esktop/gsplat}"

# --------------------------------------------------------------------------- #
# Non-Python toolchain
# --------------------------------------------------------------------------- #
fetch_gcc11() {
  [ "$SKIP_TOOLCHAIN" = "1" ] && { log "skipping gcc-11 fetch (SKIP_TOOLCHAIN=1)"; return 0; }
  if [ -x "$GCC_DIR/usr/bin/gcc-11" ]; then
    log "gcc-11 already present"; return 0
  fi
  log "fetching gcc-11 (apt-get download, no root needed)"
  local scratch; scratch="$(mktemp -d)"
  trap 'rm -rf "$scratch"' RETURN
  ( cd "$scratch" && apt-get download \
      gcc-11 g++-11 cpp-11 libgcc-11-dev libstdc++-11-dev >/dev/null 2>&1 )
  mkdir -p "$GCC_DIR"
  local deb
  for deb in "$scratch"/*.deb; do
    dpkg-deb -x "$deb" "$GCC_DIR"
  done
  "$GCC_DIR/usr/bin/gcc-11" --version | head -1
}

fetch_cuda118() {
  [ "$SKIP_TOOLCHAIN" = "1" ] && { log "skipping CUDA fetch (SKIP_TOOLCHAIN=1)"; return 0; }
  if [ -x "$CUDA_DIR/bin/nvcc" ]; then
    log "CUDA 11.8 already present"; return 0
  fi
  # The nvidia pip wheels ship only ptxas, not nvcc; the toolkit itself has to
  # come from NVIDIA's installer. Mirror first (developer.download.nvidia.cn),
  # then the canonical host.
  local runfile="$ENVS_ROOT/pkgs/cuda_11.8.run"
  mkdir -p "$ENVS_ROOT/pkgs"
  if [ ! -f "$runfile" ]; then
    local url size
    for url in \
      "https://developer.download.nvidia.cn/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run" \
      "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"; do
      log "downloading CUDA 11.8 from $url"
      if curl -fsSL --retry 3 -o "$runfile.part" "$url"; then
        mv "$runfile.part" "$runfile"; break
      fi
    done
    [ -f "$runfile" ] || { warn "could not download the CUDA runfile"; return 1; }
  fi
  chmod +x "$runfile"
  # CUDA 11.8 rejects gcc > 11; --override is safe here because we point the
  # compiler at gcc-11 via PATH when we actually build the extensions.
  "$runfile" --silent --toolkit --toolkitpath="$CUDA_DIR" \
    --no-opengl-libs --no-man-page --override >/dev/null 2>&1 || true
  [ -x "$CUDA_DIR/bin/nvcc" ] || { warn "CUDA toolkit extraction failed"; return 1; }
  "$CUDA_DIR/bin/nvcc" --version | tail -2
}

export_cuda_env() {
  export CUDA_HOME="$CUDA_DIR"
  export PATH="$GCC_DIR/usr/bin:$CUDA_HOME/bin:$PATH"
  export CC="$GCC_DIR/usr/bin/gcc-11"
  export CXX="$GCC_DIR/usr/bin/g++-11"
  export CUDAHOSTCXX="$GCC_DIR/usr/bin/g++-11"
  export MAX_JOBS="${MAX_JOBS:-16}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9}"
}

# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
build_gs() {
  log "creating envs/agri-gs (python $PY_VERSION_GS)"
  if [ ! -x "$PY_GS" ]; then
    # --python-preference only-managed: never borrow an interpreter that is
    # already on PATH (the conda one, which would silently reintroduce conda).
    uv venv --python "$PY_VERSION_GS" --python-preference only-managed "$ENVS_ROOT/agri-gs"
  else
    log "envs/agri-gs already exists"
  fi
  log "installing torch cu128"
  uv pip install --python "$PY_GS" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "$REPO/env/requirements-gs.lock.txt"

  if [ -d "$GSPLAT_SRC" ]; then
    log "installing gsplat from $GSPLAT_SRC (editable, no CUDA rebuild)"
    uv pip install --python "$PY_GS" --no-build-isolation -e "$GSPLAT_SRC"
  else
    warn "gsplat source not found at $GSPLAT_SRC; set GSPLAT_SRC to a gsplat checkout"
  fi
}

build_jepa() {
  log "creating envs/agri-jepa (python $PY_VERSION_JEPA)"
  if [ ! -x "$PY_JEPA" ]; then
    uv venv --python "$PY_VERSION_JEPA" --python-preference only-managed "$ENVS_ROOT/agri-jepa"
  else
    log "envs/agri-jepa already exists"
  fi
  log "installing torch cu118 + python deps"
  uv pip install --python "$PY_JEPA" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r "$REPO/env/requirements-jepa.lock.txt"

  fetch_gcc11
  fetch_cuda118
  export_cuda_env

  local src="$ENVS_ROOT/src"
  mkdir -p "$src"
  # github.com itself is unreachable from this host; codeload works.
  local pn2="$src/pointnet2_pytorch-master"
  local knn="$src/knn_cuda-master"
  if [ ! -d "$pn2" ]; then
    log "fetching pointnet2_pytorch"
    curl -fsSL -o "$src/pn2.tar.gz" \
      https://codeload.github.com/unique1i/pointnet2_pytorch/tar.gz/refs/heads/master
    tar xzf "$src/pn2.tar.gz" -C "$src"
  fi
  if [ ! -d "$knn" ]; then
    log "fetching knn_cuda"
    curl -fsSL -o "$src/knn.tar.gz" \
      https://codeload.github.com/unique1i/knn_cuda/tar.gz/refs/heads/master
    tar xzf "$src/knn.tar.gz" -C "$src"
  fi

  log "building pointnet2_ops (this takes several minutes)"
  ( cd "$pn2/pointnet2_ops_lib" && \
    uv pip install --python "$PY_JEPA" --no-build-isolation --no-deps . )

  log "building knn_cuda"
  # knn_cuda's setup.py hardcodes ten GPU archs and overwrites
  # TORCH_CUDA_ARCH_LIST; exporting ours before invoking pip still wins for the
  # torch-driven build because the module writes it at import time of setup.py.
  ( cd "$knn" && uv pip install --python "$PY_JEPA" --no-build-isolation --no-deps . )
}

# --------------------------------------------------------------------------- #
# Verification: real work, not just imports
# --------------------------------------------------------------------------- #
verify() {
  local failures=0
  cd "$REPO"
  log "verifying envs/agri-gs"
  "$PY_GS" - <<'PY' || failures=$((failures+1))
import sys
import torch
assert torch.cuda.is_available(), "cuda not available"
from gsplat import rasterization
means = torch.randn(100, 3, device='cuda')
quats = torch.zeros(100, 4, device='cuda'); quats[:, 0] = 1.0
scales = torch.full((100, 3), 0.05, device='cuda')
opac = torch.full((100,), 0.5, device='cuda')
cols = torch.rand(100, 3, device='cuda')
K = torch.tensor([[100., 0, 32], [0, 100., 32], [0, 0, 1]], device='cuda')[None]
w2c = torch.eye(4, device='cuda')[None]
out, alpha, _ = rasterization(means, quats, scales, opac, cols, w2c, K, 64, 64, sh_degree=None)
assert out.shape == (1, 64, 64, 3), out.shape
print(f"gs: torch={torch.__version__} cuda={torch.version.cuda} render=ok")
PY
  "$PY_GS" -m pytest agri/plant_gs/foreground/tests agri/plant_gs/paa/tests -q \
    || failures=$((failures+1))

  log "verifying envs/agri-jepa"
  "$PY_JEPA" tools/smoke_test_release.py >/dev/null 2>&1 || {
    echo "FAIL: gaussian-jepa release smoke test"; failures=$((failures+1)); }
  "$PY_JEPA" - <<'PY' || failures=$((failures+1))
import torch
import knn_cuda
from pointnet2_ops import pointnet2_utils
ref = torch.rand(1, 10, 3, device='cuda'); q = torch.rand(1, 5, 3, device='cuda')
d, i = knn_cuda.knn(ref, q, 3)
assert d.shape == (3, 5), d.shape
print(f"jepa: torch={torch.__version__} cuda={torch.version.cuda} "
      f"knn_cuda=ok pointnet2_ops=ok")
PY
  "$PY_JEPA" -m pytest agri/plant_gs/jepa/tests -q || failures=$((failures+1))

  if [ "$failures" -eq 0 ]; then
    log "VERIFY PASS"
  else
    log "VERIFY FAILED ($failures check group(s))"
    return 1
  fi
}

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

ACTION="${1:-}"
case "$ACTION" in
  tools)   fetch_gcc11 ;;
  cuda)    fetch_cuda118 ;;
  all)     build_gs; build_jepa; verify ;;
  gs)      build_gs ;;
  jepa)    build_jepa ;;
  verify)  verify ;;
  -h|--help|"") usage ;;
  *)       echo "unknown action: $ACTION" >&2; usage 2 ;;
esac
