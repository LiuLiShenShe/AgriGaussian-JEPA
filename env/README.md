# Environments (uv, no conda)

This project no longer uses conda. Two independent environments are declared as
locked uv requirement sets and built by `env/bootstrap.sh`:

| environment | path (default) | Python | PyTorch | why it is separate |
|---|---|---|---|---|
| 3DGS | `envs/agri-gs` | 3.13 | 2.9.1+cu128 | training, export, PLY validation, semantic labelling |
| Gaussian-JEPA | `envs/agri-jepa` | 3.9 | 2.0.1+cu118 | forward/backward, optimizer probe, compiled extensions |

They cannot be merged. The Gaussian-JEPA code base targets PyTorch 2.0.1 /
CUDA 11.8 and its two CUDA extensions are built against that ABI; the 3DGS side
uses gsplat, which needs a current PyTorch. So the repo always names the
interpreter a command needs (`$AGRI_PY_GS` or `$AGRI_PY_JEPA`) instead of
activating one global environment.

## Quick start

```bash
bash env/bootstrap.sh all      # first time: build both + toolchain, then verify
source env/activate.sh         # every shell
```

`source env/activate.sh` exports `AGRI_PY_GS`, `AGRI_PY_JEPA`, `PY_GS` and
`PY_JEPA`. The repo's scripts read `PY_GS` / `PY_JEPA`, so they work with no
further configuration.

The CUDA-11.8 / gcc-11 variables are **not** exported by `activate.sh`, because
they belong to the JEPA build only: exporting them while running 3DGS work makes
torch (cu128) refuse to build anything with a CUDA 11.8 `CUDA_HOME`. Source
`env/activate-jepa-build.sh` for a manual extension build; `bootstrap.sh` does
this internally.

Useful actions:

```bash
bash env/bootstrap.sh verify     # import both, run gsplat render, run smoke tests
bash env/bootstrap.sh gs         # (re)build only the 3DGS env
bash env/bootstrap.sh jepa       # (re)build only the JEPA env
python -m agri.plant_gs.paa.env_resolve --check
```

## Declarative files

| file | meaning |
|---|---|
| `requirements-gs.in` / `.lock.txt` | direct deps / fully pinned set for 3DGS |
| `requirements-jepa.in` / `.lock.txt` | direct deps / fully pinned set for JEPA |

Regenerate a lock after editing its `.in` file:

```bash
cd env
uv pip compile requirements-gs.in -o requirements-gs.lock.txt --python-version 3.13 \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
uv pip compile requirements-jepa.in -o requirements-jepa.lock.txt --python-version 3.9 \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --index-strategy unsafe-best-match
```

## What is not on PyPI, and how it is handled

Three pieces cannot come from a package index. `bootstrap.sh` treats each one
explicitly rather than pretending they are normal dependencies.

**gsplat** is an editable checkout, not a released wheel. `bootstrap.sh` installs
it from `$GSPLAT_SRC` (default `/fj/VR-esktop/gsplat`). The checkout ships a
prebuilt `gsplat/csrc.so`; installing with `BUILD_NO_CUDA=1` keeps that object
instead of rebuilding it against whatever `nvcc` is on `PATH`. Rebuilding it is
what produced `RuntimeError: CUDA version 12.6 mismatches 13.0` during the
migration: uv's build isolation installed a *newer* PyTorch (CUDA 13) to build
against, while the runtime env had cu128.

**pointnet2_ops and knn_cuda** are compiled from source tarballs
(`unique1i/pointnet2_pytorch`, `unique1i/knn_cuda`). `github.com` is not
reachable from this host; `codeload.github.com` is, so they are fetched as
tarballs. Both need CUDA 11.8 and gcc ≤ 11.

**The compiler and CUDA toolkit are fetched without root and without conda:**

- gcc-11 comes from the distro archive via `apt-get download` (no privileges
  required), then `dpkg-deb -x` into `envs/toolchain/gcc-11`. Ubuntu 24.04's
  default gcc-13 cannot compile CUDA 11.8 headers.
- CUDA 11.8 comes from NVIDIA's runfile with
  `--silent --toolkit --toolkitpath=envs/toolchain/cuda-11.8 --override`.
  The `--override` is only bypassing the installer's compiler check; the actual
  compilation uses the gcc-11 above.
  The `nvidia-cuda-nvcc-cu11` wheel ships `ptxas` but **not** `nvcc`, so the
  wheel alone is insufficient.
- `knn_cuda` JIT-compiles on first import into
  `site-packages/knn_cuda/csrc/_ext/knn/`. The bootstrap pre-builds it once, so
  a normal import is a no-op and needs no toolchain. If the environment is moved,
  or torch is reinstalled, the cached build is invalidated and the next import
  triggers a rebuild -- source `env/activate-jepa-build.sh` before that import.

## What changed relative to the conda setup

| conda | now |
|---|---|
| `conda env create -f environment.yml` | `bash env/bootstrap.sh all` |
| `conda activate gaussian_jepa` | `source env/activate.sh` |
| `python` after activation | `$AGRI_PY_GS` or `$AGRI_PY_JEPA` (explicit) |
| `CUDA_HOME=$CONDA_PREFIX` | `CUDA_HOME=envs/toolchain/cuda-11.8` |
| conda `gxx_linux-64=11` | extracted `gcc-11` under `envs/toolchain` |
| conda `cuda-nvcc=11.8` + cccl | CUDA 11.8 toolkit + gcc-11 |

Environment notes for the record:

- uv's `--python-preference only-managed` is mandatory. Without it, uv finds the
  conda interpreter already on `PATH` and builds the venv on top of it, which
  quietly reintroduces the dependency this migration removes.
- `setuptools<81` is pinned because PyTorch 2.0.1's `cpp_extension` imports
  `pkg_resources`, which setuptools 81+ removed.

## Verification performed

`bash env/bootstrap.sh verify` checks real work, not just imports:

- 3DGS env: CUDA visible, a real `gsplat.rasterization` call producing a
  `(1, 64, 64, 3)` image, and the foreground/PAA test suites.
- JEPA env: the release smoke test (`tools/smoke_test_release.py`), a real
  `knn_cuda.knn` call, `pointnet2_ops` import, and the JEPA test suite.
- End-to-end, additionally checked by hand during the migration: a 60-step
  foreground training run, and `smoke_real_plant.py` on
  `Cotton_1_repro_v1/final/point_cloud.ply`, which reproduced the recorded
  losses exactly (geo 0.463949 / app 0.459002 / ground 0.201430, gradient norm
  2.281867) — i.e. the new environment is numerically equivalent to the old one.

Historical reports under `reports/` and `results/` are kept verbatim because
they record what was actually run; each carries a note pointing here.
