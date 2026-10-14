# Real-plant Gaussian-JEPA smoke test

This module proves the **real 3DGS PLY -> dataset -> sampling -> CUDA FPS/KNN ->
full Gaussian-JEPA -> finite loss -> finite, nonzero trainable gradients** path.
It does **not** measure plant recognition, disease prediction, generalization,
or the usefulness of learned representations. No model/optimizer weights are
saved or updated. In train mode, BatchNorm running buffers can change in memory.

The official model, configurations, datasets and random-input smoke test are
left unchanged. `PlantGaussianDataset` is an isolated map-style PyTorch Dataset
returning a dictionary of points and provenance. It is not yet registered with
the official full training runner; adapting its five-field training interface
and a multi-plant manifest is a later task.

## Files

- `dataset.py`: standard 3DGS PLY decoding, validated normalization profiles,
  deterministic whole-row subsampling and full-asset CPU caching.
- `smoke_real_plant.py`: one real-input forward/backward with strict finite
  checks, branch gradient validation, optional strict checkpoint loading,
  eval feature probes, JSON report and NPZ artifacts.
- `validation.py`: trainable/frozen gradient inspection.
- `tests/`: CPU contract tests, including missing/non-finite PLY attributes,
  normalization equivalence to official code, sampling and quaternion edges.

## Input contract

Use a standard **Gaussian** `point_cloud.ply`, NOT an XYZ/RGB centers PLY.
The required fields are `x,y,z,opacity,scale_0..2,rot_0..3,f_dc_0..2`.
Raw opacity is a logit and scales are logarithmic. The output order is:

```
xyz(3), sigmoid(opacity)(1), exp(scale)(3), normalized quaternion(4), SH DC(3)
```

High-order SH fields and normals are ignored by the official 14-D model.
The official reader is reused. A local wrapper fixes the exact `qw == 0`
quaternion sign boundary without changing upstream code; zero quaternions fail.

### Normalization is explicit

- `--normalization pretrain` (default): full-object XYZ mean/radius, physical
  scales divided by that radius, opacity mapped to [-1,1], scales centered and
  scaled by their maximum centered norm, SH DC converted/clipped as in the
  official `ShapeNetGaussian.pc_norm_gs` with all normalization attributes.
- `--normalization downstream`: full-object XYZ mean/radius and physical scales
  divided by that radius, matching `tools/extract_features.py`.

The statistics are computed **before sampling** and stored in the report.
Pretrain-normalized scale channels can be negative; they are model features,
not physical Gaussian sizes and must not be exported as renderable raw scales.

Sampling is uniform, seeded, without replacement when N >= budget; otherwise
existing complete rows are sampled with replacement (never zero padded).
`set_epoch()` changes the sampling seed deterministically. Different samples
of the same PLY are not independent plants or a valid train/validation split.
The optional `--model-seed` separates initialization from sampling variation.

## Reproduce

From `/fj/AgriGaussian-JEPA/Gaussian-JEPA`:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/test/biosoft/enter/envs/gaussian_jepa/bin/python \
  agri/plant_gs/jepa/smoke_real_plant.py \
  --ply /fj/AgriGaussian-JEPA/data/plant_gs/XianKeLai1_fg_20261014/final/point_cloud.ply \
  --normalization pretrain --num-gaussians 1024 --seed 0 --model-seed 0 \
  --output /fj/AgriGaussian-JEPA/data/plant_gs/XianKeLai1_fg_20261014/jepa_smoke_20261014/real_seed0.json
```

For a **trusted** compatible full pretraining checkpoint, add `--checkpoint`.
The default is explicitly random initialization, NOT pretrained weights.
The official loader is reused with strict model state loading and the checkpoint
SHA-256 is recorded. Only use checkpoints from a trusted source.

The source PLY and parameter SHA-256 are checked before/after the run. Outputs:

- `.json`: source path/hash, normalization stats, input contract, losses,
  gradients, checkpoint status, environment, timing/memory and scope limits.
- `.npz`: sampled 14-D attributes and original indices, token features, pooled
  embedding and group centers. Arrays come from eval mode before BN train
  buffers change. Random-weight embeddings are marked as such.

Feature probes verify identical-input repeatability and sensitivity when SH DC
or all non-XYZ channels are zeroed. These are **model-space ablations**, not
valid renderable Gaussians. Differences demonstrate input sensitivity only,
not useful semantics or downstream performance.

## 2026-10-14 result

Source: 287,079 Gaussians, plant + pot + blue block (all original SAM foreground).
The original source PLY is unchanged.

- Official synthetic smoke: PASS (`synthetic_smoke.log`).
- Real sampling seeds 0, 1, 2: all PASS with the same random initialization.
- Input `[1,1024,14]`; token features `[1,64,384]`; pooled feature `[1,768]`.
- All 215 trainable parameter tensors have finite gradients; frozen teacher
  tensors have none. Required encoder/predictor/projection branches have
  nonzero gradients.
- Total losses: 1.124135 / 1.124407 / 1.123125.
- Gradient L2 norms: 2.226385 / 2.259852 / 2.324147.
- Repeated identical eval input: maximum absolute token difference 0.
- No optimizer step, no EMA update, model parameters unchanged.
- Semantic effectiveness: **NOT EVALUATED**.
- RTX A6000 physical GPU 1, Python 3.9.23, torch 2.0.1, CUDA build 11.8.

Artifacts:
`/fj/AgriGaussian-JEPA/data/plant_gs/XianKeLai1_fg_20261014/jepa_smoke_20261014/`

Tests (default Python has pytest; dedicated JEPA env has CUDA operators):

```bash
OMP_NUM_THREADS=4 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q agri/plant_gs/jepa/tests agri/plant_gs/foreground/tests
# 24 passed (17 JEPA contracts + 7 foreground reconstruction regression tests)
```

## Next gate, before large-scale data collection

The engineering gate is passed. Start with a small set of **independent plants**
to establish sampling coverage, plant-only foreground if needed, a real optimizer
+ EMA training pilot and an independent-plant downstream/retrieval evaluation.
Only after defining task/labels and validating that pilot should collection and
pretraining be scaled up. One plant and finite smoke losses cannot decide the
benefit of pretraining versus a frozen encoder/other baselines.

Official references:
- https://github.com/Amazingren/Gaussian-JEPA/blob/main/tools/smoke_test_release.py
- https://github.com/Amazingren/Gaussian-JEPA/blob/main/datasets/ShapeNet55Gaussian.py
- https://github.com/Amazingren/Gaussian-JEPA/blob/main/tools/extract_features.py
- https://github.com/Amazingren/Gaussian-JEPA/blob/main/CHECKPOINTS.md
