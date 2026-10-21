#!/usr/bin/env python
"""Stage 4D: the evaluation protocol Stage 5 must follow, made executable.

Nothing here trains or changes a model. It builds, from the assets that
actually exist, the split lists and stability measurements a Stage 5 run would
need -- and it writes the caveats into the same file as the numbers, so a
reader cannot lift a split without also seeing that seven plants cannot support
a generalisation claim.

The four protocols
------------------
1. **Plant-grouped split.** The unit is ``<crop>/<plant_id>``; every acquisition
   of a plant lands in the same split. Same-plant leakage is checked by name.
2. **Repeated-sampling stability.** The same asset is drawn many times at the
   same budget under different seeds; the spread of voxel coverage and class
   composition is the noise floor any JEPA comparison must clear.
3. **Partial-occlusion strata.** ``gaussian_classes.npz`` records how many views
   saw each Gaussian. Gaussians seen by few views are the occlusion analogue of
   a held-out view; the strata are reported with their plant fraction so a
   future "robustness" claim has a definition to be measured against.
4. **Leave-one-crop-out.** Six crops means six folds; the per-fold spread is the
   only honest way to read a small multi-crop library.

Config templates
----------------
Three templates are written under ``stage5_configs/``: Random Encoder,
Gaussian-JEPA and Gaussian-MAE. The Gaussian-JEPA template copies the official
model block verbatim -- the model and objective are not Stage 4D's to change.
The other two are *comparison* arms, and their point is that the encoder and
the objective differ, not that either is expected to win on seven plants.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agri.plant_gs.jepa.dataset import (  # noqa: E402
    load_gaussian_ply, normalize_gaussians, sample_gaussians)

SCHEMA_VERSION = "paa-stage4d-stage5-protocol-1"

BACKGROUND, PLANT, POT, SOIL, SUPPORT = 0, 1, 2, 3, 4


def log(message: str) -> None:
    print(message, flush=True)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# 1. the split
# --------------------------------------------------------------------------- #
def eligible(entries: list) -> list:
    return [e for e in entries
            if not e.get("sealed")
            and (e.get("states") or {}).get("states", {}).get(
                "reconstruction_verified")]


def plant_split(entries: list, seed: int = 20261020,
                ratios=(0.6, 0.2, 0.2)) -> dict:
    import random
    groups = sorted({e["split_group"] for e in entries if e.get("split_group")})
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * ratios[0]))) if n else 0
    n_val = max(1, int(round(n * ratios[1]))) if n > 2 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 2) if n > 2 else n
        n_val = 1 if n > 2 else 0
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train:n_train + n_val])
    test = sorted(shuffled[n_train + n_val:])
    coverage = {name: sorted(e["asset_id"] for e in entries
                             if e["split_group"] in set(members))
                for name, members in
                (("train", train), ("val", val), ("test", test))}
    return {"groups": {"train": train, "val": val, "test": test},
            "assets": coverage, "seed": seed,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]}}


def leakage_check(entries: list, split: dict) -> dict:
    """Same-plant leakage, plus the softer signals a reader should see."""
    placed = {}
    for name, groups in split["groups"].items():
        for group in groups:
            placed.setdefault(group, []).append(name)
    duplicate = {g: names for g, names in placed.items() if len(set(names)) > 1}

    plants_across_dates = {}
    for entry in entries:
        if entry.get("plant_id") is None:
            continue
        plants_across_dates.setdefault(entry["split_group"], set()).add(
            entry.get("acquisition_date"))
    multi_date = {group: sorted(d for d in dates if d)
                  for group, dates in plants_across_dates.items()
                  if len({d for d in dates if d}) > 1}

    crops_across_splits = {}
    for entry in entries:
        split_of = next((name for name, groups in split["groups"].items()
                         if entry["split_group"] in set(groups)), None)
        if split_of:
            crops_across_splits.setdefault(entry.get("crop"), set()).add(split_of)
    return {
        "split_group_in_two_splits": duplicate,
        "leakage": bool(duplicate),
        "plants_with_multiple_acquisitions": multi_date,
        "same_plant_multi_date_note": (
            "every acquisition of a plant must stay in one split; the list above "
            "shows which plants actually have more than one date, i.e. where a "
            "date-level split would have leaked"),
        "crops_present_in_multiple_splits": {
            k: sorted(v) for k, v in crops_across_splits.items() if len(v) > 1},
        "crop_overlap_is_not_plant_leakage": (
            "a crop appearing in train and test is allowed and is what makes "
            "leave-one-crop-out a different question"),
    }


def leave_one_crop_out(entries: list) -> dict:
    crops = sorted({e["crop"] for e in entries if e.get("crop")})
    folds = {}
    for crop in crops:
        test = sorted(e["asset_id"] for e in entries if e["crop"] == crop)
        train = sorted(e["asset_id"] for e in entries if e["crop"] != crop)
        folds[crop] = {"train_assets": train, "test_assets": test,
                       "n_train_plants": len({e["split_group"] for e in entries
                                              if e["crop"] != crop}),
                       "n_test_plants": len({e["split_group"] for e in entries
                                             if e["crop"] == crop})}
    return {"folds": folds, "n_folds": len(folds),
            "caveat": ("with 2-3 plants in several crops a fold has a training "
                       "set of 5-7 plants; report per-fold spread, never a "
                       "single mean")}


# --------------------------------------------------------------------------- #
# 2. repeated-sampling stability
# --------------------------------------------------------------------------- #
def _voxel_keys(points: np.ndarray, low: np.ndarray, extent: np.ndarray,
                grid: int = 16) -> np.ndarray:
    """Relative 16^3 lattice over the asset's bounding box.

    An absolute voxel size (0.05 m, say) is meaningless across crops whose
    world scales differ; the same lattice for every asset is what makes the
    coverage numbers comparable.
    """
    cell = np.clip(((points - low) / extent * grid).astype(np.int64), 0, grid - 1)
    return np.unique(cell[:, 0] * grid * grid + cell[:, 1] * grid + cell[:, 2])


def sampling_stability(entry: dict, budget: int = 1024, seeds=(0, 1, 2, 3, 4, 5, 6, 7),
                       grid: int = 16) -> dict:
    path = entry.get("gaussian_ply_path")
    if not path or not Path(path).is_file():
        return {"status": "skipped", "reason": "no PLY"}
    decoded = load_gaussian_ply(path)
    normalized, _metadata = normalize_gaussians(decoded, "pretrain")
    xyz = decoded[:, :3]
    low = xyz.min(axis=0)
    extent = np.maximum(xyz.max(axis=0) - low, 1e-9)
    reference = _voxel_keys(xyz, low, extent, grid)
    coverages, centroids = [], []
    for seed in seeds:
        _, indices = sample_gaussians(normalized, budget, seed)
        drawn = _voxel_keys(xyz[indices], low, extent, grid)
        coverages.append(len(np.intersect1d(reference, drawn)) / max(len(reference), 1))
        centroids.append(xyz[indices].mean(axis=0))
    return {
        "status": "measured",
        "budget": budget,
        "seeds": list(seeds),
        "grid": grid,
        "reference_voxels": int(len(reference)),
        "voxel_coverage_mean": float(np.mean(coverages)),
        "voxel_coverage_std": float(np.std(coverages)),
        "voxel_coverage_min": float(np.min(coverages)),
        "voxel_coverage_max": float(np.max(coverages)),
        "centroid_std": np.std(np.stack(centroids), axis=0).tolist(),
        "interpretation": ("a JEPA difference smaller than this seed-to-seed "
                           "spread is not a difference"),
    }


# --------------------------------------------------------------------------- #
# 3. partial-occlusion strata
# --------------------------------------------------------------------------- #
def occlusion_strata(entry: dict) -> dict:
    workspace = Path(entry.get("workspace") or "")
    path = workspace / "plant_only" / "gaussian_classes.npz"
    if not path.is_file():
        return {"status": "skipped", "reason": "no gaussian_classes.npz"}
    payload = np.load(path)
    seen = payload["seen"].astype(np.float64)
    classes = payload["classes"]
    quantiles = np.quantile(seen, [0.0, 0.25, 0.5, 0.75, 1.0])
    strata = []
    for low, high in zip(quantiles[:-1], quantiles[1:]):
        mask = (seen >= low) & (seen <= high)
        if not mask.any():
            continue
        strata.append({
            "seen_min": float(low), "seen_max": float(high),
            "n_gaussians": int(mask.sum()),
            "plant_fraction": float((classes[mask] == PLANT).mean()),
            "undecided_fraction": float((classes[mask] < 0).mean()),
        })
    return {
        "status": "measured",
        "n_gaussians": int(len(seen)),
        "seen_range": [float(seen.min()), float(seen.max())],
        "strata_by_visibility_quartile": strata,
        "protocol": ("report representation stability separately for the "
                     "lowest and highest visibility strata; a model that only "
                     "works on Gaussians seen by most views has not learned "
                     "partial occlusion"),
    }


# --------------------------------------------------------------------------- #
# 4. config templates
# --------------------------------------------------------------------------- #
JEPA_MODEL_BLOCK = """model:
  NAME: GaussianJEPA
  num_targets: 4
  target_sizes: [11, 9, 7, 5]
  group_size: 32
  num_group: 64
  attribute: [xyz, opacity, scale, rotation, sh]
  group_attribute: [xyz]
  soft_knn: false
  lambda_ground: 0.1
  num_swd_projections: 256
  transformer_config:
    mask_ratio: 0.55
    mask_type: block
    trans_dim: 384
    encoder_dims: 384
    depth: 12
    drop_path_rate: 0.1
    num_heads: 6
    decoder_depth: 4
    decoder_num_heads: 6
    pred_dim: 192
"""

COMMON = """seed: 20261021
num_gaussians: 1024
batch_size: 8
epochs: 100
learning_rate: 0.0003
weight_decay: 0.05
normalization: pretrain
optimizer: AdamW
scheduler: cosine
data:
  train_split: results/agri_paa/stage4d/stage5_protocol.json#/split/assets/train
  val_split: results/agri_paa/stage4d/stage5_protocol.json#/split/assets/val
  test_split: results/agri_paa/stage4d/stage5_protocol.json#/split/assets/test
  split_unit: <crop>/<plant_id>
evaluation:
  leave_one_crop_out: true
  repeated_sampling: {seeds: 8, budget: 1024}
  occlusion_strata: true
  report_per_fold: true
  claims_allowed: [within-library ranking, stability under resampling]
  claims_forbidden: [generalisation to unseen plants, segmentation accuracy,
                     semantic validity from loss alone]
"""

CONFIGS = {
    "random_encoder.yaml": COMMON + """
# Comparison arm 1: the Gaussian-JEPA architecture with a randomly initialised
# encoder that is never updated. It is the floor a learned encoder must beat.
model:
  NAME: RandomEncoder
  freeze_encoder: true
  encoder_arch: gaussian_jepa_encoder
  trans_dim: 384
  depth: 12
  num_heads: 6
  objective: none_linear_probe_only
""",
    "gaussian_jepa.yaml": COMMON + """
# The official model and objective, unchanged. Stage 4D does not modify either.
""".rstrip() + "\n" + JEPA_MODEL_BLOCK + """
ema_momentum_start: 0.996
ema_momentum_end: 0.9999
npoints: 1024
npoints_fps: false
total_bs: 256
step_per_update: 1
max_epoch: 100
""",
    "gaussian_mae.yaml": COMMON + """
# Comparison arm 2: the same encoder, mask-and-reconstruct in attribute space
# with no EMA target and no JEPA predictor. It isolates the objective.
""".rstrip() + "\n" + JEPA_MODEL_BLOCK.replace(
        "NAME: GaussianJEPA", "NAME: GaussianMAE").replace(
        "  lambda_ground: 0.1\n", "").replace(
        "  num_swd_projections: 256\n", "") + """
ema_momentum_start: 0.0
ema_momentum_end: 0.0
npoints: 1024
total_bs: 256
max_epoch: 100
""",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path,
                    default=Path("results/agri_paa/stage4c/assets.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/agri_paa/stage4d/stage5_protocol.json"))
    ap.add_argument("--config-dir", type=Path,
                    default=Path("results/agri_paa/stage4d/stage5_configs"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args()

    library = json.loads(args.library.read_text())
    entries = eligible(library["entries"])
    split = plant_split(entries)
    leakage = leakage_check(entries, split)
    locv = leave_one_crop_out(entries)

    stability = {}
    strata = {}
    if not args.skip_stability:
        for entry in entries:
            asset_id = entry["asset_id"]
            stability[asset_id] = sampling_stability(
                entry, budget=args.budget, seeds=tuple(range(args.seeds)))
            strata[asset_id] = occlusion_strata(entry)

    crops = sorted({e["crop"] for e in entries})
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc(),
        "library": str(args.library),
        "n_eligible_assets": len(entries),
        "n_plants": len({e["split_group"] for e in entries}),
        "n_crops": len(crops),
        "split": split,
        "leakage": leakage,
        "leave_one_crop_out": locv,
        "repeated_sampling": stability,
        "occlusion_strata": strata,
        "config_templates": {},
        "claims": {
            "allowed": ["within-library ranking of objectives",
                        "stability under resampling and visibility strata"],
            "forbidden": [
                "generalisation to unseen plants from this split",
                "segmentation accuracy (no human-confirmed label exists)",
                "semantic validity inferred from a lower training loss",
            ],
            "reason": (f"{len(entries)} assets from "
                       f"{len({e['split_group'] for e in entries})} plants is "
                       "enough to validate the evaluation machinery, not to "
                       "establish a representation claim"),
        },
    }

    args.config_dir.mkdir(parents=True, exist_ok=True)
    for name, text in CONFIGS.items():
        (args.config_dir / name).write_text(text)
        report["config_templates"][name] = str((args.config_dir / name).resolve())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps({
        "n_eligible_assets": report["n_eligible_assets"],
        "n_plants": report["n_plants"],
        "split": split["groups"],
        "leakage": leakage["leakage"],
        "loco_folds": locv["n_folds"],
        "stability_measured": sum(1 for v in stability.values()
                                  if v.get("status") == "measured"),
        "strata_measured": sum(1 for v in strata.values()
                               if v.get("status") == "measured"),
    }, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
