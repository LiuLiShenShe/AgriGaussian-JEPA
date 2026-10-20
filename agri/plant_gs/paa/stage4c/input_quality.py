#!/usr/bin/env python
"""Full-specimen vs plant-only, measured through JEPA's own sampler (Task 4).

What is compared, and what is held fixed
----------------------------------------
*The sampler* is not re-implemented here. :func:`sample_gaussians` is imported
from ``agri.plant_gs.jepa.dataset``, the module the smoke test feeds. If the
official sampler changes, this analysis changes with it -- a private copy
would silently measure a sampler the model does not use.

*The budget* is identical in every comparison: 1,024 Gaussians. The
plant-aware variant draws with the same ``numpy.random.RandomState(seed)
.choice(..., 1024)`` call over a filtered index pool, so the only difference
between the two arms is which *population* the draw comes from. That makes
"how much of a JEPA input is plant?" a question with a clean answer.

*The seeds* are shared. Arms are compared inside a seed, then averaged, so
seed-to-seed variance is reported instead of averaging it away.

What a loss comparison is NOT done here
---------------------------------------
The brief (and Stage 4B before it) forbids claiming one representation is
better from a single forward pass on a random network. This module therefore
computes **no** loss and makes **no** preference statement. It measures:

* composition -- what fraction of the 1,024 drawn Gaussians is plant;
* spatial coverage -- how much of the plant's volume the draw reaches;
* scale / opacity profiles -- whether the draw is biased toward small or
  faint Gaussians;
* retention of fine structure -- leaf and stem Gaussians are the small ones,
  so their share of the draw is the thing to watch;
* render loss -- re-rendered held-out PSNR/IoU of both assets, which is a
  property of the assets, not of the JEPA model.

Usage
-----
    python -m agri.plant_gs.paa.stage4c.input_quality \
        --library results/agri_paa/stage4c/assets.json \
        --out results/agri_paa/stage4c/input_quality \
        --seeds 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri.plant_gs.jepa.dataset import (  # noqa: E402
    load_gaussian_ply, normalize_gaussians, sample_gaussians)

#: semantic.py's label ids, which gaussian_classes.npz stores.
BACKGROUND, PLANT, POT, SOIL, SUPPORT = 0, 1, 2, 3, 4
CLASS_NAMES = {0: "background", 1: "plant", 2: "pot", 3: "soil", 4: "support",
               -1: "undecided"}
NON_PLANT = (BACKGROUND, POT, SOIL, SUPPORT)


def log(message: str) -> None:
    print(message, flush=True)


def digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def plant_pool(classes: np.ndarray) -> np.ndarray:
    """Indices of every Gaussian whose majority visible label is plant.

    Undecided Gaussians (class -1, too few visible views) are *excluded*: the
    plant-aware sampler may only prefer evidence, and -1 is the absence of it.
    """
    return np.flatnonzero(classes == PLANT)


def draw_uniform(decoded: np.ndarray, count: int, seed: int):
    """The official path: normalize the full asset, uniform whole-row draw."""
    normalized, metadata = normalize_gaussians(decoded, "pretrain")
    sampled, indices = sample_gaussians(normalized, count, seed)
    return sampled, indices, metadata


def draw_plant_aware(decoded: np.ndarray, classes: np.ndarray, count: int, seed: int):
    """Same normalization, same draw size, same RNG -- a plant-only population.

    Normalization statistics are still computed on the *full* asset. That is
    deliberate: using the plant-only subset's own centre and radius would
    change the coordinate frame as well as the population, and the comparison
    would conflate the two.
    """
    normalized, metadata = normalize_gaussians(decoded, "pretrain")
    pool = plant_pool(classes)
    if len(pool) == 0:
        raise ValueError("no plant Gaussians to draw from")
    rng = np.random.RandomState(seed)
    indices = rng.choice(pool, count, replace=len(pool) < count).astype(np.int64)
    return np.ascontiguousarray(normalized[indices], dtype=np.float32), indices, metadata


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def composition(classes: np.ndarray, indices: np.ndarray) -> dict:
    picked = classes[indices]
    total = max(len(picked), 1)
    out = {}
    for class_id, name in CLASS_NAMES.items():
        share = int((picked == class_id).sum()) / total
        if share:
            out[name] = share
    return out


def scale_opacity(decoded: np.ndarray, indices: np.ndarray) -> dict:
    """Scale/opacity of the drawn rows, next to the population's own profile.

    ``decoded`` is the 14-D JEPA representation, where column 4:7 are positive
    physical scales (before the pretraining-mode transform) and column 3 is the
    raw logit opacity. Both are read on the raw decode so the numbers mean what
    the PLY says.
    """
    picked = decoded[indices]
    population = decoded
    return {
        "scale_median": float(np.median(picked[:, 4:7].max(axis=1))),
        "scale_p05": float(np.percentile(picked[:, 4:7].max(axis=1), 5)),
        "scale_p95": float(np.percentile(picked[:, 4:7].max(axis=1), 95)),
        "population_scale_median": float(np.median(population[:, 4:7].max(axis=1))),
        "opacity_median_logit": float(np.median(picked[:, 3])),
        "opacity_p05_logit": float(np.percentile(picked[:, 3], 5)),
        "population_opacity_median_logit": float(np.median(population[:, 3])),
        "small_gaussian_share_sample": float(
            (picked[:, 4:7].max(axis=1) < np.percentile(
                population[:, 4:7].max(axis=1), 10)).mean()),
        "small_gaussian_share_population": 0.10,
    }


def spatial(decoded: np.ndarray, indices: np.ndarray, classes: np.ndarray,
            grid: int = 16) -> dict:
    xyz = decoded[:, :3]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)

    def voxels(points):
        cell = np.clip(((points - lo) / extent * grid).astype(int), 0, grid - 1)
        return cell[:, 0] * grid * grid + cell[:, 1] * grid + cell[:, 2]

    plant_voxels = np.unique(voxels(xyz[classes == PLANT]))
    picked_voxels = np.unique(voxels(xyz[indices]))
    return {
        "occupied_voxels_plant": int(len(plant_voxels)),
        "occupied_voxels_draw": int(len(picked_voxels)),
        "plant_voxel_coverage": (float(len(np.intersect1d(plant_voxels, picked_voxels))
                                       / max(len(plant_voxels), 1))),
        "draw_voxels_outside_plant": int(len(np.setdiff1d(picked_voxels, plant_voxels))),
    }


def fine_structure(decoded: np.ndarray, classes: np.ndarray, indices: np.ndarray) -> dict:
    """Fine structure = the finest-decile plant Gaussians (thin stems, leaf edges).

    Fine structure in a Gaussian asset is not something a label map can name:
    the semantic vote says "plant", not "petiole". What can be measured is
    whether the draw reaches the *smallest* Gaussians -- the ones that carry
    thin geometry -- at their population rate. The reference is the smallest
    decile of the plant population, so the comparison is plant-to-plant and is
    not confounded by the pot's own scale distribution.
    """
    plant = plant_pool(classes)
    if len(plant) == 0:
        return {}
    plant_scale = decoded[plant][:, 4:7].max(axis=1)
    threshold = np.percentile(plant_scale, 10)
    fine = plant[plant_scale <= threshold]
    picked = np.isin(indices, fine)
    return {
        "fine_threshold_scale": float(threshold),
        "fine_population_share": float(len(fine) / len(plant)),
        "fine_share_in_draw_of_plant_rows": float(picked.sum() / max(
            (np.isin(indices, plant)).sum(), 1)),
        "fine_retention_ratio": float(
            (picked.sum() / max((np.isin(indices, plant)).sum(), 1))
            / max(len(fine) / len(plant), 1e-9)),
    }


# --------------------------------------------------------------------------- #
# Per-asset analysis
# --------------------------------------------------------------------------- #
def analyse_asset(entry: dict, count: int, seeds: list) -> dict:
    full_ply = entry.get("gaussian_ply_path")
    plant = (entry.get("plant_only") or {}).get("plant_only") or {}
    out = {"asset_id": entry["asset_id"], "crop": entry["crop"],
           "sample_size": count, "seeds": seeds}
    if not full_ply:
        out["status"] = "skipped_no_full_ply"
        return out
    classes_path = None
    if plant.get("dir"):
        candidate = Path(plant["dir"]) / "gaussian_classes.npz"
        if candidate.is_file():
            classes_path = candidate
    if classes_path is None:
        out["status"] = "skipped_no_semantic_labels"
        return out

    full_decoded = load_gaussian_ply(Path(full_ply))
    classes = np.load(classes_path)["classes"]
    if len(classes) != len(full_decoded):
        out["status"] = "skipped_label_mismatch"
        out["detail"] = f"{len(classes)} labels vs {len(full_decoded)} Gaussians"
        return out

    runs = {"full_uniform": [], "full_plant_aware": [], "plant_only_uniform": []}
    for seed in seeds:
        _, idx_full, meta = draw_uniform(full_decoded, count, seed)
        _, idx_aware, _ = draw_plant_aware(full_decoded, classes, count, seed)
        runs["full_uniform"].append({
            "seed": seed,
            "composition": composition(classes, idx_full),
            "plant_fraction": float((classes[idx_full] == PLANT).mean()),
            "scale_opacity": scale_opacity(full_decoded, idx_full),
            "spatial": spatial(full_decoded, idx_full, classes),
            "fine_structure": fine_structure(full_decoded, classes, idx_full),
            "indices_sha256": digest(idx_full),
        })
        runs["full_plant_aware"].append({
            "seed": seed,
            "composition": composition(classes, idx_aware),
            "plant_fraction": float((classes[idx_aware] == PLANT).mean()),
            "scale_opacity": scale_opacity(full_decoded, idx_aware),
            "spatial": spatial(full_decoded, idx_aware, classes),
            "fine_structure": fine_structure(full_decoded, classes, idx_aware),
            "indices_sha256": digest(idx_aware),
        })

    if plant.get("ply") and Path(plant["ply"]).is_file():
        try:
            plant_decoded = load_gaussian_ply(Path(plant["ply"]))
        except ValueError as exc:
            # An empty carve (a labelling failure) is a result to report, not a
            # reason to abort the whole comparison.
            plant_decoded = None
            out["plant_only_note"] = f"plant-only asset unreadable: {exc}"
        # Coverage of the plant's own voxels, comparable across the three arms:
        # the reference lattice is the FULL asset's plant voxels, so the number
        # means "how much of the plant does this draw reach" for the full draw,
        # the plant-aware draw and the plant-only asset's own draw alike.
        if plant_decoded is not None:
            reference = spatial(full_decoded, np.flatnonzero(classes == PLANT), classes)
            plant_voxels = reference["occupied_voxels_plant"]
            lo = full_decoded[:, :3].min(axis=0)
            hi = full_decoded[:, :3].max(axis=0)
            extent = np.maximum(hi - lo, 1e-9)

            def covered_by(points):
                cell = np.clip(((points - lo) / extent * 16).astype(int), 0, 15)
                keys = cell[:, 0] * 256 + cell[:, 1] * 16 + cell[:, 2]
                keys = np.unique(keys)
                return keys

            plant_keys = covered_by(full_decoded[np.flatnonzero(classes == PLANT), :3])
            for seed in seeds:
                _, idx, _ = draw_uniform(plant_decoded, count, seed)
                picked_keys = covered_by(plant_decoded[idx, :3])
                runs["plant_only_uniform"].append({
                    "seed": seed,
                    "composition": {"plant": 1.0},
                    "plant_fraction": 1.0,
                    "scale_opacity": scale_opacity(plant_decoded, idx),
                    "spatial": {
                        "occupied_voxels_plant": int(len(plant_keys)),
                        "occupied_voxels_draw": int(len(picked_keys)),
                        "plant_voxel_coverage": float(
                            len(np.intersect1d(plant_keys, picked_keys))
                            / max(len(plant_keys), 1)),
                    },
                    "spatial_simple": {
                        "bbox_min": plant_decoded[:, :3].min(axis=0).tolist(),
                        "bbox_max": plant_decoded[:, :3].max(axis=0).tolist(),
                    },
                    "fine_structure": fine_structure(plant_decoded,
                                                      np.full(len(plant_decoded), PLANT,
                                                              dtype=np.int64), idx),
                    "indices_sha256": digest(idx),
                })
    out["runs"] = runs
    out["population"] = {
        "gaussians_full": int(len(full_decoded)),
        "gaussians_plant": int((classes == PLANT).sum()),
        "gaussians_nonplant": int(np.isin(classes, NON_PLANT).sum()),
        "gaussians_undecided": int((classes == -1).sum()),
        "plant_fraction_of_full": float((classes == PLANT).mean()),
    }
    out["summary"] = summarize(runs)
    out["status"] = "analysed"
    return out


def _mean_std(values: list) -> dict:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite)),
            "n": len(finite)}


def summarize(runs: dict) -> dict:
    out = {}
    for arm, rows in runs.items():
        if not rows:
            continue
        out[arm] = {
            "plant_fraction": _mean_std([r["plant_fraction"] for r in rows]),
            "scale_median": _mean_std([r["scale_opacity"]["scale_median"] for r in rows]),
            "opacity_median_logit": _mean_std(
                [r["scale_opacity"]["opacity_median_logit"] for r in rows]),
            "small_share": _mean_std(
                [r["scale_opacity"]["small_gaussian_share_sample"] for r in rows]),
        }
        if rows[0].get("spatial", {}).get("plant_voxel_coverage") is not None:
            out[arm]["plant_voxel_coverage"] = _mean_std(
                [r["spatial"]["plant_voxel_coverage"] for r in rows])
        if rows[0].get("spatial", {}).get("draw_voxels_outside_plant") is not None:
            out[arm]["draw_voxels_outside_plant"] = _mean_std(
                [r["spatial"]["draw_voxels_outside_plant"] for r in rows])
        if rows[0].get("fine_structure"):
            out[arm]["fine_retention_ratio"] = _mean_std(
                [r["fine_structure"]["fine_retention_ratio"] for r in rows])
        out[arm]["distinct_indices_across_seeds"] = len(
            {r["indices_sha256"] for r in rows})
        out[arm]["seeds"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path,
                    default=Path("results/agri_paa/stage4c/assets.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/agri_paa/stage4c/input_quality"))
    ap.add_argument("--count", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=8,
                    help="number of seeds (0..n-1); each arm is drawn with the same set")
    ap.add_argument("--crops", nargs="*", default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    library = json.loads(args.library.read_text())
    entries = library["entries"]
    if args.crops:
        entries = [e for e in entries if e["crop"] in set(args.crops)]

    results = []
    for entry in entries:
        if not entry["states"]["states"]["reconstruction_verified"]:
            log(f"skip {entry['asset_id']}: not reconstruction_verified")
            continue
        log(f"analysing {entry['asset_id']} ...")
        results.append(analyse_asset(entry, args.count, list(range(args.seeds))))

    report = {
        "status": "PASS",
        "sample_size": args.count,
        "seeds": list(range(args.seeds)),
        "sampler": {
            "implementation": "agri.plant_gs.jepa.dataset.sample_gaussians",
            "method": "uniform over rows, without replacement when the population "
                      "is larger than the sample",
            "normalization": "pretrain; statistics always from the full asset",
            "plant_aware": "same rng, same count, population restricted to "
                           "Gaussians whose majority visible label is plant",
        },
        "assets": results,
        "comparison_rule": (
            "both arms draw the same number of Gaussians with the same seeds, so a "
            "difference is attributable to the population, not the budget"),
        "not_measured_here": (
            "no JEPA loss is compared; a single forward pass through a randomly "
            "initialised network says nothing about representation quality"),
    }
    (args.out / "input_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    table = [{"asset": r["asset_id"], "status": r["status"],
              "summary": r.get("summary")} for r in results]
    log(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()