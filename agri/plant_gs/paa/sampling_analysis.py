#!/usr/bin/env python
"""Analyse what Gaussian-JEPA's input sampling actually picks up.

Why this matters
----------------
Gaussian-JEPA reads a Gaussian asset, normalises it, and draws a fixed budget
of Gaussians per sample. The official sampler is **uniform over rows**: every
Gaussian is equally likely, whatever its size, opacity or semantic role. For a
plant asset that is not a neutral choice. A pot rebuilt with many small,
low-opacity Gaussians and a canopy rebuilt with a few large ones will hand the
model very different views of the plant.

This module measures that, on the Full-specimen and Plant-only assets:

* the **semantic composition** of a 1,024-Gaussian sample (plant / pot / soil /
  support / undecided), against the composition of the population it is drawn
  from -- a uniform sample of a skewed population is skewed;
* how often **tiny** Gaussians are drawn, relative to their share of the
  population, and with what scale profile;
* the **spatial distribution** of the drawn points, as a 3D scatter and as
  occupancy statistics;
* **uniform vs a structure-balanced sampler**, so the cost of the current
  choice is a number rather than an opinion.

The class labels come from ``plant_only.py``'s visibility-weighted vote, so they
are the same candidate labels the plant-only asset was built from. They are not
ground truth; this is an analysis of the sampling, not of the segmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASS_NAMES = {0: "background", 1: "plant", 2: "pot", 3: "soil", 4: "support",
               -1: "undecided"}


def log(message: str) -> None:
    print(message, flush=True)


def read_ply(path: Path):
    """Read a 3DGS PLY into a dict of arrays."""
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"].data
    out = {}
    for name in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                 "f_dc_0", "f_dc_1", "f_dc_2"):
        if name in vertex.dtype.names:
            out[name] = np.asarray(vertex[name], dtype=np.float64)
    xyz = np.stack([out["x"], out["y"], out["z"]], axis=1)
    scales = np.stack([out["scale_0"], out["scale_1"], out["scale_2"]], axis=1)
    return {"xyz": xyz, "opacity": out["opacity"], "scale_log": scales,
            "scale": np.exp(scales), "sh_dc": np.stack(
                [out["f_dc_0"], out["f_dc_1"], out["f_dc_2"]], axis=1),
            "count": len(vertex)}


def sample_uniform(rng, n_total: int, count: int) -> np.ndarray:
    """The official sampler: uniform over rows, no replacement when possible."""
    return rng.choice(n_total, count, replace=n_total < count)


def sample_balanced(rng, xyz: np.ndarray, count: int, grid: int = 32):
    """Structure-balanced: one draw per occupied voxel cell, then fill.

    Uniform sampling over rows is uniform over *Gaussians*, not over *space*.
    Where the reconstruction is dense, a uniform draw returns many points from
    a small volume; where it is sparse, it returns few from a large one. This
    sampler instead gives every occupied cell of a ``grid``^3 lattice at most
    one point, which equalises spatial coverage, and fills the remainder from
    the Gaussians not already chosen if the grid cannot supply enough points.

    The fill excludes the cells' representatives. Without that, a fill draw can
    land on a Gaussian already picked as its cell's representative and the
    returned index set silently shrinks.
    """
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)
    cell = np.clip(((xyz - lo) / extent * grid).astype(int), 0, grid - 1)
    keys = (cell[:, 0] * grid + cell[:, 1]) * grid + cell[:, 2]

    # One representative per cell, taken from a stable sort so the choice is
    # the lowest-index member and therefore reproducible.
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    representatives = order[starts]
    chosen = rng.permutation(representatives)[:count]

    if len(chosen) < count:
        available = np.ones(len(xyz), dtype=bool)
        available[chosen] = False
        pool = np.flatnonzero(available)
        take = min(count - len(chosen), len(pool))
        chosen = np.concatenate([chosen, rng.choice(pool, take, replace=False)])
    return np.ascontiguousarray(chosen[:count], dtype=np.int64)


def composition(classes: np.ndarray, indices: np.ndarray) -> dict:
    picked = classes[indices]
    total = len(picked)
    out = {}
    for cid, name in CLASS_NAMES.items():
        count = int((picked == cid).sum())
        if count:
            out[name] = {"count": count, "fraction": count / total}
    return out


def scale_profile(scale: np.ndarray, indices: np.ndarray) -> dict:
    picked = scale[indices].max(axis=1)
    population = scale.max(axis=1)
    small_threshold = 1e-6
    small_pop = float((population < small_threshold).mean())
    small_pick = float((picked < small_threshold).mean())
    return {
        "population": {
            "median_max_scale": float(np.median(population)),
            "p05": float(np.percentile(population, 5)),
            "p95": float(np.percentile(population, 95)),
            "max": float(population.max()),
            "fraction_below_1e-6": small_pop,
        },
        "sample": {
            "median_max_scale": float(np.median(picked)),
            "p05": float(np.percentile(picked, 5)),
            "p95": float(np.percentile(picked, 95)),
            "fraction_below_1e-6": small_pick,
        },
        "small_gaussian_oversampling": (
            float(small_pick / small_pop) if small_pop > 0 else None),
        "note": ("oversampling factor is (share of the sample below 1e-6) divided by "
                 "(share of the population below 1e-6); 1.0 means the sample is "
                 "unbiased with respect to Gaussian size"),
    }


def spatial_profile(xyz: np.ndarray, indices: np.ndarray) -> dict:
    picked = xyz[indices]
    centre = xyz.mean(axis=0)
    radius_all = np.linalg.norm(xyz - centre, axis=1)
    radius_pick = np.linalg.norm(picked - centre, axis=1)
    # occupancy of a 16^3 lattice over the population's bounding box
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)

    def occupied(points):
        cell = np.clip(((points - lo) / extent * 16).astype(int), 0, 15)
        keys = cell[:, 0] * 256 + cell[:, 1] * 16 + cell[:, 2]
        return len(np.unique(keys))

    occ_all = occupied(xyz)
    occ_pick = occupied(picked)
    return {
        "population_bbox_min": lo.tolist(), "population_bbox_max": hi.tolist(),
        "sample_bbox_min": picked.min(axis=0).tolist(),
        "sample_bbox_max": picked.max(axis=0).tolist(),
        "radius_population_median": float(np.median(radius_all)),
        "radius_sample_median": float(np.median(radius_pick)),
        "radius_sample_p05": float(np.percentile(radius_pick, 5)),
        "radius_sample_p95": float(np.percentile(radius_pick, 95)),
        "occupied_voxels_16cube_population": int(occ_all),
        "occupied_voxels_16cube_sample": int(occ_pick),
        "voxel_coverage_ratio": float(occ_pick / max(occ_all, 1)),
    }


def nearest_neighbour_stats(xyz: np.ndarray, indices: np.ndarray) -> dict:
    from scipy.spatial import cKDTree
    picked = xyz[indices]
    tree = cKDTree(picked)
    distances, _ = tree.query(picked, k=2)
    nn = distances[:, 1]
    return {"nn_distance_median": float(np.median(nn)),
            "nn_distance_p05": float(np.percentile(nn, 5)),
            "nn_distance_p95": float(np.percentile(nn, 95))}


def analyse(ply: Path, classes: np.ndarray | None, count: int, seed: int,
            label: str) -> dict:
    cloud = read_ply(ply)
    n = cloud["count"]
    if classes is None or len(classes) != n:
        classes = np.full(n, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)

    uniform = sample_uniform(rng, n, count)
    balanced = sample_balanced(rng, cloud["xyz"], count)

    result = {
        "label": label,
        "ply": str(ply.resolve()),
        "gaussians": int(n),
        "sample_size": int(count),
        "seed": int(seed),
        "uniform": {
            "composition": composition(classes, uniform),
            "scale": scale_profile(cloud["scale"], uniform),
            "spatial": spatial_profile(cloud["xyz"], uniform),
            "nearest_neighbour": nearest_neighbour_stats(cloud["xyz"], uniform),
            "indices_sha256": _digest(uniform),
        },
        "balanced": {
            "composition": composition(classes, balanced),
            "scale": scale_profile(cloud["scale"], balanced),
            "spatial": spatial_profile(cloud["xyz"], balanced),
            "nearest_neighbour": nearest_neighbour_stats(cloud["xyz"], balanced),
            "indices_sha256": _digest(balanced),
        },
    }
    result["comparison"] = {
        "uniform_voxel_coverage": result["uniform"]["spatial"]["voxel_coverage_ratio"],
        "balanced_voxel_coverage": result["balanced"]["spatial"]["voxel_coverage_ratio"],
        "uniform_small_oversampling": result["uniform"]["scale"]["small_gaussian_oversampling"],
        "balanced_small_oversampling": result["balanced"]["scale"]["small_gaussian_oversampling"],
    }
    return result, cloud, uniform, balanced


def _digest(indices: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()


def plot_scatter(cloud, uniform, balanced, classes, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz = cloud["xyz"]
    centre = xyz.mean(axis=0)
    fig = plt.figure(figsize=(15, 5))
    for column, (name, indices) in enumerate([("uniform (official)", uniform),
                                               ("structure-balanced", balanced)]):
        ax = fig.add_subplot(1, 3, column + 1, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.4, c="#c9c9c9", alpha=0.25)
        picked = xyz[indices]
        if classes is not None:
            colours = np.array([{1: "#28c846", 2: "#3c5ae6", 3: "#8c5a28",
                                 4: "#ebd73c"}.get(int(c), "#888888")
                                for c in classes[indices]])
        else:
            colours = "#d04040"
        ax.scatter(picked[:, 0], picked[:, 1], picked[:, 2], s=8, c=colours,
                   alpha=0.9, depthshade=False)
        ax.set_title(f"{title}\n{name} (n={len(indices)})", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect((1, 1, 1))
    ax = fig.add_subplot(1, 3, 3)
    ax.hist(np.linalg.norm(xyz - centre, axis=1), bins=60, color="#c9c9c9",
            label="population", density=True)
    ax.hist(np.linalg.norm(xyz[uniform] - centre, axis=1), bins=60, histtype="step",
            color="#d04040", lw=1.6, label="uniform", density=True)
    ax.hist(np.linalg.norm(xyz[balanced] - centre, axis=1), bins=60, histtype="step",
            color="#1f77b4", lw=1.6, label="balanced", density=True)
    ax.set_xlabel("radius from centroid"); ax.set_ylabel("density")
    ax.set_title("radial distribution", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-ply", type=Path, required=True)
    ap.add_argument("--plant-ply", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True,
                    help="gaussian_classes.npz from plant_only.py (full asset)")
    ap.add_argument("--plant-ply-is-plant", action="store_true", default=True,
                    help="label every Gaussian of the plant-only asset as plant; it "
                         "was selected on that basis, so its composition is 100%% "
                         "plant by construction and the interesting question is "
                         "what the FULL asset's sampler draws")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8,
                    help="repeat the sampling this many times for stable statistics")
    args = ap.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.classes)
    classes = payload["classes"]

    full, cloud_full, uniform_full, balanced_full = analyse(
        args.full_ply, classes, args.count, args.seed, "full_specimen")
    plant_classes = (np.full(read_ply(args.plant_ply)["count"], 1, dtype=np.int64)
                     if args.plant_ply_is_plant else None)
    plant, cloud_plant, uniform_plant, balanced_plant = analyse(
        args.plant_ply, plant_classes, args.count, args.seed, "plant_only")

    # Repeat over seeds so the composition numbers are not one draw's luck.
    repeat = {"uniform": {"plant": [], "pot": [], "soil": [], "support": []},
               "balanced": {"plant": [], "pot": [], "soil": [], "support": []}}
    for s in range(args.seeds):
        rng = np.random.default_rng(1000 + s)
        for strategy, picker in (("uniform",
                                  lambda r: sample_uniform(r, cloud_full["count"],
                                                           args.count)),
                                 ("balanced",
                                  lambda r: sample_balanced(r, cloud_full["xyz"],
                                                            args.count))):
            indices = picker(rng)
            comp = composition(classes, indices)
            for name in repeat[strategy]:
                repeat[strategy][name].append(
                    comp.get(name, {}).get("fraction", 0.0))
    full["seed_sweep"] = {
        "seeds": args.seeds,
        "uniform_mean_fraction": {k: float(np.mean(v)) for k, v in
                                  repeat["uniform"].items()},
        "balanced_mean_fraction": {k: float(np.mean(v)) for k, v in
                                   repeat["balanced"].items()},
        "uniform_std": {k: float(np.std(v)) for k, v in repeat["uniform"].items()},
    }

    report = {
        "status": "PASS",
        "sample_size": args.count,
        "seed": args.seed,
        "seed_sweep_seeds": args.seeds,
        "full_specimen": full,
        "plant_only": plant,
        "population_class_composition": {
            name: int((classes == cid).sum()) / len(classes)
            for cid, name in CLASS_NAMES.items() if (classes == cid).any()},
        "seconds": round(time.time() - started, 1),
        "scope": ("class labels are the candidate semantic labels from "
                  "plant_only.py, not ground truth; this module measures the "
                  "sampling behaviour, not segmentation accuracy"),
    }
    (args.out / "sampling_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    plot_scatter(cloud_full, uniform_full, balanced_full, classes,
                 args.out / "sampling_full_specimen.png", "Full-specimen")
    plot_scatter(cloud_plant, uniform_plant, balanced_plant, None,
                 args.out / "sampling_plant_only.png", "Plant-only")

    log(json.dumps({
        "population_class_composition": report["population_class_composition"],
        "full_uniform": full["uniform"]["composition"],
        "full_balanced": full["balanced"]["composition"],
        "plant_uniform": plant["uniform"]["composition"],
        "comparison": full["comparison"],
        "seed_sweep": full["seed_sweep"],
    }, indent=2))


if __name__ == "__main__":
    main()
