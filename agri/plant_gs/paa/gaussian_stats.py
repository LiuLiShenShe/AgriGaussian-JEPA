#!/usr/bin/env python
"""Statistical profile of an exported 3DGS PLY, for reconstruction QA.

Reports the distributions the brief asks for (position, scale, opacity), plus
three things a single summary number would hide:

* **Floater / tail structure.** Gaussians far from the specimen centroid, or
  with scales far above the median spacing, are the usual visible artefact.
  This reports radial and scale percentiles rather than a mean, because the
  mean of a heavy tail says nothing useful.

* **The activation convention actually in the file.** `opacity` must be a raw
  logit and `scale_*` a raw log. Those are checked, not assumed, because a file
  written with sigmoid(opacity) or exp(scale) trains and loads perfectly well
  while producing a wrong model downstream.

* **Non-finite values.** Any NaN/Inf anywhere is reported with the field names,
  since a single non-finite Gaussian can poison a whole render.

    python agri/plant_gs/paa/gaussian_stats.py plant.ply --json stats.json \
        [--mask-consistency cleanup_scores.npz] [--orange-preview out.jpg]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814


def percentile_row(values: np.ndarray) -> dict:
    q = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    keys = ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"]
    return {k: float(v) for k, v in zip(keys, q)}


def analyse(path: Path, mask_consistency: Path | None = None) -> dict:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"{path} has no vertex element")
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names)

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    opacity_raw = np.asarray(vertex["opacity"], dtype=np.float64)
    scale_names = sorted((n for n in names if n.startswith("scale_")),
                         key=lambda s: int(s.split("_")[-1]))
    rot_names = sorted((n for n in names if n.startswith("rot_")),
                       key=lambda s: int(s.split("_")[-1]))
    dc_names = sorted((n for n in names if n.startswith("f_dc_")),
                      key=lambda s: int(s.split("_")[-1]))
    f_rest_names = sorted((n for n in names if n.startswith("f_rest_")),
                          key=lambda s: int(s.split("_")[-1]))

    scales_raw = np.column_stack([vertex[n] for n in scale_names]).astype(np.float64)
    rots = np.column_stack([vertex[n] for n in rot_names]).astype(np.float64)
    f_dc = np.column_stack([vertex[n] for n in dc_names]).astype(np.float64)

    # ---- finite-value audit over every float property in the file ----
    nonfinite = {}
    for name in vertex.data.dtype.names:
        column = np.asarray(vertex[name])
        if column.dtype.kind != "f":
            continue
        bad = int((~np.isfinite(column)).sum())
        if bad:
            nonfinite[name] = bad

    opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity_raw, -60, 60)))
    scales = np.exp(np.clip(scales_raw, -60, 60))
    quat_norm = np.linalg.norm(rots, axis=1)
    rgb = np.clip(f_dc * C0 + 0.5, 0.0, 1.0)

    centroid = np.median(xyz, axis=0)
    radial = np.linalg.norm(xyz - centroid, axis=1)

    out = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "gaussians": int(len(xyz)),
        "properties": len(vertex.data.dtype.names),
        "attribute_sets": {
            "xyz": all(f"x{i}" if False else n in names for n in ("x", "y", "z")),
            "scale": len(scale_names), "rotation": len(rot_names),
            "f_dc": len(dc_names), "f_rest": len(f_rest_names),
        },
        "has_all_14_gsjepa_attributes": bool(
            {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"} <= names
            and len(scale_names) == 3 and len(rot_names) == 4),
        "nonfinite_fields": nonfinite,
        "all_finite": not nonfinite,
        "box": {
            "min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist(),
            "extent": (xyz.max(axis=0) - xyz.min(axis=0)).tolist(),
            "centroid_median": centroid.tolist(),
        },
        "radial_distance_from_centroid": percentile_row(radial),
        "scale_linear": {
            "percentiles_all_axes": percentile_row(scales.reshape(-1)),
            "per_axis": {f"scale_{i}": percentile_row(scales[:, i]) for i in range(scales.shape[1])},
            "degenerate_below_1e-6": int((scales < 1e-6).sum()),
            "large_above_0.25": int((scales.max(axis=1) > 0.25).sum()),
        },
        "opacity_activated": {
            "percentiles": percentile_row(opacity),
            "below_0.01": int((opacity < 0.01).sum()),
            "below_0.05": int((opacity < 0.05).sum()),
            "above_0.99": int((opacity > 0.99).sum()),
        },
        "opacity_raw_is_logit": bool(opacity_raw.min() < 0 or opacity_raw.max() > 1),
        "scale_raw_is_log": bool(scales_raw.max() < 0),
        "rotation": {
            "norm": percentile_row(quat_norm),
            "non_unit_beyond_1e-3": int((np.abs(quat_norm - 1.0) > 1e-3).sum()),
        },
        "dc_colour_after_decode": {
            "rgb_linear_percentiles": {
                ch: percentile_row(rgb[:, i]) for i, ch in enumerate("rgb")},
            "fraction_clipped_to_0_or_1": float(
                ((rgb <= 0) | (rgb >= 1)).mean()),
        },
    }

    # ---- floppy-tail sketch: how many Gaussians sit far outside the body ----
    p95_radial = out["radial_distance_from_centroid"]["p95"]
    out["floater_indicators"] = {
        "radial_p99_over_p50": (out["radial_distance_from_centroid"]["p99"]
                                / max(out["radial_distance_from_centroid"]["median"], 1e-9)),
        "beyond_1.5x_p95_radius": int((radial > 1.5 * p95_radial).sum()),
        "low_opacity_fraction": float((opacity < 0.01).mean()),
        "large_scale_fraction": float((scales.max(axis=1) > 0.25).mean()),
    }

    if mask_consistency and Path(mask_consistency).is_file():
        blob = np.load(mask_consistency)
        out["mask_consistency"] = {
            "source": str(mask_consistency),
            "arrays": list(blob.files),
        }
        for key in ("mask_ratio", "observed_views", "opacity", "keep"):
            if key in blob:
                arr = np.asarray(blob[key])
                out["mask_consistency"][f"{key}_percentiles"] = percentile_row(
                    arr.astype(np.float64))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--mask-consistency", type=Path, default=None)
    args = ap.parse_args()
    stats = analyse(args.ply, args.mask_consistency)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"{stats['path']}")
    print(f"  gaussians            {stats['gaussians']:,}  ({stats['properties']} properties)")
    print(f"  all finite           {stats['all_finite']}  non-finite fields: {stats['nonfinite_fields']}")
    print(f"  14 JEPA attributes   {stats['has_all_14_gsjepa_attributes']}")
    print(f"  bbox extent          {[round(v,3) for v in stats['box']['extent']]}")
    r = stats["radial_distance_from_centroid"]
    print(f"  radius from centroid median {r['median']:.4f} p95 {r['p95']:.4f} p99 {r['p99']:.4f} max {r['max']:.4f}")
    s = stats["scale_linear"]
    print(f"  scale (linear)       median {s['percentiles_all_axes']['median']:.2e} "
          f"p95 {s['percentiles_all_axes']['p95']:.2e} max {s['percentiles_all_axes']['max']:.2e}")
    print(f"     degenerate<1e-6 {s['degenerate_below_1e-6']}  large>0.25 {s['large_above_0.25']}")
    o = stats["opacity_activated"]
    print(f"  opacity (activated)  median {o['percentiles']['median']:.4f} "
          f"p5 {o['percentiles']['p5']:.4f} p95 {o['percentiles']['p95']:.4f}")
    print(f"     <0.01 {o['below_0.01']}  <0.05 {o['below_0.05']}  >0.99 {o['above_0.99']}")
    print(f"  conventions          opacity-is-logit {stats['opacity_raw_is_logit']}, "
          f"scale-is-log {stats['scale_raw_is_log']}")
    print(f"  rotation norm        min {stats['rotation']['norm']['min']:.6f} "
          f"max {stats['rotation']['norm']['max']:.6f} non-unit {stats['rotation']['non_unit_beyond_1e-3']}")
    f = stats["floater_indicators"]
    print(f"  floaters             radial p99/p50 {f['radial_p99_over_p50']:.2f}, "
          f"beyond 1.5*p95 {f['beyond_1.5x_p95_radius']}, "
          f"low-opacity {f['low_opacity_fraction']:.4f}, large-scale {f['large_scale_fraction']:.4f}")
    if args.json:
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()