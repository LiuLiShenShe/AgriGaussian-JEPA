#!/usr/bin/env python
"""Per-crop semantic-separation quality, measured against the candidate labels.

What this adds to ``plant_only.py``'s report
--------------------------------------------
``plant_only.py`` already reports, per held-out view, the PSNR and IoU of the
full and carved assets inside the plant region, and the non-plant residue. That
answers "how good is the carve overall". It cannot answer the brief's specific
concerns, which are all about *which pixels* are lost:

* green leaf blades complete or chewed at the edges;
* white plant tissue (cotton bolls, white flowers) thrown away with the support;
* thin things -- stems, petioles, leaf margins, veins -- gone;
* pot, soil and support still painted by the "plant-only" asset.

Each of those is a measurement on a *subset of the plant region*, so this module
splits the region four ways and reports coverage for each:

``thick``    plant pixels that survive a morphological opening -- blades,
             blooms, the bulk of the plant;
``thin``     plant pixels that do **not** survive it -- stems, petioles,
             margins, veins. This is the morphological definition of fine
             structure: a pixel set whose opening removes it.
``green``    plant pixels that are green in hue.
``white``    plant pixels that are bright and desaturated -- the cotton-boll
             case. Where a *candidate* label calls such a pixel ``support``,
             that is exactly the failure mode the brief names.

All of it is measured against the candidate labels, which are not ground truth.
The numbers therefore say "the carve is faithful to the labels", and the label
question is what the annotation benchmark exists to settle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agri" / "plant_gs"))
sys.path.insert(0, str(ROOT / "agri" / "plant_gs" / "paa"))
sys.path.insert(0, str(ROOT / "agri" / "plant_gs" / "foreground"))

from agri.plant_gs.paa.plant_only import (  # noqa: E402
    Captures, load_splats_from_ply, render)
from agri.plant_gs.paa.semantic import PLANT, POT, SOIL, SUPPORT  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def _ply_vertex_count(path: Path) -> int:
    """Vertex count from the PLY header; 0 for an empty carve."""
    with Path(path).open("rb") as handle:
        for _ in range(64):
            line = handle.readline()
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
    return 0


def _opening(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary morphological opening with a square structuring element.

    A disk would be ideal; a square of the same radius is close enough for a
    *classification* of pixels into thin and thick, and it is deterministic and
    dependency-free. The radius is recorded in the output so the definition is
    not implicit.
    """
    from scipy.ndimage import binary_erosion, binary_dilation
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(binary_erosion(mask, structure), structure)


def analyse_view(image: np.ndarray, specimen: np.ndarray, labels: np.ndarray,
                 alpha_full: np.ndarray, alpha_plant: np.ndarray,
                 target: np.ndarray, radius: int) -> dict:
    plant = (labels == PLANT) & specimen
    nonplant = specimen & (labels != PLANT) & (labels != 0)
    opened = _opening(plant, radius)
    thick = plant & opened
    thin = plant & ~opened

    hsv = np.asarray(Image.fromarray(image).convert("HSV")).astype(np.float32)
    hue, saturation, value = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    green = plant & (hue >= 60) & (hue <= 180) & (saturation > 0.20) & (value > 0.12)
    white = plant & (value > 0.60) & (saturation < 0.20)

    def coverage(selection: np.ndarray, alpha: np.ndarray) -> float | None:
        if selection.sum() < 50:
            return None
        return float((alpha[selection] > 0.5).mean())

    # The cotton-boll question: white *specimen* pixels -- a boll, a white
    # flower, a pale stem -- that the candidate labels put outside the plant
    # class. Measuring this on ``plant`` pixels alone would be circular
    # (they are already plant by construction), so the denominator here is
    # every bright, desaturated pixel inside the specimen mask.
    white_specimen = specimen & (value > 0.60) & (saturation < 0.20)
    white_outside_plant = white_specimen & ~plant
    white_labelled_support = int((white_specimen & (labels == SUPPORT)).sum())
    white_labelled_potsoil = int((white_specimen
                                  & ((labels == POT) | (labels == SOIL))).sum())
    white_labelled_other = int((white_specimen
                                & ~np.isin(labels, (PLANT, SUPPORT, POT, SOIL))).sum())

    return {
        "plant_region_pixels": int(plant.sum()),
        "thin_pixels": int(thin.sum()),
        "thin_fraction_of_plant": float(thin.sum() / max(plant.sum(), 1)),
        "green_pixels": int(green.sum()),
        "white_pixels": int(white.sum()),
        "white_share_of_plant": float(white.sum() / max(plant.sum(), 1)),
        "coverage_plant_only_plant": coverage(plant, alpha_plant),
        "coverage_plant_only_thick": coverage(thick, alpha_plant),
        "coverage_plant_only_thin": coverage(thin, alpha_plant),
        "coverage_full_thin": coverage(thin, alpha_full),
        "coverage_plant_only_green": coverage(green, alpha_plant),
        "coverage_plant_only_white": coverage(white, alpha_plant),
        "residue_alpha_on_nonplant": (float(alpha_plant[nonplant].mean())
                                      if nonplant.sum() else None),
        "residue_opaque_fraction_on_nonplant": (
            float((alpha_plant[nonplant] > 0.5).mean()) if nonplant.sum() else None),
        "white_pixels_labelled_support": white_labelled_support,
        "white_pixels_labelled_potsoil": white_labelled_potsoil,
        "white_pixels_labelled_other": white_labelled_other,
        # The failure rate the brief asks about: of every bright, desaturated
        # specimen pixel, how many did the candidate labels remove from the
        # plant class? A white support stake counts in the numerator only if
        # the *human* who reads this panel agrees it is not plant; the labels
        # cannot tell us that, which is the point of the annotation benchmark.
        "white_support_rate": (white_labelled_support / white_specimen.sum()
                               if white_specimen.sum() else None),
        "white_potsoil_rate": (white_labelled_potsoil / white_specimen.sum()
                               if white_specimen.sum() else None),
        "white_outside_plant_rate": (white_outside_plant.sum() / white_specimen.sum()
                                     if white_specimen.sum() else None),
        "white_specimen_pixels": int(white_specimen.sum()),
        "alpha_mean_thin_difference": (
            float(alpha_full[thin].mean() - alpha_plant[thin].mean())
            if thin.sum() >= 50 else None),
    }


def analyse_sample(workspace: Path, labels_dir: Path, plant_dir: Path,
                   out_dir: Path, radius: int, max_views: int = 0) -> dict:
    device = torch.device("cuda")
    dataset = workspace / "dataset"
    data = Captures(dataset, device=str(device), test_every=8, downscale=1)
    full_ply = workspace / "final" / "point_cloud.ply"
    splats, degree, n_full = load_splats_from_ply(full_ply, device)
    n_plant = _ply_vertex_count(plant_dir / "plant_only.ply")
    if n_plant == 0:
        # An empty carve cannot be rendered (gsplat divides by zero). Report the
        # failure with the reason rather than crashing, so a crop whose labels
        # collapsed stays in the record as a failure case.
        summary = {
            "asset": str(workspace.name), "views": 0,
            "source_gaussians": n_full, "plant_gaussians": 0,
            "plant_fraction_of_gaussians": 0.0,
            "status": "empty_carve",
            "reason": ("the plant-only asset has no Gaussians, so no coverage or "
                       "residue can be measured; the semantic labels, not the "
                       "carve, are the failure"),
            "mean": {}, "per_view": [],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "semantic_quality.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        return summary

    kept, _, _ = load_splats_from_ply(plant_dir / "plant_only.ply", device)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    indices = list(data.test_idx)
    if max_views:
        indices = indices[:max_views]
    for index in indices:
        frame = data.frames[index]
        stem = Path(frame["name"]).stem
        label_path = labels_dir / f"{stem}.png"
        if not label_path.is_file():
            continue
        rgb, mask, K, w2c = data.batch(index)
        specimen = mask[0, ..., 0].cpu().numpy() > 0.5
        labels = np.asarray(Image.open(label_path))
        if labels.shape != specimen.shape:
            continue
        _, alpha_full, _ = render(splats, K, w2c, data.width, data.height, degree)
        _, alpha_plant, _ = render(kept, K, w2c, data.width, data.height, degree)
        image = (rgb[0].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        row = {"view": frame["name"],
               **analyse_view(image, specimen, labels,
                              alpha_full[0, ..., 0].detach().cpu().numpy(),
                              alpha_plant[0, ..., 0].detach().cpu().numpy(),
                              rgb[0].detach().clamp(0, 1).cpu().numpy(), radius)}
        rows.append(row)

    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "asset": str(workspace.name),
        "views": len(rows),
        "source_gaussians": n_full,
        "plant_gaussians": n_plant,
        "plant_fraction_of_gaussians": n_plant / max(n_full, 1),
        "status": "PASS",
        "thin_definition": f"plant mask minus its morphological opening with a "
                           f"{2 * radius + 1}x{2 * radius + 1} box",
        "mean": {key: mean(key) for key in (
            "thin_fraction_of_plant", "white_share_of_plant",
            "coverage_plant_only_plant", "coverage_plant_only_thick",
            "coverage_plant_only_thin", "coverage_full_thin",
            "coverage_plant_only_green", "coverage_plant_only_white",
            "residue_alpha_on_nonplant", "residue_opaque_fraction_on_nonplant",
            "white_support_rate", "white_potsoil_rate", "white_outside_plant_rate",
            "alpha_mean_thin_difference")},
        "per_view": rows,
    }
    summary["thin_structure_loss"] = (
        None if summary["mean"].get("coverage_plant_only_thick") is None
        or summary["mean"].get("coverage_plant_only_thin") is None else
        summary["mean"]["coverage_plant_only_thick"]
        - summary["mean"]["coverage_plant_only_thin"])
    summary["note"] = ("coverage and residue are measured against CANDIDATE labels; "
                       "they say the carve is faithful to those labels, not that the "
                       "labels are correct. Fine structure is defined "
                       "morphologically, not organ-by-organ -- no organ labels exist.")
    (out_dir / "semantic_quality.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=None,
                    help="label dir (default <workspace>/semantic/labels_merged)")
    ap.add_argument("--plant-dir", type=Path, default=None,
                    help="plant-only output dir (default <workspace>/plant_only)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--max-views", type=int, default=0)
    args = ap.parse_args()

    workspace = args.workspace
    labels = args.labels or workspace / "semantic" / "labels_merged"
    plant_dir = args.plant_dir or workspace / "plant_only"
    out = args.out or workspace / "semantic_quality"
    started = time.time()
    summary = analyse_sample(workspace, labels, plant_dir, out, args.radius,
                             args.max_views)
    summary["seconds"] = round(time.time() - started, 1)
    (out / "semantic_quality.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps({"asset": summary["asset"], "views": summary["views"],
                    "status": summary.get("status"),
                    "mean": summary["mean"],
                    "thin_structure_loss": summary.get("thin_structure_loss")},
                   indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
