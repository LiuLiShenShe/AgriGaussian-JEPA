#!/usr/bin/env python
"""Why the Stage 4C candidate labels fail on Rice, Cotton and Cabbage.

Stage 4C measured the damage (a 0-Gaussian carve, 54% of white tissue labelled
support, a 0.129 plant fraction). This module narrows the cause without
pretending to measure accuracy.

What can and cannot be claimed
------------------------------
There is no human-confirmed label for these frames, so no accuracy, precision
or recall figure can be reported. What *can* be measured is:

* **candidate disagreement** -- how far two independent labellings put the
  plant boundary apart (A vs B, and the intersection C vs both);
* **stability across an orbit** -- the plant is the same plant in all 102
  frames, so a method whose plant area swings from 45% to 0% of the specimen is
  unstable regardless of which answer is right;
* **white-tissue exposure** -- of the bright, desaturated specimen pixels, how
  many does each method refuse to call plant, and how many does it call
  support;
* **thin-structure exposure** -- how much of a method's own plant mask is
  morphologically thin, and how much of that the intersection rule keeps.

A method that is stable, keeps white tissue and keeps thin structure is a
better *candidate*. That is weaker than "more accurate", and it is stated as
such everywhere.

The five arms the brief asks about
----------------------------------
A. SegFormer          measured here.
B. SAM + CLIP         measured here.
C. intersection       measured here.
D. SAM 2 / Grounded SAM 2   **not run**, with the exact dependency/weight state
                            recorded. No accuracy is implied either way.
E. human prompts + multi-view constraints   **not run**; there is no human
                            prompt, and inventing one would be exactly the
                            substitution this project forbids. The orbit
                            stability figure above is the closest available
                            proxy and is labelled as a proxy.

Nothing here re-carves, rewrites a PLY, or changes an asset.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
Image.MAX_IMAGE_PIXELS = None

SCHEMA_VERSION = "paa-stage4d-diagnosis-1"

#: semantic.py class ids, restated to keep this module importable in the venv
#: that has no gsplat.
BACKGROUND, PLANT, POT, SOIL, SUPPORT = 0, 1, 2, 3, 4

WHITE_VALUE_MIN = 0.85
WHITE_SATURATION_MAX = 0.15
THIN_KERNEL = 7


def log(message: str) -> None:
    print(message, flush=True)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def binary_opening(mask: np.ndarray, kernel: int = THIN_KERNEL) -> np.ndarray:
    try:
        import cv2
        k = np.ones((kernel, kernel), np.uint8)
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                k).astype(bool)
    except Exception:
        from scipy.ndimage import binary_dilation, binary_erosion
        structure = np.ones((kernel, kernel), dtype=bool)
        return binary_dilation(binary_erosion(mask, structure), structure)


def white_mask(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.where(maximum > 0,
                          (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
    return (maximum >= WHITE_VALUE_MIN) & (saturation <= WHITE_SATURATION_MAX)


# --------------------------------------------------------------------------- #
# One frame
# --------------------------------------------------------------------------- #
def measure_frame(image: np.ndarray, specimen: np.ndarray | None,
                  segformer: np.ndarray | None,
                  sam_clip: np.ndarray | None) -> dict:
    a = (segformer == PLANT) if segformer is not None else None
    b = (sam_clip == PLANT) if sam_clip is not None else None
    c = (a & b) if (a is not None and b is not None) else None
    union = (a | b) if (a is not None and b is not None) else None

    specimen_pixels = int(specimen.sum()) if specimen is not None else None
    row = {
        "specimen_pixels": specimen_pixels,
        "plant_pixels": {},
        "plant_fraction_of_specimen": {},
        "thin_fraction_of_own_plant": {},
        "white_specimen_pixels": 0,
        "white_not_plant_fraction": {},
        "white_called_support_fraction": {},
    }
    for name, mask in (("A_segformer", a), ("B_sam_clip", b), ("C_intersection", c)):
        if mask is None:
            continue
        pixels = int(mask.sum())
        row["plant_pixels"][name] = pixels
        if specimen_pixels:
            row["plant_fraction_of_specimen"][name] = pixels / specimen_pixels
        if pixels:
            thin = mask & ~binary_opening(mask)
            row["thin_fraction_of_own_plant"][name] = float(thin.sum() / pixels)

    if specimen is not None:
        white = white_mask(image) & specimen
        total = int(white.sum())
        row["white_specimen_pixels"] = total
        if total:
            for name, mask in (("A_segformer", a), ("B_sam_clip", b),
                               ("C_intersection", c)):
                if mask is not None:
                    row["white_not_plant_fraction"][name] = float(
                        (white & ~mask).sum() / total)
            if segformer is not None:
                row["white_called_support_fraction"]["A_segformer"] = float(
                    (white & (segformer == SUPPORT)).sum() / total)
            if sam_clip is not None:
                row["white_called_support_fraction"]["B_sam_clip"] = float(
                    (white & (sam_clip == SUPPORT)).sum() / total)

    if a is not None and b is not None:
        inter = int((a & b).sum())
        uni = int((a | b).sum())
        row["A_B_plant_iou"] = (inter / uni) if uni else None
        row["intersection_share_of_union"] = (inter / uni) if uni else None
        row["A_coverage_of_B"] = (int(a.sum()) / int(b.sum())) if b.sum() else None
        row["B_coverage_of_A"] = (int(b.sum()) / int(a.sum())) if a.sum() else None
    return row


# --------------------------------------------------------------------------- #
# One workspace
# --------------------------------------------------------------------------- #
def _labels(semantic: Path, stem: str) -> tuple:
    seg = semantic / "labels_segformer" / f"{stem}.png"
    sam = semantic / "labels_sam_clip" / f"{stem}.png"
    return (np.asarray(Image.open(seg)) if seg.is_file() else None,
            np.asarray(Image.open(sam)) if sam.is_file() else None)


def _mask(frame: dict) -> np.ndarray | None:
    path = frame.get("mask")
    if not path or not Path(path).is_file():
        return None
    return np.asarray(Image.open(path)) > 0


def _mean(values: list):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _spread(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    mean = float(np.mean(values))
    return {"n": len(values), "mean": mean, "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "cv": (float(np.std(values) / mean) if mean else None)}


def diagnose_workspace(workspace: Path, every: int = 1) -> dict:
    workspace = Path(workspace)
    dataset, semantic = workspace / "dataset", workspace / "semantic"
    frames = json.loads((dataset / "frames.json").read_text())
    if every > 1:
        frames = frames[::every]

    per_frame = []
    for frame in frames:
        stem = Path(frame["name"]).stem
        image_path = frame.get("image")
        if not image_path or not Path(image_path).is_file():
            continue
        segformer, sam_clip = _labels(semantic, stem)
        if segformer is None and sam_clip is None:
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"))
        row = measure_frame(image, _mask(frame), segformer, sam_clip)
        row["stem"] = stem
        per_frame.append(row)

    summary = {}
    for arm in ("A_segformer", "B_sam_clip", "C_intersection"):
        fractions = [f["plant_fraction_of_specimen"].get(arm) for f in per_frame]
        summary[f"plant_fraction_of_specimen_{arm}"] = _spread(fractions)
        empty = sum(1 for v in fractions
                    if v is not None and v < 0.01)
        summary[f"near_empty_frames_{arm}"] = empty
        summary[f"thin_fraction_{arm}"] = _mean(
            [f["thin_fraction_of_own_plant"].get(arm) for f in per_frame])
        summary[f"white_not_plant_{arm}"] = _mean(
            [f["white_not_plant_fraction"].get(arm) for f in per_frame])
        summary[f"white_called_support_{arm}"] = _mean(
            [f["white_called_support_fraction"].get(arm) for f in per_frame])
    summary["A_B_plant_iou"] = _spread([f.get("A_B_plant_iou") for f in per_frame])
    summary["intersection_share_of_union"] = _spread(
        [f.get("intersection_share_of_union") for f in per_frame])
    summary["frames_measured"] = len(per_frame)

    # A method's plant area over a full 360-degree orbit should not swing by an
    # order of magnitude: the plant does not disappear. A high coefficient of
    # variation is a label-stability failure, whatever the ground truth is.
    stability = {arm: summary[f"plant_fraction_of_specimen_{arm}"].get("cv")
                 for arm in ("A_segformer", "B_sam_clip", "C_intersection")}

    arms = {
        "A_segformer": {"status": "measured",
                        "what": "nvidia/segformer-b5-finetuned-ade-640-640"},
        "B_sam_clip": {"status": "measured",
                       "what": "SAM vit_h proposals + CLIP ViT-B/32 zero-shot "
                               "text scoring"},
        "C_intersection": {"status": "measured",
                           "what": "plant where both A and B agree"},
        "D_sam2_grounded": {
            "status": "not_run",
            "reason": ("no SAM 2 / Grounded SAM 2 weights or code are installed "
                       "on this host (import sam2 fails; the SAM vit_h "
                       "checkpoint the Stage 4C backend referenced is also no "
                       "longer present under /data/models). No accuracy is "
                       "implied in either direction."),
        },
        "E_human_prompts": {
            "status": "not_run",
            "reason": ("requires a human to place prompts; no human is "
                       "available, and simulating prompts with a model would "
                       "repeat the substitution this project forbids. The "
                       "orbit-stability figure is the closest available proxy "
                       "and is labelled a proxy, not arm E."),
        },
    }
    return {
        "workspace": str(workspace.resolve()),
        "diagnosed_utc": _utc(),
        "schema_version": SCHEMA_VERSION,
        "frames": per_frame,
        "summary": summary,
        "orbit_stability_cv": stability,
        "arms": arms,
        "claims": {
            "candidate_disagreement_only": True,
            "no_accuracy_claim": ("no human-confirmed label was used; every "
                                  "number compares model outputs with each "
                                  "other or measures stability"),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path,
                    default=Path("data/plant_gs/PAA/Rice_1"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    result = diagnose_workspace(args.workspace, every=args.every)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({"summary": result["summary"],
                          "orbit_stability_cv": result["orbit_stability_cv"],
                          "arms": result["arms"]}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2,
                                       ensure_ascii=False) + "\n")
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
