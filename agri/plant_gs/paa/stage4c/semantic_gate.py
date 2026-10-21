#!/usr/bin/env python
"""Turn human-confirmed labels into an auditable semantic verdict per asset.

The gap this closes
-------------------
Stage 4C could only ever write ``semantic_verified = 0``: ``assets.py`` hard
coded ``benchmark.status = "annotation_pending"`` because no human label
existed. That was honest, but it meant the annotation benchmark and the asset
eligibility decision were two disconnected artefacts -- a confirmed label file
would have had no path into ``governance.json``.

This module is that path, and it is deliberately narrow:

    human confirmation -> per-view metrics -> per-asset gate -> per-crop summary
                                                                   |
                                                    assets.py reads the result

Three rules keep it from certifying too much
--------------------------------------------
1. **Per asset, not per crop.** One confirmed photograph of one plant says
   nothing about the other acquisitions of the same crop, so a crop summary is
   reported but never used to award an asset a verdict. Broccoli has three
   assets and one annotated acquisition; the other two stay pending.
2. **Enough views, and not one lucky view.** An asset needs at least
   ``MIN_CONFIRMED_VIEWS_PER_ASSET`` confirmed views, a per-view floor (no
   single view may be catastrophic) *and* a pass rate across views. A single
   image that happens to score well cannot carry a verdict.
3. **Only the label set the carve actually used is scored for verification.**
   The carve consumes ``labels_merged``; ``labels_segformer`` and
   ``labels_sam_clip`` are reported for diagnosis but do not decide anything.

No human labels exist on this machine, so the real run produces
``annotation_pending`` for every asset. That is the correct output, not a
failure: the module is exercised by integration tests that build a synthetic
confirmation set, and those tests are the evidence the state can flip only
under the conditions above.

Usage
-----
    python -m agri.plant_gs.paa.stage4c.semantic_gate \
        --annotation results/agri_paa/stage4c/annotation \
        --out       results/agri_paa/stage4d/semantic_gate.json
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

from agri.plant_gs.paa.stage4c import annotation as annotation_mod  # noqa: E402

SCHEMA_VERSION = "paa-stage4d-semantic-gate-1"

#: The label set the plant-only carve consumes. Only this one can earn a verdict.
VERIFICATION_BACKEND = "labels_merged"

#: Minimum confirmed views before an asset may be scored at all.
MIN_CONFIRMED_VIEWS_PER_ASSET = 6
#: Crop summaries are only published with this many confirmed views.
MIN_CONFIRMED_VIEWS_PER_CROP = 6

#: Gate values. Chosen to be strict enough that a mostly-right label set still
#: fails: a carve that deletes white bolls cannot reach 0.80 plant recall.
MIN_PLANT_IOU = 0.70
MIN_PLANT_RECALL = 0.80
MAX_FALSE_PLANT_RATE = 0.10
MIN_PER_VIEW_IOU = 0.50
MIN_VIEW_PASS_RATE = 0.80

#: A pixel is "bright and desaturated" above these, in HSV.
WHITE_VALUE_MIN = 0.85
WHITE_SATURATION_MAX = 0.15
#: Morphological opening kernel; pixels a plant mask loses are "thin".
THIN_KERNEL = 7


def log(message: str) -> None:
    print(message, flush=True)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Region definitions, measured on the HUMAN label
# --------------------------------------------------------------------------- #
def _opening(mask: np.ndarray, kernel: int) -> np.ndarray:
    """Binary opening without OpenCV, so this runs in either venv."""
    size = (kernel, kernel)
    # Erode: a pixel survives only if the whole window is set.
    try:
        import cv2
        k = np.ones((kernel, kernel), np.uint8)
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
        return opened.astype(bool)
    except Exception:
        pass
    from scipy.ndimage import binary_erosion, binary_dilation
    structure = np.ones(size, dtype=bool)
    return binary_dilation(binary_erosion(mask, structure), structure)


def region_masks(image: np.ndarray, truth: np.ndarray) -> dict:
    """Thin-structure and white-tissue regions *inside the human plant label*.

    Both regions are derived from the human label, never from a model
    prediction. ``thin`` is what a 7x7 opening removes -- stems, petioles,
    leaf margins and veins. ``white`` is bright, desaturated plant tissue --
    cotton bolls, white flowers, pale stems -- which is exactly the tissue the
    Stage 4C candidate labels deleted most often.
    """
    plant = truth == annotation_mod.PLANT
    if not plant.any():
        return {"plant": plant, "thin": np.zeros_like(plant),
                "white": np.zeros_like(plant)}
    opened = _opening(plant, THIN_KERNEL)
    thin = plant & ~opened

    rgb = image.astype(np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
    white = plant & (maximum >= WHITE_VALUE_MIN) & (saturation <= WHITE_SATURATION_MAX)
    return {"plant": plant, "thin": thin, "white": white}


def region_recall(image: np.ndarray, truth: np.ndarray,
                  prediction: np.ndarray) -> dict:
    """Recall of a prediction on thin and white plant regions."""
    masks = region_masks(image, truth)
    predicted_plant = prediction == annotation_mod.PLANT
    out = {}
    for name in ("thin", "white"):
        region = masks[name]
        total = int(region.sum())
        hit = int((region & predicted_plant).sum())
        out[f"{name}_pixels"] = total
        out[f"{name}_recall"] = (hit / total) if total else None
    return out


# --------------------------------------------------------------------------- #
# Scoring one confirmed task
# --------------------------------------------------------------------------- #
def score_task(task: dict, label_path: Path, preannotation_dir: Path,
               image_path: Path | None) -> dict:
    truth = np.asarray(Image.open(label_path))
    image = None
    if image_path is not None and image_path.is_file():
        image = np.asarray(Image.open(image_path).convert("RGB"))
        if image.shape[:2] != truth.shape[:2]:
            image = np.asarray(Image.fromarray(image).resize(
                (truth.shape[1], truth.shape[0]), Image.Resampling.BILINEAR))
    row = {"crop": task.get("crop"), "asset_id": task.get("asset_id"),
           "task_id": task.get("task_id"), "predictions": {}}
    for path in sorted(preannotation_dir.glob("*.png")) if preannotation_dir.is_dir() else []:
        prediction = np.asarray(Image.open(path))
        score = annotation_mod.score_against(truth, prediction)
        if image is not None:
            score.update(region_recall(image, truth, prediction))
        else:
            score.update({"thin_pixels": None, "thin_recall": None,
                          "white_pixels": None, "white_recall": None})
        row["predictions"][path.stem] = score
    return row


# --------------------------------------------------------------------------- #
# Aggregation and the gate
# --------------------------------------------------------------------------- #
def _mean(values: list):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _minimum(values: list):
    values = [v for v in values if v is not None]
    return float(np.min(values)) if values else None


def aggregate(scores: list) -> dict:
    keys = ("plant_iou", "plant_precision", "plant_recall", "plant_false_rate",
            "uncertain_fraction", "thin_recall", "white_recall")
    out = {"n_views": len(scores)}
    for key in keys:
        out[key] = _mean([s.get(key) for s in scores])
        out[f"{key}_min"] = _minimum([s.get(key) for s in scores])
    return out


def gate_asset(rows: list, total_tasks: int) -> dict:
    """Decide one asset. ``rows`` are the confirmed, scored tasks of that asset."""
    n_confirmed = len(rows)
    reasons = []
    if n_confirmed == 0:
        return {"status": "annotation_pending",
                "verified": False,
                "n_confirmed_views": 0,
                "n_tasks": total_tasks,
                "n_required_views": MIN_CONFIRMED_VIEWS_PER_ASSET,
                "reasons": ["no human-confirmed view for this asset"],
                "metrics": {}}
    if n_confirmed < MIN_CONFIRMED_VIEWS_PER_ASSET:
        return {"status": "insufficient_coverage",
                "verified": False,
                "n_confirmed_views": n_confirmed,
                "n_tasks": total_tasks,
                "n_required_views": MIN_CONFIRMED_VIEWS_PER_ASSET,
                "reasons": [f"{n_confirmed} confirmed view(s); "
                            f"{MIN_CONFIRMED_VIEWS_PER_ASSET} required"],
                "metrics": {}}

    backend_scores = [row["predictions"][VERIFICATION_BACKEND] for row in rows
                      if VERIFICATION_BACKEND in row.get("predictions", {})]
    if not backend_scores:
        return {"status": "failed", "verified": False,
                "n_confirmed_views": n_confirmed, "n_tasks": total_tasks,
                "n_required_views": MIN_CONFIRMED_VIEWS_PER_ASSET,
                "reasons": [f"no {VERIFICATION_BACKEND} prediction exists for "
                            "the confirmed views"],
                "metrics": {}}

    metrics = aggregate(backend_scores)
    if (metrics["plant_iou"] or 0) < MIN_PLANT_IOU:
        reasons.append(f"plant IoU {metrics['plant_iou']:.3f} < {MIN_PLANT_IOU}")
    if (metrics["plant_recall"] or 0) < MIN_PLANT_RECALL:
        reasons.append(f"plant recall {metrics['plant_recall']:.3f} < {MIN_PLANT_RECALL}")
    if (metrics["plant_false_rate"] or 0) > MAX_FALSE_PLANT_RATE:
        reasons.append(f"false plant rate {metrics['plant_false_rate']:.3f} > "
                       f"{MAX_FALSE_PLANT_RATE}")
    per_view = [s.get("plant_iou") or 0.0 for s in backend_scores]
    worst = min(per_view)
    if worst < MIN_PER_VIEW_IOU:
        reasons.append(f"worst view IoU {worst:.3f} < {MIN_PER_VIEW_IOU}")
    pass_rate = sum(1 for v in per_view if v >= MIN_PER_VIEW_IOU) / len(per_view)
    if pass_rate < MIN_VIEW_PASS_RATE:
        reasons.append(f"only {pass_rate:.0%} of views reach the per-view floor")

    return {
        "status": "evaluated" if not reasons else "failed",
        "verified": not reasons,
        "n_confirmed_views": n_confirmed,
        "n_tasks": total_tasks,
        "n_required_views": MIN_CONFIRMED_VIEWS_PER_ASSET,
        "reasons": reasons,
        "metrics": metrics,
        "gate": {
            "verification_backend": VERIFICATION_BACKEND,
            "min_plant_iou": MIN_PLANT_IOU,
            "min_plant_recall": MIN_PLANT_RECALL,
            "max_false_plant_rate": MAX_FALSE_PLANT_RATE,
            "min_per_view_iou": MIN_PER_VIEW_IOU,
            "min_view_pass_rate": MIN_VIEW_PASS_RATE,
            "min_confirmed_views": MIN_CONFIRMED_VIEWS_PER_ASSET,
        },
    }


def evaluate(directory: Path) -> dict:
    """Score every confirmed task and decide every asset that appears in them."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no annotation manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    confirmed = annotation_mod.load_confirmed(directory, manifest)

    confirmed_ids = [k for k, v in confirmed.items()
                     if v.get("status") == "human_confirmed"]
    by_task = {t["task_id"]: t for t in manifest["tasks"]}

    per_task = {}
    per_asset_rows = {}
    for task_id in confirmed_ids:
        task = by_task[task_id]
        image_path = directory / task["image"] if task.get("image") else None
        row = score_task(task, confirmed[task_id]["label"],
                         directory / "preannotations" / task_id, image_path)
        row["annotator"] = confirmed[task_id]["record"].get("annotator")
        row["confirmed_utc"] = confirmed[task_id]["record"].get("confirmed_utc")
        per_task[task_id] = row
        per_asset_rows.setdefault(task.get("asset_id"), []).append(row)

    task_counts = {}
    for task in manifest["tasks"]:
        task_counts[task.get("asset_id")] = task_counts.get(task.get("asset_id"), 0) + 1

    per_asset = {}
    for asset_id, total in sorted(task_counts.items()):
        rows = per_asset_rows.get(asset_id, [])
        verdict = gate_asset(rows, total)
        verdict["asset_id"] = asset_id
        verdict["crop"] = (rows[0]["crop"] if rows else
                           next((t["crop"] for t in manifest["tasks"]
                                 if t.get("asset_id") == asset_id), None))
        verdict["per_backend"] = {
            backend: aggregate([r["predictions"][backend] for r in rows
                                if backend in r.get("predictions", {})])
            for backend in sorted({b for r in rows for b in r.get("predictions", {})})
        }
        per_asset[asset_id] = verdict

    # Crop summaries are descriptive. ``certifies_assets`` is explicitly False
    # so no reader can mistake a good crop average for a per-asset verdict.
    per_crop = {}
    for crop in sorted({t["crop"] for t in manifest["tasks"]}):
        asset_ids = sorted({t["asset_id"] for t in manifest["tasks"]
                            if t["crop"] == crop})
        rows = [r for aid in asset_ids for r in per_asset_rows.get(aid, [])]
        n_confirmed = len(rows)
        scores = [r["predictions"][VERIFICATION_BACKEND] for r in rows
                  if VERIFICATION_BACKEND in r.get("predictions", {})]
        per_crop[crop] = {
            "assets": asset_ids,
            "assets_annotated": sum(1 for aid in asset_ids
                                    if per_asset[aid]["n_confirmed_views"] > 0),
            "assets_verified": sum(1 for aid in asset_ids
                                   if per_asset[aid]["verified"]),
            "n_confirmed_views": n_confirmed,
            "n_required_views": MIN_CONFIRMED_VIEWS_PER_CROP,
            "status": ("annotation_pending" if n_confirmed == 0 else
                       "insufficient_coverage" if n_confirmed < MIN_CONFIRMED_VIEWS_PER_CROP
                       else "evaluated"),
            "metrics": aggregate(scores) if scores else {},
            "certifies_assets": False,
            "note": ("crop averages are reported for diagnosis only; each asset "
                     "needs its own confirmed views"),
        }

    verified = sorted(a for a, v in per_asset.items() if v["verified"])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc(),
        "annotation_dir": str(directory.resolve()),
        "policy": {
            "unit_of_verification": "asset (one acquisition of one plant)",
            "crop_summary_certifies_nothing": True,
            "verification_backend": VERIFICATION_BACKEND,
            "model_predictions_are_not_truth": True,
        },
        "tasks_total": len(manifest["tasks"]),
        "tasks_human_confirmed": len(confirmed_ids),
        "tasks_unconfirmed_ignored": sum(
            1 for v in confirmed.values() if v.get("status") == "unconfirmed_ignored"),
        "annotators": sorted({per_task[t]["annotator"] for t in per_task}),
        "per_task": per_task,
        "per_asset": per_asset,
        "per_crop": per_crop,
        "assets_semantic_verified": verified,
        "caveat": ("A verdict exists only for an asset that has enough "
                   "human-confirmed views and passes the per-view and aggregate "
                   "gate. With no confirmed labels every asset stays pending; "
                   "that is the honest state, not a missing measurement."),
    }


def asset_benchmark(report: dict | None, asset_id: str | None) -> dict:
    """The slice of the gate report that ``assets.py`` copies into an entry."""
    if not report or not asset_id:
        return {"status": "annotation_pending", "truth_source": None,
                "verified": False, "n_confirmed_views": 0,
                "n_required_views": MIN_CONFIRMED_VIEWS_PER_ASSET,
                "reasons": ["no semantic gate report was loaded"], "metrics": {}}
    verdict = (report.get("per_asset") or {}).get(asset_id)
    if verdict is None:
        return {"status": "annotation_pending", "truth_source": None,
                "verified": False, "n_confirmed_views": 0,
                "n_required_views": MIN_CONFIRMED_VIEWS_PER_ASSET,
                "reasons": ["this asset has no annotation tasks"], "metrics": {}}
    truth = "human_confirmed" if verdict.get("n_confirmed_views") else None
    return {
        "status": verdict.get("status"),
        "truth_source": truth,
        "verified": bool(verdict.get("verified")),
        "n_confirmed_views": verdict.get("n_confirmed_views"),
        "n_required_views": verdict.get("n_required_views"),
        "reasons": verdict.get("reasons") or [],
        "metrics": verdict.get("metrics") or {},
        "gate_source": report.get("annotation_dir"),
        "per_backend": verdict.get("per_backend") or {},
        "crop_summary_signals_only": True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation", type=Path,
                    default=Path("results/agri_paa/stage4c/annotation"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/agri_paa/stage4d/semantic_gate.json"))
    args = ap.parse_args()

    report = evaluate(args.annotation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps({
        "tasks_total": report["tasks_total"],
        "tasks_human_confirmed": report["tasks_human_confirmed"],
        "assets_semantic_verified": report["assets_semantic_verified"],
        "per_crop": {c: v["status"] for c, v in report["per_crop"].items()},
    }, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
