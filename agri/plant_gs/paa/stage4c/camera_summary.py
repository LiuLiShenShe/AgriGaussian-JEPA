#!/usr/bin/env python
"""One cross-crop camera table from the per-sample COLMAP audits.

Why a separate summary
----------------------
Each sample's ``colmap_audit.json`` answers "is this sample usable", and the
pipeline acts on that answer. What it does not answer is the *library-level*
question the brief asks: how many distinct camera models are represented, are
any of them distorted, and does the 90-degree roll handling that ``prepare_paa``
applies actually hold across crops. Those are properties of the collection, and
they are the reason a cross-crop asset library needs a cross-crop camera record.

The roll column matters more than it looks: a frame whose stored EXIF
orientation disagrees with the COLMAP camera size must be rolled to match, and
if that decision is made per-sample by inspection rather than by the audit, a
future crop can be rolled the wrong way and still train.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path, default=None):
    try:
        if path.is_file():
            return json.loads(path.read_text())
    except Exception:
        return default
    return default


def summarise_sample(entry: dict) -> dict:
    workspace = Path(entry["workspace"])
    audit = read_json(workspace / "colmap_audit.json") or {}
    preparation = read_json(workspace / "preparation.json") or {}
    cameras = audit.get("cameras") or {}
    poses = audit.get("poses") or {}
    return {
        "asset_id": entry["asset_id"],
        "crop": entry["crop"],
        "colmap_status": audit.get("status"),
        "n_cameras": len(cameras),
        "camera_models": sorted({(c or {}).get("model") for c in cameras.values()
                                 if c}) or [],
        "distortion_free": audit.get("camera_model_distortion_free"),
        "needing_undistortion": audit.get("camera_models_needing_undistortion") or [],
        "unsupported_models": audit.get("camera_models_unsupported") or [],
        "trainer_ready_without_undistortion":
            audit.get("trainer_ready_without_undistortion"),
        "per_camera": {
            key: {
                "model": (camera or {}).get("model"),
                "images": (camera or {}).get("images"),
                "width": (camera or {}).get("width"),
                "height": (camera or {}).get("height"),
                "horizontal_fov_deg": (camera or {}).get("horizontal_fov_deg"),
                "distortion_free": (camera or {}).get("distortion_free"),
            } for key, camera in sorted(cameras.items())
        },
        "frames_rolled_ccw90": preparation.get("frames_rolled_ccw90"),
        "frames_used": preparation.get("frames_used"),
        "registered_images": preparation.get("registered_images"),
        "reprojection_error_mean_px": preparation.get("reprojection_error_mean_px"),
        "reprojection_error_p95_px": (audit.get("reprojection_error_px") or {}).get("p95"),
        "orbit_radius": preparation.get("orbit_radius"),
        "poses": poses if isinstance(poses, dict) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path,
                    default=Path("results/agri_paa/stage4c/assets.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/agri_paa/stage4c/camera_summary.json"))
    args = ap.parse_args()

    library = json.loads(args.library.read_text())
    rows = [summarise_sample(e) for e in library["entries"]]

    models = sorted({m for row in rows for m in row["camera_models"]})
    distorted = sorted({m for row in rows for m in row["needing_undistortion"]})
    rolled = {row["asset_id"]: row["frames_rolled_ccw90"] for row in rows
              if row["frames_rolled_ccw90"]}
    roll_consistent = all(
        row["frames_rolled_ccw90"] is None or
        (row["frames_used"] and row["frames_rolled_ccw90"] <= row["frames_used"])
        for row in rows)
    report = {
        "status": "PASS" if all(r["colmap_status"] == "PASS" for r in rows) else "PARTIAL",
        "n_samples": len(rows),
        "crops": sorted({row["crop"] for row in rows}),
        "camera_models_seen": models,
        "n_distinct_camera_models": len(models),
        "models_needing_undistortion": distorted,
        "roll_applied": rolled,
        "roll_bounds_consistent": roll_consistent,
        "all_distortion_handled": all(not r["unsupported_models"] for r in rows),
        "samples": rows,
        "note": ("a camera model appearing here is a COLMAP model name observed in a "
                 "PAA sparse reconstruction, with the frame counts that use it. It is "
                 "not a statement about the physical camera"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps({k: report[k] for k in
                    ("status", "n_samples", "camera_models_seen", "roll_applied",
                     "all_distortion_handled")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
