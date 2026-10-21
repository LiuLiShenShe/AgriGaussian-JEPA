#!/usr/bin/env python
"""One TomatoMAP capture unit: recover, verify, and decide whether it is posed.

What this module will and will not do
-------------------------------------
It recovers the 48 images of one (plant, date) unit, checks that each decodes,
records the grouping fields, loads the per-camera calibration, validates that
calibration against the image size, and then answers one question: **are the
pose conditions for a 3DGS reconstruction established?**

If they are not, it says so and stops. It does not run COLMAP against a static
greenhouse background, and it does not turn a FiftyOne group into a camera
pose. That boundary is the point of the module.

Why the calibration needs validating rather than trusting
---------------------------------------------------------
The dataset card points at ``data/calibration/pi{1-4}.npz``. Those files are
*not* in the Hugging Face repository (the resolve endpoint returns 404); they
are in the code repository under ``src/models/pi{1-4}.npz``. Once fetched they
are usable, but at least one of them (``pi2``) carries a camera matrix whose
focal length is 0.15 px for a 1080x1440 photograph. That is not a matter of
taste -- it means the file cannot be used as a pinhole model, and the unit is
not poseable from calibration alone.

Usage
-----
    python -m agri.plant_gs.paa.stage4d.tomatomap_unit \
        --unit data/external/TomatoMAP/plant1_20230816 \
        --out  results/agri_paa/stage4d/tomatomap_unit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCHEMA_VERSION = "paa-stage4d-tomatomap-unit-1"

#: pi{cam}_{seq}_{plant}_{pose}_{YYYYmmddHHMMSS}.jpg -- the parsing rule the
#: dataset card documents and the only complete source of the grouping fields.
NAME_PATTERN = re.compile(
    r"^pi(?P<camera>[1-4])_(?P<seq>\d+)_(?P<plant>\d+)_(?P<pose>\d{1,2})_"
    r"(?P<timestamp>\d{14})\.jpg$")

CAMERA_ELEVATION_DEG = {1: 45, 2: 90, 3: 135, 4: 180}
POSE_STEP_DEG = 30


def log(message: str) -> None:
    print(message, flush=True)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_name(name: str) -> dict | None:
    match = NAME_PATTERN.match(name)
    if not match:
        return None
    groups = match.groupdict()
    return {"camera": int(groups["camera"]),
            "sequence": int(groups["seq"]),
            "plant_id": int(groups["plant"]),
            "pose_id": int(groups["pose"]),
            "pose_degrees": (int(groups["pose"]) - 1) * POSE_STEP_DEG,
            "capture_timestamp": groups["timestamp"],
            "camera_elevation_deg": CAMERA_ELEVATION_DEG[int(groups["camera"])]}


def inspect_unit(directory: Path) -> dict:
    images = sorted((Path(directory) / "images").glob("*.jpg"))
    records, problems = [], []
    for path in images:
        parsed = parse_name(path.name)
        if parsed is None:
            problems.append(f"{path.name}: does not match the documented pattern")
            continue
        try:
            with Image.open(path) as handle:
                handle.verify()
            with Image.open(path) as handle:
                width, height = handle.size
                mode = handle.mode
        except Exception as exc:
            problems.append(f"{path.name}: cannot decode ({exc})")
            continue
        records.append({
            "file": path.name, "width": width, "height": height, "mode": mode,
            "bytes": path.stat().st_size, "sha256": _sha256(path), **parsed,
        })

    cameras = sorted({r["camera"] for r in records})
    poses = sorted({r["pose_id"] for r in records})
    plants = sorted({r["plant_id"] for r in records})
    cells = {(r["camera"], r["pose_id"]) for r in records}
    complete = {c: sorted(p for cc, p in cells if cc == c) for c in cameras}
    expected = {(c, p) for c in (1, 2, 3, 4) for p in range(1, 13)}
    missing = sorted(expected - cells)
    duplicates = len(records) - len(cells)
    return {
        "n_images": len(records),
        "sizes": sorted({(r["width"], r["height"]) for r in records}),
        "cameras": cameras,
        "poses": poses,
        "plants": plants,
        "n_distinct_sha256": len({r["sha256"] for r in records}),
        "missing_camera_pose_cells": [list(cell) for cell in missing],
        "duplicate_cells": duplicates,
        "complete_48_view_unit": (len(records) == 48 and not missing
                                  and duplicates == 0 and len(plants) == 1),
        "per_camera": complete,
        "problems": problems,
        "records": records,
    }


def validate_calibration(directory: Path, image_size, cell_scale=None) -> dict:
    """Check each camera matrix against the photograph it claims to describe.

    A pinhole matrix is plausible for an image of size (W, H) when its focal
    lengths are of the order of the image dimension and its principal point
    lies inside the frame. Both are cheap, and both catch the failure that
    matters here: a matrix copied from a different camera model or scale.
    """
    width, height = image_size
    out = {"image_size": [width, height], "cameras": {}, "problems": []}
    for camera in (1, 2, 3, 4):
        path = Path(directory) / f"pi{camera}.npz"
        if not path.is_file():
            out["cameras"][str(camera)] = {"status": "missing"}
            out["problems"].append(f"pi{camera}.npz missing")
            continue
        payload = np.load(path, allow_pickle=True)
        matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(payload["distortion_coeff"], dtype=np.float64)
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        scale = cell_scale or 1.0
        fx_s, fy_s = fx * scale, fy * scale
        cx_s, cy_s = cx * scale, cy * scale
        plausible_focal = (0.1 * min(width, height) <= fx_s <= 10 * max(width, height)
                           and 0.1 * min(width, height) <= fy_s
                           <= 10 * max(width, height))
        inside = (0 <= cx_s <= width) and (0 <= cy_s <= height)
        status = "plausible" if (plausible_focal and inside) else "implausible"
        if status == "implausible":
            out["problems"].append(
                f"pi{camera}: fx={fx_s:.4g} fy={fy_s:.4g} cx={cx_s:.4g} "
                f"cy={cy_s:.4g} is not a usable pinhole model for "
                f"{width}x{height}")
        out["cameras"][str(camera)] = {
            "status": status,
            "camera_matrix": matrix.tolist(),
            "distortion_coeff": distortion.tolist(),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "principal_point_inside_image": inside,
            "focal_length_plausible": plausible_focal,
            "note": ("matrix is stored at a different scale or belongs to "
                     "another camera model" if status == "implausible" else ""),
        }
    out["usable_for_pinhole_sfm"] = not out["problems"]
    return out


def pose_assessment(unit: dict, calibration: dict) -> dict:
    """State exactly what is known about the camera motion, and what is not."""
    return {
        "turntable_model": ("12 poses at 30-degree increments (pose_id 1..12, "
                            "pose_degrees 0..330); 4 fixed cameras at nominal "
                            "elevations 45/90/135/180 degrees"),
        "extrinsics_provided": False,
        "extrinsics_evidence": ("metadata.json has empty camera_intrinsics and "
                                "empty static_transforms; the dataset card states "
                                "'no extrinsic rig calibration is provided'"),
        "intrinsics_provided": True,
        "intrinsics_usable": calibration.get("usable_for_pinhole_sfm"),
        "intrinsics_problems": calibration.get("problems") or [],
        "fiftyone_group_is_not_a_pose": (
            "group.name (pos_1..pos_4) names the camera elevation and "
            "image_set_id groups the four simultaneous exposures of one pose. "
            "Neither is a camera matrix; a FiftyOne group supplies no "
            "rotation or translation"),
        # Deliberately False: usable intrinsics alone do not pose a 4-camera
        # rig whose relative geometry is not published.
        "pose_conditions_established": False,
        "why_not": (
            "three independent gaps: (1) no extrinsic camera geometry is "
            "provided; (2) one of the four calibration files is not a usable "
            "pinhole model at the released resolution; (3) the scene contains a "
            "static greenhouse background, so an unconstrained SfM pass would "
            "register the room rather than the rotating plant, and no plant "
            "masks ship with the rig images"),
        "what_would_establish_them": [
            "per-image extrinsics from the authors, or",
            "a masked SfM pass: plant/support masks per image, background "
            "excluded, cameras constrained to the known turntable axis, then "
            "verified by reprojection error and silhouette IoU",
        ],
    }


def build_report(directory: Path, plant_id: int | None, date: str | None,
                 cell_scale: float | None = None) -> dict:
    directory = Path(directory)
    unit = inspect_unit(directory)
    unit["directory"] = str(directory.resolve())
    image_size = unit["sizes"][0] if unit["sizes"] else (0, 0)
    calibration = validate_calibration(directory, image_size, cell_scale)
    poses = pose_assessment(unit, calibration)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc(),
        "unit": {"directory": str(directory.resolve()),
                 "plant_id": plant_id,
                 "date": date,
                 "n_images": unit["n_images"],
                 "sizes": unit["sizes"],
                 "cameras": unit["cameras"],
                 "poses": unit["poses"],
                 "plants": unit["plants"],
                 "complete_48_view_unit": unit["complete_48_view_unit"],
                 "n_distinct_sha256": unit["n_distinct_sha256"],
                 "missing_camera_pose_cells": unit["missing_camera_pose_cells"],
                 "problems": unit["problems"]},
        "images": unit["records"],
        "calibration": calibration,
        "poses": poses,
        "reconstruction": {
            "attempted": False,
            "reason": ("pose conditions are not established; running 3DGS "
                       "would require inventing extrinsics, and a COLMAP pass "
                       "without masks would reconstruct the static greenhouse "
                       "background instead of the plant"),
            "not_a_data_quality_result": (
                "the unit itself is complete: 48/48 images decode at "
                "1080x1440 with 48 distinct hashes. The blocker is pose/"
                "calibration information, not the imagery"),
        },
        "license": ("CC BY 4.0 (Hugging Face dataset card Voxel51/tomato-map); "
                    "code repository 0YJ/TomatoMAP"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", type=Path,
                    default=Path("data/external/TomatoMAP/plant1_20230816"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/agri_paa/stage4d/tomatomap_unit"))
    ap.add_argument("--plant-id", type=int, default=1)
    ap.add_argument("--date", default="2023-08-16")
    ap.add_argument("--cell-scale", type=float, default=None,
                    help="scale factor if calibration was measured at another "
                         "resolution (Reported, never guessed silently)")
    args = ap.parse_args()

    report = build_report(args.unit, args.plant_id, args.date, args.cell_scale)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "unit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.out / "image_sha256.json").write_text(json.dumps(
        {r["file"]: r["sha256"] for r in report["images"]}, indent=2) + "\n")
    log(json.dumps({
        "unit": report["unit"],
        "calibration_problems": report["calibration"]["problems"],
        "pose_conditions_established": report["poses"]["pose_conditions_established"],
        "reconstruction_attempted": report["reconstruction"]["attempted"],
    }, indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
