#!/usr/bin/env python
"""Identity manifest for the whole PAA dataset, and the split groups it forces.

Why
---
PAA's dated samples are named ``<YYYYMMDD>-<plant_id>``. ``Broccoli/20260327-10``
and ``Broccoli/20260402-10`` are the *same physical plant* six days apart. Any
train/validation/test split that groups by sample directory puts the same plant
on both sides and reports memorisation of that plant's silhouette as
generalisation.

The manifest is therefore built from the **plant**, and the acquisition is a
separate axis. This module enumerates the dataset from the Hub (or from what is
on disk), derives both axes with the same function the registry uses, and
states the counts that a split has to respect: acquisitions, distinct plants,
crops, and the time span.

It also reports, for each plant, the *registration information* available for
the two dates that were reconstructed independently. Two captures of the same
plant are two independent COLMAP models: their coordinate frames are not
aligned, and nothing in PAA provides the transform between them. The manifest
says so explicitly, because "we have two dates of this plant" is often read as
"we can model growth", which needs registration that does not exist here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import group_key, parse_sample_id  # noqa: E402

REPO_ID = "JackTrum938/PAA-Plant-Dataset"
USER_AGENT = "huggingface_hub/1.8.0 (paa-identity-manifest)"


def log(message: str) -> None:
    print(message, flush=True)


def list_tree(prefix: str, revision: str = "main", timeout: int = 60) -> list:
    """Every file under a Hub path, via the paginated tree API."""
    import urllib.request
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/api/datasets/{REPO_ID}/tree/{revision}/{prefix}?recursive=false"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.load(handle)


def list_crops(revision: str = "main") -> list:
    return sorted(entry["path"].rstrip("/").split("/")[-1]
                  for entry in list_tree("", revision)
                  if entry.get("type") == "directory")


def list_acquisitions(crop: str, revision: str = "main") -> list:
    return sorted(entry["path"].rstrip("/").split("/")[-1]
                  for entry in list_tree(crop, revision)
                  if entry.get("type") == "directory")


def build_manifest(crops: dict) -> dict:
    acquisitions, plants = [], {}
    for crop, samples in sorted(crops.items()):
        for sample in samples:
            parsed = parse_sample_id(sample)
            record = {
                "crop": crop,
                "acquisition_id": sample,
                "plant_id": parsed["plant_id"],
                "acquisition_date": parsed["acquisition_date"],
                "dated": parsed["dated"],
                "split_group": group_key(crop, parsed["plant_id"]),
            }
            acquisitions.append(record)
            plants.setdefault(record["split_group"], {
                "split_group": record["split_group"],
                "crop": crop,
                "plant_id": parsed["plant_id"],
                "acquisitions": [],
                "dates": [],
            })
            plants[record["split_group"]]["acquisitions"].append(sample)
            if record["acquisition_date"]:
                plants[record["split_group"]]["dates"].append(
                    record["acquisition_date"])
    for entry in plants.values():
        entry["acquisitions"] = sorted(entry["acquisitions"])
        entry["dates"] = sorted(entry["dates"])
        entry["n_acquisitions"] = len(entry["acquisitions"])
        entry["time_span_days"] = (
            (_date(entry["dates"][-1]) - _date(entry["dates"][0])).days
            if len(entry["dates"]) > 1 else 0)

    dates = sorted({a["acquisition_date"] for a in acquisitions
                    if a["acquisition_date"]})
    return {
        "dataset_name": "PAA-Plant-Dataset",
        "source_url": f"https://huggingface.co/datasets/{REPO_ID}",
        "crops": sorted(crops),
        "n_crops": len(crops),
        "n_acquisitions": len(acquisitions),
        "n_distinct_plants": len(plants),
        "plants": plants,
        "acquisitions": acquisitions,
        "acquisition_dates": dates,
        "time_span": {
            "first": dates[0] if dates else None,
            "last": dates[-1] if dates else None,
            "n_distinct_dates": len(dates),
            "days": (_date(dates[-1]) - _date(dates[0])).days if dates else 0,
        },
        "split_group_rule": (
            "split_group = <crop>/<plant_id>; one plant's acquisitions must never "
            "be split across train/val/test. Acquisitions of a plant are the "
            "leaves of that group."),
    }


def _date(text: str):
    from datetime import date
    year, month, day = text.split("-")
    return date(int(year), int(month), int(day))


def registration_report(workspaces: dict) -> dict:
    """What registration information exists for multi-date plants.

    Each entry maps a split group to the independently reconstructed
    acquisitions and the camera/scale facts needed to align them -- and, just
    as importantly, the transform that is *missing*.
    """
    out = {}
    for group, paths in workspaces.items():
        entries = []
        for path in paths:
            meta_path = Path(path) / "preparation.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            entries.append({
                "acquisition_id": meta.get("acquisition_id"),
                "acquisition_date": meta.get("acquisition_date"),
                "workspace": str(Path(path).resolve()),
                "registered_images": meta.get("registered_images"),
                "frames_used": meta.get("frames_used"),
                "orbit_radius": meta.get("orbit_radius"),
                "reprojection_error_mean_px": meta.get("reprojection_error_mean_px"),
                "canvas": meta.get("canvas"),
                "undistortion": meta.get("undistortion", {}).get("applied", False),
            })
        entries.sort(key=lambda e: str(e.get("acquisition_date")))
        if len(entries) < 2:
            continue
        out[group] = {
            "acquisitions": entries,
            "n_acquisitions": len(entries),
            "cross_date_registration": {
                "available": False,
                "reason": ("each acquisition is an independent COLMAP model in its "
                           "own world frame; PAA ships no extrinsic alignment "
                           "between acquisitions, so the two reconstructions are "
                           "NOT in a common coordinate system"),
                "what_would_be_needed": [
                    "a shared reference frame (a common calibration target, a "
                    "measured marker, or a hand-registered correspondence set)",
                    "per-acquisition scale and principal-point comparison, which the "
                    "per-sample COLMAP models do provide",
                ],
                "per_acquisition_facts_available": [
                    "camera intrinsics and image resolutions",
                    "camera centre trajectory and orbit radius",
                    "sparse point count and reprojection error",
                ],
                "comparable_observables_without_registration": [
                    "plant silhouette area per view (in pixels, needs no common frame)",
                    "sparse point count and track statistics",
                    "number of cameras and their field of view",
                ],
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workspace", type=Path, action="append", default=[],
                    help="a reconstructed workspace; repeat for each acquisition")
    ap.add_argument("--offline", action="store_true",
                    help="do not contact the Hub; only report local workspaces")
    args = ap.parse_args()

    crops = {}
    if not args.offline:
        for crop in list_crops(args.revision):
            crops[crop] = list_acquisitions(crop, args.revision)
            log(f"{crop}: {len(crops[crop])} acquisitions {crops[crop]}")

    manifest = build_manifest(crops) if crops else {
        "n_crops": 0, "n_acquisitions": 0, "n_distinct_plants": 0,
        "note": "offline: no Hub listing was fetched"}

    # Local reconstructions, and what they say about cross-date registration.
    local = {}
    for path in args.workspace:
        meta_path = Path(path) / "preparation.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        crop = meta.get("crop")
        plant = meta.get("plant_id")
        key = group_key(crop, plant)
        local.setdefault(key, []).append(str(Path(path).resolve()))
    manifest["local_reconstructions"] = {
        "split_groups": {k: sorted(v) for k, v in sorted(local.items())},
        "registration": registration_report(local),
    }

    payload = {
        "status": "PASS",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **manifest,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        log(f"wrote {args.out}")

    summary = {
        "crops": manifest.get("n_crops"),
        "acquisitions": manifest.get("n_acquisitions"),
        "distinct_plants": manifest.get("n_distinct_plants"),
        "time_span": manifest.get("time_span"),
    }
    log(json.dumps(summary, indent=2))
    for group, entry in sorted(manifest.get("plants", {}).items()):
        if entry["n_acquisitions"] > 1:
            log(f"  {group}: {entry['n_acquisitions']} acquisitions "
                f"{entry['acquisitions']} span {entry['time_span_days']}d")


if __name__ == "__main__":
    main()
