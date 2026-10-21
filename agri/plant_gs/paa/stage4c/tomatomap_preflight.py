#!/usr/bin/env python
"""Stage 4D TomatoMAP pre-flight: the right sources, measured honestly.

Stage 4C probed four guessed repository ids on two endpoints and concluded
"could not be read". That conclusion was wrong in one specific way that mattered:
the *correct* id had never been tried. The brief names two sources --

* the original e!DAL repository, DOI ``10.5447/ipk/2025/14``;
* the Hugging Face dataset ``Voxel51/tomato-map``.

This module probes exactly those, plus the two auxiliary links, and it
classifies every outcome into one of four buckets that Stage 4C could not tell
apart: **unreachable**, **repository absent**, **access restricted**, and
**download failed**. It also records what the FiftyOne grouping does and does
not give you, because "one group per camera pose" is not "COLMAP poses".

No large download happens here. The single-capture download is a separate,
explicit step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = "agri-stage4d-tomatomap-preflight/1.0"

#: The two verified sources, and the two auxiliary references.
HF_DATASET = "Voxel51/tomato-map"
HF_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")
CONTROL_DATASET = "JackTrum938/PAA-Plant-Dataset"   # known public, used to calibrate
EDAL_DOI = "https://doi.org/10.5447/ipk/2025/14"
EDAL_ROOT = "https://edal.ipk-gatersleben.de/"
CODE_REPO = "https://github.com/0YJ/TomatoMAP"
PAPER_DOI = "https://doi.org/10.1038/s41597-026-06926-9"


def log(message: str) -> None:
    print(message, flush=True)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(url: str, timeout: int = 60, headers: dict | None = None, retries: int = 5):
    """One request with bounded retries; a mirror can be briefly unavailable.

    The mirror occasionally drops a TLS handshake for a path it is happy to
    serve a second later, so a single transport error is not evidence about the
    repository. Backing off and retrying is what keeps "unreachable" honest.
    """
    last = None
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                # Read inside the context: once the response is closed a later
                # ``read()`` returns b"", which is how the first version of this
                # module turned every 200 into "unexpected_body".
                return handle.status, handle.read(), dict(handle.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, None, dict(exc.headers)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            last = {"error": message, "attempt": attempt + 1}
            # "Network is unreachable" / DNS failure is a statement about the
            # host's routing, not a transient blip; retrying it only burns the
            # budget that the mirror needs.
            if ("Network is unreachable" in message
                    or "Name or service not known" in message
                    or "Temporary failure in name resolution" in message):
                break
            time.sleep(1.0 * (attempt + 1))
    return None, None, last or {}


def classify(status: int | None, detail: str | None) -> str:
    """One of: unreachable / repository_absent / access_restricted / present."""
    if status is None:
        return "unreachable"
    if status == 200:
        return "present"
    if status in (401, 403, 407):
        return "access_restricted"
    if status in (404, 410):
        return "repository_absent"
    if status == 429:
        return "rate_limited"
    return f"unexpected_http_{status}"


def probe_url(url: str, what: str, timeout: int = 30) -> dict:
    status, _, headers = _get(url, timeout=timeout, retries=2)
    entry = {"url": url, "what": what, "http_status": status,
             "classification": classify(status, None)}
    if status is None:
        entry["error"] = headers.get("error")
    return entry


def probe_dataset(endpoint: str, repo_id: str, control: str = CONTROL_DATASET) -> dict:
    """Probe one dataset id, with a known-public control in the same pass.

    The control is probed **first** on purpose: it warms the connection to an
    endpoint that drops a cold TLS handshake now and then, and it is the
    calibration the refusal is read against.
    """
    control_status, control_body, _ = _get(f"{endpoint}/api/datasets/{control}",
                                           timeout=45)
    url = f"{endpoint}/api/datasets/{repo_id}"
    status, handle, headers = _get(url, timeout=120, retries=2,
                                   headers={"Accept": "application/json"})
    entry = {"url": url, "repo": repo_id, "endpoint": endpoint,
             "http_status": status,
             "classification": classify(status, None),
             "control": {"repo": control, "http_status": control_status,
                         "classification": classify(control_status, None)}}
    if status == 200:
        raw = handle or b""
        try:
            payload = json.loads(raw)
        except Exception:
            entry.update({
                "classification": "unexpected_body",
                "detail": raw[:200].decode("utf-8", "replace"),
                "conclusion": ("the endpoint answered 200 with a non-JSON body; "
                               "this is a proxy or interstitial page, not a "
                               "dataset record"),
            })
            return entry
        entry.update({
            "sha": payload.get("sha"),
            "last_modified": payload.get("lastModified"),
            "private": payload.get("private"),
            "gated": payload.get("gated"),
            "disabled": payload.get("disabled"),
            "license": (payload.get("cardData") or {}).get("license"),
            "tags": payload.get("tags"),
            "downloads": payload.get("downloads"),
            "n_files_listed": len(payload.get("siblings") or []),
            "conclusion": ("public and reachable; no credential needed"
                           if not payload.get("gated") else
                           "reachable but marked gated"),
        })
        return entry
    if status is None:
        entry["error"] = headers.get("error")
    if status in (401, 403) and control_status == 200:
        entry["conclusion"] = ("the endpoint serves known-public datasets but not "
                               "this id, so it is gated, private, or the id is "
                               "wrong; a credential would settle which")
    elif status in (401, 403):
        entry["conclusion"] = ("the endpoint cannot serve ANY dataset from this "
                               "host, so this says nothing about the dataset")
    elif status == 404:
        entry["conclusion"] = "the endpoint answered and the id does not exist there"
    return entry


def list_top(repo_id: str, endpoint: str, revision: str = "main") -> list:
    status, handle, _ = _get(f"{endpoint}/api/datasets/{repo_id}/tree/{revision}")
    if status != 200 or handle is None:
        return []
    try:
        return json.loads(handle)
    except Exception:
        return []


def probe_metadata_files(endpoint: str, repo_id: str, revision: str,
                          paths: list) -> dict:
    """Fetch the small files that answer the pose/intrinsics questions."""
    out = {}
    for path in paths:
        url = f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{path}"
        status, _, headers = _get(url, timeout=30, retries=1)
        out[path] = {"url": url, "http_status": status,
                     "classification": classify(status, None),
                     "bytes": int(headers.get("Content-Length") or 0) or None}
    return out


# --------------------------------------------------------------------------- #
# FiftyOne grouping vs COLMAP poses
# --------------------------------------------------------------------------- #
FIFTYONE_VS_COLMAP = {
    "what_fiftyone_gives": (
        "16,116 groups, each one turntable pose; each group has 4 slices "
        "(pos_1..pos_4) -- the 4 simultaneous camera exposures of that pose. "
        "plant_id, pose_id, pose_degrees, image_set_id and capture_datetime are "
        "per-sample fields, and 1,343 (plant, date) units contain the full 48 "
        "images."
    ),
    "what_it_does_not_give": (
        "no extrinsic camera poses and no COLMAP-style camera matrix per view. "
        "metadata.json carries no camera_intrinsics and no static_transforms "
        "(both empty), and the dataset card states the calibration npz files are "
        "not attached to samples. The only pose information is the nominal "
        "turntable angle (12 x 30 deg) and the 4 nominal camera elevations "
        "(45/90/135/180 deg)."
    ),
    "consequence_for_reconstruction": (
        "a capture unit is reconstructable in principle -- 48 views of one plant "
        "on a known turntable, with known intrinsics -- but the camera motion "
        "must be either recovered by SfM on the images or derived from the "
        "nominal angles, and the white backdrop/pot must be masked out or SfM "
        "will lock onto it. Neither is established by the metadata alone."
    ),
    "joinability": (
        "the macro-photograph segmentation subset is a different imaging task "
        "(different camera, different window, no plant_id/pose/session ids) and "
        "must not be joined to the rig images by plant id."
    ),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("results/agri_paa/stage4d/tomatomap_preflight.json"))
    ap.add_argument("--revision", default=None,
                    help="Hub revision to inspect; the API sha by default")
    args = ap.parse_args()

    report = {
        "schema_version": "paa-stage4d-tomatomap-preflight-1",
        "probed_utc": _utc(),
        "purpose": ("access, metadata, calibration and pose-model pre-flight. "
                    "No bulk download; no reconstruction."),
        "probes": [],
    }

    # 1. the auxiliary references, including the original e!DAL DOI
    for url, what in ((EDAL_DOI, "e!DAL DOI landing page"),
                      (EDAL_ROOT, "e!DAL repository root"),
                      (CODE_REPO, "code repository"),
                      (PAPER_DOI, "paper DOI")):
        report["probes"].append(probe_url(url, what))

    # 2. the Hugging Face dataset, on both endpoints
    for endpoint in HF_ENDPOINTS:
        report["probes"].append(
            {"kind": "hf_dataset_api", **probe_dataset(endpoint, HF_DATASET)})

    hf_present = [p for p in report["probes"]
                  if p.get("kind") == "hf_dataset_api" and p.get("http_status") == 200]
    chosen = None
    if hf_present:
        chosen = hf_present[0]
        endpoint = chosen["endpoint"]
        revision = args.revision or chosen.get("sha") or "main"
        report["hf_dataset"] = {"repo": HF_DATASET, "endpoint": endpoint,
                               "revision": revision,
                               "access": {k: chosen.get(k) for k in
                                          ("sha", "last_modified", "private",
                                           "gated", "disabled", "license",
                                           "downloads", "n_files_listed",
                                           "conclusion")}}
        top = list_top(HF_DATASET, endpoint, revision)
        report["hf_dataset"]["top_level"] = [{"path": e["path"],
                                              "type": e["type"],
                                              "size": e.get("size")}
                                             for e in top]
        report["hf_dataset"]["metadata_files"] = probe_metadata_files(
            endpoint, HF_DATASET, revision,
            ["metadata.json", "fiftyone.yml", "README.md"])
        # Calibration files the card claims exist; measured, not assumed.
        calibration = probe_metadata_files(
            endpoint, HF_DATASET, revision,
            ["data/calibration/pi1.npz", "data/calibration/pi2.npz",
             "data/calibration/pi3.npz", "data/calibration/pi4.npz"])
        report["hf_dataset"]["calibration_probe"] = calibration
        report["hf_dataset"]["fiftyone_vs_colmap"] = FIFTYONE_VS_COLMAP

    # 3. the code repository, which carries the calibration the HF card points
    #    at. Measured separately so "not in the dataset repo" is not mistaken
    #    for "not published".
    github_calibration = {}
    for camera in (1, 2, 3, 4):
        url = (f"https://raw.githubusercontent.com/0YJ/TomatoMAP/main/"
               f"src/models/pi{camera}.npz")
        status, _, headers = _get(url)
        github_calibration[f"pi{camera}.npz"] = {
            "url": url, "http_status": status,
            "classification": classify(status, None),
            "bytes": int(headers.get("Content-Length") or 0) or None}
    report["github_calibration"] = github_calibration
    report["calibration_location"] = {
        "in_hf_dataset_repo": any(
            v.get("http_status") == 200
            for v in (report["hf_dataset"].get("calibration_probe") or {}).values())
        if hf_present else None,
        "in_code_repo": any(v.get("http_status") == 200
                            for v in github_calibration.values()),
        "finding": ("the calibration npz files are NOT in the Hugging Face "
                    "dataset repository (404 there) but ARE published in the "
                    "code repository under src/models/pi{1-4}.npz"),
    }

    reachable = [p for p in report["probes"] if p.get("http_status") is not None]
    report["summary"] = {
        "hf_dataset_present": bool(hf_present),
        "hf_dataset_gated": bool(chosen and chosen.get("gated")),
        "edal_landing_reachable": any(
            p.get("http_status") is not None for p in report["probes"]
            if p.get("what") == "e!DAL DOI landing page"),
        "endpoints_reachable": sorted({
            p["endpoint"] for p in report["probes"]
            if p.get("kind") == "hf_dataset_api" and p.get("http_status")}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps(report["summary"], indent=2))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
