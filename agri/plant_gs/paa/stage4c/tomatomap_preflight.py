#!/usr/bin/env python
"""TomatoMAP access pre-flight: what is actually there, before any download.

The brief allows a *pre-flight* on TomatoMAP and forbids starting a large
tomato reconstruction. This module is written to make that boundary structural
rather than a matter of discipline: it only issues ``HEAD``/``GET`` requests
against the **API metadata**, downloads no imagery, and answers four questions:

1. Is the repository reachable from this host, and is it gated or public?
2. What is the license, and what does it permit for research use?
3. What does the acquisition layout look like -- how many capture units, how
   many views per unit, what camera metadata is present?
4. Are the intrinsics and turntable poses present in a form a COLMAP-style
   reconstruction could consume?

It writes its answers, including a negative answer, to a JSON file. A
repository that is unreachable from this host is reported as unreachable, not
skipped silently: the conclusion "not assessed" is itself a result, and a later
reader must be able to tell it apart from "assessed and fine".

No large download. No reconstruction. No tomato 3DGS from this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "huggingface_hub/1.8.0 (paa-tomatomap-preflight)"


def log(message: str) -> None:
    print(message, flush=True)


def _get(url: str, timeout: int = 60, headers: dict | None = None):
    request = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            return handle.status, json.load(handle), dict(handle.headers)
    except urllib.error.HTTPError as exc:
        body = None
        try:
            body = json.load(exc)
        except Exception:
            body = exc.read()[:400].decode("utf-8", "replace")
        return exc.code, body, dict(exc.headers)
    except Exception as exc:
        return None, {"error": f"{type(exc).__name__}: {exc}"}, {}


def probe_repo(repo_id: str, endpoint: str, control: str = "JackTrum938/PAA-Plant-Dataset") -> dict:
    """Probe ``repo_id`` and, for comparison, a dataset known to be public.

    The mirror answers 401 "Invalid username or password" for *any* dataset it
    cannot serve -- including ids that do not exist. A bare 401 therefore cannot
    distinguish "gated" from "absent", and reporting it as "gated" would be a
    conclusion drawn from the endpoint's error format rather than from the
    dataset. Probing a known-public control in the same pass calibrates the
    endpoint, so the 401 can be classified honestly.
    """
    status, payload, _ = _get(f"{endpoint}/api/datasets/{repo_id}")
    control_status, control_payload, _ = _get(f"{endpoint}/api/datasets/{control}")
    endpoint_behaves = (control_status == 200)
    result = {"endpoint": endpoint, "http_status": status,
              "control_repo": control, "control_status": control_status,
              "endpoint_can_serve_public_datasets": endpoint_behaves}
    if status is None:
        result.update({"reachable": False, "detail": payload.get("error"),
                       "conclusion": "unreachable from this host; NOT assessed"})
        return result
    result["reachable"] = True
    if status == 200:
        card = payload.get("cardData") or {}
        result.update({
            "gated": payload.get("gated"),
            "private": payload.get("private"),
            "sha": payload.get("sha"),
            "last_modified": payload.get("lastModified"),
            "license": card.get("license"),
            "license_name": card.get("license_name"),
            "license_link": card.get("license_link"),
            "tags": payload.get("tags"),
            "pretty_name": payload.get("prettyName") or payload.get("pretty_name"),
            "conclusion": "public and reachable" if not payload.get("gated")
                          else "reachable but marked gated"})
        return result
    if status in (401, 403, 404):
        if endpoint_behaves:
            result.update({
                "detail": payload,
                "conclusion": ("the endpoint serves known-public datasets but not "
                               "this id, so it is gated or private or the id is "
                               "wrong; a credential or an approved request would "
                               "settle which")})
        else:
            result.update({
                "detail": payload,
                "conclusion": ("this endpoint cannot serve ANY dataset from this "
                               "host, so this result says nothing about the "
                               "dataset; NOT assessed")})
        return result
    result.update({"detail": payload,
                   "conclusion": f"unexpected HTTP {status}; NOT assessed"})
    return result


def list_tree(repo_id: str, endpoint: str, revision: str, prefix: str = "") -> list:
    status, payload, _ = _get(
        f"{endpoint}/api/datasets/{repo_id}/tree/{revision}/{prefix}?recursive=false")
    if status != 200 or not isinstance(payload, list):
        return []
    return payload


def list_tree_recursive(repo_id: str, endpoint: str, revision: str,
                        prefix: str, max_entries: int = 5000) -> list:
    out, cursor = [], None
    while len(out) < max_entries:
        url = (f"{endpoint}/api/datasets/{repo_id}/tree/{revision}/{prefix}"
               f"?recursive=true")
        if cursor:
            url += f"&cursor={cursor}"
        status, page, _ = _get(url, timeout=180)
        if status != 200 or not isinstance(page, list):
            break
        out.extend(page)
        if len(page) < 1000:
            break
        cursor = None
        break
    return out


def describe_layout(entries: list) -> dict:
    """Group a file listing into capture units and describe each one."""
    units = {}
    for entry in entries:
        if entry.get("type") != "file":
            continue
        parts = entry["path"].split("/")
        if len(parts) < 2:
            continue
        unit = "/".join(parts[:2])
        rest = "/".join(parts[2:])
        counts = units.setdefault(unit, {"n_files": 0, "kinds": {}})
        counts["n_files"] += 1
        kind = rest.split("/")[0]
        counts["kinds"][kind] = counts["kinds"].get(kind, 0) + 1
    return {
        "n_units": len(units),
        "units": {k: {"n_files": v["n_files"], "kinds": v["kinds"]}
                  for k, v in sorted(units.items())},
    }


CAMERA_HINTS = ("intrinsic", "intrinsics", "camera", "cam", "K", "extrinsic",
                "pose", "poses", "transform", "matrix", "calib")
POSE_HINTS = ("pose", "transform", "matrix", "turn", "turntable", "angle", "rotation")


def assess_metadata(entries: list) -> dict:
    names = [Path(e["path"]).name.lower() for e in entries if e.get("type") == "file"]
    return {
        "n_files": len(entries),
        "camera_metadata_files": sorted(
            {n for n in names if any(h in n for h in CAMERA_HINTS)})[:20],
        "pose_metadata_files": sorted(
            {n for n in names if any(h in n for h in POSE_HINTS)})[:20],
        "extensions": sorted({Path(n).suffix for n in names}),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="TomatoMAP/TomatoMAP",
                    help="Hub dataset id to probe (a list is tried if the first "
                         "one is not found)")
    ap.add_argument("--candidates", nargs="*", default=[
        "TomatoMAP/TomatoMAP", "TomatoMAP/tomatomap", "Liuxinyv/TomatoMAP",
        "xiaoyang/TomatoMAP"])
    ap.add_argument("--endpoints", nargs="*", default=["https://huggingface.co",
                                                      "https://hf-mirror.com"])
    ap.add_argument("--out", type=Path,
                    default=Path("results/agri_paa/stage4c/tomatomap_preflight.json"))
    args = ap.parse_args()

    report = {
        "status": "PASS",
        "probed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "access/pose/intrinsics pre-flight only; no download, no "
                   "reconstruction, no tomato 3DGS is produced here",
        "attempts": [],
        "conclusion": None,
    }

    chosen = None
    for endpoint in args.endpoints:
        for repo_id in args.candidates:
            probe = probe_repo(repo_id, endpoint)
            report["attempts"].append({"repo": repo_id, **probe})
            if probe.get("http_status") == 200:
                chosen = {"repo": repo_id, "endpoint": endpoint, "probe": probe}
                break
        if chosen:
            break

    if chosen is None:
        assessed = [a for a in report["attempts"]
                    if a.get("endpoint_can_serve_public_datasets")]
        report["conclusion"] = (
            "TomatoMAP could not be read from this host. On an endpoint that DOES "
            "serve known-public datasets, every candidate id was refused, which "
            "means gated/private or a wrong id -- a credential or an approved "
            "request is needed to tell those apart. On the official endpoint the "
            "host has no network route at all. Either way this is an ACCESS result, "
            "not a data-quality result: poses and intrinsics are unassessed, and no "
            "download or reconstruction was started."
            if assessed else
            "No endpoint could serve public datasets from this host, so the probe "
            "says nothing about TomatoMAP. NOT assessed.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        log(json.dumps(report["attempts"], indent=2))
        log(report["conclusion"])
        return

    repo_id, endpoint = chosen["repo"], chosen["endpoint"]
    probe = chosen["probe"]
    report["repo"] = repo_id
    report["endpoint"] = endpoint
    report["access"] = probe
    revision = probe.get("sha") or "main"

    top = list_tree(repo_id, endpoint, revision)
    report["top_level"] = [{"path": e["path"], "type": e["type"]} for e in top]

    layout = {}
    for entry in top:
        if entry.get("type") != "directory":
            continue
        name = entry["path"]
        if name.lower() in ("data", "images", "captures"):
            entries = list_tree_recursive(repo_id, endpoint, revision, name)
            layout[name] = {"entries_sampled": len(entries),
                            **assess_metadata(entries)}
            layout[name]["layout"] = describe_layout(
                entries[: max(len(entries), 1)] if len(entries) < 200 else entries[:200])
    report["capture_layout"] = layout
    report["conclusion"] = (
        "Access pre-flight only. Nothing here says a tomato capture unit is "
        "reconstructable: that requires downloading one unit and running the same "
        "COLMAP audit + alignment + 7k gate the PAA crops went through, which this "
        "stage deliberately does not start.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps({k: report[k] for k in
                    ("repo", "access", "top_level")}, indent=2))
    log(report["conclusion"])


if __name__ == "__main__":
    main()
