#!/usr/bin/env python
"""Download one PAA sample (a single plant + acquisition) with provenance.

The full PAA-Plant-Dataset is ~28 GB across 41 samples. Every stage of this
project needs *one* plant at a time, so this downloads only the directories a
run actually consumes and records exactly which revision it came from:

    Camera/**          the COLMAP model and the frames it was solved on (required)
    RGB_seg/masks/**   the plant silhouettes (required for masked supervision)
    RGB_seg/whitebg/**  optional, but ``prepare_paa`` uses it to re-verify the
                        mask registration per image, so it is on by default
    RGB/**             optional, kept off by default; the raw photographs are
                        not what COLMAP was solved on

The dataset is public and ungated (Apache-2.0), so no token is required. The
resolved commit SHA is fetched from the Hub API and written into
``provenance.json`` next to the data, so a run can be tied to an exact revision
rather than to ``main``.

    python agri/plant_gs/paa/download_paa.py \
        --crop Broccoli --acquisition 20260327-10 \
        --root data/external/PAA-Plant-Dataset

Re-running is a no-op: huggingface_hub resumes from its local cache and only
fetches files that are missing or changed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ID = "JackTrum938/PAA-Plant-Dataset"
REPO_TYPE = "dataset"
DEFAULT_ROOT = Path("data/external/PAA-Plant-Dataset")

GROUPS = {
    "camera": ("Camera/**", True),
    "masks": ("RGB_seg/masks/**", True),
    "whitebg": ("RGB_seg/whitebg/**", True),
    "rgb": ("RGB/**", False),
}


def resolve_revision(repo_id: str) -> dict:
    """Ask the Hub for the commit SHA and licence of ``main``."""
    import urllib.request
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    try:
        with urllib.request.urlopen(url, timeout=60) as handle:
            payload = json.load(handle)
    except Exception as exc:                              # noqa: BLE001
        return {"resolved": False, "error": f"{type(exc).__name__}: {exc}"}
    card = payload.get("cardData") or {}
    return {
        "resolved": True,
        "sha": payload.get("sha"),
        "last_modified": payload.get("lastModified"),
        "private": payload.get("private"),
        "gated": payload.get("gated"),
        "license": card.get("license"),
        "pretty_name": card.get("pretty_name"),
        "tags": card.get("tags"),
    }


def list_sample_files(crop: str, acquisition: str, revision: str | None,
                      timeout: int = 60) -> list:
    """Enumerate the sample's files through the paginated tree API."""
    import urllib.request
    prefix = f"{crop}/{acquisition}"
    rev = revision or "main"
    api = (f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{rev}/"
           f"{prefix}?recursive=true")
    files, cursor = [], None
    while True:
        url = api + (f"&cursor={cursor}" if cursor else "")
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            page = json.load(handle)
            link = handle.headers.get("Link", "")
        files.extend(e for e in page if e.get("type") == "file")
        if 'rel="next"' not in link or "cursor=" not in link:
            break
        cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crop", required=True, help="e.g. Broccoli")
    ap.add_argument("--acquisition", required=True, help="e.g. 20260327-10")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--include-rgb", action="store_true",
                    help="also fetch RGB/ (raw photographs; not the COLMAP frames)")
    ap.add_argument("--skip-whitebg", action="store_true")
    ap.add_argument("--revision", default=None, help="branch/tag/sha (default: main)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    prefix = f"{args.crop}/{args.acquisition}"
    wanted = ["camera", "masks"]
    if not args.skip_whitebg:
        wanted.append("whitebg")
    if args.include_rgb:
        wanted.append("rgb")
    patterns = [f"{prefix}/{GROUPS[key][0]}" for key in ("camera", "masks", "whitebg", "rgb")
                if key in wanted]

    started = time.time()
    print(f"repo      {REPO_ID}")
    print(f"sample    {prefix}")
    print(f"patterns  {patterns}")
    path = snapshot_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, revision=args.revision,
        allow_patterns=patterns, local_dir=str(args.root),
        max_workers=args.workers,
    )
    elapsed = time.time() - started
    print(f"downloaded to {path} in {elapsed:.1f}s")

    sample_dir = Path(path) / prefix
    listing = {}
    for key, (pattern, _) in GROUPS.items():
        pattern_dir = sample_dir / pattern.replace("/**", "")
        listing[key] = sorted(p.name for p in pattern_dir.glob("*")) if pattern_dir.is_dir() else []
    counts = {k: len(v) for k, v in listing.items()}
    print(f"files on disk: {counts}")

    provenance = {
        "dataset_name": "PAA-Plant-Dataset",
        "source_url": f"https://huggingface.co/datasets/{REPO_ID}",
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "requested_revision": args.revision or "main",
        "crop": args.crop,
        "plant_id": f"{args.crop}/{args.acquisition}",
        "acquisition_id": args.acquisition,
        "acquisition_date": args.acquisition.split("-")[0],
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "download_seconds": round(elapsed, 1),
        "include_rgb": args.include_rgb,
        "file_counts": counts,
        "sample_dir": str(sample_dir.resolve()),
        "hub": resolve_revision(REPO_ID),
    }
    out = sample_dir / "provenance.json"
    out.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in provenance.items()
                      if k in ("hub", "file_counts", "plant_id")}, indent=2))


if __name__ == "__main__":
    main()
