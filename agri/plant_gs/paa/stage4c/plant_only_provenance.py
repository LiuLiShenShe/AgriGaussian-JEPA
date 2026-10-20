#!/usr/bin/env python
"""Provenance for the *plant-only* Gaussian asset (source PLY -> carved PLY).

Why a second chain
------------------
The Stage 4B-Fix chain is ``checkpoint -> full PLY``. The plant-only asset is
a **different PLY**: a subset of the full asset's rows, selected by a
visibility-weighted semantic vote. If that carve is not itself traceable, the
plant-only file is an orphan in exactly the way the checkpoint->PLY chain was
an orphan before: it loads, it renders, and nothing on disk says which rows of
which asset it came from.

So the chain is ``checkpoint -> full PLY -> plant-only PLY`` and the middle
link is the *keep mask*. The verification is the same idea as the full chain's,
one level down:

* the source PLY's bytes must still hash to the recorded value;
* the keep mask must still select the same rows (its digest is recorded);
* the carved PLY must have exactly as many rows as the mask keeps;
* every attribute of every carved row must equal the corresponding source row,
  compared **by index mapping** and not by position.

The last point is the one that catches real errors. The carved PLY is shorter
than the source, so a positional comparison would either be impossible or
would silently compare the wrong rows. A row-count check alone would pass for
a carve that kept the wrong rows -- which is precisely the failure mode that
produced the Cotton/1 incident.

Usage
-----
    python -m agri.plant_gs.paa.stage4c.plant_only_provenance \
        data/plant_gs/PAA/Cotton_1_repro_v1/plant_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: The attributes a plant-only PLY must reproduce from its source rows.
ATTRIBUTES = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ply(path: Path):
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"].data
    return vertex


def verify(directory: Path) -> dict:
    directory = Path(directory)
    problems = []
    report_path = directory / "plant_only_report.json"
    classes_path = directory / "gaussian_classes.npz"
    ply_path = directory / "plant_only.ply"
    for path in (report_path, classes_path, ply_path):
        if not path.is_file():
            return {"status": "fail", "problems": [f"missing {path}"],
                    "checked_utc": _utc(), "directory": str(directory.resolve())}
    report = json.loads(report_path.read_text())
    source_ply = report.get("source_ply")
    if not source_ply:
        problems.append("plant_only_report.json has no source_ply; the carve's "
                        "source cannot be identified")
        return {"status": "fail", "problems": problems, "checked_utc": _utc(),
                "directory": str(directory.resolve())}
    source_path = Path(source_ply)
    if not source_path.is_file():
        problems.append(f"source PLY missing: {source_path}")
        return {"status": "fail", "problems": problems, "checked_utc": _utc(),
                "directory": str(directory.resolve())}

    payload = np.load(classes_path)
    if "keep" not in payload:
        problems.append("gaussian_classes.npz has no 'keep' mask")
        return {"status": "fail", "problems": problems, "checked_utc": _utc(),
                "directory": str(directory.resolve())}
    keep = payload["keep"]
    retained = np.flatnonzero(keep).astype(np.int64)
    keep_digest = hashlib.sha256(retained.tobytes()).hexdigest()

    source_sha = sha256_file(source_path)
    recorded_source_sha = report.get("source_ply_sha256")
    if recorded_source_sha and recorded_source_sha != source_sha:
        problems.append("source PLY changed since the plant-only export "
                        f"({recorded_source_sha[:12]} -> {source_sha[:12]})")

    source = _read_ply(source_path)
    carved = _read_ply(ply_path)
    details = {
        "source_ply": str(source_path.resolve()),
        "source_ply_sha256": source_sha,
        "source_rows": int(len(source)),
        "plant_only_ply": str(ply_path.resolve()),
        "plant_only_ply_sha256": sha256_file(ply_path),
        "plant_only_rows": int(len(carved)),
        "keep_retained": int(len(retained)),
        "keep_index_sha256": keep_digest,
    }

    if len(carved) != len(retained):
        problems.append(f"plant-only PLY has {len(carved)} rows but the keep mask "
                        f"selects {len(retained)}")
    if len(source) != len(keep):
        problems.append(f"keep mask has {len(keep)} entries but the source PLY has "
                        f"{len(source)} rows")

    errors = {}
    if len(carved) == len(retained) and len(source) == len(keep):
        for name in ATTRIBUTES:
            if name not in source.dtype.names or name not in carved.dtype.names:
                problems.append(f"attribute {name} missing from one of the PLYs")
                continue
            expected = np.asarray(source[name])[retained]
            observed = np.asarray(carved[name])
            delta = float(np.abs(expected.astype(np.float64)
                                 - observed.astype(np.float64)).max()) if len(observed) else 0.0
            errors[name] = delta
            tolerance = 2e-6 if observed.dtype.itemsize == 4 else 1e-9
            if delta > tolerance:
                problems.append(f"attribute {name}: max |delta| = {delta:.3e}")
    details["max_abs_error"] = errors

    result = {
        "status": "pass" if not problems else "fail",
        "problems": problems,
        "details": details,
        "checked_utc": _utc(),
        "directory": str(directory.resolve()),
        "scope": ("this proves the plant-only PLY is the recorded subset of the "
                  "recorded source PLY. It does not prove the semantic vote was "
                  "correct -- that needs human labels, which is a separate state"),
    }
    (directory / "plant_only_provenance_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path,
                    help="a plant_only output directory (with plant_only.ply, "
                         "plant_only_report.json and gaussian_classes.npz)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    result = verify(args.directory)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
