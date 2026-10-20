#!/usr/bin/env python
"""Build a plant-only training dataset from a prepared PAA dataset.

The prepared dataset's masks cover the whole specimen (plant + pot + support).
Re-optimising 3DGS under a plant-only mask needs a dataset whose masks are the
plant only, so this writes a parallel dataset: the same frames, the same
intrinsics, the same poses, and masks taken from the semantic label maps.

Images are symlinked, not copied -- a second copy of 102 JPEGs is 40 MB of
nothing, and the frames must be byte-identical to the original run for the
comparison to mean anything.

``--min-visible``-style thresholds do not apply here: the label map is already
the per-pixel decision. What *is* recorded is how much of each frame's original
specimen mask the plant mask covers, because a frame whose plant mask is tiny
contributes almost no supervision and would otherwise silently distort the
comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic import PLANT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-mask-coverage", type=float, default=0.005,
                    help="drop a frame whose plant mask covers less than this "
                         "fraction of the original specimen mask")
    ap.add_argument("--link-images", action="store_true", default=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(exist_ok=True)
    (args.out / "masks").mkdir(exist_ok=True)

    frames = json.loads((args.dataset / "frames.json").read_text())
    kept_frames, coverage, dropped = [], [], []
    for frame in frames:
        stem = Path(frame["name"]).stem
        label_path = args.labels / f"{stem}.png"
        mask_path = args.dataset / "masks" / f"{stem}.png"
        if not label_path.is_file() or not mask_path.is_file():
            dropped.append({"frame": frame["name"], "reason": "missing label or mask"})
            continue
        plant = np.asarray(Image.open(label_path)) == PLANT
        original = np.asarray(Image.open(mask_path).convert("L")) > 127
        share = float(plant.sum() / max(original.sum(), 1))
        if share < args.min_mask_coverage:
            dropped.append({"frame": frame["name"],
                             "reason": f"plant mask covers only {share:.4f} of the specimen mask"})
            continue
        Image.fromarray((plant * 255).astype(np.uint8)).save(
            args.out / "masks" / f"{stem}.png")
        target = args.out / "images" / frame["name"]
        if target.exists() or target.is_symlink():
            target.unlink()
        if args.link_images:
            target.symlink_to(Path(frame["image"]).resolve())
        else:
            shutil.copy2(frame["image"], target)
        coverage.append(share)
        kept_frames.append({**frame, "image": str(target.resolve()),
                            "mask": str((args.out / "masks" / f"{stem}.png").resolve())})

    (args.out / "frames.json").write_text(json.dumps(kept_frames, indent=2) + "\n")
    for name in ("points3D_plant.npz", "meta.json", "sharpness.json"):
        source = args.dataset / name
        if source.is_file():
            shutil.copy2(source, args.out / name)

    summary = {
        "status": "PASS",
        "source_dataset": str(args.dataset.resolve()),
        "labels": str(args.labels.resolve()),
        "frames_in": len(frames),
        "frames_kept": len(kept_frames),
        "frames_dropped": len(dropped),
        "dropped": dropped,
        "plant_mask_share_of_specimen": {
            "mean": float(np.mean(coverage)) if coverage else None,
            "min": float(np.min(coverage)) if coverage else None,
            "max": float(np.max(coverage)) if coverage else None,
        },
        "note": ("images are symlinked so the frames are byte-identical to the "
                 "original run; only the masks differ"),
    }
    (args.out / "plant_dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "dropped"}, indent=2))


if __name__ == "__main__":
    main()
