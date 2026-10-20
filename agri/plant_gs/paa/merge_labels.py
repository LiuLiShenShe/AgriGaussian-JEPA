#!/usr/bin/env python
"""Merge two candidate label maps into one, conservatively.

Rule
----
* **plant** only where the two backends agree on plant;
* **pot / soil / support** taken from the zero-shot backend, which is the one
  that distinguishes the container classes at all (the ADE20K model has no
  class for a plant support stake);
* everything else is background.

The plant class is the one that decides what ends up in the plant-only asset,
so it is taken by *intersection*: a pixel enters the plant class only if two
independent labellings -- a supervised ADE20K segmenter and a zero-shot
SAM+CLIP classifier -- both call it plant. The cost of this rule is recall
(plant that one method misses is dropped); the benefit is that pot, soil and
support pixels labelled plant by one method do not leak into the asset.

The merge is written as a third label directory so that every later step --
the Gaussian vote, the plant-only export, the sampling analysis -- reads one
consistent set of maps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic import CLASS_NAMES, PLANT  # noqa: E402


def merge(plant_dir: Path, other_dir: Path, out: Path) -> dict:
    """Merge every frame in ``plant_dir``/``other_dir`` into ``out``."""
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    stats = {"frames": 0, "plant_both": 0, "plant_only_plant_dir": 0,
             "plant_only_other_dir": 0, "disagreement": 0}
    per_frame = {}
    for path in sorted(plant_dir.glob("*.png")):
        other = other_dir / path.name
        if not other.is_file():
            continue
        a = np.asarray(Image.open(path))
        b = np.asarray(Image.open(other))
        plant_a, plant_b = a == PLANT, b == PLANT
        merged = b.copy()
        merged[plant_a & plant_b] = PLANT
        merged[plant_a & ~plant_b] = 0
        merged[~plant_a & plant_b] = 0
        Image.fromarray(merged.astype(np.uint8)).save(out / path.name)
        stats["frames"] += 1
        stats["plant_both"] += int((plant_a & plant_b).sum())
        stats["plant_only_plant_dir"] += int((plant_a & ~plant_b).sum())
        stats["plant_only_other_dir"] += int((plant_b & ~plant_a).sum())
        stats["disagreement"] += int((a != b).sum())
        per_frame[path.stem] = {
            "plant_both": int((plant_a & plant_b).sum()),
            "plant_only_plant_dir": int((plant_a & ~plant_b).sum()),
            "plant_only_other_dir": int((plant_b & ~plant_a).sum()),
            "pixels": int(a.size),
        }
    return {"totals": stats, "per_frame": per_frame}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant-dir", type=Path, required=True,
                    help="label maps whose PLANT class is the one both must agree on")
    ap.add_argument("--other-dir", type=Path, required=True,
                    help="label maps supplying pot / soil / support")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged = merge(args.plant_dir, args.other_dir, args.out)
    stats = merged["totals"]
    summary = {
        "status": "PASS",
        "plant_dir": str(args.plant_dir.resolve()),
        "other_dir": str(args.other_dir.resolve()),
        "out": str(args.out.resolve()),
        "rule": ("plant where both backends agree plant; pot/soil/support from the "
                 "other backend; everything else background"),
        "totals": stats,
        "plant_agreement_fraction": (stats["plant_both"] /
                                      max(stats["plant_both"] + stats["plant_only_plant_dir"]
                                          + stats["plant_only_other_dir"], 1)),
        # `per_frame` lives in merge()'s return value; referencing the local
        # name here raised NameError after the label PNGs had already been
        # written, which left a merged directory with no summary and looked
        # like "the merge did not run" to every later reader.
        "per_frame": merged["per_frame"],
    }
    (args.out.parent / "merge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
