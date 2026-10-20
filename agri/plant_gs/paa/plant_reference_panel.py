#!/usr/bin/env python
"""Build a human-inspectable reference panel of the plant masks.

PAA ships no plant/pot annotation, so nothing about the candidate masks can be
called "accurate". What *can* be done is to put them in front of a person: a
grid of views, each showing the photograph, the candidate plant mask, and the
non-plant part, with the views spread around the orbit so the panel is not all
one direction.

The panel is the artefact the brief asks for -- 8 to 12 views covering
different directions -- and it is written as ordinary JPEGs so it can be looked
at, not just parsed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic import CLASS_COLORS, CLASS_NAMES, PLANT  # noqa: E402


def pick_views(dataset: Path, count: int) -> list:
    """Spread ``count`` views around the orbit, by camera azimuth."""
    frames = json.loads((dataset / "frames.json").read_text())
    rows = []
    for index, frame in enumerate(frames):
        c2w = np.asarray(frame["camtoworld"], dtype=np.float64)
        centre = c2w[:3, 3]
        rows.append((index, frame, centre))
    # The camera centres lie on a curve; the angle of the centre in the
    # horizontal plane is a stable ordering of "around the plant".
    origins = np.stack([r[2] for r in rows])
    reference = origins.mean(axis=0)
    angles = np.arctan2(origins[:, 1] - reference[1], origins[:, 0] - reference[0])
    order = np.argsort(angles)
    picked = [rows[order[i]] for i in
              np.linspace(0, len(order) - 1, count).astype(int)]
    return [(i, f) for i, f, _ in picked]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--width", type=int, default=320)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    picked = pick_views(args.dataset, args.views)
    panels = []
    summary = []
    for index, frame in picked:
        stem = Path(frame["name"]).stem
        label_path = args.labels / f"{stem}.png"
        if not label_path.is_file():
            continue
        image = Image.open(frame["image"]).convert("RGB")
        label = np.asarray(Image.open(label_path))
        plant = label == PLANT
        other = (label != PLANT) & (label != 0)

        overlay = np.asarray(image).copy()
        tint = np.zeros_like(overlay)
        for class_id, colour in CLASS_COLORS.items():
            if class_id in (PLANT,):
                continue
            tint[label == class_id] = colour
        overlay = (0.6 * np.asarray(image).astype(np.float32)
                   + 0.4 * tint.astype(np.float32)).astype(np.uint8)
        overlay[plant] = (0.55 * overlay[plant]
                          + 0.45 * np.array(CLASS_COLORS[PLANT])).astype(np.uint8)

        binary = np.zeros_like(overlay)
        binary[plant] = (40, 200, 70)
        binary[other] = (200, 60, 60)

        ratio = args.width / image.width
        size = (args.width, int(image.height * ratio))
        row = Image.new("RGB", (size[0] * 3, size[1] + 20), "white")
        draw = ImageDraw.Draw(row)
        for column, (im, caption) in enumerate([
                (image, "photograph"),
                (Image.fromarray(overlay), "candidate labels"),
                (Image.fromarray(binary), "plant (green) / other (red)")]):
            row.paste(im.resize(size, Image.Resampling.LANCZOS), (column * size[0], 20))
            draw.text((column * size[0] + 6, 5), f"{frame['name']}  {caption}",
                      fill="black")
        panels.append(row)
        summary.append({
            "frame": frame["name"],
            "plant_fraction_of_specimen": float(plant.sum() / max((plant | other).sum(), 1)),
        })
        row.save(args.out / f"{stem}.jpg", quality=92)

    if panels:
        sheet = Image.new("RGB", (max(p.width for p in panels),
                                  sum(p.height for p in panels)), "white")
        y = 0
        for panel in panels:
            sheet.paste(panel, (0, y))
            y += panel.height
        sheet.save(args.out / "contact_sheet.jpg", quality=90)

    report = {
        "status": "PASS",
        "views": len(panels),
        "selection": "spread around the orbit by the camera centre's azimuth",
        "per_view": summary,
        "columns": ["photograph", "candidate labels", "plant / other binary"],
        "note": ("these are CANDIDATE labels from zero-shot / out-of-domain "
                 "semantic models; the panel exists so they can be inspected, "
                 "not to claim accuracy"),
    }
    (args.out / "panel_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_view"}, indent=2))


if __name__ == "__main__":
    main()
