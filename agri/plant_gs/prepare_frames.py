#!/usr/bin/env python
"""Prepare frame set for COLMAP + gsplat.

- Verifies every source JPEG actually decodes (3 files in the upload are
  truncated/random bytes, not JPEGs, and cannot be recovered).
- Writes working-resolution images and the matching SAM masks.
- Masks only exist for frames 0000-0207, so those are the trainable frames.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

SRC_RGB = "/fj/AgriGaussian-JEPA/data/rgb/XianKeLai1"
SRC_SAM = "/fj/AgriGaussian-JEPA/data/sam/XianKeLai1"
WS = "/fj/AgriGaussian-JEPA/data/plant_gs/plant_001"
OUT_IMG = os.path.join(WS, "images")
OUT_MASK = os.path.join(WS, "masks")
SCALE = 0.5  # 2160x3840 -> 1080x1920

Image.MAX_IMAGE_PIXELS = None


def do_frame(i):
    name = "%04d" % i
    src = os.path.join(SRC_RGB, name + ".jpg")
    try:
        im = Image.open(src)
        im.load()
        im = im.convert("RGB")
    except Exception as e:
        return {"frame": i, "status": "unreadable", "error": f"{type(e).__name__}: {e}"}

    w, h = im.size
    im = im.resize((int(round(w * SCALE)), int(round(h * SCALE))), Image.LANCZOS)
    im.save(os.path.join(OUT_IMG, name + ".jpg"), quality=97, subsampling=0)

    rec = {"frame": i, "status": "ok", "width": im.size[0], "height": im.size[1],
           "has_mask": False}
    mpath = os.path.join(SRC_SAM, "mask_%04d.png" % i)
    if os.path.exists(mpath):
        m = Image.open(mpath).convert("L")
        m = m.resize(im.size, Image.LANCZOS)
        a = (np.asarray(m) > 127).astype(np.uint8) * 255
        Image.fromarray(a, mode="L").save(os.path.join(OUT_MASK, name + ".png"))
        rec["has_mask"] = True
        rec["mask_coverage"] = float((a > 0).mean())
    return rec


if __name__ == "__main__":
    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_MASK, exist_ok=True)
    n = len([f for f in os.listdir(SRC_RGB) if f.endswith(".jpg")])
    frames = sorted(int(f[:-4]) for f in os.listdir(SRC_RGB) if f.endswith(".jpg"))
    assert frames == list(range(len(frames))), "frame numbering is not contiguous"

    with ProcessPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(do_frame, frames))

    ok = [r for r in recs if r["status"] == "ok"]
    bad = [r for r in recs if r["status"] != "ok"]
    masked = [r for r in ok if r["has_mask"]]
    summary = {
        "source_frames": n,
        "decodable_frames": len(ok),
        "unreadable_frames": bad,
        "frames_with_masks": len(masked),
        "working_resolution": [ok[0]["width"], ok[0]["height"]],
        "scale": SCALE,
        "trainable_frames": [r["frame"] for r in masked],
        "colmap_frames": [r["frame"] for r in ok],
    }
    cov = [r["mask_coverage"] for r in masked]
    summary["mask_coverage_min"] = min(cov)
    summary["mask_coverage_max"] = max(cov)
    summary["mask_coverage_mean"] = float(np.mean(cov))

    with open(os.path.join(WS, "prepare_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("trainable_frames", "colmap_frames")}, indent=2))
    print("images:", len(os.listdir(OUT_IMG)), "masks:", len(os.listdir(OUT_MASK)))