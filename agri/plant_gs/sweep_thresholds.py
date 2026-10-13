#!/usr/bin/env python
"""Choose the extraction thresholds by measuring, not guessing.

Tight thresholds give a clean silhouette but drop the low-opacity Gaussians
that carry the plant's appearance, so the extracted model renders dim and dark.
Loose thresholds keep the look but let room Gaussians back in.

This sweeps the thresholds and scores each setting on two axes measured against
the training views:
  inside MAE   - how well the extracted plant reproduces the real plant
  outside ink  - how much non-plant content survives
Pick the knee, not the extreme.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_plant_gsplat as T  # noqa: E402
from extract_plant import _Dummy, visibility_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--ckpt", default="results/train/splats.pt")
    ap.add_argument("--min-sharp-pct", type=float, default=20.0)
    ap.add_argument("--views", type=int, default=8)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = T.Captures(a.data, device, test_every=10 ** 9, min_sharp_pct=a.min_sharp_pct)
    ck = torch.load(a.ckpt, map_location=device)
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device).float())
                                     for k, v in ck["splats"].items()})
    sh = ck.get("sh_degree", 3)
    print("%d frames, %d scene Gaussians" % (len(data.frames), splats["means"].shape[0]),
          flush=True)

    score, seen = visibility_scores(splats, data, sh_degree=sh)
    opac = torch.sigmoid(splats["opacities"].detach()).cpu().numpy()

    view_idx = list(range(0, len(data.frames), max(1, len(data.frames) // a.views)))[:a.views]
    cache = [data.batch(np.array([i])) for i in view_idx]

    rows = []
    for min_score in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for min_op in [0.01, 0.03, 0.05]:
            for min_vis in [2, 3]:
                keep = (score >= min_score) & (seen >= min_vis) & (opac >= min_op)
                if keep.sum() < 500:
                    continue
                sub = {k: v.detach()[torch.from_numpy(keep).to(v.device)].contiguous()
                       for k, v in splats.items()}
                maes, inks = [], []
                with torch.no_grad():
                    for (pixels, masks, K, c2w) in cache:
                        colors, _, _ = T.render(sub, c2w, K, data.width, data.height,
                                                sh, _Dummy())
                        ren = colors[0].cpu().numpy()
                        tl = pixels[0].cpu().numpy()
                        m = masks[0, ..., 0].cpu().numpy().astype(bool)
                        maes.append(float(np.abs(ren[m] - tl[m]).mean() * 255))
                        inks.append(float((ren[~m].max(axis=1) > 12 / 255).mean()))
                rows.append({"min_score": min_score, "min_opacity": min_op,
                             "min_visible": min_vis, "kept": int(keep.sum()),
                             "inside_MAE": round(float(np.mean(maes)), 2),
                             "outside_ink": round(float(np.mean(inks)), 4)})
                print(json.dumps(rows[-1]), flush=True)

    with open("dataset/threshold_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n--- best inside MAE with outside ink < 0.05 ---")
    ok = [r for r in rows if r["outside_ink"] < 0.05]
    for r in sorted(ok, key=lambda r: r["inside_MAE"])[:6]:
        print(json.dumps(r))


if __name__ == "__main__":
    main()