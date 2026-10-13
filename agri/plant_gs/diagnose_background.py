#!/usr/bin/env python
"""Diagnostic: where does the rendered background come from?

Renders (a) the freshly initialised model and (b) a trained checkpoint, then
reports how bright the render is OUTSIDE the plant mask. If (a) is already
bright, the initial point cloud is not plant-localised. If only (b) is bright,
training is spreading Gaussians into the room.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_plant_gsplat as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--ckpt", default=None, help="optional splats .pt to load")
    ap.add_argument("--views", type=int, default=6)
    a = ap.parse_args()

    data = T.Captures(a.data, "cuda", test_every=8)
    points = torch.from_numpy(data.points.astype(np.float32)).cuda()
    rgbs = torch.from_numpy(data.points_rgb.astype(np.float32)).cuda()

    class C:
        sh_degree = 3
        init_scale = 1.0
        init_opacity = 0.1
        max_init_scale_mult = 3.0
        means_lr = scales_lr = quats_lr = opacities_lr = sh0_lr = shN_lr = 0.0
    splats, _ = T.init_splats(points, rgbs, data.radius, C, "cuda")

    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cuda")
        for k, v in (sd.items() if "splats" not in sd else sd["splats"].items()):
            if k in splats:
                splats[k].data = v.to("cuda")
        print("loaded", a.ckpt)

    class S:
        absgrad = False
    strategy = S()

    print("N =", splats["means"].shape[0])
    tot_out, tot_in = [], []
    for i in data.test_idx[: a.views]:
        pixels, masks, K, c2w = data.batch(np.array([i]))
        colors, alphas, _ = T.render(splats, c2w, K, data.width, data.height, 3, strategy)
        ren = colors[0].detach().cpu().numpy()
        m = masks[0, ..., 0].cpu().numpy().astype(bool)
        lum = ren.mean(axis=2)
        tot_in.append(lum[m].mean() if m.any() else 0)
        tot_out.append(lum[~m].mean())
        comp = np.concatenate([
            (pixels[0].cpu().numpy() * 255).astype(np.uint8),
            (np.clip(ren, 0, 1) * 255).astype(np.uint8),
            (m[..., None].astype(np.uint8) * 255).repeat(3, -1),
        ], axis=1)
        Image.fromarray(comp).save(f"/tmp/agri_vis/diag_view{i:04d}.jpg", quality=90)
    print("render luma: inside mask %.1f | OUTSIDE mask %.1f  (outside must be ~0 for a clean plant-only model)"
          % (np.mean(tot_in), np.mean(tot_out)))

    # how many Gaussians project outside the mask across views?
    ratio, seen = T.mask_consistency(splats, data)
    print("mask consistency: median %.3f | frac >=0.5: %.3f | frac <0.2: %.3f"
          % (np.median(ratio), (ratio >= 0.5).mean(), (ratio < 0.2).mean()))


if __name__ == "__main__":
    main()