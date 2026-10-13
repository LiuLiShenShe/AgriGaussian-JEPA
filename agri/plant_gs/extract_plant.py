#!/usr/bin/env python
"""Extract the plant from a full-scene 3DGS model, using the SAM masks.

Why a visibility test is essential
----------------------------------
The scene (plant + room) is reconstructed with a standard, well-posed 3DGS loss.
The plant is then carved out. The obvious test -- "does this Gaussian project
inside the plant silhouette?" -- does not work on its own, because a point on
the wall *behind* the plant projects inside the plant silhouette in every view
where the plant occludes it, and a full orbit means the plant hides it most of
the time. Those points therefore pass a naive mask test at ~100%.

What separates them is visibility. A Gaussian counts as evidence only in views
where it is the frontmost surface. A wall point is never frontmost (the plant is
in front of it), so it collects no in-mask evidence; a plant Gaussian is frontmost
from its own side and always lands inside the silhouette. Scores separate cleanly.

So per Gaussian we accumulate, over all views where it is visible and in frame:

    score = (# visible views landing inside the mask) / (# visible views)

and keep the Gaussians above a threshold.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_plant_gsplat as T  # noqa: E402


@torch.no_grad()
def visibility_scores(splats, data, sh_degree=3, depth_tol=0.10, alpha_thresh=0.5,
                      chunk=16384, views=None):
    """Per-Gaussian fraction of *visible* views that land inside the plant mask."""
    device = data.device
    means = splats["means"].detach()
    n = means.shape[0]
    K_all = torch.from_numpy(data.Ks.astype(np.float32)).to(device)
    w2c_all = torch.from_numpy(np.linalg.inv(data.camtoworlds).astype(np.float32)).to(device)
    masks = data.masks_t
    H, W = data.height, data.width

    hits = torch.zeros(n, device=device)
    seen = torch.zeros(n, device=device)

    idx_views = views if views is not None else range(len(data.frames))
    for c in idx_views:
        # Expected depth (alpha-weighted) is the front surface's depth, which is
        # what we compare a candidate Gaussian's depth against.
        out, alpha, _ = T.render(
            splats, torch.from_numpy(data.camtoworlds[c:c + 1].astype(np.float32)).to(device),
            K_all[c:c + 1], W, H, sh_degree, _Dummy(), render_mode="RGB+ED",
        )
        depths = out[0, :, :, 3]
        solid = alpha[0, :, :, 0] > alpha_thresh

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            p = means[s:e]
            cam = p @ w2c_all[c][:3, :3].T + w2c_all[c][:3, 3]
            z = cam[:, 2]
            in_front = z > 1e-4
            u = K_all[c][0, 0] * cam[:, 0] / z.clamp_min(1e-6) + K_all[c][0, 2]
            v = K_all[c][1, 1] * cam[:, 1] / z.clamp_min(1e-6) + K_all[c][1, 2]
            in_frame = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not bool(in_frame.any()):
                continue
            ui = u.long().clamp(0, W - 1)
            vi = v.long().clamp(0, H - 1)
            d_at = depths[vi, ui]
            # Frontmost surface test: this Gaussian must sit on the rendered
            # surface, not behind it.
            vis = in_frame & solid[vi, ui] & ((z - d_at).abs() <= depth_tol * z.clamp_min(1e-6))
            idx = torch.nonzero(vis).squeeze(-1)
            if idx.numel() == 0:
                continue
            seen[s:e] += vis.float()
            hits[s + idx] += masks[c][vi[idx], ui[idx]].float()

    return (hits / seen.clamp_min(1e-6)).cpu().numpy(), seen.cpu().numpy()


class _Dummy:
    """Stand-in for the strategy object (render() only reads .absgrad)."""
    absgrad = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--ckpt", default="results/train/splats.pt")
    ap.add_argument("--out", default="point_cloud.ply")
    ap.add_argument("--min-score", type=float, default=0.80)
    ap.add_argument("--min-visible", type=int, default=3)
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--depth-tol", type=float, default=0.10)
    ap.add_argument("--prune-floater-mult", type=float, default=4.0,
                   help="drop Gaussians whose 4-NN distance exceeds this "
                        "multiple of the median (isolated floater removal); "
                        "0 disables")
    ap.add_argument("--min-sharp-pct", type=float, default=20.0,
                   help="must match the value used for training")
    ap.add_argument("--render-check", default=None, help="dir for render sanity images")
    ap.add_argument("--max-radius", type=float, default=None,
                    help="optional hard cap on distance from the orbit centre")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use the SAME frame set as training: the blur filter decides which frames
    # were fitted, and the visibility test should be judged on those views.
    data = T.Captures(a.data, device, test_every=10 ** 9,
                     min_sharp_pct=a.min_sharp_pct)
    print("loaded %d frames" % len(data.frames), flush=True)

    ck = torch.load(a.ckpt, map_location=device)
    sd = ck["splats"]
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device).float())
                                     for k, v in sd.items()})
    n = splats["means"].shape[0]
    print("checkpoint %s: %d Gaussians, sh_degree=%s"
          % (a.ckpt, n, ck.get("sh_degree")), flush=True)

    print("computing visibility-aware mask scores...", flush=True)
    sh_deg = ck.get("sh_degree") or 3
    score, seen = visibility_scores(splats, data, sh_degree=sh_deg,
                                   depth_tol=a.depth_tol)
    opac = torch.sigmoid(splats["opacities"].detach()).cpu().numpy()
    scales = torch.exp(splats["scales"].detach()).cpu().numpy().max(axis=1)

    keep = (score >= a.min_score) & (seen >= a.min_visible) & (opac >= a.min_opacity)
    # Distances are measured in the NORMALISED frame the model was trained in:
    # the trainer maps the orbit centre to the origin and divides by `radius`,
    # so ||means|| * radius is the distance in world units.
    m_np = splats["means"].detach().cpu().numpy()
    scale = float(ck.get("normalization", {}).get("scale", data.scale))
    d_norm = np.linalg.norm(m_np, axis=1)          # in normalised units
    d_world = d_norm / scale                        # back to COLMAP units
    if a.max_radius is not None:
        keep &= d_world <= a.max_radius
    print("kept %d / %d (%.1f%%)" % (keep.sum(), n, 100.0 * keep.sum() / n), flush=True)
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        print("   score >= %.2f : %d" % (thr, int((score >= thr).sum())))

    # --- floater removal ------------------------------------------------------
    # A pure score threshold still leaves stray splats: optimiser debris that
    # happens to land in the silhouette in a couple of views, plus real room
    # Gaussians adjacent to the plant. Those are isolated in 3D, so drop them by
    # local density -- keep only Gaussians that have neighbours nearby.
    if a.prune_floater_mult > 0:
        from scipy.spatial import cKDTree
        pts = splats["means"].detach().cpu().numpy()[keep]
        tree = cKDTree(pts)
        d_nn, _ = tree.query(pts, k=min(5, len(pts)))
        nn = d_nn[:, -1]                       # distance to the 4th neighbour
        med = float(np.median(nn))
        cut = med * a.prune_floater_mult
        dense = nn <= cut
        print("floater prune: nn(4) median %.5f, cut %.5f -> drop %d isolated Gaussians"
              % (med, cut, int((~dense).sum())), flush=True)
        idx_kept = np.nonzero(keep)[0]
        keep = np.zeros(n, dtype=bool)
        keep[idx_kept[dense]] = True

    if keep.sum() < 500:
        print("WARNING: only %d Gaussians passed; relaxing min-score" % keep.sum(),
              flush=True)
        keep = (seen >= 2) & (opac >= a.min_opacity) & (score >= 0.5)

    kept = {k: v.detach()[torch.from_numpy(keep).to(v.device)].contiguous()
            for k, v in splats.items()}
    T.export_ply(kept, a.out)
    np.savez(os.path.splitext(a.out)[0] + "_filter_stats.npz",
             score=score, seen=seen, opacity=opac, keep=keep)

    d_all = d_world
    report = {
        "out": a.out,
        "source_checkpoint": a.ckpt,
        "sh_degree": sh_deg,
        "scene_gaussians": int(n),
        "kept_gaussians": int(keep.sum()),
        "removed": int((~keep).sum()),
        "thresholds": {"min_score": a.min_score, "min_visible": a.min_visible,
                       "min_opacity": a.min_opacity, "depth_tol": a.depth_tol},
        "world_units_per_normalised_unit": 1.0 / scale,
        "kept_dist_to_center_world": {
            "p50": float(np.percentile(d_all[keep], 50)),
            "p90": float(np.percentile(d_all[keep], 90)),
            "max": float(d_all[keep].max())},
        "removed_dist_to_center_world_p50": float(np.percentile(d_all[~keep], 50)),
        "score_histogram": {
            str(t): int((score >= t).sum()) for t in [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]},
    }
    if a.render_check:
        render_check(kept, data, sh_deg, a.render_check)
        report["render_check_dir"] = a.render_check
    with open(os.path.splitext(a.out)[0] + "_extract_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


@torch.no_grad()
def render_check(kept, data, sh_degree, outdir, limit=4):
    """Render the extracted model beside the target and the mask."""
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    for i in data.test_idx[:limit]:
        pixels, masks, K, c2w = data.batch(np.array([i]))
        colors, _, _ = T.render(kept, c2w, K, data.width, data.height,
                                sh_degree, _Dummy())
        m = masks[0, ..., 0].cpu().numpy().astype(bool)
        comp = np.concatenate([
            (pixels[0].cpu().numpy() * 255).astype(np.uint8),
            (np.clip(colors[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8),
            (m[..., None].astype(np.uint8) * 255).repeat(3, -1),
        ], axis=1)
        Image.fromarray(comp).save(os.path.join(outdir, "check_%04d.jpg" % i), quality=90)
    print("render check written to", outdir, flush=True)


if __name__ == "__main__":
    main()