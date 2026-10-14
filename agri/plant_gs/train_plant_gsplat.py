#!/usr/bin/env python
"""Object-centric 3D Gaussian Splatting with gsplat, on masked plant captures.

Why this is not just gsplat's example trainer
---------------------------------------------
The capture is a handheld orbit of ONE plant in a cluttered room. COLMAP recovers
poses from the room, but the deliverable is a Gaussian cloud of the PLANT only.
That changes two things:

1. **Supervision is masked.** The loss is computed only inside the SAM mask, and
   the sparse initialisation is filtered to points that project inside the mask.
   Without this, background Gaussians land in masked-out pixels, receive no
   gradient, survive training untouched, and pollute the exported PLY.

2. **The export is filtered again.** After training, every Gaussian is scored by
   what fraction of the views it could be observed in actually see it inside the
   mask. Background floaters score ~0 and are dropped; plant Gaussians score ~1.

Everything else follows the standard 3DGS recipe (SH warm-up, L1 + SSIM, and
gsplat's DefaultStrategy for densification) so the result is a plain, portable
3DGS PLY: x/y/z, f_dc_*, f_rest_*, opacity, scale_* (log), rot_*.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from PIL import Image

C0 = 0.28209479177387814


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0


def knn_mean_dist(points, k=4, chunk=4096):
    """Mean distance to the k nearest neighbours, for initial scale."""
    n = points.shape[0]
    out = torch.empty(n, device=points.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(points[s:e], points)
        out[s:e] = d.topk(k + 1, largest=False).values[:, 1:].mean(dim=1)
    return out.clamp_min(1e-7)


def _gauss_window(ws, sigma, channels, device, dtype):
    c = torch.arange(ws, dtype=dtype, device=device) - (ws - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2 = g[:, None] @ g[None, :]
    return w2.expand(channels, 1, ws, ws).contiguous()


def ssim(img1, img2, window_size=11, sigma=1.5):
    """SSIM on [B,C,H,W] in [0,1], 'valid' convolution."""
    b, c, h, w = img1.shape
    win = _gauss_window(window_size, sigma, c, img1.device, img1.dtype)
    pad = 0  # valid
    mu1 = F.conv2d(img1, win, padding=pad, groups=c)
    mu2 = F.conv2d(img2, win, padding=pad, groups=c)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s11 = F.conv2d(img1 * img1, win, padding=pad, groups=c) - mu1s
    s22 = F.conv2d(img2 * img2, win, padding=pad, groups=c) - mu2s
    s12 = F.conv2d(img1 * img2, win, padding=pad, groups=c) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s11 + s22 + C2))
    return m.mean()


def psnr(a, b, mask=None):
    if mask is not None:
        m = mask
        mse = (((a - b) ** 2) * m).sum() / m.sum().clamp_min(1.0)
    else:
        mse = ((a - b) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Captures:
    def __init__(self, root, device, test_every=8, min_sharp_pct=0.0):
        self.device = device
        with open(os.path.join(root, "frames.json")) as f:
            frames = json.load(f)

        # Motion-blur filtering. The capture is a handheld orbit, and ~20% of the
        # frames are blurred enough that the variometer disagrees with sharp
        # frames of the same surface. Those frames cannot be fit jointly with
        # the sharp ones, and they drag validation PSNR down. COLMAP still used
        # all frames for poses -- for geometry, more views help -- this only
        # governs which frames supervise appearance.
        self.sharpness = None
        if min_sharp_pct > 0:
            sp_path = os.path.join(root, "sharpness.json")
            with open(sp_path) as f:
                blob = json.load(f)
            table = blob["sharpness"]
            vals = np.array(list(table.values()))
            thr = float(np.percentile(vals, min_sharp_pct))
            keep = [f for f in frames if table.get(os.path.basename(f["image"]), 0) >= thr]
            self.sharpness = {"threshold": thr, "dropped": len(frames) - len(keep)}
            if len(keep) < 30:
                print("WARNING: sharpness filter would keep only %d frames; ignoring"
                      % len(keep), flush=True)
            else:
                print("sharpness filter: drop %d/%d frames below Laplacian var %.1f"
                      % (len(frames) - len(keep), len(frames), thr), flush=True)
                frames = keep

        pts = np.load(os.path.join(root, "points3D_plant.npz"))
        xyz, rgb = pts["xyz"], pts["rgb"]

        c2w = np.stack([np.array(f["camtoworld"]) for f in frames])
        cam_centers = c2w[:, :3, 3]

        # --- normalise: centre on the cameras, scale so the cloud is ~unit ----
        # Cameras rather than points: the points are the plant, the cameras are
        # what must stay outside it, and the cameras define the orbit radius.
        self.center = cam_centers.mean(axis=0)
        dists = np.linalg.norm(cam_centers - self.center, axis=1)
        self.radius = float(dists.max())
        self.scale = 1.0 / self.radius
        # gsplat measures scene_scale AFTER normalisation (its Parser computes
        # max camera distance from the already-transformed cameras), so in a
        # unit-normalised scene this is ~1.0. Passing the world-space radius
        # instead scales grow_scale3d / prune_scale3d by the same factor and
        # the scene over-densifies by an order of magnitude.
        self.scene_scale = float(self.radius * self.scale)  # == 1.0

        T = np.eye(4)
        T[:3, :3] *= self.scale
        T[:3, 3] = -self.scale * self.center
        self.transform = T

        self.transform_ = T
        self.Ks = np.stack([np.array(f["K"]) for f in frames])
        self.camtoworlds = T[None] @ c2w
        self.width = int(frames[0]["width"])
        self.height = int(frames[0]["height"])

        self.points = (xyz - self.center) * self.scale
        self.points_rgb = rgb / 255.0

        self.frames = frames
        self.train_idx = [i for i in range(len(frames)) if i % test_every != 0]
        self.test_idx = [i for i in range(len(frames)) if i % test_every == 0]

        self.images, self.masks = [], []
        for f in frames:
            self.images.append(np.asarray(Image.open(f["image"]).convert("RGB"), dtype=np.uint8))
            self.masks.append(np.asarray(Image.open(f["mask"]).convert("L")) > 127)
        self.images_t = torch.from_numpy(np.stack(self.images)).to(device)
        self.masks_t = torch.from_numpy(np.stack(self.masks)).to(device)

    def batch(self, idx):
        return (
            self.images_t[idx].float() / 255.0,
            self.masks_t[idx].unsqueeze(-1),
            torch.from_numpy(self.Ks[idx].astype(np.float32)).to(self.device),
            torch.from_numpy(self.camtoworlds[idx].astype(np.float32)).to(self.device),
        )


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def init_splats(points, rgbs, scene_scale, cfg, device):
    n = points.shape[0]
    dist = knn_mean_dist(points, 4)
    # Cap the initial size. Scale is seeded from the 3-NN distance, so an
    # isolated point (the tail of a sparse SfM cloud) is handed a splat metres
    # across, which smears the whole frame. Capping at a robust multiple of the
    # median spacing keeps that tail from ever being able to cover the scene.
    med = dist.median()
    cap = med * cfg.max_init_scale_mult
    clipped = int((dist > cap).sum())
    if clipped:
        print("init scale cap: %d/%d points clamped to %.4g (median nn %.4g)"
              % (clipped, n, cap, med), flush=True)
    scales = torch.log((dist * cfg.init_scale).clamp(max=cap)).unsqueeze(-1).repeat(1, 3)
    quats = torch.rand((n, 4), device=device)
    opac = torch.logit(torch.full((n,), cfg.init_opacity, device=device))
    shs = torch.zeros((n, (cfg.sh_degree + 1) ** 2, 3), device=device)
    shs[:, 0, :] = rgb_to_sh(rgbs)

    params = {
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opac),
        "sh0": torch.nn.Parameter(shs[:, :1, :].contiguous()),
        "shN": torch.nn.Parameter(shs[:, 1:, :].contiguous()),
    }
    lrs = {
        "means": cfg.means_lr * scene_scale,
        "scales": cfg.scales_lr,
        "quats": cfg.quats_lr,
        "opacities": cfg.opacities_lr,
        "sh0": cfg.sh0_lr,
        "shN": cfg.shN_lr,
    }
    params = torch.nn.ParameterDict(params)
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k]}], eps=1e-15)
            for k in params}
    return params, opts


def render(splats, camtoworld, K, width, height, sh_degree, strategy, bg=None,
           render_mode="RGB"):
    viewmats = torch.inverse(camtoworld)
    colors, alphas, info = rasterization(
        means=splats["means"],
        quats=F.normalize(splats["quats"], dim=-1),
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
        viewmats=viewmats,
        Ks=K,
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
        absgrad=strategy.absgrad,
        backgrounds=bg,
        render_mode=render_mode,
    )
    return colors, alphas, info


# --------------------------------------------------------------------------- #
# post-training multi-view mask filter
# --------------------------------------------------------------------------- #
def mask_consistency(splats, data, chunk=8192):
    """Fraction of observable views that see each Gaussian inside the mask."""
    means = splats["means"].detach()
    n = means.shape[0]
    K = torch.from_numpy(data.Ks.astype(np.float32)).to(data.device)
    w2c = torch.from_numpy(np.linalg.inv(data.camtoworlds).astype(np.float32)).to(data.device)
    masks = data.masks_t
    H, W = data.height, data.width

    hits = torch.zeros(n, device=data.device)
    seen = torch.zeros(n, device=data.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        p = means[s:e]
        for c in range(K.shape[0]):
            cam = p @ w2c[c][:3, :3].T + w2c[c][:3, 3]
            z = cam[:, 2]
            ok = z > 1e-4
            u = K[c][0, 0] * cam[:, 0] / z.clamp_min(1e-6) + K[c][0, 2]
            v = K[c][1, 1] * cam[:, 1] / z.clamp_min(1e-6) + K[c][1, 2]
            ui = u.long().clamp(0, W - 1)
            vi = v.long().clamp(0, H - 1)
            inside = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            hit = torch.zeros(e - s, dtype=torch.bool, device=data.device)
            idx = torch.nonzero(inside).squeeze(-1)
            if idx.numel():
                m = masks[c][vi[idx], ui[idx]]
                hit[idx] = m
            seen[s:e] += inside.float()
            hits[s:e] += hit.float()
    return (hits / seen.clamp_min(1e-6)).cpu().numpy(), seen.cpu().numpy()


def export_ply(splats, path):
    """Write a standard 3DGS PLY.

    The 3DGS convention stores the RAW optimisation parameters, not activated
    values: `opacity` is a logit, `scale_*` is a log-scale. Consumers apply
    sigmoid()/exp() themselves -- Gaussian-JEPA's own loader does exactly that
    in datasets/ShapeNet55Gaussian.py (`np_sigmoid(vertex["opacity"])`,
    `np.exp(scales)`). Passing sigmoid(opacity) or exp(scale) here would double
    apply the activation and silently produce a wrong model.
    Rotation is the exception: the raw quaternion is an unconstrained 4-vector,
    so it is normalised here. Every consumer we care about normalises again on
    load (Gaussian-JEPA divides by its norm), which is idempotent for an
    already-unit quaternion -- but a consumer that does NOT normalise would
    otherwise read a wrong rotation.
    """
    # gsplat's PLY writer emits whatever tensor it is handed.
    export_splats(
        means=splats["means"],
        scales=splats["scales"],                       # raw log-scale
        quats=F.normalize(splats["quats"], dim=-1),    # unit quaternion
        opacities=splats["opacities"],                 # raw logit
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=path,
    )


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/fj/AgriGaussian-JEPA/data/plant_gs/plant_001/dataset")
    p.add_argument("--result", default="/fj/AgriGaussian-JEPA/data/plant_gs/plant_001/results/train")
    p.add_argument("--max-steps", type=int, default=30_000)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--sh-interval", type=int, default=1000)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--min-sharp-pct", type=float, default=20.0,
                   help="drop frames whose Laplacian variance is below this "
                        "percentile (motion blur); 0 keeps every frame")
    p.add_argument("--mask-mode", choices=["none", "masked"], default="none",
                   help="'none' = standard 3DGS over the full frame (well-posed), "
                        "then extract the plant afterwards. 'masked' = restrict the "
                        "photometric loss to the silhouette (kept for comparison; "
                        "it is ill-posed and fills the silhouette with a blur)")
    p.add_argument("--ssim-lambda", type=float, default=0.2)
    p.add_argument("--bg-alpha-weight", type=float, default=1.0,
                   help="weight pushing accumulated alpha to zero outside the mask; "
                        "0 reproduces plain masked training (which leaves background haze)")
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-init-scale-mult", type=float, default=3.0,
                   help="cap initial scale at this multiple of the median 3-NN distance")
    p.add_argument("--init-opacity", type=float, default=0.1)
    p.add_argument("--means-lr", type=float, default=1.6e-4)
    p.add_argument("--scales-lr", type=float, default=5e-3)
    p.add_argument("--quats-lr", type=float, default=1e-3)
    p.add_argument("--opacities-lr", type=float, default=5e-2)
    p.add_argument("--sh0-lr", type=float, default=2.5e-3)
    p.add_argument("--shN-lr", type=float, default=2.5e-3 / 20)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--mask-consistency", type=float, default=0.5,
                   help="drop Gaussians seen inside the mask in fewer than this "
                        "fraction of their observable views")
    p.add_argument("--min-opacity", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--absgrad", action="store_true",
                   help="AbsGS-style absolute gradients for densification; "
                        "off by default (classic 3DGS settings, which "
                        "densify far less aggressively on this scene)")
    p.add_argument("--mask-weight", type=float, default=8.0,
                   help="loss weight inside the plant mask (1.0 = uniform)")
    cfg = p.parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda"
    os.makedirs(cfg.result, exist_ok=True)

    data = Captures(cfg.data, device, cfg.test_every, min_sharp_pct=cfg.min_sharp_pct)
    print("frames: %d train / %d val | res %dx%d | init points %d"
          % (len(data.train_idx), len(data.test_idx), data.width, data.height,
             data.points.shape[0]), flush=True)
    print("normalisation: center=%s radius=%.4f scale=%.4f"
          % (np.round(data.center, 4), data.radius, data.scale), flush=True)

    points = torch.from_numpy(data.points.astype(np.float32)).to(device)
    rgbs = torch.from_numpy(data.points_rgb.astype(np.float32)).to(device)
    splats, opts = init_splats(points, rgbs, data.scene_scale * 1.1, cfg, device)

    # Refinement schedule scales with the run so that short smoke runs still
    # exercise densification instead of silently skipping it.
    refine_start = max(100, min(500, cfg.max_steps // 20))
    refine_stop = max(refine_start + 1, min(15_000, cfg.max_steps // 2))
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=0.0008 if cfg.absgrad else 0.0002,
        grow_scale3d=0.01,
        grow_scale2d=0.05,
        prune_scale3d=0.1,
        prune_scale2d=0.15,
        refine_start_iter=refine_start,
        refine_stop_iter=refine_stop,
        refine_every=100,
        reset_every=max(300, cfg.max_steps // 10),
        absgrad=cfg.absgrad,
    )
    strategy.check_sanity(splats, opts)
    state = strategy.initialize_state(scene_scale=data.scene_scale * 1.1)

    train_idx = np.array(data.train_idx)
    order = np.random.permutation(train_idx)
    cursor = 0
    best = {"psnr_full": -1.0, "psnr_masked": -1.0, "step": -1}
    log_path = os.path.join(cfg.result, "train_log.jsonl")
    t0 = time.time()

    for step in range(cfg.max_steps):
        if cursor >= len(order):
            order = np.random.permutation(train_idx)
            cursor = 0
        i = order[cursor]
        cursor += 1

        pixels, masks, K, camtoworld = data.batch(np.array([i]))
        sh_deg = min(step // cfg.sh_interval, cfg.sh_degree)

        colors, alphas, info = render(splats, camtoworld, K, data.width, data.height,
                                      sh_deg, strategy)
        strategy.step_pre_backward(params=splats, optimizers=opts, state=state,
                                   step=step, info=info)

        m = masks
        m3 = m.repeat(1, 1, 1, 3)
        if cfg.mask_mode == "masked":
            # Photometric terms restricted to the plant silhouette.
            l1 = ((colors - pixels).abs() * m3).sum() / m3.sum().clamp_min(1.0)
            s = ssim((colors * m3).permute(0, 3, 1, 2),
                     (pixels * m3).permute(0, 3, 1, 2))
        else:
            # Full-frame (well-posed) reconstruction, but with the plant
            # up-weighted. A plain global PSNR is dominated by the room, which
            # is ~3/4 of every frame, so the plant comes out under-fit -- and
            # the plant is the only part we keep. Weighting keeps the objective
            # well-posed while directing capacity where it matters.
            w3 = (1.0 + (cfg.mask_weight - 1.0) * m).repeat(1, 1, 1, 3)
            l1 = ((colors - pixels).abs() * w3).sum() / w3.sum().clamp_min(1.0)
            s = ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - s)

        if cfg.mask_mode == "masked" and cfg.bg_alpha_weight > 0:
            # Outside the silhouette the scene is known to be EMPTY, but a masked
            # photometric loss never says so: a splat out there gets no gradient
            # and survives at its birth opacity, hazing the render. Supervising
            # alpha towards zero explicitly is what clears the room.
            bg = (~m).float()
            loss = loss + cfg.bg_alpha_weight * (alphas * bg).sum() / bg.sum().clamp_min(1.0)
        loss.backward()

        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        strategy.step_post_backward(params=splats, optimizers=opts, state=state,
                                    step=step, info=info, packed=False)

        if step % 500 == 0 or step == cfg.max_steps - 1:
            n = splats["means"].shape[0]
            rec = {"step": step, "loss": float(loss), "l1": float(l1),
                   "ssim": float(s), "n_gauss": int(n),
                   "sh_degree": int(sh_deg), "elapsed": round(time.time() - t0, 1)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("step %6d loss %.5f l1 %.5f ssim %.4f N %7d  %.0fs"
                  % (step, float(loss), float(l1), float(s), n, time.time() - t0),
                  flush=True)

        if step > 0 and (step % cfg.save_every == 0 or step == cfg.max_steps - 1):
            export_ply(splats, os.path.join(cfg.result, f"point_cloud_raw_{step:06d}.ply"))

        if step > 0 and (step % cfg.eval_every == 0 or step == cfg.max_steps - 1):
            m = render_eval(splats, data, strategy, cfg.sh_degree)
            print("  EVAL step %d: PSNR %.2f (masked) %.2f (full), N %d"
                  % (step, m["psnr_masked"], m["psnr_full"],
                     splats["means"].shape[0]), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "eval": m}) + "\n")
            if m["psnr_full"] > best["psnr_full"]:
                best = {"psnr_full": m["psnr_full"], "psnr_masked": m["psnr_masked"],
                        "step": step}
                torch.save({"splats": {k: v.detach().cpu() for k, v in splats.items()},
                            "step": step},
                           os.path.join(cfg.result, "splats_best.pt"))
                export_ply(splats, os.path.join(cfg.result, "point_cloud_best.ply"))
                save_eval_images(splats, data, strategy,
                                 os.path.join(cfg.result, "renders"), step,
                                 cfg.sh_degree)

    # ------------------------------------------------------------------ #
    # final: persist the trained model. Plant extraction is a separate step
    # (extract_plant.py) because it needs depth-aware visibility testing, which
    # is not something the training loop should be doing.
    # ------------------------------------------------------------------ #
    n_final = int(splats["means"].shape[0])
    print("final Gaussian count: %d" % n_final, flush=True)

    ckpt = {
        "splats": {k: v.detach().cpu() for k, v in splats.items()},
        "sh_degree": cfg.sh_degree,
        "mask_mode": cfg.mask_mode,
        "step": cfg.max_steps,
        "normalization": {"center": data.center.tolist(),
                          "radius": data.radius, "scale": data.scale},
    }
    torch.save(ckpt, os.path.join(cfg.result, "splats.pt"))

    scene_path = os.path.join(cfg.result, "point_cloud_scene.ply")
    export_ply(splats, scene_path)

    summary = {
        "scene_ply": scene_path,
        "checkpoint": os.path.join(cfg.result, "splats.pt"),
        "gaussians": n_final,
        "mask_mode": cfg.mask_mode,
        "best_eval_psnr_masked": best["psnr_masked"],
        "best_eval_psnr_full": best["psnr_full"],
        "best_eval_step": best["step"],
        "steps": cfg.max_steps,
        "n_train_frames": len(data.train_idx),
        "n_val_frames": len(data.test_idx),
        "normalization": {"center": data.center.tolist(),
                          "radius": data.radius, "scale": data.scale},
    }
    with open(os.path.join(cfg.result, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def render_eval(splats, data, strategy, sh_degree):
    vals_m, vals_f = [], []
    for i in data.test_idx:
        pixels, masks, K, c2w = data.batch(np.array([i]))
        colors, alphas, _ = render(splats, c2w, K, data.width, data.height,
                                   sh_degree, strategy)
        vals_m.append(psnr(colors, pixels, masks))
        vals_f.append(psnr(colors, pixels))
    return {"psnr_masked": float(np.mean(vals_m)), "psnr_full": float(np.mean(vals_f))}


def save_eval_images(splats, data, strategy, outdir, step, sh_degree, limit=4):
    os.makedirs(outdir, exist_ok=True)
    for k, i in enumerate(data.test_idx[:limit]):
        pixels, masks, K, c2w = data.batch(np.array([i]))
        colors, alphas, _ = render(splats, c2w, K, data.width, data.height,
                                   sh_degree, strategy)
        colors = colors.detach()
        comp = np.concatenate([
            (pixels[0].cpu().numpy() * 255).astype(np.uint8),
            (np.clip(colors[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8),
            (masks[0, ..., 0].cpu().numpy().astype(np.uint8) * 255)[..., None].repeat(3, -1),
        ], axis=1)
        Image.fromarray(comp).save(os.path.join(outdir, f"step{step:06d}_{i:04d}.jpg"),
                                  quality=90)


if __name__ == "__main__":
    main()