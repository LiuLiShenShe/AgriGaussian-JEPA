#!/usr/bin/env python
"""Associate per-view semantic labels with optimised 3D Gaussians, and build
a Plant-only Gaussian asset.

The problem
-----------
Stage 4A produced a *specimen* asset: plant + pot + support, because PAA's
masks cover all three. To get a plant-only asset the semantic decision has to
be lifted from pixels into 3D, and it cannot be lifted by projecting a mask:
a point on the pot that is hidden behind the plant in every view projects
inside the plant silhouette in every view. That is why the separation here is
**visibility-aware**, the same defence ``agri/plant_gs/extract_plant.py`` uses
for its SAM masks.

Method
------
1. **Render the depth** of the trained Gaussians for every view
   (``RGB+ED`` gives the front surface's expected depth).
2. **Project every Gaussian** into every view and keep only the observations
   where the Gaussian *is* the frontmost surface: its depth matches the
   rendered depth within a tolerance, and the rendered alpha there is solid.
   A Gaussian behind the plant is never frontmost, so it collects no evidence.
3. **Vote.** Each visible observation contributes the semantic class of the
   label map at the projected pixel. A Gaussian's class is the majority over
   its visible observations, and its confidence is that majority's share.

A Gaussian seen in 12 views therefore has 12 chances to be contradicted. A
one-frame mistake is outvoted; a systematic mistake in one view is outvoted by
the other eleven.

Outputs
-------
``plant_only.ply``              the retained Gaussians, standard 3DGS PLY
``gaussian_classes.npz``        per-Gaussian class, confidence, visible count
``plant_only_report.json``      retention, residue, counts, timings
``renders/``                    full-specimen vs plant-only comparisons
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
# The foreground trainer is a package that imports its siblings by bare module
# name (`from common import ...`), so its own directory has to be on the path.
sys.path.insert(0, str(ROOT / "agri" / "plant_gs"))
sys.path.insert(0, str(ROOT / "agri" / "plant_gs" / "paa"))
sys.path.insert(0, str(ROOT / "agri" / "plant_gs" / "foreground"))

from foreground.train import Captures, export_ply, render  # noqa: E402
from paa.semantic import BACKGROUND, CLASS_NAMES, PLANT, POT, SOIL, SUPPORT  # noqa: E402


def load_splats_from_ply(path: Path, device):
    """Rebuild the trainer's splat tensors from a standard 3DGS PLY.

    The PLY stores raw log-scale and logit-opacity, which is exactly what the
    renderer wants, so no activation is applied here. ``sh0``/``shN`` are split
    the way the trainer keeps them, and any ``f_rest_*`` beyond the requested
    degree is dropped -- the same thing ``verify_gsjepa_compatible`` reports.
    """
    from plyfile import PlyData
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names)
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    scales = np.stack([vertex[f"scale_{i}"] for i in range(3)],
                      axis=1).astype(np.float32)
    quats = np.stack([vertex[f"rot_{i}"] for i in range(4)],
                     axis=1).astype(np.float32)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
    # Numeric order, not `sorted()`: f_rest_10 would sort before f_rest_2 in a
    # lexicographic sort, and the coefficients would be interleaved wrongly.
    rest_names = sorted((n for n in names if n.startswith("f_rest_")),
                        key=lambda n: int(n.rsplit("_", 1)[-1]))
    if rest_names:
        # The 3DGS PLY stores the higher-order SH **channel-major**: the 15
        # coefficients of channel 0 are f_rest_0..f_rest_14, channel 1 is
        # f_rest_15..f_rest_29, and so on. So the flat array reshapes to
        # (N, 3, coefficients) and is transposed to the trainer's
        # (N, coefficients, 3). The other order is a silent corruption: the
        # model still loads and still renders, but with scrambled SH, which
        # costs PSNR for reasons that have nothing to do with the experiment.
        rest = np.stack([vertex[n] for n in rest_names], axis=1).astype(np.float32)
        rest = rest.reshape(len(means), 3, -1).transpose(0, 2, 1)
        degree = int(round(np.sqrt(rest.shape[1] + 1))) - 1
    else:
        rest = np.zeros((len(means), 0, 3), dtype=np.float32)
        degree = 0
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(means).to(device)),
        "scales": torch.nn.Parameter(torch.from_numpy(scales).to(device)),
        "quats": torch.nn.Parameter(torch.from_numpy(quats).to(device)),
        "opacities": torch.nn.Parameter(torch.from_numpy(opacity).to(device)),
        "sh0": torch.nn.Parameter(torch.from_numpy(dc).to(device)[:, None, :]),
        "shN": torch.nn.Parameter(torch.from_numpy(rest).to(device)),
    })
    return splats, degree, len(means)


class _Dummy:
    """Stand-in for the strategy object (render() only reads .absgrad)."""
    absgrad = False


@torch.no_grad()
def render_with_depth(splats, K, w2c, width, height, degree):
    """Rasterise and also return the front surface's expected depth.

    The foreground trainer's ``render`` does not expose a depth channel, and the
    visibility test needs one: a Gaussian counts as evidence only in the views
    where it *is* the frontmost surface. ``RGB+ED`` appends the expected depth
    to the colour buffer, which is what ``agri/plant_gs/extract_plant.py``
    already relies on.
    """
    from gsplat import rasterization
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    return rasterization(
        means=splats["means"], quats=splats["quats"],
        scales=splats["scales"].exp(), opacities=splats["opacities"].sigmoid(),
        colors=colors, viewmats=w2c, Ks=K, width=width, height=height,
        sh_degree=degree, near_plane=0.01, packed=False, absgrad=False,
        backgrounds=None, render_mode="RGB+ED")


def log(message: str) -> None:
    print(message, flush=True)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_empty_ply(path: Path) -> None:
    """A 0-row 3DGS PLY, so an empty asset is distinguishable from a missing one.

    A downstream tool that opens the file sees a valid header and zero
    vertices. Writing nothing at all would let a reader treat the failure as
    "the step never ran".
    """
    header = ["ply", "format binary_little_endian 1.0",
              "comment 3D Gaussians (empty carve: no Gaussian met the plant "
              "criteria)", "element vertex 0",
              "property float x", "property float y", "property float z",
              "property float opacity"]
    header += [f"property float scale_{i}" for i in range(3)]
    header += [f"property float rot_{i}" for i in range(4)]
    header += [f"property float f_dc_{i}" for i in range(3)]
    header += [f"property float f_rest_{i}" for i in range(45)]
    header += ["end_header"]
    Path(path).write_text("\n".join(header) + "\n")


def load_label_maps(directory: Path, frames, device) -> dict:
    """Load one label map per frame, as int16 tensors on the device."""
    maps = {}
    for frame in frames:
        stem = Path(frame["name"]).stem
        path = directory / f"{stem}.png"
        if not path.is_file():
            continue
        array = np.asarray(Image.open(path))
        maps[frame["name"]] = torch.from_numpy(array.astype(np.int16)).to(device)
    return maps


@torch.no_grad()
def accumulate_votes(splats, data, label_maps, sh_degree, depth_tol=0.10,
                      alpha_thresh=0.5, chunk=32768):
    """Per-Gaussian class votes over the views where the Gaussian is visible."""
    device = data.device
    means = splats["means"].detach()
    n = means.shape[0]
    K_all = data.Ks.to(torch.float32)
    w2c_all = data.w2cs.to(torch.float32)
    H, W = data.height, data.width

    votes = torch.zeros(n, 6, device=device)     # one column per class id
    seen = torch.zeros(n, device=device)

    for c in range(len(data.frames)):
        name = data.frames[c]["name"]
        label = label_maps.get(name)
        if label is None:
            continue
        out, alpha, _ = render_with_depth(
            splats, K_all[c:c + 1], w2c_all[c:c + 1], W, H, sh_degree)
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
            vis = in_frame & solid[vi, ui] & ((z - d_at).abs() <= depth_tol * z.clamp_min(1e-6))
            idx = torch.nonzero(vis).squeeze(-1)
            if idx.numel() == 0:
                continue
            seen[s:e] += vis.float()
            classes = label[vi[idx], ui[idx]].long()
            flat = (s + idx) * 6 + classes
            votes.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=votes.dtype))
    return votes, seen


def class_assignment(votes, seen, min_visible=3):
    """Majority class per Gaussian plus its confidence."""
    totals = votes.sum(dim=1)
    best = votes.argmax(dim=1)
    confidence = votes.gather(1, best[:, None]).squeeze(1) / totals.clamp_min(1e-6)
    valid = (seen >= min_visible) & (totals > 0)
    classes = best.clone()
    classes[~valid] = -1          # -1: not enough evidence to call it
    return classes, confidence, seen, valid


@torch.no_grad()
def evaluate_region_quality(splats, data, keep, label_maps, sh_degree, plant_id=PLANT):
    """Compare the full asset and the plant-only asset inside the plant region.

    Five numbers per held-out view, all measured against the *candidate* plant
    label map:

    * ``psnr_full`` / ``psnr_plant_only`` -- photometric error inside the plant
      region only, so the pot's pixels cannot flatter the plant-only render;
    * ``iou_full`` / ``iou_plant_only`` -- does the rendered silhouette agree
      with the plant region;
    * ``plant_coverage`` -- fraction of the plant region the plant-only render
      actually covers (a hole in the canopy shows up here);
    * ``residue_alpha`` -- mean alpha of the plant-only render *outside* the
      plant region. This is the non-plant residue: a plant-only asset that
      still paints the pot scores high here.
    """
    kept = {k: v.detach()[keep] for k, v in splats.items()}
    rows = []
    for i in data.test_idx:
        name = data.frames[i]["name"]
        label = label_maps.get(name)
        if label is None:
            continue
        rgb, mask, K, w2c = data.batch(i)
        target = rgb[0].cpu().numpy()
        specimen = mask[0, ..., 0].cpu().numpy() > 0.5
        region = (label.cpu().numpy() == plant_id) & specimen
        outside = specimen & ~region
        if region.sum() < 100:
            continue

        colour_full, alpha_full, _ = render(splats, K, w2c, data.width,
                                             data.height, sh_degree)
        colour_plant, alpha_plant, _ = render(kept, K, w2c, data.width,
                                               data.height, sh_degree)
        full = colour_full[0].clamp(0, 1).cpu().numpy()
        plant = colour_plant[0].clamp(0, 1).cpu().numpy()
        alpha_full = alpha_full[0, ..., 0].cpu().numpy()
        alpha_plant = alpha_plant[0, ..., 0].cpu().numpy()

        def psnr(image, selection):
            if selection.sum() == 0:
                return None
            mse = float(((image[selection] - target[selection]) ** 2).mean())
            return float(-10 * np.log10(max(mse, 1e-12)))

        def iou(prediction, selection):
            inter = float((prediction & selection).sum())
            union = float((prediction | selection).sum())
            return inter / union if union else None

        rows.append({
            "view": name,
            "psnr_full_in_plant": psnr(full, region),
            "psnr_plant_only_in_plant": psnr(plant, region),
            "psnr_full_in_specimen": psnr(full, specimen),
            "psnr_plant_only_in_specimen": psnr(plant, specimen),
            "iou_full_vs_plant": iou(alpha_full > 0.5, region),
            "iou_plant_only_vs_plant": iou(alpha_plant > 0.5, region),
            "plant_coverage": float(((alpha_plant > 0.5) & region).sum() / region.sum()),
            "residue_alpha_outside_plant": (float(alpha_plant[outside].mean())
                                            if outside.sum() else None),
            "residue_opaque_fraction_outside_plant": (
                float((alpha_plant[outside] > 0.5).mean()) if outside.sum() else None),
            "plant_region_pixels": int(region.sum()),
            "specimen_pixels": int(specimen.sum()),
        })
    if not rows:
        return {"views": 0}

    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "views": len(rows),
        "per_view": rows,
        "psnr_full_in_plant": mean("psnr_full_in_plant"),
        "psnr_plant_only_in_plant": mean("psnr_plant_only_in_plant"),
        "psnr_full_in_specimen": mean("psnr_full_in_specimen"),
        "psnr_plant_only_in_specimen": mean("psnr_plant_only_in_specimen"),
        "iou_full_vs_plant": mean("iou_full_vs_plant"),
        "iou_plant_only_vs_plant": mean("iou_plant_only_vs_plant"),
        "plant_coverage": mean("plant_coverage"),
        "residue_alpha_outside_plant": mean("residue_alpha_outside_plant"),
        "residue_opaque_fraction_outside_plant": mean(
            "residue_opaque_fraction_outside_plant"),
    }


@torch.no_grad()
def render_comparison(splats, data, keep_mask, sh_degree, out_dir, limit=6):
    """Render the full asset and the plant-only asset side by side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = {k: v.detach()[keep_mask] for k, v in splats.items()}
    panels = []
    indices = list(data.test_idx)
    for i in indices[:limit]:
        rgb, mask, K, w2c = data.batch(i)
        color_full, alpha_full, _ = render(splats, K, w2c, data.width, data.height,
                                            sh_degree)
        color_plant, alpha_plant, _ = render(kept, K, w2c, data.width, data.height,
                                              sh_degree)
        def to_image(t):
            return (t[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        full = to_image(color_full * mask)
        plant = to_image(color_plant * mask)
        m = (mask[0, ..., 0].cpu().numpy() * 255).astype(np.uint8)
        panel = np.concatenate([full, plant, np.repeat(m[..., None], 3, axis=2)], axis=1)
        Image.fromarray(panel).save(out_dir / f"view{i:04d}.jpg", quality=90)
        panels.append(panel)
    if panels:
        contact = np.concatenate(panels, axis=0)
        Image.fromarray(contact).save(out_dir / "contact_sheet.jpg", quality=90)
    return {"views_rendered": len(panels), "columns": ["full_specimen", "plant_only", "mask"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="trained splats checkpoint; not needed with --from-ply")
    ap.add_argument("--labels", type=Path, required=True,
                    help="directory of per-frame label PNGs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plant-source", default="labels_segformer",
                    help="label directory name to take the plant mask from")
    ap.add_argument("--eval-labels", type=Path, default=None,
                    help="label directory to score the result against; defaults to "
                         "--labels. Scoring two variants against their own masks "
                         "is not a comparison, so this exists to give them one "
                         "common yardstick.")
    ap.add_argument("--min-plant-fraction", type=float, default=0.5,
                    help="a Gaussian is plant if at least this share of its "
                         "visible observations are labelled plant")
    ap.add_argument("--min-visible", type=int, default=3)
    ap.add_argument("--depth-tol", type=float, default=0.10)
    ap.add_argument("--min-opacity", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--keep-all", action="store_true",
                    help="do not carve: evaluate the checkpoint's asset as-is. Used "
                         "to score a model that was *re-optimised* under plant-only "
                         "masks, so it is compared like for like with a carved one "
                         "instead of being carved a second time.")
    ap.add_argument("--from-ply", type=Path, default=None,
                    help="read the asset from a PLY instead of a checkpoint. The "
                         "exported PLY is the deliverable, and it is a cleaned "
                         "subset of the checkpoint, so class labels computed on the "
                         "checkpoint do not line up with it one-to-one. Working "
                         "from the PLY makes the labels and the analysed asset the "
                         "same set of Gaussians.")
    args = ap.parse_args()

    started = time.time()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    data = Captures(args.dataset, device=str(device), test_every=8, downscale=1)
    if args.from_ply is not None:
        splats, sh_degree, n = load_splats_from_ply(args.from_ply, device)
        log(f"asset {args.from_ply}: {n} Gaussians")
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt["splats"]
        splats = torch.nn.ParameterDict({
            k: torch.nn.Parameter(v.to(device).float()) for k, v in state.items()})
        sh_degree = ckpt.get("sh_degree") or 3
        n = splats["means"].shape[0]
        log(f"checkpoint {args.checkpoint}: {n} Gaussians, sh_degree={sh_degree}")

    labels = load_label_maps(args.labels, data.frames, device)
    log(f"loaded {len(labels)} label maps from {args.labels}")
    if not labels:
        raise SystemExit(f"no label maps found in {args.labels}")

    log("accumulating visibility-weighted semantic votes ...")
    votes, seen = accumulate_votes(splats, data, labels, sh_degree,
                                   depth_tol=args.depth_tol)
    classes, confidence, seen, valid = class_assignment(votes, seen, args.min_visible)
    total_votes = votes.sum(dim=1)
    plant_votes = votes[:, PLANT]
    plant_fraction = plant_votes / total_votes.clamp_min(1e-6)

    opacity = torch.sigmoid(splats["opacities"].detach())
    is_plant = ((classes == PLANT) & valid
                & (plant_fraction >= args.min_plant_fraction)
                & (opacity >= args.min_opacity))
    is_nonplant = ((classes != PLANT) & valid
                   & (plant_fraction < args.min_plant_fraction)
                   & (opacity >= args.min_opacity))
    undecided = ~(is_plant | is_nonplant)

    counts = {CLASS_NAMES[c]: int((classes == c).sum()) for c in CLASS_NAMES}
    counts["undecided"] = int(undecided.sum())
    log(f"Gaussian classes: {counts}")
    log(f"plant: {int(is_plant.sum())} | non-plant: {int(is_nonplant.sum())} | "
        f"undecided: {int(undecided.sum())}")

    keep = is_plant
    if args.keep_all:
        log("--keep-all: evaluating every Gaussian (no carving)")
        keep = torch.ones(n, dtype=torch.bool, device=device)
    metrics = {}
    eval_labels = labels
    if args.eval_labels is not None:
        eval_labels = load_label_maps(args.eval_labels, data.frames, device)
    log(f"evaluating full vs plant-only inside the plant region "
        f"(reference: {args.eval_labels or args.labels}) ...")
    # A carve that keeps nothing is a *result*, not a crash. Rendering an empty
    # Gaussian tensor made gsplat divide by zero and the process died with
    # SIGFPE, losing the report that would have explained why (Rice/1: both
    # backends labelled the stalks "support"/"soil", so no Gaussian had a plant
    # majority). Evaluate only when there is something to render, and record
    # the empty carve explicitly so it survives as evidence.
    if int(keep.sum()) > 0:
        metrics["region_quality"] = evaluate_region_quality(
            splats, data, keep, eval_labels, sh_degree, plant_id=PLANT)
        if not args.no_export:
            metrics["render_comparison"] = render_comparison(
                splats, data, keep, sh_degree, args.out / "renders")
    else:
        metrics["region_quality"] = {
            "views": 0,
            "reason": ("the carve retained no Gaussians, so there is nothing to "
                       "render; this is a semantic-labelling failure, not a "
                       "rendering failure"),
        }
        metrics["render_comparison"] = {"views_rendered": 0, "reason": "empty carve"}
    if not args.no_export:
        kept = {k: v.detach()[keep] for k, v in splats.items()}
        if int(kept["means"].shape[0]) == 0:
            # gsplat's exporter cannot write a 0-row PLY ("cannot reshape
            # tensor of 0 elements"), so an empty carve died here and took the
            # whole report with it. The empty carve is a legitimate outcome of
            # a labelling failure, so it is written as a header-only PLY and
            # the report explains why there are no rows.
            _write_empty_ply(args.out / "plant_only.ply")
            log("carve retained 0 Gaussians; wrote a header-only plant_only.ply "
                "so downstream tools can distinguish an empty asset from a "
                "missing one")
        else:
            export_ply(kept, args.out / "plant_only.ply")
            log(f"wrote {args.out / 'plant_only.ply'}")

    np.savez_compressed(
        args.out / "gaussian_classes.npz",
        classes=classes.cpu().numpy(), confidence=confidence.cpu().numpy(),
        seen=seen.cpu().numpy(), valid=valid.cpu().numpy(),
        plant_fraction=plant_fraction.cpu().numpy(),
        opacity=opacity.cpu().numpy(), keep=keep.cpu().numpy())

    report = {
        "status": "PASS",
        "source_checkpoint": str(args.checkpoint),
        # The carve is a *subset of rows* of a source asset. Naming that asset
        # and hashing its bytes is what lets stage4c/plant_only_provenance.py
        # check the subset against it later; without it the plant-only PLY is
        # an orphan that loads and renders perfectly.
        "source_ply": str(args.from_ply.resolve()) if args.from_ply else None,
        "source_ply_sha256": _sha256_file(args.from_ply) if args.from_ply else None,
        "labels": str(args.labels),
        "plant_source": args.plant_source,
        "source_gaussians": int(n),
        "plant_gaussians": int(is_plant.sum()),
        "nonplant_gaussians": int(is_nonplant.sum()),
        "undecided_gaussians": int(undecided.sum()),
        "plant_fraction_of_gaussians": float(is_plant.sum() / n),
        "class_counts": counts,
        "thresholds": {
            "min_plant_fraction": args.min_plant_fraction,
            "min_visible": args.min_visible,
            "depth_tol": args.depth_tol,
            "min_opacity": args.min_opacity,
        },
        "visibility": {
            "mean_visible_views_per_gaussian": float(seen.float().mean()),
            "gaussians_with_no_evidence": int((seen == 0).sum()),
            "label_vote_total": int(total_votes.sum()),
        },
        "retention": {
            "plant_kept_of_labelled_plant": float(
                is_plant.sum() / max(int((classes == PLANT).sum()), 1)),
            "nonplant_kept_of_labelled_nonplant": float(
                (is_plant & (classes != PLANT)).sum() / max(int(is_nonplant.sum()), 1)),
            "plant_gaussians_dropped_for_low_plant_fraction": int(
                ((classes == PLANT) & valid
                 & (plant_fraction < args.min_plant_fraction)).sum()),
            "note": ("'plant retention' is the share of Gaussians whose majority "
                     "visible label is plant that the plant-only asset keeps; "
                     "'non-plant residue' is the share of the kept asset whose "
                     "majority visible label is not plant. Both are measured "
                     "against the CANDIDATE labels, not ground truth."),
        },
        "region_quality": metrics.get("region_quality"),
        "renders": metrics.get("render_comparison"),
        "seconds": round(time.time() - started, 1),
        "scope": ("semantic separation is limited by the quality of the candidate "
                  "per-view labels; PAA has no plant/pot ground truth, so these are "
                  "consistency figures and not segmentation accuracy"),
    }
    (args.out / "plant_only_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(json.dumps({k: report[k] for k in
                    ("source_gaussians", "plant_gaussians", "nonplant_gaussians",
                     "undecided_gaussians", "plant_fraction_of_gaussians")}, indent=2))


if __name__ == "__main__":
    main()
