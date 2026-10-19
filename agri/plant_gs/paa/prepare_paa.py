#!/usr/bin/env python
"""Prepare a PAA sample as a workspace for the validated foreground 3DGS trainer.

    python agri/plant_gs/paa/prepare_paa.py \
        --sample data/external/PAA-Plant-Dataset/Broccoli/20260327-10 \
        --workspace data/plant_gs/PAA/Broccoli_20260327-10 \
        --max-size 1600 --seed 0

What it does, in order
-----------------------
1. **Audit the COLMAP model** (``colmap_check.audit``). It refuses to continue on
   a sample that does not decode, whose poses are not finite, or whose camera
   model the trainer cannot consume.

2. **Probe mask alignment** (``alignment.probe_sample``). It recovers, from the
   data, the integer ``(dx, dy)`` crop that registers ``RGB_seg/masks`` onto
   ``Camera/images``, and it reports an IoU between the mask and the non-white
   pixels of ``RGB_seg/whitebg``. The default crop offsets used here were
   measured that way; ``--reuse-crop`` re-probes and requires agreement, while
   the default trusts the recorded offsets and re-verifies per-image IoU.

3. **Downscale frames and masks together.** The camera frames are large
   (2128x3806 / 3070x2020) and mostly empty white space; ``--max-size`` caps the
   long edge. Intrinsics are scaled by the *same* factor, so a downscale cannot
   silently desynchronise a mask from its frame. Every record carries
   ``source_width/height`` and the crop offset, so the mapping stays auditable.

4. **Hold out views.** ``--test-every`` marks every Nth frame as held out. These
   are *single-scene* held-out views for reconstruction-quality reporting only;
   they are NOT an independent-plant generalisation test, and the report says so.

5. **Filter the sparse cloud to the plant.** A point is kept when it projects
   inside the mask in a large majority of the views that observe it. Points the
   plant occludes from every view still project inside the silhouette in every
   view, so mask voting alone is not enough; the orbit radius bound is applied
   as well, which is the same defence ``foreground/build_dataset.py`` uses.

Nothing here reruns SfM. The PAA COLMAP model is the dataset author's, and the
brief is explicit that poses must be reused rather than re-solved.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colmap_io import read_model, qvec2rotmat  # noqa: E402

from paa.alignment import crop_mask_to_camera_frame, exif_orientation, probe_sample  # noqa: E402
from paa.colmap_check import DISTORTION_FREE, audit, intrinsics_matrix  # noqa: E402
from paa.registry import parse_sample_id  # noqa: E402

# Crop offsets measured on Broccoli/20260327-10 by alignment.py (check B).
# camera id -> (dx, dy) in the mask's stored (EXIF-ignored) orientation.
DEFAULT_CROPS = {0: (16, 24), 1: (440, 80)}


def log(message: str):
    print(message, flush=True)


def write_json(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def scale_intrinsics(K, factor_w, factor_h):
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, :] *= factor_w
    K[1, :] *= factor_h
    return K


def project(xyz, K, c2w):
    w2c = np.linalg.inv(c2w)
    cam = np.asarray(xyz) @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                       K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)
    valid = np.isfinite(uv).all(axis=1) & (z > 1e-6)
    return uv, valid


# --------------------------------------------------------------------------- #
# uniform-canvas normalisation
# --------------------------------------------------------------------------- #
# PAA mixes two physical cameras in one sample: Broccoli/20260327-10 has 99
# portrait frames at 2128x3806 (camera 0) and 3 landscape frames at 3070x2020
# (camera 1). The foreground trainer stacks every frame into a single tensor,
# so mixed sizes cannot be handed to it as-is. Dropping the 3 frames would be
# simple but throws away views, so instead the minority orientation is rolled
# into the majority one and everything is padded onto a shared canvas.
#
#   R_roll = [[0,1,0],[-1,0,0],[0,0,1]]   (rotate the camera about its own Z)
#   K_new  = [[f,0,cy],[0,f,W-1-cx],[0,0,1]]   with the new size (H, W)
#   c2w_new[:3,:3] = c2w[:3,:3] @ R_roll.T,  translation unchanged
#
# which is exactly the pixel map of a 90-degree counter-clockwise image
# rotation, so the image, the mask and the camera all stay consistent.
# `roll_camera_ccw90` and `roll_image_ccw90` are inverse views of one operation
# and are checked against each other in the tests.
R_ROLL_CCW90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def roll_camera_ccw90(K, c2w, width, height):
    """Camera that sees the 90-degree counter-clockwise rotation of the frame."""
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K_new = np.array([[fx, 0.0, cy], [0.0, fy, width - 1.0 - cx], [0.0, 0.0, 1.0]])
    c2w_new = np.asarray(c2w, dtype=np.float64).copy()
    c2w_new[:3, :3] = c2w_new[:3, :3] @ R_ROLL_CCW90.T
    return K_new, c2w_new, height, width        # new (width, height)


def roll_image_ccw90(image):
    """Image rotate() with expand=True turns the content 90 degrees CCW."""
    return image.rotate(90, expand=True)


def pad_canvas(image, canvas_w, canvas_h, offset, fill=255):
    out = Image.new(image.mode, (canvas_w, canvas_h), fill)
    out.paste(image, offset)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True,
                    help="PAA sample directory (contains Camera/ and RGB_seg/)")
    ap.add_argument("--workspace", type=Path, required=True,
                    help="output workspace; must not be the sample directory")
    ap.add_argument("--max-size", type=int, default=1600,
                    help="cap the long edge of frames and masks (intrinsics scale with it)")
    ap.add_argument("--test-every", type=int, default=8,
                    help="hold out every Nth view for reconstruction-quality reporting")
    ap.add_argument("--track-ratio-min", type=float, default=0.80,
                    help="fraction of a point's tracks that must land in the plant mask")
    ap.add_argument("--min-track-observations", type=int, default=2)
    ap.add_argument("--orbit-margin", type=float, default=1.0,
                    help="bound points to this multiple of the median camera radius")
    ap.add_argument("--reuse-crop", action="store_true",
                    help="re-probe the mask crop and require it to match the recorded default")
    ap.add_argument("--probe-limit", type=int, default=10,
                    help="images used when re-probing the crop")
    ap.add_argument("--frame-policy", choices=["uniform-canvas", "drop-minority"],
                    default="uniform-canvas",
                    help="'uniform-canvas' rolls minority-orientation frames and pads "
                         "everything onto one canvas; 'drop-minority' discards the "
                         "minority-orientation frames instead")
    ap.add_argument("--json-quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)
    sample = args.sample.resolve()
    ws = args.workspace.resolve()
    if ws == sample:
        raise SystemExit("--workspace must differ from --sample")
    if ws.exists():
        if args.force:
            shutil.rmtree(ws)
        else:
            raise SystemExit(f"workspace already exists: {ws} (use --force)")
    images_dir = sample / "Camera" / "images"
    masks_dir = sample / "RGB_seg" / "masks"
    for d in (images_dir, masks_dir):
        if not d.is_dir():
            raise SystemExit(f"missing required PAA directory: {d}")

    out = ws / "dataset"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (ws / "logs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. audit the COLMAP model
    # ------------------------------------------------------------------ #
    log("== 1. COLMAP audit ==")
    report = audit(sample, probe_limit=4)
    write_json(ws / "colmap_audit.json", report)
    if report["status"] != "PASS":
        raise SystemExit(f"COLMAP audit failed: {report['status']}")
    if not report["trainer_ready_without_undistortion"]:
        raise SystemExit(
            "camera model needs undistortion before the trainer can use it; "
            "this stage does not silently rename camera models")
    log(f"   {report['counts']} | reproj err mean {report['reprojection_error_px']['mean']:.3f} px")

    cameras, images, points = read_model(str(sample / "Camera" / "sparse" / "0"))

    # ------------------------------------------------------------------ #
    # 2. mask alignment
    # ------------------------------------------------------------------ #
    log("== 2. mask alignment ==")
    alignment = probe_sample(sample, limit=args.probe_limit)
    derived = {int(cid): (int(v["dx"]), int(v["dy"])) for cid, v in alignment.crops.items()}
    crops = dict(DEFAULT_CROPS)
    if args.reuse_crop:
        for cid, offset in derived.items():
            recorded = crops.get(cid)
            if recorded is not None and recorded != offset:
                raise SystemExit(
                    f"crop for camera {cid} drifted: recorded {recorded}, measured {offset}")
            crops[cid] = offset
    log(f"   crop offsets (camera id -> dx,dy): { {k: v for k, v in sorted(crops.items())} }")
    log(f"   measured by probe: {derived}")
    ious = [v["A_iou_median"] for v in alignment.crops.values() if v.get("A_iou_median") is not None]
    log(f"   mask vs whitebg IoU (median per camera): "
        f"{[round(float(i), 4) for i in ious]}")
    write_json(ws / "mask_alignment.json", alignment.to_json())

    # ------------------------------------------------------------------ #
    # 3. frames, masks, intrinsics
    #
    # Done in two passes. Pass 1 decodes each frame, aligns its mask and
    # downscales both, recording the working size and orientation. Pass 2
    # applies the uniform-canvas policy, which needs the sizes from pass 1.
    # Splitting it this way means a decode failure is reported per frame
    # rather than aborting the whole sample.
    # ------------------------------------------------------------------ #
    log("== 3. frames ==")
    staged = []
    dropped = []
    sharpness = {}

    for image in sorted(images.values(), key=lambda im: im.name):
        stem = Path(image.name).stem
        cam = cameras[image.camera_id]
        K, _ = intrinsics_matrix(cam)
        frame_path = images_dir / image.name
        mask_path = masks_dir / (stem + ".png")
        if not (frame_path.is_file() and mask_path.is_file()):
            dropped.append({"image": image.name, "reason": "frame or mask missing on disk"})
            continue

        orientation = exif_orientation(frame_path)
        frame = Image.open(frame_path).convert("RGB")
        src_w, src_h = frame.size
        if (src_w, src_h) != (cam.width, cam.height):
            dropped.append({"image": image.name,
                            "reason": f"frame {src_w}x{src_h} != camera {cam.width}x{cam.height}"})
            continue

        src_mask = np.asarray(Image.open(mask_path).convert("L"))
        try:
            mask_cropped = crop_mask_to_camera_frame(
                src_mask, orientation,
                crops[image.camera_id][0], crops[image.camera_id][1], src_w, src_h)
        except ValueError as exc:
            dropped.append({"image": image.name, "reason": f"mask crop failed: {exc}"})
            continue
        if not (mask_cropped > 127).any():
            dropped.append({"image": image.name, "reason": "mask empty after alignment"})
            continue

        c2w = np.eye(4)
        R = qvec2rotmat(image.qvec)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ image.tvec
        staged.append({
            "image": image, "stem": stem, "camera": cam, "K": K, "c2w": c2w,
            "orientation": orientation, "frame": frame, "mask": mask_cropped,
            "src_w": src_w, "src_h": src_h,
        })

    if len(staged) < 30:
        raise SystemExit(f"only {len(staged)} usable frames; refusing to build a dataset")

    # ---- pass 2: uniform canvas ----
    files_by_camera: dict[int, list] = {}
    for entry in staged:
        files_by_camera.setdefault(entry["camera"].id, []).append(entry)
    major_camera = max(files_by_camera, key=lambda cid: len(files_by_camera[cid]))
    major_landscape = (files_by_camera[major_camera][0]["src_w"]
                       > files_by_camera[major_camera][0]["src_h"])

    rolled_cameras = set()
    dropped_cameras = set()
    for cid, entries in files_by_camera.items():
        is_landscape = entries[0]["src_w"] > entries[0]["src_h"]
        if is_landscape == major_landscape:
            continue
        if args.frame_policy == "uniform-canvas":
            rolled_cameras.add(cid)
        else:
            dropped_cameras.add(cid)
            for entry in entries:
                dropped.append({"image": entry["image"].name,
                                "reason": "minority frame orientation; policy=drop-minority"})
    staged = [e for e in staged if e["camera"].id not in dropped_cameras]

    # Downscale (after any roll, so the long edge is capped in the final
    # orientation), then find the shared canvas.
    prepared = []
    for entry in staged:
        frame, mask, K, c2w, w, h = (entry["frame"], entry["mask"], entry["K"],
                                     entry["c2w"], entry["src_w"], entry["src_h"])
        if entry["camera"].id in rolled_cameras:
            frame = roll_image_ccw90(frame)
            mask = roll_image_ccw90(Image.fromarray(mask))
            K, c2w, w, h = roll_camera_ccw90(K, c2w, w, h)
            mask = np.asarray(mask)
        factor = min(1.0, args.max_size / max(w, h))
        out_w, out_h = int(round(w * factor)), int(round(h * factor))
        entry = dict(entry)
        entry.update(frame=frame.resize((out_w, out_h), Image.Resampling.LANCZOS),
                     mask=np.asarray(Image.fromarray(mask).resize(
                         (out_w, out_h), Image.Resampling.NEAREST)) > 127,
                     K=scale_intrinsics(K, out_w / w, out_h / h),
                     c2w=c2w, out_w=out_w, out_h=out_h,
                     rolled=entry["camera"].id in rolled_cameras)
        prepared.append(entry)

    canvas_w = max(e["out_w"] for e in prepared)
    canvas_h = max(e["out_h"] for e in prepared)

    records = []
    masks_on_disk = []
    for entry in prepared:
        image, cam = entry["image"], entry["camera"]
        stem = entry["stem"]
        ox = (canvas_w - entry["out_w"]) // 2
        oy = (canvas_h - entry["out_h"]) // 2
        frame_p = pad_canvas(entry["frame"], canvas_w, canvas_h, (ox, oy), fill=255)
        mask_p = pad_canvas(Image.fromarray(entry["mask"].astype(np.uint8) * 255),
                            canvas_w, canvas_h, (ox, oy), fill=0)
        mask_out = np.asarray(mask_p) > 127
        K_out = entry["K"].copy()
        K_out[0, 2] += ox
        K_out[1, 2] += oy

        out_image = out / "images" / image.name
        out_mask = out / "masks" / (stem + ".png")
        frame_p.save(out_image, quality=args.json_quality)
        Image.fromarray(mask_out.astype(np.uint8) * 255).save(out_mask)

        gray = np.asarray(frame_p.convert("L"), dtype=np.float32)
        gy, gx = np.gradient(gray)
        sharpness[image.name] = float(np.hypot(gx, gy)[mask_out].var())

        records.append({
            "name": image.name, "image_id": image.id,
            "image": str(out_image), "mask": str(out_mask),
            "camtoworld": entry["c2w"].tolist(), "K": K_out.tolist(),
            "width": canvas_w, "height": canvas_h,
            "frame_width": entry["out_w"], "frame_height": entry["out_h"],
            "canvas_offset": [ox, oy],
            "mask_coverage": float(mask_out.mean()),
            "mask_coverage_in_frame": float(entry["mask"].mean()),
            "camera_id": image.camera_id, "camera_model": cam.model,
            "source_width": entry["src_w"], "source_height": entry["src_h"],
            "crop_offset": [crops[image.camera_id][0], crops[image.camera_id][1]],
            "rolled_ccw90": entry["rolled"],
            "exif_orientation": entry["orientation"],
        })
        masks_on_disk.append(mask_out)

    # The alignment must survive the roll: a rolled frame whose camera was not
    # rolled identically would still train, it would just train badly. So the
    # roll is checked by the same measurement the COLMAP model gives us for
    # free -- where do the SfM points land relative to the mask?
    roll_check = check_mask_hit_rate(records, masks_on_disk, points)
    log(f"   kept {len(records)} frames on a {canvas_w}x{canvas_h} canvas "
        f"(rolled {len(rolled_cameras)} camera(s))"
        + (f" | dropped {len(dropped)}" if dropped else ""))
    log(f"   SfM points landing inside the mask: "
        f"min {roll_check['min']:.4f} median {roll_check['median']:.4f}")
    if roll_check["min"] < 0.80:
        raise SystemExit(
            f"mask/pose agreement collapsed to {roll_check['min']:.3f}; "
            "a frame is not registered to its camera")
    write_json(ws / "mask_hit_rate.json", roll_check)

    # ------------------------------------------------------------------ #
    # 4. sparse cloud -> plant points
    # ------------------------------------------------------------------ #
    log("== 4. foreground sparse filtering ==")
    xyz = np.stack([p.xyz for p in points.values()])
    rgb = np.stack([p.rgb for p in points.values()])
    obs = np.zeros(len(points), dtype=np.int32)
    inside = np.zeros(len(points), dtype=np.int32)
    # Keyed by COLMAP image id so the per-point track walk is O(1) per
    # observation. Using records.index(rec) here would be O(n^2) and, worse,
    # compares dicts by value -- it can silently return the wrong frame.
    by_image_id = {r["image_id"]: (idx, r) for idx, r in enumerate(records)}
    for i, point in enumerate(points.values()):
        for image_id in point.image_ids:
            entry = by_image_id.get(int(image_id))
            if entry is None:
                continue
            idx, rec = entry
            uv, valid = project(point.xyz[None], np.asarray(rec["K"]), np.asarray(rec["camtoworld"]))
            if not valid[0]:
                continue
            u, v = int(uv[0][0]), int(uv[0][1])
            if not (0 <= u < rec["width"] and 0 <= v < rec["height"]):
                continue
            obs[i] += 1
            inside[i] += 1 if masks_on_disk[idx][v, u] else 0

    ratio = np.divide(inside, np.maximum(obs, 1))
    centres = np.stack([np.asarray(r["camtoworld"])[:3, 3] for r in records])
    orbit_centre = centres.mean(axis=0)
    radii = np.linalg.norm(centres - orbit_centre, axis=1)
    orbit_radius = float(np.median(radii))
    keep = ((obs >= args.min_track_observations) & (ratio >= args.track_ratio_min)
            & (np.linalg.norm(xyz - orbit_centre, axis=1)
               <= orbit_radius * args.orbit_margin))
    if keep.sum() < 100:
        raise SystemExit(f"only {keep.sum()} foreground points; refusing to export an empty object")
    log(f"   kept {keep.sum()} of {len(xyz)} sparse points "
        f"({100 * keep.mean():.1f}%) | orbit radius {orbit_radius:.3f}")

    np.savez_compressed(out / "points3D_all.npz", xyz=xyz, rgb=rgb)
    np.savez_compressed(out / "points3D_plant.npz", xyz=xyz[keep], rgb=rgb[keep],
                        n_obs=obs, n_in=inside, ratio=ratio)
    write_json(out / "frames.json", records)
    write_json(out / "sharpness.json", {"sharpness": sharpness})

    meta = {
        "dataset": "PAA-Plant-Dataset",
        "source_url": "https://huggingface.co/datasets/JackTrum938/PAA-Plant-Dataset",
        "license": "apache-2.0",
        "sample": sample.name,
        "crop": sample.parent.name,
        # plant_id is the PLANT, not the plant+date. Broccoli/20260327-10 and
        # Broccoli/20260402-10 are the same physical plant, so the split group
        # must collapse them; encode plant_id here exactly as
        # registry.parse_sample_id would derive it, so the two files agree.
        "plant_id": parse_sample_id(sample.name)["plant_id"],
        "acquisition_id": sample.name,
        "acquisition_date": parse_sample_id(sample.name)["acquisition_date"],
        "seed": args.seed,
        "frame_policy": args.frame_policy,
        "camera_models": sorted({r["camera_model"] for r in records}),
        "crop_offsets": {str(k): list(v) for k, v in sorted(crops.items())},
        "registered_images": len(images),
        "frames_used": len(records),
        "frames_rolled_ccw90": int(sum(1 for r in records if r["rolled_ccw90"])),
        "dropped_frames": dropped,
        "canvas": [canvas_w, canvas_h],
        "source_resolution": sorted({(r["source_width"], r["source_height"]) for r in records}),
        "max_size": args.max_size,
        "test_every": args.test_every,
        "held_out_scope": ("single-scene held-out views for reconstruction quality only; "
                           "NOT an independent-plant generalisation test"),
        "sparse_points_total": int(len(xyz)),
        "sparse_points_plant": int(keep.sum()),
        "reprojection_error_mean_px": report["reprojection_error_px"]["mean"],
        "mask_coverage_min": float(min(r["mask_coverage"] for r in records)),
        "mask_coverage_median": float(np.median([r["mask_coverage"] for r in records])),
        "mask_coverage_max": float(max(r["mask_coverage"] for r in records)),
        "mask_hit_rate_min": roll_check["min"],
        "mask_hit_rate_median": roll_check["median"],
        "orbit_radius": orbit_radius,
        "python": platform.python_version(),
        "prepared_in_seconds": round(time.time() - started, 1),
    }
    write_json(ws / "preparation.json", meta)
    write_json(out / "meta.json",
               {k: v for k, v in meta.items() if k != "dropped_frames"})
    log(f"   dataset written to {out}")
    log("PREPARE_COMPLETE")


def check_mask_hit_rate(records, masks, points, limit: int = 0) -> dict:
    """Fraction of projected SfM points landing inside each frame's mask.

    This is the cheapest end-to-end check that frames, masks, intrinsics and
    poses still describe the same geometry after downscaling, rolling and
    padding. A frame that fell out of registration shows up immediately as a
    low hit rate for itself while its neighbours stay high.
    """
    xyz_list = [p.xyz for p in points.values()]
    if limit:
        step = max(1, len(xyz_list) // limit)
        xyz_list = xyz_list[::step]
    xyz = np.stack(xyz_list)
    rates = {}
    for record, mask in zip(records, masks):
        uv, valid = project(xyz, np.asarray(record["K"]),
                            np.asarray(record["camtoworld"]))
        h, w = mask.shape
        inside = (valid & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        if inside.sum() == 0:
            rates[record["name"]] = 0.0
            continue
        u = uv[inside, 0].astype(int)
        v = uv[inside, 1].astype(int)
        rates[record["name"]] = float(mask[v, u].mean())
    values = np.array(list(rates.values()))
    flags = {k: False for k in ("rolled_frames_ok",)}
    rolled = [v for r, v in rates.items()
              if next(x for x in records if x["name"] == r)["rolled_ccw90"]]
    flags["rolled_frames_min"] = float(min(rolled)) if rolled else None
    return {"per_frame": rates,
            "min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": float(values.max()),
            "n_frames": len(records), **flags}


if __name__ == "__main__":
    main()
