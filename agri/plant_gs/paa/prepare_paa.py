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
   pixels of ``RGB_seg/whitebg``.

   Stage 4A measured the offsets on ``Broccoli/20260327-10`` and hard-coded them
   as :data:`DEFAULT_CROPS`. That is a cross-sample leak: a second plant whose
   camera framing shifted would have been silently registered with the first
   plant's offsets. **Every sample now measures its own offsets** and refuses to
   continue if the probe is inconclusive. ``--trust-default-crops`` re-enables
   the old behaviour for one named sample only, and only when the measurement
   confirms it (see :func:`resolve_crops`).

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

from paa.alignment import (exif_orientation, probe_sample,  # noqa: E402
                           _bbox_corner_error, _centroid_shift,
                           verify_alignment, warp_mask_to_camera)
from paa.colmap_check import DISTORTION_FREE, audit, intrinsics_matrix  # noqa: E402
from paa.registry import parse_sample_id  # noqa: E402

# Crop offsets measured on Broccoli/20260327-10 by alignment.py (check B).
# camera id -> (dx, dy) in the mask's stored (EXIF-ignored) orientation.
#
# These are NOT defaults any more. They are the Stage 4A reference values, kept
# so `--trust-default-crops` can prove a new sample agrees with them; every
# other run measures its own offsets. See `resolve_crops`.
STAGE4A_REFERENCE_CROPS = {0: (16, 24), 1: (440, 80)}
STAGE4A_REFERENCE_SAMPLE = "20260327-10"


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


# Minimum template-match confidence before a measured crop offset is trusted.
# Stage 4A measured NCC median 0.9930 (cam 0) / 0.9956 (cam 1) with min 0.9826 /
# 0.9883 across all 102 frames; 0.95 sits below every observed value with room
# for JPEG noise on a different plant, and far above coincidence on white
# studio backgrounds.
CROP_NCC_MIN = 0.95
CROP_MIN_SUPPORT_FRACTION = 0.6

# Acceptance gates for the Stage 4B similarity fit. Measured across the four
# samples in this stage: rms 0.06-1.67 px, inliers >= 1922, and a translation
# spread of < 0.5 px (cam 1: < 1.4 px) between frames of one camera. The limits
# sit well outside those with room for a new plant.
SIM_RMS_MAX_PX = 3.0
SIM_MIN_INLIERS = 200
SIM_TRANSLATION_SPREAD_MAX_PX = 6.0
SIM_SCALE_SPREAD_MAX = 0.01


def resolve_transforms(sample_name: str, alignment, trust_defaults: bool) -> dict:
    """Decide the per-camera mask-to-camera transforms, or refuse to continue.

    Returns ``{camera_id: fit}`` where ``fit`` is the similarity dict produced by
    ``alignment.match_similarity``.

    The transform binds a mask to a camera frame, so it is measured on *this*
    sample from *this* sample's data. Anything that makes a measurement
    untrustworthy raises rather than falling back:

    * no fit for a camera that has frames;
    * a fit with too few inliers or a residual above :data:`SIM_RMS_MAX_PX`;
    * a fit that disagrees with itself across the camera's frames (the
      translation or scale drifts), which means the correspondence is not a
      single rigid-ish transform.

    ``trust_defaults`` exists only to reproduce the Stage 4A sample, whose
    integer crop was already published. It still re-measures and requires
    agreement, so it verifies rather than assumes.
    """
    measured = {int(cid): fit for cid, fit in alignment.crops.items()}

    for cid, fit in sorted(measured.items(), key=lambda kv: int(kv[0])):
        spread = fit.get("spread", {})
        # A per-camera median built from a minority of the frames is not the
        # camera's transform; it is whichever subset happened to match. Refuse
        # rather than silently register the plant with a half-determined mask.
        support = fit.get("support_images", 0)
        total = fit.get("total_images", 0)
        if total and support / total < CROP_MIN_SUPPORT_FRACTION:
            raise SystemExit(
                f"camera {cid}: only {support}/{total} frames yielded a fit; "
                f"the mask-to-camera transform is not determined for this sample")
        # Drift alone is not a failure. On Broccoli/20260327-5 the per-frame
        # `ty` spans 14 px because ORB matches slightly different point sets on
        # each frame; the *resulting* warped mask agrees with its image to
        # within 0.001 IoU whether the per-camera or the per-frame transform is
        # used. So the gate that matters is the end-to-end warp, which
        # `verify_alignment` measures -- spread is reported, not enforced.
        if (fit.get("inliers_min", 0) < SIM_MIN_INLIERS):
            raise SystemExit(
                f"camera {cid}: only {fit.get('inliers_min')} correspondence inliers; "
                f"the mask-to-camera transform is not established for this sample")
        if fit.get("rms", float("inf")) > SIM_RMS_MAX_PX:
            raise SystemExit(
                f"camera {cid}: similarity fit residual {fit.get('rms'):.3f}px exceeds "
                f"{SIM_RMS_MAX_PX}px; the whitebg-to-camera registration is too loose")
        if max(spread.get("tx", 0.0), spread.get("ty", 0.0)) > \
                SIM_TRANSLATION_SPREAD_MAX_PX:
            log(f"   note: camera {cid} fit drifts {max(spread.get('tx', 0.0), spread.get('ty', 0.0)):.1f}px "
                f"across frames; relying on the end-to-end warp check instead")

    if trust_defaults:
        if sample_name != STAGE4A_REFERENCE_SAMPLE:
            raise SystemExit(
                f"--trust-default-crops is only valid for {STAGE4A_REFERENCE_SAMPLE}; "
                f"{sample_name!r} must be measured on its own")
        for cid, fit in measured.items():
            recorded = STAGE4A_REFERENCE_CROPS.get(cid)
            if recorded is None:
                continue
            measured_int = (int(round(fit["tx"])), int(round(fit["ty"])))
            if measured_int != recorded:
                raise SystemExit(
                    f"camera {cid}: measured translation {measured_int} disagrees with "
                    f"the Stage 4A reference {recorded}; the recorded offsets are not "
                    f"reusable here")
        log("   measured transforms agree with the Stage 4A reference crops")
    return measured


def undistortion_maps(camera):
    """cv2 remap tables that rectify a distorted camera into a pinhole one.

    Only needed for samples whose camera model gsplat cannot consume (Stage 4B
    met FULL_OPENCV on Broccoli/20260402-10). Returns ``(map_x, map_y, new_K,
    report)`` or ``None`` when the camera is already distortion-free.

    The maps are applied to the **image and its warped mask together**, in the
    main frame loop. Undistorting the image alone would move every pixel while
    leaving the mask where it was -- a registration break that still trains,
    just on the wrong silhouette. Applying one map to both is what keeps them
    locked together.

    ``newCameraMatrix`` is derived with ``alpha=0`` so the rectified frame is the
    minimal pinhole frame containing the valid pixels; the resulting intrinsics
    are written into ``frames.json`` alongside it.
    """
    import cv2

    if camera.model in DISTORTION_FREE:
        return None
    p = [float(v) for v in camera.params]
    if camera.model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = p
        D = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        D = np.array([k1, k2, p1, p2], dtype=np.float64)
    elif camera.model == "RADIAL":
        f, cx, cy, k1, k2 = p
        fx = fy = f
        D = np.array([k1, k2, 0, 0], dtype=np.float64)
    elif camera.model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p
        fx = fy = f
        D = np.array([k1, 0, 0, 0], dtype=np.float64)
    else:
        raise ValueError(f"cannot undistort unsupported camera model {camera.model}")

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    size = (camera.width, camera.height)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=0, newImgSize=size)
    map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_32FC1)
    report = {
        "applied": True, "model": camera.model,
        "distortion": D.tolist(), "image_size": list(size),
        "alpha": 0, "valid_roi": [int(v) for v in roi],
        "K_before": K.tolist(), "K_after": new_K.tolist(),
        "note": ("image and warped mask are rectified by the SAME remap table, so "
                 "their registration is preserved; the source sample is untouched"),
    }
    return map_x, map_y, new_K, report


def rectify(frame: np.ndarray, map_x, map_y, fill) -> np.ndarray:
    """Apply the undistortion tables to one image (nearest for binary masks)."""
    import cv2
    interpolation = cv2.INTER_NEAREST if frame.ndim == 2 else cv2.INTER_LINEAR
    return cv2.remap(np.ascontiguousarray(frame), map_x, map_y, interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def resolve_crops(sample_name: str, alignment, trust_defaults: bool) -> dict:
    """Stage 4B kept a crop-shaped name in the shell path; delegate to transforms.

    Returns ``{camera_id: (dx, dy)}`` rounded from the similarity so existing
    log lines and callers keep working. New code should call
    :func:`resolve_transforms`.
    """
    fits = resolve_transforms(sample_name, alignment, trust_defaults)
    return {cid: (int(round(f["tx"])), int(round(f["ty"]))) for cid, f in fits.items()}


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
                    help="DEPRECATED, no-op: crops are always measured per sample now")
    ap.add_argument("--trust-default-crops", action="store_true",
                    help=f"verify this sample against the Stage 4A reference offsets "
                         f"({STAGE4A_REFERENCE_CROPS}); only valid for "
                         f"{STAGE4A_REFERENCE_SAMPLE}")
    ap.add_argument("--probe-limit", type=int, default=10,
                    help="images used when re-probing the mask transform")
    ap.add_argument("--verify-limit", type=int, default=10,
                    help="images used for the end-to-end mask warp check (0 = skip)")
    ap.add_argument("--frame-policy", choices=["uniform-canvas", "drop-minority"],
                    default="uniform-canvas",
                    help="'uniform-canvas' rolls minority-orientation frames and pads "
                         "everything onto one canvas; 'drop-minority' discards the "
                         "minority-orientation frames instead")
    ap.add_argument("--json-quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="rebuild the derived dataset even if it already exists; only "
                         "the derived files are replaced, never the source sample")
    ap.add_argument("--resume", action="store_true",
                    help="accept an existing dataset that is already complete for the "
                         "same sample and settings, and stop without rewriting it")
    args = ap.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)
    sample = args.sample.resolve()
    ws = args.workspace.resolve()
    if ws == sample:
        raise SystemExit("--workspace must differ from --sample")
    # "already exists" means "already has prepared artefacts", not "the
    # directory is there". The pilot script creates $WS/logs before calling
    # this, so a bare `ws.exists()` check refuses a perfectly clean workspace.
    has_dataset = (ws / "dataset").exists()
    has_meta = (ws / "preparation.json").is_file()
    if (has_dataset or has_meta) and not (args.force or args.resume):
        raise SystemExit(f"workspace already holds a prepared dataset: {ws} "
                         f"(use --resume to continue a completed run, or --force "
                         f"to rebuild the derived dataset)")
    if (has_dataset or has_meta) and args.force:
        # Deliberately narrow: only the derived dataset directory and the
        # per-sample reports this script writes are removed. The source sample
        # is never inside the workspace, and training results/checkpoints in
        # `results/` are the expensive artefacts, so they survive a rebuild of
        # the frames.
        for stale in (ws / "dataset",):
            if stale.exists():
                shutil.rmtree(stale)
        for stale in ("colmap_audit.json", "mask_alignment.json",
                      "mask_hit_rate.json", "preparation.json"):
            path = ws / stale
            if path.is_file():
                path.unlink()
        log(f"   --force: cleared derived dataset in {ws} (training results kept)")
    images_dir = sample / "Camera" / "images"
    masks_dir = sample / "RGB_seg" / "masks"
    for d in (images_dir, masks_dir):
        if not d.is_dir():
            raise SystemExit(f"missing required PAA directory: {d}")

    out = ws / "dataset"
    if args.resume:
        meta_path = ws / "preparation.json"
        if meta_path.is_file() and (out / "frames.json").is_file():
            previous = json.loads(meta_path.read_text())
            same = (previous.get("acquisition_id") == sample.name
                    and previous.get("max_size") == args.max_size
                    and previous.get("frame_policy") == args.frame_policy
                    and previous.get("seed") == args.seed)
            if same:
                log(f"PREPARE_COMPLETE (resumed; dataset already prepared at {out})")
                return
            raise SystemExit(
                "existing dataset was prepared with different settings "
                f"(sample={previous.get('acquisition_id')}, max_size={previous.get('max_size')}, "
                f"frame_policy={previous.get('frame_policy')}, seed={previous.get('seed')}); "
                "use --force to rebuild")
        log("   --resume: no complete dataset found, preparing from scratch")
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
    log(f"   {report['counts']} | reproj err mean {report['reprojection_error_px']['mean']:.3f} px")

    cameras, images, points = read_model(str(sample / "Camera" / "sparse" / "0"))
    undistortion = {}
    if not report["trainer_ready_without_undistortion"]:
        # Not a failure any more: build the rectification here, and fail only if
        # no camera can be rectified. The old behaviour (abort) is what made
        # Broccoli/20260402-10 unusable in Stage 4A.
        for cid, cam in cameras.items():
            maps = undistortion_maps(cam)
            if maps is not None:
                undistortion[cid] = maps
                log(f"   camera {cid} is {cam.model}; it will be rectified "
                    f"(image and mask by the same remap)")
        if not undistortion:
            raise SystemExit(
                "camera models need undistortion but none could be rectified: "
                f"{report['camera_models_needing_undistortion']}")

    # ------------------------------------------------------------------ #
    # 2. mask alignment
    # ------------------------------------------------------------------ #
    log("== 2. mask alignment ==")
    alignment = probe_sample(sample, limit=args.probe_limit)
    # Measured on THIS sample. `resolve_transforms` raises when the measurement
    # is inconclusive rather than falling back to another sample's offsets.
    transforms = resolve_transforms(sample.name, alignment, args.trust_default_crops)
    crops = {cid: (int(round(f["tx"])), int(round(f["ty"]))) for cid, f in transforms.items()}
    # A camera that carries frames but yielded no transform would KeyError deep
    # in the frame loop; fail here with the camera id instead.
    cameras_with_frames = {int(im.camera_id) for im in images.values()}
    unresolved = sorted(cameras_with_frames - set(transforms))
    if unresolved:
        raise SystemExit(
            f"no mask-to-camera transform could be measured for camera(s) {unresolved}; "
            f"this sample's mask registration is unresolved, not inherited")
    if args.reuse_crop:
        log("   note: --reuse-crop is a no-op; transforms are always measured per sample")
    for cid, fit in sorted(transforms.items()):
        log(f"   camera {cid}: sx={fit['sx']:.6f} sy={fit['sy']:.6f} "
            f"tx={fit['tx']:.2f} ty={fit['ty']:.2f} | fit rms {fit['rms']:.2f}px, "
            f"{fit['support_images']}/{fit['total_images']} frames")
    write_json(ws / "mask_alignment.json", alignment.to_json())

    # The transforms that were actually used, next to the evidence that produced
    # them, so a reader never has to guess whether a sample inherited them.
    write_json(ws / "crop_offsets.json", {
        "sample": sample.name,
        "method": "similarity",
        "measured_per_sample": True,
        "trust_default_crops_used": bool(args.trust_default_crops),
        "reference_crops": {str(k): list(v) for k, v in STAGE4A_REFERENCE_CROPS.items()},
        "crops": {str(k): list(v) for k, v in sorted(crops.items())},
        "transforms": {str(cid): fit for cid, fit in sorted(transforms.items())},
        "acceptance": {
            "rms_max_px": SIM_RMS_MAX_PX,
            "min_inliers": SIM_MIN_INLIERS,
            "translation_spread_max_px": SIM_TRANSLATION_SPREAD_MAX_PX,
            "scale_spread_max": SIM_SCALE_SPREAD_MAX,
        },
    })

    # End-to-end: warp each mask into its camera frame and score it against the
    # frame and the SfM cloud. This is the check that would catch a transform
    # that fits check A but is wrong about the camera.
    verify = verify_alignment(sample, alignment, limit=args.verify_limit,
                              undistortion=undistortion)
    write_json(ws / "mask_alignment_verify.json", verify)
    for cid, check in sorted(verify.items()):
        log(f"   verify camera {cid}: {check}")
        if check.get("status") != "ok":
            raise SystemExit(
                f"camera {cid}: end-to-end mask warp check failed: {check}")
        if check.get("iou_vs_nonwhite_min", 0.0) < MASK_IOU_MIN:
            raise SystemExit(
                f"camera {cid}: warped mask matches its image at only IoU "
                f"{check['iou_vs_nonwhite_min']:.3f} (< {MASK_IOU_MIN}) on at least one "
                f"frame; this sample's masks are not registered to its cameras")
        if check.get("sfm_hit_rate_min", 0.0) < HIT_RATE_ABSOLUTE_MIN:
            raise SystemExit(
                f"camera {cid}: SfM hit rate dropped to "
                f"{check['sfm_hit_rate_min']:.3f} (< {HIT_RATE_ABSOLUTE_MIN}); a frame "
                f"is not registered to its camera")

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
            # Similarity (scale + translation), not a bare crop: Cotton/1 cam 1
            # is larger than its mask canvas, so no crop exists for it.
            mask_cropped = warp_mask_to_camera(
                src_mask, transforms[image.camera_id], src_w, src_h, orientation)
        except ValueError as exc:
            dropped.append({"image": image.name, "reason": f"mask warp failed: {exc}"})
            continue
        if not (mask_cropped > 127).any():
            dropped.append({"image": image.name, "reason": "mask empty after alignment"})
            continue

        frame_arr = np.asarray(frame)
        if image.camera_id in undistortion:
            map_x, map_y, new_K, _ = undistortion[image.camera_id]
            frame_arr = rectify(frame_arr, map_x, map_y, fill=(255, 255, 255))
            mask_cropped = rectify(mask_cropped, map_x, map_y, fill=0)
            K = new_K
            frame = Image.fromarray(frame_arr)

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
        f"min {roll_check['min']:.4f} median {roll_check['median']:.4f} | "
        f"mask-vs-image IoU min {roll_check['iou_min']:.4f} "
        f"median {roll_check['iou_median']:.4f}")
    if roll_check["iou_min"] is not None and roll_check["iou_min"] < MASK_IOU_MIN:
        raise SystemExit(
            f"frame {roll_check['iou_worst_frame']} warped mask matches its image at "
            f"only IoU {roll_check['iou_min']:.3f} (< {MASK_IOU_MIN}); that mask is "
            f"not registered to its camera")
    if roll_check["worst_frame_hit_rate_over_median"] < HIT_RATE_RELATIVE_MIN:
        raise SystemExit(
            f"frame {roll_check['worst_frame']} has SfM hit rate "
            f"{roll_check['worst_frame_hit_rate']:.3f}, only "
            f"{roll_check['worst_frame_hit_rate_over_median']:.2f}x the sample median "
            f"{roll_check['median']:.3f}; it is not registered like its neighbours")
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
        "crop_offsets_measured_per_sample": True,
        "mask_alignment_method": "similarity",
        "mask_transforms": {str(cid): {
            "sx": v["sx"], "sy": v["sy"], "tx": v["tx"], "ty": v["ty"],
            "rms_px": v["rms"], "support": f"{v['support_images']}/{v['total_images']}",
        } for cid, v in sorted(transforms.items())},
        "undistortion": {
            str(cid): {k: v for k, v in maps[3].items()}
            for cid, maps in sorted(undistortion.items())
        } or {"applied": False},
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
    """Per-frame evidence that each mask is registered to its camera.

    Two measurements per frame, because one of them alone is not a registration
    test:

    * ``iou`` -- the warped mask against the camera frame's non-white pixels.
      This is a *direct* registration check: if the mask were mis-registered,
      it would not sit on the silhouette. It is independent of the 3D cloud.
    * ``hit_rate`` -- the fraction of projected SfM points landing inside the
      warped mask. This is what Stage 4A gated on, and on the samples here it
      behaves as expected: 0.96-0.99 nearly everywhere.

      It is **not** a valid gate on its own. ``Broccoli/20260402-10`` frame
      101 photographs the specimen along a thin direction, so the silhouette
      covers only 13% of the frame instead of the usual 24-38%. The sparse cloud
      contains non-specimen points (its extent is 13 x 8 x 16 world units for a
      specimen about 2 units across) that land on the silhouette from most
      directions but fall on white background from this one. That frame scores
      0.759 -- yet its mask matches its image at IoU 0.985 and its pose
      reprojects at 0.47 px. A global 0.80 floor would reject a perfectly
      registered sample.

    So the gate is: every frame's ``iou`` must clear :data:`MASK_IOU_MIN`, and no
    frame's ``hit_rate`` may collapse to a small fraction of the sample's median.
    """
    xyz_list = [p.xyz for p in points.values()]
    if limit:
        step = max(1, len(xyz_list) // limit)
        xyz_list = xyz_list[::step]
    xyz = np.stack(xyz_list)
    rates, ious, bbox_errors, centroid_shifts = {}, {}, {}, {}
    for record, mask in zip(records, masks):
        uv, valid = project(xyz, np.asarray(record["K"]),
                            np.asarray(record["camtoworld"]))
        h, w = mask.shape
        inside = (valid & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        if inside.sum() == 0:
            rates[record["name"]] = 0.0
        else:
            u = uv[inside, 0].astype(int)
            v = uv[inside, 1].astype(int)
            rates[record["name"]] = float(mask[v, u].mean())

        # Direct registration check: the warped mask against the frame it was
        # written beside. Padding is white in the image and 0 in the mask, so
        # the canvas border counts as background on both sides.
        frame = np.asarray(Image.open(record["image"]).convert("RGB"))
        nonwhite = frame.min(axis=2) < 235
        union = int(np.logical_or(mask, nonwhite).sum())
        ious[record["name"]] = (float(np.logical_and(mask, nonwhite).sum() / union)
                                if union else 0.0)
        a = np.nonzero(mask)
        b = np.nonzero(nonwhite)
        bbox_errors[record["name"]] = _bbox_corner_error(mask, nonwhite)
        centroid_shifts[record["name"]] = _centroid_shift(mask, nonwhite)

    values = np.array(list(rates.values()))
    iou_values = np.array(list(ious.values())) if ious else None
    bbox_values = np.array([v for v in bbox_errors.values()]) if bbox_errors else None
    median = float(np.median(values))
    worst = min(rates.items(), key=lambda kv: kv[1])
    flags = {
        "worst_frame": worst[0],
        "worst_frame_hit_rate": float(worst[1]),
        "worst_frame_hit_rate_over_median": float(worst[1] / max(median, 1e-9)),
        "iou_min": float(iou_values.min()) if iou_values is not None else None,
        "iou_median": float(np.median(iou_values)) if iou_values is not None else None,
        "iou_worst_frame": (min(ious.items(), key=lambda kv: kv[1])[0]
                            if ious else None),
        "bbox_corner_error_px_max": (float(np.nanmax(bbox_values))
                                     if bbox_values is not None else None),
        "bbox_corner_error_px_median": (float(np.nanmedian(bbox_values))
                                        if bbox_values is not None else None),
        "centroid_shift_px_max": (float(np.nanmax(list(centroid_shifts.values())))
                                  if centroid_shifts else None),
    }
    rolled = [v for r, v in rates.items()
              if next(x for x in records if x["name"] == r)["rolled_ccw90"]]
    flags["rolled_frames_min"] = float(min(rolled)) if rolled else None
    return {"per_frame": rates,
            "per_frame_iou": ious,
            "per_frame_bbox_corner_error_px": bbox_errors,
            "per_frame_centroid_shift_px": centroid_shifts,
            "min": float(values.min()), "median": median,
            "mean": float(values.mean()), "max": float(values.max()),
            "n_frames": len(records), **flags}


# Calibration: across the four samples of this stage the per-frame warped-mask
# vs camera-image IoU is 0.873-0.993 (median 0.93-0.99). Cotton is lowest
# because a white cotton plant is poorly separated by a "not white" threshold.
# A mask registered to the wrong frame scores near 0; a half-frame offset
# scores about 0.5. 0.80 separates those from every good frame observed.
MASK_IOU_MIN = 0.80
HIT_RATE_RELATIVE_MIN = 0.50  # a frame this far below the median is broken
HIT_RATE_ABSOLUTE_MIN = 0.85  # per-camera end-to-end floor; samples here exceed 0.85


if __name__ == "__main__":
    main()
