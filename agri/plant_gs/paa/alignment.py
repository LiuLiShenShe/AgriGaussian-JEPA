"""Recover and verify how PAA's segmentation masks align with its COLMAP frames.

Background
----------
PAA ships four views of the same capture and they are *not* pixel-registered:

  ``Camera/images``   2128x3806 (cam 0) or 3070x2020 (cam 1) -- the frames the
                      COLMAP sparse model in ``Camera/sparse/0`` was solved on.
  ``RGB_seg/masks``   2160x3840 (portrait, every image, no EXIF).
  ``RGB_seg/whitebg`` 2160x3840 or 3840x2160, with EXIF orientation 6 on the
                      landscape subset.

Two separate hazards follow, and both have to be closed before a masked 3DGS
run can be trusted:

1. **Size.** ``Camera/images`` is smaller than the mask by a *different* ratio
   per camera (2128/2160 vs 3070/3840), so the relationship is a crop, not a
   resize. Resizing a mask to fit would silently smear the silhouette.

2. **Orientation.** COLMAP's intrinsics for camera 1 (3070x2020, principal
   point at 1535/1010) match the **raw, EXIF-ignored** frame, not the
   viewer-oriented one. Verified independently in ``colmap_check.py`` by
   keypoint-gradient contrast at projected points (3.8x raw vs 1.8x EXIF).
   The masks carry no EXIF and are portrait, i.e. they live in the *display*
   orientation of the source frames.

Method
------
Stage 4A modelled the relationship as an integer *crop*: it located the camera
frame inside the whitebg frame with ``cv2.matchTemplate`` and reported one
``(dx, dy)`` per camera. That is only valid when one grid contains the other.
Stage 4B found samples where it does not: ``Cotton/1`` camera 1 is
3858x2136 and its whitebg is 3840x2160, so neither frame contains the other and
``matchTemplate`` reports "whitebg frame smaller than the camera frame" for
every image.

The general relationship is a **similarity** (per-axis scale + translation, no
rotation), which is what a crop is a special case of. It is estimated from ORB
correspondences with a RANSAC homography gate, then fitted by least squares on
the inliers. The estimator is self-checking: on ``Broccoli/20260402-10``, whose
camera frames and whitebg are the *same* 2160x3840 pixels, it returns
``sx = sy = 0.999999`` and ``tx = ty = 0.00`` -- identity to five decimals. On
every other sample it recovers a per-camera scale within 1e-3 of 1 and a
translation consistent to well under a pixel across frames.

Three checks, deliberately using different signal, so the composition is
auditable rather than assumed:

  **A. mask vs non-white pixels of whitebg (display frame).**
     ``whitebg`` is the plant composited onto white, so its non-white pixels
     *are* the segmentation. Reported as IoU.

  **B. camera frame vs whitebg (similarity fit + residual).**
     Recovers the per-camera scale and translation, with an RMS residual in
     pixels. A wrong correspondence cannot fit this tightly: residuals are
     ~0.1-1.7 px on frames with thousands of inliers.

  **C. mask warped into the camera frame vs the camera frame's non-white
     pixels, and the fraction of SfM points that land inside the warped mask.**
     This is the end-to-end check that mask, intrinsics and pose agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colmap_io import read_model, qvec2rotmat  # noqa: E402
from paa.colmap_check import DISTORTION_FREE, intrinsics_matrix  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

EXIF_ORIENT = 274
NONWHITE_THRESHOLD = 235          # whitebg background is pure 255 white


# --------------------------------------------------------------------------- #
# orientation helpers
# --------------------------------------------------------------------------- #
def exif_orientation(path: Path):
    """Return the EXIF orientation tag, or None when absent."""
    try:
        return Image.open(path).getexif().get(EXIF_ORIENT)
    except Exception:
        return None


def display_image(path: Path) -> Image.Image:
    """Load as a viewer renders it (EXIF applied), as RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def raw_image(path: Path) -> Image.Image:
    """Load with the EXIF orientation ignored -- the frame COLMAP solved on."""
    return Image.open(path).convert("RGB")


def rotate_to_raw(display: Image.Image, orientation) -> Image.Image:
    """Undo an EXIF rotation so the image is back in its stored orientation."""
    if orientation == 6:
        return display.rotate(90, expand=True)
    if orientation == 8:
        return display.rotate(-90, expand=True)
    if orientation == 3:
        return display.rotate(180, expand=True)
    return display


# --------------------------------------------------------------------------- #
# check A: does the mask label the whitebg plant?
# --------------------------------------------------------------------------- #
def mask_vs_whitebg(mask: np.ndarray, whitebg_display: np.ndarray):
    """IoU between a binary mask and the non-white pixels of the whitebg frame.

    Returns None when the two arrays do not share a grid -- that is a real
    finding (the sample is malformed), not a zero score.
    """
    if mask.shape != whitebg_display.shape[:2]:
        return None
    nonwhite = whitebg_display.min(axis=2) < NONWHITE_THRESHOLD
    m = mask > 127
    if not m.any() or not nonwhite.any():
        return None
    inter = np.logical_and(m, nonwhite).sum()
    union = np.logical_or(m, nonwhite).sum()
    return {
        "iou": float(inter / union),
        "intersection": int(inter), "union": int(union),
        "mask_coverage": float(m.mean()),
        "whitebg_nonwhite_coverage": float(nonwhite.mean()),
        "shapes_equal": True,
    }


# --------------------------------------------------------------------------- #
# check B: how does the camera frame map into the whitebg frame?
# --------------------------------------------------------------------------- #
def _fit_similarity(src, dst):
    """Least-squares per-axis scale + translation mapping ``src`` -> ``dst``.

    No rotation term: PAA's four views are related by a resize and a shift (the
    EXIF rotation is undone separately), so allowing rotation would let noise
    fit a tilt that is not there. Returns ``(sx, sy, tx, ty, rms)``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    A = np.zeros((2 * len(src), 4))
    b = np.zeros(2 * len(src))
    A[0::2, 0] = src[:, 0]
    A[0::2, 2] = 1.0
    A[1::2, 1] = src[:, 1]
    A[1::2, 3] = 1.0
    b[0::2] = dst[:, 0]
    b[1::2] = dst[:, 1]
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = A @ solution - b
    return (*[float(v) for v in solution],
            float(np.sqrt((residual ** 2).mean())))


def match_similarity(small: np.ndarray, large: np.ndarray, nfeatures: int = 4000):
    """Estimate the similarity that maps ``small`` onto ``large``.

    Returns a dict with ``sx, sy, tx, ty``, the fit ``rms`` in pixels, the
    inlier and match counts, or ``None`` when too few correspondences survive.
    A homography is used only as a RANSAC gate on the correspondences; the
    reported transform is the constrained similarity, so a spurious rotation
    cannot leak into it.
    """
    small = np.ascontiguousarray(small)
    large = np.ascontiguousarray(large)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    if large.ndim == 3:
        large = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)
    if min(small.shape) < 16 or min(large.shape) < 16:
        return None

    orb = cv2.ORB_create(nfeatures)
    key_small, desc_small = orb.detectAndCompute(small, None)
    key_large, desc_large = orb.detectAndCompute(large, None)
    if desc_small is None or desc_large is None:
        return None
    if len(key_small) < 20 or len(key_large) < 20:
        return None

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_small, desc_large, k=2)
    good = [a for a, b in matches if a.distance < 0.75 * b.distance]
    if len(good) < 20:
        return None
    src = np.float32([key_small[m.queryIdx].pt for m in good])
    dst = np.float32([key_large[m.trainIdx].pt for m in good])

    homography, inlier_mask = cv2.findHomography(
        src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    if inliers.sum() < 20:
        return None

    sx, sy, tx, ty, rms = _fit_similarity(src[inliers], dst[inliers])
    return {"sx": sx, "sy": sy, "tx": tx, "ty": ty, "rms": rms,
            "inliers": int(inliers.sum()), "matches": len(good),
            "source_shape_hw": list(small.shape),
            "target_shape_hw": list(large.shape)}


def crop_mask_to_camera_frame(mask: np.ndarray, orientation, dx: int, dy: int,
                              width: int, height: int) -> np.ndarray:
    """Deprecated crop-only helper, kept for tests and Stage 4A compatibility.

    Use :func:`warp_mask_to_camera` instead: a crop cannot express the scale
    difference present in ``Cotton/1``.
    """
    mask_image = Image.fromarray(mask)
    if orientation == 6:
        mask_image = mask_image.rotate(90, expand=True)
    elif orientation == 8:
        mask_image = mask_image.rotate(-90, expand=True)
    elif orientation == 3:
        mask_image = mask_image.rotate(180, expand=True)
    arr = np.asarray(mask_image)
    if arr.shape[0] < dy + height or arr.shape[1] < dx + width:
        raise ValueError(
            f"mask {arr.shape[1]}x{arr.shape[0]} cannot yield {width}x{height} "
            f"at offset ({dx},{dy})")
    return arr[dy:dy + height, dx:dx + width]


def similarity_to_affine(fit: dict) -> np.ndarray:
    """2x3 affine taking a **mask** (whitebg grid) into the camera frame."""
    sx, sy = fit["sx"], fit["sy"]
    if abs(sx) < 1e-9 or abs(sy) < 1e-9:
        raise ValueError(f"degenerate similarity scale: {fit}")
    return np.array([[1.0 / sx, 0.0, -fit["tx"] / sx],
                     [0.0, 1.0 / sy, -fit["ty"] / sy]], dtype=np.float64)


def warp_mask_to_camera(mask: np.ndarray, fit: dict, width: int, height: int,
                        orientation) -> np.ndarray:
    """Bring a display-orientation mask into a COLMAP camera frame.

    Undo the EXIF rotation first (the mask carries none, so it is in the display
    orientation), then apply the inverse similarity. Nearest-neighbour
    interpolation keeps the result binary; any smoothing here would invent
    silhouette that the label does not contain.
    """
    mask_image = Image.fromarray(mask)
    if orientation == 6:
        mask_image = mask_image.rotate(90, expand=True)
    elif orientation == 8:
        mask_image = mask_image.rotate(-90, expand=True)
    elif orientation == 3:
        mask_image = mask_image.rotate(180, expand=True)
    warped = cv2.warpAffine(np.ascontiguousarray(np.asarray(mask_image)),
                            similarity_to_affine(fit), (int(width), int(height)),
                            flags=cv2.INTER_NEAREST, borderValue=0)
    return warped


def _legacy_match_template_offset(small: np.ndarray, large: np.ndarray,
                                  scales=(8, 4, 2, 1)):
    """Stage 4A crop locator, kept for reference and for tests.

    Only valid when ``small`` fits entirely inside ``large``. ``match_similarity``
    supersedes it; see the module docstring for why.
    """
    small = np.ascontiguousarray(small)
    large = np.ascontiguousarray(large)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    if large.ndim == 3:
        large = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)
    sh, sw = small.shape
    lh, lw = large.shape
    if sh > lh or sw > lw:
        return None
    best = None
    for factor in scales:
        if sh // factor < 8 or sw // factor < 8 or lh // factor < 8 or lw // factor < 8:
            continue
        s = cv2.resize(small, (sw // factor, sh // factor), interpolation=cv2.INTER_AREA)
        l = cv2.resize(large, (lw // factor, lh // factor), interpolation=cv2.INTER_AREA)
        if s.shape[0] > l.shape[0] or s.shape[1] > l.shape[1]:
            continue
        result = cv2.matchTemplate(l, s, cv2.TM_CCOEFF_NORMED)
        _, max_ncc, _, loc = cv2.minMaxLoc(result)
        dx = int(round(loc[0] * factor))
        dy = int(round(loc[1] * factor))
        dx = max(0, min(dx, lw - sw))
        dy = max(0, min(dy, lh - sh))
        if best is None or max_ncc > best[2]:
            best = (dx, dy, float(max_ncc))
    return best


def mean_abs_difference(small: np.ndarray, large: np.ndarray, dx: int, dy: int) -> float:
    """Mean absolute RGB difference of ``large[dy:dy+h, dx:dx+w]`` against ``small``."""
    small = np.asarray(small, dtype=np.int16)
    large = np.asarray(large, dtype=np.int16)
    h, w = small.shape[:2]
    crop = large[dy:dy + h, dx:dx + w]
    if crop.shape != small.shape:
        return float("inf")
    return float(np.abs(crop - small).mean())


# Backwards-compatible alias: Stage 4A scripts and tests import this name.
match_template_offset = _legacy_match_template_offset


# --------------------------------------------------------------------------- #
# per-sample probe
# --------------------------------------------------------------------------- #
@dataclass
class SampleAlignment:
    """Result of the alignment probe for one PAA sample directory."""

    sample_dir: Path
    n_images: int = 0
    cameras: dict = field(default_factory=dict)     # camera_id -> model/params
    camera_group_of: dict = field(default_factory=dict)  # stem -> camera_id
    crops: dict = field(default_factory=dict)       # camera_id -> similarity fit
    per_image: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    method: str = "similarity"

    def to_json(self):
        return {
            "sample_dir": str(self.sample_dir),
            "n_images": self.n_images,
            "method": self.method,
            "cameras": {str(k): v for k, v in self.cameras.items()},
            "crops": {str(k): v for k, v in self.crops.items()},
            "per_image": self.per_image,
            "errors": self.errors,
        }


def probe_sample(sample_dir: Path, colmap_models_dir: Path | None = None,
                 limit: int | None = None) -> SampleAlignment:
    """Run the checks across a PAA sample and derive one transform per camera.

    ``limit`` subsamples the orbit; every probed frame is still measured
    independently, and the per-camera transform is the median of those fits, so
    a single bad frame cannot move it. The spread across frames is reported, and
    ``resolve_transforms`` treats a wide spread as a failure.
    """
    sample_dir = Path(sample_dir)
    camera_dir = sample_dir / "Camera"
    images_dir = camera_dir / "images"
    masks_dir = sample_dir / "RGB_seg" / "masks"
    whitebg_dir = sample_dir / "RGB_seg" / "whitebg"
    for d in (images_dir, masks_dir, whitebg_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"required PAA directory missing: {d}")

    sparse = colmap_models_dir or (camera_dir / "sparse" / "0")
    cameras, images, _ = read_model(str(sparse))

    out = SampleAlignment(sample_dir=sample_dir)
    out.cameras = {cid: {"model": c.model, "width": c.width, "height": c.height,
                         "params": [float(v) for v in c.params]}
                   for cid, c in cameras.items()}

    stems = sorted((p.stem for p in images_dir.glob("*.jpg")), key=lambda s: (len(s), s))
    if limit is not None:
        idx = np.linspace(0, len(stems) - 1, min(limit, len(stems))).astype(int)
        stems = [stems[i] for i in sorted(set(idx))]

    cam_to_stems: dict[int, list[str]] = {}
    for stem in stems:
        name = stem + ".jpg"
        image = next((im for im in images.values() if im.name == name), None)
        if image is None:
            out.errors.append(f"{name}: present on disk but absent from the COLMAP model")
            continue
        out.camera_group_of[stem] = image.camera_id
        cam_to_stems.setdefault(image.camera_id, []).append(stem)

    out.n_images = len(stems)

    # ---- per-image checks ----
    for stem in stems:
        cid = out.camera_group_of.get(stem)
        if cid is None:
            continue
        cam = cameras[cid]
        cam_raw = np.asarray(raw_image(images_dir / (stem + ".jpg")))
        mask = np.asarray(Image.open(masks_dir / (stem + ".png")).convert("L"))
        whitebg_display = np.asarray(display_image(whitebg_dir / (stem + ".jpg")))
        orientation = exif_orientation(images_dir / (stem + ".jpg"))

        rec = {
            "camera_id": cid,
            "camera_model": cam.model,
            "camera_wxh": [cam.width, cam.height],
            "camera_frame_shape_hw": list(cam_raw.shape[:2]),
            "exif_orientation": orientation,
            "mask_shape_hw": list(mask.shape),
            "whitebg_display_shape_hw": list(whitebg_display.shape),
        }

        # A: mask (display) vs whitebg non-white (display). Both are already in
        # the display orientation, so they can be compared directly.
        rec["A_mask_vs_whitebg"] = mask_vs_whitebg(mask, whitebg_display)

        # B: camera frame vs whitebg rotated back to the camera's stored
        # orientation. Either may be the larger one -- see the module docstring.
        whitebg_raw = rotate_to_raw(
            display_image(whitebg_dir / (stem + ".jpg")), orientation)
        whitebg_raw_arr = np.asarray(whitebg_raw)
        fit = match_similarity(cam_raw, whitebg_raw_arr)
        if fit is None:
            rec["B_similarity"] = None
            out.errors.append(f"{stem}: no similarity could be fitted (too few matches)")
        else:
            rec["B_similarity"] = fit
        out.per_image[stem] = rec

    # ---- one transform per camera, by per-parameter median ----
    for cid, group in cam_to_stems.items():
        fits = [out.per_image[s]["B_similarity"] for s in group
                if out.per_image[s].get("B_similarity")]
        if not fits:
            out.errors.append(f"camera {cid}: no usable camera-to-whitebg fit")
            continue
        keys = ("sx", "sy", "tx", "ty", "rms")
        median = {k: float(np.median([f[k] for f in fits])) for k in keys}
        spread = {k: float(np.max([f[k] for f in fits]) - np.min([f[k] for f in fits]))
                  for k in ("sx", "sy", "tx", "ty")}
        ious = [out.per_image[s]["A_mask_vs_whitebg"]["iou"] for s in group
                if out.per_image[s].get("A_mask_vs_whitebg")]
        cam = cameras[cid]
        out.crops[cid] = {
            **median,
            "method": "similarity",
            "camera_wxh": [cam.width, cam.height],
            "support_images": len(fits),
            "total_images": len(group),
            "unanimous": len(fits) == len(group),
            "spread": spread,
            "inliers_min": int(min(f["inliers"] for f in fits)),
            "matches_min": int(min(f["matches"] for f in fits)),
            "A_iou_min": float(min(ious)) if ious else None,
            "A_iou_median": float(np.median(ious)) if ious else None,
        }
    return out


def derive_crop(alignment: SampleAlignment) -> dict[int, tuple[int, int]]:
    """Deprecated: rounded integer crop, for display only.

    The crop model cannot express a scale change, so this is not what the
    pipeline uses any more; it is kept so Stage 4A logs stay readable.
    """
    return {int(cid): (int(round(v["tx"])), int(round(v["ty"])))
            for cid, v in alignment.crops.items()}


def crop_mask_to_camera_frame(mask: np.ndarray, orientation, dx: int, dy: int,
                              width: int, height: int) -> np.ndarray:
    """Bring a display-orientation mask into a COLMAP camera frame.

    The mask has no EXIF, so it is in the display orientation of its source
    frame. Rotating it back into the camera's stored orientation and cropping
    by the offset recovered in check B produces a mask that is pixel-aligned
    with the image COLMAP's intrinsics describe.
    """
    mask_image = Image.fromarray(mask)
    if orientation == 6:
        mask_image = mask_image.rotate(90, expand=True)
    elif orientation == 8:
        mask_image = mask_image.rotate(-90, expand=True)
    elif orientation == 3:
        mask_image = mask_image.rotate(180, expand=True)
    arr = np.asarray(mask_image)
    if arr.shape[0] < dy + height or arr.shape[1] < dx + width:
        raise ValueError(
            f"mask {arr.shape[1]}x{arr.shape[0]} cannot yield {width}x{height} "
            f"at offset ({dx},{dy})")
    return arr[dy:dy + height, dx:dx + width]


def _bbox_corner_error(mask: np.ndarray, nonwhite: np.ndarray,
                       q: float = 0.005) -> float:
    """Max *robust* corner displacement between two binary regions.

    IoU is the wrong tool when the subject is not clearly darker than its
    background. ``Cotton/1`` is a white cotton plant: at the ``< 235``
    threshold its white bolls and bright leaves fall inside the "background"
    class, so IoU reads 0.90 while the two regions agree to a few pixels.

    The corners are taken at the ``q`` / ``1-q`` percentiles rather than the
    extremes. A single dark speck on the white backdrop -- or one stray bright
    pixel inside the specimen -- moves a min/max bounding box by hundreds of
    pixels while the actual segmentation is unchanged, which would make a
    perfectly registered frame look broken.
    """
    a = np.nonzero(mask)
    b = np.nonzero(nonwhite)
    if not (len(a[0]) and len(b[0])):
        return float("inf")
    error = 0.0
    for axis in (0, 1):
        av, bv = a[axis], b[axis]
        lo = abs(np.percentile(av, 100 * q) - np.percentile(bv, 100 * q))
        hi = abs(np.percentile(av, 100 * (1 - q)) - np.percentile(bv, 100 * (1 - q)))
        error += max(lo, hi)
    return float(error)


def _centroid_shift(mask: np.ndarray, nonwhite: np.ndarray) -> float:
    """Distance between the two regions' centroids, in pixels."""
    a = np.nonzero(mask)
    b = np.nonzero(nonwhite)
    if not (len(a[0]) and len(b[0])):
        return float("inf")
    return float(np.hypot(a[0].mean() - b[0].mean(), a[1].mean() - b[1].mean()))


def verify_alignment(sample_dir: Path, alignment: SampleAlignment,
                     limit: int = 10, rng=None, undistortion: dict | None = None) -> dict:
    """Check C: warp each mask into its camera frame and score the agreement.

    Two independent readings of the same warp:

    * IoU between the warped mask and the camera frame's non-white pixels. Both
      are in the camera frame, so this is a pixel-level registration check.
    * The fraction of SfM points that land inside the warped mask. This is the
      check that ties the mask to the *poses and intrinsics*, not just to the
      photograph; if the transform were wrong, the points would fall outside.

    A sample whose masks cannot be registered reports low values here even when
    check A (mask vs whitebg) is high, because A never involves the camera.

    ``undistortion`` maps camera id -> the tuple :func:`prepare_paa` builds, so a
    distorted camera can still be checked: the frame and mask are rectified by
    the same table before the SfM points are projected with the new K.
    """
    sample_dir = Path(sample_dir)
    rng = rng or np.random.default_rng(0)
    undistortion = undistortion or {}
    camera_dir = sample_dir / "Camera"
    images_dir = camera_dir / "images"
    masks_dir = sample_dir / "RGB_seg" / "masks"
    cameras, images, points = read_model(str(camera_dir / "sparse" / "0"))
    xyz = np.stack([p.xyz for p in points.values()])

    by_camera: dict[int, list] = {}
    for image in images.values():
        by_camera.setdefault(image.camera_id, []).append(image)

    results = {}
    for cid, records in sorted(by_camera.items()):
        fit = alignment.crops.get(cid)
        if fit is None:
            results[str(cid)] = {"status": "no_transform"}
            continue
        records = sorted(records, key=lambda im: im.name)
        idx = np.linspace(0, len(records) - 1,
                          min(limit, len(records))).astype(int)
        picked = [records[i] for i in sorted(set(idx))]
        ious, hits, bbox_errors, centroid_shifts = [], [], [], []
        for image in picked:
            stem = Path(image.name).stem
            orientation = exif_orientation(images_dir / image.name)
            mask = np.asarray(Image.open(masks_dir / (stem + ".png")).convert("L"))
            camera = cameras[cid]
            try:
                warped = warp_mask_to_camera(mask, fit, camera.width,
                                             camera.height, orientation)
            except ValueError as exc:
                results[str(cid)] = {"status": f"warp_failed: {exc}"}
                break
            inside_mask = warped > 127

            frame = np.asarray(raw_image(images_dir / image.name))
            nonwhite = frame.min(axis=2) < NONWHITE_THRESHOLD
            if cid in undistortion:
                map_x, map_y, new_K, _ = undistortion[cid]
                import cv2
                frame = cv2.remap(np.ascontiguousarray(frame), map_x, map_y,
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
                inside_mask = cv2.remap(np.ascontiguousarray(inside_mask.astype(np.uint8)),
                                        map_x, map_y, cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0) > 0
                nonwhite = frame.min(axis=2) < NONWHITE_THRESHOLD
                K, _ = new_K, None
            else:
                K, _ = _pinhole_matrix(camera)
            union = int(np.logical_or(inside_mask, nonwhite).sum())
            ious.append(float(np.logical_and(inside_mask, nonwhite).sum() / union)
                        if union else 0.0)
            bbox_errors.append(_bbox_corner_error(inside_mask, nonwhite))
            centroid_shifts.append(_centroid_shift(inside_mask, nonwhite))

            R = qvec2rotmat(image.qvec)
            cam = xyz @ R.T + image.tvec
            z = cam[:, 2]
            ok = z > 1e-6
            with np.errstate(divide="ignore", invalid="ignore"):
                u = K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]
                v = K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]
            inb = (np.isfinite(u) & np.isfinite(v)
                   & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height))
            if inb.sum():
                hits.append(float(inside_mask[v[inb].astype(int),
                                               u[inb].astype(int)].mean()))
        else:
            results[str(cid)] = {
                "status": "ok",
                "images_checked": len(picked),
                "iou_vs_nonwhite_median": float(np.median(ious)) if ious else None,
                "iou_vs_nonwhite_min": float(np.min(ious)) if ious else None,
                "bbox_corner_error_px_max": float(np.max(bbox_errors)) if bbox_errors else None,
                "bbox_corner_error_px_median": (float(np.median(bbox_errors))
                                                if bbox_errors else None),
                "centroid_shift_px_max": float(np.max(centroid_shifts)) if centroid_shifts else None,
                "centroid_shift_px_median": (float(np.median(centroid_shifts))
                                             if centroid_shifts else None),
                "sfm_hit_rate_median": float(np.median(hits)) if hits else None,
                "sfm_hit_rate_min": float(np.min(hits)) if hits else None,
            }
    return results


def _pinhole_matrix(camera):
    """K for a distortion-free camera; raises for models needing undistortion."""
    if camera.model not in DISTORTION_FREE:
        raise ValueError(
            f"camera model {camera.model} needs undistortion before its masks "
            f"can be verified against pinhole projections")
    return intrinsics_matrix(camera)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, required=True,
                    help="PAA sample directory, e.g. .../PAA-Plant-Dataset/Broccoli/20260327-10")
    ap.add_argument("--sparse", type=Path, default=None,
                    help="COLMAP sparse model directory (default: <sample>/Camera/sparse/0)")
    ap.add_argument("--limit", type=int, default=8,
                    help="probe this many images, spread across the orbit (0 = all)")
    ap.add_argument("--verify-limit", type=int, default=10,
                    help="images used for the end-to-end warp check (0 = skip)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    alignment = probe_sample(args.sample_dir, args.sparse,
                             limit=None if args.limit == 0 else args.limit)
    payload = alignment.to_json()
    if args.verify_limit:
        try:
            payload["verify"] = verify_alignment(args.sample_dir, alignment,
                                                 limit=args.verify_limit)
        except ValueError as exc:
            payload["verify"] = {"status": f"skipped: {exc}"}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"sample: {alignment.sample_dir}")
    print(f"images probed: {alignment.n_images} | method: {alignment.method}")
    for cid, cam in alignment.cameras.items():
        fit = alignment.crops.get(cid)
        if fit is None:
            print(f"  camera {cid} {cam['model']} {cam['width']}x{cam['height']}: NO FIT")
            continue
        print(f"  camera {cid} {cam['model']} {cam['width']}x{cam['height']}: "
              f"sx={fit['sx']:.6f} sy={fit['sy']:.6f} "
              f"tx={fit['tx']:.2f} ty={fit['ty']:.2f} "
              f"(support {fit['support_images']}/{fit['total_images']}, "
              f"unanimous={fit['unanimous']})")
        print(f"      fit rms {fit['rms']:.2f}px | inliers min {fit['inliers_min']} | "
              f"spread sx {fit['spread']['sx']:.2e} tx {fit['spread']['tx']:.2f}px")
        print(f"      check A IoU median {fit['A_iou_median']} min {fit['A_iou_min']}")
    for cid, check in (payload.get("verify") or {}).items():
        print(f"  check C camera {cid}: {check}")
    for problem in alignment.errors:
        print("  ERROR:", problem)
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
