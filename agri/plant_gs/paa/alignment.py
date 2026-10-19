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
Two checks that use deliberately different signal, so the composition of them
is auditable rather than assumed:

  **A. mask vs non-white pixels of whitebg (display frame).**
     ``whitebg`` is the plant composited onto white, so its non-white pixels
     *are* the segmentation. If the mask and the non-white region agree, the
     mask is a faithful label for the whitebg pixels. Reported as IoU.

  **B. whitebg vs Camera/images (NCC + mean-abs-diff).**
     Where does the smaller camera frame sit inside the bigger whitebg frame?
     ``cv2.matchTemplate(TM_CCOEFF_NORMED)`` on a scale pyramid gives the
     offset, and the mean absolute difference at that offset confirms it is a
     genuine crop and not a coincidental structural match.

Composing A (display frame) with B (display frame -> raw frame) yields
mask -> Camera/images(raw), which is the frame COLMAP's intrinsics expect.
``derive_crop`` turns that into a concrete integer offset per camera.
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
from colmap_io import read_model  # noqa: E402

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
# check B: where does the camera frame sit inside the whitebg frame?
# --------------------------------------------------------------------------- #
def match_template_offset(small: np.ndarray, large: np.ndarray,
                          scales=(8, 4, 2, 1)):
    """Locate ``small`` inside ``large`` by multi-scale NCC.

    Returns ``(dx, dy, ncc)`` such that ``small == large[dy:dy+h, dx:dx+w]``,
    or ``None`` when ``small`` cannot fit. ``cv2`` needs contiguous uint8
    arrays; arrays handed out by ``np.asarray(PIL)`` are read-only views that
    OpenCV 5 rejects.
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
    crops: dict = field(default_factory=dict)       # camera_id -> {"dx","dy",...}
    per_image: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def to_json(self):
        return {
            "sample_dir": str(self.sample_dir),
            "n_images": self.n_images,
            "cameras": {str(k): v for k, v in self.cameras.items()},
            "crops": {str(k): v for k, v in self.crops.items()},
            "per_image": self.per_image,
            "errors": self.errors,
        }


def probe_sample(sample_dir: Path, colmap_models_dir: Path | None = None,
                 limit: int | None = None) -> SampleAlignment:
    """Run checks A and B across a PAA sample and derive one crop per camera."""
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

        # B: whitebg (rotated back to the camera's stored orientation) vs the
        # camera frame. The camera frame is the smaller one.
        whitebg_raw = rotate_to_raw(
            display_image(whitebg_dir / (stem + ".jpg")), orientation)
        whitebg_raw_arr = np.asarray(whitebg_raw)
        found = match_template_offset(cam_raw, whitebg_raw_arr)
        if found is None:
            rec["B_offset"] = None
            out.errors.append(f"{stem}: whitebg frame smaller than the camera frame")
        else:
            dx, dy, ncc = found
            mad = mean_abs_difference(cam_raw, whitebg_raw_arr, dx, dy)
            rec["B_offset"] = {"dx": dx, "dy": dy, "ncc": ncc, "mean_abs_diff": mad,
                               "whitebg_raw_shape_hw": list(whitebg_raw_arr.shape)}
        out.per_image[stem] = rec

    # ---- one crop offset per camera, by majority vote ----
    for cid, group in cam_to_stems.items():
        votes: dict[tuple, list] = {}
        for stem in group:
            off = out.per_image[stem].get("B_offset")
            if off is None:
                continue
            votes.setdefault((off["dx"], off["dy"]), []).append(stem)
        if not votes:
            out.errors.append(f"camera {cid}: no usable whitebg offset")
            continue
        (dx, dy), supporters = max(votes.items(), key=lambda kv: len(kv[1]))
        nccs = [out.per_image[s]["B_offset"]["ncc"] for s in supporters]
        mads = [out.per_image[s]["B_offset"]["mean_abs_diff"] for s in supporters]
        ious = [out.per_image[s]["A_mask_vs_whitebg"]["iou"] for s in supporters
                if out.per_image[s].get("A_mask_vs_whitebg")]
        cam = cameras[cid]
        out.crops[cid] = {
            "dx": dx, "dy": dy,
            "camera_wxh": [cam.width, cam.height],
            "support_images": len(supporters),
            "total_images": len(group),
            "unanimous": len(supporters) == len(group),
            "offset_votes": {f"{k[0]},{k[1]}": len(v) for k, v in sorted(votes.items())},
            "ncc_min": float(min(nccs)), "ncc_median": float(np.median(nccs)),
            "mean_abs_diff_max": float(max(mads)),
            "A_iou_min": float(min(ious)) if ious else None,
            "A_iou_median": float(np.median(ious)) if ious else None,
        }
    return out


def derive_crop(alignment: SampleAlignment) -> dict[int, tuple[int, int]]:
    """Extract the per-camera ``(dx, dy)`` mask crop offsets."""
    return {int(cid): (int(v["dx"]), int(v["dy"])) for cid, v in alignment.crops.items()}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, required=True,
                    help="PAA sample directory, e.g. .../PAA-Plant-Dataset/Broccoli/20260327-10")
    ap.add_argument("--sparse", type=Path, default=None,
                    help="COLMAP sparse model directory (default: <sample>/Camera/sparse/0)")
    ap.add_argument("--limit", type=int, default=8,
                    help="probe this many images, spread across the orbit (0 = all)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    alignment = probe_sample(args.sample_dir, args.sparse,
                             limit=None if args.limit == 0 else args.limit)
    payload = alignment.to_json()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"sample: {alignment.sample_dir}")
    print(f"images probed: {alignment.n_images}")
    for cid, cam in alignment.cameras.items():
        crop = alignment.crops.get(cid)
        if crop is None:
            print(f"  camera {cid} {cam['model']} {cam['width']}x{cam['height']}: NO CROP")
            continue
        print(f"  camera {cid} {cam['model']} {cam['width']}x{cam['height']}: "
              f"crop dx={crop['dx']} dy={crop['dy']} "
              f"(support {crop['support_images']}/{crop['total_images']}, "
              f"unanimous={crop['unanimous']})")
        print(f"      NCC median {crop['ncc_median']:.4f} min {crop['ncc_min']:.4f} | "
              f"max mean|diff| {crop['mean_abs_diff_max']:.3f}")
        print(f"      check A IoU median {crop['A_iou_median']} min {crop['A_iou_min']}")
    for problem in alignment.errors:
        print("  ERROR:", problem)
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
