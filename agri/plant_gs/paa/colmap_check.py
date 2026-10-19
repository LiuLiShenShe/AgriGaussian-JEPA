"""Integrity and geometric audit of a PAA COLMAP sparse model.

Deliberately recomputes quantities the file claims rather than trusting them.
Two findings in the PAA sample make that necessary:

* **The stored per-point `error` field is identically zero.** PAA's export zeroed
  it, so "mean reprojection error = 0.0" from the file is meaningless. The real
  error is recomputed here by projecting every track observation with the
  stored intrinsics and poses.
* **Camera 1's intrinsics describe the EXIF-ignored frame.** Its principal point
  (1535, 1010) sits at the centre of 3070x2020, and the image on disk is only
  3070x2020 wide *before* EXIF rotation. ``orientation_probe`` confirms which
  reading is correct by measuring image-gradient energy at the projected
  points: the right orientation puts the SfM keypoints on real structure.

Every number this module prints is measured from the sample, not assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colmap_io import read_model, qvec2rotmat  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

# Camera models gsplat can consume without an undistortion pass. Anything else
# needs a real geometric conversion before it may be used for training.
DISTORTION_FREE = {"PINHOLE", "SIMPLE_PINHOLE"}


def intrinsics_matrix(camera):
    """(K, distortion) for the models PAA is expected to use."""
    p = [float(v) for v in camera.params]
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = p
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), np.zeros(4)
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), np.zeros(4)
    if camera.model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), np.array([k1, 0, 0, 0])
    raise ValueError(f"unsupported camera model {camera.model}")


def project(points_xyz, camera, image):
    """Project world points through one COLMAP image record."""
    K, _ = intrinsics_matrix(camera)
    R = qvec2rotmat(image.qvec)
    cam = np.asarray(points_xyz) @ R.T + image.tvec
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                       K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=1)
    return uv, z, np.isfinite(uv).all(axis=1) & (z > 1e-6)


def reprojection_error(cameras, images, points):
    """Recompute per-observation reprojection error from the stored model.

    This is the honest number; PAA's own ``error`` field is zeroed.
    """
    errors = []
    per_image = {}
    for pid, point in points.items():
        for image_id, obs_idx in zip(point.image_ids, point.point2D_idxs):
            image = images.get(int(image_id))
            if image is None or obs_idx >= len(image.xys):
                continue
            camera = cameras[image.camera_id]
            uv, _, ok = project(point.xyz[None], camera, image)
            if not ok[0]:
                continue
            d = float(np.linalg.norm(uv[0] - image.xys[obs_idx]))
            errors.append(d)
            per_image.setdefault(image.name, []).append(d)
    return np.asarray(errors, dtype=np.float64), per_image


def gradient_contrast(gray: np.ndarray, uv: np.ndarray, rng) -> tuple:
    """Mean |grad| at ``uv`` divided by the same at random pixels.

    A ratio near 1 means the projected points land on nothing in particular;
    a ratio well above 1 means they land on image structure -- i.e. the
    intrinsics, the pose and the image orientation agree.
    """
    gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.hypot(gx, gy)
    h, w = magnitude.shape
    uv = uv[np.isfinite(uv).all(axis=1)]
    inside = ((uv[:, 0] >= 2) & (uv[:, 0] < w - 2) &
              (uv[:, 1] >= 2) & (uv[:, 1] < h - 2))
    uv = uv[inside]
    if len(uv) < 50:
        return None, 0
    at_points = magnitude[uv[:, 1].astype(int), uv[:, 0].astype(int)].mean()
    ry = rng.integers(2, h - 2, len(uv))
    rx = rng.integers(2, w - 2, len(uv))
    at_random = magnitude[ry, rx].mean()
    return float(at_points / max(at_random, 1e-6)), int(len(uv))


def orientation_probe(cameras, images, points, images_dir: Path, stems, rng):
    """Decide, from the data, which orientation COLMAP's intrinsics describe."""
    xyz = np.stack([p.xyz for p in points.values()])
    rows = []
    for stem in stems:
        name = stem + ".jpg"
        image = next((im for im in images.values() if im.name == name), None)
        if image is None:
            continue
        camera = cameras[image.camera_id]
        uv, _, _ = project(xyz, camera, image)
        stored = np.asarray(Image.open(images_dir / name).convert("L"))
        viewed = np.asarray(ImageOps.exif_transpose(
            Image.open(images_dir / name)).convert("L"))
        stored_ratio, n_used = gradient_contrast(stored, uv, rng)
        viewed_ratio, _ = gradient_contrast(viewed, uv, rng)
        rows.append({
            "image": name, "camera_id": image.camera_id,
            "stored_shape_hw": list(stored.shape),
            "viewed_shape_hw": list(viewed.shape),
            "camera_wxh": [camera.width, camera.height],
            "intrinsics_match_stored": (stored.shape[1] == camera.width
                                        and stored.shape[0] == camera.height),
            "contrast_stored": stored_ratio, "contrast_viewed": viewed_ratio,
            "projected_points_used": n_used,
        })
    return rows


def audit(sample_dir: Path, sparse: Path | None = None, probe_limit: int = 0) -> dict:
    sample_dir = Path(sample_dir)
    camera_dir = sample_dir / "Camera"
    images_dir = camera_dir / "images"
    sparse = sparse or (camera_dir / "sparse" / "0")
    rng = np.random.default_rng(0)

    report: dict = {"sample_dir": str(sample_dir.resolve()), "sparse": str(sparse.resolve())}

    files = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        p = sparse / name
        files[name] = {"exists": p.is_file(),
                       "bytes": p.stat().st_size if p.is_file() else 0}
    report["files"] = files
    missing = [k for k, v in files.items() if not v["exists"]]
    if missing:
        report["status"] = "FAIL_MISSING_FILES"
        report["missing"] = missing
        return report

    cameras, images, points = read_model(str(sparse))
    report["counts"] = {"cameras": len(cameras), "registered_images": len(images),
                        "sparse_points": len(points)}

    # ---- camera models ----
    report["cameras"] = {}
    unsupported = []
    distorted = []
    for cid, cam in cameras.items():
        n = sum(1 for im in images.values() if im.camera_id == cid)
        K, dist = intrinsics_matrix(cam)
        report["cameras"][str(cid)] = {
            "model": cam.model, "width": cam.width, "height": cam.height,
            "params": [float(v) for v in cam.params],
            "images": n, "distortion": [float(v) for v in dist],
            "distortion_free": cam.model in DISTORTION_FREE,
            "fx_fy_equal": bool(abs(K[0, 0] - K[1, 1]) < 1e-6),
            "principal_point_normalised": [float(K[0, 2] / cam.width),
                                           float(K[1, 2] / cam.height)],
            "horizontal_fov_deg": float(np.degrees(2 * np.arctan(cam.width / 2 / K[0, 0]))),
            "vertical_fov_deg": float(np.degrees(2 * np.arctan(cam.height / 2 / K[1, 1]))),
        }
        if cam.model not in DISTORTION_FREE:
            distorted.append(cam.model)
        if cam.model not in DISTORTION_FREE and cam.model not in (
                "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"):
            unsupported.append(cam.model)
    report["camera_model_distortion_free"] = not distorted
    report["camera_models_needing_undistortion"] = sorted(set(distorted))
    report["camera_models_unsupported"] = sorted(set(unsupported))

    # ---- image / disk correspondence ----
    on_disk = {p.name for p in images_dir.glob("*.jpg")}
    registered = {im.name for im in images.values()}
    report["correspondence"] = {
        "on_disk": len(on_disk), "registered": len(registered),
        "registered_missing_on_disk": sorted(registered - on_disk),
        "on_disk_not_registered": sorted(on_disk - registered),
        "exact": registered == on_disk,
    }

    # ---- per-image decoding and size agreement (stored vs EXIF-ignored) ----
    bad_decode, size_mismatch, exif_images = [], [], []
    for im in images.values():
        p = images_dir / im.name
        try:
            with Image.open(p) as handle:
                stored = handle.size            # (w, h) as written on disk
                orientation = handle.getexif().get(274)
        except Exception as exc:                # noqa: BLE001 - report, don't crash
            bad_decode.append({"image": im.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        cam = cameras[im.camera_id]
        if (stored[0], stored[1]) != (cam.width, cam.height):
            size_mismatch.append({"image": im.name, "on_disk_wxh": list(stored),
                                  "camera_wxh": [cam.width, cam.height]})
        if orientation not in (None, 1):
            exif_images.append({"image": im.name, "orientation": orientation})
    report["image_decoding"] = {
        "undecodable": bad_decode,
        "size_mismatch_stored_vs_camera": size_mismatch,
        "all_sizes_match_stored_frame": not size_mismatch,
        "images_carrying_exif_orientation": exif_images,
    }

    # ---- pose sanity ----
    dets, nan_pose = [], []
    centers = []
    for im in images.values():
        if not (np.isfinite(im.qvec).all() and np.isfinite(im.tvec).all()):
            nan_pose.append(im.name)
            continue
        R = qvec2rotmat(im.qvec)
        dets.append(float(np.linalg.det(R)))
        centers.append(-R.T @ im.tvec)
    dets = np.asarray(dets)
    centers = np.asarray(centers)
    report["poses"] = {
        "non_finite": nan_pose,
        "rotation_determinant_min": float(dets.min()),
        "rotation_determinant_max": float(dets.max()),
        "rotations_are_proper": bool(np.allclose(dets, 1.0, atol=1e-6)),
        "camera_center_bbox_min": centers.min(axis=0).tolist(),
        "camera_center_bbox_max": centers.max(axis=0).tolist(),
        "camera_center_centroid": centers.mean(axis=0).tolist(),
        "camera_radius_median": float(np.median(np.linalg.norm(centers - centers.mean(0), axis=1))),
        "camera_radius_min": float(np.linalg.norm(centers - centers.mean(0), axis=1).min()),
        "camera_radius_max": float(np.linalg.norm(centers - centers.mean(0), axis=1).max()),
    }

    # ---- sparse points ----
    xyz = np.stack([p.xyz for p in points.values()])
    stored_err = np.array([float(p.error) for p in points.values()])
    tracks = np.array([len(p.image_ids) for p in points.values()])
    report["points3D"] = {
        "count": int(len(xyz)),
        "finite": bool(np.isfinite(xyz).all()),
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
        "stored_error_all_zero": bool(np.all(stored_err == 0)),
        "stored_error_note": ("PAA zeroed this field; the recomputed value below is "
                              "the meaningful one" if np.all(stored_err == 0) else None),
        "track_length_min": int(tracks.min()), "track_length_median": float(np.median(tracks)),
        "track_length_max": int(tracks.max()),
        "observations_total": int(tracks.sum()),
    }

    # ---- recomputed reprojection error ----
    errs, per_image = reprojection_error(cameras, images, points)
    report["reprojection_error_px"] = {
        "observations": int(len(errs)),
        "mean": float(errs.mean()), "median": float(np.median(errs)),
        "p95": float(np.percentile(errs, 95)), "max": float(errs.max()),
        "finite": bool(np.isfinite(errs).all()),
        "per_image_median_worst5": sorted(
            ({"image": k, "median": float(np.median(v)), "n": len(v)}
             for k, v in per_image.items()),
            key=lambda r: -r["median"])[:5],
    }

    # ---- orientation probe ----
    stems = sorted((p.stem for p in images_dir.glob("*.jpg")), key=lambda s: (len(s), s))
    if probe_limit and probe_limit < len(stems):
        idx = np.linspace(0, len(stems) - 1, probe_limit).astype(int)
        stems = [stems[i] for i in sorted(set(idx))]
    report["orientation_probe"] = orientation_probe(
        cameras, images, points, images_dir, stems, rng)

    # ---- supported for the downstream 3DGS trainer ----
    ok = (not missing
          and report["image_decoding"]["all_sizes_match_stored_frame"]
          and report["poses"]["rotations_are_proper"]
          and not report["poses"]["non_finite"]
          and report["points3D"]["finite"]
          and report["reprojection_error_px"]["finite"]
          and not report["camera_models_unsupported"])
    report["status"] = "PASS" if ok else "FAIL"
    report["trainer_ready_without_undistortion"] = bool(
        report["camera_model_distortion_free"] and not size_mismatch)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--sparse", type=Path, default=None)
    ap.add_argument("--probe-limit", type=int, default=6)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    report = audit(args.sample_dir, args.sparse, args.probe_limit)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")

    print(f"status: {report['status']}")
    print(f"files: { {k: v['exists'] for k, v in report['files'].items()} }")
    print(f"counts: {report['counts']}")
    print(f"correspondence: {report['correspondence']}")
    for cid, cam in report.get("cameras", {}).items():
        print(f"camera {cid}: {cam['model']} {cam['width']}x{cam['height']} "
              f"images={cam['images']} distortion_free={cam['distortion_free']} "
              f"hfov={cam['horizontal_fov_deg']:.1f}deg")
    print(f"sizes match stored frame: {report['image_decoding']['all_sizes_match_stored_frame']}")
    print(f"exif-carrying images: {len(report['image_decoding']['images_carrying_exif_orientation'])}")
    print(f"poses finite/proper: {report['poses']['rotation_determinant_min']:.6f}"
          f"..{report['poses']['rotation_determinant_max']:.6f}")
    print(f"reprojection error (recomputed): mean={report['reprojection_error_px']['mean']:.3f} "
          f"median={report['reprojection_error_px']['median']:.3f} "
          f"p95={report['reprojection_error_px']['p95']:.3f} px")
    print("orientation probe (contrast at projected points; higher = correct frame):")
    for row in report["orientation_probe"]:
        print(f"   {row['image']:>10} cam={row['camera_wxh']} stored={row['stored_shape_hw']} "
              f"viewed={row['viewed_shape_hw']} | stored {row['contrast_stored']} "
              f"viewed {row['contrast_viewed']}")
    print(f"trainer ready without undistortion: {report['trainer_ready_without_undistortion']}")
    if args.json:
        print(f"wrote {args.json}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()