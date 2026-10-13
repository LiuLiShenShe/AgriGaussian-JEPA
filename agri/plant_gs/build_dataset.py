#!/usr/bin/env python
"""Turn the COLMAP sparse model into a masked, pinhole, gsplat-ready dataset.

Why this exists
---------------
The capture is a *handheld orbit of a plant in a cluttered room*. COLMAP gets
its poses from the room (plenty of parallax, plenty of features), but the 3DGS
model must contain ONLY the plant. So the pipeline is deliberately asymmetric:

  poses  <- full frames including background   (COLMAP, already done)
  pixels <- masked frames, plant only          (this script + training)

Two consequences drive the code below:

1. Images are undistorted from the OPENCV model to a plain PINHOLE model, so the
   trainer needs no distortion math. `colmap image_undistorter` cannot also
   undistort masks, so the masks are remapped here with the *same* mapping
   (original OPENCV K + distortion -> output PINHOLE K), computed with
   cv2.initUndistortRectifyMap. COLMAP's output K already has the ROI crop baked
   into the principal point, so this reproduces its mapping exactly.

2. Only frames that have a SAM mask are kept, so every training pixel is
   labelled. Background Gaussians would otherwise receive no gradient over a
   masked-out region and survive training as floaters in the exported PLY.

Additionally the sparse point cloud is written twice: `points3D_all.npz` and
`points3D_plant.npz`. The latter keeps only points that project inside the mask
in a majority of the views that observe them -- this is what the trainer uses
for initialisation, and it is the main defence against background floaters.
"""
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import image_camtoworld, read_model  # noqa: E402

WS = "/fj/AgriGaussian-JEPA/data/plant_gs/plant_001"
IMG = os.path.join(WS, "images")
MASK = os.path.join(WS, "masks")
SPARSE = os.path.join(WS, "colmap", "sparse", "0")
UNDIST = os.path.join(WS, "undistorted")
LOG = os.path.join(WS, "logs")
OUT = os.path.join(WS, "dataset")

Image.MAX_IMAGE_PIXELS = None


def dist_params(name, params):
    """Return (K 3x3, distortion coefficients in cv2 order) for a COLMAP model."""
    if name == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        K, d = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array([k1, k2, p1, p2])
    elif name == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        d = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
    elif name == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        K, d = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.array([k1, 0, 0, 0])
    elif name == "RADIAL":
        f, cx, cy, k1, k2 = params
        K, d = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.array([k1, k2, 0, 0])
    elif name == "PINHOLE":
        fx, fy, cx, cy = params
        K, d = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.zeros(4)
    elif name == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K, d = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.zeros(4)
    else:
        raise ValueError("unsupported camera model %s" % name)
    return K, d


def main():
    os.makedirs(OUT, exist_ok=True)

    cameras, images, points3D = read_model(SPARSE)
    print("sparse model: %d cameras, %d registered images, %d points"
          % (len(cameras), len(images), len(points3D)))

    mask_files = {f[:-4] for f in os.listdir(MASK) if f.endswith(".png")}
    keep = [im for im in images.values() if os.path.splitext(im.name)[0] in mask_files]
    keep.sort(key=lambda im: im.name)
    print("registered images with a SAM mask: %d" % len(keep))
    if not keep:
        raise SystemExit("no camouflaged frames: masks and COLMAP registration do not overlap")

    # --- 1. undistort via COLMAP, restricted to the masked frames -------------
    list_path = os.path.join(WS, "undistort_image_list.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(im.name for im in keep) + "\n")

    if os.path.exists(UNDIST):
        shutil.rmtree(UNDIST)
    cmd = [
        "colmap", "image_undistorter",
        "--image_path", IMG,
        "--input_path", SPARSE,
        "--output_path", UNDIST,
        "--output_type", "COLMAP",
        "--image_list_path", list_path,
        "--copy_policy", "copy",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=open(os.path.join(LOG, "colmap_05_undistort.log"), "w"),
                   stderr=subprocess.STDOUT)

    ucameras, uimages, upoints = read_model(os.path.join(UNDIST, "sparse"))
    print("undistorted: %d camera(s), %d images, %d points"
          % (len(ucameras), len(uimages), len(upoints)))

    # --- 2. build the per-frame records, undistorting masks to match ----------
    uimg_dir = os.path.join(UNDIST, "images")
    out_mask_dir = os.path.join(OUT, "masks")
    os.makedirs(out_mask_dir, exist_ok=True)

    # The undistorted model keeps COLMAP's image names; map name -> record.
    u_by_name = {im.name: im for im in uimages.values()}

    records = []
    for im in keep:
        uim = u_by_name.get(im.name)
        if uim is None:
            continue
        ocam = cameras[im.camera_id]
        ucam = ucameras[uim.camera_id]
        if ucam.model != "PINHOLE":
            raise SystemExit("expected PINHOLE after undistortion, got %s" % ucam.model)

        K_orig, dist = dist_params(ocam.model, ocam.params)
        fx, fy, cx, cy = ucam.params
        K_new = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Sanity: undistorted size must match the undistorted image on disk.
        upath = os.path.join(uimg_dir, im.name)
        ui = Image.open(upath)
        uw, uh = ui.size
        if (uw, uh) != (ucam.width, ucam.height):
            raise SystemExit("undistorted size mismatch for %s: %s vs %s"
                             % (im.name, (uw, uh), (ucam.width, ucam.height)))

        mapx, mapy = cv2.initUndistortRectifyMap(
            K_orig, dist, np.eye(3), K_new, (uw, uh), cv2.CV_32FC1
        )
        m = np.asarray(Image.open(os.path.join(MASK, os.path.splitext(im.name)[0] + ".png")).convert("L"))
        m = cv2.remap(m, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out_mask = os.path.join(out_mask_dir, im.name)
        Image.fromarray((m > 127).astype(np.uint8) * 255, mode="L").save(out_mask)

        c2w = image_camtoworld(uim)
        records.append({
            "name": im.name,
            "image": upath,
            "mask": out_mask,
            "camtoworld": c2w.tolist(),
            "K": K_new.tolist(),
            "width": int(uw),
            "height": int(uh),
            "mask_coverage": float((m > 127).mean()),
        })

    print("dataset frames: %d" % len(records))

    # --- 3. split the sparse cloud: all points vs plant-only points -----------
    pts = np.stack([p.xyz for p in upoints.values()]) if upoints else np.zeros((0, 3))
    col = np.stack([p.rgb for p in upoints.values()]) if upoints else np.zeros((0, 3))
    err = np.array([p.error for p in upoints.values()]) if upoints else np.zeros(0)
    np.savez(os.path.join(OUT, "points3D_all.npz"), xyz=pts, rgb=col, error=err)

    # For every point, ask in how many observing views it lands on the plant.
    rec_by_name = {r["name"]: r for r in records}
    masks_cache = {}
    n_obs = np.zeros(len(upoints), dtype=np.int32)
    n_in = np.zeros(len(upoints), dtype=np.int32)
    for i, p in enumerate(upoints.values()):
        for image_id, _ in zip(p.image_ids, p.point2D_idxs):
            rec = rec_by_name.get(uimages[image_id].name) if image_id in uimages else None
            if rec is None:
                continue
            n_obs[i] += 1
            K = np.array(rec["K"])
            c2w = np.array(rec["camtoworld"])
            cam = np.linalg.inv(c2w) @ np.append(p.xyz, 1.0)
            if cam[2] <= 0:
                continue
            u = K[0, 0] * cam[0] / cam[2] + K[0, 2]
            v = K[1, 1] * cam[1] / cam[2] + K[1, 2]
            if not (0 <= u < rec["width"] and 0 <= v < rec["height"]):
                continue
            key = rec["mask"]
            if key not in masks_cache:
                masks_cache[key] = np.asarray(Image.open(key)) > 127
            if masks_cache[key][int(v), int(u)]:
                n_in[i] += 1

    ratio = np.divide(n_in, np.maximum(n_obs, 1))
    # Keep points that are consistently explained by plant pixels. A point seen
    # only once has no majority to appeal to, so require an absolute floor too.
    plant_sel = (n_obs >= 2) & (ratio >= 0.5) & (n_in >= 2)
    # Fall back to a looser rule if the strict one is too aggressive.
    if plant_sel.sum() < 100:
        plant_sel = (n_obs >= 1) & (ratio >= 0.34)

    # Geometric pruning. Mask voting alone cannot separate the plant from what
    # sits directly behind it: a point on the far wall is occluded by the plant
    # in *every* view, so it projects inside the plant silhouette in every view
    # and passes the mask-consistency test with a perfect score. The plant is
    # what the cameras orbit, so bound it by the orbit itself: the median camera
    # radius is the scale of the swept volume, and the room lies beyond it.
    cam_centers = np.stack([np.array(r["camtoworld"])[:3, 3] for r in records])
    orbit_center = cam_centers.mean(axis=0)
    cam_r = np.linalg.norm(cam_centers - orbit_center, axis=1)
    orbit_radius = float(np.median(cam_r))
    d_center = np.linalg.norm(pts - orbit_center, axis=1)
    geo_ok = d_center <= orbit_radius
    n_before = int(plant_sel.sum())
    plant_sel = plant_sel & geo_ok
    print("geometric prune: orbit centre %s, median orbit radius %.3f | dropped %d of %d kept "
          "points outside the swept volume (plant pts kept: %.1f%%)"
          % (np.round(orbit_center, 3), orbit_radius, n_before - int(plant_sel.sum()),
             n_before, 100.0 * plant_sel.sum() / max(n_before, 1)))
    print("sparse points: %d total, %d plant-selected (median obs %.1f)"
          % (len(pts), plant_sel.sum(), float(np.median(n_obs)) if len(n_obs) else 0))

    np.savez(os.path.join(OUT, "points3D_plant.npz"),
             xyz=pts[plant_sel], rgb=col[plant_sel], error=err[plant_sel],
             n_obs=n_obs, n_in=n_in)

    meta = {
        "source": "COLMAP 3.9.1 SfM on full frames; geometry from background, pixels from masks",
        "camera_model_original": sorted({c.model for c in cameras.values()}),
        "camera_model_output": "PINHOLE",
        "n_registered_total": len(images),
        "n_frames_used": len(records),
        "n_sparse_points_all": int(len(pts)),
        "n_sparse_points_plant": int(plant_sel.sum()),
        "resolution": [records[0]["width"], records[0]["height"]],
        "mask_coverage_min": float(min(r["mask_coverage"] for r in records)),
        "mask_coverage_mean": float(np.mean([r["mask_coverage"] for r in records])),
        "mask_coverage_max": float(max(r["mask_coverage"] for r in records)),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(OUT, "frames.json"), "w") as f:
        json.dump(records, f, indent=1)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()