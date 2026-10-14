#!/usr/bin/env python
"""Same-map undistortion and foreground-only sparse initialization."""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colmap_io import read_model, image_camtoworld
from common import write_json, silhouette_support, project_points


def camera_params(camera):
    v = camera.params
    if camera.model == 'OPENCV':
        fx, fy, cx, cy, *dist = v
    elif camera.model == 'PINHOLE':
        fx, fy, cx, cy = v
        dist = np.zeros(4)
    elif camera.model in ['SIMPLE_RADIAL', 'RADIAL', 'SIMPLE_PINHOLE']:
        f, cx, cy = v[:3]
        fx = fy = f
        dist = np.zeros(4)
        dist[:len(v)-3] = v[3:]
    else:
        raise ValueError('Unsupported camera: ' + camera.model)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64), np.asarray(dist, np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--max-size', type=int, default=1280)
    args = p.parse_args()
    ws = args.workspace.resolve()
    models = []
    for model in sorted((ws / 'colmap/sparse').iterdir()):
        if (model / 'images.bin').exists():
            c, im, pts = read_model(str(model))
            models.append((len(im), model, c, im, pts))
    if not models:
        raise SystemExit('No sparse model')
    _, model, cameras, images, points = max(models, key=lambda x: x[0])
    masks_on_disk = {p.stem: p for p in (ws / 'masks').glob('*.png')}
    selected = [im for im in sorted(images.values(), key=lambda im: im.name) if Path(im.name).stem in masks_on_disk]
    if len(selected) < 30:
        raise SystemExit(f'Only {len(selected)} registered masked views')
    out = ws / 'dataset'
    for sub in ['images', 'masks']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    mappings, records, masks, sharpness = {}, [], [], {}
    for im in selected:
        cam = cameras[im.camera_id]
        if im.camera_id not in mappings:
            K, dist = camera_params(cam)
            Knew, _ = cv2.getOptimalNewCameraMatrix(K, dist, (cam.width, cam.height), 0)
            factor = min(1., args.max_size / max(cam.width, cam.height))
            width, height = round(cam.width*factor), round(cam.height*factor)
            Knew[0] *= width / cam.width
            Knew[1] *= height / cam.height
            mx, my = cv2.initUndistortRectifyMap(K, dist, np.eye(3), Knew,
                                               (width, height), cv2.CV_32FC1)
            mappings[im.camera_id] = (Knew, mx, my, width, height)
        Knew, mx, my, width, height = mappings[im.camera_id]
        rgb = cv2.imread(str(ws / 'images' / im.name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks_on_disk[Path(im.name).stem]), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 127
        if not mask.any():
            raise ValueError('Mask empty after undistortion: ' + im.name)
        ipath, mpath = out / 'images' / im.name, out / 'masks' / (Path(im.name).stem + '.png')
        cv2.imwrite(str(ipath), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        Image.fromarray(mask.astype(np.uint8)*255).save(mpath)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness[im.name] = float(lap[mask].var())
        records.append({'name': im.name, 'image_id': im.id, 'image': str(ipath), 'mask': str(mpath),
                        'camtoworld': image_camtoworld(im).tolist(), 'K': Knew.tolist(),
                        'width': width, 'height': height, 'mask_coverage': float(mask.mean())})
        masks.append(mask)
    xyz = np.stack([p.xyz for p in points.values()])
    colors = np.stack([p.rgb for p in points.values()])
    error = np.array([float(p.error) for p in points.values()])
    by_id = {r['image_id']: (r, mask) for r, mask in zip(records, masks)}
    # Track-based evidence is especially important: an occluded wall point can
    # project inside a mask without actually being observed on the object.
    obs = np.zeros(len(points), np.int32)
    positives = np.zeros(len(points), np.int32)
    for i, point in enumerate(points.values()):
        for image_id in point.image_ids:
            pair = by_id.get(int(image_id))
            if pair is None:
                continue
            frame, mask = pair
            xy, _, valid = project_points(point.xyz[None], frame)
            if valid[0]:
                uv = xy[0].astype(int)
                obs[i] += 1
                positives[i] += mask[uv[1], uv[0]]
    tolerated = [cv2.dilate(m.astype(np.uint8), np.ones((7, 7),np.uint8)) > 0 for m in masks]
    inside, viewed = silhouette_support(xyz, records, tolerated)
    track_ratio = positives / np.maximum(obs, 1)
    silhouette_ratio = inside / np.maximum(viewed, 1)
    keep = (positives >= 2) & (track_ratio >= .8) & (silhouette_ratio >= .75) & (viewed >= 10) & (error <= 3.)
    if keep.sum() < 100:
        raise SystemExit(f'Insufficient reliable foreground SfM points: {keep.sum()}')
    np.savez_compressed(out / 'points3D_all.npz', xyz=xyz, rgb=colors, error=error)
    np.savez_compressed(out / 'points3D_plant.npz', xyz=xyz[keep], rgb=colors[keep], error=error[keep])
    write_json(out / 'frames.json', records)
    write_json(out / 'sharpness.json', {'sharpness': sharpness})
    report = {'sparse_model': str(model), 'registered_rgb': len(images), 'masked_registered_views': len(records),
              'masked_unregistered_rgb': sorted(set(masks_on_disk) - {Path(r['name']).stem for r in records}),
              'sfm_points_all': len(points), 'foreground_points': int(keep.sum()),
              'mean_reprojection_error': float(error.mean()), 'working_size': [width, height],
              'foreground_track_ratio_min': .8, 'foreground_silhouette_ratio_min': .75,
              'camera': {str(k): {'model': v.model, 'params': v.params.tolist()} for k,v in cameras.items()},
              'target': 'all verified SAM foreground (plant/pot/blue block)'}
    write_json(out / 'meta.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
