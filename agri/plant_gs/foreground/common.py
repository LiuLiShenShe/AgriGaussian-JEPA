"""Auditable pairing and projection helpers for foreground-only reconstruction."""
from pathlib import Path
import json

import numpy as np


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def find_rgb_match(crop, mask, rgb_samples, names, max_mae=0.01, min_exact=0.999):
    """Match a masked image to source RGB; never assume matching filenames.

    Arrays use the SAME sampling grid (not resize); foreground RGB is unchanged
    by SAM export. Fail closed on uncertain/ambiguous correspondence.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError('empty foreground mask')
    target = np.asarray(crop)[mask].astype(np.int16)
    candidates = np.asarray(rgb_samples)[:, mask].astype(np.int16)
    diffs = np.abs(candidates - target[None])
    errors = diffs.mean(axis=(1, 2))
    exact = (diffs == 0).all(axis=-1).mean(axis=-1)
    good = np.flatnonzero((errors <= max_mae) & (exact >= min_exact))
    if len(good) != 1:
        raise ValueError(f'expected unique exact RGB match, got {len(good)}; '
                         f'best MAE={errors.min():.4f}, exact={exact[errors.argmin()]:.4f}')
    i = int(good[0])
    return {'rgb': names[i], 'mae': float(errors[i]),
            'exact_fraction': float(exact[i]), 'sampled_pixels': int(mask.sum())}


def project_points(points, frame):
    """Project world points to a PINHOLE image, with explicit valid visibility."""
    w2c = np.linalg.inv(np.asarray(frame['camtoworld'], dtype=np.float64))
    camera = np.asarray(points) @ w2c[:3, :3].T + w2c[:3, 3]
    projected = camera @ np.asarray(frame['K'], dtype=np.float64).T
    xy = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
    valid = (camera[:, 2] > 1e-6) & np.isfinite(xy).all(axis=1)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < frame['width'])
    valid &= (xy[:, 1] >= 0) & (xy[:, 1] < frame['height'])
    return xy, camera[:, 2], valid


def silhouette_support(points, frames, masks):
    """Count positive and total silhouette tests (NOT a surface visibility test).

    Outside the frame/behind the camera is unknown, not evidence of background.
    Occluded points can project outside noisy masks: threshold conservatively.
    """
    inside = np.zeros(len(points), dtype=np.int32)
    observed = np.zeros(len(points), dtype=np.int32)
    for frame, mask in zip(frames, masks):
        xy, _, valid = project_points(points, frame)
        idx = np.flatnonzero(valid)
        uv = xy[idx].astype(np.int64)
        observed[idx] += 1
        inside[idx] += np.asarray(mask, dtype=bool)[uv[:, 1], uv[:, 0]]
    return inside, observed


def evaluation_due(step, every, total_steps, reset_every, refine_stop, recovery_steps):
    """Avoid evaluating immediately after DefaultStrategy opacity resets.

    Training and densification are unchanged; only the measurement is delayed
    by one pass through the training views while resets are still active.
    """
    if step == total_steps - 1:
        return True
    if step <= 0:
        return False
    if step % every == 0:
        reset_now = step < refine_stop and step % reset_every == 0
        return not reset_now
    scheduled = step - recovery_steps
    return (scheduled > 0 and scheduled < refine_stop
            and scheduled % every == 0 and scheduled % reset_every == 0)
