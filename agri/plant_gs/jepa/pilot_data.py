#!/usr/bin/env python3
"""Audit the fixed real-plant Gaussian set and persist an object-level split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri.plant_gs.jepa.dataset import REQUIRED_FIELDS, load_gaussian_ply

DATA_ROOT = Path('/fj/AgriGaussian-JEPA/data/plant_gs/F2DMAS_fg_20261014')
EXPERIMENT_ROOT = ROOT / 'experiments/agri_pilot'
RESULT_ROOT = ROOT / 'results/agri_pilot'
EXPECTED_PLANT_COUNT = 19
SPLIT_SEED = 20261015


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    jobs = manifest.get('jobs', [])
    usable = [job for job in jobs if job.get('status') in {'complete', 'reused'}]
    skipped = [job for job in jobs if job.get('status') not in {'complete', 'reused'}]
    names = [job['name'] for job in usable]
    if len(names) != len(set(names)):
        raise ValueError('duplicate plant IDs in manifest')
    if len(usable) != EXPECTED_PLANT_COUNT:
        raise ValueError(
            f'expected {EXPECTED_PLANT_COUNT} usable plant PLYs, found {len(usable)}'
        )

    records = []
    for job in sorted(usable, key=lambda item: item['name']):
        ply_path = Path(job['ply']).resolve()
        if not ply_path.is_file():
            raise FileNotFoundError(f"{job['name']}: missing PLY: {ply_path}")
        ply = PlyData.read(str(ply_path))
        if 'vertex' not in ply:
            raise ValueError(f"{job['name']}: PLY has no vertex element")
        vertex = ply['vertex']
        fields = set(vertex.data.dtype.names or ())
        missing = sorted(set(REQUIRED_FIELDS) - fields)
        if missing:
            raise ValueError(f"{job['name']}: missing PLY fields: {missing}")

        decoded = load_gaussian_ply(ply_path)
        declared = job.get('gaussians')
        if declared is not None and int(declared) != len(decoded):
            raise ValueError(
                f"{job['name']}: manifest says {declared} Gaussians, PLY has {len(decoded)}"
            )
        if decoded.shape != (len(vertex), 14) or not np.isfinite(decoded).all():
            raise ValueError(f"{job['name']}: decoded input is not finite N x 14")

        alpha = decoded[:, 3]
        records.append({
            'plant_id': job['name'],
            'status': job['status'],
            'ply': str(ply_path),
            'ply_sha256': sha256_file(ply_path),
            'gaussian_count': int(len(decoded)),
            'decoded_shape': list(decoded.shape),
            'attribute_order': [
                'x', 'y', 'z', 'opacity', 'scale_x', 'scale_y', 'scale_z',
                'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2',
            ],
            'finite_values': True,
            'positive_decoded_scales': bool((decoded[:, 4:7] > 0).all()),
            'opacity_probability_min': float(alpha.min()),
            'opacity_probability_mean': float(alpha.mean()),
            'opacity_probability_max': float(alpha.max()),
            'foreground_psnr': job.get('foreground_psnr'),
            'silhouette_iou': job.get('silhouette_iou'),
            'background_alpha': job.get('background_alpha'),
            'quality_warnings': job.get('quality_warnings', []),
            'gsjepa_compatible_manifest_flag': job.get('gsjepa_compatible'),
            'reconstruction_preview': job.get('preview'),
            'registered_rgb': job.get('registered_rgb'),
            'masked_registered_views': job.get('masked_registered_views'),
            'paired_masks': job.get('paired_masks'),
        })

    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest': str(manifest_path),
        'target_definition': manifest.get('target'),
        'usable_plant_count': len(records),
        'skipped_jobs': [
            {'plant_id': job.get('name'), 'status': job.get('status'),
             'reason': job.get('error')}
            for job in skipped
        ],
        'all_decoded_arrays_finite': True,
        'all_assets_have_14_model_attributes': True,
        'records': records,
        'confounds': [
            'SAM foreground includes any retained pot and marker objects; the model receives no semantic plant-only label.',
            'Residual background alpha and imperfect silhouettes remain, especially in entries with manifest quality warnings.',
            'The set contains 18 newly reconstructed assets and one reused XianKeLai1 reconstruction.',
        ],
    }


def fixed_split(plant_ids: list[str], seed: int = SPLIT_SEED,
                test_count: int = 4) -> dict:
    ids = sorted(plant_ids)
    if len(ids) != EXPECTED_PLANT_COUNT or len(ids) != len(set(ids)):
        raise ValueError(f'expected {EXPECTED_PLANT_COUNT} unique plant IDs')
    test_ids = sorted(random.Random(seed).sample(ids, test_count))
    test_set = set(test_ids)
    train_ids = [plant_id for plant_id in ids if plant_id not in test_set]
    return {
        'seed': seed,
        'unit': 'plant_id / one reconstructed PLY per plant',
        'train_plant_ids': train_ids,
        'test_plant_ids': test_ids,
        'train_count': len(train_ids),
        'test_count': len(test_ids),
        'leakage_check': not bool(set(train_ids) & set(test_ids)),
    }


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DATA_ROOT / 'manifest.json')
    parser.add_argument('--audit-output', type=Path,
                        default=EXPERIMENT_ROOT / 'data_audit.json')
    parser.add_argument('--split-output', type=Path,
                        default=EXPERIMENT_ROOT / 'split.json')
    args = parser.parse_args()

    audit = audit_manifest(args.manifest)
    split = fixed_split([row['plant_id'] for row in audit['records']])
    write_json(args.audit_output, audit)
    write_json(args.split_output, split)
    print(json.dumps({
        'audit_output': str(args.audit_output.resolve()),
        'split_output': str(args.split_output.resolve()),
        'usable_plant_count': audit['usable_plant_count'],
        'skipped_jobs': audit['skipped_jobs'],
        'train_plant_ids': split['train_plant_ids'],
        'test_plant_ids': split['test_plant_ids'],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
