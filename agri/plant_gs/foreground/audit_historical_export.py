#!/usr/bin/env python
"""Retrospective provenance audit for runs that predate the provenance chain.

The Stage 4B assets were produced before checkpoints and PLYs were linked by
hash. Their mtimes and report texts cannot prove that ``final/point_cloud.ply``
came from the checkpoint the report names -- and for Cotton/1 they demonstrably
did not, because a second training process overwrote ``splats_best.pt`` after
the export.

This tool asks the question the mtime cannot: **does the PLY still agree with
the checkpoint it claims to come from?** It re-derives the mapping the exporter
used -- apply the keep-mask to the checkpoint's gaussians, then compare the
result row-for-row with the PLY -- and checks that the step recorded inside
the checkpoint matches the step the report quotes.

Three outcomes:

``pass``
    The PLY matches the named checkpoint, attribute for attribute, and the
    steps agree. The chain is *consistent*, even though no hash was recorded
    at the time.
``fail``
    The PLY does not match the named checkpoint. The chain is broken.
``insufficient``
    A required artefact is missing (no keep-mask, no checkpoint, no report),
    so the question cannot be answered either way.

The status vocabulary is shared with ``verify_provenance.py`` so one registry
reader can consume either file. This tool additionally sets
``retrospective: true``, because it runs after the fact rather than alongside
the export that produced the artefact.

Read-only: it never writes to the workspace it audits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

import provenance
from provenance import PLY_ATTRIBUTE_MAP


def _attribute_values(tensor_name, tensor, index):
    values = np.asarray(tensor)
    if tensor_name == 'quats':
        values = provenance.normalize_quaternions(values)
    return provenance.attribute_column(tensor_name, values, index)


def audit(workspace: Path) -> dict:
    workspace = Path(workspace)
    result = {
        'workspace': str(workspace.resolve()),
        'checked_utc': provenance.utc_now(),
        'status': 'insufficient',
        # Marks this verification as reconstructed after the fact, so a
        # registry row can say "consistent, but checked later" rather than
        # implying a hash chain was recorded when the run was made.
        'retrospective': True,
        'problems': [],
        'evidence': {},
    }

    report_path = workspace / 'final' / 'report.json'
    ply_path = workspace / 'final' / 'point_cloud.ply'
    scores_path = workspace / 'final' / 'cleanup_scores.npz'

    if not report_path.is_file():
        result['problems'].append(f'missing {report_path}')
        return result
    report = json.loads(report_path.read_text())

    checkpoint_path = Path(report.get('source_checkpoint', ''))
    if not checkpoint_path.is_file():
        result['problems'].append(
            f"named source checkpoint is missing: {checkpoint_path}")
        return result
    if not ply_path.is_file():
        result['problems'].append(f'missing {ply_path}')
        return result
    if not scores_path.is_file():
        result['problems'].append(
            f'missing {scores_path}; without the keep-mask the row mapping '
            'from checkpoint to PLY cannot be reconstructed')
        return result

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    keep = np.load(scores_path)['keep']
    stored = checkpoint.get('splats') or {}
    if 'means' not in stored:
        result['problems'].append('checkpoint contains no gaussian tensors')
        return result

    cleaned = {k: v[torch.from_numpy(keep)] for k, v in stored.items()}
    data = PlyData.read(str(ply_path))['vertex'].data

    result['evidence'] = {
        'source_checkpoint': str(checkpoint_path),
        'source_checkpoint_sha256': provenance.sha256_file(checkpoint_path),
        'checkpoint_sha256': provenance.sha256_file(checkpoint_path),
        'ply': str(ply_path),
        'ply_sha256': provenance.sha256_file(ply_path),
        'checkpoint_step': checkpoint.get('step'),
        'report_step': report.get('step'),
        'checkpoint_config_sha256': checkpoint.get('config_sha256'),
        'report_raw_gaussians': report.get('raw_gaussians'),
        'report_retained_gaussians': report.get('retained_gaussians'),
        'ply_rows': int(len(data)),
        'kept_rows': int(len(cleaned['means'])),
        'checkpoint_gaussians': int(len(stored['means'])),
    }

    problems = result['problems']

    if int(checkpoint.get('step', -1)) != int(report.get('step', -2)):
        problems.append(
            f"checkpoint stores step {checkpoint.get('step')} but the report "
            f"quotes step {report.get('step')}")

    if len(data) != len(cleaned['means']):
        problems.append(
            f"PLY has {len(data)} rows but the keep-mask selects "
            f"{len(cleaned['means'])} checkpoint rows")
    else:
        errors = {}
        for ply_name, (tensor_name, index) in PLY_ATTRIBUTE_MAP.items():
            if tensor_name not in cleaned or ply_name not in data.dtype.names:
                continue
            column = _attribute_values(tensor_name,
                                       cleaned[tensor_name].detach().cpu().numpy(),
                                       index)
            column = np.asarray(column).reshape(len(data), -1)[:, :1].ravel()
            observed = np.asarray(data[ply_name]).ravel()
            delta = float(np.abs(column - observed).max())
            errors[ply_name] = delta
            tolerance = 2e-6 if observed.dtype.itemsize == 4 else 1e-9
            if delta > tolerance:
                problems.append(f'attribute {ply_name}: max |delta| = {delta:.3e}')
        result['evidence']['max_abs_error'] = errors

    if report.get('retained_gaussians') is not None and \
            int(report['retained_gaussians']) != len(cleaned['means']):
        problems.append(
            f"report says {report['retained_gaussians']} retained gaussians but "
            f"the keep-mask keeps {len(cleaned['means'])}")

    result['details'] = {
        'checkpoint_sha256': result['evidence']['checkpoint_sha256'],
        'ply_sha256': result['evidence']['ply_sha256'],
        'checkpoint_step': result['evidence'].get('checkpoint_step'),
        'gaussians_before_filter': result['evidence'].get('checkpoint_gaussians'),
        'gaussians_after_filter': result['evidence'].get('kept_rows'),
        'ply_rows': result['evidence'].get('ply_rows'),
    }
    result['status'] = 'fail' if problems else 'pass'
    result['problems'] = problems
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.workspace)
    if args.json:
        provenance.write_record(args.json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
