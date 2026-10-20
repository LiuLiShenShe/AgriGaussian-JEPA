#!/usr/bin/env python
"""Independently re-check a checkpoint -> PLY provenance chain.

The exporter *writes* the provenance record. This tool re-derives it from the
bytes on disk, using nothing but the checkpoint, the PLY, the keep-mask, and
the record itself. That separation is the point: a record that merely restates
what the exporter believed is not evidence; a record whose hashes still match
the files, and whose cleaned tensors still match the PLY rows, is.

Checks, in order:

1. the checkpoint exists and its SHA256 equals the recorded value,
2. the PLY exists and its SHA256 equals the recorded value,
3. the keep-mask's retained-index digest matches the recorded digest,
4. the PLY vertex count equals the recorded post-filter count,
5. each shared attribute of every PLY row equals the corresponding checkpoint
   row selected by the keep-mask -- this is the *mapping* check, and it is why
   the two arrays are expected to differ in length,
6. the recorded source step equals the step stored inside the checkpoint.

Exits non-zero on any failure so a shell runner can gate on it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

import provenance
ATTRIBUTE_MAP = provenance.PLY_ATTRIBUTE_MAP


def _fail(problems):
    for problem in problems:
        print('FAIL', problem)
    return 1


def verify(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    record_path = output_dir / 'export.provenance.json'
    if not record_path.is_file():
        raise SystemExit(f'no provenance record at {record_path}')
    record = provenance.read_record(record_path)

    problems = list(provenance.verify_export_chain(record))
    details = {}

    checkpoint_path = Path(record['source_checkpoint']['path'])
    ply_path = Path(record['output_ply']['path'])

    # (3) keep-mask digest.
    scores_path = output_dir / 'cleanup_scores.npz'
    keep = None
    if scores_path.is_file():
        keep = np.load(scores_path)['keep']
        retained = np.flatnonzero(keep).astype(np.int64)
        digest = hashlib.sha256(retained.tobytes()).hexdigest()
        recorded = (record.get('cleanup') or {}).get('retained_index_sha256')
        details['retained_index_sha256'] = digest
        if recorded != digest:
            problems.append(
                f'keep-mask digest {digest[:12]} != recorded {str(recorded)[:12]}')
        if len(retained) != int(record.get('gaussians_after_filter', -1)):
            problems.append(
                f'keep-mask retains {len(retained)} rows but record says '
                f'{record.get("gaussians_after_filter")}')
    else:
        problems.append(f'missing {scores_path}')

    # (1),(2) are already covered by verify_export_chain; re-state for clarity.
    details['checkpoint_sha256'] = record['source_checkpoint']['sha256']
    details['ply_sha256'] = record['output_ply']['sha256']

    # (4),(5),(6) need the actual tensors.
    if keep is not None and checkpoint_path.is_file() and ply_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        stored = (checkpoint or {}).get('splats') or {}
        if 'means' not in stored:
            problems.append(
                'checkpoint does not contain the expected gaussian tensors; the '
                'file on disk is not the one the provenance record describes')
            return {
                'status': 'fail', 'problems': problems, 'details': details,
                'checked_utc': provenance.utc_now(),
                'provenance_record': str(record_path.resolve()),
            }
        cleaned = {k: v[torch.from_numpy(keep)] for k, v in stored.items()}
        data = PlyData.read(str(ply_path))['vertex'].data
        details['ply_rows'] = len(data)
        if len(data) != len(cleaned['means']):
            problems.append(
                f'PLY has {len(data)} rows but keep-mask selects '
                f'{len(cleaned["means"])} checkpoint rows')

        if len(data) == len(cleaned['means']):
            errors = {}
            for ply_name, (tensor_name, index) in ATTRIBUTE_MAP.items():
                if tensor_name not in cleaned or ply_name not in data.dtype.names:
                    continue
                tensor = cleaned[tensor_name].detach().cpu().numpy()
                if tensor_name == 'quats':
                    tensor = provenance.normalize_quaternions(tensor)
                column = provenance.attribute_column(tensor_name, tensor, index)
                column = np.asarray(column).reshape(len(data), -1)[:, :1].ravel()
                observed = np.asarray(data[ply_name]).ravel()
                delta = float(np.abs(column - observed).max())
                errors[ply_name] = delta
                tolerance = 2e-6 if observed.dtype.itemsize == 4 else 1e-9
                if delta > tolerance:
                    problems.append(f'attribute {ply_name}: max |delta| = {delta:.3e}')
            details['max_abs_error'] = errors

        stored_step = int(checkpoint.get('step', -1))
        recorded_step = int(record.get('source_checkpoint_step', -2))
        details['checkpoint_step'] = stored_step
        details['recorded_step'] = recorded_step
        if stored_step != recorded_step:
            problems.append(
                f'checkpoint stores step {stored_step} but provenance says '
                f'{recorded_step}')

        details['gaussians_before_filter'] = len(checkpoint['splats']['means'])
        details['gaussians_after_filter'] = len(cleaned['means'])

    result = {
        'status': 'pass' if not problems else 'fail',
        'problems': problems,
        'details': details,
        'checked_utc': provenance.utc_now(),
        'provenance_record': str(record_path.resolve()),
    }
    provenance.write_record(output_dir / 'provenance_verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path,
                        help='directory containing export.provenance.json')
    parser.add_argument('--json', type=Path, default=None,
                        help='write the verification result here as well')
    args = parser.parse_args()
    result = verify(args.output_dir)
    if args.json:
        provenance.write_record(args.json, result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())