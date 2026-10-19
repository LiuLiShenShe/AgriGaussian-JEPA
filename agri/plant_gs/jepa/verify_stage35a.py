#!/usr/bin/env python3
"""Independent checks on actual saved plant diagnostics (no synthetic experiment)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from agri.plant_gs.jepa.diagnostic_metrics import match_positions, unit


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(output, reference=None):
    summary = json.loads((output / 'diagnostic_summary.json').read_text())
    plan = json.loads((output / 'sampling_plan.json').read_text())
    indices = np.load(output / 'sampling_indices.npz', allow_pickle=False)
    checks = {}
    before = json.loads((output / 'protected_artifacts_before.json').read_text())
    after = json.loads((output / 'protected_artifacts_after.json').read_text())
    assert before == after
    for path, metadata in before.items():
        assert digest(Path(path)) == metadata['sha256'], path
    checks['protected_sha256_files_checked'] = len(before)
    for item in plan['queries']:
        q = indices[item['key']]
        g = indices['gallery_' + item['plant_id']]
        assert len(q) == item['query_gaussians']
        assert len(q) == len(np.unique(q))
        common = len(np.intersect1d(q, g))
        assert common == item['overlap_count']
        if item['protocol'] != 'old_shared':
            assert common == 0
        else:
            assert common == len(q)
    checks['query_index_conditions_checked'] = len(plan['queries'])
    for model in ('random_init', 'pilot_trained'):
        saved = np.load(output / f'{model}_features.npz', allow_pickle=False)
        for key in saved.files:
            if saved[key].dtype.kind in 'fci':
                assert np.isfinite(saved[key]).all(), (model, key)
        tok, pos = saved['encoder_tokens'], saved['positions']
        assert tok.shape == (4, 3, 64, 384)
        # On real tokens and centers, simultaneous permutation must not change
        # a geometry-matched pair score. No token-ID correspondence is used.
        left, right = tok[0, 0], tok[1, 0]
        lp, rp = pos[0, 0], pos[1, 0]
        a, b, distance = match_positions(lp, rp)
        permutation = np.arange(63, -1, -1)
        aa, bb, dd = match_positions(lp[permutation], rp[permutation])
        score = np.sum(unit(left[a]) * unit(right[b]), axis=-1).mean()
        perm_score = np.sum(unit(left[permutation][aa]) * unit(right[permutation][bb]), axis=-1).mean()
        assert abs(score - perm_score) < 1e-12
        assert abs(distance.sum() - dd.sum()) < 1e-12
        a, b, dd = match_positions(lp, lp[permutation])
        assert np.max(dd) == 0
        assert np.array_equal(left[a], left[permutation][b])
        checks[model + '_spatial_permutation_invariance'] = True
    # Cosine invariance under L2, translation invariance of Euclidean distances,
    # and the n=4 centered rank bound are checked in saved statistics.
    groups = {}
    for row in summary['feature_statistics']:
        groups.setdefault((row['model'], row['feature']), {})[row['space']] = row
        assert row['effective_rank_participation'] <= 3 + 1e-9
    for spaces in groups.values():
        for metric in ('same_cosine', 'different_cosine'):
            assert abs(spaces['raw'][metric] - spaces['l2'][metric]) < 1e-12
        for metric in ('same_l2', 'different_l2'):
            assert abs(spaces['raw'][metric] - spaces['centered_raw'][metric]) < 1e-10
    checks['feature_invariance_groups_checked'] = len(groups)
    with (output / 'retrieval_per_query.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    checked = 0
    for model in ('random_init', 'pilot_trained'):
        features = np.load(output / f'{model}_retrieval_features.npz', allow_pickle=False)
        all_ids = features['gallery_ids'].tolist()
        for row in rows:
            if row['model'] != model:
                continue
            ids = summary['test_plant_ids'] if row['candidate_count'] == '4' else all_ids
            gallery = features['gallery_' + row['feature']][[all_ids.index(p) for p in ids]]
            query = features[row['key'] + '_' + row['feature']]
            sims = unit(gallery) @ unit(query)
            identity = ids.index(row['plant_id'])
            wrong = np.delete(sims, identity)
            order = np.argsort(-sims, kind='stable')
            assert abs(float(row['true_cosine']) - sims[identity]) < 1e-13
            assert abs(float(row['retrieval_margin']) - (sims[identity] - wrong.max())) < 1e-13
            assert abs(float(row['top1_top2_gap']) - (sims[order[0]] - sims[order[1]])) < 1e-13
            assert row['retrieved_id'] == ids[order[0]]
            assert int(row['hit']) == int(order[0] == identity)
            checked += 1
    checks['retrieval_rows_independently_recomputed'] = checked
    completion_digest = digest(output / 'run_completion.json')
    rejected = subprocess.run([sys.executable, '-m', 'agri.plant_gs.jepa.diagnose_stage35a',
                               '--output', str(output)], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert 'refusing to overwrite existing directory' in rejected.stderr
    assert completion_digest == digest(output / 'run_completion.json')
    checks['existing_output_overwrite_rejected'] = True
    for csv_path in output.glob('*.csv'):
        with csv_path.open() as stream:
            for row in csv.DictReader(stream):
                for value in row.values():
                    try:
                        numeric = float(value)
                    except (ValueError, TypeError):
                        continue
                    assert np.isfinite(numeric), (csv_path, value)
    checks['all_csv_numeric_fields_finite'] = True
    if reference is not None:
        reference = reference.resolve()
        assert reference != output
        assert json.loads((reference / 'run_completion.json').read_text())['complete']
        checked_csv, checked_npz = [], []
        for path in sorted(output.glob('*.csv')):
            assert path.read_bytes() == (reference / path.name).read_bytes(), path.name
            checked_csv.append(path.name)
        array_count = 0
        for path in sorted(output.glob('*.npz')):
            actual = np.load(path, allow_pickle=False)
            expected = np.load(reference / path.name, allow_pickle=False)
            assert actual.files == expected.files, path.name
            for key in actual.files:
                assert np.array_equal(actual[key], expected[key]), (path.name, key)
                array_count += 1
            checked_npz.append(path.name)
        checks['cross_process_reproduction'] = {
            'reference_directory': str(reference), 'csv_byte_identical': checked_csv,
            'npz_all_arrays_elementwise_identical': checked_npz,
            'arrays_compared': array_count, 'passed': True}
    checks['all_checks_passed'] = True
    (output / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path,
                        help='fresh-process rerun directory: compare CSV bytes and every saved array exactly')
    args = parser.parse_args()
    verify(args.input.resolve(), args.reference)
