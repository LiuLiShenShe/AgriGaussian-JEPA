#!/usr/bin/env python3
"""Verify actual Stage 3.5-B plant outputs and optional fresh-process replay."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from agri.plant_gs.jepa.pilot_data import sha256_file
from agri.plant_gs.jepa.pilot_eval import spatial_cap_keep_mask
from agri.plant_gs.jepa.pooling_stage35b import ROOT, dump_json, load_clouds, validate_source_hashes


def check_close(actual, expected, label, tolerance=5e-13):
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(f'{label}: {actual} != {expected}')


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def finite_json(value):
    if isinstance(value, dict):
        for v in value.values():
            finite_json(v)
    elif isinstance(value, list):
        for v in value:
            finite_json(v)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise AssertionError('nonfinite JSON number')


def select_rows(all_rows, aggregate):
    scope, group = aggregate['scope'], aggregate['group']
    if scope == 'overall':
        return all_rows
    if scope == 'ratio':
        return [r for r in all_rows if float(r['missing_ratio']) == float(group)]
    if scope == 'scheme':
        return [r for r in all_rows if r['scheme'] == group]
    if scope == 'scheme_ratio':
        scheme, ratio = group.split('/')
        return [r for r in all_rows if r['scheme'] == scheme and float(r['missing_ratio']) == float(ratio)]
    if scope == 'direction':
        return [r for r in all_rows if r['direction'] == group]
    parts = group.split('/')
    subset = [r for r in all_rows if r['plant_id'] == parts[0]]
    if scope == 'plant':
        return subset
    if scope == 'plant_ratio':
        return [r for r in subset if float(r['missing_ratio']) == float(parts[1])]
    if scope == 'plant_scheme':
        return [r for r in subset if r['scheme'] == parts[1]]
    if scope == 'plant_scheme_ratio':
        return [r for r in subset if r['scheme'] == parts[1] and float(r['missing_ratio']) == float(parts[2])]
    raise ValueError(scope)


def verify(output, reference=None):
    protocol = json.loads((output / 'protocol.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    plan = json.loads((output / 'sampling_plan.json').read_text())['items']
    indices = np.load(output / 'sampling_indices.npz', allow_pickle=False)
    features = np.load(output / 'encoder_features.npz', allow_pickle=False)
    before = json.loads((output / 'protected_artifacts_before.json').read_text())
    after = json.loads((output / 'protected_artifacts_after.json').read_text())
    if before != after:
        raise AssertionError('protected before/after snapshots differ')
    for path, metadata in before.items():
        assert sha256_file(Path(path)) == metadata['sha256'], path
    checks = {'protected_sha256_files_checked': len(before)}
    assert sha256_file(output / 'protocol.json') == summary['protocol_sha256']
    amendment = validate_source_hashes(protocol, output)
    if amendment:
        checks['validation_amendment'] = amendment
        note = ('\n## 数值校验修订记录\n\n'
                '首次验证的 float64 单位范数阈值 1e-15 过严；768 维向量实际最大误差 '
                '1.7763568394002505e-15。仅把验证阈值改为 32 × float64 epsilon '
                '（约 7.11e-15），并记录源代码哈希修订。原始 protocol.json 字节、'
                '全部种子、采样索引、Token、两种读出及分数均未改变；'
                '评分/统计模块 pooling_stage35b_report.py 的哈希仍与预注册一致。'
                'validation_amendment.json 和 verification.json 保存修订与实际误差。\n')
        report = output / 'diagnosis_report.md'
        if '## 数值校验修订记录' not in report.read_text():
            report.write_text(report.read_text() + note)
    assert not set(protocol['new_seeds']) & set(protocol['historical_seed_audit']['used_seeds'])
    assert len(protocol['new_seeds']) == len(set(protocol['new_seeds'])) == 34
    assert protocol['plant_ids'] == ['DouBanLv2', 'XianKeLai1', 'XianKeLai2', 'XiangPiShu2']
    checks['new_seeds_disjoint_from_historical_seeds'] = True
    assert summary['model_state_sha256_before'] == summary['model_state_sha256_after']
    for key in ('all_parameters_frozen', 'all_model_modules_eval', 'all_parameter_gradients_none',
                'same_tokens_for_both_poolings', 'inference_started_after_protocol_created'):
        assert summary[key] is True, key
    assert len(plan) == summary['encoder_forwards'] == 132
    assert sum(p['kind'] == 'query' for p in plan) == 108
    assert sum(p['kind'] == 'gallery' for p in plan) == 12
    assert sum(p['kind'] == 'resampling' for p in plan) == 12
    clouds, _ = load_clouds(protocol)
    schemes = {s['id']: s for s in protocol['schemes']}
    for item in plan:
        key, plant, seed = item['key'], item['plant_id'], item['seed']
        selected = indices[key]
        assert selected.dtype == np.int64 and selected.shape == (1024,)
        assert len(np.unique(selected)) == 1024
        assert selected.min() >= 0 and selected.max() < len(clouds[plant])
        scheme = schemes[item['scheme']]
        if item['kind'] == 'gallery':
            assert seed == scheme['gallery_seed']
            expected = np.random.RandomState(seed).choice(len(clouds[plant]), 1024, replace=False)
        else:
            gallery = indices[item['gallery_key']]
            assert len(np.intersect1d(selected, gallery)) == item['overlap_count'] == 0
            assert item['overlap_fraction'] == 0.0
            if item['kind'] == 'query':
                ratio = item['missing_ratio']
                ri = protocol['missing_ratios'].index(ratio)
                ai = list(protocol['directions']).index(item['direction'])
                assert seed == scheme['query_seeds'][ri][ai]
                direction = np.asarray(protocol['directions'][item['direction']])
                keep = spatial_cap_keep_mask(clouds[plant][:, :3], ratio, direction)
                assert np.all(keep[selected])
                assert int((~keep).sum()) == round(len(keep) * ratio)
                eligible = np.setdiff1d(np.flatnonzero(keep), gallery)
                assert len(eligible) == item['eligible_count']
                check_close(item['actual_missing_fraction'], 1 - keep.mean(), key)
                projection = clouds[plant][:, :3].astype(np.float64) @ direction
                check_close(item['projection_threshold'], projection[keep].max(), key)
            else:
                assert seed == scheme['resampling_seed']
                eligible = np.setdiff1d(np.arange(len(clouds[plant])), gallery)
            expected = np.random.RandomState(seed).choice(eligible, 1024, replace=False)
        assert np.array_equal(selected, expected), key
    checks['sampling_indices_seed_regenerated_and_spatial_constraints_checked'] = len(plan)
    keys = features['keys'].tolist()
    assert keys == [p['key'] for p in plan]
    assert features['tokens'].shape == (132, 64, 384)
    assert features['positions'].shape == (132, 64, 3)
    for name in features.files:
        if features[name].dtype.kind in 'fci':
            assert np.isfinite(features[name]).all(), name
    # Rebuild the exact official float32 reductions from each saved Token tensor,
    # without running the encoder again. This tests the shared-token constraint.
    device = torch.device(summary['runtime']['device'])
    with torch.inference_mode():
        for i in range(len(plan)):
            tokens = torch.as_tensor(features['tokens'][i].copy(), device=device)[None]
            mean = tokens.mean(1)[0].cpu().numpy()
            maximum = tokens.max(1).values[0].cpu().numpy()
            assert np.array_equal(mean, features['raw_mean'][i])
            assert np.array_equal(maximum, features['raw_max'][i])
    checks['l2_unit_norm_max_abs_error'] = {}
    for pooling in ('mean', 'mean_max'):
        raw = features['raw_mean'].astype(np.float64)
        if pooling == 'mean_max':
            raw = np.concatenate([raw, features['raw_max'].astype(np.float64)], axis=1)
        expected = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        assert features[pooling].dtype == np.float64
        assert np.max(np.abs(expected - features[pooling])) < 1e-15
        norm_error = float(np.max(np.abs(np.linalg.norm(features[pooling], axis=1) - 1)))
        assert norm_error < 32 * np.finfo(np.float64).eps
        checks['l2_unit_norm_max_abs_error'][pooling] = norm_error
    checks['same_token_readouts_exactly_reconstructed'] = 132
    lookup = {key: i for i, key in enumerate(keys)}
    rows = read_csv(output / 'retrieval_per_query.csv')
    assert len(rows) == 108
    for row in rows:
        assert float(row['overlap_fraction']) == int(row['overlap_count']) == 0
        qindex = lookup[row['input_key']]
        gallery_indices = [lookup[row['scheme'] + '_' + p + '_gallery'] for p in protocol['plant_ids']]
        identity = protocol['plant_ids'].index(row['plant_id'])
        for pooling in ('mean_max', 'mean'):
            # Independent implementation: individual dot products + Python sort.
            query = features[pooling][qindex]
            values = [float(np.clip(np.dot(features[pooling][g], query), -1., 1.))
                      for g in gallery_indices]
            order = sorted(range(4), key=lambda j: (-values[j], j))
            correct, wrong = values[identity], max(v for j, v in enumerate(values) if j != identity)
            for metric, expected in [('correct_cosine', correct), ('strongest_wrong_cosine', wrong),
                                     ('retrieval_margin', correct - wrong),
                                     ('top1_top2_gap', values[order[0]] - values[order[1]])]:
                check_close(row[pooling + '_' + metric], expected, row['input_key'] + '/' + pooling)
            assert row[pooling + '_prediction'] == protocol['plant_ids'][order[0]]
            assert int(row[pooling + '_correct']) == int(order[0] == identity)
        check_close(row['delta_margin'], float(row['mean_retrieval_margin']) -
                    float(row['mean_max_retrieval_margin']), 'paired delta')
    checks['paired_query_scores_independently_recomputed'] = len(rows)
    aggregate_count = 0
    for name in ('pooling_comparison.csv', 'per_plant_comparison.csv'):
        for aggregate in read_csv(output / name):
            selected = select_rows(rows, aggregate)
            assert int(aggregate['queries']) == len(selected) > 0
            for pooling in ('mean_max', 'mean'):
                hits = [int(r[pooling + '_correct']) for r in selected]
                margins = np.asarray([float(r[pooling + '_retrieval_margin']) for r in selected])
                for metric, expected in [('hits', sum(hits)), ('r1', np.mean(hits)),
                                         ('mean_margin', np.mean(margins)),
                                         ('median_margin', np.median(margins)),
                                         ('min_margin', np.min(margins)),
                                         ('positive_margin_fraction', np.mean(margins > 0))]:
                    check_close(aggregate[pooling + '_' + metric], expected, name + '/' + metric)
            wins = sum(int(r['mean_correct']) > int(r['mean_max_correct']) for r in selected)
            losses = sum(int(r['mean_correct']) < int(r['mean_max_correct']) for r in selected)
            assert int(aggregate['mean_hit_wins']) == wins
            assert int(aggregate['mean_hit_losses']) == losses
            assert int(aggregate['hit_ties']) == len(selected) - wins - losses
            aggregate_count += 1
    checks['stratified_aggregates_independently_recomputed'] = aggregate_count
    drift = read_csv(output / 'sampling_drift.csv')
    assert len(drift) == 12
    for row in drift:
        assert float(row['overlap_fraction']) == int(row['overlap_count']) == 0
        qi, gi = lookup[row['input_key']], lookup[row['gallery_key']]
        for pooling in ('mean_max', 'mean'):
            q, g = features[pooling][qi], features[pooling][gi]
            check_close(row[pooling + '_l2_drift'], np.linalg.norm(q - g), 'resampling drift')
            check_close(row[pooling + '_cosine'], np.clip(np.dot(q, g), -1, 1), 'resampling cosine')
        check_close(row['delta_l2_drift'], float(row['mean_l2_drift']) -
                    float(row['mean_max_l2_drift']), 'drift delta')
    checks['complete_resampling_pairs_checked'] = len(drift)
    for pooling in ('mean_max', 'mean'):
        check_close(summary['sampling_drift'][pooling + '_mean_l2'],
                    np.mean([float(r[pooling + '_l2_drift']) for r in drift]), 'mean drift')
    assert {p: sum(r['plant_id'] == p for r in rows) for p in protocol['plant_ids']} == {
        p: 27 for p in protocol['plant_ids']}
    assert summary['biological_n'] == 4
    # R@1 is reported as both balanced query average and plant macro average;
    # no repeated query is treated as an independent biological sample.
    for pooling in ('mean_max', 'mean'):
        check_close(summary['overall'][pooling + '_r1'],
                    np.mean([np.mean([int(r[pooling + '_correct']) for r in rows if r['plant_id'] == p])
                             for p in protocol['plant_ids']]), 'plant macro R@1')
    for path in output.glob('*.csv'):
        for row in read_csv(path):
            for text in row.values():
                try:
                    number = float(text)
                except ValueError:
                    continue
                assert math.isfinite(number), (path, text)
    for path in output.glob('*.json'):
        finite_json(json.loads(path.read_text()))
    checks['all_npz_csv_json_numeric_values_finite'] = True
    # Both CLI entry points must reject overwriting a populated directory.
    artifact_hashes = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    for flag in ('--prepare', '--run'):
        process = subprocess.run([sys.executable, '-m', 'agri.plant_gs.jepa.pooling_stage35b',
                                  flag, '--output', str(output)], cwd=ROOT,
                                 capture_output=True, text=True)
        assert process.returncode != 0 and 'refusing to overwrite' in process.stderr
    assert artifact_hashes == {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    checks['prepare_and_run_existing_output_overwrite_rejected'] = True
    from PIL import Image
    with Image.open(output / 'retrieval_margin.png') as image:
        assert image.width > 1000 and image.height > 500
        assert np.asarray(image).std() > 1
    assert (output / 'retrieval_margin.pdf').stat().st_size > 1000
    checks['plot_files_valid'] = True
    if reference is not None:
        reference = reference.resolve()
        assert reference != output
        identical = []
        comparable = ['protocol.json', 'sampling_plan.json', 'summary.json', 'diagnosis_report.md',
                     'retrieval_per_query.csv', 'pooling_comparison.csv',
                     'per_plant_comparison.csv', 'sampling_drift.csv']
        if amendment:
            comparable.append('validation_amendment.json')
        for name in comparable:
            assert (output / name).read_bytes() == (reference / name).read_bytes(), name
            identical.append(name)
        array_count = 0
        for name in ['sampling_indices.npz', 'encoder_features.npz']:
            with np.load(output / name, allow_pickle=False) as left, np.load(reference / name, allow_pickle=False) as right:
                assert left.files == right.files
                for key in left.files:
                    assert np.array_equal(left[key], right[key]), (name, key)
                    array_count += 1
        checks['cross_process_reproduction'] = {
            'reference_directory': str(reference), 'byte_identical': identical,
            'npz_arrays_elementwise_identical': array_count, 'passed': True}
    elif (output / 'verification.json').is_file():
        previous = json.loads((output / 'verification.json').read_text())
        if 'cross_process_reproduction' in previous:
            checks['cross_process_reproduction'] = previous['cross_process_reproduction']
    checks['all_checks_passed'] = True
    dump_json(output / 'verification.json', checks)
    reproduced = checks.get('cross_process_reproduction', {}).get('passed', False)
    dump_json(output / 'run_completion.json', {
        'complete': reproduced, 'numerical_validation_passed': True,
        'cross_process_reproduction_passed': reproduced,
        'protocol_sha256': summary['protocol_sha256'], 'protected_files_unchanged': len(before)})
    print(json.dumps(checks, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    verify(args.output.resolve(), args.reference)


if __name__ == '__main__':
    main()
