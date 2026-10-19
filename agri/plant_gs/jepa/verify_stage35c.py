#!/usr/bin/env python3
"""Independently check Stage 3.5-C tables and a separate-process reproduction.

Only NumPy, saved Stage 3.5-A vectors/indices, and generated diagnostic tables
are read. There is no model import, inference, resampling or training.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'results/agri_diagnostics/stage_3_5a'
OUTPUT = ROOT / 'results/agri_diagnostics/stage_3_5c'
ATOL = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def check(actual, expected, label):
    if isinstance(expected, str):
        if actual != expected:
            raise AssertionError(f'{label}: {actual!r} != {expected!r}')
    elif not math.isfinite(float(actual)) or abs(float(actual) - expected) > ATOL:
        raise AssertionError(f'{label}: {actual!r} != {expected!r}')


def check_fields(actual, expected, label):
    for key, value in expected.items():
        check(actual[key], value, f'{label}/{key}')


def cat(plant):
    return re.sub(r'\d+$', '', plant)


def reject_constant(value):
    raise ValueError(f'nonfinite JSON number: {value}')


def rank_stats(records):
    return {
        'queries': len(records),
        'mean_cosine': float(np.mean([r['cosine'] for r in records])),
        'min_cosine': min(r['cosine'] for r in records),
        'max_cosine': max(r['cosine'] for r in records),
        'mean_rank_19': float(np.mean([r['rank_19'] for r in records])),
        'median_rank_19': float(np.median([r['rank_19'] for r in records])),
        'min_rank_19': min(r['rank_19'] for r in records),
        'max_rank_19': max(r['rank_19'] for r in records),
        'top3_count': sum(r['rank_19'] <= 3 for r in records),
        'above_true_count': sum(r['beats_true'] for r in records),
        'false_top1_count': sum(r['is_top1_error'] for r in records),
    }


def query_stats(records):
    wrong = [r for r in records if not r['hit']]
    margin = [r['retrieval_margin'] for r in records]
    return {
        'queries': len(records), 'hits': sum(r['hit'] for r in records), 'errors': len(wrong),
        'r_at_1': sum(r['hit'] for r in records) / len(records),
        'mean_margin': float(np.mean(margin)), 'median_margin': float(np.median(margin)),
        'min_margin': min(margin), 'max_margin': max(margin),
        'mean_true_rank_19': float(np.mean([r['true_rank_19'] for r in records])),
        'seen_train_errors': sum(r['retrieved_split'] == 'seen_train' for r in wrong),
        'unseen_test_errors': sum(r['retrieved_split'] == 'unseen_test' for r in wrong),
        'same_named_category_errors': sum(r['same_named_category'] for r in wrong),
        'different_named_category_errors': sum(not r['same_named_category'] for r in wrong),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--reproduction', type=Path, required=True)
    args = parser.parse_args()
    output, reproduction = args.output.resolve(), args.reproduction.resolve()
    if output == reproduction:
        raise ValueError('a separate output directory is required for reproduction')
    for name in ('verification.json', 'run_completion.json'):
        if (output / name).exists():
            raise FileExistsError(f'refusing to overwrite {output / name}')
    original = [r for r in rows(SOURCE / 'retrieval_per_query.csv')
                if (r['model'], r['feature'], r['protocol'], r['candidate_scope']) ==
                ('pilot_trained', 'mean_max', 'independent_full_cap_1024', 'mixed19_auxiliary')]
    original.sort(key=lambda r: r['key'])
    keys = [r['key'] for r in original]
    plan = json.loads((SOURCE / 'sampling_plan.json').read_text())
    split = json.loads((ROOT / 'experiments/agri_pilot/split.json').read_text())
    plants, training = split['test_plant_ids'], split['train_plant_ids']
    with np.load(SOURCE / 'pilot_trained_retrieval_features.npz', allow_pickle=False) as f:
        ids = f['gallery_ids'].tolist()
        gallery = f['gallery_mean_max'].astype(np.float64)
        query = np.stack([f[key + '_mean_max'] for key in keys]).astype(np.float64)
    # Deliberately use a separate batched einsum implementation, not the
    # analysis script's per-query matrix-vector multiplication.
    gallery /= np.sqrt(np.sum(gallery * gallery, axis=1, keepdims=True))
    query /= np.sqrt(np.sum(query * query, axis=1, keepdims=True))
    scores = np.einsum('qd,gd->qg', query, gallery, optimize=False)
    if scores.shape != (36, 19) or not np.isfinite(scores).all():
        raise AssertionError('invalid independently recomputed score matrix')
    output_queries = {r['query_key']: r for r in rows(output / 'query_error_details.csv')}
    output_negatives = {(r['query_key'], r['candidate_plant_id']): r
                        for r in rows(output / 'hard_negative_ranking.csv')}
    if set(output_queries) != set(keys) or len(output_negatives) != 648:
        raise AssertionError('query/negative table coverage is wrong')
    rebuilt_queries, rebuilt_negatives = [], []
    max_score_error = 0.0
    with np.load(SOURCE / 'sampling_indices.npz', allow_pickle=False) as indices:
        for qi, old in enumerate(original):
            key, plant = old['key'], old['plant_id']
            order = np.argsort(-scores[qi], kind='stable').tolist()
            true = ids.index(plant)
            best_wrong = next(i for i in order if i != true)
            ranks = {i: order.index(i) + 1 for i in range(19)}
            if np.intersect1d(indices[key], indices['gallery_' + plant]).size:
                raise AssertionError('original query/gallery primitives overlap')
            rebuilt = {
                'query_plant_id': plant, 'query_named_category': cat(plant),
                'missing_ratio': float(old['missing_ratio']), 'axis': int(old['axis']),
                'retrieved_id': ids[order[0]], 'hit': int(order[0] == true),
                'true_cosine': float(scores[qi, true]),
                'strongest_wrong_cosine': float(scores[qi, best_wrong]),
                'retrieval_margin': float(scores[qi, true] - scores[qi, best_wrong]),
                'top1_top2_gap': float(scores[qi, order[0]] - scores[qi, order[1]]),
                'true_rank_19': ranks[true],
                'retrieved_split': plan['gallery_split'][ids[order[0]]],
                'same_named_category': int(cat(plant) == cat(ids[order[0]])),
                'query_correct_gallery_overlap_count': 0, 'query_correct_gallery_overlap_fraction': 0,
                'query_gaussians': 1024, 'gallery_gaussians': 1024,
                'query_seed': int(old['query_seed']), 'gallery_seed': int(old['gallery_seed']),
            }
            check_fields(output_queries[key], rebuilt, key)
            for field in ('true_cosine', 'strongest_wrong_cosine', 'retrieval_margin'):
                max_score_error = max(max_score_error, abs(float(old[field]) - rebuilt[field]))
            check_fields(old, {k: rebuilt[k] for k in ('retrieved_id', 'hit', 'true_cosine',
                                                       'strongest_wrong_cosine', 'retrieval_margin')}, key + '/old')
            heldout_order = [i for i in order if ids[i] in plants]
            rank4 = heldout_order.index(true) + 1
            added_ahead = sum(ids[i] in training for i in order[:ranks[true] - 1])
            check_fields(output_queries[key], {'true_rank_4': rank4, 'saved_hit_4': int(rank4 == 1),
                                              'added_seen_candidates_ahead': added_ahead}, key + '/rank-audit')
            if ranks[true] != rank4 + added_ahead:
                raise AssertionError('rank arithmetic failed')
            rebuilt_queries.append(rebuilt)
            for ci, candidate in enumerate(ids):
                if ci == true:
                    continue
                negative = {
                    'candidate_split': plan['gallery_split'][candidate],
                    'candidate_named_category': cat(candidate),
                    'same_named_category': int(cat(plant) == cat(candidate)),
                    'cosine': float(scores[qi, ci]), 'rank_19': ranks[ci],
                    'true_cosine': float(scores[qi, true]), 'true_rank_19': ranks[true],
                    'candidate_minus_true_cosine': float(scores[qi, ci] - scores[qi, true]),
                    'beats_true': int(scores[qi, ci] > scores[qi, true]),
                    'is_top1_error': int(ranks[ci] == 1),
                }
                if (key, candidate) not in output_negatives:
                    raise AssertionError(f'missing negative {key}/{candidate}')
                check_fields(output_negatives[key, candidate], negative, key + '/' + candidate)
                rebuilt_negatives.append({**negative, 'candidate_plant_id': candidate,
                                          'query_plant_id': plant, 'missing_ratio': float(old['missing_ratio'])})
    for row in rows(output / 'per_plant_errors.csv'):
        subset = [r for r in rebuilt_queries if r['query_plant_id'] == row['query_plant_id']]
        check_fields(row, query_stats(subset), 'per-plant/' + row['query_plant_id'])
        distribution = collections.Counter(r['retrieved_id'] for r in subset if not r['hit'])
        if json.loads(row['wrong_candidate_distribution']) != dict(distribution):
            raise AssertionError('per-plant wrong candidate distribution differs')
    for name, key in (('occlusion_errors.csv', 'missing_ratio'), ('axis_errors.csv', 'axis')):
        for row in rows(output / name):
            subset = [r for r in rebuilt_queries if r[key] == float(row[key])]
            check_fields(row, query_stats(subset), name + '/' + row[key])
    for row in rows(output / 'candidate_error_summary.csv'):
        subset = [r for r in rebuilt_negatives if r['candidate_plant_id'] == row['candidate_plant_id']]
        check_fields(row, rank_stats(subset), 'candidate/' + row['candidate_plant_id'])
    for row in rows(output / 'focus_candidate_statistics.csv'):
        subset = [r for r in rebuilt_negatives if r['candidate_plant_id'] == row['candidate_plant_id']
                  and r['query_plant_id'] == row['query_plant_id']
                  and (row['missing_ratio'] == 'all' or r['missing_ratio'] == float(row['missing_ratio']))]
        check_fields(row, rank_stats(subset), 'focus/' + row['candidate_plant_id'])
    for row in rows(output / 'confusion_matrix.csv'):
        distribution = collections.Counter(r['retrieved_id'] for r in rebuilt_queries
                                           if r['query_plant_id'] == row['query_plant_id'])
        check_fields(row, {candidate: distribution[candidate] for candidate in ids}, 'confusion')
    summary = json.loads((output / 'summary.json').read_text())
    check_fields(summary['overall'], query_stats(rebuilt_queries), 'overall')
    for path, expected in summary['source_sha256'].items():
        if sha256(path) != expected:
            raise AssertionError(f'source changed: {path}')
    before = json.loads((output / 'protected_artifacts_before.json').read_text())
    after = json.loads((output / 'protected_artifacts_after.json').read_text())
    if before != after:
        raise AssertionError('before/after protected artifact inventories differ')
    for path, fingerprint in before.items():
        if Path(path).stat().st_size != fingerprint['bytes'] or sha256(path) != fingerprint['sha256']:
            raise AssertionError(f'protected input now differs: {path}')
    csv_files = sorted(p.name for p in output.glob('*.csv'))
    numeric_cells = 0
    for name in csv_files:
        for row in rows(output / name):
            for value in row.values():
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    raise AssertionError(f'nonfinite CSV value in {name}')
                numeric_cells += 1
    for path in output.glob('*.json'):
        json.loads(path.read_text(), parse_constant=reject_constant)
    comparison_files = csv_files + ['summary.json', 'analysis_checks.json', 'error_analysis.md',
                                    'error_case_manifest.json', 'protected_artifacts_before.json',
                                    'protected_artifacts_after.json']
    for name in comparison_files:
        if (output / name).read_bytes() != (reproduction / name).read_bytes():
            raise AssertionError(f'independent-process reproduction differs: {name}')
    with np.load(output / 'saved_cosine_scores.npz', allow_pickle=False) as primary, \
            np.load(reproduction / 'saved_cosine_scores.npz', allow_pickle=False) as replay:
        if primary.files != replay.files:
            raise AssertionError('reproduction score archive keys differ')
        for key in primary.files:
            if not np.array_equal(primary[key], replay[key]):
                raise AssertionError(f'reproduction array differs: {key}')
        if not np.allclose(primary['cosine_scores'], scores, rtol=0, atol=ATOL):
            raise AssertionError('independent score calculation differs')
        array_count = len(primary.files)
    from PIL import Image
    for name in ('confusion_matrix.png', 'error_cases.png'):
        with Image.open(output / name) as a, Image.open(reproduction / name) as b:
            if not np.array_equal(np.asarray(a), np.asarray(b)):
                raise AssertionError(f'reproduced plot pixels differ: {name}')
    verification = {
        'passed': True, 'independent_score_formula': 'float64 L2 then batched numpy.einsum',
        'saved_score_max_abs_error': max_score_error, 'score_atol': ATOL,
        'queries_checked': len(rebuilt_queries), 'gallery_candidates': len(ids),
        'hard_negative_rows_checked': len(rebuilt_negatives), 'numeric_csv_cells_finite': numeric_cells,
        'all_json_numeric_literals_finite': True,
        'all_original_index_intersections_zero': True, 'all_summary_tables_reconciled': True,
        'protected_files_unchanged': True, 'protected_file_count': len(before),
        'independent_process_reproduction': {
            'directory': str(reproduction), 'byte_identical_files': comparison_files,
            'npz_arrays_elementwise_identical': array_count,
            'plot_png_pixels_identical': ['confusion_matrix.png', 'error_cases.png'],
            'pdf_byte_comparison_excluded_reason': 'PDF creation-time metadata can differ; PNG pixels compared instead.',
        },
    }
    (output / 'verification.json').write_text(json.dumps(verification, indent=2, allow_nan=False) + '\n')
    (output / 'run_completion.json').write_text(json.dumps({
        'stage': '3.5-C', 'complete': True, 'verification_passed': True,
        'no_model_forward_or_training': True, 'no_new_sampling_or_candidate_expansion': True,
        'official_mean_max_preserved': True, 'next_stage_started': False,
        'error_analysis_sha256': sha256(output / 'error_analysis.md'),
        'summary_sha256': sha256(output / 'summary.json'),
    }, indent=2, allow_nan=False) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
