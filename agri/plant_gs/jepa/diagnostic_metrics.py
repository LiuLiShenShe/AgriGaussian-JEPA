"""Read-only Stage 3.5-A feature probes; no model or training code changes."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-15)


def spectrum(x):
    """Centered covariance spectrum via the small sample Gram matrix."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    eigen = np.maximum(np.linalg.eigvalsh(x @ x.T / len(x))[::-1], 0)
    total = eigen.sum()
    proportions = eigen / total if total > 0 else eigen * 0
    positive = proportions[proportions > 1e-14]
    return {
        'dimension_variance': float(np.square(x).mean()),
        'effective_rank_participation': float(1 / np.square(proportions).sum()) if total else 0.,
        'effective_rank_entropy': float(np.exp(-(positive * np.log(positive)).sum())) if total else 0.,
        'pc1_explained': float(proportions[0]),
        'pc2_explained': float(proportions[1]) if len(proportions) > 1 else 0.,
        'pc3_explained': float(proportions[2]) if len(proportions) > 2 else 0.,
        'maximum_centered_rank': min(len(x) - 1, x.shape[1]),
        'covariance_eigenvalues': eigen.tolist(),
    }


def match_positions(left, right):
    """Geometry-only one-to-one assignment. Token indices are never homologs."""
    cost = np.square(left[:, None, :].astype(np.float64) - right[None, :, :]).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    # This environment's SciPy extension returns arrays from a separate NumPy
    # runtime. Copy through Python integers so subsequent operations use ours.
    rows = np.asarray(rows.tolist(), dtype=np.int64)
    cols = np.asarray(cols.tolist(), dtype=np.int64)
    return rows, cols, np.sqrt(cost[rows, cols])


def align_tokens(tokens, positions):
    """Reference-center alignment for descriptive covariance, not retrieval."""
    flat_t = tokens.reshape(-1, *tokens.shape[-2:])
    flat_p = positions.reshape(-1, *positions.shape[-2:])
    aligned, assignments, distances = [], [], []
    for t, p in zip(flat_t, flat_p):
        rows, cols, distance = match_positions(flat_p[0], p)
        assert np.array_equal(rows, np.arange(len(rows)))
        aligned.append(t[cols].reshape(-1))
        assignments.append(cols)
        distances.append(distance)
    return (np.asarray(aligned).reshape(*tokens.shape[:2], -1),
            np.asarray(assignments), np.asarray(distances))


def vector_statistics(values, model, feature):
    """All 3 resamplings are repeated measures, never 12 independent plants."""
    p, r, d = values.shape
    raw = np.asarray(values, dtype=np.float64).reshape(p * r, d)
    ids = np.repeat(np.arange(p), r)
    mean = raw.mean(0)
    centered = raw - mean
    spaces = {'raw': raw, 'l2': unit(raw),
              'centered_raw': centered, 'centered_l2': unit(centered)}
    rows, matrices = [], {}
    pairs = list(itertools.combinations(range(len(raw)), 2))
    for space, x in spaces.items():
        sim = unit(x) @ unit(x).T
        matrices[space] = sim
        distances = np.linalg.norm(x[:, None] - x[None], axis=-1)
        same = [(i, j) for i, j in pairs if ids[i] == ids[j]]
        other = [(i, j) for i, j in pairs if ids[i] != ids[j]]
        norms = np.linalg.norm(x, axis=-1)
        stats = spectrum(x.reshape(p, r, d).mean(1))
        row = {
            'model': model, 'feature': feature, 'space': space,
            'plants': p, 'resamplings_per_plant': r, 'dimensions': d,
            'norm_mean': float(norms.mean()), 'norm_std': float(norms.std()),
            'norm_min': float(norms.min()), 'norm_max': float(norms.max()),
            'mean_vector_norm_to_mean_sample_norm': float(np.linalg.norm(x.mean(0)) / max(norms.mean(), 1e-15)),
            'same_cosine': float(np.mean([sim[i, j] for i, j in same])),
            'different_cosine': float(np.mean([sim[i, j] for i, j in other])),
            'same_l2': float(np.mean([distances[i, j] for i, j in same])),
            'different_l2': float(np.mean([distances[i, j] for i, j in other])),
            'same_pair_measurements': len(same), 'different_pair_measurements': len(other),
            'independent_plant_pairs': p * (p - 1) // 2,
            'residual_norm_to_raw_norm_mean': float(np.mean(np.linalg.norm(centered, axis=-1) / np.maximum(np.linalg.norm(raw, axis=-1), 1e-15))),
            **{k: v for k, v in stats.items() if k != 'covariance_eigenvalues'},
            'repeated_measure_effective_rank': spectrum(x)['effective_rank_participation'],
        }
        rows.append(row)
    return rows, matrices


def token_statistics(tokens, positions, model, stage='encoder_tokens'):
    """Both pairwise spatial matching and unpooled token dispersion are saved."""
    flat = tokens.astype(np.float64).reshape(-1, *tokens.shape[-2:])
    centers = positions.reshape(-1, *positions.shape[-2:])
    plant_ids = np.repeat(np.arange(tokens.shape[0]), tokens.shape[1])
    common = flat.reshape(-1, flat.shape[-1]).mean(0)
    centered = flat - common
    norms = np.linalg.norm(flat, axis=-1)
    rows, assignments = [], []
    for i, j in itertools.combinations(range(len(flat)), 2):
        a, b, distances = match_positions(centers[i], centers[j])
        assignments.append([i, j, *b.tolist()])
        for space, values in [('raw', flat), ('l2', unit(flat)),
                              ('centered_raw', centered), ('centered_l2', unit(centered))]:
            left, right = values[i, a], values[j, b]
            rows.append({
                'model': model, 'stage': stage, 'left_sample': i, 'right_sample': j,
                'same_plant': bool(plant_ids[i] == plant_ids[j]), 'space': space,
                'spatial_center_distance_mean': float(distances.mean()),
                'spatial_center_distance_max': float(distances.max()),
                'token_cosine': float(np.sum(unit(left) * unit(right), axis=-1).mean()),
                'token_l2': float(np.linalg.norm(left - right, axis=-1).mean()),
            })
    all_token_spec = spectrum(flat.reshape(-1, flat.shape[-1]))
    # Spectrum over 768 tokens is descriptive; it is not 768 biological samples.
    summary = {
        'model': model, 'stage': stage, 'per_token_norm_mean': float(norms.mean()),
        'per_token_norm_std': float(norms.std()),
        'common_norm_to_mean_token_norm': float(np.linalg.norm(common) / norms.mean()),
        'within_cloud_token_dimension_variance': float(np.var(flat, axis=1).mean()),
        'all_tokens_dimension_variance': float(np.var(flat, axis=(0, 1)).mean()),
        'across_plant_token_mean_dimension_variance': float(np.var(tokens.astype(np.float64).mean((1, 2)), axis=0).mean()),
        'all_token_spectrum_descriptive_only': all_token_spec,
    }
    for space in ('raw', 'l2', 'centered_raw', 'centered_l2'):
        for same, label in ((True, 'same'), (False, 'different')):
            subset = [r for r in rows if r['space'] == space and r['same_plant'] == same]
            summary[f'{space}_{label}_cosine'] = float(np.mean([r['token_cosine'] for r in subset]))
            summary[f'{space}_{label}_l2'] = float(np.mean([r['token_l2'] for r in subset]))
    summary['different_pair_center_distance_mean'] = float(np.mean([
        r['spatial_center_distance_mean'] for r in rows if r['space'] == 'raw' and not r['same_plant']]))
    return rows, summary, np.asarray(assignments, dtype=np.int64)


def retrieval_rows(query, gallery, gallery_ids, identity, metadata):
    similarities = unit(gallery) @ unit(query)
    target = gallery_ids.index(identity)
    order = np.argsort(-similarities, kind='stable')
    incorrect = similarities.copy()
    incorrect[target] = -np.inf
    return {
        **metadata, 'plant_id': identity, 'candidate_count': len(gallery_ids),
        'retrieved_id': gallery_ids[int(order[0])], 'hit': int(order[0] == target),
        'true_cosine': float(similarities[target]),
        'strongest_wrong_cosine': float(incorrect.max()),
        'retrieval_margin': float(similarities[target] - incorrect.max()),
        'top1_top2_gap': float(similarities[order[0]] - similarities[order[1]]),
        'true_rank': int(np.where(order == target)[0][0] + 1),
    }


def aggregate_retrieval(rows):
    keys = ('model', 'feature', 'protocol', 'candidate_scope', 'candidate_count', 'missing_ratio')
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in groups.items():
        result.append({
            **dict(zip(keys, key)), 'query_plants': len(set(r['plant_id'] for r in group)),
            'query_measurements': len(group), 'r_at_1': float(np.mean([r['hit'] for r in group])),
            **{name + '_mean': float(np.mean([r[name] for r in group]))
               for name in ('true_cosine', 'strongest_wrong_cosine', 'retrieval_margin',
                            'top1_top2_gap', 'query_overlap_fraction', 'gallery_overlap_fraction')},
            'retrieval_margin_min': float(min(r['retrieval_margin'] for r in group)),
            'retrieval_margin_max': float(max(r['retrieval_margin'] for r in group)),
        })
    return result
