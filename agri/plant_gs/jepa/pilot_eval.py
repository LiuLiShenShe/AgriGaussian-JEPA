#!/usr/bin/env python3
"""Paired before/after representation probes on the fixed held-out plants."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri.plant_gs.jepa.dataset import load_gaussian_ply, normalize_gaussians
from agri.plant_gs.jepa.pilot_data import EXPERIMENT_ROOT, RESULT_ROOT, write_json
from agri.plant_gs.jepa.pilot_train import safe_torch_load
from utils.config import cfg_from_yaml_file

DEFAULT_CONFIG = Path(__file__).with_name('pilot_config.yaml')
DEFAULT_AUDIT = EXPERIMENT_ROOT / 'data_audit.json'
DEFAULT_SPLIT = EXPERIMENT_ROOT / 'split.json'
DIRECTIONS = np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
SAMPLE_SEEDS = (4101, 4102, 4103)
MASK_RATIOS = (0.30, 0.55, 0.70)


def spatial_cap_keep_mask(xyz: np.ndarray, ratio: float,
                          direction: np.ndarray) -> np.ndarray:
    """Remove an exact upper projection cap, a contiguous half-space region."""
    xyz = np.asarray(xyz)
    direction = np.asarray(direction, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'expected N x 3 xyz, got {xyz.shape}')
    if not 0 < ratio < 1 or len(xyz) < 3:
        raise ValueError('ratio must be between zero and one and N >= 3')
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError('direction must be a finite 3-vector')
    remove_count = min(len(xyz) - 1, max(1, int(round(len(xyz) * ratio))))
    projection = xyz.astype(np.float64) @ direction
    order = np.argsort(projection, kind='stable')
    keep = np.ones(len(xyz), dtype=bool)
    keep[order[-remove_count:]] = False
    return keep


def load_test_data(audit: dict, plant_ids: list[str]) -> dict[str, np.ndarray]:
    lookup = {row['plant_id']: row for row in audit['records']}
    clouds = {}
    for plant_id in plant_ids:
        decoded = load_gaussian_ply(Path(lookup[plant_id]['ply']))
        normalized, _ = normalize_gaussians(decoded, 'pretrain')
        clouds[plant_id] = normalized
    return clouds


def make_fixed_samples(clouds: dict[str, np.ndarray], num_gaussians: int,
                       index_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    plant_ids = sorted(clouds)
    if index_path.is_file():
        saved = np.load(index_path, allow_pickle=False)
        saved_ids = saved['plant_ids'].tolist()
        indices = saved['indices']
        if saved_ids != plant_ids or indices.shape != (
                len(plant_ids), len(SAMPLE_SEEDS), num_gaussians):
            raise ValueError('saved evaluation sampling protocol does not match current split/config')
        samples = np.stack([
            np.ascontiguousarray(clouds[plant_id][indices[i]])
            for i, plant_id in enumerate(plant_ids)
        ])
        return plant_ids, indices, samples

    index_rows, sample_rows = [], []
    for plant_id in plant_ids:
        data = clouds[plant_id]
        plant_indices, plant_samples = [], []
        for seed in SAMPLE_SEEDS:
            indices = np.random.RandomState(seed).choice(
                len(data), num_gaussians, replace=len(data) < num_gaussians
            ).astype(np.int64)
            plant_indices.append(indices)
            plant_samples.append(data[indices])
        index_rows.append(plant_indices)
        sample_rows.append(plant_samples)
    indices = np.asarray(index_rows, dtype=np.int64)
    samples = np.asarray(sample_rows, dtype=np.float32)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        plant_ids=np.asarray(plant_ids),
        sample_seeds=np.asarray(SAMPLE_SEEDS),
        indices=indices,
    )
    return plant_ids, indices, samples


@torch.inference_mode()
def encode_points(model, points: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(points, dtype=torch.float32, device=device).unsqueeze(0)
    neighborhoods, centers = model.group_divider(tensor)
    positions = centers[..., model.JEPA_encoder.pos_feature_dim]
    tokens = model.JEPA_encoder.forward_all(neighborhoods, positions)
    pooled = F.normalize(torch.cat([tokens.mean(1), tokens.max(1).values], dim=-1), dim=-1)
    if not torch.isfinite(tokens).all() or not torch.isfinite(pooled).all():
        raise FloatingPointError('non-finite evaluation embedding')
    return pooled[0].float().cpu().numpy()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(denom, 1e-12))


def calculate_metrics(model, plant_ids: list[str], samples: np.ndarray,
                      device: torch.device) -> tuple[dict, dict]:
    model.eval()
    full = np.empty((len(plant_ids), len(SAMPLE_SEEDS), 768), dtype=np.float32)
    for plant_index in range(len(plant_ids)):
        for sample_index in range(len(SAMPLE_SEEDS)):
            full[plant_index, sample_index] = encode_points(
                model, samples[plant_index, sample_index], device
            )

    per_plant_drift = []
    same_plant_cos = []
    for plant_index, plant_id in enumerate(plant_ids):
        drift_cos = [cosine(full[plant_index, 0], full[plant_index, j])
                     for j in range(1, len(SAMPLE_SEEDS))]
        drift_l2 = [float(np.linalg.norm(full[plant_index, 0] - full[plant_index, j]))
                    for j in range(1, len(SAMPLE_SEEDS))]
        per_plant_drift.append({
            'plant_id': plant_id,
            'mean_cosine_to_reference': float(np.mean(drift_cos)),
            'mean_l2_to_reference': float(np.mean(drift_l2)),
        })
        same_plant_cos.extend(
            cosine(full[plant_index, a], full[plant_index, b])
            for a, b in itertools.combinations(range(len(SAMPLE_SEEDS)), 2)
        )
    different_plant_cos = [
        cosine(full[i, a], full[j, b])
        for i, j in itertools.combinations(range(len(plant_ids)), 2)
        for a in range(len(SAMPLE_SEEDS))
        for b in range(len(SAMPLE_SEEDS))
    ]

    complete_reference = full[:, 0, :]
    partial_metrics = {}
    retrieval = {}
    partial_features = {}
    for ratio in MASK_RATIOS:
        ratio_key = f'{int(ratio * 100)}'
        row_metrics = []
        hits = []
        ratio_features = []
        for plant_index, plant_id in enumerate(plant_ids):
            full_sample = samples[plant_index, 0]
            for direction_index, direction in enumerate(DIRECTIONS):
                keep = spatial_cap_keep_mask(full_sample[:, :3], ratio, direction)
                partial_embedding = encode_points(model, full_sample[keep], device)
                complete_embedding = complete_reference[plant_index]
                similarities = np.asarray([
                    cosine(partial_embedding, gallery) for gallery in complete_reference
                ])
                predicted = int(similarities.argmax())
                hits.append(int(predicted == plant_index))
                row_metrics.append({
                    'plant_id': plant_id,
                    'direction': direction.tolist(),
                    'removed_ratio_requested': ratio,
                    'removed_gaussian_ratio_actual': float(1 - keep.mean()),
                    'partial_to_same_complete_cosine': cosine(partial_embedding, complete_embedding),
                    'partial_to_same_complete_l2': float(np.linalg.norm(partial_embedding - complete_embedding)),
                    'retrieved_plant_id': plant_ids[predicted],
                    'retrieval_correct': bool(predicted == plant_index),
                })
                ratio_features.append(partial_embedding)
        partial_metrics[ratio_key] = row_metrics
        retrieval[ratio_key] = {
            'r_at_1': float(np.mean(hits)),
            'correct_queries': int(sum(hits)),
            'query_count': len(hits),
            'independent_query_plants': len(plant_ids),
            'chance_r_at_1': 1 / len(plant_ids),
        }
        partial_features[ratio_key] = np.asarray(ratio_features, dtype=np.float32)

    plant_means = full.mean(axis=1).astype(np.float64)
    centered = plant_means - plant_means.mean(axis=0, keepdims=True)
    covariance_eigenvalues = np.linalg.eigvalsh(centered @ centered.T / len(plant_ids))
    covariance_eigenvalues = np.maximum(covariance_eigenvalues, 0)
    total_variance = float(covariance_eigenvalues.sum())
    participation_ratio = (
        float(total_variance ** 2 / np.square(covariance_eigenvalues).sum())
        if np.square(covariance_eigenvalues).sum() > 0 else 0.0
    )
    embedding_variance = float(np.var(plant_means, axis=0).mean())
    same_mean = float(np.mean(same_plant_cos))
    different_mean = float(np.mean(different_plant_cos))
    metrics = {
        'independent_test_plants': len(plant_ids),
        'complete_resampling_drift': {
            'per_plant': per_plant_drift,
            'mean_cosine_to_reference': float(np.mean([
                row['mean_cosine_to_reference'] for row in per_plant_drift
            ])),
            'mean_l2_to_reference': float(np.mean([
                row['mean_l2_to_reference'] for row in per_plant_drift
            ])),
            'sample_seeds': list(SAMPLE_SEEDS),
        },
        'same_vs_different_plant_cosine': {
            'same_plant_mean': same_mean,
            'same_plant_std': float(np.std(same_plant_cos)),
            'same_plant_pair_measurements': len(same_plant_cos),
            'different_plant_mean': different_mean,
            'different_plant_std': float(np.std(different_plant_cos)),
            'different_plant_pair_measurements': len(different_plant_cos),
            'different_independent_plant_pairs': len(plant_ids) * (len(plant_ids) - 1) // 2,
            'warning': 'Resampled inputs are repeated measurements within one plant, not independent biological samples.',
        },
        'spatial_contiguous_missing': {
            'mask_type': 'upper projection cap along a fixed XYZ axis; exact-ratio top-projection Gaussians removed before regrouping',
            'directions': DIRECTIONS.tolist(),
            'ratios': partial_metrics,
            'partial_to_complete_retrieval': retrieval,
        },
        'embedding_statistics': {
            'mean_dimension_variance_across_plant_means': embedding_variance,
            'participation_ratio_effective_rank': participation_ratio,
            'maximum_centered_rank_with_four_plants': max(0, len(plant_ids) - 1),
            'covariance_eigenvalues': covariance_eigenvalues.tolist(),
            'collapse_heuristic': {
                'variance_below_1e-8': embedding_variance < 1e-8,
                'effective_rank_below_1.1': participation_ratio < 1.1,
                'different_plant_cosine_above_0.99': different_mean > 0.99,
                'interpretation': 'Descriptive warning flags only; four held-out plants are insufficient to establish population-level collapse/non-collapse.',
            },
        },
    }
    arrays = {
        'plant_ids': np.asarray(plant_ids),
        'sample_seeds': np.asarray(SAMPLE_SEEDS),
        'full_embeddings': full,
        'plant_mean_embeddings': plant_means.astype(np.float32),
        **{f'partial_{key}_embeddings': value for key, value in partial_features.items()},
    }
    return metrics, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--trained', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--output', type=Path, default=RESULT_ROOT)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('official Gaussian grouping requires CUDA')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = cfg_from_yaml_file(str(args.config))
    audit = json.loads(args.audit.read_text())
    split = json.loads(args.split.read_text())
    plant_ids = sorted(split['test_plant_ids'])
    clouds = load_test_data(audit, plant_ids)
    index_path = args.output / 'fixed_test_sampling_indices.npz'
    plant_ids, indices, samples = make_fixed_samples(
        clouds, config.num_gaussians, index_path
    )
    del indices

    from models import build_model_from_cfg
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = {
        'test_plant_ids': plant_ids,
        'fixed_sample_indices': str(index_path.resolve()),
        'sample_seeds': list(SAMPLE_SEEDS),
        'num_gaussians_per_sample': config.num_gaussians,
        'models': {},
    }
    for label, checkpoint_path in (('random_init', args.baseline), ('pilot_trained', args.trained)):
        model = build_model_from_cfg(config.model).to(device)
        checkpoint = safe_torch_load(checkpoint_path)
        model.load_state_dict(checkpoint['base_model'], strict=True)
        metrics, arrays = calculate_metrics(model, plant_ids, samples, device)
        comparison['models'][label] = {
            'checkpoint': str(checkpoint_path.resolve()),
            'metrics': metrics,
        }
        np.savez_compressed(args.output / f'{label}_embeddings.npz', **arrays)
        write_json(args.output / f'{label}_metrics.json', comparison['models'][label])
        print(f'evaluated={label} plants={len(plant_ids)}')
        del model
        torch.cuda.empty_cache()

    write_json(args.output / 'evaluation_comparison.json', comparison)
    rows = []
    for label, data in comparison['models'].items():
        metrics = data['metrics']
        row = {
            'model': label,
            'resampling_mean_cosine': metrics['complete_resampling_drift']['mean_cosine_to_reference'],
            'resampling_mean_l2': metrics['complete_resampling_drift']['mean_l2_to_reference'],
            'same_plant_cosine_mean': metrics['same_vs_different_plant_cosine']['same_plant_mean'],
            'different_plant_cosine_mean': metrics['same_vs_different_plant_cosine']['different_plant_mean'],
            'embedding_dimension_variance': metrics['embedding_statistics']['mean_dimension_variance_across_plant_means'],
            'effective_rank': metrics['embedding_statistics']['participation_ratio_effective_rank'],
        }
        for ratio, score in metrics['spatial_contiguous_missing']['partial_to_complete_retrieval'].items():
            row[f'partial_r1_{ratio}pct'] = score['r_at_1']
        rows.append(row)
    import csv
    csv_path = args.output / 'metrics_comparison.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'comparison={args.output / "evaluation_comparison.json"}')


if __name__ == '__main__':
    main()
