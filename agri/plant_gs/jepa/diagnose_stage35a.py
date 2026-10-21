#!/usr/bin/env python3
"""Frozen-checkpoint, real-PLY diagnosis. Refuses to overwrite any output directory.

Run from Gaussian-JEPA:
  $AGRI_PY_JEPA -m \
      agri.plant_gs.jepa.diagnose_stage35a --device cuda:0
Reproduce into a fresh directory with --output /absolute/fresh/directory.
No optimizer, backward, checkpoint write, or training-mode forward is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri.plant_gs.jepa.diagnostic_metrics import (
    aggregate_retrieval, align_tokens, retrieval_rows, token_statistics, vector_statistics,
)
from agri.plant_gs.jepa.pilot_data import EXPERIMENT_ROOT, RESULT_ROOT, sha256_file
from agri.plant_gs.jepa.pilot_eval import (
    DEFAULT_CONFIG, DIRECTIONS, MASK_RATIOS, SAMPLE_SEEDS, load_test_data,
    spatial_cap_keep_mask,
)
from agri.plant_gs.jepa.pilot_train import safe_torch_load
from utils.config import cfg_from_yaml_file

DEFAULT_OUTPUT = ROOT / 'results/agri_diagnostics/stage_3_5a'
SEED = 20261015
QUERY_SEED = 351000
FEATURES = ('mean', 'max', 'mean_max')
PROTOCOLS = ('old_shared', 'disjoint_same_cap_same_count',
             'independent_full_cap_matched_count', 'independent_full_cap_1024')


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def dump_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protected_paths(audit):
    paths = []
    for directory in (EXPERIMENT_ROOT, RESULT_ROOT):
        paths.extend(p for p in directory.rglob('*') if p.is_file())
    for directory in ('models', 'datasets', 'utils'):
        paths.extend((ROOT / directory).rglob('*.py'))
    paths.extend((ROOT / 'agri/plant_gs/jepa' / f) for f in (
        'dataset.py', 'pilot_eval.py', 'pilot_train.py', 'pilot_data.py', 'pilot_config.yaml'))
    paths.extend(Path(r['ply']) for r in audit['records'])
    return sorted(set(paths))


def snapshot(paths):
    return {str(path): {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
            for path in paths}


def overlap(query, gallery):
    query, gallery = np.unique(query), np.unique(gallery)
    common = len(np.intersect1d(query, gallery))
    return {'query_unique_count': len(query), 'gallery_unique_count': len(gallery),
            'overlap_count': common, 'query_overlap_fraction': common / len(query),
            'gallery_overlap_fraction': common / len(gallery),
            'overlap_jaccard': common / len(np.union1d(query, gallery))}


def make_plan(clouds, plant_ids, fixed_indices):
    """Index plan shared by checkpoints; exclusion is on original PLY row IDs."""
    plan, arrays = [], {}
    for pi, plant_id in enumerate(plant_ids):
        cloud = clouds[plant_id]
        gallery_indices = fixed_indices[pi, 0]
        arrays[f'gallery_{plant_id}'] = gallery_indices
        for ri, ratio in enumerate(MASK_RATIOS):
            for axis, direction in enumerate(DIRECTIONS):
                keep_old = spatial_cap_keep_mask(cloud[gallery_indices, :3], ratio, direction)
                old_indices = gallery_indices[keep_old]
                keep_full = spatial_cap_keep_mask(cloud[:, :3], ratio, direction)
                projection = cloud[:, :3].astype(np.float64) @ direction
                old_threshold = float(projection[old_indices].max())
                same_region = projection <= old_threshold
                query_seed = QUERY_SEED + ri * 100 + axis
                for protocol in PROTOCOLS:
                    if protocol == 'old_shared':
                        indices = old_indices
                        eligible_count = len(old_indices)
                        threshold = old_threshold
                        pool_missing = 1 - len(old_indices) / len(gallery_indices)
                    else:
                        region = same_region if protocol == 'disjoint_same_cap_same_count' else keep_full
                        candidates = np.flatnonzero(region)
                        eligible = np.setdiff1d(candidates, gallery_indices, assume_unique=False)
                        n = 1024 if protocol == 'independent_full_cap_1024' else len(old_indices)
                        if len(eligible) < n:
                            raise ValueError(f'cannot obtain disjoint {n}-primitive query: {plant_id}')
                        indices = np.random.RandomState(query_seed).choice(eligible, n, replace=False)
                        eligible_count = len(eligible)
                        threshold = float(projection[candidates].max())
                        pool_missing = 1 - len(candidates) / len(cloud)
                    key = f'q{len(plan):03d}'
                    arrays[key] = indices.astype(np.int64)
                    item = {'key': key, 'plant_id': plant_id, 'protocol': protocol,
                            'missing_ratio': ratio, 'axis': axis, 'direction': direction.tolist(),
                            'query_seed': query_seed, 'gallery_seed': SAMPLE_SEEDS[0],
                            'query_gaussians': len(indices), 'original_gaussians': len(cloud),
                            'eligible_disjoint_count': eligible_count,
                            'actual_pool_missing_fraction': float(pool_missing),
                            'projection_threshold': threshold,
                            **overlap(indices, gallery_indices)}
                    if protocol != 'old_shared':
                        assert item['overlap_count'] == 0
                    plan.append(item)
    return plan, arrays


@torch.inference_mode()
def extract(model, points, device):
    tensor = torch.as_tensor(np.ascontiguousarray(points), device=device).unsqueeze(0)
    neighborhoods, centers = model.group_divider(tensor)
    positions = centers[..., model.JEPA_encoder.pos_feature_dim]
    captured = []
    handle = model.JEPA_encoder.encoder.register_forward_hook(
        lambda module, args, output: captured.append(output.detach()))
    try:
        tokens = model.JEPA_encoder.forward_all(neighborhoods, positions)
    finally:
        handle.remove()
    if tokens.shape != (1, 64, 384):
        raise ValueError(f'expected 1x64x384 encoder tokens, got {tokens.shape}')
    result = {'encoder_tokens': tokens[0].float().cpu().numpy(),
              'tokenizer_tokens': captured[0][0].float().cpu().numpy(),
              'positions': positions[0].float().cpu().numpy(),
              'mean': tokens.mean(1)[0].float().cpu().numpy(),
              'max': tokens.max(1).values[0].float().cpu().numpy()}
    result['mean_max'] = np.concatenate([result['mean'], result['max']])
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f'non-finite {name}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing directory: {output}')
    # Avoid accidental writes inside either immutable Stage 3 directory.
    for path in (RESULT_ROOT.resolve(), EXPERIMENT_ROOT.resolve()):
        if output == path or path in output.parents:
            raise ValueError('diagnostics must be outside Stage 3 output directories')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'agri/plant-gs-experiments':
        raise RuntimeError('run only on agri/plant-gs-experiments')
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # Preserve Stage 3's cuDNN convolution precision; disabling this changes
    # the saved random-init embeddings by ~2.9e-5. Repeated inference is tested.
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = cfg_from_yaml_file(str(DEFAULT_CONFIG))
    audit = json.loads((EXPERIMENT_ROOT / 'data_audit.json').read_text())
    split = json.loads((EXPERIMENT_ROOT / 'split.json').read_text())
    plant_ids = sorted(split['test_plant_ids'])
    all_ids = sorted(split['test_plant_ids'] + split['train_plant_ids'])
    paths = protected_paths(audit)
    print(f'fingerprinting {len(paths)} protected files', flush=True)
    before = snapshot(paths)
    for row in audit['records']:
        if before[row['ply']]['sha256'] != row['ply_sha256']:
            raise ValueError(f"PLY changed since Stage 3 audit: {row['plant_id']}")
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / 'protected_artifacts_before.json', before)
    saved = np.load(RESULT_ROOT / 'fixed_test_sampling_indices.npz', allow_pickle=False)
    assert saved['plant_ids'].tolist() == plant_ids
    indices = saved['indices']
    assert indices.shape == (4, 3, 1024)
    clouds = load_test_data(audit, all_ids)
    plan, plan_arrays = make_plan(clouds, plant_ids, indices)
    repeated_plan, repeated_arrays = make_plan(clouds, plant_ids, indices)
    assert plan == repeated_plan
    assert all(np.array_equal(plan_arrays[k], repeated_arrays[k]) for k in plan_arrays)
    # All19 gallery uses seed4101, the test galleries are exactly the Stage 3 galleries.
    gallery_indices = {}
    for plant_id in all_ids:
        if plant_id in plant_ids:
            gi = indices[plant_ids.index(plant_id), 0]
        else:
            gi = np.random.RandomState(SAMPLE_SEEDS[0]).choice(len(clouds[plant_id]), 1024, replace=False)
        gallery_indices[plant_id] = gi
        plan_arrays[f'gallery_{plant_id}'] = gi
    np.savez_compressed(output / 'sampling_indices.npz', complete_test_indices=indices,
                        plant_ids=np.asarray(plant_ids), sample_seeds=np.asarray(SAMPLE_SEEDS),
                        **plan_arrays)
    dump_json(output / 'sampling_plan.json', {
        'test_plant_ids': plant_ids, 'auxiliary_gallery_ids': all_ids,
        'gallery_split': {p: 'unseen_test' if p in plant_ids else 'seen_train' for p in all_ids},
        'query_seed_base': QUERY_SEED, 'global_seed': SEED, 'queries': plan,
        'normalization': 'Stage 3 full-asset pretrain normalization; never renormalize a partial query',
        'cap': 'remove largest projections; full-cap protocols use full original PLY; exact-count stable ties',
        'same_cap_control': 'full-PLY threshold equals largest projection retained in old sampled cap',
    })
    all_feature_rows, pooling_rows, token_pair_rows = [], [], []
    retrieval, token_summaries, reproducibility, replay = [], [], [], []
    plot_data = {}
    from models import build_model_from_cfg
    checkpoints = {'random_init': EXPERIMENT_ROOT / 'initial_random.pt',
                   'pilot_trained': EXPERIMENT_ROOT / 'pilot/checkpoint_last.pt'}
    for label, checkpoint_path in checkpoints.items():
        print(f'extracting {label}', flush=True)
        model = build_model_from_cfg(config.model).to(device)
        checkpoint = safe_torch_load(checkpoint_path)
        model.load_state_dict(checkpoint['base_model'], strict=True)
        del checkpoint
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        repeat_max_abs, repeat_count = 0., 0

        def checked_extract(points):
            nonlocal repeat_max_abs, repeat_count
            first = extract(model, points, device)
            second = extract(model, points, device)
            for name in first:
                error = float(np.max(np.abs(first[name].astype(np.float64) - second[name])))
                repeat_max_abs = max(repeat_max_abs, error)
                if error > 1e-6:
                    raise AssertionError(f'nonrepeatable {label}/{name}: {error}')
            repeat_count += 1
            return first

        complete = []
        for pi, plant_id in enumerate(plant_ids):
            complete.append([checked_extract(clouds[plant_id][ix]) for ix in indices[pi]])
        arrays = {name: np.asarray([[s[name] for s in plant] for plant in complete])
                  for name in complete[0][0]}
        arrays['plant_ids'] = np.asarray(plant_ids)
        matrices = {}
        for feature in FEATURES:
            rows, matrix = vector_statistics(arrays[feature], label, feature)
            all_feature_rows.extend(rows)
            pooling_rows.extend(rows)
            matrices[feature] = matrix
        for stage in ('encoder_tokens', 'tokenizer_tokens'):
            aligned, assignment, distances = align_tokens(arrays[stage], arrays['positions'])
            arrays[stage + '_reference_assignments'] = assignment
            arrays[stage + '_reference_center_distances'] = distances
            rows, matrix = vector_statistics(aligned, label, stage + '_spatial_aligned')
            all_feature_rows.extend(rows)
            pairs, stats, pair_assignments = token_statistics(arrays[stage], arrays['positions'], label, stage)
            token_pair_rows.extend(pairs)
            token_summaries.append(stats)
            arrays[stage + '_pairwise_assignments'] = pair_assignments
            matrices[stage + '_spatial_aligned'] = matrix
        # Reproduce saved Stage 3 encoder embeddings before interpreting any new result.
        old = np.load(RESULT_ROOT / f'{label}_embeddings.npz', allow_pickle=False)
        normalized = arrays['mean_max'].astype(np.float64)
        normalized /= np.linalg.norm(normalized, axis=-1, keepdims=True)
        full_error = float(np.abs(normalized - old['full_embeddings']).max())
        if full_error > 2e-5:
            raise AssertionError(f'Stage 3 full embeddings not reproduced: {full_error}')
        galleries = {}
        for plant_id in all_ids:
            galleries[plant_id] = (complete[plant_ids.index(plant_id)][0] if plant_id in plant_ids
                                    else checked_extract(clouds[plant_id][gallery_indices[plant_id]]))
        partial_arrays, old_partial = {}, {}
        for qi, item in enumerate(plan):
            plant_id = item['plant_id']
            features = checked_extract(clouds[plant_id][plan_arrays[item['key']]])
            for feature in FEATURES:
                partial_arrays[item['key'] + '_' + feature] = features[feature]
            if item['protocol'] == 'old_shared':
                n = features['mean_max'].astype(np.float64)
                n = n / np.linalg.norm(n)
                key = f"partial_{int(item['missing_ratio'] * 100)}_embeddings"
                stored = old[key][plant_ids.index(plant_id) * len(DIRECTIONS) + item['axis']]
                old_partial[item['key']] = float(np.max(np.abs(n - stored)))
            for feature in FEATURES:
                for scope, ids in (('heldout4_primary', plant_ids), ('mixed19_auxiliary', all_ids)):
                    metadata = {k: v for k, v in item.items() if k != 'direction'}
                    metadata.update({'model': label, 'feature': feature, 'candidate_scope': scope,
                                     'query_split': 'unseen_test',
                                     'candidate_seen_train': len(ids) - len(plant_ids)})
                    retrieval.append(retrieval_rows(
                        features[feature], np.stack([galleries[p][feature] for p in ids]),
                        ids, plant_id, metadata))
            if (qi + 1) % 36 == 0:
                print(f'{label}: {qi + 1}/{len(plan)} query conditions (each repeated)', flush=True)
        partial_error = max(old_partial.values())
        if partial_error > 2e-5:
            raise AssertionError(f'Stage 3 partial embeddings not reproduced: {partial_error}')
        np.savez_compressed(output / f'{label}_features.npz', **arrays)
        np.savez_compressed(output / f'{label}_retrieval_features.npz',
                            gallery_ids=np.asarray(all_ids),
                            **{'gallery_' + f: np.stack([galleries[p][f] for p in all_ids]) for f in FEATURES},
                            **partial_arrays)
        replay.append({'model': label, 'full_embedding_max_abs_error': full_error,
                       'partial_embedding_max_abs_error': partial_error,
                       'full_and_partial_reproduced_within': 2e-5})
        reproducibility.append({'model': label, 'repeated_forward_pairs': repeat_count,
                                'maximum_absolute_difference_all_features': repeat_max_abs,
                                'accepted_tolerance': 1e-6})
        plot_data[label] = {'features': {f: arrays[f] for f in FEATURES}, 'matrices': matrices}
        del model
        torch.cuda.empty_cache()

    comparison = aggregate_retrieval(retrieval)
    dump_csv(output / 'feature_statistics.csv', all_feature_rows)
    dump_csv(output / 'pooling_comparison.csv', pooling_rows)
    dump_csv(output / 'spatial_token_pair_statistics.csv', token_pair_rows)
    dump_csv(output / 'retrieval_per_query.csv', retrieval)
    dump_csv(output / 'retrieval_protocol_comparison.csv', comparison)
    dump_json(output / 'local_token_statistics.json', {'statistics': token_summaries})
    from agri.plant_gs.jepa.diagnostic_report import make_plots, write_report
    make_plots(output, plot_data, plant_ids)
    after = snapshot(protected_paths(audit))
    if before != after:
        raise AssertionError('protected Stage 3/model/PLY files changed')
    dump_json(output / 'protected_artifacts_after.json', after)
    summary = {
        'stage': '3.5-A diagnostic only', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'branch': 'agri/plant-gs-experiments', 'seed': SEED,
        'test_plant_ids': plant_ids, 'train_plant_ids_excluded_from_primary': split['train_plant_ids'],
        'checkpoints': {k: str(v) for k, v in checkpoints.items()},
        'checkpoint_sha256': {k: before[str(v)]['sha256'] for k, v in checkpoints.items()},
        'model_encoder_parameters': 21826432,
        'feature_statistics': all_feature_rows, 'token_statistics': token_summaries,
        'retrieval_comparison': comparison, 'stage3_replay': replay,
        'reproducibility': reproducibility,
        'protected_artifact_check': {'unchanged': True, 'file_count': len(before)},
        'index_plan_reproducible': True, 'all_independent_queries_zero_overlap': True,
        'finite_check': 'all feature arrays checked; JSON serialization disallows NaN/Infinity',
        'runtime_seconds': time.perf_counter() - started,
        'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                        'gpu': torch.cuda.get_device_name(device), 'device': str(device),
                        'cudnn_deterministic': True, 'cudnn_benchmark': False,
                        'matmul_tf32': False, 'cudnn_tf32': True},
        'interpretation_limits': [
            'Four independent held-out plants; 3 samples and 9 masks per plant are repeated measurements.',
            'Centered plant covariance rank <=3; repeated-measure rank <=11, neither diagnoses total collapse alone.',
            'Reference spatial alignment is normalized-coordinate matching, not semantic correspondence or rigid registration.',
            'Test cohort centering and PCA are descriptive only, never reported as leakage-free retrieval.',
            'L2 rescaling cannot alter cosine direction; raw cosine equals L2 cosine up to rounding.',
            'Mixed19 gallery includes 15 training plants and is auxiliary, not independent generalization.',
            'Pot, marker, residual background and coordinate orientation may encode identity.',
            'No new training or intervention: observational evidence cannot identify a training cause.',
        ],
    }
    # Strict recursive numeric validation includes every CSV field and summary field.
    def validate(value):
        if isinstance(value, dict):
            for v in value.values():
                validate(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                validate(v)
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise FloatingPointError('non-finite diagnostic output')
    validate(summary)
    for rows in (all_feature_rows, token_pair_rows, retrieval, comparison):
        validate(rows)
    dump_json(output / 'diagnostic_summary.json', summary)
    write_report(output, summary)
    completion = {
        'complete': False, 'output': str(output), 'protected_files_unchanged': True,
        'numerical_values_finite': True, 'all_inferences_repeated': True,
        'stage3_embeds_reproduced': True, 'stage35b_started': False}
    dump_json(output / 'run_completion.json', completion)
    from agri.plant_gs.jepa.verify_stage35a import verify
    verify(output)
    completion['complete'] = True
    completion['independent_verification_passed'] = True
    dump_json(output / 'run_completion.json', completion)
    print(f'diagnosis complete: {output} ({time.perf_counter() - started:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
