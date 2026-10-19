#!/usr/bin/env python3
"""Preregister and run a paired, frozen-encoder pooling experiment on real PLYs.

Prepare once before inference, then run; --replay-from copies the exact frozen
protocol into a new directory. Existing completed results are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri.plant_gs.jepa.dataset import load_gaussian_ply, normalize_gaussians
from agri.plant_gs.jepa.diagnose_stage35a import protected_paths, snapshot
from agri.plant_gs.jepa.pilot_data import EXPERIMENT_ROOT, RESULT_ROOT, sha256_file
from agri.plant_gs.jepa.pilot_eval import DEFAULT_CONFIG, spatial_cap_keep_mask
from agri.plant_gs.jepa.pilot_train import safe_torch_load
from utils.config import cfg_from_yaml_file

STAGE_A = ROOT / 'results/agri_diagnostics/stage_3_5a'
DEFAULT_OUTPUT = ROOT / 'results/agri_diagnostics/stage_3_5b'
PLANTS = ['DouBanLv2', 'XianKeLai1', 'XianKeLai2', 'XiangPiShu2']
NEW_SCRIPTS = ['pooling_stage35b.py', 'pooling_stage35b_report.py', 'verify_stage35b.py']


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def validate_source_hashes(protocol, output):
    """Permit only explicitly logged validation fixes, preserving protocol bytes."""
    amendment_path = output / 'validation_amendment.json'
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else {}
    if amendment:
        if amendment['protocol_sha256'] != sha256_file(output / 'protocol.json'):
            raise ValueError('amendment does not refer to this frozen protocol')
        if not set(amendment['source_revisions']) <= {'pooling_stage35b.py', 'verify_stage35b.py'}:
            raise ValueError('an analysis or scoring change cannot use a validation amendment')
    for name, expected in protocol['source_sha256'].items():
        actual = sha256_file(Path(__file__).with_name(name))
        if actual != expected:
            revision = amendment.get('source_revisions', {}).get(name, {})
            if revision.get('original_sha256') != expected or revision.get('revised_sha256') != actual:
                raise ValueError(f'script changed after preregistration without matching amendment: {name}')
    return amendment


def audit_seeds():
    """Collect saved seeds AND epoch-derived seeds, with source fingerprints."""
    used, evidence = set(), []

    def walk(value, label='', active=False):
        found = []
        if isinstance(value, dict):
            for key, item in value.items():
                found.extend(walk(item, label + '/' + key, active or 'seed' in key.lower()))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                found.extend(walk(item, label + '/' + str(i), active))
        elif active and isinstance(value, int) and not isinstance(value, bool):
            found.append((label, value))
        return found

    for directory in (EXPERIMENT_ROOT, RESULT_ROOT, STAGE_A):
        for path in sorted(directory.rglob('*.json')):
            # Hash inventories contain filenames, not experimental seeds.
            if path.name.startswith('protected_artifacts_'):
                continue
            found = walk(json.loads(path.read_text()))
            if found:
                vals = sorted({value for _, value in found})
                used.update(vals)
                evidence.append({'source': str(path), 'sha256': sha256_file(path),
                                 'seeds': vals, 'method': 'recursive saved JSON seed fields'})
        for path in sorted(directory.rglob('*.npz')):
            vals = set()
            with np.load(path, allow_pickle=False) as saved:
                for key in saved.files:
                    if 'seed' in key.lower():
                        vals.update(int(v) for v in np.asarray(saved[key]).reshape(-1))
            if vals:
                used.update(vals)
                evidence.append({'source': str(path), 'sha256': sha256_file(path),
                                 'seeds': sorted(vals), 'method': 'saved NPZ seed arrays'})
    for name in ('sanity', 'pilot'):
        path = EXPERIMENT_ROOT / name / 'run_config.json'
        cfg = json.loads(path.read_text())
        vals = list(range(cfg['seed'], cfg['seed'] + cfg['epochs']))
        used.update(vals)
        evidence.append({'source': str(path), 'sha256': sha256_file(path), 'seeds': vals,
                         'method': 'PlantGaussianDataset: base seed + zero-based epoch; '
                                   'also DataLoader shuffle; sanity has no saved index archive'})
    # Verify source constants used by the two historical evaluation runners.
    from agri.plant_gs.jepa.diagnose_stage35a import QUERY_SEED, SEED
    from agri.plant_gs.jepa.pilot_eval import SAMPLE_SEEDS
    source_values = sorted(set(SAMPLE_SEEDS) | {SEED, QUERY_SEED} |
                           {QUERY_SEED + r * 100 + a for r in range(3) for a in range(3)})
    used.update(source_values)
    source_names = ['dataset.py', 'pilot_train.py', 'pilot_eval.py',
                    'diagnose_stage35a.py', 'pilot_config.yaml']
    return {'used_seeds': sorted(used), 'evidence': evidence,
            'source_derived_evaluation_seeds': source_values,
            'reviewed_source_sha256': {n: sha256_file(Path(__file__).with_name(n))
                                      for n in source_names},
            'scope': 'all saved Stage 3/3.5-A JSON and NPZ seed fields, '
                     'both training runs epoch formula, and historical evaluation source constants'}


def protected_files():
    audit = json.loads((EXPERIMENT_ROOT / 'data_audit.json').read_text())
    paths = set(protected_paths(audit))
    paths.update(p for p in STAGE_A.rglob('*') if p.is_file())
    paths.update(p for p in Path(__file__).parent.rglob('*')
                 if p.is_file() and p.suffix in ('.py', '.yaml') and p.name not in NEW_SCRIPTS)
    return sorted(paths)


def validate_output(output):
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'agri/plant-gs-experiments':
        raise RuntimeError('run only on agri/plant-gs-experiments')
    for protected in (EXPERIMENT_ROOT, RESULT_ROOT, STAGE_A):
        protected = protected.resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('output must be outside all protected Stage 3/3.5-A directories')


def prepare(output, replay_from=None):
    validate_output(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing directory: {output}')
    if replay_from:
        protocol_text = (replay_from / 'protocol.json').read_text()
        protocol = json.loads(protocol_text)
    else:
        audit = audit_seeds()
        schemes = []
        for i in range(1, 4):
            base = 35200000 + i * 1000
            schemes.append({'id': f'scheme_{i}', 'gallery_seed': base + 1,
                            'resampling_seed': base + 2,
                            'query_seeds': [[base + 100 + r * 10 + a for a in range(3)]
                                            for r in range(3)]})
        new_seeds = [35200000]
        for scheme in schemes:
            new_seeds += [scheme['gallery_seed'], scheme['resampling_seed']]
            new_seeds += [s for row in scheme['query_seeds'] for s in row]
        if len(new_seeds) != len(set(new_seeds)) or set(new_seeds) & set(audit['used_seeds']):
            raise AssertionError('new seed reuse')
        data = json.loads((EXPERIMENT_ROOT / 'data_audit.json').read_text())
        records = {r['plant_id']: r for r in data['records']}
        split = json.loads((EXPERIMENT_ROOT / 'split.json').read_text())
        if sorted(split['test_plant_ids']) != PLANTS:
            raise ValueError('held-out plant IDs changed')
        checkpoint = EXPERIMENT_ROOT / 'pilot/checkpoint_last.pt'
        protocol = {
            'stage': '3.5-B', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'written_before_any_stage35b_inference': True,
            'branch': 'agri/plant-gs-experiments', 'plant_ids': PLANTS,
            'biological_sample_size': 4, 'gallery_candidates': 4,
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
            'config': str(DEFAULT_CONFIG), 'config_sha256': sha256_file(DEFAULT_CONFIG),
            'split': str(EXPERIMENT_ROOT / 'split.json'),
            'split_sha256': sha256_file(EXPERIMENT_ROOT / 'split.json'),
            'assets': [{'plant_id': p, 'ply': records[p]['ply'],
                        'sha256': records[p]['ply_sha256'],
                        'gaussian_count': records[p]['gaussian_count']} for p in PLANTS],
            'runtime_seed': new_seeds[0], 'new_seeds': new_seeds,
            'historical_seed_audit': audit, 'schemes': schemes,
            'num_gaussians': 1024, 'token_shape': [64, 384],
            'missing_ratios': [0.30, 0.55, 0.70],
            'directions': {'X': [1., 0., 0.], 'Y': [0., 1., 0.], 'Z': [0., 0., 1.]},
            'normalization': 'dataset.normalize_gaussians(full PLY, pretrain); '
                             'no partial renormalization or attribute changes',
            'occlusion': 'pilot_eval.spatial_cap_keep_mask on full normalized PLY; '
                         'remove round(N * ratio) largest projections, stable-sort ties',
            'query_sampling': 'RandomState(query_seed).choice(kept original row IDs minus '
                              'correct gallery IDs, 1024, replace=False); overlap must equal zero',
            'complete_resampling': 'one full-asset probe per plant per scheme; '
                                   'independent seed, exclude gallery IDs, 1024 without replacement',
            'pooling': {'mean_max': 'L2(concat(token mean, token max)), 768 dimensions',
                        'mean': 'L2(token mean), 384 dimensions'},
            'precision': 'official float32 encoder and torch mean/max; shared tokens; '
                         'cast pooled vectors to float64 before L2, cosine, margin and drift',
            'retrieval_ties': 'stable descending cosine, gallery ordered by fixed plant IDs; '
                              'exact ties choose first ID, separately recorded',
            'paired_outcomes': 'win/loss refer to Mean hit and Mean+Max miss or reverse; '
                               'both hit or both miss is a hit tie; margin ties use abs(delta)<=1e-12',
            'drift_primary': 'Euclidean distance of independently resampled L2-normalized '
                             'full-asset embeddings; also report cosine distance',
            'decision_rule': {
                'scope': 'conservative descriptive rule, not a statistical significance test',
                'stable_r1_gain': 'overall Mean R@1 strictly higher; no scheme lower and at '
                                  'least two schemes higher; at least two plants higher and '
                                  'no plant lower; no missing-ratio aggregate lower',
                'tradeoffs': 'report margin and resampling drift separately; '
                            'do not equate higher R@1 with improvement in these metrics',
                'action': 'if stable_r1_gain false retain original Mean+Max; '
                          'otherwise Mean is a candidate only, disclose all tradeoffs'},
            'statistics': 'plant is independent unit n=4; all 108 queries are repeated '
                          'measurements; report per-plant paired deltas without query-level p-values',
            'no_training': True, 'no_centering_whitening_projection': True,
            'runtime_precision': {'cudnn_deterministic': True, 'cudnn_benchmark': False,
                                  'cuda_matmul_tf32': False, 'cudnn_tf32': True},
            'source_sha256': {n: sha256_file(Path(__file__).with_name(n)) for n in NEW_SCRIPTS},
        }
        protocol_text = json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    print('fingerprinting protected Stage 3 and 3.5-A artifacts', flush=True)
    before = snapshot(protected_files())
    for asset in protocol['assets']:
        if before[asset['ply']]['sha256'] != asset['sha256']:
            raise ValueError(f"PLY changed: {asset['plant_id']}")
    if before[protocol['checkpoint']]['sha256'] != protocol['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(protocol_text)
    dump_json(output / 'protected_artifacts_before.json', before)
    if replay_from and (replay_from / 'validation_amendment.json').is_file():
        (output / 'validation_amendment.json').write_bytes(
            (replay_from / 'validation_amendment.json').read_bytes())
    print(f'protocol frozen before inference: {output / "protocol.json"}', flush=True)


def load_clouds(protocol):
    clouds, metadata = {}, {}
    for asset in protocol['assets']:
        cloud, info = normalize_gaussians(load_gaussian_ply(Path(asset['ply'])), 'pretrain')
        if len(cloud) != asset['gaussian_count']:
            raise ValueError('PLY row count changed')
        clouds[asset['plant_id']], metadata[asset['plant_id']] = cloud, info
    return clouds, metadata


def make_plan(protocol, clouds):
    items, indices = [], {}
    n = protocol['num_gaussians']
    for scheme in protocol['schemes']:
        for plant in protocol['plant_ids']:
            cloud = clouds[plant]
            key = scheme['id'] + '_' + plant
            gallery = np.random.RandomState(scheme['gallery_seed']).choice(len(cloud), n, replace=False)
            base = {'scheme': scheme['id'], 'plant_id': plant, 'gallery_seed': scheme['gallery_seed']}
            gallery_key = key + '_gallery'
            items.append({**base, 'key': gallery_key, 'kind': 'gallery',
                          'seed': scheme['gallery_seed']})
            indices[gallery_key] = gallery.astype(np.int64)
            eligible = np.setdiff1d(np.arange(len(cloud)), gallery)
            probe = np.random.RandomState(scheme['resampling_seed']).choice(eligible, n, replace=False)
            probe_key = key + '_resampling'
            indices[probe_key] = probe.astype(np.int64)
            items.append({**base, 'key': probe_key, 'kind': 'resampling', 'gallery_key': gallery_key,
                          'seed': scheme['resampling_seed'], 'overlap_count': 0,
                          'overlap_fraction': 0.0})
            for ri, ratio in enumerate(protocol['missing_ratios']):
                for ai, (axis, direction) in enumerate(protocol['directions'].items()):
                    keep = spatial_cap_keep_mask(cloud[:, :3], ratio, np.asarray(direction))
                    eligible = np.setdiff1d(np.flatnonzero(keep), gallery)
                    if len(eligible) < n:
                        raise ValueError(f'not enough disjoint primitives: {plant}/{ratio}/{axis}')
                    seed = scheme['query_seeds'][ri][ai]
                    query = np.random.RandomState(seed).choice(eligible, n, replace=False)
                    qkey = key + f'_query_{int(round(ratio * 100))}_{axis}'
                    common = len(np.intersect1d(query, gallery))
                    if common != 0:
                        raise AssertionError('nonzero Query/Gallery overlap')
                    indices[qkey] = query.astype(np.int64)
                    projection = cloud[:, :3].astype(np.float64) @ np.asarray(direction)
                    items.append({**base, 'key': qkey, 'kind': 'query', 'gallery_key': gallery_key,
                                  'seed': seed, 'missing_ratio': ratio, 'direction': axis,
                                  'eligible_count': len(eligible), 'overlap_count': common,
                                  'overlap_fraction': 0.0,
                                  'actual_missing_fraction': float(1 - keep.mean()),
                                  'projection_threshold': float(projection[keep].max())})
    return items, indices


@torch.inference_mode()
def extract_once(model, points, device):
    tensor = torch.as_tensor(np.ascontiguousarray(points), dtype=torch.float32, device=device)[None]
    groups, centers = model.group_divider(tensor)
    positions = centers[..., model.JEPA_encoder.pos_feature_dim]
    tokens = model.JEPA_encoder.forward_all(groups, positions)
    if tokens.shape != (1, 64, 384):
        raise ValueError(f'unexpected encoder token shape: {tokens.shape}')
    # Both readouts below use this SAME tensor; no separate model forward.
    mean, maximum = tokens.mean(1)[0], tokens.max(1).values[0]
    result = {'tokens': tokens[0].float().cpu().numpy(),
              'positions': positions[0].float().cpu().numpy(),
              'raw_mean': mean.float().cpu().numpy(), 'raw_max': maximum.float().cpu().numpy()}
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f'nonfinite {name}')
    for name, value in [('mean', result['raw_mean']),
                        ('mean_max', np.concatenate([result['raw_mean'], result['raw_max']]))]:
        value = value.astype(np.float64)
        norm = np.linalg.norm(value)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise FloatingPointError('invalid embedding norm')
        result[name] = value / norm
    return result


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(output, device_name):
    validate_output(output)
    expected_files = {'protocol.json', 'protected_artifacts_before.json'}
    if output.is_dir() and (output / 'validation_amendment.json').is_file():
        expected_files.add('validation_amendment.json')
    if not output.is_dir() or {p.name for p in output.iterdir()} != expected_files:
        raise FileExistsError('run requires a prepared directory; refusing to overwrite existing results')
    protocol = json.loads((output / 'protocol.json').read_text())
    protocol_hash = sha256_file(output / 'protocol.json')
    validate_source_hashes(protocol, output)
    before = json.loads((output / 'protected_artifacts_before.json').read_text())
    if snapshot([Path(p) for p in before]) != before:
        raise ValueError('protected artifacts changed since preregistration')
    seed = protocol['runtime_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    clouds, normalization = load_clouds(protocol)
    plan, indices = make_plan(protocol, clouds)
    dump_json(output / 'sampling_plan.json', {'normalization': normalization, 'items': plan})
    np.savez_compressed(output / 'sampling_indices.npz', **indices)
    if sum(p['kind'] == 'query' for p in plan) != 108 or len(plan) != 132:
        raise AssertionError('incomplete balanced plan')
    from models import build_model_from_cfg
    config = cfg_from_yaml_file(protocol['config'])
    model = build_model_from_cfg(config.model).to(device)
    checkpoint = safe_torch_load(Path(protocol['checkpoint']))
    model.load_state_dict(checkpoint['base_model'], strict=True)
    checkpoint_epoch = int(checkpoint['epoch'])
    del checkpoint
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state_before = state_digest(model)
    features = []
    print(f'extracting {len(plan)} inputs, once each; two readouts share every token tensor', flush=True)
    for i, item in enumerate(plan):
        features.append(extract_once(model, clouds[item['plant_id']][indices[item['key']]], device))
        if (i + 1) % 33 == 0:
            print(f'completed {i + 1}/{len(plan)} encoder forwards', flush=True)
    state_after = state_digest(model)
    if state_before != state_after or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('model state changed or gradients were populated')
    if any(m.training for m in model.modules()):
        raise AssertionError('model entered training mode')
    saved = {key: np.stack([f[key] for f in features]) for key in features[0]}
    np.savez_compressed(output / 'encoder_features.npz', keys=np.asarray([p['key'] for p in plan]), **saved)
    from agri.plant_gs.jepa.pooling_stage35b_report import analyze, write_outputs
    summary, tables = analyze(protocol, plan, saved)
    summary.update({
        'protocol_sha256': protocol_hash, 'checkpoint_epoch_zero_based': checkpoint_epoch,
        'encoder_forwards': len(plan), 'same_tokens_for_both_poolings': True,
        'model_state_sha256_before': state_before, 'model_state_sha256_after': state_after,
        'all_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
        'all_model_modules_eval': all(not m.training for m in model.modules()),
        'all_parameter_gradients_none': all(p.grad is None for p in model.parameters()),
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                    'device': device_name, 'gpu': torch.cuda.get_device_name(device)},
        'inference_started_after_protocol_created': True,
    })
    write_outputs(output, protocol, summary, tables)
    after = snapshot([Path(p) for p in before])
    if after != before or sha256_file(output / 'protocol.json') != protocol_hash:
        raise AssertionError('protected inputs or fixed protocol changed')
    dump_json(output / 'protected_artifacts_after.json', after)
    dump_json(output / 'run_completion.json', {'complete': False, 'protocol_sha256': protocol_hash,
                                             'protected_files_unchanged': len(before)})
    print('inference and tables complete; run verify_stage35b before considering the experiment complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--replay-from', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output)
    elif args.replay_from:
        prepare(output, args.replay_from.resolve())
        run(output, args.device)
    else:
        run(output, args.device)


if __name__ == '__main__':
    main()
