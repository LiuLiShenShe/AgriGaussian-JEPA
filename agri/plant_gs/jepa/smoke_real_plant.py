#!/usr/bin/env python3
"""Real-plant Gaussian-JEPA contract test, with NO optimizer/EMA weight updates.

One actual PLY is sampled through PlantGaussianDataset, fed into CUDA grouping
and the unmodified full JEPA, and checked for finite losses and gradients. Eval
feature probes measure repeatability and input sensitivity only, NOT accuracy
or plant semantics. Training-mode BN running buffers may change in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri.plant_gs.jepa.dataset import PlantGaussianDataset
from agri.plant_gs.jepa.validation import gradient_report


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_sha256(model) -> str:
    """Hash weights only; BatchNorm running buffers are deliberately excluded."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def encode(points, model):
    neighborhoods, centers = model.group_divider(points)
    positions = centers[..., model.JEPA_encoder.pos_feature_dim]
    tokens = model.JEPA_encoder.forward_all(neighborhoods, positions)
    pooled = F.normalize(torch.cat([tokens.mean(1), tokens.max(1).values], -1), dim=-1)
    if not torch.isfinite(tokens).all() or not torch.isfinite(pooled).all():
        raise RuntimeError('non-finite eval features')
    return tokens, pooled, centers


def feature_probe(points, model):
    """Extract before train() mutates BN buffers; this is not a downstream test."""
    model.eval()
    tokens, embedding, centers = encode(points, model)
    repeated_tokens, repeated_embedding, _ = encode(points, model)
    if not torch.allclose(tokens, repeated_tokens, rtol=1e-5, atol=1e-6):
        raise RuntimeError('identical eval input did not reproduce token features')
    dc_zero = points.clone()
    dc_zero[..., 11:14] = 0
    _, dc_embedding, _ = encode(dc_zero, model)
    appearance_zero = points.clone()
    appearance_zero[..., 3:] = 0
    _, geometry_embedding, _ = encode(appearance_zero, model)
    return {
        'scope': 'repeatability_and_input_sensitivity_not_semantic_effectiveness',
        'token_shape': list(tokens.shape), 'embedding_shape': list(embedding.shape),
        'embedding_l2_norm': float(embedding.norm().cpu()),
        'repeat_max_abs_token_difference': float((tokens - repeated_tokens).abs().max().cpu()),
        'repeat_embedding_cosine': float(F.cosine_similarity(embedding, repeated_embedding).cpu()[0]),
        'zero_dc_embedding_l2_difference': float((embedding - dc_embedding).norm().cpu()),
        'zero_dc_embedding_cosine': float(F.cosine_similarity(embedding, dc_embedding).cpu()[0]),
        'zero_non_xyz_embedding_l2_difference': float((embedding - geometry_embedding).norm().cpu()),
        'zero_non_xyz_embedding_cosine': float(F.cosine_similarity(embedding, geometry_embedding).cpu()[0]),
        'ablations_are_model_space_diagnostics_not_valid_renderable_gaussians': True,
    }, {
        'token_features': tokens[0].cpu().numpy(),
        'embedding': embedding[0].cpu().numpy(),
        'group_centers': centers[0].cpu().numpy(),
    }


def load_optional_checkpoint(model, path):
    if path is None:
        return {'loaded': False, 'initialization': 'random', 'path': None}
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Reuse the official strict loader without importing tools/__init__.py.
    spec = importlib.util.spec_from_file_location('official_release_smoke', ROOT / 'tools/smoke_test_release.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.load_checkpoint_strict(model, path)
    return {'loaded': True, 'initialization': 'checkpoint', 'path': str(path),
            'sha256': file_sha256(path), 'strict': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'cfgs/pretrain/gaussian_jepa.yaml')
    parser.add_argument('--checkpoint', type=Path, help='Optional TRUSTED Gaussian-JEPA checkpoint; loaded strictly')
    parser.add_argument('--output', type=Path, required=True, help='JSON report; same-stem NPZ contains sampled data/features')
    parser.add_argument('--num-gaussians', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-seed', type=int, default=0, help='Separate model initialization from sampling seed')
    parser.add_argument('--normalization', choices=['pretrain', 'downstream'], default='pretrain')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The official FPS/KNN grouping requires an available CUDA device')
    if args.output.suffix.lower() != '.json':
        raise ValueError('--output must have .json extension')
    if args.seed < 0 or args.model_seed < 0:
        raise ValueError('seeds must be non-negative')
    from models import build_model_from_cfg
    from utils.config import cfg_from_yaml_file

    source_hash = file_sha256(args.ply)
    config = cfg_from_yaml_file(str(args.config))
    if args.num_gaussians < max(config.model.num_group, config.model.group_size):
        raise ValueError('sample budget is smaller than the FPS/KNN grouping requirements')
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    model = build_model_from_cfg(config.model)
    checkpoint = load_optional_checkpoint(model, args.checkpoint)
    model = model.to(device)
    weights_before = parameter_sha256(model)

    dataset = PlantGaussianDataset(
        [args.ply], num_gaussians=args.num_gaussians, seed=args.seed,
        normalization=args.normalization,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=True)
    batch = next(iter(loader))
    sample = dataset[0]
    points = batch['points'].to(device, dtype=torch.float32, non_blocking=True).contiguous()
    if tuple(points.shape) != (1, args.num_gaussians, 14) or not torch.isfinite(points).all():
        raise RuntimeError(f'invalid real PLY input: {tuple(points.shape)}')
    probes, feature_arrays = feature_probe(points, model)

    model.train()
    model.zero_grad(set_to_none=True)
    # Reset masks/projections RNG after eval probes, keeping the training test
    # independent of the number of diagnostic eval passes above.
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses = model(points)
    if not isinstance(losses, dict) or not losses:
        raise TypeError(f'expected non-empty loss dictionary, got {type(losses)!r}')
    loss_values = {}
    for name, value in losses.items():
        if value.ndim != 0 or not torch.isfinite(value):
            raise RuntimeError(f'invalid loss {name}: {value}')
        loss_values[name] = float(value.detach().cpu())
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise RuntimeError('non-finite total loss')
    total.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    gradients = gradient_report(model)
    if not gradients['all_trainable_grads_finite'] or not gradients['has_nonzero_gradient']:
        raise RuntimeError(f'invalid backward gradients: {gradients}')
    if gradients['tensors_without_grad'] or gradients['frozen_tensors_with_grad']:
        raise RuntimeError(f'unexpected missing/frozen gradients: {gradients}')
    required_branches = ['JEPA_encoder', 'pred_input_proj', 'predictor', 'pred_geo_out',
                         'pred_app_out', 'teacher_geo_proj', 'teacher_app_proj']
    for name in required_branches:
        if gradients['branches'].get(name, {}).get('global_l2_norm', 0) <= 0:
            raise RuntimeError(f'no nonzero gradient for required branch: {name}')
    weights_after = parameter_sha256(model)
    if weights_before != weights_after:
        raise RuntimeError('model parameters unexpectedly changed without an optimizer step')
    if source_hash != file_sha256(args.ply):
        raise RuntimeError('source PLY changed during validation')

    features_path = args.output.with_suffix('.npz')
    indices = sample['indices'].numpy()
    result = {
        'status': 'PASS', 'source_ply': str(args.ply.resolve()), 'source_sha256': source_hash,
        'source_gaussians': sample['num_source_gaussians'],
        'sampled_gaussians': int(points.shape[1]), 'input_shape': list(points.shape),
        'input_dtype': str(points.dtype), 'input_contiguous': points.is_contiguous(),
        'seed': args.seed, 'model_seed': args.model_seed,
        'normalization': sample['normalization'], 'sampling': {
            'method': 'uniform', 'replacement': sample['num_source_gaussians'] < args.num_gaussians,
            'unique_indices': int(len(np.unique(indices))),
            'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
            'sample_sha256': hashlib.sha256(sample['points'].numpy().tobytes()).hexdigest(),
        },
        'losses': loss_values, 'total_loss': float(total.detach().cpu()),
        'gradients': gradients, 'feature_probe': probes,
        'optimizer_step': False, 'ema_update': False,
        'weights_unchanged': weights_before == weights_after, 'weights_sha256': weights_before,
        'note': 'train() may update BN running buffers in memory; no weights/checkpoints saved',
        'checkpoint': checkpoint,
        'semantic_effectiveness': 'NOT_EVALUATED',
        'device': str(points.device), 'gpu_name': torch.cuda.get_device_name(device),
        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'torch': torch.__version__, 'cuda_build': torch.version.cuda,
        'python': sys.version.split()[0], 'python_executable': sys.executable,
        'forward_backward_seconds': elapsed,
        'peak_allocated_memory_gib': torch.cuda.max_memory_allocated(device) / 2**30,
        'config': str(args.config.resolve()), 'config_sha256': file_sha256(args.config),
        'feature_artifact': str(features_path.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(features_path, **feature_arrays,
                        sampled_gaussians=sample['points'].numpy(), sampled_indices=indices,
                        source_ply=np.asarray(str(args.ply.resolve())),
                        source_sha256=np.asarray(source_hash), seed=np.asarray(args.seed),
                        model_seed=np.asarray(args.model_seed),
                        normalization=np.asarray(args.normalization),
                        checkpoint=np.asarray(checkpoint['path'] or 'RANDOM_INITIALIZATION'))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('Gaussian-JEPA real-plant smoke test: PASS')
    print(f"  source={args.ply} ({sample['num_source_gaussians']:,} Gaussians)")
    print(f'  initialization={checkpoint["initialization"]}; normalization={args.normalization}; sampling_seed={args.seed}')
    print(f'  input={tuple(points.shape)}; losses={loss_values}')
    print(f'  finite_grad_tensors={gradients["tensors_with_finite_grad"]}; norm={gradients["global_l2_norm"]:.6f}')
    print(f'  tokens={probes["token_shape"]}; embedding={probes["embedding_shape"]}')
    print(f'  no optimizer step; weights unchanged; semantic effectiveness NOT EVALUATED')
    print(f'  report={args.output}; features={features_path}')


if __name__ == '__main__':
    main()
