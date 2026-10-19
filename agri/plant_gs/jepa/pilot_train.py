#!/usr/bin/env python3
"""Short real-plant pretraining using the unchanged official Gaussian-JEPA."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri.plant_gs.jepa.dataset import PlantGaussianDataset
from agri.plant_gs.jepa.pilot_data import EXPERIMENT_ROOT, sha256_file, write_json
from utils.config import cfg_from_yaml_file

DEFAULT_CONFIG = Path(__file__).with_name('pilot_config.yaml')
DEFAULT_AUDIT = EXPERIMENT_ROOT / 'data_audit.json'
DEFAULT_SPLIT = EXPERIMENT_ROOT / 'split.json'


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_inputs(audit_path: Path, split_path: Path):
    audit = json.loads(audit_path.read_text())
    split = json.loads(split_path.read_text())
    records = {item['plant_id']: item for item in audit['records']}
    all_ids = set(records)
    train_ids = split['train_plant_ids']
    test_ids = split['test_plant_ids']
    if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != all_ids:
        raise ValueError('fixed split must be a disjoint partition of audited plant IDs')
    return records, split


def build_model(config, seed: int, device: torch.device):
    from models import build_model_from_cfg
    seed_everything(seed)
    model = build_model_from_cfg(config.model).to(device)
    return model


def save_initialization(config_path: Path, output_path: Path, seed: int,
                        device: torch.device) -> None:
    config = cfg_from_yaml_file(str(config_path))
    model = build_model(config, seed, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    torch.save({
        'base_model': model.state_dict(),
        'epoch': -1,
        'seed': seed,
        'initialization': 'random_from_scratch',
        'model_config': dict(config.model),
    }, temp_path)
    temp_path.replace(output_path)


def build_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith('.bias') or 'token' in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': config.weight_decay},
    ], lr=config.learning_rate)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=config.epochs,
        lr_min=1e-6,
        cycle_decay=0.1,
        warmup_lr_init=1e-6,
        warmup_t=max(1, int(config.epochs * 0.1)),
        cycle_limit=1,
        t_in_epochs=True,
    )
    return optimizer, scheduler


def safe_torch_load(path: Path):
    # Checkpoints contain trusted, locally generated optimizer/RNG state.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError as error:
        if 'weights_only' not in str(error):
            raise
        return torch.load(path, map_location='cpu')


def atomic_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temp_path)
    temp_path.replace(path)


def get_encoder_grad_norm(model) -> tuple[float, bool, int]:
    squared_norm = 0.0
    all_finite = True
    count = 0
    for parameter in model.JEPA_encoder.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        all_finite = all_finite and bool(torch.isfinite(grad).all())
        squared_norm += float(grad.float().square().sum().item())
        count += 1
    return math.sqrt(squared_norm), all_finite, count


@torch.no_grad()
def ema_parameter_delta(model, momentum: float) -> float:
    student = list(model.JEPA_encoder.parameters())
    teacher = list(model.teacher_encoder.parameters())
    before = sum((t.detach().float() - s.detach().float()).square().sum()
                 for s, t in zip(student, teacher))
    model._update_teacher(momentum=momentum)
    after = sum((t.detach().float() - s.detach().float()).square().sum()
                for s, t in zip(student, teacher))
    return float(torch.sqrt(before).item() - torch.sqrt(after).item())


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def train(args) -> None:
    config = cfg_from_yaml_file(str(args.config))
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('official Gaussian grouping requires an available CUDA device')
    torch.cuda.set_device(device)
    records, split = load_inputs(args.audit, args.split)
    all_train_ids = split['train_plant_ids']
    if args.stage == 'sanity':
        selected_ids = all_train_ids[:args.sanity_plants]
        if len(selected_ids) < 2:
            raise ValueError('sanity check needs at least two training plants')
    else:
        selected_ids = all_train_ids
    if len(selected_ids) < 2:
        raise ValueError('training needs at least two independent plants')

    seed_everything(args.seed)
    model = build_model(config, args.seed, device)
    if args.initial_checkpoint:
        initial = safe_torch_load(args.initial_checkpoint)
        model.load_state_dict(initial['base_model'], strict=True)

    optimizer, scheduler = build_optimizer(model, config)
    start_epoch = 0
    resumed_config = None
    if args.resume:
        checkpoint = safe_torch_load(args.resume)
        resumed_config = checkpoint['run_config']
        for key, expected in (
            ('train_plant_ids', selected_ids), ('seed', args.seed),
            ('num_gaussians', config.num_gaussians), ('batch_size', config.batch_size),
        ):
            if resumed_config[key] != expected:
                raise ValueError(f'resume mismatch for {key}')
        model.load_state_dict(checkpoint['base_model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch']) + 1
        if checkpoint.get('torch_rng_state') is not None:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if checkpoint.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        if checkpoint.get('numpy_rng_state') is not None:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if checkpoint.get('python_rng_state') is not None:
            random.setstate(checkpoint['python_rng_state'])

    dataset = PlantGaussianDataset(
        [records[plant_id]['ply'] for plant_id in selected_ids],
        num_gaussians=config.num_gaussians,
        seed=args.seed,
        normalization=config.normalization,
    )
    batches_per_epoch = math.ceil(len(dataset) / config.batch_size)
    total_steps = batches_per_epoch * args.epochs
    if start_epoch >= args.epochs:
        raise ValueError(f'checkpoint is already at/after requested {args.epochs} epochs')

    args.output.mkdir(parents=True, exist_ok=True)
    run_meta = {
        'stage': args.stage,
        'train_plant_ids': selected_ids,
        'batch_size': config.batch_size,
        'epochs': args.epochs,
        'num_gaussians': config.num_gaussians,
        'initialization': (resumed_config['initialization'] if resumed_config else
                           str(args.initial_checkpoint.resolve()) if args.initial_checkpoint
                           else 'random_from_scratch'),
        'seed': args.seed,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device),
        'model_config': dict(config.model),
        'test_plant_ids_excluded_from_training': split['test_plant_ids'],
        'normalization': config.normalization,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'ema_momentum_start': config.ema_momentum_start,
        'ema_momentum_end': config.ema_momentum_end,
        'scheduler_epochs': config.epochs,
        'total_model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'student_encoder_parameters': sum(p.numel() for p in model.JEPA_encoder.parameters()),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'official_model_source_sha256': sha256_file(ROOT / 'models/Gaussian_JEPA_ExpMultiScale.py'),
        'split_path': str(args.split.resolve()),
        'split_sha256': sha256_file(args.split),
        'config_path': str(args.config.resolve()),
        'config_sha256': sha256_file(args.config),
        'resumed_from': str(args.resume.resolve()) if args.resume else None,
        'start_epoch_zero_based': start_epoch,
    }
    write_json(args.output / 'run_config.json', run_meta)
    log_path = args.output / 'train_metrics.csv'
    step_offset = start_epoch * batches_per_epoch

    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        loader_generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        epoch_rows = []
        sampled_paths, sampled_indices = [], []
        sums = {'geo': 0.0, 'app': 0.0, 'ground': 0.0, 'total': 0.0}

        for batch_index, batch in enumerate(loader):
            sampled_paths.extend(batch['source_ply'])
            sampled_indices.extend(batch['indices'].numpy())
            points = batch['points'].to(device, dtype=torch.float32,
                                        non_blocking=True).contiguous()
            if points.shape[1:] != (config.num_gaussians, 14):
                raise RuntimeError(f'unexpected agricultural input shape {points.shape}')
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(points)
            if set(loss_dict) != {'geo', 'app', 'ground'}:
                raise RuntimeError(f'unexpected official loss keys: {sorted(loss_dict)}')
            total_loss = sum(loss_dict.values())
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f'non-finite loss at epoch {epoch}')
            total_loss.backward()
            grad_norm, grad_finite, grad_tensors = get_encoder_grad_norm(model)
            if not grad_finite or grad_norm <= 0:
                raise FloatingPointError('student Encoder has invalid/zero gradients')

            optimizer.step()
            global_step = step_offset + batch_index
            momentum = config.ema_momentum_start + (
                config.ema_momentum_end - config.ema_momentum_start
            ) * global_step / max(total_steps, 1)
            teacher_delta = ema_parameter_delta(model, momentum)

            losses = {key: float(value.detach().item()) for key, value in loss_dict.items()}
            values = {**losses, 'total': float(total_loss.detach().item())}
            for key, value in values.items():
                sums[key] += value
            epoch_rows.append({
                'epoch': epoch + 1,
                'batch': batch_index + 1,
                'global_step': global_step + 1,
                **values,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'student_encoder_grad_l2': grad_norm,
                'student_encoder_grad_tensors': grad_tensors,
                'all_student_encoder_grads_finite': grad_finite,
                'ema_momentum': momentum,
                'teacher_parameter_delta_after_ema': teacher_delta,
                'teacher_updated': teacher_delta > 0,
                'gpu_memory_allocated_gib': torch.cuda.memory_allocated(device) / 2**30,
                'gpu_memory_reserved_gib': torch.cuda.memory_reserved(device) / 2**30,
            })
        scheduler.step(epoch + 1)
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        sampling_path = args.output / 'sampling_indices' / f'epoch_{epoch + 1:03d}.npz'
        sampling_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sampling_path,
            source_ply=np.asarray(sampled_paths),
            indices=np.asarray(sampled_indices, dtype=np.int64),
            epoch=np.asarray(epoch + 1),
            sampling_seed=np.asarray(args.seed + epoch),
        )
        for row in epoch_rows:
            row['epoch_seconds'] = train_seconds
            row['epoch_mean_geo'] = sums['geo'] / len(epoch_rows)
            row['epoch_mean_app'] = sums['app'] / len(epoch_rows)
            row['epoch_mean_ground'] = sums['ground'] / len(epoch_rows)
            row['epoch_mean_total'] = sums['total'] / len(epoch_rows)
            row['epoch_peak_gpu_memory_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
            row['sampling_indices_path'] = str(sampling_path.resolve())

        checkpoint = {
            'base_model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': step_offset + len(epoch_rows),
            'run_config': run_meta,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        ckpt_path = args.output / 'checkpoint_last.pt'
        checkpoint_started = time.perf_counter()
        atomic_checkpoint(ckpt_path, checkpoint)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        for row in epoch_rows:
            row['checkpoint_path'] = str(ckpt_path.resolve())
            row['checkpoint_seconds'] = checkpoint_seconds
        append_csv(log_path, epoch_rows)
        print(json.dumps({
            'stage': args.stage,
            'epoch': epoch + 1,
            'epochs': args.epochs,
            'plants': len(selected_ids),
            'losses': {key: sums[key] / len(epoch_rows) for key in sums},
            'learning_rate_next_epoch': optimizer.param_groups[0]['lr'],
            'epoch_seconds': round(train_seconds, 2),
            'peak_gpu_memory_gib': round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            'checkpoint': str(ckpt_path),
        }, ensure_ascii=False))

        if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['sanity', 'pilot'], default='pilot')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--sanity-plants', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--initialize-only', action='store_true')
    args = parser.parse_args()

    config = cfg_from_yaml_file(str(args.config))
    if args.initialize_only:
        output = args.output or EXPERIMENT_ROOT / 'initial_random.pt'
        save_initialization(args.config, output, args.seed, torch.device(args.device))
        print(f'initialization_checkpoint={output.resolve()}')
        return
    args.epochs = args.epochs or config.epochs
    args.output = args.output or EXPERIMENT_ROOT / args.stage
    train(args)


if __name__ == '__main__':
    main()
