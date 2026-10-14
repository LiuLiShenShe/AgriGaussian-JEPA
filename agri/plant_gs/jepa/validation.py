"""Validation helpers for real-input Gaussian-JEPA smoke tests."""
from __future__ import annotations

import math

import torch


def gradient_report(model: torch.nn.Module) -> dict:
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    finite = []
    missing = []
    nonfinite = []
    for index, parameter in enumerate(trainable):
        if parameter.grad is None:
            missing.append(index)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(index)
        else:
            finite.append(parameter)
    frozen_with_grad = [index for index, p in enumerate(frozen) if p.grad is not None]
    squared = sum(torch.sum(p.grad.detach().float() ** 2) for p in finite)
    norm = float(torch.sqrt(squared).cpu()) if finite else 0.0
    branches = {}
    for name, module in model.named_children():
        params = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
        if params:
            b2 = sum(torch.sum(p.grad.detach().float() ** 2) for p in params)
            branches[name] = {
                'trainable_tensors_with_grad': len(params),
                'global_l2_norm': float(torch.sqrt(b2).cpu()),
            }
    return {
        'trainable_tensors': len(trainable),
        'tensors_with_finite_grad': len(finite),
        'tensors_without_grad': missing,
        'tensors_with_nonfinite_grad': nonfinite,
        'frozen_tensors': len(frozen),
        'frozen_tensors_with_grad': frozen_with_grad,
        'global_l2_norm': norm,
        'branches': branches,
        'all_trainable_grads_finite': not nonfinite,
        'has_nonzero_gradient': math.isfinite(norm) and norm > 0,
    }
