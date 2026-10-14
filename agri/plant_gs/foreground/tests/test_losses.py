import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses import foreground_losses


def test_background_alpha_is_penalized_even_for_black_gaussians():
    rgb = torch.zeros(1, 8, 8, 3)
    mask = torch.zeros(1, 8, 8, 1)
    mask[:, 2:6, 2:6] = 1
    alpha = mask.clone().requires_grad_(True)
    alpha.data[0, 0, 0, 0] = .5
    losses = foreground_losses(rgb, alpha, rgb, mask)
    assert losses['background_alpha'] > 0
    assert losses['foreground_rgb'] == 0
    losses['background_alpha'].backward()
    assert alpha.grad[0, 0, 0, 0] > 0
    assert alpha.grad[0, 3, 3, 0] == 0
