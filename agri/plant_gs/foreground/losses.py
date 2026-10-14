"""Balanced foreground appearance + silhouette (including zero-alpha background)."""
import torch
import torch.nn.functional as F


def foreground_losses(rendered, alpha, rgb, mask):
    foreground = mask.to(rendered.dtype)
    background = 1 - foreground
    fg_rgb = ((rendered - rgb).abs() * foreground).sum() / (3 * foreground.sum()).clamp_min(1)
    bg_alpha = (alpha * background).sum() / background.sum().clamp_min(1)
    fg_alpha = ((1 - alpha).abs() * foreground).sum() / foreground.sum().clamp_min(1)
    return {'foreground_rgb': fg_rgb, 'background_alpha': bg_alpha, 'foreground_alpha': fg_alpha}


def masked_ssim(rendered, rgb, mask):
    """Patch SSIM only near the foreground; black exterior has no RGB target."""
    a = (rendered * mask).permute(0, 3, 1, 2)
    b = (rgb * mask).permute(0, 3, 1, 2)
    win = 7
    mu_a = F.avg_pool2d(a, win, stride=1)
    mu_b = F.avg_pool2d(b, win, stride=1)
    var_a = F.avg_pool2d(a * a, win, stride=1) - mu_a.square()
    var_b = F.avg_pool2d(b * b, win, stride=1) - mu_b.square()
    cov = F.avg_pool2d(a * b, win, stride=1) - mu_a * mu_b
    score = ((2*mu_a*mu_b + .01**2) * (2*cov + .03**2)) / (
        (mu_a.square()+mu_b.square()+.01**2) * (var_a+var_b+.03**2))
    support = F.avg_pool2d(mask.permute(0, 3, 1, 2), win, stride=1)
    return (score * support).sum() / (3 * support.sum()).clamp_min(1)
