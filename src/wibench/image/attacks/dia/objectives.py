"""Key-free frequency objectives for a surrogate inverted noise latent.

These experimental objectives are not part of the original DIA paper.
Only the image-derived reference and an independent random target are used.
"""

import math

import torch


def spectral_distance(
    latent: torch.Tensor,
    target: torch.Tensor,
    weights: tuple[float, ...] = (4.0, 2.0, 1.0, 1.0),
) -> torch.Tensor:
    """Compare unit complex FFT vectors per channel and broad radial band.

    Frequencies use cycles per latent pixel. The DC coefficient is omitted:
    the separate scale penalty controls channel means. Complex phase and
    relative coefficient magnitudes participate; a uniform rescaling cannot
    improve this term. FFT and reductions run in float32 even with fp16 UNet.
    """
    if latent.shape != target.shape or latent.ndim != 4:
        raise ValueError("Expected matching BCHW latent and target tensors")
    height, width = latent.shape[-2:]
    fy = torch.fft.fftfreq(height, device=latent.device)
    fx = torch.fft.fftfreq(width, device=latent.device)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    edges = (0.0, 0.125, 0.25, 0.5, math.sqrt(0.5) + 1e-6)
    current_fft = torch.fft.fft2(latent.float(), norm="ortho")
    target_fft = torch.fft.fft2(target.detach().float(), norm="ortho")
    losses, used_weights = [], []
    for lower, upper, weight in zip(edges[:-1], edges[1:], weights):
        mask = (radius >= lower) & (radius < upper) & (radius > 0)
        if weight == 0 or not mask.any():
            continue
        current, goal = current_fft[..., mask], target_fft[..., mask]
        current = current / current.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        goal = goal / goal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        losses.append(weight * (current - goal).abs().square().sum(-1).mean())
        used_weights.append(weight)
    if not losses:
        raise ValueError("No nonempty frequency bands with positive weight")
    return torch.stack(losses).sum() / sum(used_weights)


def latent_scale_distance(
    latent: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Penalize changes in each channel's mean and log standard deviation."""
    current, baseline = latent.float(), reference.detach().float()
    dims = (-2, -1)
    std = current.std(dims, correction=0).clamp_min(1e-6)
    ref_std = baseline.std(dims, correction=0).clamp_min(1e-6)
    mean_delta = (current.mean(dims) - baseline.mean(dims)) / ref_std
    return (mean_delta.square() + (std.log() - ref_std.log()).square()).mean()
