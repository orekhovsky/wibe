"""DIA objective for Gaussian Shading's sign-based readout."""

import torch
from torch.nn.functional import softplus

from .spectral import RandomTargetAttack


class GaussianShadingAttack(RandomTargetAttack):
    """Match independent random signs using a differentiable sign margin."""

    objective_name = "gaussian_shading"
    loss_name = "sign_loss"
    trace_name = "DIA_GAUSSIAN_SHADING_TRACE"

    def __init__(self, model, *, sign_temperature: float, **config) -> None:
        super().__init__(model, **config)
        self.sign_temperature = sign_temperature

    def _target_loss(self, target, reference):
        signs = (target > 0).float().mul_(2).sub_(1)
        # Fix the scale using the clean inversion. Centering would move the
        # zero boundary used by Gaussian Shading.
        scale = reference.float().std(dim=(-2, -1), correction=0, keepdim=True)
        normalizer = scale.clamp_min(1e-6) * self.sign_temperature

        def loss(latent: torch.Tensor) -> torch.Tensor:
            return softplus(-signs * latent.float() / normalizer).mean()

        return loss
