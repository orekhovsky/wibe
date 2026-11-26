"""Random-target spectral adaptation of DIA's recomputed inversion gradient.

The only attack input is the image. The terminal loss does not receive a
watermark key, detector, true mask, source generation noise, or detector score.
"""

from collections.abc import Callable, Sequence
import json

import torch
from tqdm import trange

from .objectives import latent_scale_distance, spectral_distance
from .vendor.dia_pt import METHOD2 as ProcessTrajectory


def recomputed_vjp(
    initial: torch.Tensor,
    steps: Sequence,
    step_fn: Callable,
    loss_fn: Callable,
    loss_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate a terminal loss by recomputing one step at a time.

    Forward and backward invoke the same deterministic step_fn. Gradients
    are returned in float32. loss_scale protects fp16 intermediate cotangents
    from underflow and is removed from the returned gradient.
    """
    states = []
    with torch.no_grad():
        value = initial.detach()
        for step in steps:
            states.append(value)
            value = step_fn(step, value).detach()
    terminal = value.detach().requires_grad_(True)
    loss = loss_fn(terminal)
    adjoint = torch.autograd.grad(loss * loss_scale, terminal)[0].detach()
    for step, state in zip(reversed(steps), reversed(states)):
        leaf = state.detach().requires_grad_(True)
        with torch.enable_grad():
            output = step_fn(step, leaf)
            adjoint = torch.autograd.grad(output, leaf, grad_outputs=adjoint)[0].detach()
    return adjoint.float() / loss_scale, loss.detach(), value


class RandomTargetAttack(ProcessTrajectory):
    """Optimize the inverted latent towards a fixed random target.

    Uses the authors' surrogate model and DDIM equations, with consistent
    deterministic VAE mode() in both passes. A clean start and float32 pixel
    master copy replace the original PT initialization. Returned candidates
    are selected solely by the surrogate objective, never by detector output.
    """

    def __init__(self, model, *, quality_weight: float,
                 latent_scale_weight: float, **config) -> None:
        super().__init__(model, **config)
        self.quality_weight = quality_weight
        self.latent_scale_weight = latent_scale_weight
        self.history: list[dict] = []

    def _target_loss(self, target, reference):
        raise NotImplementedError

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        dtype = next(self.model.vae.parameters()).dtype
        scale = self.model.vae.config.scaling_factor
        return self.model.vae.encode(pixels.to(dtype)).latent_dist.mode() * scale

    def _step(self, timestep, value: torch.Tensor) -> torch.Tensor:
        return self.custom_diffusion(
            timestep, value, self.forward_text_embeddings[0],
            self.forward_text_embeddings[1], self.forward_text_embeddings[2],
            self.forward_cfg, inversion=True,
        )

    def _invert(self, pixels: torch.Tensor, steps: list) -> torch.Tensor:
        value = self._encode(pixels)
        for step in steps:
            value = self._step(step, value)
        return value

    def compute_grad(self, source_latents, **attack_args):
        self.model.scheduler.set_timesteps(self.num_inference_steps)
        steps = list(reversed(self.model.scheduler.timesteps.to(source_latents.device)))
        self.attn_controller.state = False
        self.attn_controller.zero_attn_probs()
        source = source_latents.detach().float()
        with torch.no_grad():
            reference = self._invert(source, steps).detach()
            target = torch.randn(reference.shape, device=reference.device, dtype=torch.float32)
            target_loss = self._target_loss(target, reference)
        current = source.clone()
        best, best_score = source.clone(), float("inf")
        self.history = []

        def terminal_loss(latent):
            return (target_loss(latent)
                    + self.latent_scale_weight * latent_scale_distance(latent, reference))

        # The public budgets have been converted to [-1, 1] by backend.py.
        for iteration in trange(self.iters + 1, desc=self.objective_name):
            leaf = current.detach().requires_grad_(True)
            quality = self.quality_weight * ((leaf - source) / self.eps).square().mean()
            if iteration < self.iters:
                with torch.no_grad():
                    initial = self._encode(leaf)
                loss_scale = 1024.0 if initial.dtype == torch.float16 else 1.0
                adjoint, loss, terminal = recomputed_vjp(
                    initial, steps, self._step, terminal_loss, loss_scale
                )
                encoded = self._encode(leaf)
                grad = torch.autograd.grad(
                    encoded, leaf,
                    grad_outputs=(adjoint * loss_scale).to(encoded.dtype),
                )[0].float() / loss_scale
                grad = grad + torch.autograd.grad(quality, leaf)[0]
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"Non-finite {self.objective_name} DIA pixel gradient")
            else:
                with torch.no_grad():
                    terminal = self._invert(leaf, steps)
                    loss = terminal_loss(terminal)
            score = float(loss) + float(quality.detach())
            if not torch.isfinite(loss) or not torch.isfinite(quality):
                raise RuntimeError(f"Non-finite {self.objective_name} DIA loss")
            if score < best_score:
                best, best_score = current.detach().clone(), score
            with torch.no_grad():
                self.history.append(dict(
                    iteration=iteration, loss=score,
                        **{self.loss_name: float(target_loss(terminal))},
                    scale=float(latent_scale_distance(terminal, reference)),
                    pixel_penalty=float(quality.detach()),
                    linf_0_1=float((current-source).abs().max()/2),
                ))
                if iteration < self.iters:
                    current = current - self.lr * grad.sign()
                    current = torch.maximum(torch.minimum(current, source+self.eps), source-self.eps)
                    current.clamp_(-1, 1)
        print(self.trace_name, json.dumps(dict(
            initial=self.history[0], final=self.history[-1], best_loss=best_score,
            selected_linf_0_1=float((best-source).abs().max()/2),
        )), flush=True)
        return best


class SpectralAttack(RandomTargetAttack):
    """Minimize a normalized FFT loss with bounded pixel updates."""

    objective_name = "spectral"
    loss_name = "frequency"
    trace_name = "DIA_SPECTRAL_TRACE"

    def __init__(self, model, *, band_weights: tuple[float, ...],
                 quality_weight: float, latent_scale_weight: float, **config) -> None:
        super().__init__(model, quality_weight=quality_weight,
                         latent_scale_weight=latent_scale_weight, **config)
        self.band_weights = band_weights

    def _target_loss(self, target, reference):
        return lambda latent: spectral_distance(latent, target, self.band_weights)
