"""
Denoising Diffusion Probabilistic Models — Shared Utilities

Paper: https://arxiv.org/abs/2006.11239
Implements: Noise schedule (§4), forward process q(x_t|x_0) (§3, Eq. 4),
            and reverse sampling (Algorithm 2).

Section references:
  §2 — Forward and reverse diffusion processes
  §3 — Training objective derivation
  §4 — Noise schedule specification
  Algorithm 1 — Training
  Algorithm 2 — Sampling
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


def linear_noise_schedule(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """§4 — Linear variance schedule.

    "We set the forward process variances to constants increasing linearly
     from β_1 = 10^-4 to β_T = 0.02."

    Precomputes all quantities needed for training and sampling.

    Args:
        timesteps: §4 — T = 1000
        beta_start: §4 — β_1 = 10^-4
        beta_end: §4 — β_T = 0.02

    Returns:
        Dict with precomputed schedule tensors
    """
    # §4 — linear schedule
    betas = torch.linspace(beta_start, beta_end, timesteps)

    # §2 — α_t = 1 - β_t
    alphas = 1.0 - betas

    # §2 — α̅_t = Π_{s=1}^{t} α_s  (cumulative product)
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # α̅_{t-1} — needed for posterior q(x_{t-1} | x_t, x_0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    # Precompute quantities for q(x_t | x_0) = N(x_t; √α̅_t x_0, (1-α̅_t)I)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Precompute for sampling (Algorithm 2)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)

    # §3.2, Eq. 7 — Posterior variance β̃_t = β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """§3, Eq. 4 — Forward process: sample x_t from q(x_t | x_0).

    "A notable property is that we can sample x_t at any arbitrary time step t
     in closed form: q(x_t | x_0) = N(x_t; √α̅_t x_0, (1 - α̅_t)I)"

    x_t = √α̅_t * x_0 + √(1 - α̅_t) * ε, where ε ~ N(0, I)

    Args:
        x_0: (batch, C, H, W) — clean images
        t: (batch,) — timestep indices
        schedule: precomputed noise schedule
        noise: optional pre-sampled noise (for reproducibility)

    Returns:
        x_t: (batch, C, H, W) — noisy images at timestep t
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    # Extract schedule values for timestep t, reshape for broadcasting
    sqrt_alpha_cumprod = schedule["sqrt_alphas_cumprod"][t]          # (batch,)
    sqrt_one_minus_alpha_cumprod = schedule["sqrt_one_minus_alphas_cumprod"][t]  # (batch,)

    # Reshape for broadcasting with (batch, C, H, W)
    sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1)

    # §3, Eq. 4 — x_t = √α̅_t * x_0 + √(1 - α̅_t) * ε
    return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_index: int,
    schedule: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Algorithm 2, lines 3-4 — Single reverse step: sample x_{t-1} from p_θ(x_{t-1} | x_t).

    "x_{t-1} = 1/√α_t * (x_t - β_t/√(1-α̅_t) * ε_θ(x_t, t)) + σ_t * z"

    where z ~ N(0, I) if t > 1, else z = 0.

    Args:
        model: noise prediction network ε_θ
        x_t: (batch, C, H, W) — current noisy sample
        t: (batch,) — current timestep (as tensor for model input)
        t_index: integer timestep (for indexing schedule)
        schedule: precomputed noise schedule

    Returns:
        x_{t-1}: (batch, C, H, W) — denoised sample one step
    """
    # Predict noise ε_θ(x_t, t)
    predicted_noise = model(x_t, t)  # (batch, C, H, W)

    # Extract schedule values
    beta_t = schedule["betas"][t_index]
    sqrt_recip_alpha_t = schedule["sqrt_recip_alphas"][t_index]
    sqrt_one_minus_alpha_cumprod_t = schedule["sqrt_one_minus_alphas_cumprod"][t_index]

    # Algorithm 2, line 4 — Compute mean of p_θ(x_{t-1} | x_t)
    # μ_θ = 1/√α_t * (x_t - β_t/√(1-α̅_t) * ε_θ(x_t, t))
    mean = sqrt_recip_alpha_t * (
        x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise
    )

    if t_index == 0:
        # Algorithm 2, line 3 — z = 0 when t = 1 (final step)
        return mean
    else:
        # Algorithm 2, line 3 — z ~ N(0, I) when t > 1
        # §3.4 — σ²_t = β_t (fixed small variance)
        sigma_t = torch.sqrt(schedule["betas"][t_index])
        noise = torch.randn_like(x_t)
        return mean + sigma_t * noise


@torch.no_grad()
def sample(
    model: nn.Module,
    schedule: Dict[str, torch.Tensor],
    image_shape: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Algorithm 2 — Full reverse sampling process.

    "Algorithm 2 Sampling
     1: x_T ~ N(0, I)
     2: for t = T, ..., 1 do
     3:   z ~ N(0, I) if t > 1, else z = 0
     4:   x_{t-1} = 1/√α_t (x_t - β_t/√(1-α̅_t) ε_θ(x_t, t)) + σ_t z
     5: end for
     6: return x_0"

    Args:
        model: noise prediction network ε_θ (should be in eval mode, ideally EMA weights)
        schedule: precomputed noise schedule
        image_shape: (batch, C, H, W) — shape of images to generate
        device: torch device

    Returns:
        x_0: (batch, C, H, W) — generated images
    """
    model.eval()
    timesteps = len(schedule["betas"])

    # Algorithm 2, line 1: x_T ~ N(0, I)
    x = torch.randn(image_shape, device=device)

    # Algorithm 2, lines 2-5: reverse iterate from t=T to t=1
    for t_index in reversed(range(timesteps)):
        t = torch.full((image_shape[0],), t_index, device=device, dtype=torch.long)
        x = p_sample(model, x, t, t_index, schedule)

    return x


class EMA:
    """§4 — Exponential Moving Average of model parameters.

    "We report sample quality metrics using an exponential moving average (EMA)
     of model parameters with a decay factor of 0.9999."

    The EMA weights are used for sampling/evaluation, not for training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: the model to track
            decay: §4 — "decay factor of 0.9999"
        """
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights after each training step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module):
        """Load EMA weights into model (for evaluation/sampling)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model weights (after evaluation)."""
        # NOTE: This requires storing original weights separately.
        # The caller should save model.state_dict() before calling apply().
        pass
