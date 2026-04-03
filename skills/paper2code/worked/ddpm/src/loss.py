"""
Denoising Diffusion Probabilistic Models — Training Objective (Simplified Loss)

Paper: https://arxiv.org/abs/2006.11239
Authors: Ho, Jain, Abbeel (2020)

Implements: L_simple from §3.4, Eq. 14

"We have shown that the variational bound... can be optimized with a
 simplified objective... which resembles denoising score matching..."

  L_simple = E_{t, x_0, ε} [ ||ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)||² ]

This is MSE between the true noise ε and the model's prediction ε_θ.
The expectation is over:
  - t ~ Uniform({1, ..., T})
  - x_0 ~ q(x_0)  (data distribution)
  - ε ~ N(0, I)

§3.4 — "We found it beneficial to sample quality (and simpler to implement)
to train on the following variant of the variational bound... L_simple"

§3.4 — "Algorithm 1" describes the training procedure that uses this loss.
"""

import torch
import torch.nn as nn


class DDPMLoss(nn.Module):
    """§3.4, Eq. 14 — Simplified training objective L_simple.

    Computes MSE between true noise and predicted noise:
        L = ||ε − ε_θ(x_t, t)||²

    This module handles only the loss computation. The caller (training loop)
    is responsible for sampling t, computing x_t from x_0, and calling the model.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        §3.4, Eq. 14 — L_simple = E[||ε − ε_θ(x_t, t)||²]

        Args:
            noise_pred: (batch, C, H, W) — predicted noise ε_θ(x_t, t)
            noise_true: (batch, C, H, W) — true noise ε ~ N(0, I)

        Returns:
            scalar — mean squared error loss
        """
        # §3.4 — Simple MSE between predicted and true noise
        # "equivalent to (a re-weighted variant of) the ELBO"
        return nn.functional.mse_loss(noise_pred, noise_true)
