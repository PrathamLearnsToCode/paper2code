"""
Denoising Diffusion Probabilistic Models — Training Loop

Paper: https://arxiv.org/abs/2006.11239
Authors: Ho, Jain, Abbeel (2020)

Implements: Algorithm 1 — Training procedure

  1: repeat
  2:   x_0 ~ q(x_0)                         ← sample data
  3:   t ~ Uniform({1, ..., T})              ← sample timestep
  4:   ε ~ N(0, I)                           ← sample noise
  5:   Take gradient step on
          ∇_θ ||ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)||²
  6: until converged

Hyperparameters from §4 and Appendix B:
  - Adam optimizer, learning rate 2e-4 (§B)
  - No lr schedule/warmup [UNSPECIFIED — not mentioned in paper]
  - Gradient clipping at 1.0 [FROM_OFFICIAL_CODE]
  - EMA with decay 0.9999 (§4)
  - T = 1000 (§4)
  - Trained for 800K steps on CIFAR-10 (Appendix B)
"""

import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from model import UNet, UNetConfig
from loss import DDPMLoss
from data import get_dataloaders
from utils import linear_noise_schedule, q_sample, EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)


def train(config_path: str = "configs/base.yaml"):
    """Algorithm 1 — DDPM Training.

    Args:
        config_path: Path to YAML config file
    """
    # --- Load config ---
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")

    diff_cfg = cfg["diffusion"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Noise schedule ---
    # §2, Eq. 4 — linear schedule β_1 = 0.0001, β_T = 0.02
    T = diff_cfg["T"]
    betas = linear_noise_schedule(T, diff_cfg["beta_start"], diff_cfg["beta_end"])
    betas = betas.to(device)

    alphas = 1.0 - betas                                    # α_t = 1 − β_t
    alpha_bar = torch.cumprod(alphas, dim=0)                # ᾱ_t = ∏_{s=1}^{t} α_s
    sqrt_alpha_bar = torch.sqrt(alpha_bar)                  # √ᾱ_t
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)  # √(1−ᾱ_t)

    # --- Model ---
    unet_config = UNetConfig(
        image_channels=model_cfg.get("image_channels", 3),
        base_channels=model_cfg.get("base_channels", 128),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 2, 2])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [16])),
        dropout=model_cfg.get("dropout", 0.0),
        num_groups=model_cfg.get("num_groups", 32),
        image_size=data_cfg.get("image_size", 32),
    )
    model = UNet(unet_config).to(device)
    logger.info(f"Model: {model}")

    # --- EMA ---
    # §4 — "we also report results with an exponential moving average of
    # model parameters with a decay factor of 0.9999"
    ema = EMA(model, decay=train_cfg.get("ema_decay", 0.9999))

    # --- Optimizer ---
    # Appendix B — "Adam, lr = 2 × 10^-4"
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
    )

    # --- Loss ---
    criterion = DDPMLoss()

    # --- Data ---
    train_loader, _ = get_dataloaders(
        data_dir=data_cfg.get("data_dir", "./data"),
        batch_size=train_cfg.get("batch_size", 128),
        num_workers=data_cfg.get("num_workers", 4),
        image_size=data_cfg.get("image_size", 32),
    )

    # --- Training loop: Algorithm 1 ---
    total_steps = train_cfg.get("total_steps", 800_000)
    log_every = train_cfg.get("log_every", 1000)
    save_every = train_cfg.get("save_every", 50_000)
    save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    grad_clip = train_cfg.get("gradient_clip", 1.0)

    step = 0
    model.train()

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            # Algorithm 1, line 2: x_0 ~ q(x_0)
            x_0 = batch[0].to(device)  # (batch, C, H, W), labels discarded (unconditional)
            batch_size = x_0.shape[0]

            # Algorithm 1, line 3: t ~ Uniform({1, ..., T})
            t = torch.randint(1, T + 1, (batch_size,), device=device)

            # Algorithm 1, line 4: ε ~ N(0, I)
            noise = torch.randn_like(x_0)

            # Algorithm 1, line 5: compute x_t and predict noise
            x_t = q_sample(x_0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise)
            noise_pred = model(x_t, t)

            # L_simple — §3.4, Eq. 14
            loss = criterion(noise_pred, noise)

            # Algorithm 1, line 5: gradient step
            optimizer.zero_grad()
            loss.backward()

            # [FROM_OFFICIAL_CODE] gradient clipping
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # §4 — EMA update
            ema.update()

            step += 1

            if step % log_every == 0:
                logger.info(f"Step {step}/{total_steps} — loss: {loss.item():.6f}")

            if step % save_every == 0:
                checkpoint = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.shadow_params,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "config": cfg,
                }
                ckpt_path = save_dir / f"ddpm_step_{step}.pt"
                torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = save_dir / "ddpm_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow_params,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }, final_path)
    logger.info(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DDPM — Algorithm 1")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
