"""
Denoising Diffusion Probabilistic Models — Evaluation (FID Score)

Paper: https://arxiv.org/abs/2006.11239
Authors: Ho, Jain, Abbeel (2020)

§4 — "We report FID score and Inception Score... Our best results
are FID: 3.17, IS: 9.46 (unconditional) on CIFAR-10."

This module provides a wrapper for sampling from a trained DDPM model
and computing FID scores using the pytorch-fid library.

FID (Fréchet Inception Distance) is the primary evaluation metric
used in §4 and Table 1. Lower FID = better quality.

NOTE: FID computation requires:
  1. Generating 50K samples (standard for CIFAR-10)
  2. Computing Inception features for real and generated images
  3. Computing the Fréchet distance between the two feature distributions

This is computationally expensive. For quick validation, generate a
small batch and visually inspect.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torchvision
import yaml

from model import UNet, UNetConfig
from utils import linear_noise_schedule, sample, EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> tuple:
    """Load a trained DDPM model from checkpoint.

    §4 — "we also report results with an exponential moving average"
    The EMA parameters typically produce better samples.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Target device
        use_ema: Whether to load EMA parameters (recommended)

    Returns:
        (model, config_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]

    unet_config = UNetConfig(
        image_channels=model_cfg.get("image_channels", 3),
        base_channels=model_cfg.get("base_channels", 128),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 2, 2])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [16])),
        dropout=model_cfg.get("dropout", 0.0),
        num_groups=model_cfg.get("num_groups", 32),
        image_size=cfg["data"].get("image_size", 32),
    )

    model = UNet(unet_config).to(device)

    if use_ema and "ema_state_dict" in checkpoint:
        # Load EMA parameters
        ema_params = checkpoint["ema_state_dict"]
        for name, param in model.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name])
        logger.info("Loaded EMA parameters")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded model parameters (no EMA)")

    return model, cfg


@torch.no_grad()
def generate_samples(
    model: UNet,
    config: dict,
    num_samples: int = 64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate samples using Algorithm 2 — Sampling.

    §3.4 — "sampling from p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_t I)"

    Args:
        model: Trained UNet model
        config: Config dict from checkpoint
        num_samples: Number of images to generate
        device: Target device

    Returns:
        (num_samples, C, H, W) — generated images in [0, 1] range
    """
    if device is None:
        device = next(model.parameters()).device

    diff_cfg = config["diffusion"]
    data_cfg = config["data"]

    T = diff_cfg["T"]
    betas = linear_noise_schedule(T, diff_cfg["beta_start"], diff_cfg["beta_end"]).to(device)

    image_size = data_cfg.get("image_size", 32)
    image_channels = config["model"].get("image_channels", 3)
    shape = (num_samples, image_channels, image_size, image_size)

    model.eval()
    samples = sample(model, shape, T, betas, device)

    # Convert from [-1, 1] to [0, 1] for saving
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    return samples


def save_samples(
    samples: torch.Tensor,
    output_dir: str,
    prefix: str = "sample",
    make_grid: bool = True,
    nrow: int = 8,
):
    """Save generated samples as images.

    Args:
        samples: (N, C, H, W) in [0, 1]
        output_dir: Directory to save images
        prefix: Filename prefix
        make_grid: If True, also save a grid image
        nrow: Number of images per row in grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if make_grid:
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        grid_path = output_dir / f"{prefix}_grid.png"
        torchvision.utils.save_image(grid, grid_path)
        logger.info(f"Saved grid: {grid_path}")

    # Save individual images
    for i, img in enumerate(samples):
        img_path = output_dir / f"{prefix}_{i:05d}.png"
        torchvision.utils.save_image(img, img_path)

    logger.info(f"Saved {len(samples)} individual images to {output_dir}")


def compute_fid(
    generated_dir: str,
    real_stats_path: Optional[str] = None,
    batch_size: int = 50,
    device: str = "cuda",
    dims: int = 2048,
) -> float:
    """Compute FID score between generated samples and real data.

    §4 — "We report FID score... Our best results are FID: 3.17"

    Requires the pytorch-fid package: pip install pytorch-fid

    For CIFAR-10, you need pre-computed stats for the real training set,
    or provide a directory of real images.

    Args:
        generated_dir: Directory containing generated .png images
        real_stats_path: Path to pre-computed .npz stats for real data,
                         OR directory containing real images
        batch_size: Batch size for Inception feature extraction
        device: Device for computation
        dims: Inception feature dimensionality (2048 = pool3)

    Returns:
        FID score (float). Lower is better.
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        logger.error(
            "pytorch-fid not installed. Install with: pip install pytorch-fid\n"
            "Then re-run evaluation."
        )
        raise

    if real_stats_path is None:
        raise ValueError(
            "Must provide real_stats_path: either a .npz file with pre-computed "
            "Inception statistics, or a directory of real CIFAR-10 images."
        )

    fid = fid_score.calculate_fid_given_paths(
        [generated_dir, real_stats_path],
        batch_size=batch_size,
        device=torch.device(device),
        dims=dims,
    )

    logger.info(f"FID score: {fid:.2f}")
    return fid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DDPM Evaluation — Generate samples and compute FID")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="generated",
                        help="Output directory for generated images")
    parser.add_argument("--fid", action="store_true",
                        help="Compute FID score (requires --real_stats)")
    parser.add_argument("--real_stats", type=str, default=None,
                        help="Path to real data stats (.npz) or directory")
    parser.add_argument("--no_ema", action="store_true",
                        help="Don't use EMA parameters")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    samples = generate_samples(model, cfg, args.num_samples, device)
    save_samples(samples, args.output_dir)

    if args.fid:
        compute_fid(args.output_dir, args.real_stats, device=args.device)
