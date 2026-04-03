"""
Attention Is All You Need — Minimal Training Example

Paper: https://arxiv.org/abs/1706.03762
Implements: Minimal training loop demonstrating optimizer setup and LR schedule.

This is NOT a full training pipeline. The paper's contribution is primarily
architectural (§3). This script shows how to set up training with the
paper's specified optimizer and schedule (§5.3) for reference.

For actual training, you need:
  - WMT 2014 data with BPE tokenization (§5.1)
  - 8 P100 GPUs for the reported training speed (§5.2)
  - Token-count batching (~25k source + 25k target tokens per batch) (§5.1)
"""

import math

import torch
import yaml

from src.model import Transformer, TransformerConfig
from src.loss import LabelSmoothedCrossEntropy
from src.utils import get_transformer_lr_schedule


def train(config_path: str = "configs/base.yaml"):
    """Minimal training example with paper-specified optimizer and schedule."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model — §3, Table 3
    model_config = TransformerConfig(
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        n_encoder_layers=config["model"]["n_encoder_layers"],
        n_decoder_layers=config["model"]["n_decoder_layers"],
        dropout=config["model"]["dropout"],
        vocab_size=config["model"]["vocab_size"],
        norm_eps=config["model"]["norm_eps"],
        tie_weights=config["model"]["tie_weights"],
    )
    model = Transformer(model_config).to(device)
    print(model)

    # §5.3 — "We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ε = 10^−9"
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # LR is computed by the schedule, not a fixed value
        betas=tuple(config["training"]["betas"]),
        eps=config["training"]["eps"],
    )

    # §5.3, Eq. 3 — Custom LR schedule
    scheduler = get_transformer_lr_schedule(
        optimizer,
        d_model=config["model"]["d_model"],
        warmup_steps=config["training"]["warmup_steps"],
    )

    # §5.4 — Label smoothing with ε_ls = 0.1
    loss_fn = LabelSmoothedCrossEntropy(
        smoothing=config["training"]["label_smoothing"],
    )

    # --- Minimal forward/backward example with random data ---
    model.train()
    batch_size, src_len, tgt_len = 4, 20, 15

    src = torch.randint(3, model_config.vocab_size, (batch_size, src_len), device=device)
    tgt = torch.randint(3, model_config.vocab_size, (batch_size, tgt_len), device=device)
    tgt_labels = torch.randint(3, model_config.vocab_size, (batch_size, tgt_len), device=device)

    # Forward pass
    optimizer.zero_grad()
    logits = model(src, tgt)                             # (batch, tgt_len, vocab_size)
    loss = loss_fn(logits, tgt_labels)
    loss.backward()

    # Gradient clipping — [UNSPECIFIED] paper does not mention gradient clipping
    # Using: no clipping (following paper's silence on this)

    optimizer.step()
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"Step 1 | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
    print(f"Logits shape: {logits.shape}")
    print(f"\nTraining setup verified. For actual training, implement data loading from §5.1.")


if __name__ == "__main__":
    train()
