"""
Attention Is All You Need — Shared Utilities

Paper: https://arxiv.org/abs/1706.03762
Implements: Learning rate schedule (§5.3, Eq. 3) and masking utilities.
"""

import math

import torch
import torch.optim as optim


def get_transformer_lr_schedule(
    optimizer: optim.Optimizer,
    d_model: int,
    warmup_steps: int = 4000,
) -> optim.lr_scheduler.LambdaLR:
    """§5.3, Eq. 3 — Transformer learning rate schedule.

    "We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ε = 10^−9.
     We varied the learning rate over the course of training, according to the formula:

         lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

     This corresponds to increasing the learning rate linearly for the first
     warmup_steps training steps, and decreasing it thereafter proportionally
     to the inverse square root of the step number."

    Args:
        optimizer: the optimizer to schedule
        d_model: model dimension — §3, Table 3: 512
        warmup_steps: §5.3 — "warmup_steps = 4000"

    Returns:
        LambdaLR scheduler (step per training step, not per epoch)
    """
    def lr_lambda(step: int) -> float:
        step = max(step, 1)  # avoid division by zero at step 0
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_padding_mask(tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Create a padding mask for source sequences.

    Args:
        tokens: (batch, seq_len) — token IDs
        pad_idx: the padding token index

    Returns:
        (batch, 1, 1, seq_len) — True where tokens are NOT padding, False where padding
    """
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """§3.2.3 — Create a causal (autoregressive) mask.

    "We also modify the self-attention sub-layer in the decoder stack to prevent
     positions from attending to subsequent positions."

    Args:
        seq_len: length of the target sequence
        device: torch device

    Returns:
        (1, 1, seq_len, seq_len) — True for allowed positions, False for masked
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def create_decoder_mask(
    tgt: torch.Tensor, pad_idx: int = 0
) -> torch.Tensor:
    """Create combined causal + padding mask for decoder self-attention.

    Args:
        tgt: (batch, tgt_len) — target token IDs
        pad_idx: padding token index

    Returns:
        (batch, 1, tgt_len, tgt_len) — combined mask
    """
    batch_size, tgt_len = tgt.size()

    # Padding mask: (batch, 1, 1, tgt_len)
    pad_mask = create_padding_mask(tgt, pad_idx)

    # Causal mask: (1, 1, tgt_len, tgt_len)
    causal_mask = create_causal_mask(tgt_len, tgt.device)

    # Combine: both conditions must be True
    return pad_mask & causal_mask  # (batch, 1, tgt_len, tgt_len)
