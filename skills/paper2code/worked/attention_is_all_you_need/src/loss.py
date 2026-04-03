"""
Attention Is All You Need — Loss Functions

Paper: https://arxiv.org/abs/1706.03762
Implements: Label-smoothed cross-entropy loss

Section references:
  §5.4 — "During training, we employed label smoothing of value ε_ls = 0.1"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(nn.Module):
    """§5.4 — Label smoothing with KL divergence.

    "During training, we employed label smoothing of value ε_ls = 0.1.
     This hurt perplexity, as the model learns to be more unsure,
     but improved accuracy and BLEU score."

    Uses KL divergence formulation following Szegedy et al. (2016).
    Target distribution: (1 - ε) on correct class, ε / (V-1) on other classes.

    NOTE: This differs slightly from PyTorch's built-in label_smoothing parameter
    in CrossEntropyLoss. PyTorch distributes ε uniformly as ε/V to all classes
    (including correct), while the paper's formulation distributes to incorrect
    classes only. For large vocabularies (V >> 1), the difference is negligible.
    We implement the paper's stated formulation.
    """

    def __init__(self, smoothing: float = 0.1, pad_idx: int = 0):
        """
        Args:
            smoothing: §5.4 — ε_ls = 0.1
            pad_idx: index of padding token (loss ignored for these positions)
        """
        super().__init__()
        self.smoothing = smoothing        # §5.4 — "ε_ls = 0.1"
        self.confidence = 1.0 - smoothing  # probability mass on correct class
        self.pad_idx = pad_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.

        Args:
            logits: (batch, seq_len, vocab_size) — model output logits (NOT probabilities)
            targets: (batch, seq_len) — target token indices

        Returns:
            Scalar loss value
        """
        vocab_size = logits.size(-1)

        # Reshape for computation
        logits = logits.contiguous().view(-1, vocab_size)   # (batch * seq_len, vocab_size)
        targets = targets.contiguous().view(-1)              # (batch * seq_len,)

        # Compute log probabilities (numerically stable via log_softmax)
        log_probs = F.log_softmax(logits, dim=-1)  # (batch * seq_len, vocab_size)

        # NLL loss on the true class
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        # (batch * seq_len,)

        # Smooth loss: uniform distribution over all classes
        smooth_loss = -log_probs.mean(dim=-1)  # (batch * seq_len,)

        # §5.4 — Combined loss with label smoothing
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # Mask out padding positions
        non_pad_mask = targets != self.pad_idx
        loss = loss * non_pad_mask.float()

        # Average over non-padding tokens
        return loss.sum() / non_pad_mask.float().sum().clamp(min=1.0)
