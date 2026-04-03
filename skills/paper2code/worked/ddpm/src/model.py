"""
Denoising Diffusion Probabilistic Models — U-Net Noise Prediction Network

Paper: https://arxiv.org/abs/2006.11239
Authors: Ho, Jain, Abbeel (2020)

Implements: U-Net architecture for noise prediction ε_θ(x_t, t)
The U-Net is adapted from PixelCNN++ (Salimans et al., 2017) and the architecture
used in score matching (Song & Ermon, 2019). It is NOT the paper's core contribution
but is required as the backbone for the diffusion process.

Section references:
  §3.3 — "The neural network... is similar to an unmasked PixelCNN++ with
          group normalization and self-attention"
  Appendix B — Architecture details (channel counts, attention resolutions)

NOTE: This U-Net implementation follows the architecture from the official code
(github.com/hojonathanho/diffusion) since the paper describes it only briefly
in §3.3 and Appendix B. Many details are [FROM_OFFICIAL_CODE].
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class UNetConfig:
    """U-Net configuration.

    Values from Appendix B (CIFAR-10 config) and official code unless marked.
    """
    image_channels: int = 3           # §4 — RGB images
    base_channels: int = 128          # Appendix B — "128 base channels"
    channel_mults: tuple = (1, 2, 2, 2)  # [FROM_OFFICIAL_CODE] — channel multipliers per level
    num_res_blocks: int = 2           # [FROM_OFFICIAL_CODE] — residual blocks per resolution level
    attention_resolutions: tuple = (16,)  # Appendix B — "attention at 16×16 resolution"
    dropout: float = 0.0             # Appendix B — "dropout 0.0" for CIFAR-10
    time_embed_dim: int = 512        # [FROM_OFFICIAL_CODE] — 4 * base_channels
    num_groups: int = 32             # [FROM_OFFICIAL_CODE] — groups for GroupNorm
    image_size: int = 32             # CIFAR-10 is 32×32


# ---------------------------------------------------------------------------
# Time embedding — sinusoidal (borrowed from Transformer positional encoding)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding, following the Transformer positional encoding.

    §3.3 — "The diffusion time t is specified by adding the Transformer
    sinusoidal position embedding into each residual block."

    [FROM_OFFICIAL_CODE] The embedding dimension and MLP projection follow
    the official implementation.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) — integer timesteps

        Returns:
            (batch, embed_dim) — sinusoidal embedding
        """
        half_dim = self.embed_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch, embed_dim)
        return emb


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with time embedding conditioning.

    §3.3 — "group normalization throughout... Transformer sinusoidal position
    embedding into each residual block"

    [FROM_OFFICIAL_CODE] Structure: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Dropout -> Conv + residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        num_groups: int = 32,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection (1x1 conv if channel count changes)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
            t_emb: (batch, time_embed_dim)

        Returns:
            (batch, out_channels, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)                                  # (batch, in_channels, H, W)
        h = self.conv1(h)                              # (batch, out_channels, H, W)

        # Add time embedding
        t = self.time_proj(F.silu(t_emb))              # (batch, out_channels)
        h = h + t.unsqueeze(-1).unsqueeze(-1)          # (batch, out_channels, H, W) — broadcast

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)                              # (batch, out_channels, H, W)

        return h + self.skip(x)                         # residual connection


class AttentionBlock(nn.Module):
    """Self-attention block for the U-Net.

    §3.3 — "self-attention... at the 16×16 feature map resolution"
    Appendix B — "We add one head of self-attention at the 16×16 resolution"

    [FROM_OFFICIAL_CODE] Uses a single attention head with GroupNorm.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            (batch, channels, H, W)
        """
        batch, channels, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(batch, channels, h * w)        # (batch, channels, H*W)

        qkv = self.qkv(x)                          # (batch, 3*channels, H*W)
        q, k, v = qkv.chunk(3, dim=1)              # each: (batch, channels, H*W)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(channels)
        attn = torch.bmm(q.transpose(1, 2), k) * scale  # (batch, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))   # (batch, channels, H*W)
        out = self.proj(out)                         # (batch, channels, H*W)
        out = out.view(batch, channels, h, w)        # (batch, channels, H, W)

        return out + residual


class Downsample(nn.Module):
    """Spatial downsampling by factor 2. [FROM_OFFICIAL_CODE]"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (batch, C, H, W) -> (batch, C, H/2, W/2)


class Upsample(nn.Module):
    """Spatial upsampling by factor 2. [FROM_OFFICIAL_CODE]"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)  # (batch, C, H, W) -> (batch, C, 2H, 2W)


# ---------------------------------------------------------------------------
# §3.3 — Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """§3.3, Appendix B — U-Net noise prediction network ε_θ(x_t, t).

    "We use a U-Net backbone similar to an unmasked PixelCNN++ with
     group normalization throughout, and we add one head of self-attention
     at the 16×16 feature map resolution."

    The U-Net takes a noisy image x_t and a timestep t, and predicts the
    noise ε that was added. This is NOT the paper's core contribution —
    it is the backbone model that enables the diffusion process.

    Architecture (for CIFAR-10 32×32):
      Down: 32→32→16→8→4 (with skip connections)
      Middle: bottleneck with attention
      Up: 4→8→16→32→32 (with skip connections from down path)
    """

    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        ch = config.base_channels

        # Time embedding: sinusoidal -> MLP
        # §3.3 — "Transformer sinusoidal position embedding"
        time_embed_dim = config.time_embed_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(ch),
            nn.Linear(ch, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial convolution
        self.input_conv = nn.Conv2d(config.image_channels, ch, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        channels = [ch]
        current_res = config.image_size
        in_ch = ch

        for level, mult in enumerate(config.channel_mults):
            out_ch = ch * mult
            for _ in range(config.num_res_blocks):
                layers = [ResidualBlock(in_ch, out_ch, time_embed_dim,
                                        config.dropout, config.num_groups)]
                if current_res in config.attention_resolutions:
                    layers.append(AttentionBlock(out_ch, config.num_groups))
                self.down_blocks.append(nn.ModuleList(layers))
                channels.append(out_ch)
                in_ch = out_ch

            if level < len(config.channel_mults) - 1:
                self.down_samples.append(Downsample(out_ch))
                channels.append(out_ch)
                current_res //= 2
            else:
                self.down_samples.append(nn.Identity())

        # Middle (bottleneck)
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_embed_dim,
                                         config.dropout, config.num_groups)
        self.mid_attn = AttentionBlock(in_ch, config.num_groups)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_embed_dim,
                                         config.dropout, config.num_groups)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(len(config.channel_mults))):
            mult = config.channel_mults[level]
            out_ch = ch * mult
            for i in range(config.num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [ResidualBlock(in_ch + skip_ch, out_ch, time_embed_dim,
                                        config.dropout, config.num_groups)]
                if current_res in config.attention_resolutions:
                    layers.append(AttentionBlock(out_ch, config.num_groups))
                self.up_blocks.append(nn.ModuleList(layers))
                in_ch = out_ch

            if level > 0:
                self.up_samples.append(Upsample(out_ch))
                current_res *= 2
            else:
                self.up_samples.append(nn.Identity())

        # Output
        self.output_norm = nn.GroupNorm(config.num_groups, in_ch)
        self.output_conv = nn.Conv2d(in_ch, config.image_channels, kernel_size=3, padding=1)

        # [UNSPECIFIED] Zero-initialize the final conv (from official code)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise ε_θ(x_t, t).

        Args:
            x: (batch, C, H, W) — noisy image x_t
            t: (batch,) — integer timesteps

        Returns:
            (batch, C, H, W) — predicted noise ε_θ
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_embed_dim)

        # Initial conv
        h = self.input_conv(x)  # (batch, base_channels, H, W)

        # Downsampling with skip connections
        skips = [h]
        block_idx = 0
        for level in range(len(self.config.channel_mults)):
            for _ in range(self.config.num_res_blocks):
                layers = self.down_blocks[block_idx]
                h = layers[0](h, t_emb)  # ResidualBlock
                if len(layers) > 1:
                    h = layers[1](h)      # AttentionBlock (if present)
                skips.append(h)
                block_idx += 1

            h = self.down_samples[level](h)
            if not isinstance(self.down_samples[level], nn.Identity):
                skips.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling with skip connections
        block_idx = 0
        for level in reversed(range(len(self.config.channel_mults))):
            for _ in range(self.config.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)  # Concatenate skip connection
                layers = self.up_blocks[block_idx]
                h = layers[0](h, t_emb)  # ResidualBlock
                if len(layers) > 1:
                    h = layers[1](h)      # AttentionBlock (if present)
                block_idx += 1

            h = self.up_samples[level - len(self.config.channel_mults)](h) if level > 0 else h

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        return self.output_conv(h)  # (batch, C, H, W) — predicted noise

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"UNet(\n"
            f"  image_size={self.config.image_size}, base_channels={self.config.base_channels},\n"
            f"  channel_mults={self.config.channel_mults}, num_res_blocks={self.config.num_res_blocks},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )
