"""
Attention Is All You Need — Model Architecture

Paper: https://arxiv.org/abs/1706.03762
Authors: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017)

Implements: The Transformer model — an encoder-decoder architecture based entirely
on multi-head self-attention, with no recurrence or convolution.

Section references:
  §3.1 — Encoder and Decoder Stacks
  §3.2 — Attention (Scaled Dot-Product and Multi-Head)
  §3.3 — Position-wise Feed-Forward Networks
  §3.4 — Embeddings and Softmax
  §3.5 — Positional Encoding

Usage:
    from src.model import Transformer, TransformerConfig

    config = TransformerConfig()
    model = Transformer(config)
    output = model(src_tokens, tgt_tokens)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    """All model hyperparameters.

    Values from Vaswani et al. 2017, Table 3 ("base model") unless marked [UNSPECIFIED].
    """
    # Architecture — §3, Table 3
    d_model: int = 512          # §3.1, Table 3 — "d_model = 512"
    n_heads: int = 8            # §3.2.2, Table 3 — "h = 8"
    d_ff: int = 2048            # §3.3, Table 3 — "d_ff = 2048"
    n_encoder_layers: int = 6   # §3.1, Table 3 — "N = 6"
    n_decoder_layers: int = 6   # §3.1, Table 3 — "N = 6"
    dropout: float = 0.1        # §5.4, Table 3 — "P_drop = 0.1"
    vocab_size: int = 37000     # §5.1 — "~37000 tokens" (EN-DE BPE)
    max_seq_len: int = 5000     # [UNSPECIFIED] — max length for positional encoding
    norm_eps: float = 1e-6      # [UNSPECIFIED] — LayerNorm epsilon not stated. Alternatives: 1e-5 (PyTorch default)
    tie_weights: bool = True    # §3.4 — "we share the same weight matrix"


# ---------------------------------------------------------------------------
# §3.2.1 — Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> torch.Tensor:
    """§3.2.1, Eq. 1 — Attention(Q, K, V) = softmax(QK^T / √d_k) V

    "We compute the attention function on a set of queries simultaneously,
     packed together into a matrix Q."

    Args:
        query: (batch, n_heads, seq_q, d_k)
        key:   (batch, n_heads, seq_k, d_k)
        value: (batch, n_heads, seq_k, d_v)
        mask:  (batch, 1, seq_q, seq_k) or (1, 1, seq_q, seq_k) — True = attend, False = mask
        dropout: optional dropout on attention weights

    Returns:
        (batch, n_heads, seq_q, d_v) — attention output
    """
    d_k = query.size(-1)

    # §3.2.1 — "We compute the dot products of the query with all keys,
    # divide each by √d_k, and apply a softmax"
    # (batch, n_heads, seq_q, d_k) @ (batch, n_heads, d_k, seq_k) -> (batch, n_heads, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # §3.2.3 — "We implement this inside of scaled dot-product attention by masking out
        # (setting to −∞) all values in the input of the softmax which correspond to
        # illegal connections"
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # (batch, n_heads, seq_q, seq_k)
    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # (batch, n_heads, seq_q, seq_k) @ (batch, n_heads, seq_k, d_v) -> (batch, n_heads, seq_q, d_v)
    return torch.matmul(attn_weights, value)


# ---------------------------------------------------------------------------
# §3.2.2 — Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """§3.2.2 — Multi-Head Attention.

    "Multi-head attention allows the model to jointly attend to information
     from different representation subspaces at different positions."

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads  # §3.2.2 — "d_k = d_v = d_model/h = 64"

        # §3.2.2 — "We employ h = 8 parallel attention layers, or heads"
        # "We use d_k = d_v = d_model/h = 64"
        # [UNSPECIFIED] Paper does not state whether projection layers have bias.
        # Using: bias=True (common default)
        # Alternatives: bias=False (used in some reimplementations like LLaMA)
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  (batch, 1, seq_q, seq_k) or broadcastable shape

        Returns:
            (batch, seq_q, d_model)
        """
        batch_size = query.size(0)

        # §3.2.2 — "linearly project the queries, keys and values h times with different,
        # learned linear projections to d_k, d_k, and d_v dimensions"
        # (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # §3.2.1 — apply scaled dot-product attention
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        # (batch, n_heads, seq_q, d_k)

        # §3.2.2 — "concatenated and once again projected"
        # (batch, n_heads, seq_q, d_k) -> (batch, seq_q, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection W^O
        return self.W_o(attn_out)  # (batch, seq_q, d_model)


# ---------------------------------------------------------------------------
# §3.3 — Position-wise Feed-Forward Networks
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """§3.3 — Position-wise Feed-Forward Networks.

    "FFN(x) = max(0, xW_1 + b_1)W_2 + b_2" (Eq. 2)

    "The dimensionality of input and output is d_model = 512,
     and the inner-layer has dimensionality d_ff = 2048."
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        # §3.3, Eq. 2 — two linear transformations with ReLU in between
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # §3.3, Eq. 2 — FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        x = self.linear1(x)          # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = F.relu(x)                # §3.3, Eq. 2 — ReLU activation
        x = self.dropout(x)          # §5.4 — "applied... to the output of each sub-layer"
        x = self.linear2(x)          # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return x


# ---------------------------------------------------------------------------
# §3.5 — Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """§3.5 — Positional Encoding (sinusoidal).

    "We use sine and cosine functions of different frequencies:"
      PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    "We chose this function because we hypothesized it would allow the model
     to easily learn to attend by relative positions."
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

        # §3.5 — precompute positional encodings
        pe = torch.zeros(config.max_seq_len, config.d_model)               # (max_len, d_model)
        position = torch.arange(0, config.max_seq_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float()
            * (-math.log(10000.0) / config.d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices: cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — for batch broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — embedding output (already scaled by √d_model)
        Returns:
            (batch, seq_len, d_model) — with positional encoding added
        """
        # §3.5 — "We apply dropout to the sums of the embeddings and the positional encodings"
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# §3.1 — Encoder Layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """§3.1 — Single encoder layer.

    "Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
     and the second is a simple, position-wise fully connected feed-forward network."

    "We employ a residual connection around each of the two sub-layers,
     followed by layer normalization."

    LayerNorm(x + Sublayer(x))  — post-norm formulation
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        # §3.1 — two sub-layers with residual connections and layer norm
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        # §3.1 — "layer normalization" after each residual connection
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        # §5.4 — "Residual Dropout: We apply dropout to the output of each sub-layer,
        # before it is added to the sub-layer input and normalized"
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            src_mask: (batch, 1, 1, seq_len) — padding mask
        Returns:
            (batch, seq_len, d_model)
        """
        # §3.1 — Sub-layer 1: Multi-head self-attention
        # Post-norm: LayerNorm(x + Dropout(Sublayer(x)))
        attn_out = self.self_attn(x, x, x, mask=src_mask)  # (batch, seq_len, d_model)
        x = self.norm1(x + self.dropout1(attn_out))          # (batch, seq_len, d_model)

        # §3.1 — Sub-layer 2: Feed-forward network
        ff_out = self.feed_forward(x)                         # (batch, seq_len, d_model)
        x = self.norm2(x + self.dropout2(ff_out))             # (batch, seq_len, d_model)

        return x


# ---------------------------------------------------------------------------
# §3.1 — Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """§3.1 — Single decoder layer.

    "In addition to the two sub-layers in each encoder layer, the decoder inserts
     a third sub-layer, which performs multi-head attention over the output of the
     encoder stack."

    Three sub-layers:
      1. Masked multi-head self-attention (causal)
      2. Multi-head cross-attention (over encoder output)
      3. Position-wise feed-forward network
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        # §3.1 — three sub-layers
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model) — decoder input
            memory: (batch, src_len, d_model) — encoder output
            tgt_mask: (1, 1, tgt_len, tgt_len) — causal mask
            memory_mask: (batch, 1, 1, src_len) — source padding mask
        Returns:
            (batch, tgt_len, d_model)
        """
        # §3.1 — Sub-layer 1: Masked self-attention
        # §3.2.3 — "We need to prevent leftward information flow in the decoder
        # to preserve the auto-regressive property"
        self_attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))  # (batch, tgt_len, d_model)

        # §3.1 — Sub-layer 2: Cross-attention over encoder output
        # "queries come from the previous decoder layer, and the memory keys and values
        # come from the output of the encoder"
        cross_attn_out = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))  # (batch, tgt_len, d_model)

        # §3.1 — Sub-layer 3: Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))  # (batch, tgt_len, d_model)

        return x


# ---------------------------------------------------------------------------
# Encoder and Decoder Stacks
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """§3.1 — Encoder: "The encoder is composed of a stack of N = 6 identical layers." """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_encoder_layers)
        ])

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, src_len, d_model) — embedded + positionally-encoded source
            mask: (batch, 1, 1, src_len) — source padding mask
        Returns:
            (batch, src_len, d_model) — encoder output (memory)
        """
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return x


class Decoder(nn.Module):
    """§3.1 — Decoder: "The decoder is also composed of a stack of N = 6 identical layers." """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_decoder_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model) — embedded + positionally-encoded target
            memory: (batch, src_len, d_model) — encoder output
            tgt_mask: (1, 1, tgt_len, tgt_len) — causal mask
            memory_mask: (batch, 1, 1, src_len) — source padding mask
        Returns:
            (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x


# ---------------------------------------------------------------------------
# §3 — Full Transformer Model
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """§3 — The Transformer (Encoder-Decoder).

    "Most competitive neural sequence transduction models have an encoder-decoder structure.
     Here, the encoder maps an input sequence of symbol representations (x_1, ..., x_n) to
     a sequence of continuous representations z = (z_1, ..., z_n). Given z, the decoder then
     generates an output sequence (y_1, ..., y_m) of symbols one element at a time."

    Composed of:
      - Token embedding with √d_model scaling (§3.4)
      - Positional encoding (§3.5)
      - Encoder stack — N=6 layers (§3.1)
      - Decoder stack — N=6 layers (§3.1)
      - Linear output projection with weight tying (§3.4)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # §3.4 — "Similarly to other sequence transduction models, we use learned embeddings
        # to convert the input tokens and output tokens to vectors of dimension d_model"
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # §3.4 — "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation"
        if config.tie_weights:
            self.tgt_embedding.weight = self.src_embedding.weight

        # §3.5 — Positional encoding (shared between encoder and decoder)
        self.pos_encoding = PositionalEncoding(config)

        # §3.1 — Encoder and decoder stacks
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # §3.4 — Output linear projection
        # "we share the same weight matrix between the two embedding layers
        # and the pre-softmax linear transformation"
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.output_projection.weight = self.src_embedding.weight

        # §3.4 — embedding scale factor: √d_model
        self.embed_scale = math.sqrt(config.d_model)

        # [UNSPECIFIED] Paper does not describe weight initialization.
        # Using: Xavier uniform for all linear layers (common for transformers)
        # Alternatives: PyTorch defaults, normal(0, 0.02)
        self._init_weights()

    def _init_weights(self):
        """[UNSPECIFIED] — Weight initialization not described in paper.
        Using Xavier uniform, a common choice for transformer models.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full encoder-decoder forward pass.

        Args:
            src: (batch, src_len) — source token IDs
            tgt: (batch, tgt_len) — target token IDs (shifted right for teacher forcing)
            src_mask: (batch, 1, 1, src_len) — source padding mask. Auto-generated if None.
            tgt_mask: (1, 1, tgt_len, tgt_len) — causal mask. Auto-generated if None.

        Returns:
            (batch, tgt_len, vocab_size) — logits over vocabulary
        """
        # Auto-generate masks if not provided
        if src_mask is None:
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
        if tgt_mask is None:
            tgt_mask = self._make_causal_mask(tgt.size(1), tgt.device)  # (1, 1, tgt_len, tgt_len)

        # §3.4 — "In the embedding layers, we multiply those weights by √d_model"
        src_emb = self.src_embedding(src) * self.embed_scale  # (batch, src_len, d_model)
        tgt_emb = self.tgt_embedding(tgt) * self.embed_scale  # (batch, tgt_len, d_model)

        # §3.5 — Add positional encoding
        src_emb = self.pos_encoding(src_emb)  # (batch, src_len, d_model)
        tgt_emb = self.pos_encoding(tgt_emb)  # (batch, tgt_len, d_model)

        # §3.1 — Encode
        memory = self.encoder(src_emb, mask=src_mask)  # (batch, src_len, d_model)

        # §3.1 — Decode
        dec_out = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )  # (batch, tgt_len, d_model)

        # §3.4 — Linear projection to vocabulary
        logits = self.output_projection(dec_out)  # (batch, tgt_len, vocab_size)
        return logits

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """§3.2.3 — Create causal (autoregressive) mask for decoder self-attention.

        "We also modify the self-attention sub-layer in the decoder stack to prevent
         positions from attending to subsequent positions."

        Returns:
            (1, 1, seq_len, seq_len) boolean tensor. True = attend, False = mask.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"Transformer(\n"
            f"  config=TransformerConfig(d_model={self.config.d_model}, n_heads={self.config.n_heads}, "
            f"d_ff={self.config.d_ff}, n_layers={self.config.n_encoder_layers}),\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )
