# Knowledge: Transformer Components

## Purpose
Canonical correct implementations of transformer building blocks that papers frequently reference but don't re-explain. When a paper says "standard transformer encoder," this file tells you what that means and what mistakes to avoid.

---

## Multi-Head Attention

### Canonical implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # head dimension

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # Project and reshape: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # (batch, n_heads, seq_q, d_k) @ (batch, n_heads, d_k, seq_k) -> (batch, n_heads, seq_q, seq_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_q, seq_k)
        attn_weights = self.dropout(attn_weights)

        # (batch, n_heads, seq_q, seq_k) @ (batch, n_heads, seq_k, d_k) -> (batch, n_heads, seq_q, d_k)
        context = torch.matmul(attn_weights, v)

        # Reshape back: (batch, n_heads, seq_q, d_k) -> (batch, seq_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)  # (batch, seq_q, d_model)
```

### Common mistakes

1. **Scaling by sqrt(d_model) instead of sqrt(d_k)**
   - The scale factor is `sqrt(d_k)` where `d_k = d_model / n_heads`
   - NOT `sqrt(d_model)`. This is the single most common mistake.
   - Vaswani et al. §3.2.1: "We suspect that for large values of d_k, the dot products grow large in magnitude"

2. **Wrong mask convention**
   - Additive mask: add a large negative number (e.g., -1e9 or -inf) to scores BEFORE softmax
   - Multiplicative mask: multiply attention weights by 0/1 AFTER softmax
   - Both are valid, but papers rarely specify which. Additive is more common and numerically cleaner.
   - Using -inf is cleaner than -1e9 (avoids non-zero attention for -1e9 with float16)

3. **Forgetting `.contiguous()` after transpose**
   - After `transpose(1, 2)`, the tensor may not be contiguous
   - `.view()` requires a contiguous tensor
   - This will crash, not silently fail — but it's a common "why doesn't my code run" bug

4. **Causal mask shape**
   - Should be `(1, 1, seq_len, seq_len)` for broadcasting with `(batch, n_heads, seq_len, seq_len)` scores
   - Mask where `mask[i][j] = 1` if position `j` is allowed for position `i`
   - Upper triangular = disallowed, not lower triangular (common mistake)

### Causal masking

```python
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Creates a causal (autoregressive) attention mask.
    Returns a (1, 1, seq_len, seq_len) boolean tensor where True = attend, False = mask.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

---

## Positional Encodings

### Sinusoidal (Vaswani et al., 2017)

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from 'Attention Is All You Need'.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Common mistakes with sinusoidal PE:
- Using `arange(0, d_model)` instead of `arange(0, d_model, 2)` for div_term
- Off-by-one in position indexing (should start at 0)
- Forgetting to `register_buffer` (so it's not a parameter but moves with the model to GPU)

### Learned positional embeddings

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device)  # (seq_len,)
        return x + self.embedding(positions)  # broadcast over batch
```

### Rotary Position Embedding (RoPE) — Su et al., 2021

```python
class RotaryPositionalEmbedding(nn.Module):
    """RoPE: Enhanced Transformer with Rotary Position Embedding.

    Applied to each head individually within the attention computation,
    AFTER the Q and K projections but BEFORE the dot product.
    """
    def __init__(self, d_head: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_head)
        return emb.cos(), emb.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to query or key tensor.
    x: (batch, n_heads, seq_len, d_head)
    """
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([
        x1 * cos[..., :d_half] - x2 * sin[..., :d_half],
        x2 * cos[..., d_half:] + x1 * sin[..., d_half:]
    ], dim=-1)
```

### Key difference: RoPE is applied to Q and K individually, NOT summed onto embeddings like sinusoidal PE.

### ALiBi (Press et al., 2022)

ALiBi doesn't use positional embeddings at all. Instead, it adds a linear bias to the attention scores:

```python
def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each head.
    Head i gets slope 2^(-8i/n_heads) for i = 1, ..., n_heads
    """
    ratio = 2 ** (-8 / n_heads)
    slopes = torch.tensor([ratio ** i for i in range(1, n_heads + 1)])
    return slopes  # (n_heads,)

def apply_alibi(scores: torch.Tensor, slopes: torch.Tensor) -> torch.Tensor:
    """Apply ALiBi bias to attention scores.
    scores: (batch, n_heads, seq_q, seq_k)
    slopes: (n_heads,)
    """
    seq_q, seq_k = scores.size(-2), scores.size(-1)
    # Position difference: relative distance between query and key positions
    positions = torch.arange(seq_k, device=scores.device).unsqueeze(0) - \
                torch.arange(seq_q, device=scores.device).unsqueeze(1)  # (seq_q, seq_k)
    bias = slopes.unsqueeze(-1).unsqueeze(-1) * positions.unsqueeze(0)  # (n_heads, seq_q, seq_k)
    return scores + bias.unsqueeze(0)  # broadcast over batch
```

---

## Layer Normalization

### Pre-norm vs Post-norm — THIS MATTERS ENORMOUSLY

**Post-norm (original Transformer):**
```python
# Post-norm: normalize AFTER the residual addition
x = self.norm(x + self.sublayer(x))
```

**Pre-norm (GPT-2, most modern transformers):**
```python
# Pre-norm: normalize BEFORE the sublayer, residual OUTSIDE the norm
x = x + self.sublayer(self.norm(x))
```

**Why it matters:**
- Post-norm requires learning rate warmup and careful initialization
- Pre-norm is much more stable to train at scale
- They produce different quality models — not interchangeable
- Many papers show post-norm in figures but use pre-norm in experiments — ALWAYS CHECK

### RMSNorm (Zhang & Sennrich, 2019)

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Used in LLaMA, T5. Simpler than LayerNorm (no centering, no bias).
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

---

## Feed-Forward Network

### Standard (Vaswani et al.)
```python
class FeedForward(nn.Module):
    """Two-layer feed-forward network with expansion factor.
    FFN(x) = W_2 * activation(W_1 * x + b_1) + b_2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, d_model) -> (batch, seq, d_ff) -> (batch, seq, d_model)
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

### SwiGLU (Shazeer, 2020) — used in LLaMA, PaLM
```python
class SwiGLU(nn.Module):
    """Gated feed-forward with SiLU activation.
    SwiGLU(x) = (SiLU(W_1 * x) ⊙ W_3 * x) * W_2
    Note: uses 3 weight matrices, not 2. This changes parameter count.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## Embedding with Weight Tying

```python
class TransformerEmbedding(nn.Module):
    """Token + positional embedding with optional weight tying to output projection.

    Weight tying (Press & Wolf, 2017): The embedding matrix and the output
    projection matrix are the SAME tensor. This reduces parameters and often
    improves performance. Many papers do this without mentioning it explicitly.
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int,
                 dropout: float = 0.0, scale: bool = True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEmbedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model) if scale else 1.0
        # Vaswani et al. §3.4: "we multiply those weights by sqrt(d_model)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) of token IDs
        tok = self.token_emb(x) * self.scale  # (batch, seq_len, d_model)
        return self.dropout(self.pos_emb(tok))
```

**Weight tying note:** If the paper ties embedding and output weights, the output projection is `F.linear(x, model.embedding.token_emb.weight)` — not a separate `nn.Linear`. Many papers do this without stating it. Check the parameter count in the paper against your model — if yours is higher, weight tying might be missing.

---

## Complete Transformer Block

### Post-norm variant (original)
```python
class TransformerBlockPostNorm(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x
```

### Pre-norm variant (modern standard)
```python
class TransformerBlockPreNorm(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x
```

**Key difference:** In pre-norm, LayerNorm is applied BEFORE each sublayer. The residual connection adds the UN-normalized input. This is more stable for training deep models.
