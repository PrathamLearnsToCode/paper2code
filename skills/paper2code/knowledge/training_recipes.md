# Knowledge: Training Recipes

## Purpose
Common optimizer configurations, learning rate schedules, and training details that papers frequently assume you know but don't re-explain. When a paper says "we use standard training settings," this file helps you determine what they probably mean (and helps you flag what's still ambiguous).

---

## Optimizers

### Adam (Kingma & Ba, 2014)

Default parameters in the paper: β₁=0.9, β₂=0.999, ε=1e-8

**PyTorch defaults match the Adam paper.**

But in practice:
- Transformer papers often use β₂=0.98 (from Vaswani et al.)
- Some NLP papers use β₂=0.95
- ε values vary: 1e-8 (default), 1e-6 (BERT), 1e-9 (Vaswani et al.)
- "We use Adam" does NOT mean these defaults — you must check

### AdamW (Loshchilov & Hutter, 2019)

AdamW is NOT the same as Adam with L2 regularization:
- Adam + L2: weight decay is applied to the gradient, which scales with the adaptive learning rate
- AdamW: weight decay is applied directly to the weights, decoupled from the gradient update
- This difference matters for large learning rates

```python
# Adam with L2 (WRONG if paper says "AdamW" or "decoupled weight decay"):
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.01)

# AdamW (correct decoupled weight decay):
optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
```

Critical detail: **Weight decay is usually NOT applied to bias terms and LayerNorm parameters.** If the paper doesn't state this, implement it anyway and flag as [ASSUMPTION]:

```python
# [ASSUMPTION] Not applying weight decay to biases and normalization layers
# (standard practice, but paper does not specify)
no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
param_groups = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'weight_decay': config.weight_decay},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(param_groups, lr=config.lr)
```

### SGD with momentum

Common in vision papers (ResNet, EfficientNet):
- Momentum: 0.9 (almost always)
- Weight decay: 1e-4 (vision standard, not universal)
- Nesterov: sometimes on, sometimes off — papers often don't specify

---

## Learning rate schedules

### Linear warmup then constant
```python
def get_warmup_schedule(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Linear warmup then linear decay
```python
def get_linear_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Linear warmup then cosine decay (very common)
```python
def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Transformer schedule (Vaswani et al., 2017)

The original transformer uses a unique schedule that most papers reference:
```python
def get_transformer_schedule(optimizer, d_model, warmup_steps):
    """lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))"""
    def lr_lambda(step):
        step = max(step, 1)  # avoid division by zero
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

Note: This schedule doesn't need a base learning rate — it computes the LR from scratch.
When papers say "following Vaswani et al. for learning rate," this is what they mean.

---

## Batch size semantics

### The nomenclature problem

When a paper says "batch size 256," it could mean:
1. **Per-GPU batch size**: each GPU processes 256 samples
2. **Global batch size**: 256 total across all GPUs
3. **Effective batch size after gradient accumulation**: might be 32 per-GPU × 8 accumulation steps

This is CRITICAL for reproduction because learning rate scaling depends on batch size.

### How to determine which

Check for:
- "per-GPU batch size" or "micro-batch size" → clear
- "total batch size" or "effective batch size" → clear
- Just "batch size" → ambiguous. Look for:
  - Number of GPUs mentioned → if N GPUs and batch size B, probably B total
  - "gradient accumulation steps = K" → if batch B and K steps, effective batch = B × K
  - If they mention both batch size and number of GPUs but don't clarify → flag as PARTIALLY_SPECIFIED

### Learning rate linear scaling rule

Many papers implicitly use the linear scaling rule (Goyal et al., 2017):
- If base LR is `lr` for batch size `B`, then LR for batch size `B'` is `lr * B' / B`
- This applies to SGD. For Adam, the scaling is usually sqrt: `lr * sqrt(B' / B)`
- If the paper trains on 8 GPUs with batch 256 and you use 1 GPU with batch 32, adjust the LR
- **Flag this if you change the batch size from what the paper specifies**

---

## Gradient clipping

### Max gradient norm (most common)
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Apply AFTER `loss.backward()` and BEFORE `optimizer.step()`.

Common values:
- 1.0 (most common)
- 0.5 (some NLP papers)
- 5.0 (some RL papers)
- If paper says "gradient clipping" without a value → UNSPECIFIED

### Max gradient value (less common)
```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```
Different from norm clipping — clips each gradient element independently.

---

## Mixed precision training

### What it means
Train with FP16 (or BF16) for speed, but keep master weights in FP32 for stability.

### PyTorch implementation
```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### What papers don't tell you
- BF16 doesn't need loss scaling (GradScaler) — it has the same dynamic range as FP32
- FP16 DOES need loss scaling to prevent gradient underflow
- Some operations must stay in FP32: softmax, layer norm, loss computation
- PyTorch autocast handles most of this automatically, but custom operations may not be covered
- "We use mixed precision" without specifying FP16 vs BF16 is **PARTIALLY_SPECIFIED**

---

## Exponential Moving Average (EMA)

Common in diffusion models, GANs, and some vision models. Papers often mention EMA without specifying the decay rate.

```python
class EMA:
    """Maintains exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                      for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Load EMA weights into model for evaluation."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
```

Common decay values:
- 0.9999 (diffusion models — DDPM, DDIM)
- 0.999 (some GAN papers)
- 0.99 (faster averaging)
- If paper says "EMA" but not the decay rate → UNSPECIFIED

---

## Common training recipes by domain

### Language models (GPT-style)
- Optimizer: AdamW with β₁=0.9, β₂=0.95, ε=1e-8
- Weight decay: 0.1 (not on biases/norms)
- LR schedule: cosine decay with linear warmup
- Gradient clipping: 1.0 (max norm)
- Batch size: typically reported as total tokens/batch (e.g., "batch of 0.5M tokens")

### Vision transformers (ViT-style)
- Optimizer: AdamW with β₁=0.9, β₂=0.999
- Weight decay: 0.05-0.3
- LR schedule: cosine decay with linear warmup (5-10 epochs warmup)
- Data augmentation: RandAugment, Mixup, CutMix, random erasing (but specific combination varies)
- Label smoothing: 0.1

### Diffusion models (DDPM-style)
- Optimizer: Adam with β₁=0.9, β₂=0.999 (or 0.9999)
- LR: constant (often 2e-4 or 1e-4)
- No LR warmup (sometimes)
- EMA: 0.9999
- Gradient clipping: sometimes but not always

### Contrastive learning (SimCLR-style)
- Optimizer: SGD with momentum 0.9 (or LARS)
- LR schedule: cosine decay after linear warmup (10 epochs)
- Weight decay: 1e-6 (very small)
- Batch size: LARGE (4096+) — crucial for performance
- Temperature: 0.1 or 0.5

**WARNING:** These are guidelines, not specifications. If a paper says "we use standard training," it MIGHT mean the recipe above — but it's still UNSPECIFIED unless the paper states specifics. Use this knowledge to choose reasonable defaults, not to skip flagging.
