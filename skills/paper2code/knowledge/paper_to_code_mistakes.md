# Knowledge: Paper-to-Code Translation Mistakes

## Purpose
A detailed catalog of patterns where naive paper-to-code translation produces wrong results. These are not bugs in the usual sense — the code runs fine but implements something different from what the paper describes. This file exists because these mistakes are systematic and predictable.

---

## Notation mismatches

### Batch normalization momentum

**The problem:** PyTorch and TensorFlow define momentum differently.

Paper says: `running_mean = (1 - α) * running_mean + α * batch_mean` where α (or momentum) = 0.1

- **PyTorch** `nn.BatchNorm2d(momentum=0.1)`: uses `running_mean = (1 - momentum) * running_mean + momentum * batch_mean` → matches the formula above with momentum=0.1
- **TensorFlow** `tf.keras.layers.BatchNormalization(momentum=0.99)`: uses `running_mean = momentum * running_mean + (1 - momentum) * batch_mean` → to get the same behavior, set momentum=0.9

**Translation rule:** PyTorch `momentum=x` ≈ TensorFlow `momentum=1-x`

If a paper reports momentum without specifying the framework convention → PARTIALLY_SPECIFIED. Check which framework the official code uses.

### Dropout rate vs keep probability

**The problem:** Older papers (and TensorFlow v1) use "keep probability." Newer papers and PyTorch use "drop probability."

- Paper says "dropout 0.1" → almost certainly means drop probability = 0.1 (keep 90% of neurons)
- Paper says "dropout rate 0.9" → probably means keep probability = 0.9 (same thing)
- Paper says "we keep 90% of neurons" → keep probability = 0.9

```python
# PyTorch: p = drop probability
nn.Dropout(p=0.1)  # drops 10%, keeps 90%
```

**When in doubt:** Check the dropout paper (Srivastava et al., 2014) which uses keep probability notation. Most post-2018 papers use drop probability.

### Convolution padding

**The problem:** "same padding" means different things.

- Paper says "same convolution" → output size = input size
- PyTorch: `nn.Conv2d(padding='same')` or manually compute `padding = kernel_size // 2`
- TensorFlow: `tf.keras.layers.Conv2D(padding='same')` handles it automatically
- For even kernel sizes, "same" can't be symmetric → PyTorch requires explicit asymmetric padding via `nn.functional.pad`

### Tensor dimension ordering

- PyTorch images: `(batch, channels, height, width)` — NCHW
- TensorFlow images: `(batch, height, width, channels)` — NHWC (default)
- If converting a TensorFlow paper to PyTorch, every convolution, pooling, normalization, and reshape must account for this

---

## Activation function gotchas

### GELU approximation

There are two common GELU implementations:

```python
# Exact GELU (PyTorch default since 1.12):
nn.GELU()
# = x * Φ(x) where Φ is the standard normal CDF

# Tanh approximation (used in GPT-2, BERT):
nn.GELU(approximate='tanh')
# = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

These give slightly different results. Some papers use one, some the other. BERT uses the tanh approximation. Modern papers usually use exact GELU but often don't specify.

### Swish / SiLU

```python
# These are the same thing:
nn.SiLU()  # PyTorch name
# = x * sigmoid(x)

# Some papers call it Swish with a trainable β:
# swish(x) = x * sigmoid(β * x)
# When β=1, this is SiLU. PyTorch's SiLU always uses β=1.
```

If the paper says "Swish" without specifying β, use β=1 (SiLU) and flag it.

### Leaky ReLU slope

Papers rarely specify the negative slope for Leaky ReLU:
- Paper default: 0.01
- PyTorch default: 0.01 (matches)
- But some implementations use 0.1 or 0.2 (especially in GANs)

---

## Weight initialization gotchas

### Xavier/Glorot initialization

```python
# Xavier uniform (Glorot & Bengio, 2010):
nn.init.xavier_uniform_(layer.weight)  # U(-a, a) where a = sqrt(6 / (fan_in + fan_out))

# Xavier normal:
nn.init.xavier_normal_(layer.weight)  # N(0, 2 / (fan_in + fan_out))
```

**The problem:** "Xavier initialization" doesn't specify uniform or normal. They're different. Papers rarely specify which one.

### Kaiming/He initialization

```python
# Kaiming uniform (He et al., 2015):
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

# Kaiming normal:
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

**The problem:** `mode='fan_in'` vs `mode='fan_out'` matters. PyTorch defaults to `fan_in`, but some papers require `fan_out` for the decoder.

### PyTorch default initialization

If the paper doesn't mention initialization, PyTorch applies:
- `nn.Linear`: Kaiming uniform (fan_in)
- `nn.Conv2d`: Kaiming uniform (fan_in)
- `nn.Embedding`: Normal(0, 1)
- `nn.LayerNorm`: weight=1, bias=0
- `nn.BatchNorm`: weight=1, bias=0

These defaults are reasonable but may not match what the paper used (and didn't specify).

---

## Dimension and reshaping gotchas

### view() vs reshape()

```python
# view() requires contiguous memory — will error if not contiguous
x = x.view(batch, -1)

# reshape() handles non-contiguous tensors by copying if necessary
x = x.reshape(batch, -1)
```

After `transpose()` or `permute()`, tensors are NOT contiguous. Use `.contiguous()` before `.view()` or just use `.reshape()`.

### einsum notation

Papers increasingly use einsum notation. Common patterns:

```python
# Matrix multiplication:
torch.einsum('bij,bjk->bik', A, B)  # same as torch.bmm(A, B)

# Attention scores:
torch.einsum('bhqd,bhkd->bhqk', q, k)  # (batch, heads, q_len, k_len)

# Bilinear:
torch.einsum('bi,ij,bj->b', x, W, y)
```

**Gotcha:** einsum is correct but can be slow for some patterns. For standard operations (matmul, dot product), use `torch.matmul` or `@` operator — they're optimized.

---

## Loss function gotchas

### Cross-entropy: logits vs probabilities

If you see `loss = -sum(y * log(p))` in a paper, this is cross-entropy applied to probabilities.
PyTorch `nn.CrossEntropyLoss` expects LOGITS and applies log-softmax internally.

```python
# WRONG: passing softmax output to CrossEntropyLoss
probs = F.softmax(logits, dim=-1)
loss = nn.CrossEntropyLoss()(probs, targets)  # WRONG — double softmax

# CORRECT:
loss = nn.CrossEntropyLoss()(logits, targets)  # applies log-softmax internally
```

### MSE reduction

Papers often specify loss as a sum or average differently:
- PyTorch `MSELoss(reduction='mean')`: averages over ALL elements (batch × features)
- Some papers average only over features, then sum over batch
- Some papers sum everything, no averaging

Check the paper's equation: does it have $\frac{1}{N}$ (mean) or $\sum$ (sum)?

### KL divergence direction

$KL(P || Q) \neq KL(Q || P)$

- Forward KL ($KL(P || Q)$): "I want Q to cover everywhere P has mass" — mode-covering
- Reverse KL ($KL(Q || P)$): "I want Q to be 0 wherever P is 0" — mode-seeking

PyTorch's `nn.KLDivLoss` computes `KL(target || input)` — note the argument order!

```python
# PyTorch expects LOG probabilities as input, regular probabilities as target
loss = nn.KLDivLoss(reduction='batchmean')(
    F.log_softmax(predicted_logits, dim=-1),  # LOG probs
    target_probs  # regular probs
)
```

---

## Training loop gotchas

### optimizer.zero_grad() placement

```python
# CORRECT: zero gradients BEFORE forward pass
optimizer.zero_grad()
output = model(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()

# ALSO CORRECT (set_to_none is faster):
optimizer.zero_grad(set_to_none=True)
```

### Learning rate scheduler step

Different schedulers step at different rates:
```python
# Per-step schedulers (most warmup schedules):
for step in range(total_steps):
    train_step()
    scheduler.step()  # after every training step

# Per-epoch schedulers (ReduceLROnPlateau, StepLR):
for epoch in range(epochs):
    train_epoch()
    scheduler.step()  # after every epoch
```

Papers often don't specify when the scheduler steps. If it's a warmup schedule, it's per-step.

### Gradient accumulation

```python
# If effective_batch = micro_batch × accumulation_steps:
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # DIVIDE by accumulation steps
    loss.backward()  # accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Common mistake:** Forgetting to divide the loss by `accumulation_steps`. Without this, the effective learning rate is scaled by the number of accumulation steps.

---

## Evaluation gotchas

### BLEU score implementations

Different BLEU implementations give different numbers:
- `sacrebleu` (recommended, reproducible): uses standard tokenization
- `nltk.translate.bleu_score`: requires pre-tokenized input
- `torchtext.data.metrics.bleu_score`: deprecated
- Papers before 2020 may use any of these

**Difference can be 1-2 BLEU points** — significant for reporting.

### FID (Fréchet Inception Distance)

- Must use the SAME Inception v3 model weights (the default TF weights)
- PyTorch and TensorFlow Inception models give slightly different features
- Number of samples used for FID computation matters (more ≈ more stable)
- `clean-fid` package is the recommended implementation

### Top-k accuracy

```python
def topk_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy.
    output: (batch, num_classes) — logits
    target: (batch,) — class indices
    """
    _, pred = output.topk(k, dim=-1)  # (batch, k)
    correct = pred.eq(target.unsqueeze(-1)).any(dim=-1)  # (batch,)
    return correct.float().mean().item()
```

---

## Miscellaneous gotchas

### model.eval() vs model.train()

```python
model.eval()   # Disables dropout and uses running stats for batch norm
model.train()  # Enables dropout and computes batch stats for batch norm

# CRITICAL: Always call model.eval() before evaluation/inference
# Forgetting this is a very common source of poor eval performance
```

### torch.no_grad() vs torch.inference_mode()

```python
# For evaluation (no gradient computation):
with torch.no_grad():
    output = model(input)
    
# For inference (even faster, more restricted):
with torch.inference_mode():
    output = model(input)
```

### Determinism

If the paper reports "we average over 3 random seeds":
```python
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
Note: `cudnn.deterministic = True` can significantly slow down training.
