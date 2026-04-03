# Stage 5: Walkthrough Notebook

## Purpose
Generate a Jupyter notebook that connects paper sections to code, making the implementation pedagogically useful. This is not for training — it's for understanding. A researcher should be able to read this notebook and understand exactly how every paper section maps to code.

## Input
- All generated code from Stage 4
- Paper sections from Stage 1
- Contribution analysis from Stage 2

## Output
- `{paper_slug}/notebooks/walkthrough.ipynb`

---

## Notebook structure

The notebook follows a strict section pattern. For each major component of the implementation:

### Cell pattern (repeat for each component)

#### Cell 1: Paper excerpt (Markdown)
A quoted block of the relevant paper passage that describes this component. Include the section number and, if applicable, the equation.

```markdown
## Multi-Head Attention (§3.2)

> "An attention function can be described as mapping a query and a set of key-value
> pairs to an output, where the query, keys, values, and output are all vectors.
> The output is computed as a weighted sum of the values, where the weight assigned
> to each value is computed by a compatibility function of the query with the
> corresponding key."
>
> — §3.2, Attention Is All You Need

**Equation 1:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
```

#### Cell 2: Code (Python)
The corresponding code from `src/model.py` or wherever this component is defined. Don't just import — inline the relevant class/function so the reader can see it without switching files.

```python
# From src/model.py — Multi-Head Attention

class MultiHeadAttention(nn.Module):
    """§3.2 — Multi-Head Attention mechanism."""
    # ... full implementation
```

#### Cell 3: Sanity check (Python)
A runnable check that verifies the component works correctly with toy dimensions:

```python
# Sanity check: verify output shapes match paper specification
config = ModelConfig(d_model=64, n_heads=4)  # Toy dimensions for CPU
attn = MultiHeadAttention(config)

batch, seq_len = 2, 10
x = torch.randn(batch, seq_len, config.d_model)

output = attn(x)
assert output.shape == (batch, seq_len, config.d_model), \
    f"Expected (2, 10, 64), got {output.shape}"
print(f"✓ MultiHeadAttention: input {x.shape} → output {output.shape}")
```

---

## Required sections

### 1. Setup and imports
```python
import torch
import torch.nn as nn
import math
# ... minimal imports

# Use tiny dimensions so everything runs on CPU quickly
BATCH_SIZE = 2
SEQ_LEN = 16
D_MODEL = 64  # Paper uses 512, we use 64 for walkthrough
# ... etc
```

Explain why dimensions are small: "We use reduced dimensions so the notebook runs instantly on CPU. The architecture is identical — only the numbers change."

### 2. Configuration
Show the config dataclass with paper citations on each field. Instantiate with toy dimensions.

### 3. One section per architectural component
Follow the paper's presentation order. If the paper describes components A, B, C used to build the model, the notebook should have sections for A, B, C in that order, then the composed model.

For each: paper excerpt → code → sanity check.

### 4. Full model assembly
Show how the components compose into the full model. Run a full forward pass with random input. Verify output shape.

```python
# Full model forward pass
model = TheModel(config)
x = torch.randn(BATCH_SIZE, SEQ_LEN)  # or appropriate input
output = model(x)
print(f"✓ Full model: input {x.shape} → output {output.shape}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 5. Loss function (if in scope)
Paper excerpt for the loss function → code → compute loss on random predictions.

### 6. Training step (if in scope)
Show one forward + backward pass. Verify gradients flow:

```python
# One training step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()

# Verify gradients exist
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"⚠ No gradient for: {name}")
    else:
        print(f"✓ {name}: grad norm = {param.grad.norm():.4f}")
```

### 7. Common pitfalls (Markdown)
A cell listing what typically goes wrong when implementing this paper:

```markdown
## Common Pitfalls

1. **Pre-norm vs post-norm**: The paper's figure shows post-norm but experiments
   use pre-norm. This significantly affects training stability. [§3.2]

2. **Attention scale factor**: Must divide by √d_k, not √d_model. If using
   multi-head attention, d_k = d_model / n_heads. [§3.2, Eq. 1]

3. **Positional encoding**: The sinusoidal encoding uses specific frequencies.
   Off-by-one errors in the position index are common. [§3.5]

4. **Label smoothing**: The paper uses ε=0.1 label smoothing. PyTorch's
   CrossEntropyLoss does not support this directly — you need a custom
   implementation. [§5.4]

5. **Learning rate schedule**: The warmup schedule has a specific formula.
   Using a generic linear warmup gives different results. [§5.3]
```

---

## Notebook generation rules

1. **Everything must run on CPU.** No GPU required. Use tiny dimensions.
2. **No external data needed.** All inputs are random tensors.
3. **No training to convergence.** Just verify shapes, gradients, and basic numerics.
4. **Each cell should be self-contained enough to understand.** Imports can be at the top, but each section should be readable on its own.
5. **Markdown cells should quote the paper directly.** Don't paraphrase — use exact quotes with section references.
6. **Sanity checks should be assertions, not just prints.** Assert the expected shapes so failures are loud.
7. **Use the paper's variable names in code.** If the paper uses Q, K, V — the notebook code should too.

---

## Mode-specific behavior

### minimal mode
Standard notebook as described above.

### educational mode
Add additional cells:
- Before each component: a "Background" markdown cell explaining the ML concept (what is attention? what is a residual connection?) for readers who are learning
- After each sanity check: a "What would happen if..." cell exploring edge cases (what if you remove the scale factor? what if you use post-norm instead of pre-norm?)
- A "Further Reading" section at the end pointing to tutorials and explanations

### full mode
Same as minimal but includes:
- Full data loading example (with mock data)
- Complete training loop (3-5 steps, not to convergence)
- Evaluation metric computation on random predictions

---

## Self-check

- [ ] Notebook runs end-to-end without errors (all cells execute)
- [ ] All assertion-based sanity checks pass
- [ ] Every paper section relevant to the implementation is quoted
- [ ] Toy dimensions are used throughout (runs on CPU in seconds)
- [ ] Common pitfalls section exists and is genuinely useful
- [ ] No external data or GPU required
