# Guardrail: Hallucination Prevention

## The core problem

The single most dangerous failure mode of paper2code is **confident invention** — generating code that looks plausible but implements details the paper never specified, without flagging them. A researcher who trusts this output will waste days debugging differences that exist because the model guessed.

This file exists to prevent that. Read it before every code generation stage. Internalize it.

---

## The bright line rule

**If a detail is not explicitly stated in the paper text, the appendix, the paper's official GitHub repository, or a well-known replication paper — it is UNSPECIFIED.**

Not "probably this." Not "standard practice." Not "everyone uses." **UNSPECIFIED.**

---

## What counts as "stated in the paper"

### Counts as SPECIFIED:
- Direct statement: "We use d_model = 512" → SPECIFIED
- Table entry: Table 3 shows "learning rate: 3e-4" → SPECIFIED
- Equation: Eq. 4 defines the loss function → SPECIFIED (for the equation, not for implementation details like numerical precision)
- Algorithm box: Algorithm 1 lines 3-5 describe the update rule → SPECIFIED
- Footnote: Footnote 3 says "we clip gradients at 1.0" → SPECIFIED
- Appendix: Appendix B Table 6 shows "warmup steps = 4000" → SPECIFIED

### Does NOT count as SPECIFIED:
- "We use standard optimization" → **UNSPECIFIED** (standard according to whom? which optimizer? what hyperparameters?)
- "Following prior work [23]" → **PARTIALLY_SPECIFIED** (you must look up [23] and report what it says, or flag that the reader needs to)
- "We use Adam" → **PARTIALLY_SPECIFIED** (Adam has β₁, β₂, ε parameters — are they stated?)
- "Similar architecture to [X]" → **PARTIALLY_SPECIFIED** (similar is not identical — what differs?)
- "Standard hyperparameters" → **UNSPECIFIED**
- "Default settings" → **UNSPECIFIED** (whose defaults? PyTorch? TensorFlow? They differ.)
- Descriptions in related work of other people's methods → **NOT A SPECIFICATION OF THIS PAPER'S METHOD**
- Blog posts, tweets, or talks by the authors → **NOT PEER-REVIEWED, note as supplementary only**

---

## The UNSPECIFIED comment protocol

When you make a choice for an UNSPECIFIED item, the code comment must have three parts:

```python
# [UNSPECIFIED] {What the paper doesn't specify}
# Using: {your choice}
# Alternatives: {other reasonable choices}
```

Example:
```python
# [UNSPECIFIED] Paper does not state activation function in the feed-forward network
# Using: GELU (most common in recent transformer implementations)
# Alternatives: ReLU (original transformer), SiLU/Swish (used in LLaMA, PaLM)
self.activation = nn.GELU()
```

You must NEVER write just:
```python
self.activation = nn.GELU()  # standard choice
```
This hides the fact that the paper didn't specify it.

---

## Equation ground truth rule

Equations are more precise than prose. If the prose description and the equation conflict:

1. Implement the equation
2. Flag the discrepancy explicitly:

```python
# §3.2, Eq. 4 — loss = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
# NOTE: The prose in §3.2 says "we average over all positive pairs" but Eq. 4
# sums rather than averages. We implement Eq. 4 as written.
```

This applies even when the equation is clearly a typo. Implement the equation, flag the likely typo, and add a comment about what the authors probably meant.

---

## The "standard" trap

Papers frequently use the word "standard" without definition. Here's what to do:

| Paper says | What to do |
|-----------|------------|
| "standard transformer" | Ask: which transformer? Pre-norm? Post-norm? How many layers? Flag as UNSPECIFIED unless the paper cites a specific architecture |
| "standard augmentation" | Ask: which augmentations? Random crop size? Flip probability? Color jitter parameters? Flag as UNSPECIFIED |
| "standard preprocessing" | Ask: what tokenizer? What normalization? What sequence length? Flag as UNSPECIFIED |
| "standard evaluation" | Ask: which metric implementation? What post-processing? Flag as UNSPECIFIED |
| "we follow standard practice" | This means nothing. Flag as UNSPECIFIED. |

---

## The number precision trap

Do not silently change numbers:

- If the paper says 512, use 512 — not 256 "for simplicity" without flagging it
- If the paper says 0.9, use 0.9 — not 0.99 because "that's what people usually use"
- If the paper says 100k steps, use 100000 — not 50000 because "it should converge faster"

Any deviation from stated numbers must be flagged:
```python
# §5.2 states d_model = 512, but this walkthrough uses d_model = 64 for CPU execution
# Set d_model = 512 for actual reproduction
```

---

## The initialization trap

Weight initialization is almost never specified in papers but matters enormously. If the paper does not specify initialization:

```python
# [UNSPECIFIED] Paper does not describe weight initialization
# Using: PyTorch defaults (Kaiming uniform for Linear, uniform for Embedding)
# Alternatives: Xavier uniform (common for transformers), normal init with std=0.02
# NOTE: Initialization can significantly affect training stability and convergence
```

If the paper specifies initialization (rare but valuable), implement it exactly and cite the section.

---

## The framework translation trap

Different frameworks have different defaults that papers don't always clarify:

### Batch normalization momentum
- Paper says `momentum = 0.1`
- PyTorch BatchNorm uses `momentum = 0.1` (but its definition is inverted: `running_mean = (1 - momentum) * running_mean + momentum * batch_mean`)
- TensorFlow BatchNorm uses `momentum = 0.99` (with convention: `running_mean = momentum * running_mean + (1 - momentum) * batch_mean`)
- PyTorch `momentum=0.1` ≈ TensorFlow `momentum=0.9`
- **Always clarify which convention the paper uses**

### Dropout rate vs keep probability
- Paper says "dropout 0.1" — does this mean drop probability = 0.1 or keep probability = 0.1?
- Almost always means drop probability = 0.1 (keep = 0.9), but older papers sometimes use keep probability
- PyTorch `nn.Dropout(p=0.1)` means drop probability = 0.1

### Layer normalization epsilon
- Papers almost never specify this
- PyTorch default: 1e-5
- Common in papers: 1e-6
- Some implementations: 1e-8
- **Flag as UNSPECIFIED and note which you chose**

---

## The "we found" trap

When a paper says "we found that X works better," this is usually an empirical observation, not a derived result. Treat it as useful information but note that:
- "Better" often means "better for our specific setup/dataset/scale"
- The alternative (not-X) might work fine for different scenarios
- This is an [ASSUMPTION] not a [SPECIFIED] item when you implement it as the default

---

## Prohibited phrases in generated code

Never write any of these in code comments:
- "standard practice" (without specifying what practice)
- "as usual" (usual for whom?)
- "obviously" (if it's obvious, you don't need to say it; and it's probably not obvious)
- "typically" (without a citation)
- "it's well known that" (without a citation)
- "for simplicity" (as justification for deviating from the paper)
- "should work" (either it's specified or it isn't)

---

## The official code shortcut

If official code exists:
1. Note its URL in REPRODUCTION_NOTES.md
2. You may use it to resolve UNSPECIFIED items — but:
   - Mark them as `[FROM_OFFICIAL_CODE]`, not as `SPECIFIED`
   - The official code may differ from the paper (bug fixes, improvements, errors)
   - Link to the exact line: `github.com/author/repo/blob/main/model.py#L42`
3. Do NOT copy-paste code. Read the official code to understand the choice, then implement it yourself with a citation

---

## Self-audit questions

Before finishing any code generation, ask yourself:
1. If I removed all my `[UNSPECIFIED]` comments, would a reader think the paper specified everything? If yes, I'm probably missing flags.
2. Did I add any implementation detail from my own ML knowledge without checking if the paper says it? Flag it.
3. Would the authors of this paper agree that my code matches their description? If I'm not sure, something needs a flag.
4. Is there a single magic number anywhere without a citation or `[UNSPECIFIED]` comment? Find it and fix it.
