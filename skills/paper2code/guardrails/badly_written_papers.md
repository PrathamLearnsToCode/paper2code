# Guardrail: Handling Badly Written Papers

## Reality check

Many papers are vague, inconsistent, or incomplete. This is not a criticism — writing a paper is hard, page limits are strict, and reviewers rarely check hyperparameter tables. But it means you will regularly encounter papers where the text alone is insufficient to produce a correct implementation.

This file tells you what to do when that happens. The answer is never "guess silently."

---

## Decision tree for resolving ambiguity

```
Is the detail stated explicitly in the paper?
├── YES → Use it. Cite the section. Done.
├── PARTIALLY → Follow the partial specification protocol (below)
└── NO →
    Is there official code on GitHub?
    ├── YES → Use it as ground truth. Cite the file and line number.
    │         Mark as [FROM_OFFICIAL_CODE], not [SPECIFIED].
    └── NO →
        Is there a well-known reimplementation?
        ├── YES → Reference it in REPRODUCTION_NOTES.md.
        │         DO NOT blindly copy — they made their own choices.
        │         Mark as [UNSPECIFIED] and note the external reference.
        └── NO →
            Is there a "standard" choice in the field?
            ├── YES → Use it. Mark as [UNSPECIFIED] with alternatives.
            └── NO →
                Write a stub with a detailed docstring.
                Explain what should go here and what the options are.
```

---

## Partial specification protocol

When the paper partially specifies something:

### "We use Adam"
The paper says an optimizer but not all parameters.
```python
# §4.1 — "We use the Adam optimizer"
# [PARTIALLY_SPECIFIED] Optimizer stated as Adam, but beta and epsilon values not specified
# Using: β₁=0.9, β₂=0.999, ε=1e-8 (PyTorch defaults)
# Alternatives: β₁=0.9, β₂=0.98, ε=1e-9 (Transformer default from Vaswani et al.)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                              betas=(0.9, 0.999), eps=1e-8)
```

### "We use dropout for regularization"
The paper mentions dropout but not the rate or placement.
```python
# §3.3 — "We use dropout for regularization"
# [PARTIALLY_SPECIFIED] Dropout mentioned but rate not specified
# Using: 0.1 (common default for transformer models)
# Placement: after attention and FFN (not specified — this is our choice)
self.dropout = nn.Dropout(0.1)
```

### "Similar to Transformer [Vaswani et al., 2017]"
The paper references another work for a component.
```python
# §3.1 — "We use an architecture similar to the Transformer [Vaswani et al., 2017]"
# [PARTIALLY_SPECIFIED] "Similar" implies differences exist but none are specified
# We implement the standard Transformer from Vaswani et al. unless other sections
# describe modifications. Reader should check if the authors intended differences.
```

---

## When the paper contradicts itself

### Figure vs text
Figures are often created early in the paper-writing process and may show an earlier version of the architecture or method. Text is usually more up-to-date.

**Rule: Trust the text. Flag the figure discrepancy.**
```python
# §3.2 — "We apply layer normalization before each sub-layer" (Pre-LN)
# NOTE: Figure 2 shows post-norm placement, inconsistent with this text.
# We implement pre-norm as stated in the text. If reproduction fails,
# try post-norm (change this to: x = self.norm(x + sublayer(x)))
```

### Equation vs prose
Equations are peer-reviewed more carefully and are more precise.

**Rule: Trust the equation. Flag the prose discrepancy.**

### Different numbers in different sections
The paper might say "learning rate of 1e-4" in one section and "learning rate of 3e-4" in another.

**Rule: If one is in the appendix/hyperparameter table, trust that. If both are in prose, flag both and use the one from the main experimental section (usually the one paired with the best results).**

---

## When the paper is genuinely incomplete

Some papers are missing critical details that make reproduction impossible without additional information. Here's what to do:

### Missing architecture details (for architecture papers)
If you cannot determine the model architecture from the paper:
1. Check for official code — this is the only reliable resolution
2. Check for well-known reimplementations (HuggingFace, lucidrains, etc.)
3. If neither exists, implement what you CAN determine and write stubs for the rest:

```python
def mysterious_component(x: torch.Tensor) -> torch.Tensor:
    """§3.4 — The paper describes a 'novel aggregation mechanism' but does not
    provide sufficient detail to implement it.
    
    What we know:
      - Input: tensor of shape (batch, seq_len, d_model) — from §3.3
      - Output: tensor of shape (batch, d_model) — inferred from §3.5 which uses the output
      - It "aggregates information across the sequence" — from §3.4
    
    What we don't know:
      - The specific aggregation operation (mean pooling? attention? learned query?)
      - Whether it has learnable parameters
      - Whether it uses masking
    
    To complete this implementation, check:
      1. The authors' official code (if released after this generation)
      2. Follow-up papers that reproduce this work
      3. Contact the authors
    """
    raise NotImplementedError(
        "Cannot implement: paper §3.4 does not provide sufficient architectural detail. "
        "See docstring for what is known and what is missing."
    )
```

### Missing training details (for training method papers)
If the training procedure is the contribution but is underspecified:
1. Implement the parts that ARE specified
2. For unspecified parts, use the most common baseline and flag it loudly
3. Add a section in REPRODUCTION_NOTES.md titled "Critical unspecified training details"

### Missing evaluation details
Note which metric version you're using and that it might not match:
```python
# [UNSPECIFIED] Paper reports "BLEU score" but does not specify which implementation
# Using: sacrebleu (the de facto standard for reproducible BLEU)
# NOTE: Different BLEU implementations can give significantly different numbers
# (sometimes 1-2 points) due to tokenization and smoothing differences
```

---

## Using official code as ground truth

When official code exists, use it to resolve ambiguities. But be careful:

### DO:
- Link to the exact file and line number
- Mark resolved items as `[FROM_OFFICIAL_CODE]`
- Note when the official code differs from the paper text
- Check the commit history — early commits are closer to the paper, later commits may reflect improvements

### DON'T:
- Assume official code is always right — it may have bugs
- Copy code style or non-essential patterns
- Use code from Pull Requests or branches — stick to main/master
- Assume the code matches the paper — sometimes best results come from later improvements not in the paper

### Format:
```python
# [FROM_OFFICIAL_CODE] Weight initialization: normal with std=0.02
# Source: github.com/author/repo/blob/main/model.py#L42
# Paper does not specify initialization (§3.1)
nn.init.normal_(self.weight, std=0.02)
```

---

## Using well-known reimplementations

HuggingFace, lucidrains, Andrej Karpathy, Phil Wang — some implementers are widely trusted. When using their implementations as reference:

1. **Reference in REPRODUCTION_NOTES.md**, not in inline comments (unless resolving a specific choice)
2. **Note that they may have made their own unspecified choices** — their choices are educated guesses, not specifications
3. **Do not assume they are correct** — well-known reimplementations sometimes have bugs or intentional deviations
4. **Use them for sanity checking**, not as ground truth

---

## Known erratum handling

If the paper has a published erratum, correction, or the authors have acknowledged an error:

```python
# §3.2, Eq. 4 — Original paper has a typo: denominator should be sqrt(d_k), not d_k
# ERRATUM: Acknowledged by authors in arxiv v2 (see changelog)
# We implement the corrected version
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
```

Check arxiv for paper versions — v2, v3 often contain corrections. The abstract page shows the version history.

---

## When to give up

There is a point where a paper is so underspecified that a meaningful implementation is impossible. Signs:

- The core method relies on a component described only as "proprietary" or "internal"
- The paper describes results but not the method (sometimes in position papers)
- The paper is a survey or review with no novel method
- The paper's contribution is purely theoretical with no implementable component

In these cases:
1. Generate what you CAN (data loading, evaluation metrics, utility functions)
2. Write a clear `REPRODUCTION_NOTES.md` explaining why a full implementation is not possible
3. List exactly what information would be needed to complete it
4. Do NOT generate fake implementations to make the output look complete
