# Stage 3: The Ambiguity Audit

## Purpose
This is the most important anti-hallucination stage. Before generating a single line of code, systematically audit every implementation-relevant detail and classify it as SPECIFIED, PARTIALLY_SPECIFIED, or UNSPECIFIED. This stage produces the raw material for `REPRODUCTION_NOTES.md`.

## Input
- Parsed paper sections from Stage 1
- Contribution analysis from Stage 2

## Output
- `.paper2code_work/{ARXIV_ID}/ambiguity_audit.md` — complete audit with classification for every item

---

## Critical mindset

**Assume nothing. Verify everything.**

When you read "we use standard settings," ask: standard according to whom? When you read "following prior work," ask: which prior work, and which specific detail from that work? When something seems obvious, ask: is it obvious because it's specified, or because you're making an assumption?

The goal of this stage is to produce a document where a researcher can look at any implementation choice and immediately know whether the paper specified it or whether you chose it.

---

## Reasoning protocol

### The complete checklist

Go through every item below. For each one, search the paper (method section, experiments section, appendix, footnotes — all of them) and classify:

- **SPECIFIED**: The paper states this explicitly. Record the exact quote and section.
- **PARTIALLY_SPECIFIED**: The paper mentions this but is ambiguous. Record the quote and what's unclear.
- **UNSPECIFIED**: The paper does not state this at all. Record common choices and which one you'll use as default.

---

### Architecture details

| Item | What to look for | Common hiding spots |
|------|------------------|---------------------|
| Layer count / depth | "N layers," "L = 6" | Table 1, first paragraph of experiments |
| Hidden dimensions | "d_model = 512," "hidden size" | Architecture figure caption, config table in appendix |
| Number of attention heads | "h = 8 heads" | Same as hidden dims |
| Head dimension | Often d_model/h, but not always stated | Sometimes only implicit |
| Feed-forward inner dimension | "d_ff = 2048," "4x expansion" | Appendix, architecture details |
| Activation functions | "ReLU," "GELU," "SiLU/Swish" | Often unstated — do NOT assume ReLU |
| Normalization type | LayerNorm, BatchNorm, RMSNorm | Often stated, but placement (pre/post) often unstated |
| Normalization placement | Pre-norm vs post-norm | CRITICAL for transformers. Check Figure vs text — they often disagree |
| Normalization epsilon | Almost never stated | Default varies: 1e-5 (PyTorch), 1e-6 (common in papers), 1e-8 (rare) |
| Initialization scheme | Xavier, Kaiming, custom | Appendix or not stated at all |
| Dropout rate | "dropout = 0.1" | Appendix, sometimes different rates for different layers |
| Dropout placement | After attention? After FFN? On embeddings? | Rarely specified precisely |
| Residual connections | Usually obvious from figures but connection pattern matters | Check if pre-norm or post-norm |
| Bias terms | Whether linear layers have bias | Almost never stated explicitly |
| Weight tying | Whether embedding and output projection share weights | Sometimes mentioned casually |
| Vocabulary size | For NLP models | Dataset description section |
| Max sequence length | For sequence models | Experiments section, dataset preprocessing |

### Training details

| Item | What to look for | Common hiding spots |
|------|------------------|---------------------|
| Optimizer | "Adam," "SGD," "AdamW" | Method or experiments section |
| Learning rate | The peak/base learning rate | Experiments, appendix |
| Adam betas | β₁, β₂ values | Appendix if anywhere — papers often say "Adam" without specifying |
| Adam epsilon | ε value | Almost never stated |
| Weight decay | Value and where applied | Appendix. Note: often NOT applied to biases and LayerNorm |
| LR schedule | Warmup type, decay type | Appendix or experiments |
| Warmup steps | How many steps/epochs of warmup | Appendix — often unstated even when warmup is mentioned |
| Total training steps/epochs | How long to train | Experiments section |
| Batch size | Total or per-GPU? | Crucial distinction — often ambiguous |
| Gradient accumulation | How many micro-batches? | Appendix or unstated |
| Gradient clipping | Max norm value | Appendix or unstated |
| Mixed precision | FP16, BF16, or FP32? | Appendix, systems section |
| Random seed | For reproducibility | Appendix or unstated |
| Number of GPUs/hardware | Training hardware | Often in experiments or appendix |
| EMA | Exponential moving average of weights? Decay rate? | Common in diffusion models, often in appendix |

### Data details

| Item | What to look for | Common hiding spots |
|------|------------------|---------------------|
| Dataset name and version | Exact dataset identifier | Experiments section |
| Preprocessing steps | Tokenization, normalization, resizing | Appendix, experiments |
| Data augmentation | Random crop, flip, color jitter, etc. | Appendix or buried in experiments |
| Train/val/test split | Standard or custom? | Usually standard but verify |
| Sequence length / image size | Input dimensions | Experiments, may differ from model max |
| Tokenizer | Which one specifically (BPE, WordPiece, sentencepiece) | Dataset section, appendix |

### Evaluation details

| Item | What to look for | Common hiding spots |
|------|------------------|---------------------|
| Metrics used | Accuracy, BLEU, FID, etc. | Experiments section |
| Metric computation details | Which BLEU? What tokenizer? Smoothing? | Appendix or reference to external package |
| Evaluation frequency | Every N steps? Every epoch? | Appendix |
| Test-time augmentation | Averaging over crops/flips? | Experiments or appendix |
| Reported numbers | What Table 1 / main results show | Results section — record these for validation |

---

### How to search the paper

For each item in the checklist:

1. **Search the Method section** — the obvious place
2. **Search the Experiments section** — often contains "Implementation Details" or "Training Details" subsection
3. **Search the Appendix** — THE most common hiding spot for training details
4. **Search figure captions** — sometimes specify architectural details not in text
5. **Search footnotes** — authors sometimes put caveats and clarifications in footnotes
6. **Search the table captions** — hyperparameter tables often have footnotes with additional info
7. **Search for the specific term** — Ctrl+F the paper text for "learning rate," "dropout," "optimizer," etc.

---

### How to handle "following X" references

When the paper says "following [X]" or "similar to [X]" or "as in [X]":

1. Note the reference — what is paper X?
2. If X is a well-known paper (e.g., "following Vaswani et al."), describe what X specifies
3. If X is an obscure paper, flag this as requiring the reader to look up that paper
4. Record: `[PARTIALLY_SPECIFIED] Paper says "following [X]" for {component} — X specifies {detail} but reader should verify`

---

### How to handle errata and contradictions

1. **Figure vs text disagreement**: Flag it explicitly. Implement what the text/equations say (they're usually more carefully reviewed than figures) but note the figure shows something different.

2. **Abstract vs method section disagreement**: Method section wins. The abstract is a summary and may oversimplify.

3. **Equation vs prose disagreement**: Equation wins. It's more precise. Flag the discrepancy.

4. **Paper vs official code disagreement**: Note both. The official code may fix bugs or reflect a later understanding, but the paper is what was peer-reviewed. Record: `Paper says X in §Y.Z, but official code does W — we implement {your choice} because {reason}`

---

## Output format

Save to `.paper2code_work/{ARXIV_ID}/ambiguity_audit.md`:

```markdown
# Ambiguity Audit: {paper_title}

## Architecture

| Component | Status | Paper Quote | Section | Our Choice | Alternatives |
|-----------|--------|-------------|---------|------------|-------------- |
| Hidden dim | SPECIFIED | "d_model = 512" | §3.1 | 512 | — |
| Activation | UNSPECIFIED | — | — | GELU | ReLU, SiLU |
| Pre/post norm | PARTIALLY_SPECIFIED | "we use layer norm" | §3.2 | Pre-norm | Post-norm (figure suggests post-norm) |

## Training
{same table format}

## Data
{same table format}

## Evaluation
{same table format}

## Contradictions found
- {description of any inconsistencies}

## References to check
- {any "following X" references that need verification}

## Official code
- URL: {if found}
- Used to resolve: {list of items resolved by reading official code}
```

---

## Self-check before proceeding

- [ ] Every item in the checklist has been classified (not just the obvious ones)
- [ ] Every SPECIFIED item has an exact quote and section reference
- [ ] Every UNSPECIFIED item has a default choice AND alternatives listed
- [ ] You checked the appendix (not just the method section)
- [ ] You checked footnotes
- [ ] You looked for an official code repository
- [ ] Any "following [X]" references are flagged
- [ ] Contradictions between text/figures/equations are noted
