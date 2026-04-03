# Reproduction Notes: Attention Is All You Need

> This document records every implementation choice, whether it was specified by the paper,
> and what alternatives exist. If you're reproducing this paper, **read this first.**

---

## Paper

- **Title:** Attention Is All You Need
- **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- **Year:** 2017
- **ArXiv:** https://arxiv.org/abs/1706.03762
- **Official code:** https://github.com/tensorflow/tensor2tensor

---

## What this implements

The Transformer model architecture — an encoder-decoder sequence-to-sequence model that replaces recurrence entirely with multi-head self-attention. The encoder maps an input sequence of symbol representations to a sequence of continuous representations, and the decoder generates an output sequence one element at a time using attention over the encoder output and previously generated symbols. This is the "base" model configuration (§3, Table 3).

---

## Verified against

- [x] Paper equations (§3.2.1 Eq. 1 — scaled dot-product attention, §3.2.2 — multi-head attention, §3.5 — positional encoding)
- [x] Paper Table 3 — architecture hyperparameters
- [x] Paper §5.3 — optimizer configuration
- [ ] Official code (tensor2tensor) — referenced but not line-by-line verified
- [x] Well-known reimplementations: Harvard NLP "The Annotated Transformer"

---

## Unspecified choices

| Component | Our Choice | Alternatives | Paper Quote (if partial) | Section |
|-----------|-----------|--------------|--------------------------|---------|
| LayerNorm epsilon | 1e-6 | 1e-5 (PyTorch default), 1e-8 | — | — |
| LayerNorm placement | Post-norm | Pre-norm (more stable at scale) | "add & norm" in Figure 1 | §3.1, Figure 1 |
| Activation in FFN | ReLU | GELU (modern default) | "ReLU" in Eq. 2 | §3.3 |
| Bias in linear projections | True | False (used in some reimplementations) | — | — |
| Bias in output projection | True | False | — | — |
| Weight initialization | Xavier uniform | PyTorch defaults, normal(0, 0.02) | — | — |
| Embedding scale factor | √d_model | 1.0 (no scaling) | "multiply those weights by √d_model" | §3.4 |
| Weight tying | Embedding + output projection shared | Separate weights | "share the same weight matrix" | §3.4 |
| Dropout on attention weights | Yes (rate 0.1) | No dropout on weights | — | §5.4 states P_drop=0.1 but placement unspecified |
| Dropout placement | After attention + after FFN + on embeddings | Various | "Residual Dropout... applied to output of each sub-layer" | §5.4 |
| Max sequence length for PE | 5000 | 512, 10000 | — | — |

---

## Known deviations

| Deviation | Paper says | We do | Reason |
|-----------|-----------|-------|--------|
| None intentional | — | — | Implementation follows paper specification closely |

---

## Expected results

| Metric | Paper's number | Dataset | Conditions |
|--------|---------------|---------|------------|
| BLEU | 27.3 | WMT 2014 EN-DE | Table 2, base model, beam search with beam size 4 |
| BLEU | 38.1 | WMT 2014 EN-FR | Table 2, big model |
| Training cost | 12 hours on 8 P100 GPUs | — | §5.2, base model, 100k steps |

**Note:** Exact reproduction requires the WMT 2014 dataset with the same preprocessing
(BPE tokenization with ~37k tokens for EN-DE), 8 P100 GPUs, and the exact optimizer
schedule. Small deviations in BLEU (0.1-0.5) are normal.

---

## Debugging tips

1. **Loss not decreasing**: Check that the learning rate schedule follows Eq. 3 exactly (`lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`). Using a constant LR or a generic warmup schedule will give different results.

2. **NaN in attention**: Check attention masking. The mask should add `-inf` (or a very large negative number) BEFORE softmax, not multiply AFTER. All-zero rows in the mask (fully masked positions) can cause NaN after softmax — ensure at least one position is always attended to.

3. **Poor translation quality**: Verify (a) label smoothing is ε=0.1 with KL divergence, (b) the embedding scaling factor √d_model is applied, (c) weight tying between embedding and output projection is enabled.

4. **Slow convergence**: The warmup schedule is critical. With 4000 warmup steps, the LR peaks around step 4000 then decays. Starting with a high LR (no warmup) often diverges.

5. **Parameter count mismatch**: The base model should have approximately 65M parameters. If significantly different, check weight tying and whether the vocabulary size matches.

---

## Scope decisions

### Implemented
- Scaled dot-product attention (§3.2.1) — core contribution
- Multi-head attention (§3.2.2) — core contribution
- Positional encoding, sinusoidal (§3.5) — core contribution
- Encoder stack with N=6 layers (§3.1) — core contribution
- Decoder stack with N=6 layers (§3.1) — core contribution
- Embedding with √d_model scaling and weight tying (§3.4) — specified in paper
- Label-smoothed cross-entropy loss (§5.4) — needed for reproduction

### Intentionally excluded
- BPE tokenization — standard preprocessing, use sentencepiece or HuggingFace tokenizers
- Beam search decoding — inference technique, not the architectural contribution
- Multi-GPU training — engineering concern, not the paper's contribution
- WMT dataset downloading — data acquisition, provide instructions only

### Needed for full reproduction (not included)
- WMT 2014 EN-DE dataset with BPE tokenization (~37k merge operations)
- 8 P100 GPUs for the reported training speed
- Beam search with beam size 4 and length penalty α=0.6 for Table 2 results

---

## References

- Vaswani et al., 2017 — primary paper
- Ba et al., 2016 (Layer Normalization) — used in each sub-layer
- Srivastava et al., 2014 (Dropout) — applied at P_drop=0.1
- Szegedy et al., 2016 (Label smoothing) — ε_ls=0.1
- The Annotated Transformer (Harvard NLP) — reference reimplementation
