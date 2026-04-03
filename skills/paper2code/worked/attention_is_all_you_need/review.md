# Review: Attention Is All You Need — paper2code Output

**Paper:** Attention Is All You Need (Vaswani et al., 2017)
**Command:** `/paper2code https://arxiv.org/abs/1706.03762`
**Reviewer:** Human review of skill output

---

## What the skill got right

### Architecture implementation (✓ Correct)
- Multi-head attention follows §3.2.1 and §3.2.2 precisely
- Scaled dot-product attention implements Eq. 1 correctly: softmax(QK^T/√d_k)V
- Scale factor uses d_k (= d_model/n_heads = 64), not d_model — this is correct and a common mistake to catch
- Positional encoding matches §3.5 exactly — sinusoidal with correct div_term formula
- Encoder/decoder stacks are N=6 layers as specified in Table 3
- Post-norm (LayerNorm after residual addition) matches Figure 1

### Citation anchoring (✓ Thorough)
- Every class and function references the correct paper section
- Equation numbers match the paper's numbering
- Shape comments are present on every tensor operation
- Variable names match paper notation (Q, K, V, d_k, d_model)

### Ambiguity flagging (✓ Honest)
- LayerNorm epsilon flagged as [UNSPECIFIED] — correct, paper doesn't mention it
- Weight initialization flagged as [UNSPECIFIED] — correct
- Bias in linear layers flagged as [UNSPECIFIED] — correct
- Max sequence length for PE flagged as [UNSPECIFIED] — correct

### Config file (✓ Complete)
- Every parameter has a paper citation or [UNSPECIFIED] flag
- β₁, β₂, ε for Adam correctly sourced from §5.3
- Label smoothing ε=0.1 from §5.4
- Warmup steps 4000 from §5.3

---

## What it correctly flagged as unspecified

1. **LayerNorm epsilon** — the paper never states this. Using 1e-6 is a reasonable choice. ✓
2. **Bias terms in linear projections** — never discussed in the paper. ✓
3. **Weight initialization** — the paper does not describe initialization. Xavier uniform is a reasonable default. ✓
4. **Max sequence length for positional encoding** — paper doesn't specify an upper bound beyond the training data. ✓
5. **Gradient clipping** — paper doesn't mention it. ✓
6. **Dropout on attention weights specifically** — paper says P_drop=0.1 dropout generally but doesn't enumerate every placement. ✓

---

## Potential issues and discussion points

### 1. Pre-norm vs Post-norm ambiguity
The implementation uses post-norm (LayerNorm after residual addition), matching Figure 1. However, some researchers argue the paper describes pre-norm in the text. The implementation correctly notes this in REPRODUCTION_NOTES.md but could be more explicit about the ambiguity in the code comments.

**Verdict:** Acceptable. The choice is reasonable and the ambiguity is documented.

### 2. Dropout placement
The paper says "Residual Dropout" in §5.4 and describes it as being applied "to the output of each sub-layer, before it is added to the sub-layer input and normalized." The implementation applies dropout after attention and after FFN, plus on embeddings. The exact dropout placement across attention weights, attention output, FFN internal, FFN output, and embeddings is somewhat ambiguous.

**Verdict:** The implementation is reasonable but could have flagged the attention weight dropout more specifically as a partial specification.

### 3. Weight tying
The implementation correctly implements three-way weight tying (source embedding, target embedding, output projection) as described in §3.4. This is a detail many implementations miss — good catch.

### 4. Embedding scaling
The √d_model scaling on embeddings (§3.4) is correctly implemented. This is another commonly missed detail.

### 5. Training scope
Correctly identified that this is an architecture paper (type a) and appropriately limited the training code to a minimal example rather than a full pipeline. Good scope decision.

---

## What could be improved

1. **The walkthrough notebook** — would benefit from more explicit sanity checks on intermediate tensor shapes, especially showing that attention weights sum to 1 after softmax.

2. **BLEU implementation note** — could mention that the original paper likely used multi-bleu.perl, which gives slightly different numbers than sacrebleu.

3. **Beam search** — the paper's reported numbers use beam search with beam_size=4 and length_penalty=0.6 (§6.1). A note about this in the training script would be helpful for anyone trying to reproduce Table 2.

---

## Overall assessment

**Quality: High.** The implementation is citation-anchored, honest about uncertainty, and correctly scoped. A researcher could use this as a starting point for reproduction with high confidence that the architecture matches the paper. The REPRODUCTION_NOTES.md is genuinely useful — it saves someone the work of auditing the paper for unspecified details.

**Coverage: Core contribution fully implemented.** All architectural components from §3 are present and correct. Training infrastructure is appropriately minimal (this is an architecture paper, not a training method paper).

**Honesty: Good.** Unspecified choices are flagged consistently. No silent assumptions detected. The review found no cases where the implementation confidently invented a detail that the paper doesn't specify.
