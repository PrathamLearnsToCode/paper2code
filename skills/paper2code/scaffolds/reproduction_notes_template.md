# Reproduction Notes: {{PAPER_TITLE}}

> This document records every implementation choice, whether it was specified by the paper,
> and what alternatives exist. If you're reproducing this paper, **read this first.**

---

## Paper

- **Title:** {{PAPER_TITLE}}
- **Authors:** {{AUTHORS}}
- **Year:** {{YEAR}}
- **ArXiv:** https://arxiv.org/abs/{{ARXIV_ID}}
- **Official code:** {{OFFICIAL_CODE_URL or "None found"}}

---

## What this implements

{{ONE_PARAGRAPH_CONTRIBUTION_STATEMENT — from Stage 2}}

---

## Verified against

{{Describe what the implementation has been checked against:}}
- [ ] Paper equations (§X.Y, Eq. N)
- [ ] Paper Algorithm box (Algorithm 1)
- [ ] Official code (if available — link specific files/lines)
- [ ] Well-known reimplementation (name, link)
- [ ] None — implementation based solely on paper text

---

## Unspecified choices

Every implementation choice that the paper does not explicitly specify.

| Component | Our Choice | Alternatives | Paper Quote (if partial) | Section |
|-----------|-----------|--------------|--------------------------|---------|
| {{component}} | {{what we used}} | {{other options}} | {{quote or "—"}} | {{§X.Y or "—"}} |

<!-- Example rows: -->
<!-- | LayerNorm epsilon | 1e-6 | 1e-5 (PyTorch default), 1e-8 | — | — | -->
<!-- | Activation in FFN | GELU | ReLU, SiLU | — | — | -->
<!-- | Weight initialization | PyTorch defaults | Xavier, normal(0, 0.02) | — | — | -->
<!-- | Dropout placement | After attn + FFN | Various | "We use dropout" | §3.3 | -->

---

## Known deviations

Anything this implementation intentionally does differently from the paper, and why.

| Deviation | Paper says | We do | Reason |
|-----------|-----------|-------|--------|
| {{what}} | {{paper specification}} | {{our implementation}} | {{why we deviate}} |

<!-- Example: -->
<!-- | Figure vs text | Figure 1 shows post-norm | We use pre-norm | Text §3.2 says "pre-norm", Figure is likely outdated | -->

---

## Expected results

What metrics should you see if the implementation is correct? From the paper's main results.

| Metric | Paper's number | Dataset | Conditions |
|--------|---------------|---------|------------|
| {{metric}} | {{value}} | {{dataset}} | {{e.g., "Table 1, best config"}} |

**Note:** Exact reproduction of paper numbers requires matching ALL unspecified choices above,
plus having the exact training data, hardware, and random seeds. Small deviations (0.1-0.5%)
are normal even with correct implementations.

---

## Debugging tips

Common failure modes when reproducing this type of paper:

1. **{{Failure mode 1}}**: {{description and how to diagnose}}
2. **{{Failure mode 2}}**: {{description and how to diagnose}}
3. **{{Failure mode 3}}**: {{description and how to diagnose}}

<!-- Example: -->
<!-- 1. **Loss not decreasing**: Check pre-norm vs post-norm. Post-norm without warmup often diverges. -->
<!-- 2. **NaN in attention**: Masking with 0 instead of -inf can cause NaN after softmax with all-zero rows. -->
<!-- 3. **Lower accuracy than paper**: Check if label smoothing is implemented — it adds ~0.5% for transformers. -->

---

## Scope decisions

### Implemented
- {{Component}} — {{reason: core contribution / necessary for X}}

### Intentionally excluded
- {{Component}} — {{reason: baseline method / standard component / not in paper scope}}

### Needed for full reproduction (not included)
- {{Component}} — {{what it is, where to get it}}

---

## References

Papers referenced by this implementation:
- {{Citation}} — {{what was taken from this reference}}
