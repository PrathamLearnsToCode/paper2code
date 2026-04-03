# Review — DDPM (2006.11239) Implementation

## Summary
This is a minimal, citation-anchored implementation of Denoising Diffusion
Probabilistic Models (Ho, Jain, Abbeel 2020) for unconditional image generation
on CIFAR-10.

---

## Coverage Assessment

### What the paper contributes (§1, §3)
| Contribution | Implemented? | Notes |
|---|---|---|
| Forward diffusion process q(x_t\|x_{t-1}) | ✓ | `utils.py: q_sample()` — §2, Eq. 2 |
| Reverse process p_θ(x_{t-1}\|x_t) | ✓ | `utils.py: p_sample()` — §3.2, Eq. 11 |
| L_simple training objective | ✓ | `loss.py: DDPMLoss` — §3.4, Eq. 14 |
| Linear noise schedule | ✓ | `utils.py: linear_noise_schedule()` — §4 |
| Algorithm 1 (Training) | ✓ | `train.py` — §3.4 |
| Algorithm 2 (Sampling) | ✓ | `utils.py: sample()` — §3.4 |
| U-Net architecture | ✓ | `model.py: UNet` — §3.3, Appendix B |
| Fixed variance σ²_t = β_t | ✓ | §3.4 recommendation |
| EMA of model parameters | ✓ | `utils.py: EMA` — §4 |

### What is NOT implemented
| Paper content | Reason for exclusion |
|---|---|
| L_vlb (full variational bound, Eq. 5) | Paper recommends L_simple; L_vlb used only for comparison in Table 2 |
| Learned variance σ²_t | Paper explores this but recommends fixed variance (§3.4) |
| Log-likelihood evaluation (Table 2) | Not the primary contribution; FID is the main metric |
| CelebA-HQ / LSUN experiments | Larger-scale experiments; architecture is CIFAR-10 config |
| Progressive coding / lossless compression (§4.3) | Interesting analysis but not the generative model itself |
| Interpolation experiments (§4.2) | Evaluation analysis, not implementation |

---

## Faithfulness to Paper

### ✓ Correctly implemented
- **Noise schedule**: Linear β_1 = 10^-4 to β_T = 0.02 for T = 1000 (§4)
- **Training objective**: Pure MSE between ε and ε_θ, matching Eq. 14
- **Forward process**: Closed-form q(x_t|x_0) via reparameterization (Eq. 4)
- **Reverse process**: Fixed variance, learned mean via Eq. 11
- **U-Net**: GroupNorm, self-attention at 16×16, residual blocks (§3.3, Appendix B)
- **EMA decay 0.9999** (§4)

### ⚠ Assumptions and gaps
- **Batch size 128**: [FROM_OFFICIAL_CODE] — paper does not explicitly state batch size
- **Gradient clipping 1.0**: [FROM_OFFICIAL_CODE] — not mentioned in paper
- **No learning rate schedule**: [UNSPECIFIED] — paper doesn't describe any schedule
- **U-Net channel multipliers [1,2,2,2]**: [FROM_OFFICIAL_CODE] — Appendix B only says "128 channels"
- **Data normalization [-1,1]**: [FROM_OFFICIAL_CODE] — paper doesn't specify range
- **Zero-init for output conv**: [FROM_OFFICIAL_CODE]

### ✗ Known differences from ideal
- **Model size**: Our U-Net is approximately correct for the CIFAR-10 config but
  may not match the exact parameter count (~35.7M reported in Appendix B)
- **Attention implementation**: We use a simple single-head self-attention; the
  official code has minor differences in normalization
- **No multi-GPU training**: Paper trained on TPU v3-8; we provide single-GPU code
- **800K steps**: Paper trains for 800K steps which takes ~50 GPU-hours on V100

---

## Reproducibility Expectations

### CIFAR-10 (Unconditional, 32×32)

| Metric | Paper (Table 1) | Expected from this code |
|---|---|---|
| FID | 3.17 | ~5-10 (due to potential minor arch differences) |
| IS | 9.46 | ~8-9 |

Achieving exact paper numbers requires:
1. Exact architecture match (some details only in official code)
2. Full 800K training steps
3. EMA parameters for evaluation
4. 50K generated samples for FID

---

## Code Quality Notes

1. **Citation anchoring**: Every function and class references the paper section
   where its specification originates
2. **Flag discipline**: All assumptions/unofficial-code-sourced details are marked
   with [FROM_OFFICIAL_CODE], [UNSPECIFIED], or [ASSUMPTION]
3. **Shape comments**: Tensor operations include shape annotations
4. **No unnecessary code**: No class-conditional generation, no classifier guidance,
   no DDIM sampling — these are all later work, not in this paper

---

## What to improve if extending

1. **DDIM sampling** (Song et al., 2020): Faster sampling with fewer steps
2. **Classifier-free guidance** (Ho & Salimans, 2022): Better conditional generation
3. **Cosine schedule** (Nichol & Dhariwal, 2021): Better sample quality
4. **Learned variance** (Nichol & Dhariwal, 2021): Avoids the fixed variance assumption
5. **Multi-GPU training**: Use PyTorch DDP for faster training
