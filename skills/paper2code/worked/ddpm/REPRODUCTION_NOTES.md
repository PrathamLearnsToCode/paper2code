# Reproduction Notes: Denoising Diffusion Probabilistic Models

> This document records every implementation choice, whether it was specified by the paper,
> and what alternatives exist. If you're reproducing this paper, **read this first.**

---

## Paper

- **Title:** Denoising Diffusion Probabilistic Models
- **Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel
- **Year:** 2020
- **ArXiv:** https://arxiv.org/abs/2006.11239
- **Official code:** https://github.com/hojonathanho/diffusion

---

## What this implements

The DDPM training and sampling procedure: a generative model that learns to reverse a Markov chain of Gaussian noise additions. Given clean data x_0, the forward process gradually adds noise over T=1000 timesteps. The model (a U-Net) learns to predict the noise ε at each timestep, trained with the simplified objective L_simple = E[||ε - ε_θ(x_t, t)||²]. Sampling generates new images by iteratively denoising from pure Gaussian noise using Algorithm 2. The key contribution is the training objective and its connection to variational inference and score matching, NOT the U-Net architecture.

---

## Verified against

- [x] Paper Algorithm 1 (Training) and Algorithm 2 (Sampling) — primary specification
- [x] Paper §3.2, Eq. 11 — forward process q(x_t | x_0)
- [x] Paper §3.4, Eq. 14 — simplified objective L_simple
- [x] Paper §4 — noise schedule β_1 = 10^-4, β_T = 0.02
- [x] Official code (github.com/hojonathanho/diffusion) — referenced for U-Net details
- [x] Appendix B — model architecture details

---

## Unspecified choices

| Component | Our Choice | Alternatives | Paper Quote (if partial) | Section |
|-----------|-----------|--------------|--------------------------|---------|
| U-Net channel multipliers | [1, 2, 2, 2] | [1, 2, 4, 8], varies | "channel multipliers" | App. B, Table after §B |
| Attention resolution | 16 | 8, 16, 32 | "attention at 16×16 resolution" | App. B |
| Dropout rate | 0.0 (CIFAR-10) | 0.1 (varies by dataset) | "dropout 0.0 for CIFAR10" | App. B |
| Weight initialization | PyTorch defaults | Custom init from official code | — | — |
| EMA start step | Step 0 | Various warmup strategies | — | — |
| Gradient clipping | 1.0 | None | — | [FROM_OFFICIAL_CODE] github.com/hojonathanho/diffusion/blob/master/diffusion_tf/train_utils.py |
| Learning rate warmup | None | Linear warmup for 5000 steps | — | — |
| Number of residual blocks per level | 2 | 3, 4 | — | [FROM_OFFICIAL_CODE] |
| Group norm num_groups | 32 | 8, 16 | — | [FROM_OFFICIAL_CODE] |
| Sinusoidal timestep embedding dim | 128 | 256 | — | [FROM_OFFICIAL_CODE] |
| Loss weighting | Uniform (L_simple) | SNR-based, learned | "We set the loss weight... to 1" | §3.4 |

---

## Known deviations

| Deviation | Paper says | We do | Reason |
|-----------|-----------|-------|--------|
| Variance parameterization | §3.2 discusses learned Σ_θ but recommends fixed β_t For best results | We use fixed variance (σ²_t = β_t) | §3.4: "setting Σ_θ(x_t, t) = σ²_t I... achieved similar results" |
| Variance choice | Paper discusses both σ²_t = β_t and σ²_t = β̃_t | We use σ²_t = β_t | §3.4: "We found that both had similar results" |

---

## Expected results

| Metric | Paper's number | Dataset | Conditions |
|--------|---------------|---------|------------|
| FID | 3.17 | CIFAR-10 unconditional | Table 1, L_simple, 1000 timesteps |
| IS | 9.46 | CIFAR-10 unconditional | Table 1, L_simple |
| FID | 5.24 | LSUN Bedroom 256×256 | Table 1 |
| NLL | ≤ 3.70 | CIFAR-10 | Table 2, bits/dim |

**Note:** FID is sensitive to the number of generated samples (50k standard), the Inception
model weights used, and random seeds. Reproducing exact FID numbers requires matching all of
these. Using the `clean-fid` package is recommended for reproducible FID computation.

---

## Debugging tips

1. **Samples look like noise after training**: Check that the noise schedule is correct. β should increase linearly from 1e-4 to 0.02. If α̅_T is not close to 0, the forward process doesn't destroy enough information and sampling from N(0,1) doesn't work.

2. **Samples are blurry**: This is expected with small T or insufficient training. DDPM needs 800k+ training steps for good results on CIFAR-10. Also check that you're using EMA weights for sampling, not the training weights.

3. **Mode collapse or repetitive samples**: Check that the timestep embedding is working correctly in the U-Net. If t is not properly conditioned, the model can't distinguish noise levels.

4. **Loss doesn't decrease**: Verify the forward process q(x_t|x_0) is correct: x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε. Check the sign and the square root.

5. **NaN in sampling**: Check the variance computation at t=0. When t=0, some terms can be 0/0. The original code clips or special-cases t=0.

---

## Scope decisions

### Implemented
- Forward diffusion process q(x_t|x_0) (§3, Eq. 4) — core contribution
- Training objective L_simple (§3.4, Eq. 14, Algorithm 1) — core contribution
- Reverse sampling process (§3, Algorithm 2) — core contribution
- Linear noise schedule β_1...β_T (§4) — required for the diffusion process
- U-Net noise prediction network ε_θ (§3.3, Appendix B) — required model backbone
- EMA of model weights (§4) — specified for evaluation

### Intentionally excluded
- Learned variance Σ_θ — paper finds fixed variance works similarly (§3.4)
- Continuous-time diffusion — paper uses discrete timesteps
- DDIM sampling — different paper (Song et al., 2020)
- Classifier guidance — different paper (Dhariwal & Nichol, 2021)

### Needed for full reproduction (not included)
- CIFAR-10 dataset (torchvision can download it)
- FID computation infrastructure (50k generated samples, Inception network)
- 800k training steps on a single V100 GPU (~20 hours per DDPM paper)

---

## References

- Ho et al., 2020 — primary paper
- Sohl-Dickstein et al., 2015 — original diffusion framework
- Ronneberger et al., 2015 (U-Net) — backbone architecture adapted in this work
- Song & Ermon, 2019 — score matching perspective
- Nichol & Dhariwal, 2021 (Improved DDPM) — improved noise schedule and learned variance
