# Knowledge: Loss Functions

## Purpose
Canonical implementations of loss functions commonly referenced in papers, with attention to numerical stability, framework-specific differences, and common implementation mistakes.

---

## Cross-Entropy Loss

### Standard cross-entropy (classification)

```python
# PyTorch built-in:
loss_fn = nn.CrossEntropyLoss()
# Expects: predictions (batch, num_classes) as LOGITS (not probabilities)
#          targets (batch,) as class indices (not one-hot)
```

**Common mistake:** Passing probabilities (after softmax) instead of logits. PyTorch's `CrossEntropyLoss` applies log-softmax internally for numerical stability.

### Cross-entropy with label smoothing

Papers describe label smoothing differently from what PyTorch does:

**Paper formula (Szegedy et al., 2016):**
For C classes and smoothing ε:
- Target for correct class: `1 - ε`
- Target for each incorrect class: `ε / (C - 1)`

**PyTorch >= 1.10 built-in:**
```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```
PyTorch's implementation distributes ε uniformly: correct class gets `1 - ε + ε/C`, others get `ε/C`. This is slightly different from the paper formula (the paper distributes to incorrect classes only).

**Custom implementation matching the paper exactly:**
```python
def label_smoothed_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                                  n_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    """Label smoothing exactly as described in Szegedy et al., 2016.
    
    Args:
        logits: (batch, n_classes) — raw logits, NOT probabilities
        targets: (batch,) — class indices
        n_classes: number of classes
        smoothing: label smoothing factor ε
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, n_classes)
    
    # NLL loss for the true class
    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Smooth loss: uniform over all classes
    smooth_loss = -log_probs.mean(dim=-1)
    
    loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
    return loss.mean()
```

**The difference matters:** For large vocabularies (NLP), the difference between PyTorch's built-in and the paper formula is negligible. For small class counts (e.g., 10), it can be noticeable.

---

## Contrastive Losses

### NT-Xent / InfoNCE (SimCLR, Oord et al.)

```python
def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """Normalized Temperature-scaled Cross Entropy loss.
    
    Used in SimCLR (Chen et al., 2020).
    
    Args:
        z_i: (batch, d) — representations of view 1
        z_j: (batch, d) — representations of view 2 (positive pairs)
        temperature: τ — scaling factor
        
    Returns:
        Scalar loss
    """
    batch_size = z_i.size(0)
    
    # Normalize representations
    z_i = F.normalize(z_i, dim=-1)  # (batch, d)
    z_j = F.normalize(z_j, dim=-1)  # (batch, d)
    
    # Concatenate representations: [z_i; z_j]
    z = torch.cat([z_i, z_j], dim=0)  # (2*batch, d)
    
    # Compute similarity matrix
    sim = torch.matmul(z, z.T) / temperature  # (2*batch, 2*batch)
    
    # Mask out self-similarity (diagonal)
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~mask, float('-inf'))
    
    # Positive pairs: (i, i+batch) and (i+batch, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ], dim=0).to(z.device)  # (2*batch,)
    
    return F.cross_entropy(sim, labels)
```

**Critical: temperature matters enormously.** SimCLR uses τ=0.5 in the paper, but the appendix shows results are sensitive to this value. CLIP uses τ as a learned parameter initialized to 0.07. Always check what temperature the paper uses.

**Common mistake:** Not normalizing the representations before computing similarity. Without normalization, the cosine similarity becomes a dot product, which is unbounded and destabilizes training.

### Triplet loss

```python
def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor,
                 negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Standard triplet margin loss.
    
    Args:
        anchor: (batch, d)
        positive: (batch, d)
        negative: (batch, d)
        margin: minimum desired distance gap
    """
    pos_dist = F.pairwise_distance(anchor, positive)  # (batch,)
    neg_dist = F.pairwise_distance(anchor, negative)  # (batch,)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()
```

---

## Diffusion Losses

### DDPM loss (Ho et al., 2020)

The simplified loss from DDPM. This is L_simple from Eq. 14:

```python
def ddpm_loss(model: nn.Module, x_0: torch.Tensor,
              noise_schedule: dict) -> torch.Tensor:
    """Denoising Diffusion Probabilistic Models simplified loss.
    
    L_simple = E_{t, x_0, ε}[||ε - ε_θ(x_t, t)||²]
    
    The model predicts the noise ε that was added, not the clean signal.
    
    Args:
        model: noise prediction network ε_θ
        x_0: (batch, C, H, W) — clean images
        noise_schedule: dict with 'betas', 'alphas_cumprod', etc.
    """
    batch_size = x_0.size(0)
    
    # Sample random timesteps
    t = torch.randint(0, len(noise_schedule['betas']), (batch_size,),
                      device=x_0.device)
    
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Get noisy image: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
    alpha_cumprod_t = noise_schedule['alphas_cumprod'][t]  # (batch,)
    alpha_cumprod_t = alpha_cumprod_t.view(-1, 1, 1, 1)  # (batch, 1, 1, 1) for broadcasting
    
    x_t = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise
    
    # Predict noise
    predicted_noise = model(x_t, t)  # (batch, C, H, W)
    
    # MSE loss between true and predicted noise
    loss = F.mse_loss(predicted_noise, noise)
    return loss
```

**Noise schedule:**
```python
def linear_noise_schedule(timesteps: int, beta_start: float = 1e-4,
                           beta_end: float = 0.02) -> dict:
    """Linear variance schedule from DDPM.
    
    β_t increases linearly from β_start to β_end over T timesteps.
    """
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1, - alphas_cumprod),
    }
```

**Common mistakes with diffusion losses:**
1. Indexing: `alphas_cumprod[t]` — make sure `t` is the right shape for broadcasting
2. The DDPM paper uses 1000 timesteps with linear schedule β₁=1e-4 to β_T=0.02
3. The model predicts noise ε, not x₀ (in the simplified loss; other parameterizations exist)
4. `sqrt_one_minus_alphas_cumprod` should NOT go to exactly 0 — check boundary conditions

---

## VAE ELBO

### Evidence Lower Bound

```python
def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0) -> tuple:
    """VAE loss = Reconstruction loss + β * KL divergence.
    
    ELBO = E_q[log p(x|z)] - β * KL(q(z|x) || p(z))
    
    Args:
        recon_x: (batch, ...) — reconstructed output
        x: (batch, ...) — original input
        mu: (batch, d_latent) — mean of q(z|x)
        log_var: (batch, d_latent) — log variance of q(z|x)
        beta: weight for KL term (β=1 gives standard VAE)
    """
    # Reconstruction loss (pixel-wise)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    # Alternative: F.binary_cross_entropy for images in [0,1]
    
    # KL divergence: KL(N(μ, σ²) || N(0, 1))
    # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
```

**Common mistakes:**
1. **Reconstruction loss type:** Binary cross-entropy for images normalized to [0,1], MSE for others. Papers often don't specify.
2. **Reduction:** `sum` vs `mean` — affects the relative weight of reconstruction vs KL. If using `mean`, the KL term effectively gets more weight relative to reconstruction for high-dimensional data.
3. **β value:** β=1 is standard VAE. β-VAE uses β>1 for more disentangled representations. If the paper says "VAE loss" without specifying β, use β=1 and flag it.
4. **KL divergence sign:** The formula gives a positive KL value. The loss ADDS KL (penalizes deviation from prior). If your KL is negative, the sign is wrong.

---

## Numerical stability patterns

### Log-sum-exp trick
When computing `log(Σ exp(x_i))`:
```python
# WRONG (overflow for large values):
result = torch.log(torch.sum(torch.exp(x)))

# CORRECT:
result = torch.logsumexp(x, dim=-1)
# Internally: max_x + log(Σ exp(x_i - max_x))
```

### Softmax stability
PyTorch's `F.softmax` and `F.log_softmax` are already numerically stable (subtract max internally). But if you implement softmax manually:
```python
# WRONG:
weights = torch.exp(scores) / torch.exp(scores).sum(dim=-1, keepdim=True)

# CORRECT:
scores = scores - scores.max(dim=-1, keepdim=True).values
weights = torch.exp(scores) / torch.exp(scores).sum(dim=-1, keepdim=True)
```

### Epsilon in denominators
When dividing by a value that could be zero (e.g., normalizing):
```python
# Not just any epsilon — match the paper's convention or use a safe default
x_normalized = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
```

### Clamping log probabilities
```python
# When computing log of probabilities that might be 0:
log_probs = torch.log(probs.clamp(min=1e-8))
# Or better: compute in log space from the start using log_softmax
```
