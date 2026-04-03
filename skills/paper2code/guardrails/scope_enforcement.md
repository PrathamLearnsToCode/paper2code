# Guardrail: Scope Enforcement

## Purpose
Determine what to implement and what to skip. Implementing too much adds noise. Implementing too little is useless. This decision tree makes scope decisions explicit and traceable.

---

## The decision tree

For each component you're considering implementing, walk through this tree:

```
Is this component the paper's core contribution?
├── YES → ALWAYS IN SCOPE. Implement fully with citation anchoring.
│
└── NO →
    Does the paper describe this component in detail (equations, pseudocode, algorithm box)?
    ├── YES →
    │   Is this component necessary to understand the core contribution?
    │   ├── YES → IN SCOPE. Implement with citations.
    │   └── NO → OUT OF SCOPE. Note its existence, don't implement.
    │
    └── NO →
        Does the paper reference another paper for this component?
        ├── YES → REFERENCE ONLY. Import or note the dependency.
        │         Add a comment: "§X.Y — uses {component} from {reference}"
        │
        └── NO →
            Is this a standard ML component (optimizer, standard layer, etc.)?
            ├── YES → USE FRAMEWORK DEFAULT. Import from PyTorch/library.
            │         Don't reimplement. Document which version you're using.
            │
            └── NO →
                You've found something the paper assumes you know but doesn't explain.
                Flag it in REPRODUCTION_NOTES.md and use a reasonable default.
```

---

## Scope by paper type

### Type (a): New architecture paper

| Component | In scope? | Notes |
|-----------|-----------|-------|
| Model architecture | ✅ ALWAYS | This IS the contribution |
| Forward pass | ✅ ALWAYS | Demonstrates the architecture |
| Loss function | ⚠️ CONDITIONAL | Only if the paper defines a custom loss |
| Training loop | ❌ MINIMAL | Provide a single forward+backward example, not full training |
| Data pipeline | ❌ SKELETON | Dataset class with TODOs, not full implementation |
| Evaluation | ⚠️ CONDITIONAL | Metric computation code if paper reports specific metrics |
| Distributed training | ❌ NEVER | Unless the architecture paper is about distributed/parallel architectures |
| Visualization | ❌ NEVER | Unless attention visualization IS the contribution |

### Type (b): New training method or loss function

| Component | In scope? | Notes |
|-----------|-----------|-------|
| Model architecture | ❌ IMPORT | Import or reference the base architecture used |
| Loss function | ✅ ALWAYS | This IS (or is part of) the contribution |
| Training loop | ✅ ALWAYS | The training procedure IS the contribution |
| Training algorithm | ✅ ALWAYS | Algorithm box should be implemented precisely |
| Data pipeline | ⚠️ CONDITIONAL | If the training method involves special data handling |
| Evaluation | ✅ YES | Need to verify the training method produces expected results |
| LR schedule | ✅ YES | Usually critical for training methods |
| Optimizer | ✅ YES | If the paper proposes or requires a specific optimizer |

### Type (c): New inference technique

| Component | In scope? | Notes |
|-----------|-----------|-------|
| Model architecture | ❌ IMPORT | The model already exists |
| Inference algorithm | ✅ ALWAYS | This IS the contribution |
| Training | ❌ NEVER | No training happens at inference time |
| Data preprocessing | ⚠️ CONDITIONAL | Only if inference requires special preprocessing |
| Evaluation | ✅ YES | Demonstrate the inference technique produces expected output |

### Type (d): New dataset or benchmark

| Component | In scope? | Notes |
|-----------|-----------|-------|
| Dataset class | ✅ ALWAYS | This IS the contribution |
| Data loading/preprocessing | ✅ ALWAYS | How to load and prepare the data |
| Evaluation framework | ✅ ALWAYS | Metrics and evaluation protocol |
| Baseline models | ❌ REFERENCE | Note what baselines were used, don't reimplement |
| Dataset download | ⚠️ INSTRUCTIONS | Provide clear instructions, don't auto-download |

### Type (e): Theoretical with empirical validation

| Component | In scope? | Notes |
|-----------|-----------|-------|
| Experimental setup | ✅ YES | The setup that validates the theory |
| Measurement code | ✅ YES | What is measured and how |
| Model architecture | ⚠️ CONDITIONAL | Only if experiments require a specific architecture |
| Training | ⚠️ CONDITIONAL | Only the specific training setup used in experiments |
| Proof verification | ❌ NEVER | Proofs are not code |

### Type (f): System/engineering paper

| Component | In scope? | Notes |
|-----------|-----------|-------|
| System design | ✅ ALWAYS | This IS the contribution |
| Performance benchmarks | ⚠️ DOCUMENT | Document the benchmark setup, may not be reproducible |
| Low-level optimizations | ⚠️ CONDITIONAL | If they can be expressed in Python; CUDA kernels are out of scope |
| Integration code | ❌ USUALLY NOT | System papers often can't be minimally reproduced |

---

## Mode-specific scope overrides

### minimal mode (default)
- Stick strictly to the decision tree above
- Training loop: only if the contribution IS training
- Data pipeline: skeleton only
- No distributed training, no checkpointing, no logging

### full mode
- Same as minimal PLUS:
- Full training loop with all paper-specified hyperparameters
- Data pipeline with preprocessing (still needs user to provide data)
- Evaluation pipeline (not just metric functions)
- Checkpointing (minimal, for training)
- Learning rate schedule visualization

### educational mode
- Same as minimal PLUS:
- Extra comments explaining ML concepts
- Extended walkthrough notebook with theory sections
- PAPER_GUIDE.md that walks through the paper section by section

---

## Explicit exclusions (never in scope regardless of mode)

1. **Baselines and comparison methods.** The paper compares against X, Y, Z — we only implement the paper's contribution, not X, Y, Z.

2. **Ablation variants.** If the paper ablates components A, B, C — we implement the full model with A+B+C, not each variant separately.

3. **Multi-GPU/distributed training infrastructure.** Unless the paper's contribution IS the distributed strategy.

4. **Experiment tracking and logging.** No wandb, no tensorboard, no MLflow setup.

5. **Docker/container configuration.** Not relevant to understanding the paper.

6. **CI/CD, linting, formatting configuration.** This is a research scaffold, not a production project.

7. **Unit tests.** The walkthrough notebook serves as a verification layer. No pytest needed.

8. **Dataset downloading.** Provide instructions, never download automatically.

9. **Pre-trained weight downloading.** Reference where to get them, never download.

10. **Web interfaces, APIs, or demos.** Out of scope entirely.

---

## How to handle scope creep

During code generation, you may be tempted to add "just one more thing" that would make the implementation more complete. Resist this unless:

1. The "one more thing" is directly specified in the paper AND
2. It is necessary for the core contribution to make sense AND
3. Leaving it out would make the implementation misleading

If it's "nice to have" — leave it out. A focused, correct implementation of the core contribution is infinitely better than a sprawling implementation with guessed peripherals.

---

## How to document scope decisions

In `REPRODUCTION_NOTES.md`, include a section:

```markdown
## Scope decisions

### Implemented
- {Component} — reason: {core contribution / necessary for X / etc.}

### Intentionally excluded
- {Component} — reason: {baseline method / standard component imported from X / not described in paper}

### Would need for full reproduction
- {Component} — what it is, where to get it, why we didn't include it
```

This makes scope decisions transparent. A researcher reading REPRODUCTION_NOTES.md should understand exactly what this implementation covers and what it doesn't.
