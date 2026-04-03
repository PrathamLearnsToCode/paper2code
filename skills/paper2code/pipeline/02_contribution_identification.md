# Stage 2: Contribution Identification

## Purpose
Before writing any code, you must understand exactly what this paper contributes and what "implementing this paper" means. This stage forces clarity. 

## Input
- Parsed paper sections from Stage 1
- `paper_metadata.json`

## Output
- `.paper2code_work/{ARXIV_ID}/contribution.md` — contains the contribution statement, paper classification, and implementation scope

---

## Reasoning protocol

### Step 1: Read the abstract and introduction 

First read: understand what the paper claims to do.
Second read: identify the **single sentence** that states the core contribution. This is usually in the abstract ("We propose...," "We introduce...," "In this work, we...") or the last paragraph of the introduction ("Our contributions are...").

Tasks:
- What is the ONE thing this paper does that didn't exist before?
- If I had to explain this paper in 10 seconds to an ML engineer, what would I say?
- What would be missing from ML if this paper didn't exist?

Write this down as a one-sentence contribution statement.

### Step 2: Classify the paper type

Determine which category best fits. This directly affects what you implement:

#### (a) New architecture
**Examples:** Transformer, ResNet, Vision Transformer, U-Net
**What "implementing the core contribution" means:** The model architecture — layer definitions, forward pass, how components connect. The architecture IS the contribution.
**Priority sections:** Model/Architecture section, Figure showing the architecture, any Algorithm boxes
**What's secondary:** Training procedure (use standard), specific dataset preprocessing

#### (b) New training method or loss function
**Examples:** DDPM, SimCLR, RLHF, DPO
**What "implementing the core contribution" means:** The training loop, the loss function, the optimization procedure. The model architecture is usually borrowed from existing work.
**Priority sections:** Training/Objective section, loss function equations, Algorithm boxes describing the training procedure
**What's secondary:** Model architecture (often standard), dataset specifics

#### (c) New inference or generation technique
**Examples:** Beam search variants, nucleus sampling, classifier-free guidance
**What "implementing the core contribution" means:** The inference algorithm. There may be no training at all.
**Priority sections:** Inference/Generation section, Algorithm boxes, any sampling or decoding procedures
**What's secondary:** Training (none or standard), model architecture (usually pre-existing)

#### (d) New dataset or benchmark
**Examples:** ImageNet, GLUE, MMLU
**What "implementing the core contribution" means:** The evaluation framework, metrics, data loading pipeline. There may be baseline models but they're not the contribution.
**Priority sections:** Dataset description, evaluation methodology, metric definitions
**What's secondary:** Baseline model architectures (reference only)

#### (e) Theoretical analysis with empirical validation
**Examples:** Lottery Ticket Hypothesis, Scaling Laws
**What "implementing the core contribution" means:** The experimental setup that validates the theory. Often involves running specific experiments with specific measurements.
**Priority sections:** Experimental setup, what is measured and how, specific procedures
**What's secondary:** The theory proofs (important to understand but not to implement)

#### (f) System or engineering paper
**Examples:** Megatron-LM, FlashAttention, vLLM
**What "implementing the core contribution" means:** The system design — often involves low-level optimizations, custom kernels, or infrastructure.
**Priority sections:** System design section, performance benchmarks, implementation details
**Note:** These papers often cannot be reproduced in a minimal implementation because the contribution IS the engineering. Flag this honestly.

### Step 3: Find the Algorithm box

Search the paper for formal algorithm descriptions (usually labeled "Algorithm 1," "Algorithm 2," etc.). These are gold:

- They are the most precise specification of the procedure
- They resolve ambiguities in prose descriptions
- They define variable names and control flow explicitly
- If the prose says one thing and the Algorithm box says another, **implement the Algorithm box** and flag the discrepancy

If there is no Algorithm box, note this — it means the implementation will rely more on equations and prose, which are more ambiguous.

### Step 4: Identify what to implement vs. reference

Make three lists:

**IMPLEMENT** (we will write this code):
- The core contribution (as identified in Step 2)
- Any component described in enough detail to implement from the paper alone

**REFERENCE** (we will import or note as dependency):
- Standard components the paper uses but didn't invent (e.g., "standard transformer encoder")
- Pre-trained models used as backbones
- Existing techniques referenced with "following [X]" or "similar to [X]"

**OUT OF SCOPE** (we will not touch):
- Baselines and comparison methods
- Ablation variants (unless the user specifically requests full mode)
- Dataset collection or annotation procedures
- Deployment-specific optimizations

### Step 5: Write the contribution statement

Write a structured contribution statement and save it to `contribution.md`:

```markdown
# Contribution Analysis

## Paper
{title} ({year})
{authors}
arxiv: {id}

## One-sentence summary
{what this paper does in one sentence}

## Paper type
{(a)-(f) classification with justification}

## Core contribution to implement
{One paragraph describing exactly what we will implement, referencing specific paper sections}

## Algorithm specification
{Whether an Algorithm box exists, and if so, which one is the primary specification}

## Implementation scope
### Will implement:
- {item 1 — with paper section reference}
- {item 2 — with paper section reference}

### Will reference (not reimplement):
- {item 1 — what it is, where to get it}

### Out of scope:
- {item 1 — why}

## Key sections for implementation
1. {Section X.Y — what it specifies}
2. {Section X.Y — what it specifies}
3. {Appendix A — what it specifies}
```

### Step 6: Self-check

Before proceeding:
- Could a competent ML engineer read my contribution statement and know exactly what to build? If not, it's too vague.
- Have I identified the paper type correctly? Read the abstract one more time.
- Have I checked the appendices for additional specification of the core contribution?
- Is there anything in my "IMPLEMENT" list that the paper doesn't actually describe in enough detail? Move it to "REFERENCE" or flag it.

---

## Common mistakes at this stage

1. **Trying to implement everything.** A paper about a new loss function doesn't need you to reimplement the backbone architecture. Import it.

2. **Misidentifying the contribution.** A paper might describe a new architecture but the actual contribution is the training recipe. Read what the authors claim, not what takes up the most pages.

3. **Ignoring paper type effects on scope.** The same checklist item (e.g., "training loop") can be in-scope or out-of-scope depending on the paper type. A training method paper MUST include the training loop. An architecture paper doesn't need more than a minimal example.

4. **Skipping the Algorithm box.** If Algorithm 1 exists and you don't read it carefully, you will implement based on noisy prose instead of the precise specification.

5. **Not checking for official code.** Search for links in the paper — often in the abstract, introduction, or a footnote. If official code exists, note its URL. It will be useful in Stage 3 for resolving ambiguities.
