---
name: paper2code
description: Converts an arxiv paper into a minimal, citation-anchored Python implementation. Trigger when user runs /paper2code with an arxiv URL or paper ID, says "implement this paper", or pastes an arxiv link asking for implementation. Flags all ambiguities honestly. Never invents implementation details not stated in the paper.
---

# paper2code — Orchestration

You are executing the paper2code skill. This file governs the high-level flow. Each stage dispatches to a detailed reasoning protocol in `pipeline/`. Do NOT skip stages. Do NOT combine stages. Execute them in order.

## Parse arguments

Extract from the user's input:
- `ARXIV_ID`: the arxiv paper ID (e.g., `2106.09685`). Strip any URL prefix.
- `MODE`: one of `minimal` (default), `full`, `educational`.
- `FRAMEWORK`: one of `pytorch` (default), `jax`, `numpy`.

If the user provided a full URL like `https://arxiv.org/abs/2106.09685`, extract the ID `2106.09685`.
If the user provided a versioned ID like `2106.09685v2`, keep the version.

## Set up working directory

Create a temporary working directory: `.paper2code_work/{ARXIV_ID}/`
This is where intermediate artifacts go. The final output goes in the current directory under `{paper_slug}/`.

## Install dependencies

Run via Bash:
```bash
pip install pymupdf4llm pdfplumber requests pyyaml
```

## Execute pipeline

### Stage 1 — Paper Acquisition and Parsing
Read and follow: `pipeline/01_paper_acquisition.md`

Run the helper script to fetch and parse the paper:
```bash
python skills/paper2code/scripts/fetch_paper.py {ARXIV_ID} .paper2code_work/{ARXIV_ID}/
```
Then run structure extraction:
```bash
python skills/paper2code/scripts/extract_structure.py .paper2code_work/{ARXIV_ID}/paper_text.md .paper2code_work/{ARXIV_ID}/
```
Verify the outputs exist before proceeding. If extraction failed, follow the fallback protocol in `pipeline/01_paper_acquisition.md`.

### Stage 2 — Contribution Identification
Read and follow: `pipeline/02_contribution_identification.md`

Read the parsed paper sections. Identify the single core contribution. Classify the paper type. Write the contribution statement. Save it to `.paper2code_work/{ARXIV_ID}/contribution.md`.

### Stage 3 — Ambiguity Audit
Read and follow: `pipeline/03_ambiguity_audit.md`

Before reading this stage, also read: `guardrails/hallucination_prevention.md`

Go through every implementation-relevant detail. Classify each as SPECIFIED, PARTIALLY_SPECIFIED, or UNSPECIFIED. Save the audit to `.paper2code_work/{ARXIV_ID}/ambiguity_audit.md`.

### Stage 4 — Code Generation
Read and follow: `pipeline/04_code_generation.md`

Before writing code, read:
- `guardrails/scope_enforcement.md` — to determine what's in and out of scope
- `guardrails/badly_written_papers.md` — if the paper is vague or inconsistent
- The relevant knowledge files in `knowledge/` for the paper's domain
- The scaffold templates in `scaffolds/` for the expected file structure

Determine the `paper_slug` from the paper title (lowercase, underscores, no special chars).
Generate all files under `{paper_slug}/` in the current working directory.

### Stage 5 — Walkthrough Notebook
Read and follow: `pipeline/05_walkthrough_notebook.md`

Generate the walkthrough notebook that connects paper sections to code with runnable sanity checks. Save to `{paper_slug}/notebooks/walkthrough.ipynb`.

## Cleanup

Remove the `.paper2code_work/` directory after successful completion.

## Final output

Print a summary:
```
✓ paper2code complete for: {paper_title}
  Output directory: {paper_slug}/
  Files generated: {list of files}
  Unspecified choices: {count} (see REPRODUCTION_NOTES.md)
  Mode: {MODE} | Framework: {FRAMEWORK}
```

## Mode-specific behavior

- **minimal** (default): Core contribution only. Training loop only if contribution involves training. No data pipeline beyond Dataset skeleton.
- **full**: Core contribution + full training loop + data pipeline + evaluation pipeline. More code, same citation rigor.
- **educational**: Same as minimal but with extra inline comments explaining ML concepts, expanded walkthrough notebook with theory sections, and a `PAPER_GUIDE.md` that walks through the paper section by section.

## Guardrails — always active

These apply at ALL stages. Read them if you haven't already:
- `guardrails/hallucination_prevention.md` — the most important file in this skill
- `guardrails/scope_enforcement.md` — what to implement and what to skip
- `guardrails/badly_written_papers.md` — what to do when the paper is unclear

## Knowledge base — consult as needed

Before implementing any of these components, read the corresponding knowledge file:
- Transformer layers, attention, positional encoding → `knowledge/transformer_components.md`
- Optimizers, LR schedules, batch size semantics → `knowledge/training_recipes.md`
- Cross-entropy, contrastive loss, diffusion loss, ELBO → `knowledge/loss_functions.md`
- Framework-specific pitfalls, notation mismatches → `knowledge/paper_to_code_mistakes.md`
