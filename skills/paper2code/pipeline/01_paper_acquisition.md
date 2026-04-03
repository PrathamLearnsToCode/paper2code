# Stage 1: Paper Acquisition and Parsing

## Purpose
Fetch the arxiv paper, extract its full text with mathematical notation preserved, and produce a structured markdown representation that downstream stages can consume section by section.

## Input
- `ARXIV_ID`: e.g., `2106.09685` or `2106.09685v2`

## Output
- `.paper2code_work/{ARXIV_ID}/paper_text.md` — full paper text in markdown
- `.paper2code_work/{ARXIV_ID}/paper_metadata.json` — title, authors, abstract, categories
- `.paper2code_work/{ARXIV_ID}/sections/` — individual section files
- `.paper2code_work/{ARXIV_ID}/algorithms/` — extracted algorithm boxes
- `.paper2code_work/{ARXIV_ID}/equations/` — extracted numbered equations
- `.paper2code_work/{ARXIV_ID}/tables/` — extracted tables (especially hyperparameter tables)
- `.paper2code_work/{ARXIV_ID}/footnotes.md` — all footnotes collected

---

## Reasoning protocol

### Step 1: Normalize the input

Ask yourself:
- Is this a full URL or a bare ID?
- Does it have a version suffix (v1, v2)?
- Is the ID format valid? (should be YYMM.NNNNN or older format like arch-ive/NNNNNNN)

Strip to just the ID. Keep version suffix if present.

### Step 2: Fetch the paper

Run `scripts/fetch_paper.py`. The script handles:
1. Downloading from `https://arxiv.org/pdf/{id}.pdf`
2. Extraction via `pymupdf4llm` (preferred — preserves math notation as LaTeX)
3. Fallback to `pdfplumber` if pymupdf4llm fails
4. Fallback to HTML from `https://ar5iv.labs.arxiv.org/html/{id}` if PDF extraction produces garbled output

**How to detect garbled output:** After extraction, scan the first 500 characters. If more than 20% are non-ASCII, non-LaTeX special characters, or if the text has no recognizable English words, the extraction is garbled. Fall back.

### Step 3: Verify extraction quality

Read the extracted `paper_text.md` and check:
- [ ] Can you identify the paper title?
- [ ] Is the abstract present and readable?
- [ ] Are section headings identifiable?
- [ ] Are equations present (even if in LaTeX notation)?
- [ ] Is the references section present at the end?

If any of these fail, attempt the ar5iv HTML fallback. If that also fails, inform the user that automatic extraction failed and ask them to paste the paper text directly.

### Step 4: Run structure extraction

Run `scripts/extract_structure.py` on the extracted text. This script:
- Identifies section boundaries using heading patterns (`#`, numbered headings, ALL CAPS headings)
- Extracts algorithm boxes (text between "Algorithm N" and the next section)
- Extracts numbered equations
- Extracts tables (especially those containing hyperparameters, learning rates, dimensions)
- Extracts footnotes

### Step 5: Verify all critical sections exist

**You MUST find these sections** (they may be named differently):
- Abstract
- Introduction (or "1 Introduction" or similar)
- Method/Model/Approach section (this is the core — it may be named anything)
- Experiments/Results
- Conclusion

**You MUST actively look for these** (authors hide crucial details here):
- Appendix — check for content AFTER the references. Many papers have appendices with implementation details, hyperparameter tables, ablation studies, prompts, and proofs that are essential for reproduction
- Supplementary material references — if the paper mentions "see supplementary" or "see appendix," note what is referenced
- Footnotes — often contain critical caveats about implementation choices

### Step 6: Special handling for appendices

This deserves its own step because it's that important:

1. After the References section, look for any additional content (Appendix A, B, C, etc.)
2. If appendices exist, extract them as separate section files
3. Pay special attention to:
   - Hyperparameter tables (often in Appendix A or B)
   - System Prompts
   - Architecture diagrams described in text
   - Training details (often Appendix C or D)
   - Ablation studies (contain information about what matters and what doesn't)
4. If the paper references a supplementary PDF, note this — you may need to fetch it separately from the arxiv page

### Step 7: Extract metadata

From the paper text or arxiv page, extract:
- Title
- Authors
- Year
- Arxiv categories (e.g., cs.LG, cs.CV)
- Abstract (first 500 words)

Save to `paper_metadata.json`.

---

## Fallback protocol

### If PDF download fails (403, 404, network error):
1. Try the ar5iv HTML version: `https://ar5iv.labs.arxiv.org/html/{id}`
2. If that also fails, try the abstract page: `https://arxiv.org/abs/{id}` to verify the paper exists
3. If the paper exists but can't be fetched, ask the user to download it manually and provide the path

### If extraction produces garbled text:
1. Try `pdfplumber` instead of `pymupdf4llm`
2. If still garbled, fetch ar5iv HTML
3. If HTML is also bad, try extracting just the text without math preservation
4. Last resort: ask the user to paste the paper text

### If the paper is very long (>50 pages):
1. Still extract everything — don't truncate
2. The section-level files in `sections/` allow reading parts individually
3. Focus on Method and Appendix sections for implementation details

---

## Quality checklist before proceeding to Stage 2

- [ ] `paper_text.md` exists and is readable
- [ ] `paper_metadata.json` has title and authors
- [ ] At least one section file exists in `sections/`
- [ ] You've checked for appendices
- [ ] You've checked for algorithm boxes
- [ ] Equations are present (even if in LaTeX form)
- [ ] The Method/Model section is identified and readable

If the Method section is garbled but other sections are fine, attempt to re-extract just that section from the HTML version. The Method section is the most critical — you cannot proceed without a readable version of it.
