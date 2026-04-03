#!/usr/bin/env python3
"""
Fetch and parse an arxiv paper.

Usage:
    python fetch_paper.py <arxiv_id_or_url> <output_dir>

Examples:
    python fetch_paper.py 2106.09685 ./output/
    python fetch_paper.py https://arxiv.org/abs/2106.09685 ./output/
    python fetch_paper.py 2106.09685v2 ./output/

Outputs:
    {output_dir}/paper_text.md      — full paper text in markdown
    {output_dir}/paper_metadata.json — title, authors, abstract, categories
"""

import json
import re
import sys
from pathlib import Path

import requests


def normalize_arxiv_id(input_str: str) -> str:
    """Extract arxiv ID from a URL or bare ID string.

    Handles:
        https://arxiv.org/abs/2106.09685
        https://arxiv.org/pdf/2106.09685.pdf
        http://arxiv.org/abs/2106.09685v2
        2106.09685
        2106.09685v2
        cs/0601007  (old-style IDs)
    """
    input_str = input_str.strip().rstrip("/")

    # Remove common URL prefixes
    for prefix in [
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
    ]:
        if input_str.startswith(prefix):
            input_str = input_str[len(prefix):]
            break

    # Remove .pdf suffix if present
    if input_str.endswith(".pdf"):
        input_str = input_str[:-4]

    # Validate format: YYMM.NNNNN(vN) or archive/NNNNNNN
    new_style = re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", input_str)
    old_style = re.match(r"^[a-z-]+/\d{7}(v\d+)?$", input_str)

    if not new_style and not old_style:
        print(f"WARNING: '{input_str}' may not be a valid arxiv ID.", file=sys.stderr)

    return input_str


def fetch_metadata(arxiv_id: str) -> dict:
    """Fetch paper metadata from the arxiv API."""
    # Strip version for API query
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    api_url = f"http://export.arxiv.org/api/query?id_list={base_id}"

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"WARNING: Could not fetch metadata from arxiv API: {e}", file=sys.stderr)
        return {"arxiv_id": arxiv_id, "title": "Unknown", "authors": [], "abstract": "", "categories": []}

    text = resp.text

    # Simple XML parsing — avoid heavy dependencies
    def extract_tag(tag: str, content: str) -> str:
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_all_tags(tag: str, content: str) -> list:
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        return [m.strip() for m in re.findall(pattern, content, re.DOTALL)]

    # Find the entry (skip the feed-level title)
    entry_match = re.search(r"<entry>(.*?)</entry>", text, re.DOTALL)
    if not entry_match:
        print("WARNING: No entry found in arxiv API response.", file=sys.stderr)
        return {"arxiv_id": arxiv_id, "title": "Unknown", "authors": [], "abstract": "", "categories": []}

    entry = entry_match.group(1)

    title = extract_tag("title", entry)
    title = re.sub(r"\s+", " ", title)  # collapse whitespace

    abstract = extract_tag("summary", entry)
    abstract = re.sub(r"\s+", " ", abstract)

    # Authors
    author_names = []
    for author_block in re.findall(r"<author>(.*?)</author>", entry, re.DOTALL):
        name = extract_tag("name", author_block)
        if name:
            author_names.append(name)

    # Categories
    categories = re.findall(r'<category[^>]*term="([^"]+)"', entry)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": author_names,
        "abstract": abstract,
        "categories": categories,
    }


def download_pdf(arxiv_id: str, output_path: Path) -> bool:
    """Download the PDF from arxiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"Downloading PDF from {pdf_url}...")

    try:
        resp = requests.get(pdf_url, timeout=60, stream=True)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = output_path.stat().st_size
        print(f"  Downloaded: {file_size / 1024:.0f} KB")
        return True

    except requests.RequestException as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return False


def extract_with_pymupdf4llm(pdf_path: Path) -> str | None:
    """Extract text using pymupdf4llm (preserves math notation as LaTeX)."""
    try:
        import pymupdf4llm
        print("Extracting with pymupdf4llm (math-preserving)...")
        text = pymupdf4llm.to_markdown(str(pdf_path))
        if text and len(text) > 500:
            print(f"  Extracted: {len(text)} characters")
            return text
        print("  WARNING: pymupdf4llm produced insufficient text.", file=sys.stderr)
        return None
    except ImportError:
        print("  pymupdf4llm not available.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  pymupdf4llm failed: {e}", file=sys.stderr)
        return None


def extract_with_pdfplumber(pdf_path: Path) -> str | None:
    """Extract text using pdfplumber (fallback)."""
    try:
        import pdfplumber
        print("Extracting with pdfplumber (fallback)...")
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"<!-- Page {i + 1} -->\n{text}")
        if pages:
            full_text = "\n\n".join(pages)
            print(f"  Extracted: {len(full_text)} characters from {len(pages)} pages")
            return full_text
        print("  WARNING: pdfplumber produced no text.", file=sys.stderr)
        return None
    except ImportError:
        print("  pdfplumber not available.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  pdfplumber failed: {e}", file=sys.stderr)
        return None


def fetch_ar5iv_html(arxiv_id: str) -> str | None:
    """Fetch HTML version from ar5iv (renders math as readable text)."""
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    html_url = f"https://ar5iv.labs.arxiv.org/html/{base_id}"
    print(f"Fetching HTML from {html_url}...")

    try:
        resp = requests.get(html_url, timeout=60)
        resp.raise_for_status()

        # Basic HTML to text conversion — strip tags but keep structure
        text = resp.text

        # Remove script and style blocks
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)

        # Convert headers to markdown
        for level in range(1, 7):
            text = re.sub(
                rf"<h{level}[^>]*>(.*?)</h{level}>",
                lambda m, lv=level: f"\n{'#' * lv} {m.group(1).strip()}\n",
                text,
                flags=re.DOTALL,
            )

        # Convert paragraphs to double newlines
        text = re.sub(r"<p[^>]*>", "\n\n", text)
        text = re.sub(r"</p>", "", text)

        # Convert list items
        text = re.sub(r"<li[^>]*>", "\n- ", text)

        # Preserve math elements (ar5iv uses MathML or LaTeX in alt text)
        text = re.sub(r'<math[^>]*alttext="([^"]*)"[^>]*>.*?</math>', r"$\1$", text, flags=re.DOTALL)

        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if len(text) > 500:
            print(f"  Extracted: {len(text)} characters from HTML")
            return text

        print("  WARNING: ar5iv HTML produced insufficient text.", file=sys.stderr)
        return None

    except requests.RequestException as e:
        print(f"  ar5iv fetch failed: {e}", file=sys.stderr)
        return None


def check_text_quality(text: str) -> bool:
    """Check if extracted text is reasonable quality (not garbled)."""
    if not text or len(text) < 500:
        return False

    # Check first 1000 chars for readability
    sample = text[:1000]

    # Count non-ASCII, non-whitespace, non-LaTeX special chars
    weird_chars = sum(
        1 for c in sample
        if ord(c) > 127 and c not in "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ∑∏∫∂∇√∞±≤≥≠≈∈∉⊂⊃∪∩"
    )
    weird_ratio = weird_chars / max(len(sample), 1)

    if weird_ratio > 0.2:
        print(f"  WARNING: Text quality check failed ({weird_ratio:.0%} non-standard characters)")
        return False

    # Check for recognizable English words
    common_words = {"the", "and", "of", "in", "to", "we", "is", "for", "that", "with"}
    words_lower = set(re.findall(r"\b[a-z]+\b", sample.lower()))
    found_common = words_lower & common_words

    if len(found_common) < 3:
        print("  WARNING: Text quality check failed (few recognizable English words)")
        return False

    return True


def find_official_code(arxiv_id: str, paper_text: str | None, metadata: dict) -> list[dict]:
    """Search for official code repositories linked to this paper.

    Checks two sources:
    1. The paper text itself — GitHub/GitLab URLs, "code available at" phrases
    2. The arxiv abstract page — authors sometimes add code links there

    Returns a list of dicts with keys: url, source, context
    """
    found = []
    seen_urls = set()

    def add_link(url: str, source: str, context: str = "") -> None:
        normalized = url.rstrip("/").lower()
        if normalized not in seen_urls:
            seen_urls.add(normalized)
            found.append({"url": url.rstrip("/"), "source": source, "context": context.strip()})

    # --- Source 1: Scan paper text for code URLs ---
    if paper_text:
        # Match GitHub/GitLab/Bitbucket repo URLs
        repo_pattern = r"https?://(?:github\.com|gitlab\.com|bitbucket\.org)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
        for match in re.finditer(repo_pattern, paper_text):
            url = match.group(0)
            # Grab surrounding context (up to 120 chars on each side)
            start = max(0, match.start() - 120)
            end = min(len(paper_text), match.end() + 120)
            context = paper_text[start:end].replace("\n", " ")
            add_link(url, "paper_text", context)

        # Match common phrases that precede code URLs
        code_phrases = [
            r"code\s+(?:is\s+)?(?:available|released|open[\s-]?sourced)\s+at\s+(https?://\S+)",
            r"(?:our|the)\s+code\s+(?:can be found|is hosted)\s+at\s+(https?://\S+)",
            r"implementation\s+(?:is\s+)?(?:available|released)\s+at\s+(https?://\S+)",
            r"source\s+code[:\s]+(https?://\S+)",
        ]
        for pattern in code_phrases:
            for match in re.finditer(pattern, paper_text, re.IGNORECASE):
                url = match.group(1).rstrip(".,;:)")
                add_link(url, "paper_text", match.group(0))

    # --- Source 2: Scan the arxiv abstract page ---
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    abs_url = f"https://arxiv.org/abs/{base_id}"
    try:
        resp = requests.get(abs_url, timeout=30)
        resp.raise_for_status()
        html = resp.text

        # arxiv shows official code links in the "Code" or "GitHub" badges / sidebar
        # Look for GitHub links in the abstract page HTML
        page_repo_matches = re.findall(
            r'href="(https?://(?:github\.com|gitlab\.com|bitbucket\.org)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"',
            html,
        )
        for url in page_repo_matches:
            add_link(url, "arxiv_page", "Link found on arxiv abstract page")

    except requests.RequestException as e:
        print(f"  WARNING: Could not fetch arxiv abstract page for code links: {e}", file=sys.stderr)

    return found


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <arxiv_id_or_url> <output_dir>", file=sys.stderr)
        sys.exit(1)

    raw_input = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Normalize ID
    arxiv_id = normalize_arxiv_id(raw_input)
    print(f"Arxiv ID: {arxiv_id}")

    # Step 2: Fetch metadata
    print("\n--- Fetching metadata ---")
    metadata = fetch_metadata(arxiv_id)
    metadata_path = output_dir / "paper_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Title: {metadata['title']}")
    print(f"  Authors: {', '.join(metadata['authors'][:5])}{'...' if len(metadata['authors']) > 5 else ''}")
    print(f"  Categories: {', '.join(metadata['categories'])}")

    # Step 3: Download and extract PDF
    paper_text = None
    pdf_path = output_dir / "paper.pdf"

    print("\n--- Downloading PDF ---")
    if download_pdf(arxiv_id, pdf_path):
        # Try pymupdf4llm first
        print("\n--- Extracting text ---")
        paper_text = extract_with_pymupdf4llm(pdf_path)

        # Check quality
        if paper_text and not check_text_quality(paper_text):
            print("  pymupdf4llm text quality poor, trying pdfplumber...")
            paper_text = None

        # Fallback to pdfplumber
        if paper_text is None:
            paper_text = extract_with_pdfplumber(pdf_path)

        if paper_text and not check_text_quality(paper_text):
            print("  pdfplumber text quality poor, trying ar5iv HTML...")
            paper_text = None

    # Step 4: Fallback to ar5iv HTML
    if paper_text is None:
        print("\n--- Trying ar5iv HTML fallback ---")
        paper_text = fetch_ar5iv_html(arxiv_id)

    # Step 5: Save results
    if paper_text is None:
        print("\nERROR: All extraction methods failed.", file=sys.stderr)
        print("Please download the paper manually and provide the text.", file=sys.stderr)
        sys.exit(1)

    text_path = output_dir / "paper_text.md"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"# {metadata['title']}\n\n")
        f.write(f"**Authors:** {', '.join(metadata['authors'])}\n\n")
        f.write(f"**ArXiv:** https://arxiv.org/abs/{arxiv_id}\n\n")
        f.write("---\n\n")
        f.write(paper_text)

    # Step 6: Search for official code repositories
    code_links = find_official_code(arxiv_id, paper_text, metadata)
    if code_links:
        metadata["official_code"] = code_links
        # Re-save metadata with code links
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        for link in code_links:
            print(f"  Found: {link['url']} (source: {link['source']})")
    else:
        metadata["official_code"] = []
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print("  No official code repositories found.")

    # Summary
    page_count = paper_text.count("<!-- Page")
    has_math = bool(re.search(r"[\$\\]|\\frac|\\sum|\\int|\\mathbb", paper_text))
    has_figures = bool(re.search(r"[Ff]igure\s+\d", paper_text))

    print(f"\n--- Extraction Summary ---")
    print(f"  Output: {text_path}")
    print(f"  Characters: {len(paper_text):,}")
    print(f"  Pages detected: {page_count if page_count > 0 else 'N/A (HTML extraction)'}")
    print(f"  Math preserved: {'Yes' if has_math else 'No'}")
    print(f"  Figure references found: {'Yes' if has_figures else 'No'}")
    print(f"  Metadata saved: {metadata_path}")
    print(f"  Official code links: {len(code_links)} found")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
