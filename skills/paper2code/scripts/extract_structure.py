#!/usr/bin/env python3
"""
Extract structure from a parsed paper text.

Usage:
    python extract_structure.py <paper_text.md> <output_dir>

Outputs:
    {output_dir}/sections/         — individual section files
    {output_dir}/algorithms/       — extracted algorithm boxes
    {output_dir}/equations/        — extracted numbered equations
    {output_dir}/tables/           — extracted tables
    {output_dir}/footnotes.md      — all footnotes collected
"""

import re
import sys
from pathlib import Path


def identify_sections(text: str) -> list[dict]:
    """Identify section boundaries using heading patterns.

    Detects:
      - Markdown headings (# , ## , ### )
      - Numbered headings (1. Introduction, 2.1 Related Work)
      - ALL CAPS headings (INTRODUCTION, RELATED WORK)
    """
    lines = text.split("\n")
    sections = []
    current_section = None
    current_lines = []

    # Patterns for section headings
    md_heading = re.compile(r"^(#{1,4})\s+(.+)$")
    numbered_heading = re.compile(
        r"^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s:,\-]+)$"
    )
    allcaps_heading = re.compile(r"^([A-Z][A-Z\s]{4,})$")

    def save_current():
        if current_section and current_lines:
            sections.append({
                "title": current_section,
                "content": "\n".join(current_lines).strip(),
            })

    for line in lines:
        heading = None

        # Check markdown heading
        m = md_heading.match(line)
        if m:
            heading = m.group(2).strip()

        # Check numbered heading
        if not heading:
            m = numbered_heading.match(line.strip())
            if m:
                heading = f"{m.group(1)} {m.group(2).strip()}"

        # Check ALL CAPS heading (only for longer titles to avoid false positives)
        if not heading:
            m = allcaps_heading.match(line.strip())
            if m and len(m.group(1).strip()) > 5:
                heading = m.group(1).strip().title()

        if heading:
            save_current()
            current_section = heading
            current_lines = []
        else:
            current_lines.append(line)

    save_current()
    return sections


def extract_algorithms(text: str) -> list[dict]:
    """Extract algorithm boxes from the paper.

    Looks for patterns like:
      Algorithm 1: Name
      ...algorithm body...
      (ends at next section heading or next Algorithm block)
    """
    algorithms = []

    # Pattern: "Algorithm N" possibly followed by colon and name
    pattern = re.compile(
        r"(Algorithm\s+\d+[:\.]?\s*[^\n]*)\n(.*?)(?=Algorithm\s+\d+[:\.]|^#{1,4}\s|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    for match in pattern.finditer(text):
        title = match.group(1).strip()
        body = match.group(2).strip()
        if body:
            algorithms.append({
                "title": title,
                "content": body,
            })

    return algorithms


def extract_equations(text: str) -> list[dict]:
    """Extract numbered equations.

    Looks for:
      - LaTeX equation environments: \\begin{equation}...\\end{equation}
      - Display math with numbering: $$ ... $$ (N)
      - Inline equation references: (1), (2), Eq. 1, Equation 1
      - Markdown math blocks
    """
    equations = []

    # LaTeX equation environments
    latex_eq = re.compile(
        r"\\begin\{(?:equation|align|gather)\*?\}(.*?)\\end\{(?:equation|align|gather)\*?\}",
        re.DOTALL,
    )
    for i, match in enumerate(latex_eq.finditer(text)):
        equations.append({
            "number": i + 1,
            "content": match.group(1).strip(),
            "raw": match.group(0),
        })

    # Display math with parenthesized numbers: $$ formula $$ (N)
    display_math = re.compile(r"\$\$(.*?)\$\$\s*\((\d+)\)", re.DOTALL)
    for match in display_math.finditer(text):
        equations.append({
            "number": int(match.group(2)),
            "content": match.group(1).strip(),
            "raw": match.group(0),
        })

    # Lines that look like equations with numbers at the end: formula (N)
    numbered_line = re.compile(r"^(.+?)\s+\((\d+)\)\s*$", re.MULTILINE)
    for match in numbered_line.finditer(text):
        content = match.group(1).strip()
        num = int(match.group(2))
        # Only include if it looks like an equation (has math-like characters)
        if any(c in content for c in "=+∑∏∫_^{}\\√∞"):
            if not any(eq["number"] == num for eq in equations):
                equations.append({
                    "number": num,
                    "content": content,
                    "raw": match.group(0),
                })

    # Sort by equation number
    equations.sort(key=lambda e: e["number"])
    return equations


def extract_tables(text: str) -> list[dict]:
    """Extract tables from the paper text.

    Looks for:
      - Markdown tables (pipes)
      - Table captions (Table N: ...)
      - Structured text that looks like a table
    """
    tables = []

    # Find table captions and associated content
    table_caption = re.compile(
        r"(Table\s+\d+[:\.]?\s*[^\n]*)\n(.*?)(?=Table\s+\d+[:\.]|^#{1,4}\s|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    for match in table_caption.finditer(text):
        caption = match.group(1).strip()
        body = match.group(2).strip()

        # Check if the body contains table-like content (pipes, tabs, or aligned columns)
        if "|" in body or "\t" in body or re.search(r"\s{3,}", body):
            tables.append({
                "caption": caption,
                "content": body[:2000],  # limit size
            })

    # Also find markdown tables without explicit captions
    md_table = re.compile(r"(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n)*)", re.MULTILINE)
    for match in md_table.finditer(text):
        table_text = match.group(1).strip()
        if not any(table_text in t["content"] for t in tables):
            tables.append({
                "caption": "Untitled table",
                "content": table_text,
            })

    return tables


def extract_footnotes(text: str) -> list[dict]:
    """Extract footnotes from the paper."""
    footnotes = []

    # Pattern: footnote markers like ¹, ², ³ or [1], [2] at start of line
    fn_pattern = re.compile(
        r"(?:^|\n)[\s]*(?:[\u00b9\u00b2\u00b3\u2074-\u2079]|\[(\d+)\]|(\d+)\.)[\s]+(.+?)(?=\n[\s]*(?:[\u00b9\u00b2\u00b3\u2074-\u2079]|\[\d+\]|\d+\.)\s|\n\n|\Z)",
        re.DOTALL,
    )

    for match in fn_pattern.finditer(text):
        content = match.group(0).strip()
        if len(content) > 10:  # skip very short matches that are likely false positives
            footnotes.append(content)

    # Also look for explicit footnote sections
    fn_section = re.compile(
        r"(?:footnote|note)s?\s*:?\s*\n(.*?)(?=\n#{1,4}\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    for match in fn_section.finditer(text):
        content = match.group(1).strip()
        if content and content not in footnotes:
            footnotes.append(content)

    return [{"content": fn} for fn in footnotes]


def save_list_to_dir(items: list[dict], output_dir: Path, name_key: str = "title"):
    """Save a list of extracted items as individual files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(items):
        # Create a clean filename
        name = item.get(name_key, item.get("caption", f"item_{i+1}"))
        name = str(name)
        clean_name = re.sub(r"[^\w\s-]", "", name)
        clean_name = re.sub(r"\s+", "_", clean_name).strip("_").lower()
        if not clean_name:
            clean_name = f"item_{i+1}"
        clean_name = clean_name[:80]  # limit filename length

        filepath = output_dir / f"{i+1:02d}_{clean_name}.md"

        content = f"# {name}\n\n{item.get('content', item.get('raw', ''))}\n"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <paper_text.md> <output_dir>", file=sys.stderr)
        sys.exit(1)

    paper_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not paper_path.exists():
        print(f"ERROR: {paper_path} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting structure from: {paper_path}")
    text = paper_path.read_text(encoding="utf-8")
    print(f"  Total characters: {len(text):,}")

    # Extract sections
    print("\n--- Extracting sections ---")
    sections = identify_sections(text)
    if sections:
        save_list_to_dir(sections, output_dir / "sections")
        print(f"  Found {len(sections)} sections:")
        for s in sections:
            print(f"    - {s['title']} ({len(s['content'])} chars)")
    else:
        print("  WARNING: No sections detected. The paper text may not have clear headings.")
        # Save the entire text as a single section
        (output_dir / "sections").mkdir(parents=True, exist_ok=True)
        (output_dir / "sections" / "01_full_text.md").write_text(text, encoding="utf-8")

    # Extract algorithms
    print("\n--- Extracting algorithm boxes ---")
    algorithms = extract_algorithms(text)
    if algorithms:
        save_list_to_dir(algorithms, output_dir / "algorithms")
        print(f"  Found {len(algorithms)} algorithms:")
        for a in algorithms:
            print(f"    - {a['title']}")
    else:
        print("  No algorithm boxes found.")

    # Extract equations
    print("\n--- Extracting equations ---")
    equations = extract_equations(text)
    if equations:
        save_list_to_dir(equations, output_dir / "equations", name_key="number")
        print(f"  Found {len(equations)} numbered equations")
    else:
        print("  No numbered equations found (may be inline or in non-standard format).")

    # Extract tables
    print("\n--- Extracting tables ---")
    tables = extract_tables(text)
    if tables:
        save_list_to_dir(tables, output_dir / "tables", name_key="caption")
        print(f"  Found {len(tables)} tables:")
        for t in tables:
            print(f"    - {t['caption']}")
    else:
        print("  No tables found.")

    # Extract footnotes
    print("\n--- Extracting footnotes ---")
    footnotes = extract_footnotes(text)
    footnotes_path = output_dir / "footnotes.md"
    if footnotes:
        with open(footnotes_path, "w", encoding="utf-8") as f:
            f.write("# Footnotes\n\n")
            for i, fn in enumerate(footnotes):
                f.write(f"## Footnote {i + 1}\n\n{fn['content']}\n\n---\n\n")
        print(f"  Found {len(footnotes)} footnotes")
    else:
        with open(footnotes_path, "w", encoding="utf-8") as f:
            f.write("# Footnotes\n\nNo footnotes extracted.\n")
        print("  No footnotes found.")

    # Summary
    print(f"\n--- Extraction Summary ---")
    print(f"  Sections:   {len(sections)}")
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Equations:  {len(equations)}")
    print(f"  Tables:     {len(tables)}")
    print(f"  Footnotes:  {len(footnotes)}")
    print(f"  Output dir: {output_dir}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
