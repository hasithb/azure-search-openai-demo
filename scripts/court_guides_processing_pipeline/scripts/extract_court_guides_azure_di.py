#!/usr/bin/env python3
"""
Extract court guide PDFs using Azure Document Intelligence.

Parses PDFs via Azure DI prebuilt-layout model (markdown output),
splits into section-level documents, and outputs JSON matching the
legal-court-rag-index schema used by upload_with_embeddings.py.

The output JSON does NOT include embeddings — those are generated
by upload_with_embeddings.py at upload time.

Usage:
    python extract_court_guides_azure_di.py                       # All guides in sources/
    python extract_court_guides_azure_di.py --pdf sources/Patents-Court-Guide-Updated-February-2025.pdf
    python extract_court_guides_azure_di.py --dry-run             # Parse only, no output files
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Azure DI Configuration ──────────────────────────────────────────────────────

AZURE_DI_ENDPOINT = os.getenv(
    "AZURE_DOCUMENTINTELLIGENCE_ENDPOINT",
    "https://cog-di-gz2m4s637t5me.cognitiveservices.azure.com/",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_canonical_guides(sources_dir: str | Path) -> list[str]:
    """Download every configured guide into a clean, deterministic source set."""
    target = Path(sources_dir)
    target.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for filename, metadata in GUIDE_METADATA.items():
        request = Request(metadata["storageUrl"], headers={"User-Agent": "legal-court-rag-release/1.0"})
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        if not payload.startswith(b"%PDF"):
            raise ValueError(f"Canonical guide is not a PDF: {metadata['storageUrl']}")
        path = target / filename
        path.write_bytes(payload)
        downloaded.append(filename)
        logger.info("Captured %s (%d bytes, sha256=%s)", filename, len(payload), sha256_file(path))
    return downloaded

# ── Court Guide Metadata & Configuration ────────────────────────────────────────
#
# Each guide has:
#   category     — Short category name for the index filter
#   sourcefile   — Human-readable source label shown in citations
#   storageUrl   — Original download URL
#   updated      — ISO 8601 publication date
#   split_level  — Heading level at which to split into documents (2=H2, 3=H3)
#   sourcepage_style — How to format the sourcepage breadcrumb:
#       "numbered"  — "6. Statements of case (p. 3)"
#       "part_chapter" — "Part 1, Chapter 1 Introduction, Section Title (p. 14)"
#       "section_dot" — "Section 2. Pre-Action Protocol, 2.3 Exceptions (p. 12)"
#       "lettered"  — "A. Preliminary, A.1 The procedural framework (p. 15)"
#   annex_as_single — If True, group all sub-headings under an Annex into one doc

GUIDE_METADATA = {
    "14.341_JO_Commercial_Court_Guide_FINAL.pdf": {
        "category": "Commercial Court",
        "sourcefile": "Commercial Court Guide",
        "storageUrl": "https://www.judiciary.uk/wp-content/uploads/2023/06/14.341_JO_Commercial_Court_Guide_FINAL.pdf",
        "updated": "2023-07-01T00:00:00Z",
        "split_level": 3,
        "sourcepage_style": "lettered",
        "annex_as_single": False,
    },
    "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf": {
        "category": "King's Bench Division",
        "sourcefile": "King's Bench Division Guide",
        "storageUrl": "https://www.judiciary.uk/wp-content/uploads/2025/04/35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf",
        "updated": "2025-01-01T00:00:00Z",
        "split_level": 2,
        "sourcepage_style": "numbered",
        "annex_as_single": True,
        # Azure DI misidentifies hyperlink text as headings in this PDF
        "heading_blocklist": [
            "Commercial Court Guide - Courts and Tribunals Judiciary",
            "The Circuit Commercial Court Guide (judiciary.uk)",
        ],
    },
    "Chancery-Guide-2024-web.pdf": {
        "category": "Chancery Division",
        "sourcefile": "Chancery Guide",
        "storageUrl": "https://www.judiciary.uk/wp-content/uploads/2022/09/Chancery-Guide.pdf",
        "updated": "2024-12-01T00:00:00Z",
        "split_level": 3,
        "sourcepage_style": "part_chapter",
        "annex_as_single": False,
    },
    "Patents-Court-Guide-Updated-February-2025.pdf": {
        "category": "Patents Court",
        "sourcefile": "Patents Court Guide",
        "storageUrl": "https://www.judiciary.uk/wp-content/uploads/2025/02/Patents-Court-Guide-Updated-February-2025.pdf",
        "updated": "2025-02-01T00:00:00Z",
        "split_level": 2,
        "sourcepage_style": "numbered",
        "annex_as_single": True,
    },
    "The-Technology-and-Construction-Court-Guide.pdf": {
        "category": "Technology and Construction Court",
        "sourcefile": "Technology and Construction Court Guide",
        "storageUrl": "https://www.judiciary.uk/wp-content/uploads/2023/06/The-Technology-and-Construction-Court-Guide.pdf",
        "updated": "2022-10-01T00:00:00Z",
        "split_level": 4,
        "sourcepage_style": "section_dot",
        "annex_as_single": False,
    },
    "35.67_JO_Court-of-Appeal-Civil-Division-Guide_FINAL_WEB.pdf": {
        "category": "Court of Appeal Civil Division",
        "sourcefile": "Court of Appeal Civil Division Guide",
        "storageUrl": "https://www.judiciary.uk/wp-content/uploads/2025/06/35.67_JO_Court-of-Appeal-Civil-Division-Guide_FINAL_WEB.pdf",
        "updated": "2025-06-04T00:00:00Z",
        "split_level": 3,
        "sourcepage_style": "numbered",
        "annex_as_single": False,
    },
    "Senior-Courts-Costs-Office-Guide.pdf": {
        "category": "Senior Courts Costs Office",
        "sourcefile": "Senior Courts Costs Office Guide",
        "storageUrl": "https://www.judiciary.uk/wp-content/uploads/2022/09/Senior-Courts-Costs-Office-Guide-1.pdf",
        "updated": "2025-01-01T00:00:00Z",
        "split_level": 3,
        "sourcepage_style": "section_dot",
        "annex_as_single": False,
    },
}

# ── Data Classes ────────────────────────────────────────────────────────────────


@dataclass
class Section:
    """A section extracted from the document."""

    heading: str
    level: int  # 1=H1, 2=H2, 3=H3, 4=H4
    content_lines: list[str] = field(default_factory=list)
    page_number: int | None = None
    parent_headings: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class Document:
    """A document ready for the index (no embedding — that's added at upload time)."""

    id: str
    sourcepage: str
    parent_id: str
    category: str
    sourcefile: str
    storageUrl: str
    updated: str
    content: str
    oids: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sourcepage": self.sourcepage,
            "parent_id": self.parent_id,
            "category": self.category,
            "sourcefile": self.sourcefile,
            "storageUrl": self.storageUrl,
            "updated": self.updated,
            "content": self.content,
            "oids": self.oids,
            "groups": self.groups,
        }


# ── Utility Functions ───────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Convert heading text to a URL/id-safe slug matching existing convention."""
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = text.strip("_")
    return text.lower()


def sanitize_id(raw_id: str) -> str:
    """Sanitize ID for Azure Search (matching upload_with_embeddings.py convention)."""
    result = re.sub(r"[^a-zA-Z0-9_\-=]", "_", raw_id)
    result = re.sub(r"_{2,}", "___", result)
    return result.strip("_")


def clean_content(text: str) -> str:
    """Clean Azure DI markdown artifacts from content text."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if line.startswith("<!-- PageHeader=") or line.startswith('<!-- PageHeader="'):
            continue
        if re.match(r'<!-- PageNumber="?\d+"? -->', line):
            continue
        if line.strip().startswith("<!-- PageFooter="):
            continue
        if line.strip() == "<!-- PageBreak -->":
            continue
        if line.strip() in ("<figure>", "</figure>"):
            continue
        if re.match(r"^\s*<figure>\s*</figure>\s*$", line):
            continue
        cleaned.append(line)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


def strip_ogl_tail(text: str) -> str:
    """Remove Crown copyright / Open Government Licence boilerplate from end of content."""
    # Match common OGL markers that appear at the tail of the last section
    for marker in ["\nOGL\n", "\n\nOGL\n", "\nOGL ", "\n\nOGL "]:
        idx = text.find(marker)
        if idx > 0:
            return text[:idx].rstrip()
    # Also match "@ Crown copyright" or "© Crown copyright" standalone
    m = re.search(r"\n\s*[@©☒]\s*Crown copyright", text)
    if m and m.start() > len(text) * 0.5:
        return text[:m.start()].rstrip()
    return text


def find_page_at_position(page_positions: list[tuple[int, int]], line_idx: int) -> int | None:
    """Find the page number for a given line index using tracked PageNumber positions."""
    result = None
    for idx, page_num in page_positions:
        if idx <= line_idx:
            result = page_num
        else:
            break
    return result


def is_annex_heading(heading: str) -> bool:
    """Check if a heading is an Annex/Appendix header."""
    h = heading.strip().lower()
    return bool(re.match(r"^(annex|appendix)\s", h))


def is_document_title_repeat(heading: str, document_title: str) -> bool:
    """Check if a heading is just a repeat of the document title (Azure DI page headers)."""
    if not document_title:
        return False
    return heading.strip() == document_title.strip()


def is_toc_section(heading: str, content: str) -> bool:
    """Detect if a section is a Table of Contents.

    Distinguishes real ToC tables (3-column: number, title, page) from content
    tables (2-column: paragraph number, full text) which Azure DI produces for
    guides like TCC that have numbered/indented paragraphs.

    A heading of exactly "Contents" is NOT automatically a ToC — it could be a
    section describing the contents of a bundle (e.g. Commercial Court D.6.2).
    We require evidence in the content: either ToC-style tables with page numbers,
    or a high density of numbered chapter/section references.
    """
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return False
    table_lines = sum(1 for l in lines if l.startswith(("<t", "</t")))

    # Heading is exactly "Contents" — only flag as ToC if content looks like one
    if heading.lower().strip() == "contents":
        if _tables_look_like_toc(content):
            return True
        # Also catch non-table ToC pages: many lines ending with page numbers
        page_ref_lines = sum(
            1 for l in lines
            if re.search(r"\b\d{1,3}\s*$", l) and len(l) > 10
        )
        if page_ref_lines > 5:
            return True
        # Short content under a "Contents" heading with no ToC indicators → not a ToC
        return False

    if len(lines) > 10 and table_lines / len(lines) > 0.5:
        # High table ratio — but only flag as ToC if tables look like page-number
        # references rather than substantive content. ToC tables have short cells
        # with page numbers; content tables have long paragraph text in cells.
        if _tables_look_like_toc(content):
            return True
    # Also detect ToC by the heading containing "Contents" with table structure
    if "contents" in heading.lower() and table_lines > 5:
        return True
    return False


def _tables_look_like_toc(content: str) -> bool:
    """Check if HTML tables in the content look like a Table of Contents.

    ToC pattern: rows with a short numeric cell as the last <td>/<th> (page number).
    Content pattern: rows where the last cell has long text (actual content).
    """
    # Find all table rows and check last cell length
    rows = re.findall(r"<tr>(.*?)</tr>", content, re.DOTALL)
    if not rows:
        return False

    short_last_cell = 0
    total_checked = 0
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        if not cells:
            continue
        last_cell = cells[-1].strip()
        total_checked += 1
        # ToC page numbers are short (1-3 digits)
        if len(last_cell) <= 4 and re.match(r"^\d{0,3}$", last_cell):
            short_last_cell += 1

    if total_checked < 5:
        return False
    # If most rows end with a short numeric cell, it's a ToC
    return short_last_cell / total_checked > 0.6


def is_boilerplate_section(heading: str, content: str) -> bool:
    """Detect non-content boilerplate sections that should be excluded from the index."""
    h = heading.strip().lower()
    # OGL / license pages
    if h == "ogl":
        return True
    # Untitled cover page artifacts — short content with no heading is never useful
    if not heading.strip() and len(content.strip()) < 200:
        return True
    return False


# ── Core Extraction Logic ───────────────────────────────────────────────────────


def parse_with_azure_di(pdf_path: str) -> str:
    """Send PDF to Azure Document Intelligence and return markdown content."""
    credential = DefaultAzureCredential()
    client = DocumentIntelligenceClient(endpoint=AZURE_DI_ENDPOINT, credential=credential)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    logger.info("Sending %s (%d bytes) to Azure Document Intelligence...", pdf_path, len(pdf_bytes))
    t0 = time.time()

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(bytes_source=pdf_bytes),
        output_content_format="markdown",
    )
    result = poller.result()
    elapsed = time.time() - t0

    logger.info(
        "Azure DI completed in %.1fs — %d pages, %d chars",
        elapsed,
        len(result.pages),
        len(result.content),
    )
    return result.content


def extract_sections(markdown: str, split_level: int, annex_as_single: bool, metadata: dict | None = None) -> list[Section]:
    """Parse Azure DI markdown into sections.

    split_level: Create a new document at headings of this level or higher.
        split_level=2: Split at H1 and H2 (H3/H4 stay within parent)
        split_level=3: Split at H1, H2, and H3 (H4 stays within parent)

    annex_as_single: If True, when an Annex/Appendix heading is encountered,
        all subsequent content until the next Annex or top-level heading
        is grouped into one document.
    """
    lines = markdown.split("\n")

    # Index all PageNumber positions
    page_positions: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = re.search(r'PageNumber="?(\d+)"?', line)
        if m:
            page_positions.append((i, int(m.group(1))))

    sections: list[Section] = []
    current_section: Section | None = None
    heading_stack: list[tuple[int, str]] = []
    in_annex = False
    annex_level = 0

    # Detect document title from the first H1
    document_title = ""
    for line in lines[:50]:
        m = re.match(r"^#\s+(.+)$", line)
        if m and not line.startswith("## "):
            document_title = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1)).strip()
            break

    # Per-guide heading blocklist: Azure DI sometimes misidentifies hyperlink text
    # or other non-heading elements as markdown headings. This config-driven approach
    # lets us suppress them without hardcoding per-guide logic.
    heading_blocklist = set((metadata or {}).get("heading_blocklist", []))

    for i, line in enumerate(lines):
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", line)

        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_text = re.sub(r"\*\*(.+?)\*\*", r"\1", heading_text).strip()

            # Skip repeated document title headers
            if is_document_title_repeat(heading_text, document_title) and level >= 2:
                continue

            # Skip known bad headings from Azure DI misidentification
            if heading_text in heading_blocklist:
                if current_section is not None:
                    current_section.content_lines.append(line)
                continue

            # Handle annex grouping
            if annex_as_single and is_annex_heading(heading_text):
                in_annex = True
                annex_level = level
                if current_section is not None:
                    sections.append(current_section)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                parent_headings = list(heading_stack)
                heading_stack.append((level, heading_text))
                page_num = find_page_at_position(page_positions, i)
                current_section = Section(
                    heading=heading_text,
                    level=level,
                    page_number=page_num,
                    parent_headings=parent_headings,
                )
                continue

            if annex_as_single and in_annex:
                if level > annex_level:
                    # Sub-heading within annex — absorb as content
                    if current_section is not None:
                        current_section.content_lines.append(line)
                    continue
                else:
                    in_annex = False
                    if is_annex_heading(heading_text):
                        in_annex = True
                        annex_level = level
                        if current_section is not None:
                            sections.append(current_section)
                        while heading_stack and heading_stack[-1][0] >= level:
                            heading_stack.pop()
                        parent_headings = list(heading_stack)
                        heading_stack.append((level, heading_text))
                        page_num = find_page_at_position(page_positions, i)
                        current_section = Section(
                            heading=heading_text,
                            level=level,
                            page_number=page_num,
                            parent_headings=parent_headings,
                        )
                        continue

            # Normal split logic
            if level <= split_level:
                if current_section is not None:
                    sections.append(current_section)

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                parent_headings = list(heading_stack)
                heading_stack.append((level, heading_text))

                page_num = find_page_at_position(page_positions, i)
                current_section = Section(
                    heading=heading_text,
                    level=level,
                    page_number=page_num,
                    parent_headings=parent_headings,
                )
            else:
                # Sub-heading below split level — keep as content
                if current_section is not None:
                    current_section.content_lines.append(line)
                else:
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    parent_headings = list(heading_stack)
                    heading_stack.append((level, heading_text))
                    page_num = find_page_at_position(page_positions, i)
                    current_section = Section(
                        heading=heading_text,
                        level=level,
                        page_number=page_num,
                        parent_headings=parent_headings,
                    )
        elif current_section is not None:
            current_section.content_lines.append(line)
        else:
            if line.strip():
                current_section = Section(
                    heading="",
                    level=0,
                    page_number=find_page_at_position(page_positions, i),
                    parent_headings=[],
                )
                current_section.content_lines.append(line)

    if current_section is not None:
        sections.append(current_section)

    return sections


def build_sourcepage(section: Section, style: str, document_title: str = "") -> str:
    """Build the human-readable sourcepage string.

    Styles:
      "numbered"       — "6. Statements of case (p. 3)"
      "part_chapter"   — "Part 1, Chapter 1 Introduction, About this Guide (p. 15)"
      "section_dot"    — "Section 2. Pre-Action Protocol, 2.3 Exceptions (p. 12)"
      "lettered"       — "A. Preliminary, A.1 The procedural framework (p. 15)"

    Filters out H1 document title from breadcrumbs to keep citations concise.
    """
    parts = []

    for level, h in section.parent_headings:
        # Skip H1 document title — it bloats every sourcepage citation
        if level == 1 and document_title and h.strip() == document_title.strip():
            continue
        parts.append(h)

    if section.heading:
        parts.append(section.heading)

    sourcepage = ", ".join(parts) if parts else "Untitled"

    if section.page_number:
        sourcepage += f" (p. {section.page_number})"

    return sourcepage


def chunk_large_document(doc: Document, max_content_length: int = 12000) -> list[Document]:
    """Split a document that exceeds max_content_length into smaller chunks."""
    if len(doc.content) <= max_content_length:
        return [doc]

    paragraphs = doc.content.split("\n\n")
    chunks: list[Document] = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current_len + para_len > max_content_length and current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunk_num = len(chunks) + 1
            chunks.append(Document(
                id=f"{doc.id}_part_{chunk_num}",
                sourcepage=f"{doc.sourcepage} [Part {chunk_num}]",
                parent_id=doc.id,
                category=doc.category,
                sourcefile=doc.sourcefile,
                storageUrl=doc.storageUrl,
                updated=doc.updated,
                content=chunk_content,
            ))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len

    if current_chunk:
        chunk_num = len(chunks) + 1
        chunk_content = "\n\n".join(current_chunk)
        if len(chunks) == 0:
            return [doc]
        chunks.append(Document(
            id=f"{doc.id}_part_{chunk_num}",
            sourcepage=f"{doc.sourcepage} [Part {chunk_num}]",
            parent_id=doc.id,
            category=doc.category,
            sourcefile=doc.sourcefile,
            storageUrl=doc.storageUrl,
            updated=doc.updated,
            content=chunk_content,
        ))

    logger.info("Chunked '%s' (%d chars) into %d parts", doc.id, len(doc.content), len(chunks))
    return chunks


def merge_short_sections(sections: list[Section], min_content_length: int = 80) -> list[Section]:
    """Merge very short sections into the next section to avoid tiny fragments."""
    if not sections:
        return sections

    merged: list[Section] = []
    pending: Section | None = None

    for section in sections:
        content_text = clean_content("\n".join(section.content_lines))

        if pending is not None:
            merged_lines = pending.content_lines + ["", f"{'#' * section.level} {section.heading}", ""] + section.content_lines
            section = Section(
                heading=pending.heading,
                level=pending.level,
                content_lines=merged_lines,
                page_number=pending.page_number,
                parent_headings=pending.parent_headings,
            )
            pending = None
            content_text = clean_content("\n".join(section.content_lines))

        if len(content_text) < min_content_length and section.heading:
            pending = section
        else:
            merged.append(section)

    if pending is not None:
        if merged:
            last = merged[-1]
            last.content_lines.extend(["", f"{'#' * pending.level} {pending.heading}", ""] + pending.content_lines)
        else:
            merged.append(pending)

    return merged


def sections_to_documents(
    sections: list[Section],
    metadata: dict,
    max_content_length: int = 12000,
    document_title: str = "",
) -> list[Document]:
    """Convert extracted sections into index-ready documents."""
    style = metadata.get("sourcepage_style", "numbered")
    documents: list[Document] = []
    seen_ids: dict[str, int] = {}

    for section in sections:
        content_text = "\n".join(section.content_lines)
        cleaned_content = clean_content(content_text)

        if len(cleaned_content.strip()) < 50:
            continue

        if is_toc_section(section.heading, content_text):
            continue

        if is_boilerplate_section(section.heading, cleaned_content):
            continue

        # Strip OGL boilerplate from the tail of content (typically on last section)
        cleaned_content = strip_ogl_tail(cleaned_content)
        if len(cleaned_content.strip()) < 50:
            continue

        sourcepage = build_sourcepage(section, style, document_title)

        raw_id = slugify(section.heading) if section.heading else "untitled"
        doc_id = sanitize_id(raw_id)

        if doc_id in seen_ids:
            seen_ids[doc_id] += 1
            doc_id = f"{doc_id}_{seen_ids[doc_id]}"
        else:
            seen_ids[doc_id] = 0

        doc = Document(
            id=doc_id,
            sourcepage=sourcepage,
            parent_id="",
            category=metadata["category"],
            sourcefile=metadata["sourcefile"],
            storageUrl=metadata["storageUrl"],
            updated=metadata["updated"],
            content=cleaned_content,
        )

        chunked = chunk_large_document(doc, max_content_length)
        documents.extend(chunked)

    return documents


# ── Main Pipeline ───────────────────────────────────────────────────────────────


def process_pdf(pdf_path: str, output_dir: str, dry_run: bool = False) -> list[dict]:
    """Full pipeline: PDF -> Azure DI -> sections -> documents -> JSON."""
    pdf_name = os.path.basename(pdf_path)

    if pdf_name in GUIDE_METADATA:
        metadata = GUIDE_METADATA[pdf_name]
    else:
        logger.warning("No metadata for %s — using defaults", pdf_name)
        stem = Path(pdf_name).stem
        metadata = {
            "category": stem.replace("-", " ").replace("_", " "),
            "sourcefile": stem,
            "storageUrl": "",
            "updated": time.strftime("%Y-%m-%dT00:00:00Z"),
            "split_level": 2,
            "sourcepage_style": "numbered",
            "annex_as_single": True,
        }

    split_level = metadata.get("split_level", 2)
    annex_as_single = metadata.get("annex_as_single", True)

    logger.info("Processing: %s (category=%s, split_level=%d, annex_as_single=%s)",
                pdf_name, metadata["category"], split_level, annex_as_single)

    # Step 1: Parse with Azure DI (reuse cached markdown if available)
    md_path = os.path.join(output_dir, Path(pdf_name).stem + "_azure_di.md")
    if os.path.exists(md_path):
        logger.info("Reusing cached markdown from %s", md_path)
        with open(md_path) as f:
            markdown = f.read()
    else:
        markdown = parse_with_azure_di(pdf_path)
        if not dry_run:
            with open(md_path, "w") as f:
                f.write(markdown)
            logger.info("Saved raw markdown to %s", md_path)

    # Step 2: Extract sections
    sections = extract_sections(markdown, split_level, annex_as_single, metadata)
    logger.info("Extracted %d raw sections", len(sections))

    # Detect document title from first H1 (for filtering from breadcrumbs)
    document_title = ""
    for line in markdown.split("\n")[:50]:
        m = re.match(r"^#\s+(.+)$", line)
        if m and not line.startswith("## "):
            document_title = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group(1)).strip()
            break

    # Step 3: Merge short sections
    sections = merge_short_sections(sections, min_content_length=80)
    logger.info("After merging short sections: %d sections", len(sections))

    # Step 4: Convert to documents
    documents = sections_to_documents(sections, metadata, document_title=document_title)
    logger.info("Generated %d documents", len(documents))

    # Statistics
    content_lengths = [len(d.content) for d in documents]
    if content_lengths:
        logger.info(
            "Content lengths: min=%d, max=%d, avg=%d, total=%d chars",
            min(content_lengths),
            max(content_lengths),
            sum(content_lengths) // len(content_lengths),
            sum(content_lengths),
        )

    # Step 5: Output JSON
    doc_dicts = [d.to_dict() for d in documents]

    if not dry_run:
        output_name = Path(pdf_name).stem + "_processed.json"
        output_path = os.path.join(output_dir, output_name)
        with open(output_path, "w") as f:
            json.dump(doc_dicts, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d documents to %s", len(doc_dicts), output_path)

    return doc_dicts


def main():
    parser = argparse.ArgumentParser(description="Extract court guides using Azure Document Intelligence")
    parser.add_argument("--pdf", help="Path to a single PDF to process")
    parser.add_argument(
        "--sources-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sources"),
        help="Directory containing source PDFs (default: ../sources/)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs_azure_di"),
        help="Output directory for processed JSONs (default: ../outputs_azure_di/)",
    )
    parser.add_argument(
        "--capture-canonical",
        action="store_true",
        help="Download all configured canonical PDFs before extraction",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and report only, no file output")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    if args.pdf:
        pdfs = [os.path.abspath(args.pdf)]
    else:
        sources_dir = os.path.abspath(args.sources_dir)
        if args.capture_canonical:
            os.makedirs(sources_dir, exist_ok=True)
            for existing in Path(sources_dir).glob("*.pdf"):
                existing.unlink()
            capture_canonical_guides(sources_dir)
        pdfs = sorted(
            os.path.join(sources_dir, f)
            for f in os.listdir(sources_dir)
            if f.lower().endswith(".pdf")
        )
        logger.info("Found %d PDFs in %s", len(pdfs), sources_dir)

    total_docs = 0
    results = {}

    for pdf_path in pdfs:
        pdf_name = os.path.basename(pdf_path)
        try:
            docs = process_pdf(pdf_path, output_dir, dry_run=args.dry_run)
            total_docs += len(docs)
            results[pdf_name] = len(docs)
        except Exception:
            logger.exception("Failed to process %s", pdf_name)
            results[pdf_name] = "ERROR"

    # Summary
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    for name, count in results.items():
        logger.info("  %s: %s documents", name, count)
    logger.info("  Total: %d documents", total_docs)
    if not args.dry_run:
        logger.info("  Output: %s", output_dir)


if __name__ == "__main__":
    main()
