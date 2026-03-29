"""
HTML Section Extractor for CPR Pages.

Implements a 3-tier extraction strategy for UK Civil Procedure Rules pages:

Tier 1 (65% of pages): <a id="X.X"> inside h2/h3/h4 headings
  - Extract the `id` attribute directly from anchor tags inside headings
  - Covers: dotted_rule (1.1), rule_prefixed (rule44.1), named_section, etc.

Tier 2 (28% of pages): Heading text contains parseable rule number / section title
  - Sub-tier 2a: <h3>35.1</h3> — heading TEXT matches rule number pattern
  - Sub-tier 2b: <h2>1. General</h2> — numbered but not CPR rule format
  - Sub-tier 2c: <h2>Scope</h2> — section title headings only

Tier 3 (7% of pages): Simple pages with no meaningful sub-sections
  - Single block of prose (PD 1A, Welsh docs, Part 4, etc.)
  - subsection_id = document title, subsections = [document title]

Usage:
    from html_section_extractor import extract_sections, SectionInfo
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# Anchor ID classification helpers
# ---------------------------------------------------------------------------

DOTTED_RULE_RE = re.compile(r"^\d+[A-Z]?\.\d+")          # 1.1  2A.3  44.15
RULE_PREFIXED_RE = re.compile(r"^rule(\d+\.\d+)")          # rule44.1 -> 44.1
PARA_PREFIXED_RE = re.compile(r"^para(\d+\.\d+)")          # para1.1 -> 1.1
SECTION_ROMAN_RE = re.compile(r"^section([IVX]+)$", re.I)  # sectionI -> I
SINGLE_NUM_RE = re.compile(r"^\d+$")                        # 1  2  3
# Named section: at least 3 chars starting with uppercase letter (Annex, Schedule, Appendix)
NAMED_SECTION_RE = re.compile(r"^(Annex|Appendix|Schedule|Table)([-_\s]?[A-Z0-9]+)?$", re.I)
ROMAN_NUMERAL_RE = re.compile(r"^[IVX]+$")                  # I  II  III  XIV
ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z ]+$")                 # PART I  SECTION II
LEGACY_AUTOGEN_RE = re.compile(r"^(ID[A-Z0-9]+|id\d+)$")   # IDA0JICC, id3585867
FOOTNOTE_RE = re.compile(r"^fn\d+$")                        # fn1, fn2
TEXT_ANCHOR_RE = re.compile(r"^text\d+$")                   # text1, text4

# Heading text that looks like a CPR rule number
HEADING_TEXT_RULE_RE = re.compile(r"^\d+[A-Z]?\.\d+")      # 35.1

# Numbered paragraph heading (e.g. "1. General", "2. Purpose")
NUMBERED_PARA_RE = re.compile(r"^\d+\.\s+\S+")

# Section patterns in heading text: "SECTION I", "Section II – Writs"
HEADING_SECTION_RE = re.compile(r"^(SECTION\s+[IVX]+|Section\s+[IVX]+)", re.I)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SectionInfo:
    """Describes the section structure extracted from a single HTML heading."""
    anchor_id: str                  # Canonical ID (e.g. "44.1", "I", "Annex")
    heading_text: str               # The visible text of the heading
    heading_tag: str                # "h2", "h3", "h4"
    source: str                     # "anchor_id" | "heading_text" | "section_title"
    tier: int                       # 1, 2, or 3


@dataclass
class PageSections:
    """All section information extracted from one CPR page."""
    tier: int                       # 1 / 2 / 3
    tier_reason: str                # Human-readable reason
    sections: List[SectionInfo] = field(default_factory=list)
    primary_section: Optional[str] = None   # Best candidate for subsection_id
    all_section_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.all_section_ids = [s.anchor_id for s in self.sections]
        if self.sections:
            self.primary_section = self.sections[0].anchor_id


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _normalise_anchor_id(raw_id: str) -> Optional[str]:
    """
    Normalise a raw HTML anchor id to a canonical CPR reference.
    Returns None for noise / legacy auto-generated IDs.
    """
    if not raw_id:
        return None

    # Skip noise
    if LEGACY_AUTOGEN_RE.match(raw_id):
        return None
    if FOOTNOTE_RE.match(raw_id):
        return None
    if TEXT_ANCHOR_RE.match(raw_id):
        return None
    if raw_id.lower() in ("back-to-top", "contents", "top"):
        return None

    # Normalise rule-prefixed: rule44.1 -> 44.1
    m = RULE_PREFIXED_RE.match(raw_id)
    if m:
        return m.group(1)

    # Normalise para-prefixed: para1.1 -> 1.1
    m = PARA_PREFIXED_RE.match(raw_id)
    if m:
        return m.group(1)

    # Normalise section roman: sectionI -> I
    m = SECTION_ROMAN_RE.match(raw_id)
    if m:
        return m.group(1).upper()

    # Clean dotted rule — keep as-is
    if DOTTED_RULE_RE.match(raw_id):
        return raw_id

    # Single number
    if SINGLE_NUM_RE.match(raw_id):
        return raw_id

    # Bare Roman numerals: I, II, III, IV, XIV …
    if ROMAN_NUMERAL_RE.match(raw_id):
        return raw_id

    # ALL-CAPS section titles: "PART I", "SECTION II"
    if ALL_CAPS_RE.match(raw_id):
        return raw_id

    # Named section: strict allow-list only (avoid broad heading labels like "General")
    if NAMED_SECTION_RE.match(raw_id):
        return raw_id

    # Unknown / unrecognised pattern — discard (do not pass through)
    return None


def _extract_tier1_anchors(content_tag: Tag) -> List[SectionInfo]:
    """
    Tier 1: Find <a id="X"> tags that live INSIDE heading tags.
    Returns a list of SectionInfo objects in document order.
    """
    results = []
    seen_ids = set()

    for heading_tag in content_tag.find_all(["h2", "h3", "h4"]):
        for anchor in heading_tag.find_all("a", id=True):
            raw_id = anchor["id"].strip()
            canonical = _normalise_anchor_id(raw_id)
            if canonical and canonical not in seen_ids:
                seen_ids.add(canonical)
                results.append(SectionInfo(
                    anchor_id=canonical,
                    heading_text=heading_tag.get_text(strip=True),
                    heading_tag=heading_tag.name,
                    source="anchor_id",
                    tier=1,
                ))

    return results


def _extract_tier2_heading_text(content_tag: Tag) -> List[SectionInfo]:
    """
    Tier 2: No <a id> in headings, but heading TEXT contains rule numbers.
    Two sub-strategies:
    - 2a: Heading text IS a CPR rule number: "35.1", "41.3A"
    - 2b: Heading text starts with numbered paragraph: "1. General"
    """
    results = []
    seen = set()

    for tag_name in ["h2", "h3", "h4"]:
        for h in content_tag.find_all(tag_name):
            # Skip headings that already have anchor IDs (Tier 1 would handle them)
            if h.find("a", id=True):
                continue

            text = h.get_text(strip=True)
            if not text:
                continue

            canonical = None
            source = None

            # 2a: pure rule number
            if HEADING_TEXT_RULE_RE.match(text):
                # Extract just the rule part (may have trailing text)
                canonical = HEADING_TEXT_RULE_RE.match(text).group(0)
                source = "heading_text"

            # Section roman in heading: "SECTION I", "Section II – Writs"
            elif HEADING_SECTION_RE.match(text):
                # Extract the roman numeral that follows SECTION/Section keyword
                roman_match = re.search(r"(?:SECTION|section)\s+([IVX]+)", text, re.I)
                if roman_match:
                    canonical = roman_match.group(1).upper()
                    source = "heading_text"

            if canonical and canonical not in seen:
                seen.add(canonical)
                results.append(SectionInfo(
                    anchor_id=canonical,
                    heading_text=text,
                    heading_tag=tag_name,
                    source=source,
                    tier=2,
                ))

    return results


def _infer_doc_section(soup: BeautifulSoup) -> str:
    """Infer document-level section name from h1 or page title."""
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title:
        return title.get_text(strip=True).split("|")[0].strip()
    return "Document"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_sections(html_content: str) -> PageSections:
    """
    Main entry point. Given raw HTML, return a PageSections describing
    the tier, sections found, and primary subsection identifier.

    Parameters
    ----------
    html_content : str
        Raw HTML of the CPR page.

    Returns
    -------
    PageSections
        tier=1 if <a id> anchors found in headings
        tier=2 if heading text contains parseable rule/section identifiers
        tier=3 if no sub-structure found (simple document)
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Locate content div using same priority as scraper
    content = (
        soup.find("div", class_="entry-content")
        or soup.find("div", class_="article-content")
        or soup.find("div", class_="content")
        or soup.find("main")
        or soup.find("body")
    )
    if not content:
        content = soup

    # --- Tier 1: anchor IDs in headings ---
    tier1 = _extract_tier1_anchors(content)
    if tier1:
        ps = PageSections(tier=1, tier_reason="anchor_id_in_headings", sections=tier1)
        ps.all_section_ids = [s.anchor_id for s in tier1]
        ps.primary_section = tier1[0].anchor_id
        return ps

    # --- Tier 2: rule numbers in heading text ---
    tier2 = _extract_tier2_heading_text(content)
    if tier2:
        ps = PageSections(tier=2, tier_reason="heading_text_rule_numbers", sections=tier2)
        ps.all_section_ids = [s.anchor_id for s in tier2]
        ps.primary_section = tier2[0].anchor_id
        return ps

    # --- Tier 3: no sub-structure ---
    doc_section = _infer_doc_section(soup)
    return PageSections(
        tier=3,
        tier_reason="no_subsections_found",
        primary_section=doc_section,
        all_section_ids=[],
    )


def extract_sections_for_chunk(
    html_content: str,
    chunk_text: str,
) -> Tuple[Optional[str], List[str]]:
    """
    For a specific chunk of content from a page, determine:
    - subsection_id: the first/most relevant subsection in this chunk
    - subsections: all subsections referenced in this chunk

    IMPORTANT: subsections will ONLY contain section IDs that exist in the
    HTML page's structure (from extract_sections). This prevents inventing
    sections that don't actually exist in the page. Tier-3 pages (no HTML
    sub-structure) always return empty subsections.

    Parameters
    ----------
    html_content : str
        Raw HTML of the full page (used for Tier 1/2 detection).
    chunk_text : str
        The cleaned text content of this specific chunk.

    Returns
    -------
    (subsection_id, subsections)
        subsection_id: contextual label for this chunk (may be doc title for tier-3)
        subsections: list of HTML-verified section IDs found in chunk text
    """
    page_sections = extract_sections(html_content)
    header_section = _extract_section_from_chunk_header(chunk_text)
    html_section_set = set(page_sections.all_section_ids or [])

    if page_sections.tier == 3 or not page_sections.all_section_ids:
        # No subsection structure in HTML. subsection_id is for context only.
        # subsections is empty because HTML has no sections to verify against.
        subsection_id = header_section or page_sections.primary_section
        return subsection_id, []

    # Find which HTML-verified section IDs appear in this chunk's text
    found_positions: List[Tuple[int, int, str]] = []
    for sec_id in page_sections.all_section_ids:
        # Escape dots for regex
        pattern = re.escape(sec_id)
        # Match as a word boundary where possible
        match = re.search(rf"\b{pattern}\b", chunk_text)
        if match:
            # Sort by first position in chunk, then by longer IDs first if tied
            found_positions.append((match.start(), -len(sec_id), sec_id))

    chunk_sections: List[str] = []
    if found_positions:
        seen = set()
        for _, _, sec_id in sorted(found_positions):
            if sec_id not in seen:
                seen.add(sec_id)
                chunk_sections.append(sec_id)

    if chunk_sections:
        # Only use header_section if it's in the HTML-verified set
        if header_section and header_section in html_section_set:
            if header_section in chunk_sections:
                return header_section, chunk_sections
            for sec_id in chunk_sections:
                if sec_id.startswith(header_section) or header_section.startswith(sec_id):
                    return sec_id, chunk_sections
        return chunk_sections[0], chunk_sections

    # No section IDs found in chunk text via HTML map.
    # Only use header_section as subsection_id if it's in the HTML set.
    if header_section and header_section in html_section_set:
        return header_section, [header_section]

    # Chunk doesn't reference any HTML-verified sections.
    # Use page primary for context, but don't claim any sections we can't verify.
    return page_sections.primary_section, []


def _extract_section_from_chunk_header(chunk_text: str) -> Optional[str]:
    """
    Read section context injected by token chunker:
      Document: ...
      Section: <section context>
      Part N of M

    Returns a normalised section ID where possible, otherwise a cleaned label.
    """
    lines = chunk_text.splitlines()[:10]
    section_value: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("section:"):
            section_value = line.split(":", 1)[1].strip()
            break

    if not section_value:
        return None

    if section_value.lower() in {"document", ""}:
        return None

    # Common case: starts with a dotted rule (including malformed no-space variants)
    dotted = re.match(r"^(\d+[A-Z]?\.\d+[A-Z]?)", section_value)
    if dotted:
        return dotted.group(1)

    # Single number paragraph
    single = re.match(r"^(\d+)\b", section_value)
    if single:
        return single.group(1)

    # Roman section prefix
    roman = re.match(r"^([IVX]+)\b", section_value)
    if roman:
        return roman.group(1)

    # Try anchor normalisation from first token
    first_token = section_value.split()[0].strip("-–:;,.()[]")
    normalised = _normalise_anchor_id(first_token)
    if normalised:
        return normalised

    # Do not return arbitrary textual labels (e.g. "General", "Accepting offers")
    # for structured pages. Let caller fall back to HTML-derived page sections.
    return None
