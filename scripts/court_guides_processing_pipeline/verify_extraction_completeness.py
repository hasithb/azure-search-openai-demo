#!/usr/bin/env python3
"""
Tier 1 Automated Verification: PDF → JSON Content Completeness
================================================================
Runs 5 automated checks on each court guide to verify that all substantive
PDF content made it into the JSON output:

  1. Page count match (PDF pages vs Azure DI pages)
  2. N-gram coverage (PDF 6-grams found in JSON)
  3. Page-by-page coverage (per-page 6-gram match)
  4. Block-level analysis (classify every missing block)
  5. Two-extractor cross-validation (pymupdf vs Azure DI)

Usage:
    python verify_extraction_completeness.py              # All guides
    python verify_extraction_completeness.py --guide Patents  # Single guide
    python verify_extraction_completeness.py --verbose     # Show all block details
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

BASE = Path(__file__).resolve().parent
SRC_DIR = BASE / "sources"
OUT_DIR = BASE / "outputs_azure_di"

# Acceptance thresholds
MIN_NGRAM_COVERAGE = 0.85          # Overall 6-gram coverage must be >= 85%
MIN_PAGE_COVERAGE = 0.40           # Per-page 6-gram coverage must be >= 40%
MIN_CROSS_VALIDATION = 0.82        # pymupdf vs DI 8-gram agreement >= 82%
MAX_SUBSTANTIVE_MISSES = 0         # Zero substantive blocks allowed missing


@dataclass
class PageResult:
    page_num: int
    pdf_ngrams: int
    matched: int
    coverage: float
    status: str  # GOOD, PARTIAL, MISSING, SKIP


@dataclass
class MissingBlock:
    page_num: int
    text: str
    norm_length: int
    category: str
    in_di: bool = False  # True if block IS in DI markdown (pipeline drop, not extraction miss)
    in_json_via_context: bool = False  # True if block content found in JSON as part of larger passage


@dataclass
class GuideReport:
    name: str
    pdf_pages: int = 0
    di_pages: int = 0
    json_docs: int = 0
    # Check 1: page count
    page_count_match: bool = False
    page_count_note: str = ""
    # Check 2: n-gram coverage
    ngram_4_coverage: float = 0.0
    ngram_6_coverage: float = 0.0
    ngram_8_coverage: float = 0.0
    # Check 3: page-by-page
    page_results: list = field(default_factory=list)
    pages_good: int = 0
    pages_partial: int = 0
    pages_missing: int = 0
    pages_skip: int = 0
    # Check 4: block-level
    total_blocks: int = 0
    found_blocks: int = 0
    missing_blocks: list = field(default_factory=list)
    categories: dict = field(default_factory=dict)
    substantive_misses: int = 0
    substantive_in_di_only: int = 0  # In DI but not JSON — pipeline drop
    short_in_di: int = 0  # SHORT_FRAGMENT/SHORT_HEADING found in DI
    short_not_in_di: int = 0  # SHORT_FRAGMENT/SHORT_HEADING NOT in DI
    running_headers: int = 0  # Detected running headers/footers
    pipeline_drops_validated: int = 0  # DI-only blocks matching known filter patterns
    pipeline_drops_unvalidated: int = 0  # DI-only blocks NOT matching known patterns
    # Check 5: cross-validation
    pdf_only_ngrams: int = 0
    di_only_ngrams: int = 0
    cross_agreement: float = 0.0
    font_encoded_blocks: int = 0
    # Verdict
    passed: bool = False
    issues: list = field(default_factory=list)


def normalize(text: str) -> str:
    """Normalize text for comparison: strip HTML/markdown, lowercase, collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_ngrams(text: str, n: int = 6) -> set[tuple[str, ...]]:
    words = text.split()
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def is_font_encoded(text: str) -> bool:
    """Detect text that's garbled due to custom font encoding (character substitution).
    These appear as runs of consonants/shifted characters that don't form real words,
    or repeated short tokens like 'P56 P56 P56...'."""
    clean = re.sub(r"\s+", "", text)
    if len(clean) < 20:
        return False

    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

    # Pattern 1: Long garbled strings (e.g., "$QQH[*3URFHGXUH...")
    upper_ratio = sum(1 for c in clean if c.isupper()) / len(clean)
    if avg_word_len > 10 and upper_ratio > 0.3:
        return True

    # Pattern 2: Repeated short tokens (e.g., "P56 P56 P56 P56...")
    # Strip punctuation from words for comparison
    stripped = [re.sub(r"[^\w]", "", w) for w in words]
    stripped = [w for w in stripped if w]
    if len(stripped) >= 3:
        from collections import Counter
        counts = Counter(stripped)
        most_common_word, most_common_count = counts.most_common(1)[0]
        # If a single token accounts for >60% of all words, it's repeated junk
        if most_common_count / len(stripped) > 0.6 and len(most_common_word) <= 5:
            return True
        # If very few unique words relative to total
        if len(counts) <= 3 and max(len(w) for w in counts) <= 5:
            return True

    # Pattern 3: High ratio of non-dictionary-like character sequences
    # Characters shifted by a constant offset produce text with unusual bigram patterns
    consonant_clusters = len(re.findall(r"[bcdfghjklmnpqrstvwxyz]{4,}", clean.lower()))
    if consonant_clusters >= 2 and avg_word_len > 6:
        return True

    return False


def is_toc_entry(text: str) -> bool:
    """Detect ToC entries — lines like 'A. Preliminaries 15' or 'D.3 Fixing a CMC 39'.
    ToC entries typically have section markers followed by titles and page numbers."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False
    # Count lines that contain embedded page numbers (e.g., "A.1  The framework  15")
    toc_pattern = re.compile(r"(?:^[A-Z]\.|^\d+\.\d*|^Part\s).*\d{1,3}\s*$")
    toc_matches = sum(1 for l in lines if toc_pattern.search(l.strip()))
    # Also catch multi-section blocks: "Section Title  15  Section Title  16"
    page_nums_in_text = len(re.findall(r"\b\d{1,3}\b", text))
    words = text.split()
    if page_nums_in_text > 3 and page_nums_in_text > len(words) * 0.08:
        return True
    return toc_matches >= 1


def is_checklist_or_form(text: str) -> bool:
    """Detect checklist/form content — repeated question patterns, check boxes, blank fields."""
    text_lower = text.lower()
    patterns = ["question", "answer", "yes/no", "yes / no", "yes ☐", "no ☐", "tick", "checklist",
                 "please set out", "continue on a separate page", "grounds of appeal",
                 "grounds relied on", "section 3", "section 4"]
    if sum(1 for p in patterns if p in text_lower) >= 2:
        return True
    # Form-like: "Does the application ...", "Has the ... been ..." repeated
    questions = len(re.findall(r"(?i)^(?:does|has|is|are|were|will|can|should)\s", text, re.MULTILINE))
    return questions >= 2


def is_running_header(text: str, all_page_texts: list[str], page_idx: int) -> bool:
    """Detect running headers/footers — text that appears identically on multiple pages."""
    norm = normalize(text)
    if len(norm) < 10 or len(norm) > 120:
        return False
    # Count how many pages contain this exact normalized text
    matches = 0
    for i, pt in enumerate(all_page_texts):
        if i == page_idx:
            continue
        if norm in normalize(pt):
            matches += 1
            if matches >= 3:  # Appears on 4+ pages (including this one) → running header
                return True
    return False


def is_known_pipeline_filter(text: str) -> bool:
    """Check if text matches patterns that the extraction pipeline intentionally filters out.
    These are content types that Azure DI extracts but our pipeline deliberately drops.

    Mirrors the extraction pipeline's actual filters:
    - is_toc_section(): ToC pages
    - is_boilerplate_section(): OGL pages, untitled sections <200 chars
    - strip_ogl_tail(): Crown copyright / OGL at end
    - clean_content(): PageHeader/PageFooter/PageNumber/PageBreak comments
    - is_document_title_repeat(): repeated guide title headings
    """
    text_lower = text.lower()
    norm = normalize(text)

    # 1. ToC content — pipeline strips entire ToC sections
    if is_toc_entry(text):
        return True
    if any(x in text_lower for x in ["contents", "table of contents"]):
        return True
    # Section listings that look like ToC entries (e.g., "Section 2 - Entitlement  Section 3 - ...")
    if text_lower.count("section") >= 2 and re.search(r"section\s+\d", text_lower):
        return True

    # 2. Copyright / OGL / publisher boilerplate — pipeline strips via strip_ogl_tail() and is_boilerplate_section()
    if any(x in text_lower for x in [
        "crown copyright", "open government", "nationalarchives",
        "published by", "by authority of", "issued by",
        "third party copyright", "third-party copyright",
        "enquiries regarding this publication",
        "permission from the copyright holder",
    ]):
        return True

    # 3. Form/checklist content — pipeline drops appendix forms
    if is_checklist_or_form(text):
        return True
    # Single checklist question (DI splits multi-question forms into individual blocks)
    if re.match(r"(?i)^(?:does|has|is|are|were|will|can|should|have|was)\s", text.strip()):
        # Single question block ending with ? or relatively short
        if "?" in text or len(norm) < 200:
            return True
    # Conditional form questions: "If so, ...", "Where ... is ...", "In section N, ..."
    if re.match(r"(?i)^(?:if so|in section|where\s+\w+\s+(?:is|are|was|were|has|have))\b", text.strip()):
        if "?" in text or len(norm) < 200:
            return True
    # Form instructions
    if any(x in text_lower for x in [
        "this section must be completed",
        "must be completed even where",
    ]):
        return True

    # 4. Short headings that get merged into parent sections
    if re.match(r"^[\d.]+\s+\w", text_lower) and len(norm) < 80:
        return True

    # 5. Title page / cover page / chapter headings — pipeline drops via is_boilerplate_section()
    # (untitled <200 chars) and is_document_title_repeat()
    title_keywords = ["guide", "court", "division", "practice direction", "king's bench",
                      "queen's bench", "chancery", "commercial", "technology and construction",
                      "senior courts costs office", "patents court", "court of appeal"]
    if sum(1 for kw in title_keywords if kw in text_lower) >= 2 and len(norm) < 200:
        return True
    # Chapter headings that the pipeline merges into parent sections
    if re.match(r"(?i)^chapter\s+\d+", text.strip()) and len(norm) < 200:
        return True

    # 6. Form template text — legal form templates, draft orders, model forms
    if any(x in text_lower for x in [
        "[date]", "administrative form", "bill of costs",
        "pursuant to an agreement", "draft order",
        "name and address of the",
        "to be assessed if not agreed",
        "on the standard basis",
    ]):
        return True

    # 7. Section description text — short descriptions used as section sub-headings
    # (pipeline treats these as headings merged into parent, not standalone content)
    if re.match(r"(?i)^\s*(detailed assessment|further procedure|short form)\s", text.strip()):
        if len(norm) < 200:
            return True

    # 7. Materials lists (short references to bundles/documents)
    if text_lower.startswith("materials:") and len(norm) < 200:
        return True

    # 8. Contact/informational blocks not relevant to legal guidance
    if any(x in text_lower for x in [
        "information for people with disabilities",
        "hearing loop", "wheelchair access",
    ]):
        return True

    return False


def classify_missing_block(text: str) -> str:
    """Classify a missing block by category."""
    text_lower = text.lower().strip()
    norm = normalize(text)

    if is_font_encoded(text):
        return "FONT_ENCODED"
    if any(x in text_lower for x in ["……", "...", "…", ".....", "────", "⋯"]):
        return "TOC_DOTS_LINE"
    if is_toc_entry(text):
        return "TOC_ENTRY"
    if re.search(r"\d{1,3}\s*$", text_lower.strip().split("\n")[-1]) and "table" not in text_lower:
        lines = text_lower.strip().split("\n")
        if len(lines) <= 3 and any(re.search(r"\d{1,3}$", l.strip()) for l in lines):
            return "TOC_WITH_PAGE_NUMBERS"
    if any(x in text_lower for x in ["crown copyright", "open government licence", "nationalarchives"]):
        return "OGL_BOILERPLATE"
    if any(x in text_lower for x in ["published by", "by authority of", "issued by"]):
        return "PUBLISHER_INFO"
    if is_checklist_or_form(text):
        return "CHECKLIST_FORM"
    if text_lower.startswith("intentionally left blank") or text_lower.startswith("[intentionally"):
        return "BLANK_PAGE"
    if re.match(r"^[\d.]+\s+\w", text_lower) and len(norm) < 80:
        return "SHORT_HEADING"
    if len(norm) < 30:
        return "SHORT_FRAGMENT"
    if len(norm) > 80:
        return "POTENTIALLY_SUBSTANTIVE"
    return "SHORT_FRAGMENT"


def count_di_pages(md_text: str) -> int:
    """Count pages in Azure DI markdown output."""
    return len(re.findall(r'<!-- PageNumber="(\d+)" -->', md_text))


def verify_guide(pdf_path: Path, json_path: Path, md_path: Path, verbose: bool = False) -> GuideReport:
    """Run all 5 verification checks on a single guide."""
    stem = pdf_path.stem
    report = GuideReport(name=stem)

    # Load data
    doc = pymupdf.open(str(pdf_path))
    report.pdf_pages = doc.page_count

    with open(json_path) as f:
        json_docs = json.load(f)
    report.json_docs = len(json_docs)

    json_all = " ".join(d["content"] for d in json_docs)
    json_norm = normalize(json_all)

    md_text = md_path.read_text()
    di_norm = normalize(md_text)
    report.di_pages = count_di_pages(md_text)

    # ── Check 1: Page count match ──
    report.page_count_match = report.pdf_pages == report.di_pages
    if not report.page_count_match:
        diff = report.pdf_pages - report.di_pages
        report.page_count_note = (
            f"PDF={report.pdf_pages}, DI={report.di_pages} (diff={diff}, "
            f"likely blank/near-blank pages skipped by DI)"
        )

    # ── Check 2: N-gram coverage (PDF → JSON) ──
    pdf_full = ""
    page_texts = []
    for i in range(doc.page_count):
        page_text = doc[i].get_text()
        pdf_full += page_text + "\n"
        page_texts.append(page_text)
    pdf_norm = normalize(pdf_full)

    json_ngrams_4 = extract_ngrams(json_norm, 4)
    json_ngrams_6 = extract_ngrams(json_norm, 6)
    json_ngrams_8 = extract_ngrams(json_norm, 8)

    pdf_ngrams_4 = extract_ngrams(pdf_norm, 4)
    pdf_ngrams_6 = extract_ngrams(pdf_norm, 6)
    pdf_ngrams_8 = extract_ngrams(pdf_norm, 8)

    if pdf_ngrams_4:
        report.ngram_4_coverage = len(pdf_ngrams_4 & json_ngrams_4) / len(pdf_ngrams_4)
    if pdf_ngrams_6:
        report.ngram_6_coverage = len(pdf_ngrams_6 & json_ngrams_6) / len(pdf_ngrams_6)
    if pdf_ngrams_8:
        report.ngram_8_coverage = len(pdf_ngrams_8 & json_ngrams_8) / len(pdf_ngrams_8)

    if report.ngram_6_coverage < MIN_NGRAM_COVERAGE:
        report.issues.append(
            f"6-gram coverage {report.ngram_6_coverage:.1%} below threshold {MIN_NGRAM_COVERAGE:.0%}"
        )

    # ── Check 3: Page-by-page coverage ──
    for i, page_text in enumerate(page_texts):
        page_norm = normalize(page_text)
        page_6grams = extract_ngrams(page_norm, 6)

        if not page_6grams or len(page_norm) < 30:
            report.page_results.append(PageResult(i + 1, 0, 0, 0.0, "SKIP"))
            report.pages_skip += 1
            continue

        matched = len(page_6grams & json_ngrams_6)
        coverage = matched / len(page_6grams)

        if coverage >= 0.6:
            status = "GOOD"
            report.pages_good += 1
        elif coverage >= MIN_PAGE_COVERAGE:
            status = "PARTIAL"
            report.pages_partial += 1
        else:
            status = "MISSING"
            report.pages_missing += 1

        report.page_results.append(PageResult(i + 1, len(page_6grams), matched, coverage, status))

    # Classify each missing/partial page
    di_6grams = extract_ngrams(di_norm, 6)
    for pr in report.page_results:
        if pr.status in ("MISSING", "PARTIAL") and pr.page_num not in (1, report.pdf_pages):
            raw_text = page_texts[pr.page_num - 1]
            page_text = normalize(raw_text)
            if len(page_text) < 50:
                continue

            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            num_lines = len(lines)

            # Check if this page is a ToC page
            is_toc = False
            if any(x in page_text for x in ["contents", "table of contents"]):
                is_toc = True
            if is_toc_entry(raw_text):
                is_toc = True
            page_nums = len(re.findall(r"\b\d{1,3}\b", raw_text))
            words = raw_text.split()
            if page_nums > 5 and page_nums > len(words) * 0.05:
                is_toc = True

            is_boilerplate = any(
                x in page_text for x in ["crown copyright", "open government", "nationalarchives"]
            )
            is_blank = any(
                x in page_text
                for x in ["intentionally left blank", "deliberately blank", "this page intentionally"]
            )
            is_form = is_checklist_or_form(raw_text)

            # Detect title/cover pages (very few lines, contains guide name)
            is_title_page = num_lines <= 6 and any(
                x in page_text for x in ["guide", "judiciary", "court"]
            )

            # Detect part divider pages ("Part A: General matters")
            is_divider = num_lines <= 6 and any(
                re.match(r"(?i)^part\s+[a-z]", l.strip()) for l in lines
            )

            # Detect floor plan/map pages
            is_map = any(
                x in page_text for x in ["plan a", "plan b", "map of", "floor plan", "turnstile"]
            )

            # Detect contact list / email-heavy pages
            email_count = len(re.findall(r"[\w.]+@[\w.]+", raw_text))
            is_contact_list = email_count >= 3

            # Detect form/template pages (appendix, schedule, precedent, specimen)
            is_template = any(
                x in page_text
                for x in [
                    "schedule of costs", "precedent", "specimen",
                    "signed by", "notice of appeal", "appellant",
                ]
            )

            if is_toc or is_boilerplate or is_blank or is_title_page or is_divider or is_map or is_contact_list:
                continue  # Don't flag these

            if is_form or is_template:
                # Check if content IS in DI (pipeline drop)
                page_6grams_set = extract_ngrams(page_text, 6)
                if page_6grams_set:
                    di_match = len(page_6grams_set & di_6grams) / len(page_6grams_set)
                else:
                    di_match = 0.0
                if di_match > 0.5 and pr.status == "MISSING":
                    report.issues.append(
                        f"Page {pr.page_num} coverage {pr.coverage:.1%} in JSON "
                        f"but {di_match:.0%} in DI — form/checklist (pipeline drop)"
                    )
                continue

            # For remaining pages, check DI coverage
            page_6grams_set = extract_ngrams(page_text, 6)
            if page_6grams_set:
                di_match = len(page_6grams_set & di_6grams) / len(page_6grams_set)
            else:
                di_match = 0.0

            if di_match > 0.7:
                if pr.status == "MISSING":
                    report.issues.append(
                        f"Page {pr.page_num} coverage {pr.coverage:.1%} in JSON "
                        f"but {di_match:.0%} in DI — pipeline drop"
                    )
            elif num_lines <= 15 and di_match > 0.3:
                # Short page with partial DI coverage — likely cross-page attribution
                report.issues.append(
                    f"Page {pr.page_num} coverage {pr.coverage:.1%} in JSON, "
                    f"{di_match:.0%} in DI — short page, likely cross-page content"
                )
            else:
                report.issues.append(
                    f"Page {pr.page_num} coverage {pr.coverage:.1%} — may be missing content"
                )

    # ── Check 4: Block-level analysis ──
    # Helper: check if a normalized block appears in DI markdown
    def block_in_di(norm_text: str) -> bool:
        search_len = min(30, len(norm_text))
        for j in range(0, max(1, len(norm_text) - search_len), max(1, search_len // 2)):
            if norm_text[j : j + search_len] in di_norm:
                return True
        return False

    # Helper: check if a normalized block appears in JSON via wider context
    # Uses shorter 15-char windows to catch partial matches in larger passages
    def block_in_json_context(norm_text: str) -> bool:
        search_len = min(15, len(norm_text))
        if search_len < 10:
            return False
        matches = 0
        for j in range(0, max(1, len(norm_text) - search_len), search_len):
            if norm_text[j : j + search_len] in json_norm:
                matches += 1
        total_windows = max(1, (len(norm_text) - search_len) // search_len + 1)
        return matches / total_windows >= 0.5  # At least 50% of windows found

    for i in range(doc.page_count):
        page = doc[i]
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] != 0:  # skip image blocks
                continue
            text = block[4]
            norm = normalize(text)
            if len(norm) < 25:
                continue

            report.total_blocks += 1

            # Search for block in JSON using sliding window (30-char chunks, 15-char step)
            found = False
            search_len = min(30, len(norm))
            for j in range(0, max(1, len(norm) - search_len), max(1, search_len // 2)):
                chunk = norm[j : j + search_len]
                if chunk in json_norm:
                    found = True
                    break

            if found:
                report.found_blocks += 1
            else:
                # Check for running headers/footers
                is_header = is_running_header(text, page_texts, i)
                if is_header:
                    report.running_headers += 1

                category = classify_missing_block(text)

                # DI cross-check for ALL categories (not just POTENTIALLY_SUBSTANTIVE)
                in_di = block_in_di(norm)
                in_json_ctx = block_in_json_context(norm)

                if category == "POTENTIALLY_SUBSTANTIVE":
                    if in_di:
                        report.substantive_in_di_only += 1
                        # Validate: does it match a known filter pattern?
                        if is_known_pipeline_filter(text):
                            report.pipeline_drops_validated += 1
                        else:
                            report.pipeline_drops_unvalidated += 1
                elif category in ("SHORT_FRAGMENT", "SHORT_HEADING"):
                    if in_di:
                        report.short_in_di += 1
                    else:
                        report.short_not_in_di += 1

                mb = MissingBlock(
                    i + 1, text[:300].replace("\n", " "), len(norm),
                    category if not is_header else f"RUNNING_HEADER ({category})",
                    in_di, in_json_ctx,
                )
                report.missing_blocks.append(mb)

                actual_cat = mb.category
                report.categories[actual_cat] = report.categories.get(actual_cat, 0) + 1

                if category == "POTENTIALLY_SUBSTANTIVE" and not in_di:
                    report.substantive_misses += 1
                if category == "FONT_ENCODED":
                    report.font_encoded_blocks += 1

    if report.substantive_misses > MAX_SUBSTANTIVE_MISSES:
        report.issues.append(
            f"{report.substantive_misses} potentially substantive block(s) missing from BOTH JSON and DI"
        )

    # ── Check 5: Two-extractor cross-validation ──
    di_ngrams_8 = extract_ngrams(di_norm, 8)

    if pdf_ngrams_8 and di_ngrams_8:
        shared = pdf_ngrams_8 & di_ngrams_8
        report.pdf_only_ngrams = len(pdf_ngrams_8 - di_ngrams_8)
        report.di_only_ngrams = len(di_ngrams_8 - pdf_ngrams_8)
        total_union = len(pdf_ngrams_8 | di_ngrams_8)
        report.cross_agreement = len(shared) / total_union if total_union else 0

    if report.cross_agreement < MIN_CROSS_VALIDATION:
        report.issues.append(
            f"Cross-validation agreement {report.cross_agreement:.1%} below threshold {MIN_CROSS_VALIDATION:.0%}"
        )

    doc.close()

    # ── Verdict ──
    # Only real substantive misses (not in DI either) count as failures
    critical_issues = [
        i for i in report.issues if "may be missing content" in i.lower()
    ]
    report.passed = report.substantive_misses == 0 and len(critical_issues) == 0

    return report


def print_report(report: GuideReport, verbose: bool = False) -> None:
    """Print a formatted report for a single guide."""
    status = "✅ PASS" if report.passed else "❌ FAIL"
    print(f"\n{'═' * 80}")
    print(f"  {report.name}")
    print(f"  {status}  |  {report.pdf_pages} pages  |  {report.json_docs} JSON docs")
    print(f"{'═' * 80}")

    # Check 1
    pc_status = "✓" if report.page_count_match else "~"
    print(f"\n  1. Page Count:  PDF={report.pdf_pages}  DI={report.di_pages}  [{pc_status}]")
    if report.page_count_note:
        print(f"      {report.page_count_note}")

    # Check 2
    ng_status = "✓" if report.ngram_6_coverage >= MIN_NGRAM_COVERAGE else "✗"
    print(
        f"  2. N-gram Coverage:  4g={report.ngram_4_coverage:.1%}  "
        f"6g={report.ngram_6_coverage:.1%}  8g={report.ngram_8_coverage:.1%}  [{ng_status}]"
    )

    # Check 3
    print(
        f"  3. Page-by-Page:  {report.pages_good} GOOD  {report.pages_partial} PARTIAL  "
        f"{report.pages_missing} MISSING  {report.pages_skip} SKIP"
    )
    if verbose or report.pages_missing > 0:
        for pr in report.page_results:
            if pr.status in ("MISSING", "PARTIAL") or verbose:
                marker = {"GOOD": "  ", "PARTIAL": "⚠ ", "MISSING": "✗ ", "SKIP": "  "}[pr.status]
                print(
                    f"      {marker}Page {pr.page_num:3d}: {pr.coverage:5.1%} "
                    f"({pr.matched}/{pr.pdf_ngrams} 6-grams)  [{pr.status}]"
                )

    # Check 4
    block_pct = report.found_blocks / report.total_blocks * 100 if report.total_blocks else 0
    sub_status = "✓" if report.substantive_misses == 0 else "✗"
    print(
        f"  4. Block-Level:  {report.found_blocks}/{report.total_blocks} found "
        f"({block_pct:.1f}%)  [{sub_status}]"
    )
    if report.categories:
        print("      Missing blocks by category:")
        for cat, count in sorted(report.categories.items(), key=lambda x: -x[1]):
            icon = {"FONT_ENCODED": "🔤", "POTENTIALLY_SUBSTANTIVE": "⚠️ "}.get(cat, "  ")
            print(f"        {icon}{cat}: {count}")
    if report.substantive_in_di_only > 0:
        validated = report.pipeline_drops_validated
        unvalidated = report.pipeline_drops_unvalidated
        print(
            f"      Note: {report.substantive_in_di_only} \"substantive\" block(s) ARE in DI markdown "
            f"but not in JSON (pipeline drops: {validated} match known filters, "
            f"{unvalidated} unvalidated)"
        )
        if unvalidated > 0:
            for mb in report.missing_blocks:
                if mb.category == "POTENTIALLY_SUBSTANTIVE" and mb.in_di:
                    if not is_known_pipeline_filter(mb.text):
                        print(f"        ⚠ Unvalidated drop p{mb.page_num}: {mb.text[:100]}")
    if report.substantive_misses > 0:
        print(f"      ⚠ {report.substantive_misses} block(s) NOT in DI either — potential real losses:")
        for mb in report.missing_blocks:
            if mb.category == "POTENTIALLY_SUBSTANTIVE" and not mb.in_di:
                print(f"        Page {mb.page_num}: {mb.text[:120]}")
    # Short block DI cross-check summary
    short_total = report.short_in_di + report.short_not_in_di
    if short_total > 0:
        print(
            f"      Short blocks (fragments+headings): {short_total} total — "
            f"{report.short_in_di} in DI ({report.short_in_di / short_total:.0%}), "
            f"{report.short_not_in_di} not in DI ({report.short_not_in_di / short_total:.0%})"
        )
    if report.running_headers > 0:
        print(
            f"      Running headers/footers detected: {report.running_headers} "
            f"(appear on 4+ pages — not substantive content)"
        )
    if verbose:
        for mb in report.missing_blocks:
            di_tag = " [in DI]" if mb.in_di else ""
            icon = "⚠️ " if mb.category == "POTENTIALLY_SUBSTANTIVE" and not mb.in_di else "  "
            print(f"      {icon}Page {mb.page_num} [{mb.category}{di_tag}]: {mb.text[:100]}")

    # Check 5
    xv_status = "✓" if report.cross_agreement >= MIN_CROSS_VALIDATION else "✗"
    print(
        f"  5. Cross-Validation:  agreement={report.cross_agreement:.1%}  "
        f"pymupdf-only={report.pdf_only_ngrams}  DI-only={report.di_only_ngrams}  [{xv_status}]"
    )
    if report.font_encoded_blocks > 0:
        print(
            f"      Note: {report.font_encoded_blocks} block(s) use custom font encoding "
            f"(pymupdf garbles these; DI reads them correctly via OCR)"
        )

    # Issues
    if report.issues:
        print(f"\n  Issues:")
        for issue in report.issues:
            print(f"    ⚠  {issue}")


def print_summary(reports: list[GuideReport]) -> None:
    """Print overall summary across all guides."""
    total_pages = sum(r.pdf_pages for r in reports)
    total_docs = sum(r.json_docs for r in reports)
    total_blocks = sum(r.total_blocks for r in reports)
    found_blocks = sum(r.found_blocks for r in reports)
    total_substantive = sum(r.substantive_misses for r in reports)
    total_di_only = sum(r.substantive_in_di_only for r in reports)
    passed = sum(1 for r in reports if r.passed)
    failed = len(reports) - passed

    print(f"\n{'━' * 80}")
    print(f"  SUMMARY: {len(reports)} guides  |  {total_pages} PDF pages  |  {total_docs} JSON docs")
    print(f"{'━' * 80}")
    print(f"  Passed: {passed}   Failed: {failed}")
    print(f"  Blocks found: {found_blocks}/{total_blocks} ({found_blocks / total_blocks * 100:.1f}%)")
    print(f"  Substantive misses (not in JSON or DI): {total_substantive}")
    if total_di_only > 0:
        validated = sum(r.pipeline_drops_validated for r in reports)
        unvalidated = sum(r.pipeline_drops_unvalidated for r in reports)
        print(f"  Blocks in DI but not JSON (pipeline drops): {total_di_only} ({validated} validated, {unvalidated} unvalidated)")
    total_short = sum(r.short_in_di + r.short_not_in_di for r in reports)
    short_in_di = sum(r.short_in_di for r in reports)
    total_headers = sum(r.running_headers for r in reports)
    if total_short > 0:
        print(f"  Short blocks: {total_short} ({short_in_di} in DI = {short_in_di/total_short:.0%}, {total_short - short_in_di} not in DI)")
    if total_headers > 0:
        print(f"  Running headers/footers: {total_headers}")

    avg_6gram = sum(r.ngram_6_coverage for r in reports) / len(reports)
    avg_cross = sum(r.cross_agreement for r in reports) / len(reports)
    print(f"  Avg 6-gram coverage: {avg_6gram:.1%}")
    print(f"  Avg cross-validation: {avg_cross:.1%}")

    if total_substantive == 0 and failed == 0:
        print(f"\n  ✅ ALL GUIDES PASS — no substantive content losses detected")
    else:
        print(f"\n  ❌ {failed} guide(s) require investigation")
        for r in reports:
            if not r.passed:
                print(f"    - {r.name}: {', '.join(r.issues)}")

    print()


def discover_guides(filter_name: str | None = None) -> list[tuple[Path, Path, Path]]:
    """Find all guide triplets (PDF, JSON, markdown) in the standard directories."""
    guides = []
    for pdf in sorted(SRC_DIR.glob("*.pdf")):
        stem = pdf.stem
        if filter_name and filter_name.lower() not in stem.lower():
            continue
        json_path = OUT_DIR / f"{stem}_processed.json"
        md_path = OUT_DIR / f"{stem}_azure_di.md"
        if json_path.exists() and md_path.exists():
            guides.append((pdf, json_path, md_path))
        else:
            print(f"  Warning: Missing outputs for {stem}", file=sys.stderr)
            if not json_path.exists():
                print(f"    Missing: {json_path}", file=sys.stderr)
            if not md_path.exists():
                print(f"    Missing: {md_path}", file=sys.stderr)
    return guides


def main():
    parser = argparse.ArgumentParser(description="Verify PDF → JSON extraction completeness")
    parser.add_argument("--guide", help="Filter to a specific guide (partial name match)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all block/page details")
    parser.add_argument(
        "--json-output", help="Write results to a JSON file", metavar="PATH"
    )
    args = parser.parse_args()

    guides = discover_guides(args.guide)
    if not guides:
        print("No guides found. Check that sources/ and outputs_azure_di/ exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying {len(guides)} guide(s)...")
    reports = []
    for pdf, json_path, md_path in guides:
        report = verify_guide(pdf, json_path, md_path, verbose=args.verbose)
        print_report(report, verbose=args.verbose)
        reports.append(report)

    print_summary(reports)

    # Optional JSON output
    if args.json_output:
        out = []
        for r in reports:
            out.append({
                "name": r.name,
                "passed": r.passed,
                "pdf_pages": r.pdf_pages,
                "di_pages": r.di_pages,
                "json_docs": r.json_docs,
                "ngram_4_coverage": round(r.ngram_4_coverage, 4),
                "ngram_6_coverage": round(r.ngram_6_coverage, 4),
                "ngram_8_coverage": round(r.ngram_8_coverage, 4),
                "pages_good": r.pages_good,
                "pages_partial": r.pages_partial,
                "pages_missing": r.pages_missing,
                "pages_skip": r.pages_skip,
                "total_blocks": r.total_blocks,
                "found_blocks": r.found_blocks,
                "block_coverage": round(r.found_blocks / r.total_blocks, 4) if r.total_blocks else 0,
                "missing_categories": r.categories,
                "substantive_misses": r.substantive_misses,
                "substantive_in_di_only": r.substantive_in_di_only,
                "font_encoded_blocks": r.font_encoded_blocks,
                "cross_agreement": round(r.cross_agreement, 4),
                "pdf_only_ngrams": r.pdf_only_ngrams,
                "di_only_ngrams": r.di_only_ngrams,
                "issues": r.issues,
                "missing_blocks": [
                    {"page": mb.page_num, "category": mb.category, "text": mb.text[:200]}
                    for mb in r.missing_blocks
                    if mb.category == "POTENTIALLY_SUBSTANTIVE"
                ],
            })
        out_path = Path(args.json_output)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  JSON report written to {out_path}")

    # Exit code
    sys.exit(0 if all(r.passed for r in reports) else 1)


if __name__ == "__main__":
    main()
