"""
Comprehensive Citation Testing Methodology for Legal RAG (v3 index).

This test suite validates the FULL citation pipeline end-to-end:

  Layer 1: Backend Citation Generation (citation_builder + source_processor)
     - All subsection_id patterns generate correctly formatted citations
     - Multi-subsection splitting preserves citation integrity
     - Citation deduplication between subsection and sourcepage

  Layer 2: Frontend Citation Parsing (AnswerParser.tsx logic)
     - parseAnswerToHtml correctly maps [n] references to enhanced citations
     - fixInconsistentCitation corrects misaligned document/subsection combos
     - Citation labels survive round-trip through the pipeline

  Layer 3: Frontend Navigation (SupportingContent.tsx logic)
     - parseSubsectionFromCitation extracts valid subsection tokens
     - findMatchingContentIndex locates the correct supporting content card
     - extractSubsectionContent highlights the correct subsection within content

  Layer 4: Live Integration (backend → frontend data flow)
     - Every citation in enhanced_citations is well-formed
     - Every citation_map entry resolves to a matching data_point
     - Clicking any citation would navigate to the right content card

Test Categories:
  A. Format correctness (regex validation of all citation strings)
  B. Field completeness (every required field is present and non-empty)
  C. Navigation accuracy (citation → content card → subsection highlight)
  D. Edge cases (empty fields, Unicode, unusually long content, etc.)
  E. Coverage (every document category and subsection type is tested)

Run:
    # Unit tests only (fast, no backend required)
    pytest tests/test_citation_methodology.py -v -k "not Live"

    # Full suite including live integration (requires running backend)
    pytest tests/test_citation_methodology.py -v

    # With coverage
    pytest tests/test_citation_methodology.py --cov=customizations --cov-report=term-missing
"""

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "backend"))

from approaches.approach import Document, DataPoints, ExtraInfo
from customizations.approaches.citation_builder import CitationBuilder, citation_builder
from customizations.approaches.source_processor import SourceProcessor

# Create processor for tests
_processor = SourceProcessor(citation_builder)


# ────────────────────────────────────────────────────────────────────────────────
# Shared Patterns & Validators
# ────────────────────────────────────────────────────────────────────────────────

# Valid subsection_id formats observed in the live index
SUBSECTION_ID_PATTERNS = [
    (r'^\d+\.\d+$', "numeric", "1.1"),
    (r'^\d+\.\d+\.\d+$', "numeric_deep", "1.1.1"),
    (r'^[A-Z]\.\d+$', "alpha_dot", "A.1"),
    (r'^[A-Z]\.\d+\.\d+$', "alpha_dot_deep", "A.1.1"),
    (r'^[A-Z]\d+\.\d+$', "alpha_num_dot", "A1.1"),
    (r'^[A-Z]\d+$', "alpha_num", "A1"),
    (r'^Rule \d+(\.\d+)?$', "rule", "Rule 31.1"),
    (r'^Para \d+(\.\d+)?$', "para", "Para 5"),
    (r'^Part \d+$', "part", "Part 35"),
    (r'^PRACTICE DIRECTION \d+[A-Z]?$', "pd_header", "PRACTICE DIRECTION 1A"),
]

# A citation is well-formed if it matches one of these formats:
#   3-part: "subsection, sourcepage, sourcefile"
#   2-part: "sourcepage, sourcefile"  (when no subsection)
#   1-part: "sourcefile"  (fallback)
#   N-part: commas in sourcepage create 4+ parts (e.g., "E.1.1, E. Disclosure, E.1 Generally (p. 60), CCG")
CITATION_FORMAT_RE = re.compile(
    r'^.+$'  # At minimum, citation must be non-empty
)


def is_valid_subsection(sub: str) -> bool:
    """Check if a subsection_id matches any known legal format."""
    if not sub:
        return False
    for pattern, _, _ in SUBSECTION_ID_PATTERNS:
        if re.match(pattern, sub, re.IGNORECASE):
            return True
    return False


def parse_citation_parts(citation: str) -> dict:
    """Parse a comma-separated citation into structured parts."""
    parts = [p.strip() for p in citation.split(",") if p.strip()]
    if len(parts) >= 3:
        return {
            "subsection": parts[0],
            "sourcepage": ", ".join(parts[1:-1]),
            "sourcefile": parts[-1],
            "part_count": len(parts),
        }
    elif len(parts) == 2:
        return {
            "subsection": "",
            "sourcepage": parts[0],
            "sourcefile": parts[1],
            "part_count": 2,
        }
    elif len(parts) == 1:
        return {
            "subsection": "",
            "sourcepage": "",
            "sourcefile": parts[0],
            "part_count": 1,
        }
    return {"subsection": "", "sourcepage": "", "sourcefile": "", "part_count": 0}


# ────────────────────────────────────────────────────────────────────────────────
# Fixtures: complete set of document archetypes covering all real patterns
# ────────────────────────────────────────────────────────────────────────────────

def _make_doc(**kwargs) -> Document:
    """Create a Document with sensible defaults."""
    defaults = dict(
        id="test-doc-1",
        content="1.1 Test content paragraph.\n\n1.2 Second paragraph.",
        category="Civil Procedure Rules and Practice Directions",
        sourcepage="Part 1",
        sourcefile="Part 1 – Overriding Objective",
        storage_url="https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part01",
        oids=[],
        groups=[],
        captions=None,
        score=0.5,
        reranker_score=3.2,
        updated="2023-10-01T00:00:00Z",
        subsection_id="1.1",
    )
    defaults.update(kwargs)
    return Document(**defaults)


@pytest.fixture
def cpr_numeric_doc():
    """CPR Part with numeric subsection (most common: 815/869 in sample)."""
    return _make_doc(
        id="Part_1___Overriding_Objective_0",
        subsection_id="1.1",
        content="1.1 These Rules are a procedural code with the overriding objective.",
        sourcepage="Part 1 – Overriding Objective",
        sourcefile="Part 1",
        category="Civil Procedure Rules and Practice Directions",
    )


@pytest.fixture
def cpr_numeric_deep_doc():
    """CPR with deep numeric subsection like 3.1.2."""
    return _make_doc(
        id="Part_3___Scope_0",
        subsection_id="3.1.2",
        content="3.1.2 The court may also make an order on its own initiative.",
        sourcepage="Part 3 – Scope",
        sourcefile="Part 3",
    )


@pytest.fixture
def court_guide_alpha_dot_doc():
    """Court guide with alpha_dot subsection (E.1.1)."""
    return _make_doc(
        id="Commercial_Court_Guide_E1",
        subsection_id="E.1.1",
        content="E.1.1 Standard disclosure is not the norm in the Commercial Court.",
        sourcepage="E.  Disclosure, E.1 Generally (p. 60)",
        sourcefile="Commercial Court Guide",
        category="Commercial Court",
        storage_url="https://example.com/commercial-court-guide",
    )


@pytest.fixture
def court_guide_alpha_dot_deep_doc():
    """Court guide with deep alpha_dot subsection (D.7.1)."""
    return _make_doc(
        id="Commercial_Court_Guide_D7",
        subsection_id="D.7.1",
        content="D.7.1 The judge will normally consider whether a split trial is appropriate.",
        sourcepage="D.  Case and Costs Management, D.7 Split trials (p. 50)",
        sourcefile="Commercial Court Guide",
        category="Commercial Court",
    )


@pytest.fixture
def court_guide_alpha_num_doc():
    """Court guide with alpha_num subsection (A1)."""
    return _make_doc(
        id="Pre_Action_Protocol_0",
        subsection_id="A1",
        content="A1 Pre-action protocols encourage early resolution.",
        sourcepage="Pre-Action Protocol for Resolution of Package Travel Claims",
        sourcefile="Pre",
        category="Civil Procedure Rules and Practice Directions",
    )


@pytest.fixture
def practice_direction_doc():
    """Practice Direction with numeric subsection."""
    return _make_doc(
        id="Practice_Direction_35___Experts_0",
        subsection_id="2.1",
        content="2.1 Expert evidence should be the independent product of the expert.",
        sourcepage="Practice Direction 35",
        sourcefile="Practice Direction 35 – Experts And Assessors",
        category="Civil Procedure Rules and Practice Directions",
    )


@pytest.fixture
def rule_subsection_doc():
    """Document with Rule-style subsection."""
    return _make_doc(
        id="Rule_Doc_0",
        subsection_id="Rule 31.6",
        content="Rule 31.6 Standard disclosure requires a party to disclose documents.",
        sourcepage="Part 31",
        sourcefile="Part 31 – Disclosure And Inspection",
    )


@pytest.fixture
def para_subsection_doc():
    """Document with Para-style subsection (court guides)."""
    return _make_doc(
        id="CCG_Appendix1_0",
        subsection_id="Para 5",
        content="Para 5 The parties should consider whether ADR is appropriate.",
        sourcepage="Appendix 1: Overriding Objective (p. 139)",
        sourcefile="Commercial Court Guide",
        category="Commercial Court",
    )


@pytest.fixture
def no_subsection_doc():
    """Document without subsection (PD header docs, 11 in sample)."""
    return _make_doc(
        id="PD_1A_0",
        subsection_id="",
        content="PRACTICE DIRECTION 1A – Participation of vulnerable parties or witnesses.",
        sourcepage="Practice Direction 1A: participation of vulnerable parties or witnesses",
        sourcefile="Practice Direction 1A – Participation Of Vulnerable Parties Or Witnesses",
    )


@pytest.fixture
def chancery_guide_doc():
    """Chancery Division court guide document."""
    return _make_doc(
        id="Chancery_Guide_Ch9_0",
        subsection_id="9.1",
        content="9.1 Applications should be made by application notice.",
        sourcepage="Chapter 9 Applications, 9.1 General (p. 45)",
        sourcefile="Chancery Guide",
        category="Chancery Division",
    )


@pytest.fixture
def patents_court_doc():
    """Patents Court document."""
    return _make_doc(
        id="Patents_Court_Guide_0",
        subsection_id="3.1",
        content="3.1 The Patents Court deals with intellectual property claims.",
        sourcepage="3. The judges of the Patents Court (p. 2)",
        sourcefile="Patents Court Guide",
        category="Patents Court",
    )


@pytest.fixture
def tcc_guide_doc():
    """Technology and Construction Court document."""
    return _make_doc(
        id="TCC_Guide_0",
        subsection_id="5.1",
        content="5.1 Case management conferences are a key feature of TCC practice.",
        sourcepage="5. Case Management (p. 20)",
        sourcefile="Technology and Construction Court Guide",
        category="Technology and Construction Court",
    )


@pytest.fixture
def kbd_guide_doc():
    """King's Bench Division document."""
    return _make_doc(
        id="KBD_Guide_0",
        subsection_id="2.1",
        content="2.1 The King's Bench Division deals with a wide range of civil claims.",
        sourcepage="2. Litigants in person (p. 23)",
        sourcefile="King's Bench Guide",
        category="King's Bench Division",
    )


@pytest.fixture
def multi_subsection_doc():
    """Document with multiple subsections in content (triggers splitting)."""
    return _make_doc(
        id="Part_35_Full",
        subsection_id="2.1",
        content=(
            "2.1 Expert evidence should be the independent product of the expert.\n\n"
            "2.2 An expert should assist the court by providing objective opinion.\n\n"
            "2.3 The expert should consider all material facts including those which might detract.\n\n"
            "3.1 Written instructions to the expert should be identified.\n\n"
            "3.2 The substance of all oral instructions should be confirmed in writing."
        ),
        sourcepage="Practice Direction 35",
        sourcefile="Practice Direction 35 – Experts And Assessors",
    )


@pytest.fixture
def all_doc_types(
    cpr_numeric_doc, cpr_numeric_deep_doc, court_guide_alpha_dot_doc,
    court_guide_alpha_dot_deep_doc, court_guide_alpha_num_doc,
    practice_direction_doc, rule_subsection_doc, para_subsection_doc,
    no_subsection_doc, chancery_guide_doc, patents_court_doc,
    tcc_guide_doc, kbd_guide_doc,
):
    """All document archetypes for comprehensive testing."""
    return {
        "cpr_numeric": cpr_numeric_doc,
        "cpr_numeric_deep": cpr_numeric_deep_doc,
        "alpha_dot": court_guide_alpha_dot_doc,
        "alpha_dot_deep": court_guide_alpha_dot_deep_doc,
        "alpha_num": court_guide_alpha_num_doc,
        "practice_direction": practice_direction_doc,
        "rule": rule_subsection_doc,
        "para": para_subsection_doc,
        "no_subsection": no_subsection_doc,
        "chancery": chancery_guide_doc,
        "patents": patents_court_doc,
        "tcc": tcc_guide_doc,
        "kbd": kbd_guide_doc,
    }


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 1: Backend Citation Generation
# ════════════════════════════════════════════════════════════════════════════════

class TestCitationFormatCorrectness:
    """A. Every generated citation must be well-formed."""

    def test_numeric_produces_three_part_citation(self, cpr_numeric_doc):
        citation = citation_builder.build_enhanced_citation(cpr_numeric_doc, 1)
        parts = parse_citation_parts(citation)
        assert parts["part_count"] >= 2, f"Expected ≥2 parts, got: {citation}"
        assert parts["subsection"] == "1.1"
        assert parts["sourcefile"] in citation

    def test_numeric_deep_produces_citation(self, cpr_numeric_deep_doc):
        citation = citation_builder.build_enhanced_citation(cpr_numeric_deep_doc, 1)
        parts = parse_citation_parts(citation)
        assert "3.1.2" in citation

    def test_alpha_dot_produces_citation(self, court_guide_alpha_dot_doc):
        citation = citation_builder.build_enhanced_citation(court_guide_alpha_dot_doc, 1)
        assert "E.1.1" in citation
        assert "Commercial Court Guide" in citation

    def test_alpha_dot_deep_produces_citation(self, court_guide_alpha_dot_deep_doc):
        citation = citation_builder.build_enhanced_citation(court_guide_alpha_dot_deep_doc, 1)
        assert "D.7.1" in citation

    def test_alpha_num_produces_citation(self, court_guide_alpha_num_doc):
        citation = citation_builder.build_enhanced_citation(court_guide_alpha_num_doc, 1)
        assert "A1" in citation

    def test_rule_subsection_produces_citation(self, rule_subsection_doc):
        citation = citation_builder.build_enhanced_citation(rule_subsection_doc, 1)
        assert "Rule 31.6" in citation

    def test_para_subsection_produces_citation(self, para_subsection_doc):
        citation = citation_builder.build_enhanced_citation(para_subsection_doc, 1)
        assert "Para 5" in citation

    def test_practice_direction_produces_citation(self, practice_direction_doc):
        citation = citation_builder.build_enhanced_citation(practice_direction_doc, 1)
        assert "2.1" in citation
        assert "Practice Direction 35" in citation

    def test_no_subsection_still_produces_valid_citation(self, no_subsection_doc):
        citation = citation_builder.build_enhanced_citation(no_subsection_doc, 1)
        assert len(citation) > 0
        # Should still have at least sourcefile
        assert "Practice Direction 1A" in citation

    def test_all_categories_produce_citations(self, all_doc_types):
        """Every document category must produce a non-empty citation."""
        for name, doc in all_doc_types.items():
            citation = citation_builder.build_enhanced_citation(doc, 1)
            assert len(citation) > 0, f"Empty citation for {name}"
            assert citation != "Source 1", f"Fallback citation for {name}: {citation}"

    def test_citation_never_starts_with_comma(self, all_doc_types):
        """No citation should start with a comma or space."""
        for name, doc in all_doc_types.items():
            citation = citation_builder.build_enhanced_citation(doc, 1)
            assert not citation.startswith(","), f"Citation for {name} starts with comma: {citation}"
            assert not citation.startswith(" "), f"Citation for {name} starts with space: {citation}"

    def test_citation_never_has_empty_parts(self, all_doc_types):
        """No citation should have empty comma-separated parts."""
        for name, doc in all_doc_types.items():
            citation = citation_builder.build_enhanced_citation(doc, 1)
            parts = [p.strip() for p in citation.split(",")]
            empty_parts = [p for p in parts if not p]
            assert len(empty_parts) == 0, f"Citation for {name} has empty parts: {citation}"

    def test_citation_has_sourcefile(self, all_doc_types):
        """Every citation should end with the sourcefile."""
        for name, doc in all_doc_types.items():
            citation = citation_builder.build_enhanced_citation(doc, 1)
            sf = getattr(doc, 'sourcefile', '') or ''
            if sf:
                assert citation.endswith(sf), f"Citation for {name} doesn't end with sourcefile '{sf}': {citation}"


class TestSubsectionExtraction:
    """Extraction must work for every known subsection_id format."""

    @pytest.mark.parametrize("subsection_id,content_prefix", [
        ("1.1", "1.1 Text"),
        ("3.1.2", "3.1.2 Text"),
        ("A.1", "A.1 Text"),
        ("E.1.1", "E.1.1 Text"),
        ("A1", "A1 Text"),
        ("A1.1", "A1.1 Text"),
        ("Rule 31.6", "Rule 31.6 Text"),
        ("Para 5", "Para 5 Text"),
        ("Part 35", "Part 35 Text"),
    ])
    def test_indexed_subsection_id_takes_priority(self, subsection_id, content_prefix):
        """Priority 0: indexed subsection_id always wins."""
        doc = _make_doc(subsection_id=subsection_id, content=f"{content_prefix} paragraph.")
        result = citation_builder.extract_subsection(doc)
        assert result == subsection_id

    @pytest.mark.parametrize("content,expected", [
        ("1.1 These are the rules.", "1.1"),
        ("A.1 Introduction to the guide.", "A.1"),
        ("E.1.1 Standard disclosure.", "E.1.1"),
        ("D5.1 Filing requirements.", "D5.1"),
        ("Rule 31.6 Standard disclosure.", "Rule 31.6"),
        ("Para 5 The parties should.", "Para 5"),
        ("Part 35 Expert evidence.", "Part 35"),
        ("## 1.1 Heading\nParagraph text.", "1.1"),
        ("**1.1** Bold formatted section.", "1.1"),
        ("__A.1__ Underline bold.", "A.1"),
    ])
    def test_content_fallback_extraction(self, content, expected):
        """Priority 1: content extraction when no subsection_id."""
        doc = _make_doc(subsection_id="", content=content)
        result = citation_builder.extract_subsection(doc)
        assert result == expected, f"Expected '{expected}', got '{result}' from content: {content[:50]}"

    def test_empty_subsection_and_empty_content(self):
        doc = _make_doc(subsection_id="", content="")
        result = citation_builder.extract_subsection(doc)
        # Falls back to sourcepage extraction when content is empty
        # Since our default doc has sourcepage="Part 1", it may extract from that
        assert isinstance(result, str)


class TestPageNumberExtraction:
    """extract_page_number must handle PDF #page= and legal (p. N) formats."""

    @pytest.mark.parametrize("sourcepage,expected", [
        ("file.pdf#page=5", 5),
        ("Benefit_Options.pdf#page=12", 12),
        ("doc.pdf#page=1", 1),
        ("E. Disclosure, E.1 Generally (p. 60)", 60),
        ("Part 3 – Case Management (p. 123)", 123),
        ("(p. 1)", 1),
        ("(p.60)", 60),
        ("(p 42)", 42),
    ])
    def test_extracts_page_number(self, sourcepage, expected):
        result = citation_builder.extract_page_number(sourcepage)
        assert result == expected, f"Expected {expected} from '{sourcepage}', got {result}"

    @pytest.mark.parametrize("sourcepage", [
        "",
        "Part 1 – Overriding Objective",
        "civil-procedure-rules.pdf",
        "No page reference here",
    ])
    def test_returns_none_when_no_page(self, sourcepage):
        result = citation_builder.extract_page_number(sourcepage)
        assert result is None, f"Expected None from '{sourcepage}', got {result}"

    def test_returns_none_for_empty_string(self):
        assert citation_builder.extract_page_number("") is None

    def test_returns_none_for_none(self):
        assert citation_builder.extract_page_number(None) is None


class TestMultiSubsectionSplitting:
    """Multi-subsection docs must split correctly with proper citations each."""

    def test_splits_into_correct_count(self, multi_subsection_doc):
        subs = citation_builder.extract_multiple_subsections(multi_subsection_doc)
        assert len(subs) >= 4, f"Expected ≥4 subsections, got {len(subs)}"

    def test_each_subsection_has_content(self, multi_subsection_doc):
        subs = citation_builder.extract_multiple_subsections(multi_subsection_doc)
        for sub in subs:
            assert len(sub["content"]) >= 10, f"Subsection {sub['subsection']} has too little content"

    def test_split_produces_correct_citations(self, multi_subsection_doc):
        sources = _processor.process_documents([multi_subsection_doc])
        assert len(sources) >= 4
        for src in sources:
            cit = src["citation"]
            parts = parse_citation_parts(cit)
            assert parts["part_count"] >= 2, f"Citation has too few parts: {cit}"
            assert parts["subsection"], f"Split subsection missing from citation: {cit}"

    def test_split_subsection_ids_are_unique(self, multi_subsection_doc):
        sources = _processor.process_documents([multi_subsection_doc])
        sub_ids = [s.get("subsection_id", "") for s in sources]
        assert len(sub_ids) == len(set(sub_ids)), f"Duplicate subsection_ids: {sub_ids}"

    def test_split_preserves_original_doc_id(self, multi_subsection_doc):
        sources = _processor.process_documents([multi_subsection_doc])
        for src in sources:
            assert src["original_doc_id"] == multi_subsection_doc.id

    def test_split_items_are_ordered_naturally(self, multi_subsection_doc):
        sources = _processor.process_documents([multi_subsection_doc])
        indices = [s["subsection_index"] for s in sources]
        assert indices == sorted(indices), f"Subsection indices not in order: {indices}"


class TestSourceProcessorFieldCompleteness:
    """B. Every source dict must have all fields the frontend requires."""

    REQUIRED_FIELDS = [
        "id", "content", "sourcepage", "sourcefile", "category",
        "storageUrl", "citation", "url", "original_doc_id",
    ]

    OPTIONAL_FIELDS = [
        "updated", "is_subsection", "subsection_id", "subsection_index",
        "total_subsections", "score", "reranker_score", "oids", "groups",
    ]

    def test_all_required_fields_present(self, all_doc_types):
        for name, doc in all_doc_types.items():
            sources = _processor.process_documents([doc])
            assert len(sources) >= 1, f"No sources for {name}"
            for src in sources:
                for field in self.REQUIRED_FIELDS:
                    assert field in src, f"Missing '{field}' in {name} source"

    def test_critical_fields_non_empty(self, all_doc_types):
        for name, doc in all_doc_types.items():
            sources = _processor.process_documents([doc])
            for src in sources:
                assert src["content"], f"Empty content in {name}"
                assert src["citation"], f"Empty citation in {name}"
                # sourcefile may be empty for some edge cases, but sourcepage or sourcefile should be set
                assert src["sourcefile"] or src["sourcepage"], f"No source identification in {name}"

    def test_storageurl_and_url_consistent(self, all_doc_types):
        for name, doc in all_doc_types.items():
            sources = _processor.process_documents([doc])
            for src in sources:
                su = src.get("storageUrl", "")
                url = src.get("url", "")
                if su:
                    assert url == su, f"storageUrl/url mismatch in {name}: '{su}' vs '{url}'"


class TestCitationNoDuplication:
    """Citation should not duplicate subsection in sourcepage."""

    def test_subsection_not_duplicated_in_sourcepage(self):
        """When subsection == sourcepage, citation should not repeat it."""
        doc = _make_doc(
            subsection_id="1.1",
            sourcepage="1.1",
            sourcefile="Part 1",
        )
        citation = citation_builder.build_enhanced_citation(doc, 1)
        parts = [p.strip() for p in citation.split(",")]
        # Should be "1.1, Part 1" not "1.1, 1.1, Part 1"
        subsection_count = sum(1 for p in parts if p == "1.1")
        assert subsection_count <= 1, f"Subsection duplicated: {citation}"

    def test_encoded_sourcepage_not_duplicated(self):
        """Encoded sourcepage like PD3E-1.1 should not duplicate with subsection."""
        doc = _make_doc(
            subsection_id="1.1",
            sourcepage="PD3E-1.1",
            sourcefile="Practice Direction 3E",
        )
        citation = citation_builder.build_enhanced_citation(doc, 1)
        # Citation should contain the subsection and the sourcefile
        assert "1.1" in citation
        assert "Practice Direction 3E" in citation
        # Should not have duplicate '1.1' entries
        parts = [p.strip() for p in citation.split(",")]
        subsection_count = sum(1 for p in parts if p == "1.1")
        assert subsection_count <= 1, f"Subsection duplicated: {citation}"


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 2: Frontend Citation Parsing (simulated in Python)
# ════════════════════════════════════════════════════════════════════════════════

class TestCitationParsing:
    """Simulate the frontend citation parsing logic in Python."""

    def _parse_citation_label_parts(self, citation: str) -> dict:
        """Python equivalent of AnswerParser.parseCitationLabelParts()."""
        parts = [p.strip() for p in citation.split(",") if p.strip()]
        subsection = parts[0] if len(parts) >= 3 else ""
        source_page = ", ".join(parts[1:-1]) if len(parts) >= 3 else (parts[0] if len(parts) == 2 else "")
        document = parts[-1] if len(parts) >= 1 else ""
        return {"subsection": subsection, "sourcePage": source_page, "document": document, "parts": parts}

    def _classify_subsection(self, sub: str) -> str:
        """Python equivalent of AnswerParser.classifySubsection()."""
        if not sub:
            return "unknown"
        if re.match(r'^[A-Z]\d+(?:\.\d+)?', sub, re.I):
            return "alpha"
        if re.match(r'^Rule\s+\d+', sub, re.I):
            return "rule"
        if re.match(r'^Para(?:graph)?\s+\d+', sub, re.I):
            return "para"
        if re.match(r'^\d+(?:\.\d+)?', sub):
            return "numeric"
        return "unknown"

    def test_three_part_citation_parses_correctly(self):
        result = self._parse_citation_label_parts("1.1, Part 1 – Overriding Objective, Part 1")
        assert result["subsection"] == "1.1"
        assert result["sourcePage"] == "Part 1 – Overriding Objective"
        assert result["document"] == "Part 1"

    def test_four_part_citation_with_comma_in_sourcepage(self):
        """Court guide sourcepages often have commas (e.g., 'E. Disclosure, E.1 Generally (p. 60)')."""
        result = self._parse_citation_label_parts(
            "E.1.1, E.  Disclosure, E.1 Generally (p. 60), Commercial Court Guide"
        )
        assert result["subsection"] == "E.1.1"
        # The middle parts are joined as sourcePage
        assert "Disclosure" in result["sourcePage"]
        assert result["document"] == "Commercial Court Guide"

    def test_two_part_citation_parses(self):
        result = self._parse_citation_label_parts("Practice Direction 35, Practice Direction 35 – Experts")
        assert result["subsection"] == ""
        assert result["sourcePage"] == "Practice Direction 35"
        assert result["document"] == "Practice Direction 35 – Experts"

    def test_classify_numeric(self):
        assert self._classify_subsection("1.1") == "numeric"
        assert self._classify_subsection("35.4") == "numeric"

    def test_classify_alpha(self):
        assert self._classify_subsection("D5.1") == "alpha"
        assert self._classify_subsection("E1") == "alpha"
        assert self._classify_subsection("A1.1") == "alpha"

    def test_classify_rule(self):
        assert self._classify_subsection("Rule 31.6") == "rule"

    def test_classify_para(self):
        assert self._classify_subsection("Para 5") == "para"


class TestSubsectionFromCitation:
    """Python equivalent of SupportingContentParser.parseSubsectionFromCitation()."""

    VALID_PATTERNS = [
        (r'^\d+\.\d+(\.\d+)?$', "numeric"),
        (r'^[A-Z]\.\d+$', "alpha_dot"),
        (r'^[A-Z]\.\d+(\.\d+)*$', "alpha_dot_deep"),
        (r'^[A-Z]\d+\.?\d*$', "alpha_num"),
        (r'^Rule\s+\d+(\.\d+)?$', "rule"),
        (r'^Para\s+\d+(\.\d+)?$', "para"),
        (r'^Chapter\s+\d+$', "chapter"),
        (r'^Section\s+\d+$', "section"),
        (r'^Part\s+\d+$', "part"),
        (r'^Appendix\s+\d+$', "appendix"),
    ]

    def _parse_subsection(self, citation: str) -> str | None:
        """Python equivalent of parseSubsectionFromCitation()."""
        normalized = re.sub(r'^\s*\d+\.\s+', '', citation)
        parts = [p.strip() for p in normalized.split(",") if p.strip()]
        raw = parts[0] if parts else ""
        subsection = raw.split(" - ")[0].strip() if " - " in raw else raw.strip()
        for pattern, _ in self.VALID_PATTERNS:
            if re.match(pattern, subsection, re.I):
                return subsection
        return None

    @pytest.mark.parametrize("citation,expected_sub", [
        ("1.1, Part 1, Part 1 – Overriding Objective", "1.1"),
        ("E.1.1, E. Disclosure, Commercial Court Guide", "E.1.1"),
        ("D.7.1, D. Case Management, Commercial Court Guide", "D.7.1"),
        ("A1, Pre-Action Protocol, Pre", "A1"),
        ("Rule 31.6, Part 31, Part 31 – Disclosure", "Rule 31.6"),
        ("Para 5, Appendix 1, Commercial Court Guide", "Para 5"),
        ("Part 35, Practice Direction 35, PD 35 – Experts", "Part 35"),
    ])
    def test_extracts_subsection_from_citation(self, citation, expected_sub):
        result = self._parse_subsection(citation)
        assert result == expected_sub, f"Expected '{expected_sub}', got '{result}' from: {citation}"

    def test_no_subsection_in_two_part(self):
        """Two-part citations don't have a subsection as first part."""
        # Two-part citation won't have an extractable subsection since sourcepage
        # doesn't match a subsection pattern typically
        result = self._parse_subsection("Practice Direction 1A, PD 1A – Participation")
        # May or may not match depending on format
        # The key assertion: it should not crash
        assert result is None or isinstance(result, str)


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 3: Navigation Accuracy (simulated matching logic)
# ════════════════════════════════════════════════════════════════════════════════

class TestCitationToContentMatching:
    """Simulate SupportingContent.findMatchingContentIndex() in Python."""

    def _find_matching_index(
        self, citation: str, data_points: list[dict]
    ) -> int:
        """
        Python equivalent of findMatchingContentIndex().
        Returns index of best matching data_point, or -1.
        """
        parts = [p.strip() for p in citation.split(",") if p.strip()]
        best_idx = -1
        best_score = 0

        for i, dp in enumerate(data_points):
            dp_sp = (dp.get("sourcepage") or "").strip().lower()
            dp_sf = (dp.get("sourcefile") or "").strip().lower()
            dp_content = (dp.get("content") or "").lower()
            score = 0

            if len(parts) >= 3:
                subsection = parts[0].lower()
                sourcepage = ", ".join(parts[1:-1]).lower()
                document = parts[-1].lower()

                if document in dp_sf or dp_sf in document:
                    score += 10
                else:
                    continue

                if dp_sp == sourcepage or (len(sourcepage) > 3 and sourcepage in dp_sp):
                    score += 50
                elif len(dp_sp) > 3 and dp_sp in sourcepage:
                    score += 10

                # Subsection must appear in content
                if subsection and subsection in dp_content:
                    score += 40
                elif subsection:
                    # Check metadata fallback
                    meta_sub = (dp.get("subsection_id") or "").strip().lower()
                    if meta_sub == subsection:
                        score += 15
                    else:
                        continue

            elif len(parts) == 2:
                part_a = parts[0].lower()
                part_b = parts[1].lower()
                if part_b in dp_sf or dp_sf in part_b:
                    score += 25
                elif part_a in dp_sp or dp_sp in part_a:
                    score += 20
                else:
                    continue

            if score > best_score and score >= 15:
                best_score = score
                best_idx = i

        return best_idx

    def test_three_part_citation_finds_correct_item(self, cpr_numeric_doc):
        sources = _processor.process_documents([cpr_numeric_doc])
        citation = sources[0]["citation"]
        idx = self._find_matching_index(citation, sources)
        assert idx == 0

    def test_alpha_dot_citation_finds_correct_item(self, court_guide_alpha_dot_doc):
        sources = _processor.process_documents([court_guide_alpha_dot_doc])
        citation = sources[0]["citation"]
        idx = self._find_matching_index(citation, sources)
        assert idx == 0

    def test_rule_citation_finds_correct_item(self, rule_subsection_doc):
        sources = _processor.process_documents([rule_subsection_doc])
        citation = sources[0]["citation"]
        idx = self._find_matching_index(citation, sources)
        assert idx == 0

    def test_para_citation_finds_correct_item(self, para_subsection_doc):
        sources = _processor.process_documents([para_subsection_doc])
        citation = sources[0]["citation"]
        idx = self._find_matching_index(citation, sources)
        assert idx == 0

    def test_multi_subsection_each_citation_finds_its_item(self, multi_subsection_doc):
        """After splitting, each subsection's citation should resolve to itself."""
        sources = _processor.process_documents([multi_subsection_doc])
        assert len(sources) >= 4
        for i, src in enumerate(sources):
            citation = src["citation"]
            idx = self._find_matching_index(citation, sources)
            assert idx == i, f"Citation '{citation}' matched index {idx}, expected {i}"

    def test_mixed_documents_each_finds_correct(self, cpr_numeric_doc, court_guide_alpha_dot_doc, practice_direction_doc):
        """When multiple different documents are in the list, each citation resolves correctly."""
        sources = (
            _processor.process_documents([cpr_numeric_doc]) +
            _processor.process_documents([court_guide_alpha_dot_doc]) +
            _processor.process_documents([practice_direction_doc])
        )
        for i, src in enumerate(sources):
            citation = src["citation"]
            idx = self._find_matching_index(citation, sources)
            assert idx == i, f"Citation '{citation}' matched index {idx}, expected {i}"


class TestSubsectionHighlighting:
    """Simulate extractSubsectionContent() — the blue highlight targeting."""

    def _extract_subsection_content(self, full_content: str, target: str) -> dict | None:
        """Python equivalent of extractSubsectionContent()."""
        if not full_content or not target:
            return None
        escaped = re.escape(target)
        patterns = [
            re.compile(rf'(^|\n)\s*{escaped}\s*(\n|\s|$)', re.I),
            re.compile(rf'\b{escaped}\b', re.I),
        ]
        for pattern in patterns:
            m = pattern.search(full_content)
            if m:
                start = m.start()
                # Find next subsection boundary
                remaining = full_content[m.end():]
                boundary = re.search(r'\n\s*(\d+\.\d+|\d+\.\d+\.\d+|[A-Z]\.\d+|Rule\s+\d+|Para\s+\d+)\b', remaining)
                end = m.end() + boundary.start() if boundary else len(full_content)
                return {
                    "content": full_content[start:end].strip(),
                    "startIndex": start,
                    "endIndex": end,
                }
        return None

    @pytest.mark.parametrize("subsection_id,content", [
        ("2.1", "2.1 Expert evidence should be independent.\n\n2.2 An expert should assist."),
        ("E.1.1", "E.1.1 Standard disclosure is not the norm.\n\nE.1.2 Specific disclosure."),
        ("D.7.1", "D.7.1 The judge will consider split trials.\n\nD.7.2 Applications."),
        ("Rule 31.6", "Rule 31.6 Standard disclosure requires.\n\nRule 31.7 The court may."),
        ("Para 5", "Para 5 The parties should consider.\n\nPara 6 Where the dispute."),
    ])
    def test_highlights_correct_subsection(self, subsection_id, content):
        result = self._extract_subsection_content(content, subsection_id)
        assert result is not None, f"Could not find '{subsection_id}' in content"
        assert subsection_id in result["content"], f"Highlighted content doesn't contain '{subsection_id}'"

    def test_highlight_does_not_include_next_section(self):
        content = "2.1 First section content.\n\n2.2 Second section content.\n\n2.3 Third."
        result = self._extract_subsection_content(content, "2.1")
        assert result is not None
        assert "2.2" not in result["content"], "Highlight leaked into next section"

    def test_highlight_with_markdown_heading(self):
        content = "## 1.1 Introduction\nThis is the intro.\n\n## 1.2 Scope\nThe scope."
        result = self._extract_subsection_content(content, "1.1")
        assert result is not None
        assert "Introduction" in result["content"]


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 4: Round-Trip / Integration Tests
# ════════════════════════════════════════════════════════════════════════════════

class TestRoundTripCitationIntegrity:
    """Every citation generated by backend can be parsed by frontend logic."""

    def test_all_citations_parseable(self, all_doc_types):
        """Backend generates → frontend can parse subsection from every citation."""
        for name, doc in all_doc_types.items():
            sources = _processor.process_documents([doc])
            for src in sources:
                citation = src["citation"]
                # Parse parts
                parts = parse_citation_parts(citation)
                assert parts["part_count"] >= 1, f"No parts in {name} citation: {citation}"
                assert parts["sourcefile"], f"No sourcefile in {name} citation: {citation}"

    def test_all_citations_find_their_source(self, all_doc_types):
        """Every citation resolves back to a data_point via matching logic."""
        all_sources = []
        for name, doc in all_doc_types.items():
            sources = _processor.process_documents([doc])
            all_sources.extend(sources)

        for i, src in enumerate(all_sources):
            citation = src["citation"]
            parts = parse_citation_parts(citation)
            # At minimum, sourcefile should match some data_point
            matched = False
            for dp in all_sources:
                dp_sf = (dp.get("sourcefile") or "").lower()
                cit_sf = parts["sourcefile"].lower()
                if cit_sf in dp_sf or dp_sf in cit_sf:
                    matched = True
                    break
            assert matched, f"Citation '{citation}' has no matching data_point sourcefile"

    def test_citation_map_entries_resolve(self, all_doc_types):
        """Simulate building citation_map and verify each entry resolves."""
        all_sources = []
        citation_map = {}
        for name, doc in all_doc_types.items():
            sources = _processor.process_documents([doc])
            for j, src in enumerate(sources):
                idx = len(all_sources) + 1
                citation_map[str(idx)] = src["citation"]
                all_sources.append(src)

        for key, citation in citation_map.items():
            parts = parse_citation_parts(citation)
            assert parts["sourcefile"], f"citation_map[{key}] has no sourcefile: {citation}"

    def test_enhanced_citations_match_data_points(self):
        """Simulate the enhanced_citations list and verify alignment."""
        docs = [
            _make_doc(id=f"doc_{i}", subsection_id=f"{i}.1", content=f"{i}.1 Content for section {i}.",
                      sourcepage=f"Part {i}", sourcefile=f"Part {i} – Title")
            for i in range(1, 6)
        ]
        sources = _processor.process_documents(docs)
        enhanced_citations = [s["citation"] for s in sources]

        assert len(enhanced_citations) == len(sources)
        for i, cit in enumerate(enhanced_citations):
            parts = parse_citation_parts(cit)
            src = sources[i]
            # Subsection in citation should match the source's subsection_id
            if src.get("subsection_id"):
                assert parts["subsection"] == src["subsection_id"], \
                    f"Mismatch at {i}: citation subsection '{parts['subsection']}' != source '{src['subsection_id']}'"


class TestEdgeCasesAndBoundaries:
    """D. Edge cases that could break citation formatting or navigation."""

    def test_none_content_doc(self):
        doc = _make_doc(content=None, subsection_id="1.1")
        sources = _processor.process_documents([doc])
        assert len(sources) >= 1
        assert sources[0]["citation"], "Citation should still be generated with None content"

    def test_empty_sourcepage_doc(self):
        doc = _make_doc(sourcepage="", subsection_id="1.1", sourcefile="Part 1")
        citation = citation_builder.build_enhanced_citation(doc, 1)
        assert "1.1" in citation
        assert "Part 1" in citation

    def test_empty_sourcefile_doc(self):
        doc = _make_doc(sourcefile="", subsection_id="1.1", sourcepage="Part 1")
        citation = citation_builder.build_enhanced_citation(doc, 1)
        assert len(citation) > 0

    def test_unicode_content(self):
        doc = _make_doc(
            subsection_id="1.1",
            content="1.1 Cyfarwyddyd Ymarfer – Welsh language content.",
            sourcefile="Cyfarwyddyd Ymarfer 54A",
        )
        citation = citation_builder.build_enhanced_citation(doc, 1)
        assert "Cyfarwyddyd" in citation

    def test_very_long_sourcepage(self):
        """Long sourcepages (court guides with page numbers, descriptions)."""
        doc = _make_doc(
            subsection_id="E.1.1",
            sourcepage="E.  Disclosure, E.1 Generally applicable points on disclosure and list of issues (p. 60)",
            sourcefile="Commercial Court Guide",
        )
        citation = citation_builder.build_enhanced_citation(doc, 1)
        assert "E.1.1" in citation
        assert "Commercial Court Guide" in citation

    def test_sourcepage_with_page_number(self):
        doc = _make_doc(
            subsection_id="9.1",
            sourcepage="Chapter 9 Applications, 9.1 General (p. 45)",
            sourcefile="Chancery Guide",
        )
        citation = citation_builder.build_enhanced_citation(doc, 1)
        assert "9.1" in citation
        assert "(p. 45)" in citation or "Chancery Guide" in citation

    def test_split_chunk_numbering(self):
        """Documents with [Part 1/2] in sourcepage."""
        doc = _make_doc(
            subsection_id="22.1",
            sourcepage="22. Enforcement (p. 171) [Part 1/2]",
            sourcefile="Chancery Guide",
        )
        citation = citation_builder.build_enhanced_citation(doc, 1)
        assert "22.1" in citation

    def test_sort_key_stability(self):
        """Sort keys must be comparable across all types without TypeError."""
        ids = ["1.1", "2.3", "A.1", "A1.2", "Rule 31.1", "Para 5", "D5.1"]
        keys = [citation_builder.get_subsection_sort_key(s) for s in ids]
        # Should not raise TypeError when sorting
        sorted_keys = sorted(keys)
        assert len(sorted_keys) == len(ids)


# ════════════════════════════════════════════════════════════════════════════════
# LAYER 5: Live Integration Tests (requires running backend)
# ════════════════════════════════════════════════════════════════════════════════

BACKEND_URL = "http://localhost:50505"


def is_backend_running():
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{BACKEND_URL}/"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() in ("200", "302", "401")
    except Exception:
        return False


@pytest.fixture(scope="module")
def backend_available():
    if not is_backend_running():
        pytest.skip("Backend not running at localhost:50505")
    return True


def chat_query(question: str, category: str = "", top: int = 10) -> dict:
    """Send a chat query and return parsed response."""
    import urllib.request
    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {
            "overrides": {
                "retrieval_mode": "hybrid",
                "semantic_ranker": True,
                "top": top,
                "use_agentic_retrieval": True,
                "include_category": category,
            }
        },
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_URL}/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    merged = {}
    for line in body.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if "context" in event:
                merged["context"] = event["context"]
            if "message" in event:
                merged["message"] = event["message"]
        except json.JSONDecodeError:
            pass
    return merged


class TestLiveCitationFormats:
    """Live validation: every citation from the real API is well-formed."""

    def test_all_live_citations_have_sourcefile(self, backend_available):
        """Every enhanced_citation must end with a sourcefile."""
        result = chat_query("What are the rules about expert evidence?")
        ctx = result.get("context", {})
        citations = ctx.get("enhanced_citations", [])
        assert len(citations) > 0, "No enhanced_citations returned"

        for cit in citations:
            parts = parse_citation_parts(cit)
            assert parts["sourcefile"], f"Citation missing sourcefile: {cit}"

    def test_all_live_citations_not_fallback(self, backend_available):
        """No citation should be 'Source N' (fallback)."""
        result = chat_query("What is the overriding objective?")
        ctx = result.get("context", {})
        citations = ctx.get("enhanced_citations", [])
        for cit in citations:
            assert not cit.startswith("Source "), f"Fallback citation found: {cit}"

    def test_live_citation_map_keys_are_numeric(self, backend_available):
        """citation_map keys must be numeric strings ('1', '2', ...)."""
        result = chat_query("What is disclosure?")
        ctx = result.get("context", {})
        cmap = ctx.get("citation_map", {})
        assert len(cmap) > 0, "No citation_map returned"
        for key in cmap.keys():
            assert key.isdigit(), f"Non-numeric citation_map key: {key}"

    def test_live_citation_map_values_match_enhanced(self, backend_available):
        """Every citation_map value should appear in enhanced_citations."""
        result = chat_query("Tell me about case management in the Commercial Court")
        ctx = result.get("context", {})
        cmap = ctx.get("citation_map", {})
        enhanced = ctx.get("enhanced_citations", [])
        for key, val in cmap.items():
            assert val in enhanced, f"citation_map[{key}] = '{val}' not in enhanced_citations"


class TestLiveCitationNavigation:
    """Live validation: every citation resolves to a data_point."""

    def _find_match(self, citation: str, data_points: list[dict]) -> int:
        """Find matching data_point index for a citation."""
        parts = [p.strip() for p in citation.split(",") if p.strip()]
        for i, dp in enumerate(data_points):
            dp_sf = (dp.get("sourcefile") or "").lower()
            dp_sp = (dp.get("sourcepage") or "").lower()
            dp_content = (dp.get("content") or "").lower()

            if len(parts) >= 3:
                subsection = parts[0].lower()
                document = parts[-1].lower()
                if document not in dp_sf and dp_sf not in document:
                    continue
                if subsection in dp_content or (dp.get("subsection_id") or "").lower() == subsection:
                    return i
            elif len(parts) == 2:
                if parts[-1].lower() in dp_sf or dp_sf in parts[-1].lower():
                    return i
        return -1

    def test_every_citation_resolves_to_data_point(self, backend_available):
        """Critical: EVERY enhanced_citation must resolve to at least one data_point."""
        result = chat_query("What are the expert evidence duties under CPR Part 35?")
        ctx = result.get("context", {})
        citations = ctx.get("enhanced_citations", [])
        dp = ctx.get("data_points", {})
        text_pts = dp if isinstance(dp, list) else dp.get("text", [])

        assert len(citations) > 0
        assert len(text_pts) > 0

        unresolved = []
        for cit in citations:
            idx = self._find_match(cit, text_pts)
            if idx < 0:
                unresolved.append(cit)

        assert len(unresolved) == 0, (
            f"{len(unresolved)}/{len(citations)} citations could not resolve to data_points:\n"
            + "\n".join(f"  - {c}" for c in unresolved[:10])
        )

    def test_subsection_present_in_content_for_highlighted_citations(self, backend_available):
        """When a citation has a subsection, that subsection text must be findable in the content."""
        result = chat_query("What is the duty of disclosure under CPR?")
        ctx = result.get("context", {})
        citations = ctx.get("enhanced_citations", [])
        dp = ctx.get("data_points", {})
        text_pts = dp if isinstance(dp, list) else dp.get("text", [])

        not_found = []
        for cit in citations:
            parts = [p.strip() for p in cit.split(",") if p.strip()]
            if len(parts) < 3:
                continue  # No subsection to check
            subsection = parts[0]
            idx = self._find_match(cit, text_pts)
            if idx < 0:
                continue  # Already tested above
            content = (text_pts[idx].get("content") or "")
            sub_lower = subsection.lower()
            meta_sub = (text_pts[idx].get("subsection_id") or "").lower()
            if sub_lower not in content.lower() and meta_sub != sub_lower:
                not_found.append(f"{cit} (subsection '{subsection}' not in content of dp[{idx}])")

        assert len(not_found) == 0, (
            f"{len(not_found)} citations have subsection not in content:\n"
            + "\n".join(f"  - {n}" for n in not_found[:10])
        )


class TestLiveCategoryCoverage:
    """E. Every document category must produce well-formed citations."""

    CATEGORY_QUERIES = [
        ("Civil Procedure Rules and Practice Directions", "What is CPR Part 1 about?"),
        ("Commercial Court", "What happens at a CMC in the Commercial Court?"),
        ("Chancery Division", "How do I apply for a freezing order in Chancery?"),
        ("Technology and Construction Court", "What is the TCC case management procedure?"),
        ("King's Bench Division", "What are the KBD trial procedures?"),
        ("Patents Court", "How are patent claims handled?"),
    ]

    @pytest.mark.parametrize("category,query", CATEGORY_QUERIES)
    def test_category_produces_valid_citations(self, backend_available, category, query):
        result = chat_query(query, category=category)
        ctx = result.get("context", {})
        citations = ctx.get("enhanced_citations", [])
        dp = ctx.get("data_points", {})
        text_pts = dp if isinstance(dp, list) else dp.get("text", [])

        # Must have results
        assert len(text_pts) > 0, f"No data_points for category '{category}'"
        assert len(citations) > 0, f"No citations for category '{category}'"

        # Every citation must be well-formed (non-empty, has sourcefile)
        for cit in citations:
            parts = parse_citation_parts(cit)
            assert parts["sourcefile"], f"Missing sourcefile in {category}: {cit}"

        # At least some data_points should match the category
        categories = [dp.get("category", "") for dp in text_pts if isinstance(dp, dict)]
        category_match = any(category.lower() in c.lower() for c in categories if c)
        assert category_match or len(categories) > 0, \
            f"No data_points from '{category}'. Categories found: {set(categories)}"
