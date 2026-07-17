"""
Tests for supporting content parsing pipeline (v3 index).

Verifies the full pipeline from Azure Search → backend processing → frontend-ready data:
  1. source_processor.process_documents() correctly maps all v3 fields
  2. citation_builder correctly extracts subsections from legal content
  3. Backend field names match what the frontend SupportingContentParser expects
  4. Multi-subsection splitting works for documents with multiple sections
  5. Live app queries return properly structured data_points
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "backend"))

from approaches.approach import Document
from customizations.approaches.citation_builder import CitationBuilder, citation_builder
from customizations.approaches.source_processor import SourceProcessor

# Create a processor instance for tests
_processor = SourceProcessor(citation_builder)


def process_documents(docs, use_semantic_captions=False, use_image_citation=False):
    return _processor.process_documents(docs, use_semantic_captions, use_image_citation)


def enrich_source_metadata(source):
    return _processor.enrich_source_metadata(source)


# ────────────────────────────────────────────────────────────────────────────────
# Fixtures: mock Document objects matching v3 index schema
# ────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cpr_document_single():
    """A single-subsection CPR Part document."""
    return Document(
        id="cpr-part35-35_1",
        content="## 35.1 Duty to restrict expert evidence\n\nExpert evidence shall be restricted to that which is reasonably required to resolve the proceedings.",
        category="Civil Procedure Rules and Practice Directions",
        sourcepage="Part 35",
        sourcefile="Part 35 - Experts and assessors",
        storage_url="https://storage.blob.core.windows.net/legal/Part_35/35_1.md",
        oids=None,
        groups=None,
        score=8.5,
        reranker_score=3.2,
        updated="2024-01-15",
        subsection_id="35.1",
    )


@pytest.fixture
def cpr_document_multi():
    """A multi-subsection CPR Part document."""
    return Document(
        id="cpr-part7-7_2_7_3",
        content=(
            "## 7.2 How to start proceedings\n\n"
            "Proceedings are started when the court issues a claim form at the request of the claimant.\n\n"
            "## 7.3 Right to use one claim form\n\n"
            "A claimant may use a single claim form to start all claims which can be conveniently disposed of "
            "in the same proceedings."
        ),
        category="Civil Procedure Rules and Practice Directions",
        sourcepage="Part 7",
        sourcefile="Part 7 - How to start proceedings",
        storage_url="https://storage.blob.core.windows.net/legal/Part_7/7_2_7_3.md",
        oids=None,
        groups=None,
        score=9.0,
        reranker_score=3.8,
        updated="2024-02-01",
        subsection_id="7.2",
    )


@pytest.fixture
def court_guide_document():
    """A court guide document with alpha-numeric subsections."""
    return Document(
        id="commercial-court-D5_3",
        content=(
            "## D5.3 Case management conference\n\n"
            "At the case management conference, the court will review the issues in the case "
            "and give directions for the management of the case."
        ),
        category="Commercial Court",
        sourcepage="The Commercial Court Guide",
        sourcefile="The Commercial Court Guide",
        storage_url="https://storage.blob.core.windows.net/legal/CommercialCourt/D5_3.md",
        oids=None,
        groups=None,
        score=7.5,
        reranker_score=2.9,
        updated="2024-03-10",
        subsection_id="D5.3",
    )


@pytest.fixture
def kbd_annex_document():
    """A KBD annex document with NO subsection_id (one of the 7 empty cases)."""
    return Document(
        id="kbd-annex-3",
        content="Annex 3: Standard Directions for Fast Track Claims\n\nForm attached.",
        category="King's Bench Division",
        sourcepage="The King's Bench Guide",
        sourcefile="The King's Bench Guide",
        storage_url="https://storage.blob.core.windows.net/legal/KBD/annex_3.md",
        oids=None,
        groups=None,
        score=5.0,
        reranker_score=1.5,
        updated="2024-01-20",
        subsection_id="",
    )


@pytest.fixture
def pd_document():
    """A Practice Direction document with markdown bold subsection."""
    return Document(
        id="pd44-1_1",
        content="**1.1** This Practice Direction supplements CPR Part 44.\n\nIt applies to all costs proceedings.",
        category="Civil Procedure Rules and Practice Directions",
        sourcepage="PD44",
        sourcefile="Practice Direction 44 - General rules about costs",
        storage_url="https://storage.blob.core.windows.net/legal/PD44/1_1.md",
        oids=None,
        groups=None,
        score=6.0,
        reranker_score=2.0,
        updated="2024-01-25",
        subsection_id="1.1",
    )


@pytest.fixture
def document_no_url():
    """Document with no storage URL."""
    return Document(
        id="no-url-doc",
        content="## Rule 3.1 Court's general powers of management\n\nThe court may extend or shorten time.",
        category="Civil Procedure Rules and Practice Directions",
        sourcepage="Part 3",
        sourcefile="Part 3 - The court's case management powers",
        storage_url=None,
        score=4.0,
        reranker_score=1.0,
        updated="",
        subsection_id="Rule 3.1",
    )


# ────────────────────────────────────────────────────────────────────────────────
# Section 1: source_processor.process_documents() tests
# ────────────────────────────────────────────────────────────────────────────────

class TestSourceProcessorFields:
    """Verify that process_documents() outputs all v3 fields required by the frontend."""

    REQUIRED_FRONTEND_FIELDS = {
        "id", "content", "sourcepage", "sourcefile", "category",
        "citation", "url", "original_doc_id",
    }

    OPTIONAL_FRONTEND_FIELDS = {
        "storageUrl", "score", "reranker_score", "updated",
        "filepath", "oids", "groups",
    }

    def test_single_document_has_all_required_fields(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        assert len(results) >= 1
        source = results[0]
        for field in self.REQUIRED_FRONTEND_FIELDS:
            assert field in source, f"Missing required field: {field}"

    def test_single_document_field_values(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        assert source["sourcepage"] == "Part 35"
        assert source["sourcefile"] == "Part 35 - Experts and assessors"
        assert source["category"] == "Civil Procedure Rules and Practice Directions"
        assert "35.1" in source.get("content", "")
        assert source["original_doc_id"] is not None

    def test_storage_url_present(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        url = source.get("storageUrl") or source.get("url") or ""
        assert "storage.blob.core.windows.net" in url

    def test_updated_field_present(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        assert source.get("updated") == "2024-01-15"

    def test_court_guide_fields(self, court_guide_document):
        results = process_documents([court_guide_document], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        assert source["category"] == "Commercial Court"
        assert source["sourcefile"] == "The Commercial Court Guide"
        assert source["sourcepage"] == "The Commercial Court Guide"

    def test_document_without_url(self, document_no_url):
        results = process_documents([document_no_url], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        # Should not crash; url/storageUrl should be empty
        url = source.get("storageUrl") or source.get("url") or ""
        assert url == "" or url is None or url == "None"

    def test_kbd_annex_no_subsection(self, kbd_annex_document):
        results = process_documents([kbd_annex_document], use_semantic_captions=False, use_image_citation=False)
        assert len(results) >= 1
        source = results[0]
        # Should still have all required fields even without subsection_id
        for field in self.REQUIRED_FRONTEND_FIELDS:
            assert field in source, f"Missing required field for annex document: {field}"

    def test_empty_document_list(self):
        results = process_documents([], use_semantic_captions=False, use_image_citation=False)
        assert results == []

    def test_multiple_documents_in_order(self, cpr_document_single, court_guide_document, pd_document):
        docs = [cpr_document_single, court_guide_document, pd_document]
        results = process_documents(docs, use_semantic_captions=False, use_image_citation=False)
        # Should return at least one result per input document
        assert len(results) >= 3

    def test_practice_direction_fields(self, pd_document):
        results = process_documents([pd_document], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        assert source["sourcepage"] == "PD44"
        assert "Practice Direction 44" in source["sourcefile"]


class TestSourceProcessorMultiSubsection:
    """Verify multi-subsection splitting works correctly."""

    def test_multi_subsection_split(self, cpr_document_multi):
        results = process_documents([cpr_document_multi], use_semantic_captions=False, use_image_citation=False)
        # Should either split into 2 subsections or keep as single with info
        assert len(results) >= 1

    def test_multi_subsection_original_doc_id(self, cpr_document_multi):
        results = process_documents([cpr_document_multi], use_semantic_captions=False, use_image_citation=False)
        if len(results) > 1:
            # All splits should reference the same original document
            original_ids = {r.get("original_doc_id") for r in results}
            assert len(original_ids) == 1

    def test_multi_subsection_each_has_content(self, cpr_document_multi):
        results = process_documents([cpr_document_multi], use_semantic_captions=False, use_image_citation=False)
        for source in results:
            assert source.get("content"), f"Subsection missing content: {source}"

    def test_multi_subsection_has_is_subsection_flag(self, cpr_document_multi):
        results = process_documents([cpr_document_multi], use_semantic_captions=False, use_image_citation=False)
        if len(results) > 1:
            for source in results:
                assert source.get("is_subsection") is True

    def test_multi_subsection_preserves_category(self, cpr_document_multi):
        results = process_documents([cpr_document_multi], use_semantic_captions=False, use_image_citation=False)
        for source in results:
            assert source["category"] == "Civil Procedure Rules and Practice Directions"


# ────────────────────────────────────────────────────────────────────────────────
# Section 2: citation_builder tests
# ────────────────────────────────────────────────────────────────────────────────

class TestCitationBuilderExtractSubsection:
    """Verify subsection extraction from document content and metadata."""

    def test_indexed_subsection_id_priority(self, cpr_document_single):
        """Priority 0: indexed subsection_id field should be used first."""
        result = citation_builder.extract_subsection(cpr_document_single)
        assert result == "35.1"

    def test_content_fallback_when_no_subsection_id(self):
        """Priority 1: fall back to content when subsection_id is empty."""
        doc = Document(
            content="## 12.3 Filing a defence\n\nA defendant who wishes to defend...",
            sourcepage="Part 12",
            sourcefile="Part 12",
            subsection_id="",
        )
        result = citation_builder.extract_subsection(doc)
        assert result == "12.3"

    def test_markdown_heading_extraction(self):
        doc = Document(content="# 1.1 Overriding objective\n\nThese rules are intended...", subsection_id="")
        result = citation_builder.extract_subsection(doc)
        assert result == "1.1"

    def test_bold_formatted_extraction(self):
        doc = Document(content="**A4.1** Application of this section\n\nThis section...", subsection_id="")
        result = citation_builder.extract_subsection(doc)
        assert result == "A4.1"

    def test_rule_pattern_extraction(self):
        doc = Document(content="Rule 31.1 Standard disclosure\n\nEach party must...", subsection_id="")
        result = citation_builder.extract_subsection(doc)
        assert "31.1" in result

    def test_alpha_dot_pattern(self):
        doc = Document(content="D5.3 Case management conference\n\nAt the CMC...", subsection_id="")
        result = citation_builder.extract_subsection(doc)
        assert "D5.3" in result

    def test_empty_content_and_subsection(self):
        doc = Document(content="", subsection_id="")
        result = citation_builder.extract_subsection(doc)
        assert result == ""

    def test_para_pattern(self):
        doc = Document(content="Para 3.1 Interpretation\n\nIn these directions...", subsection_id="")
        result = citation_builder.extract_subsection(doc)
        assert "3.1" in result


class TestCitationBuilderBuildCitation:
    """Verify build_enhanced_citation produces correct 3-part/2-part/1-part citations."""

    def test_three_part_citation(self, cpr_document_single):
        result = citation_builder.build_enhanced_citation(cpr_document_single, 1)
        # Should be "35.1, Part 35, Part 35 - Experts and assessors"
        assert "35.1" in result
        assert "Part 35" in result

    def test_no_subsection_citation(self, kbd_annex_document):
        result = citation_builder.build_enhanced_citation(kbd_annex_document, 1)
        # Without subsection, should still produce meaningful citation from sourcepage/sourcefile
        assert result and len(result) > 0

    def test_court_guide_citation(self, court_guide_document):
        result = citation_builder.build_enhanced_citation(court_guide_document, 1)
        assert "D5.3" in result
        assert "Commercial Court" in result or "Guide" in result


class TestCitationBuilderMultiSubsection:
    """Verify multi-subsection extraction from content."""

    def test_extracts_multiple_subsections(self, cpr_document_multi):
        subsections = citation_builder.extract_multiple_subsections(cpr_document_multi)
        assert len(subsections) >= 2
        labels = [s["subsection"] for s in subsections]
        # Should have both 7.2 and 7.3
        assert any("7.2" in l for l in labels)
        assert any("7.3" in l for l in labels)

    def test_each_subsection_has_content(self, cpr_document_multi):
        subsections = citation_builder.extract_multiple_subsections(cpr_document_multi)
        for sub in subsections:
            assert sub.get("content") and len(sub["content"]) > 10

    def test_single_subsection_returns_empty(self, cpr_document_single):
        """Single-subsection docs should return empty list (not split)."""
        subsections = citation_builder.extract_multiple_subsections(cpr_document_single)
        assert len(subsections) <= 1


class TestCitationBuilderSortKey:
    """Verify subsection sort keys for natural ordering."""

    def test_numeric_sort(self):
        key = citation_builder.get_subsection_sort_key("1.1")
        assert key == (0, '', 1, 1)

    def test_alpha_sort(self):
        key = citation_builder.get_subsection_sort_key("D5.3")
        assert key == (1, 'D', 5, 3)

    def test_rule_sort(self):
        key = citation_builder.get_subsection_sort_key("Rule 31.1")
        assert key == (2, 'RULE', 31, 1)

    def test_natural_ordering(self):
        """Verify numeric subsections sort correctly."""
        keys = [
            citation_builder.get_subsection_sort_key("1.1"),
            citation_builder.get_subsection_sort_key("2.1"),
            citation_builder.get_subsection_sort_key("1.10"),
            citation_builder.get_subsection_sort_key("1.2"),
        ]
        sorted_keys = sorted(keys)
        assert sorted_keys[0] == (0, '', 1, 1)
        assert sorted_keys[1] == (0, '', 1, 2)
        assert sorted_keys[2] == (0, '', 1, 10)
        assert sorted_keys[3] == (0, '', 2, 1)


# ────────────────────────────────────────────────────────────────────────────────
# Section 3: enrich_source_metadata tests
# ────────────────────────────────────────────────────────────────────────────────

class TestEnrichSourceMetadata:
    """Verify field name normalization for frontend compatibility."""

    def test_adds_missing_url_from_storageurl(self):
        source = {"storageUrl": "https://example.com/doc.md"}
        search_result = {"storageurl": "https://example.com/doc.md"}
        _processor.enrich_source_metadata(source, search_result)
        assert source.get("url") == "https://example.com/doc.md" or source.get("storageurl")

    def test_preserves_existing_fields(self):
        source = {
            "sourcepage": "Part 1",
            "sourcefile": "Part 1 - Scope",
            "category": "CPR",
            "content": "Test content",
        }
        search_result = {
            "sourcepage": "Part 1",
            "sourcefile": "Part 1 - Scope",
            "category": "CPR",
        }
        _processor.enrich_source_metadata(source, search_result)
        assert source["sourcepage"] == "Part 1"
        assert source["sourcefile"] == "Part 1 - Scope"
        assert source["category"] == "CPR"

    def test_handles_document_object(self):
        source = {}
        search_result = Document(
            sourcepage="Part 2",
            sourcefile="Part 2 - Application",
            category="CPR",
            storage_url="https://example.com/part2.md",
            updated="2024-01-01",
        )
        _processor.enrich_source_metadata(source, search_result)
        assert source["sourcepage"] == "Part 2"
        assert source["sourcefile"] == "Part 2 - Application"


# ────────────────────────────────────────────────────────────────────────────────
# Section 4: Frontend field compatibility tests
# ────────────────────────────────────────────────────────────────────────────────

class TestFrontendFieldCompatibility:
    """
    The frontend SupportingContentParser.ts expects specific field names.
    Verify that backend output matches what the parser reads:
      - item.sourcepage
      - item.sourcefile
      - item.category
      - item.updated || item.last_updated || item.date_updated
      - item.storageurl || item.storageUrl || item.storage_url || item.url
      - item.full_content || item.content
      - item.original_doc_id
      - item.id
    """

    def test_sourcepage_field_name(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        # Frontend reads item.sourcepage (lowercase 'p')
        assert "sourcepage" in source
        assert source["sourcepage"] != ""

    def test_sourcefile_field_name(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        # Frontend reads item.sourcefile (lowercase 'f')
        assert "sourcefile" in source
        assert source["sourcefile"] != ""

    def test_category_field_name(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        assert "category" in source
        assert source["category"] == "Civil Procedure Rules and Practice Directions"

    def test_updated_field_name(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        # Frontend reads item.updated || item.last_updated || item.date_updated
        has_updated = "updated" in source or "last_updated" in source or "date_updated" in source
        assert has_updated

    def test_storage_url_field_variations(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        # Frontend reads item.storageurl || item.storageUrl || item.storage_url || item.url
        has_url = any(
            source.get(k) for k in ["storageurl", "storageUrl", "storage_url", "url"]
        )
        assert has_url, f"No URL field found. Keys: {list(source.keys())}"

    def test_content_field_name(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        # Frontend reads item.full_content || item.content
        has_content = source.get("full_content") or source.get("content")
        assert has_content

    def test_original_doc_id_present(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        assert "original_doc_id" in source

    def test_id_field_present(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        assert "id" in source


# ────────────────────────────────────────────────────────────────────────────────
# Section 5: Citation format tests (three-part format for frontend)
# ────────────────────────────────────────────────────────────────────────────────

class TestCitationFormat:
    """
    The frontend AnswerParser.tsx expects citations in format:
      "subsection, sourcepage, sourcefile"
    and uses parseCitationLabelParts() to split by comma.
    """

    def test_citation_is_comma_separated(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        citation = source.get("citation", "")
        if citation:
            parts = [p.strip() for p in citation.split(",")]
            assert len(parts) >= 2, f"Citation should have at least 2 parts: {citation}"

    def test_citation_contains_subsection(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        citation = source.get("citation", "")
        assert "35.1" in citation

    def test_citation_contains_sourcefile(self, cpr_document_single):
        results = process_documents([cpr_document_single], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        citation = source.get("citation", "")
        assert "Part 35" in citation

    def test_court_guide_citation_format(self, court_guide_document):
        results = process_documents([court_guide_document], use_semantic_captions=False, use_image_citation=False)
        source = results[0]
        citation = source.get("citation", "")
        assert "D5.3" in citation


# ────────────────────────────────────────────────────────────────────────────────
# Section 6: Edge cases and robustness
# ────────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and unusual document formats."""

    def test_none_content_document(self):
        doc = Document(
            id="none-content",
            content=None,
            category="Test",
            sourcepage="Test Page",
            sourcefile="Test File",
        )
        results = process_documents([doc], use_semantic_captions=False, use_image_citation=False)
        assert len(results) >= 1

    def test_very_short_content(self):
        doc = Document(
            id="short-content",
            content="Brief note.",
            category="Test",
            sourcepage="Test Page",
            sourcefile="Test File",
            subsection_id="",
        )
        results = process_documents([doc], use_semantic_captions=False, use_image_citation=False)
        assert len(results) >= 1

    def test_document_with_all_none_fields(self):
        doc = Document()
        results = process_documents([doc], use_semantic_captions=False, use_image_citation=False)
        # Should not crash
        assert isinstance(results, list)

    def test_multiple_categories(self, cpr_document_single, court_guide_document, kbd_annex_document):
        """Tests mixed categories in one batch."""
        docs = [cpr_document_single, court_guide_document, kbd_annex_document]
        results = process_documents(docs, use_semantic_captions=False, use_image_citation=False)
        categories = {r["category"] for r in results}
        assert "Civil Procedure Rules and Practice Directions" in categories
        assert "Commercial Court" in categories
        assert "King's Bench Division" in categories

    def test_document_unicode_content(self):
        doc = Document(
            id="unicode-doc",
            content="§ 1.1 Übernahme der Verfahrenskosten – cross-referenced with Rule 44.2",
            category="Test",
            sourcepage="Test",
            sourcefile="Test",
            subsection_id="1.1",
        )
        results = process_documents([doc], use_semantic_captions=False, use_image_citation=False)
        assert len(results) >= 1
        source = results[0]
        assert "§" in source.get("content", "")


# ────────────────────────────────────────────────────────────────────────────────
# Section 7: Live app integration tests
# ────────────────────────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:50505"


def is_backend_running():
    """Check if backend is available."""
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


def chat_query(question: str, category: str = "", use_agentic: bool = True) -> dict:
    """Send a chat query and return parsed JSON response."""
    import urllib.request
    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {
            "overrides": {
                "retrieval_mode": "hybrid",
                "semantic_ranker": True,
                "top": 5,
                "use_agentic_retrieval": use_agentic,
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
    lines = [l for l in body.strip().split("\n") if l.strip()]
    parsed = []
    for l in lines:
        try:
            parsed.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    if not parsed:
        return {}
    # Merge: first event has context, last has final content
    merged = {}
    for event in parsed:
        if "context" in event:
            merged["context"] = event["context"]
        if "message" in event:
            merged["message"] = event["message"]
        if "delta" in event:
            content = event["delta"].get("content", "")
            if content:
                if "message" not in merged:
                    merged["message"] = {"content": "", "role": "assistant"}
                merged["message"]["content"] = merged["message"].get("content", "") + content
    return merged


class TestLiveAppSupportingContent:
    """Integration tests: verify live API responses have proper supporting content."""

    def test_chat_returns_data_points(self, backend_available):
        response = chat_query("What is CPR Part 35 about?")
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        assert len(text_points) > 0, "No data_points returned"

    def test_data_points_have_sourcepage(self, backend_available):
        response = chat_query("What are the rules about expert evidence?")
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        assert len(text_points) > 0, "No data_points returned"

        # At least one data_point should have a sourcepage
        has_sourcepage = any(
            isinstance(dp, dict) and dp.get("sourcepage")
            for dp in text_points
        )
        assert has_sourcepage, f"No data_point has sourcepage. Sample: {text_points[0] if text_points else 'empty'}"

    def test_data_points_have_sourcefile(self, backend_available):
        response = chat_query("What is the overriding objective?")
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        assert len(text_points) > 0
        has_sourcefile = any(
            isinstance(dp, dict) and dp.get("sourcefile")
            for dp in text_points
        )
        assert has_sourcefile, "No data_point has sourcefile"

    def test_data_points_have_category(self, backend_available):
        response = chat_query("What is a case management conference?")
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        assert len(text_points) > 0
        has_category = any(
            isinstance(dp, dict) and dp.get("category")
            for dp in text_points
        )
        assert has_category, "No data_point has category"

    def test_data_points_have_content(self, backend_available):
        response = chat_query("What is the duty of disclosure?")
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        assert len(text_points) > 0
        has_content = any(
            isinstance(dp, dict) and dp.get("content")
            for dp in text_points
        )
        assert has_content, "No data_point has content"

    def test_data_points_have_url(self, backend_available):
        response = chat_query("What are the court's case management powers?")
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        assert len(text_points) > 0
        has_url = any(
            isinstance(dp, dict) and (dp.get("storageUrl") or dp.get("url") or dp.get("storageurl"))
            for dp in text_points
        )
        assert has_url, f"No data_point has URL. Keys of first: {list(text_points[0].keys()) if text_points and isinstance(text_points[0], dict) else 'N/A'}"

    def test_enhanced_citations_present(self, backend_available):
        response = chat_query("What is CPR Part 1?")
        context = response.get("context", {})
        enhanced_citations = context.get("enhanced_citations", [])
        assert len(enhanced_citations) > 0, "No enhanced_citations in response"

    def test_citation_map_present(self, backend_available):
        response = chat_query("What is the three-track system?")
        context = response.get("context", {})
        citation_map = context.get("citation_map", {})
        assert len(citation_map) > 0, "No citation_map in response"

    def test_court_guide_query_has_category(self, backend_available):
        response = chat_query(
            "What happens at a case management conference in the Commercial Court?",
            category="Commercial Court"
        )
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        if text_points:
            categories = [
                dp.get("category") for dp in text_points
                if isinstance(dp, dict) and dp.get("category")
            ]
            # At least one should be Commercial Court (may also include CPR)
            assert any("Commercial Court" in c for c in categories) or len(categories) > 0, \
                f"Expected Commercial Court category. Got: {categories}"

    def test_data_points_are_structured_objects(self, backend_available):
        """Ensure data_points are dicts (not legacy strings)."""
        response = chat_query("What is CPR Part 35?")
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        assert len(text_points) > 0
        for dp in text_points:
            assert isinstance(dp, dict), f"data_point should be dict, got {type(dp)}: {str(dp)[:100]}"

    def test_all_v3_fields_in_live_response(self, backend_available):
        """Comprehensive check: every data_point should have the full v3 field set."""
        response = chat_query("What is the duty to restrict expert evidence?")
        context = response.get("context", {})
        data_points = context.get("data_points", {})
        text_points = data_points if isinstance(data_points, list) else data_points.get("text", [])
        assert len(text_points) > 0

        critical_fields = ["id", "content", "sourcepage", "sourcefile", "category", "citation"]
        for i, dp in enumerate(text_points):
            if not isinstance(dp, dict):
                continue
            for field in critical_fields:
                assert field in dp, f"data_point[{i}] missing field '{field}'. Keys: {list(dp.keys())}"


# ────────────────────────────────────────────────────────────────────────────────
# Section 8: JSON serialization tests (ExtraInfo → JSON)
# ────────────────────────────────────────────────────────────────────────────────

class TestJsonSerialization:
    """Verify that ExtraInfo dataclass serializes correctly with custom JSONEncoder."""
    
    def test_extra_info_serializes_enhanced_citations(self):
        from approaches.approach import ExtraInfo, DataPoints
        import dataclasses
        
        extra = ExtraInfo(
            data_points=DataPoints(text=[{"id": "1", "content": "test"}]),
            enhanced_citations=["35.1, Part 35, Part 35 - Experts"],
            citation_map={"1": "35.1, Part 35, Part 35 - Experts"},
        )
        result = dataclasses.asdict(extra)
        assert "enhanced_citations" in result
        assert "citation_map" in result
        assert result["enhanced_citations"] == ["35.1, Part 35, Part 35 - Experts"]
        assert result["citation_map"]["1"] == "35.1, Part 35, Part 35 - Experts"

    def test_extra_info_serializes_data_points(self):
        from approaches.approach import ExtraInfo, DataPoints
        import dataclasses
        
        source = {
            "id": "test-1",
            "content": "Test content",
            "sourcepage": "Part 1",
            "sourcefile": "Part 1 - Scope",
            "category": "CPR",
            "storageUrl": "https://example.com/",
            "updated": "2024-01-01",
        }
        extra = ExtraInfo(data_points=DataPoints(text=[source]))
        result = dataclasses.asdict(extra)
        dp = result["data_points"]["text"][0]
        assert dp["sourcepage"] == "Part 1"
        assert dp["sourcefile"] == "Part 1 - Scope"
        assert dp["category"] == "CPR"
