"""Unit tests for legal SourceProcessor.

These tests target the custom structured source processing used by the backend
approach logic (subsection splitting, ordering, and metadata shaping).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock


def test_process_documents_splits_multi_subsection_document_in_order():
    from customizations.approaches.source_processor import SourceProcessor

    citation_builder = Mock()
    citation_builder.extract_multiple_subsections.return_value = [
        {"subsection": "2.1", "content": "Section 2.1 content"},
        {"subsection": "1.2", "content": "Section 1.2 content"},
    ]
    citation_builder.get_subsection_sort_key.side_effect = lambda s: tuple(int(p) for p in s.split("."))

    processor = SourceProcessor(citation_builder=citation_builder)

    doc = SimpleNamespace(
        id="doc1",
        content="irrelevant (splits come from citation_builder)",
        sourcepage="CPR Part 1#page=10",
        sourcefile="CPR Part 1",
        category="CPR",
        storage_url="https://storage/doc1",
        oids=["oid1"],
        groups=["group1"],
        score=1.23,
        reranker_score=4.56,
        updated="2026-01-11",
    )

    results = processor.process_documents([doc], use_semantic_captions=False)

    assert len(results) == 2
    assert results[0]["subsection_id"] == "1.2"
    assert results[1]["subsection_id"] == "2.1"

    first = results[0]
    assert first["is_subsection"] is True
    assert first["original_doc_id"] == "doc1"
    assert first["total_subsections"] == 2
    assert first["sourcepage"] == "CPR Part 1#page=10"
    assert first["sourcefile"] == "CPR Part 1"
    assert first["category"] == "CPR"
    assert first["storageUrl"] == "https://storage/doc1"
    assert first["citation"] == "1.2, CPR Part 1#page=10, CPR Part 1"


def test_process_documents_single_document_uses_citation_builder_and_metadata():
    from customizations.approaches.source_processor import SourceProcessor

    citation_builder = Mock()
    citation_builder.extract_multiple_subsections.return_value = []
    citation_builder.build_enhanced_citation.return_value = "1.1, p. 3, CPR Part 1"

    processor = SourceProcessor(citation_builder=citation_builder)

    doc = SimpleNamespace(
        id="doc2",
        content="1.1 Something\nBody",
        sourcepage="p. 3",
        sourcefile="CPR Part 1",
        category="",
        storage_url="",
    )

    results = processor.process_documents([doc], use_semantic_captions=False)

    assert len(results) == 1
    result = results[0]
    assert result["id"] == "doc2"
    assert result["is_subsection"] is False
    assert result["original_doc_id"] == "doc2"
    assert result["citation"] == "1.1, p. 3, CPR Part 1"

    citation_builder.build_enhanced_citation.assert_called_once()


def test_process_documents_includes_semantic_captions_and_summary():
    from customizations.approaches.source_processor import SourceProcessor

    citation_builder = Mock()
    citation_builder.extract_multiple_subsections.return_value = []
    citation_builder.build_enhanced_citation.return_value = "CPR"
    processor = SourceProcessor(citation_builder=citation_builder)

    captions = [
        SimpleNamespace(text="Caption one", highlights="hi", additional_properties={"k": "v"}),
        SimpleNamespace(text="Caption two", highlights="", additional_properties={}),
    ]
    doc = SimpleNamespace(
        id="doc3",
        content="Body",
        sourcepage="p. 1",
        sourcefile="Guide",
        captions=captions,
    )

    results = processor.process_documents([doc], use_semantic_captions=True)

    assert len(results) == 1
    result = results[0]
    assert "captions" in result
    assert len(result["captions"]) == 2
    assert result["captions"][0]["text"] == "Caption one"
    assert result["caption_summary"] == "Caption one . Caption two"


def test_focus_subsections_uses_query_hint_when_indexed_subsection_mismatches():
    """When the indexed subsection_id doesn't match any extracted header,
    the query_hint should be used to focus on the correct subsection."""
    from customizations.approaches.source_processor import SourceProcessor

    citation_builder = Mock()
    # Simulate a large CPR Part document with many subsections
    subsections = [{"subsection": f"31.{i}", "content": f"Rule 31.{i} text"} for i in range(1, 19)]
    citation_builder.extract_multiple_subsections.return_value = subsections
    citation_builder.get_subsection_sort_key.side_effect = lambda s: tuple(
        int(p) for p in s.split(".") if p.isdigit()
    )
    # subsection_id is "Part 1" (chunk label), doesn't match any "31.X"
    citation_builder.extract_subsection.return_value = "Part 1"

    processor = SourceProcessor(citation_builder=citation_builder)

    doc = SimpleNamespace(
        id="Part_31_chunk_000",
        content="Full Part 31 content...",
        sourcepage="Part 31 – Disclosure and Inspection of Documents",
        sourcefile="Part 31",
        category="Civil Procedure Rules and Practice Directions",
        subsection_id="Part 1",
    )

    # Without query_hint, falls back to first max_unfocused_subsections=4
    results_no_hint = processor.process_documents(
        [doc],
        focus_on_indexed_subsection=True,
        adjacent_subsections=1,
        max_unfocused_subsections=4,
    )
    subs_no_hint = [r["subsection_id"] for r in results_no_hint]
    assert subs_no_hint == ["31.1", "31.2", "31.3", "31.4"]

    # With query_hint mentioning 31.16, should focus around 31.16
    results_with_hint = processor.process_documents(
        [doc],
        focus_on_indexed_subsection=True,
        adjacent_subsections=1,
        max_unfocused_subsections=4,
        query_hint="CPR 31.16 pre-action disclosure before proceedings",
    )
    subs_with_hint = [r["subsection_id"] for r in results_with_hint]
    assert "31.16" in subs_with_hint
    assert "31.15" in subs_with_hint
    assert "31.17" in subs_with_hint
    # Should NOT include 31.1 (far from focus)
    assert "31.1" not in subs_with_hint
