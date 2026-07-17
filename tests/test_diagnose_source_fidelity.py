"""Read-only unit tests for the source-to-index fidelity diagnostic.

These tests do not touch Azure. They exercise the pure functions that separate
benign parser transformations from genuine substantive loss.
"""

from scripts.diagnose_source_fidelity import (
    classify_block,
    coverage_layers,
    embedding_chunk_diagnostics,
    extract_identifiers,
    provisional_cause,
    remediation_action,
    sibling_chunk_report,
    split_blocks,
    strip_breadcrumb,
)


def test_whitespace_and_markdown_do_not_reduce_substantive_coverage():
    source = "# Rule 3.1\n\nThe court **may** extend the time for compliance with any rule."
    indexed = "Rule 3.1 The court may extend the time for compliance with any rule."

    layers = coverage_layers(source, indexed)

    assert layers["substantive_coverage"] == 1.0
    assert layers["missing_identifier_count"] == 0
    assert layers["unmatched_substantive_blocks"] == []


def test_boilerplate_lowers_raw_but_not_substantive_coverage():
    legal = "The claimant must file the application notice at least three clear days before the hearing date."
    source = (
        f"{legal}\n\n"
        "© Crown copyright 2025 This publication is licensed under the terms of the "
        "Open Government Licence v3.0 visit nationalarchives.gov.uk for details.\n\n"
        "Back to top"
    )
    indexed = legal

    layers = coverage_layers(source, indexed)

    assert layers["raw_coverage"] < 1.0
    assert layers["substantive_coverage"] == 1.0
    assert "OGL_BOILERPLATE" in layers["excluded_block_counts"]
    assert "NAVIGATION" in layers["excluded_block_counts"]


def test_missing_substantive_paragraph_is_flagged_even_if_aggregate_is_high():
    kept = "The court may make an order for costs at any stage of the proceedings under this rule."
    dropped = "Where a party fails to comply with a rule the court may impose a sanction proportionate to the breach involved."
    source = f"{kept}\n\n{dropped}"
    indexed = kept  # second substantive paragraph absent from the index

    layers = coverage_layers(source, indexed)

    assert layers["substantive_coverage"] < 1.0
    assert any(
        "fails to comply" in block["snippet"] for block in layers["unmatched_substantive_blocks"]
    )


def test_breadcrumb_prefix_is_stripped_but_body_retained():
    block = "[PART 3 > 3.1] The court may extend the time for compliance."
    assert strip_breadcrumb(block) == "The court may extend the time for compliance."
    assert classify_block("[PART 3]") == "BREADCRUMB_ONLY"


def test_expanded_identifier_extraction_covers_nested_and_structural_labels():
    text = "See rule 3.1, paragraph 8A.1 and 3.1.2, Part 24, Practice Direction 51ZF, Annex 7 and Schedule 2."
    identifiers = extract_identifiers(text)

    assert "3.1" in identifiers
    assert "8A.1" in identifiers
    assert "3.1.2" in identifiers
    assert "Part 24" in identifiers
    assert any(i.lower() == "practice direction 51zf" for i in identifiers)
    assert any(i.lower() == "annex 7" for i in identifiers)
    assert any(i.lower() == "schedule 2" for i in identifiers)


def test_split_blocks_ignores_blank_separators():
    assert split_blocks("a\n\n\n  \n\nb") == ["a", "b"]


def test_sibling_chunk_report_detects_partial_sequence():
    documents = [
        {"id": "PD_27B_chunk_011", "parent_id": "PD_27B", "category": "CPR", "sourcefile": "Practice Direction 27B", "storageUrl": "https://x/27b"},
    ]

    report = sibling_chunk_report(documents, "Practice Direction 27B")

    assert report["matched_document_count"] == 1
    assert report["chunk_indices"] == [11]
    assert report["partial_chunk_sequence"] is True


def test_sibling_chunk_report_accepts_complete_sequence():
    documents = [
        {"id": "Part_3_chunk_000", "parent_id": "Part_3", "category": "CPR", "sourcefile": "Part 3"},
        {"id": "Part_3_chunk_001", "parent_id": "Part_3", "category": "CPR", "sourcefile": "Part 3"},
    ]

    report = sibling_chunk_report(documents, "Part 3")

    assert report["chunk_indices"] == [0, 1]
    assert report["partial_chunk_sequence"] is False
    assert report["metadata_variance"] is False


def test_provisional_cause_prefers_partial_index_over_source_changed():
    layers = coverage_layers("x", "y")
    sibling = {"partial_chunk_sequence": True}

    assert provisional_cause(layers, "", sibling) == "PARTIAL_INDEX"


def test_provisional_cause_reports_benign_transformation_when_only_boilerplate_differs():
    legal = "The court may make an order for costs at any stage of the proceedings under this rule at its discretion."
    layers = coverage_layers(f"{legal}\n\nBack to top", legal)
    sibling = {"partial_chunk_sequence": False}

    assert layers["substantive_coverage"] == 1.0
    assert provisional_cause(layers, "", sibling) == "BENIGN_TRANSFORMATION"


def test_provisional_cause_flags_metadata_mapping_from_reconciliation_issue():
    layers = coverage_layers("x", "y")
    sibling = {"partial_chunk_sequence": False}

    cause = provisional_cause(layers, "matched by URL or unique sourcefile; category/sourcefile differs", sibling)

    assert cause == "METADATA_MAPPING"


def test_remediation_action_keeps_partial_index_reindex_approval_gated():
    action = remediation_action("PARTIAL_INDEX", {"metadata_variance": False}, None)

    assert "source-scoped reindex" in action
    assert "approval" in action


def test_remediation_action_prioritizes_missing_identifiers_for_review():
    action = remediation_action(
        "NEEDS_REVIEW",
        {"metadata_variance": False},
        {"missing_identifier_count": 2},
    )

    assert "missing legal identifiers" in action


def test_embedding_chunk_diagnostics_reports_historical_and_token_limits():
    documents = [
        {
            "id": "short",
            "content": "Part 1.1 Scope",
            "sourcefile": "Part 1",
            "sourcepage": "1",
        },
        {
            "id": "long",
            "content": "rule " * 9000,
            "sourcefile": "Part 2",
            "sourcepage": "2",
        },
    ]

    report = embedding_chunk_diagnostics(documents)

    assert report["counts"]["document_count"] == 2
    assert report["counts"]["historical_character_truncated"] == 1
    assert report["counts"]["over_embedding_token_limit"] == 1
    assert report["examples"][0]["historical_tail_tokens_lost"] > 0
