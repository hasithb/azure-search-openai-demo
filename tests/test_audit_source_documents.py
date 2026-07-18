import json
from unittest.mock import patch

from scripts.audit_source_documents import (
    CanonicalSource,
    HtmlAuditCache,
    apply_html_fidelity,
    apply_pdf_fidelity,
    apply_block_gate,
    build_report,
    classify_remediation,
    compare_text_content,
    compare_substantive_blocks,
    extract_substantive_blocks,
    fetch_index_documents,
    load_index_snapshot,
    load_pdf_sources,
    load_web_sources,
    normalize_label,
    normalize_url,
    odata_escape,
    reconcile_sources,
    render_markdown,
    scrape_with_cache,
    serialize_index_snapshot,
    write_index_snapshot,
)


class FakeSearchResults(list):
    def __init__(self, values=(), facets=None):
        super().__init__(values)
        self.facets = facets or {}

    def get_facets(self):
        return self.facets


class FakeSearchClient:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("facets"):
            return FakeSearchResults(facets={"category": [{"value": "King's Bench"}]})
        if kwargs.get("filter"):
            return FakeSearchResults([{"id": "second"}])
        return FakeSearchResults([{"id": f"item-{number:04d}"} for number in range(1000)])


class FakeScraper:
    @staticmethod
    def scrape_page(session, action):
        return {"content": "one two three four five six seven", "_final_url": action["url"], "_redirect_count": 0}


def test_normalization_handles_legal_source_variants():
    assert normalize_label("  King’s   Bench — Division ") == "king's bench - division"
    assert normalize_url("HTTPS://WWW.JUSTICE.GOV.UK/rules/part01/?unused=1#top") == "https://justice.gov.uk/rules/part01"
    assert odata_escape("King's Bench") == "King''s Bench"


def test_pdf_action_is_classified_as_pdf_source():
    sources = load_web_sources()

    debt_claims = [source for source in sources if source.sourcefile == "Pre-Action Protocol for Debt Claims"]

    assert len(debt_claims) == 1
    assert debt_claims[0].source_type == "pdf"


def test_text_content_reports_bidirectional_coverage():
    source = "one two three four five six seven eight"
    indexed = "one two three four five six unrelated words"

    metrics = compare_text_content(source, indexed)

    assert metrics["source_ngram_count"] == 3
    assert metrics["index_ngram_count"] == 3
    assert metrics["shared_ngram_count"] == 1
    assert metrics["source_to_index_coverage"] == 1 / 3
    assert metrics["index_to_source_coverage"] == 1 / 3


def test_substantive_blocks_have_stable_ids_and_hashes():
    blocks = extract_substantive_blocks(
        "Part 1\n1.1 The overriding objective\nFootnote 1: This is operative text",
        source_type="html",
        locator_prefix="part-1",
    )

    assert [block["kind"] for block in blocks] == ["heading", "numbered_paragraph", "footnote"]
    assert blocks[1]["block_id"].startswith("part-1:2:")
    assert len(blocks[1]["normalized_hash"]) == 64
    assert blocks[1]["occurrence_ordinal"] == 1


def test_substantive_blocks_preserve_repeated_occurrences():
    blocks = extract_substantive_blocks(
        "The court must act\nThe court must act",
        source_type="html",
        locator_prefix="pd27b",
    )

    assert len(blocks) == 2
    assert blocks[0]["normalized_hash"] == blocks[1]["normalized_hash"]
    assert blocks[0]["block_id"] != blocks[1]["block_id"]
    assert [block["occurrence_ordinal"] for block in blocks] == [1, 2]


def test_occurrence_ledger_keeps_repeated_matches_independent():
    result = compare_substantive_blocks(
        "The court must act\nThe court must act",
        "The court must act\nThe court must act",
        source_type="html",
    )

    ledger = result["occurrence_ledger"]
    assert len(ledger) == 2
    assert [entry["occurrence_ordinal"] for entry in ledger] == [1, 2]
    assert [entry["source_identity"] for entry in ledger] == ["html", "html"]
    assert all(entry["status"] == "MATCHED" for entry in ledger)


def test_missing_substantive_block_fails_exact_coverage_gate():
    source = "Part 1\n1.1 The overriding objective\n1.2 The court must further the objective"
    indexed = "Part 1\n1.1 The overriding objective"

    result = compare_substantive_blocks(source, indexed, source_type="html")

    assert result["source_block_count"] == 3
    assert result["matched_block_count"] == 2
    assert result["unmatched_block_count"] == 1
    assert result["substantive_block_coverage"] < 1.0
    assert result["unmatched_blocks"][0]["text"] == "1.2 The court must further the objective"


def test_substantive_blocks_tolerate_markdown_presentation_difference():
    source = "Part 1\n1.1 The overriding objective"
    indexed = "## Part 1\n1.1: The overriding objective"

    result = compare_substantive_blocks(source, indexed, source_type="html")

    assert result["unmatched_block_count"] == 0
    assert result["ambiguous_block_count"] == 0
    assert "compact_formatting_substring" in {
        block["match_method"] for block in result["matched_blocks"]
    }


def test_compact_matching_preserves_word_boundaries():
    source = "Act"
    indexed = "Contract law applies"

    result = compare_substantive_blocks(source, indexed, source_type="html")

    assert result["unmatched_block_count"] == 1


def test_substantive_blocks_match_pdf_line_wrap_and_dehyphenation():
    source = "The inter-\nnational rule applies"
    indexed = "The international rule applies"

    result = compare_substantive_blocks(source, indexed, source_type="pdf")

    assert result["unmatched_block_count"] == 0
    assert result["ambiguous_block_count"] == 0
    assert result["matched_blocks"][0]["match_method"] == "normalized_substring"


def test_substantive_blocks_match_flattened_table_cells():
    source = "| Part | Rule |\n| 1 | The court must act |"
    indexed = "Part Rule\n1 The court must act"

    result = compare_substantive_blocks(source, indexed, source_type="pdf")

    assert result["unmatched_block_count"] == 0
    assert result["matched_block_count"] == 2
    assert all(block["match_method"] == "flattened_table_substring" for block in result["matched_blocks"])


def test_substantive_blocks_reject_ambiguous_matches():
    source = "The court must act"
    indexed = "The court must act\nThe court must act"

    result = compare_substantive_blocks(source, indexed, source_type="html")

    assert result["matched_block_count"] == 0
    assert result["unmatched_block_count"] == 0
    assert result["ambiguous_block_count"] == 1
    assert result["ambiguous_blocks"][0]["match_method"] == "ambiguous"
    assert result["ambiguous_blocks"][0]["match_count"] == 2


def test_substantive_blocks_scope_repeated_text_to_one_index_document():
    source = "The court must act"
    documents = [
        {"id": "chunk-0", "content": "The court must act\nThe court must act"},
        {"id": "chunk-1", "content": "Other content"},
    ]

    result = compare_substantive_blocks(
        source,
        "\n".join(document["content"] for document in documents),
        source_type="html",
        index_documents=documents,
    )

    assert result["ambiguous_block_count"] == 0
    assert result["matched_block_count"] == 1
    assert result["matched_blocks"][0]["matching_document_ids"] == ["chunk-0"]
    assert result["matched_blocks"][0]["match_method"].endswith("_document_scoped")


def test_unique_matches_skip_document_rescans_but_duplicates_scope_documents():
    source = "The court must act"
    documents = [
        {"id": "chunk-0", "sourcefile": "Part 1", "content": source},
        {"id": "chunk-1", "sourcefile": "Part 1", "content": "Other content"},
    ]

    with patch("scripts.audit_source_documents.count_legal_occurrences", wraps=__import__("scripts.audit_source_documents", fromlist=["count_legal_occurrences"]).count_legal_occurrences) as counter:
        compare_substantive_blocks(source, source, source_type="html", index_documents=documents)
        unique_call_count = counter.call_count
        counter.reset_mock()
        compare_substantive_blocks(
            f"{source}\n{source}",
            f"{source}\n{source}",
            source_type="html",
            index_documents=documents,
        )
        duplicate_call_count = counter.call_count

    assert unique_call_count == 1
    assert duplicate_call_count > unique_call_count


def test_impossible_token_matches_skip_expensive_occurrence_scans():
    source = "The court must act"
    indexed = "Unrelated indexed content"

    with patch("scripts.audit_source_documents.count_legal_occurrences", return_value=0) as counter:
        result = compare_substantive_blocks(source, indexed, source_type="html")

    assert result["unmatched_block_count"] == 1
    counter.assert_not_called()


def test_substantive_blocks_classify_overlap_in_other_canonical_sources():
    source = "The court must act"
    documents = [
        {"id": "pd27b-0", "sourcefile": "Practice Direction 27B", "content": "PD27B heading"},
        {"id": "protocol-0", "sourcefile": "Practice Direction 49F", "content": "The court must act"},
        {"id": "protocol-1", "sourcefile": "Practice Direction 49F", "content": "The court must act"},
    ]

    result = compare_substantive_blocks(
        source,
        "\n".join(document["content"] for document in documents),
        source_type="html",
        index_documents=documents,
        sourcefile="Practice Direction 27B",
    )

    assert result["ambiguous_block_count"] == 0
    assert result["unmatched_block_count"] == 0
    assert result["cross_document_overlap_count"] == 1
    assert result["cross_document_overlaps"][0]["matching_sourcefiles"] == ["Practice Direction 49F"]


def test_substantive_blocks_keep_same_source_duplicates_ambiguous():
    source = "The court must act"
    documents = [
        {"id": "pd27b-0", "sourcefile": "Practice Direction 27B", "content": "The court must act"},
        {"id": "pd27b-1", "sourcefile": "Practice Direction 27B", "content": "The court must act"},
    ]

    result = compare_substantive_blocks(
        source,
        "\n".join(document["content"] for document in documents),
        source_type="html",
        index_documents=documents,
        sourcefile="Practice Direction 27B",
    )

    assert result["ambiguous_block_count"] == 1
    assert result["cross_document_overlap_count"] == 0


def test_block_gate_fails_closed_on_ambiguous_matches():
    result = type("AuditResult", (), {"metrics": {"substantive_blocks": {"ambiguous_block_count": 1, "unmatched_block_count": 0}}, "status": "PASS", "issues": []})()

    apply_block_gate(result, "HTML")

    assert result.status == "FAIL"
    assert result.issues == ["1 substantive HTML block(s) have ambiguous index matches"]


def test_pdf_manifest_loads_all_local_guides():
    sources = load_pdf_sources()

    assert len(sources) == 8
    assert all(source.source_type == "pdf" for source in sources)
    assert all(source.local_path.endswith(".pdf") for source in sources)
    assert {source.sourcefile for source in sources} >= {
        "Commercial Court Guide",
        "Intellectual Property Enterprise Court Guide",
    }


def test_web_inventory_unions_action_list_with_processed_corpus(tmp_path):
    manifest = tmp_path / "manifest.py"
    manifest.write_text(
        "ACTION_LIST = [{'sourcefile': 'Part 1', 'url': 'https://example.test/part1', 'azure_id': 'part1'}]\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "Upload"
    corpus.mkdir()
    (corpus / "part1.json").write_text(
        json.dumps({"sourcefile": "Part 1", "category": "Civil Procedure Rules and Practice Directions", "storageUrl": "old"}),
        encoding="utf-8",
    )
    (corpus / "part2.json").write_text(
        json.dumps({"sourcefile": "Part 2", "category": "Civil Procedure Rules and Practice Directions", "storageUrl": "https://example.test/part2"}),
        encoding="utf-8",
    )

    sources = load_web_sources(manifest, corpus)

    assert [source.sourcefile for source in sources] == ["Part 1", "Part 2"]
    assert sources[0].url == "https://example.test/part1"


def test_live_index_reader_pages_by_category_and_escapes_apostrophes():
    client = FakeSearchClient()

    documents = fetch_index_documents(client)

    assert documents == [{"id": "second"}]
    assert client.calls[-1]["filter"] == "category eq 'King''s Bench'"
    assert all("upload_documents" not in call for call in client.calls)


def test_index_snapshot_envelope_round_trips_with_verified_provenance(tmp_path):
    documents = [{"id": "second", "category": "CPR"}, {"id": "first", "category": "CPR"}]
    path = tmp_path / "snapshot.json"

    write_index_snapshot(path, documents, "search-service", "staging-index")
    loaded_documents, provenance = load_index_snapshot(path)

    assert [document["id"] for document in loaded_documents] == ["first", "second"]
    assert provenance["verified"] is True
    assert provenance["service"] == "search-service"
    assert provenance["index"] == "staging-index"
    assert provenance["document_count"] == 2
    assert provenance["documents_sha256"] == serialize_index_snapshot(
        documents, "search-service", "staging-index", provenance["captured_at_utc"]
    )["documents_sha256"]


def test_index_snapshot_loader_accepts_legacy_array_without_verifying_provenance(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([{"id": "legacy"}]), encoding="utf-8")

    documents, provenance = load_index_snapshot(path)

    assert documents == [{"id": "legacy"}]
    assert provenance == {"verified": False, "format": "legacy_array", "document_count": 1}


def test_index_snapshot_loader_rejects_count_mismatch(tmp_path):
    path = tmp_path / "snapshot.json"
    snapshot = serialize_index_snapshot([{"id": "one"}], "service", "index", "2026-07-13T00:00:00Z")
    snapshot["document_count"] = 2
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    try:
        load_index_snapshot(path)
    except ValueError as error:
        assert "document count mismatch" in str(error)
    else:
        raise AssertionError("expected count mismatch to be rejected")


def test_index_snapshot_loader_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "snapshot.json"
    snapshot = serialize_index_snapshot([{"id": "one"}], "service", "index", "2026-07-13T00:00:00Z")
    snapshot["documents"][0]["id"] = "tampered"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    try:
        load_index_snapshot(path)
    except ValueError as error:
        assert str(error) == "Index snapshot document hash mismatch"
    else:
        raise AssertionError("expected hash mismatch to be rejected")


def test_index_snapshot_loader_rejects_empty_provenance(tmp_path):
    path = tmp_path / "snapshot.json"
    snapshot = serialize_index_snapshot([{"id": "one"}], "service", "index", "2026-07-13T00:00:00Z")
    snapshot["index"] = None
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    try:
        load_index_snapshot(path)
    except ValueError as error:
        assert str(error) == "Index snapshot provenance field is empty: index"
    else:
        raise AssertionError("expected empty provenance to be rejected")


def test_reconciliation_prefers_url_and_reports_label_mismatch():
    canonical = [
        CanonicalSource(
            source_type="pdf",
            sourcefile="King's Bench Division Guide",
            category="King's Bench Division",
            url="https://www.judiciary.uk/guide.pdf",
        )
    ]
    documents = [
        {
            "id": "kb-2",
            "sourcefile": "King’s Bench Guide",
            "category": "King’s Bench",
            "storageUrl": "https://judiciary.uk/guide.pdf?download=1",
        },
        {
            "id": "kb-1",
            "sourcefile": "King’s Bench Guide",
            "category": "King’s Bench",
            "storageUrl": "https://judiciary.uk/guide.pdf",
        },
    ]

    results = reconcile_sources(canonical, documents)

    assert len(results) == 1
    assert results[0].status == "WARN"
    assert results[0].index_document_ids == ["kb-1", "kb-2"]
    assert "category/sourcefile differs" in results[0].issues[0]


def test_reconciliation_matches_pdf_by_unique_category():
    canonical = [
        CanonicalSource(
            source_type="pdf",
            sourcefile="King's Bench Division Guide",
            category="King's Bench Division",
            url="https://example.test/2025/01/guide.pdf",
        )
    ]
    documents = [
        {
            "id": "kb-1",
            "sourcefile": "guide.pdf",
            "category": "King's Bench Division",
            "storageUrl": "https://example.test/2025/04/guide.pdf",
        }
    ]

    results = reconcile_sources(canonical, documents)

    assert len(results) == 1
    assert results[0].status == "WARN"
    assert results[0].index_document_ids == ["kb-1"]


def test_reconciliation_reports_missing_and_index_only_sources():
    canonical = [CanonicalSource(source_type="html", sourcefile="Part 1", category="CPR")]
    documents = [
        {
            "id": "circuit-1",
            "sourcefile": "Circuit Commercial Court Guide",
            "category": "Circuit Commercial Court",
            "storageUrl": "https://judiciary.uk/circuit.pdf",
        }
    ]

    results = reconcile_sources(canonical, documents)

    assert [result.status for result in results] == ["INDEX_ONLY", "MISSING_FROM_INDEX"]
    assert results[1].issues == ["canonical URL is unavailable or requires discovery"]


def test_reconciliation_can_suppress_index_only_for_focused_runs():
    canonical = [CanonicalSource(source_type="html", sourcefile="Part 1", category="CPR")]
    documents = [{"id": "other", "sourcefile": "Other", "category": "CPR"}]

    results = reconcile_sources(canonical, documents, include_index_only=False)

    assert [result.status for result in results] == ["MISSING_FROM_INDEX"]


def test_report_serialization_is_deterministic():
    canonical = [CanonicalSource(source_type="html", sourcefile="Part 1", category="CPR")]
    documents = [{"id": "part-1", "sourcefile": "Part 1", "category": "CPR", "storageUrl": ""}]

    report = build_report(reconcile_sources(canonical, documents))
    first = json.dumps(report, sort_keys=True)
    second = json.dumps(build_report(reconcile_sources(canonical, reversed(documents))), sort_keys=True)

    assert first == second
    assert report["summary"] == {
        "source_count": 1,
        "statuses": {"PASS": 1},
        "dispositions": {"VERIFIED_PRESENT": 1},
    }
    assert "| PASS | VERIFIED_PRESENT | html | CPR | Part 1 | 1 | - | - |" in render_markdown(report)


def test_report_renders_snapshot_provenance_and_generic_fail_needs_review():
    canonical = [CanonicalSource(source_type="html", sourcefile="Part 1", category="CPR")]
    documents = [{"id": "part-1", "sourcefile": "Part 1", "category": "CPR", "storageUrl": ""}]
    provenance = {"verified": False, "format": "legacy_array", "service": "service", "index": "index", "document_count": 1}

    report = build_report(reconcile_sources(canonical, documents), provenance)

    assert report["snapshot_provenance"] == provenance
    assert "Snapshot**: unverified (legacy_array)" in render_markdown(report)

    failed = reconcile_sources(
        [CanonicalSource(source_type="html", sourcefile="Part 1", category="CPR", url="https://example.test/part1")],
        [{"id": "part-1", "sourcefile": "Part 1", "category": "CPR", "storageUrl": "https://example.test/part1"}],
    )[0]
    failed.status = "FAIL"
    assert classify_remediation(failed) == "NEEDS_REVIEW"


def test_html_cache_reuses_successful_checkpoint(tmp_path):
    cache = HtmlAuditCache(tmp_path / "cache.json")
    source = CanonicalSource(
        source_type="html", sourcefile="Part 48", category="CPR", url="https://example.test/part48"
    )
    scraper = FakeScraper()
    session = object()

    first, first_url = scrape_with_cache(session, source, scraper, cache)
    second, second_url = scrape_with_cache(session, source, scraper, HtmlAuditCache(tmp_path / "cache.json"))

    assert first == second
    assert first_url == second_url == source.url


def test_html_cache_tries_known_alias_after_primary_failure(tmp_path):
    class AliasScraper:
        calls = []

        @classmethod
        def scrape_page(cls, session, action):
            cls.calls.append(action["url"])
            if action["url"] == "https://example.test/pd40f":
                return None
            return {"content": "one two three four five six seven", "_final_url": action["url"]}

    source = CanonicalSource(
        source_type="html", sourcefile="Practice Direction 40F", category="CPR", url="https://example.test/pd40f"
    )

    result, requested_url = scrape_with_cache(object(), source, AliasScraper, HtmlAuditCache(tmp_path / "cache.json"))

    assert result is not None
    assert requested_url.endswith("practice-direction-40f-proceedings-involving-declarations-of-incompatibility")
    assert AliasScraper.calls == [source.url, requested_url]


def test_pdf_fidelity_marks_missing_local_pdf_unavailable(tmp_path):
    canonical = [
        CanonicalSource(
            source_type="pdf",
            sourcefile="Test Guide",
            category="Test Court",
            local_path=str(tmp_path / "missing.pdf"),
        )
    ]
    documents = [{"id": "test-1", "sourcefile": "Test Guide", "category": "Test Court", "content": "text"}]
    results = reconcile_sources(canonical, documents)

    audited = apply_pdf_fidelity(results, canonical, documents)

    assert audited[0].status == "UNAVAILABLE"
    assert audited[0].issues == [f"local PDF not found: {tmp_path / 'missing.pdf'}"]


def test_html_fidelity_marks_low_coverage_as_failure():
    canonical = [
        CanonicalSource(
            source_type="html",
            sourcefile="Part 1",
            category="CPR",
            url="https://example.test/part1",
        )
    ]
    documents = [
        {
            "id": "part1",
            "sourcefile": "Part 1",
            "category": "CPR",
            "storageUrl": "https://example.test/part1",
            "content": "unrelated indexed words with no shared sequence",
        }
    ]

    audited = apply_html_fidelity(reconcile_sources(canonical, documents), canonical, documents, FakeScraper())

    assert audited[0].status == "FAIL"
    assert audited[0].metrics["source_to_index_coverage"] == 0