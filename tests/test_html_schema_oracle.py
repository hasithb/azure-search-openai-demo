from scripts.html_schema_oracle import (
  build_schema_census,
  capture_html_snapshot,
  compare_legal_blocks,
  extract_legal_blocks,
  write_html_snapshot,
)


HTML = """
<html><body>
  <nav>Navigation that must not enter the legal inventory</nav>
  <main id="legal-content">
    <h1>Part 1</h1>
    <p id="rule-1">The overriding objective applies.</p>
    <ol><li>First legal requirement.</li><li>Second legal requirement.</li></ol>
    <table><tr><th>Form</th><th>Purpose</th></tr><tr><td>N1</td><td>Issue a claim.</td></tr></table>
    <aside class="footnote" id="footnote-1">Footnote: this qualification matters.</aside>
    <h2>Schedule 1</h2><p>Schedule text must be retained.</p>
  </main>
</body></html>
"""


def test_oracle_inventory_preserves_dom_schema_and_ignores_navigation():
    census = build_schema_census(HTML)

    assert census["kind_counts"] == {
      "footnote": 1,
        "heading": 2,
        "li": 2,
      "p": 2,
        "table_cell": 4,
    }
    assert census["schema_counts"]["table-cell"] == 4
    assert census["schema_counts"]["footnote"] == 1
    assert census["schema_counts"]["schedule-annex-form"] == 1
    assert all("Navigation" not in block["text"] for block in census["blocks"])


def test_oracle_detects_exact_deleted_table_cell():
    original = extract_legal_blocks(HTML)
    mutated = extract_legal_blocks(HTML.replace("<td>Issue a claim.</td>", "<td></td>"))

    original_hashes = {block.normalized_hash for block in original}
    mutated_hashes = {block.normalized_hash for block in mutated}

    assert original_hashes - mutated_hashes
    assert not any(block.text == "Issue a claim." for block in mutated)


def test_oracle_ignores_navigation_when_page_has_no_content_root():
    blocks = extract_legal_blocks("<html><nav>Menu</nav><section><p>Rule text.</p></section></html>")

    assert [block.text for block in blocks] == ["Rule text."]


def test_oracle_comparison_counts_duplicate_blocks_instead_of_collapsing_them():
    expected = extract_legal_blocks("<main><p>Repeated.</p><p>Repeated.</p></main>")
    observed = extract_legal_blocks("<main><p>Repeated.</p></main>")

    comparison = compare_legal_blocks(expected, observed)

    assert comparison["matched_count"] == 1
    assert len(comparison["missing_blocks"]) == 1
    assert comparison["exact_match"] is False


def test_oracle_comparison_reports_unexpected_duplicate_blocks():
    expected = extract_legal_blocks("<main><p>One.</p></main>")
    observed = extract_legal_blocks("<main><p>One.</p><p>One.</p></main>")

    comparison = compare_legal_blocks(expected, observed)

    assert comparison["missing_blocks"] == []
    assert len(comparison["unexpected_blocks"]) == 1
    assert comparison["exact_match"] is False


def test_oracle_uses_stable_locator_and_hash_for_reordered_unrelated_content():
    source = "<main><h1>Part 1</h1><p>Alpha.</p><p>Beta.</p></main>"
    reordered = "<main><h1>Part 1</h1><p>Beta.</p><p>Alpha.</p></main>"

    first = extract_legal_blocks(source)
    second = extract_legal_blocks(reordered)

    assert {block.normalized_hash for block in first} == {block.normalized_hash for block in second}
    assert first[1].locator != second[1].locator or first[1].text != second[1].text


def test_raw_snapshot_records_response_provenance_and_census(tmp_path):
    class Response:
        content = HTML.encode("utf-8")
        encoding = "utf-8"
        url = "https://example.test/final"
        status_code = 200
        headers = {"Content-Type": "text/html"}
        history = [object()]

        def raise_for_status(self):
            return None

    class Session:
        def get(self, url, *, timeout, allow_redirects):
            assert url == "https://example.test/start"
            assert timeout == 30
            assert allow_redirects is True
            return Response()

    snapshot = capture_html_snapshot(Session(), "https://example.test/start")
    output = tmp_path / "source.json"
    write_html_snapshot(snapshot, output)

    persisted = output.read_text(encoding="utf-8")
    assert snapshot["requested_url"] == "https://example.test/start"
    assert snapshot["final_url"] == "https://example.test/final"
    assert snapshot["redirect_count"] == 1
    assert snapshot["content_sha256"]
    assert snapshot["schema_census"]["block_count"] > 0
    assert '"oracle_version": "2"' in persisted


def test_oracle_handles_deeply_nested_html_without_recursion_error():
    source = "<main>" + "<div>" * 1200 + "<p>Rule text.</p>" + "</div>" * 1200 + "</main>"

    blocks = extract_legal_blocks(source)

    assert [block.text for block in blocks] == ["Rule text."]