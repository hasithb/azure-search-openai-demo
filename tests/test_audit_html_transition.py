from types import SimpleNamespace

from scripts.audit_html_transition import _block_match, _comparison_text, _snapshot_action, audit_snapshot
from scripts.html_schema_oracle import extract_legal_blocks
from scripts.update_cpr_index_v3 import verify_scrape_target


def test_transition_audit_preserves_table_cell_and_chunk_metadata():
    source = SimpleNamespace(identity="civil::Part 1", sourcefile="Part 1")
    action = {
        "sourcefile": "Part 1",
        "url": "https://example.test/part-1",
    }
    snapshot = {
        "status": "ok",
        "final_url": action["url"],
        "redirect_count": 0,
        "html": """<main><h1>Part 1 Overriding Objective</h1><table><tr><th>Rule</th><th>Duty</th></tr><tr><td>1.1</td><td>Deal with cases justly.</td></tr></table></main>""",
    }

    result = audit_snapshot(snapshot, source, action)

    assert result["status"] == "PASS"
    assert result["raw_block_coverage"] == 1.0
    assert result["chunk_count"] == 1


def test_transition_audit_blocks_missing_raw_legal_text(monkeypatch):
    source = SimpleNamespace(identity="civil::Part 1", sourcefile="Part 1")
    action = {"sourcefile": "Part 1", "url": "https://example.test/part-1"}
    snapshot = {
        "status": "ok",
        "final_url": action["url"],
        "redirect_count": 0,
        "html": "<main><h1>Part 1 Overriding Objective</h1><p>Mandatory omitted provision with operative duty.</p></main>",
    }

    monkeypatch.setattr(
        "scripts.audit_html_transition.updater.scrape_page",
        lambda *args, **kwargs: {
            "content": "# Part 1 Overriding Objective",
            "title": "Part 1 Overriding Objective",
            "updated": "2026-01-01T00:00:00Z",
            "_final_url": action["url"],
        },
    )

    result = audit_snapshot(snapshot, source, action)

    assert result["status"] == "FAIL"
    assert result["missing_blocks"]


def test_transition_audit_builds_verified_fallback_action_from_snapshot_url():
    source = SimpleNamespace(sourcefile="Part 19")
    snapshot = {
        "sourcefile": "Part 19",
        "requested_url": "https://example.test/part-19",
        "final_url": "https://example.test/part-19",
    }

    action = _snapshot_action(snapshot, source, {}, {})

    assert action == {
        "sourcefile": "Part 19",
        "azure_id": None,
        "url": "https://example.test/part-19",
        "verified_snapshot_url": "https://example.test/part-19",
        "section": "ORACLE",
    }


def test_transition_audit_accepts_table_cell_ordering_difference():
    matched, method, score, occurrences = _block_match(
        "Central London London",
        "London Central London",
    )

    assert matched is True
    assert method == "table_token_coverage"
    assert score == 1.0
    assert occurrences == 1


def test_production_verifier_accepts_explicit_pd_40f_alias():
    verified, reason = verify_scrape_target(
        {
            "sourcefile": "Practice Direction 40F",
            "url": "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part40/practice-direction-40f-non-disclosure-injunctions-information-collection-scheme",
        },
        "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part40/practice-direction-40f-non-disclosure-injunctions-information-collection-scheme",
        "NON-DISCLOSURE ORDERS INFORMATION SCHEME",
    )

    assert verified is True
    assert reason == "ok"


def test_transition_comparison_strips_oracle_section_locator():
    assert _comparison_text("In the County Court, the Renting Homes (Wales) claim— 56.6") == (
        "in the county court, the renting homes (wales) claim-"
    )


def test_transition_comparison_preserves_standalone_bracketed_placeholders():
    assert _comparison_text("[If the claimant is legally represented]") == (
        "[if the claimant is legally represented]"
    )


def test_transition_comparison_normalizes_inline_footnote_marker_order():
    assert _comparison_text("the Landlord and Tenant Act 1927 ; 1") == "the landlord and tenant act 1927 ;"
    assert _comparison_text("the Landlord and Tenant Act 1927 1 ;") == "the landlord and tenant act 1927 ;"


def test_list_match_accepts_inserted_label_when_all_oracle_tokens_survive():
    matched, method, score, occurrences = _block_match(
        "any evidence uploaded to the in support of the claim for",
        "any evidence uploaded to the Portal in support of the claim for",
        "li",
    )

    assert (matched, method, score, occurrences) == (True, "list_token_coverage", 1.0, 1)


def test_html_oracle_preserves_inline_dom_order():
    blocks = extract_legal_blocks("<main><p>(a) accepts the <strong>accident</strong> happened;</p></main>")

    assert blocks[0].text == "(a) accepts the accident happened;"