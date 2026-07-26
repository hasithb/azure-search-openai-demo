import json

import scripts.generate_highlight_oracle as highlight_oracle


def test_load_snapshot_cases_treats_numbered_paragraphs_as_section_boundaries(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    monkeypatch.setattr(highlight_oracle, "ROOT", tmp_path)
    snapshot = {
        "status": "ok",
        "oracle_version": "2026-07-26",
        "identity": "rules::part 24",
        "content_sha256": "content-hash",
        "sourcefile": "Part 24",
        "category": "Civil Procedure",
        "schema_census": {
            "blocks": [
                {"kind": "heading", "locator": "h1", "text": "PART 24 - SUMMARY JUDGMENT"},
                {"kind": "table_cell", "locator": "cell1", "text": "Rule 24.2"},
                {"kind": "p", "locator": "p1", "text": "24.2 The court may give summary judgment."},
                {"kind": "p", "locator": "p2", "text": "24.3 The court may give summary judgment against a claimant."},
            ]
        },
    }
    (snapshot_dir / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    cases = highlight_oracle.load_snapshot_cases(snapshot_dir)

    assert [case["subsection_id"] for case in cases] == ["PART 24", "24.2", "24.3"]
    assert cases[1]["body_text"] == "24.2 The court may give summary judgment."
    assert cases[1]["next_heading"] == "24.3 The court may give summary judgment against a claimant."