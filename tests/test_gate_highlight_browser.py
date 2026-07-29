import pytest
from types import SimpleNamespace

import scripts.gate_highlight_browser as browser_gate
from scripts.gate_highlight_browser import (
    BrowserGateError,
    choose_case,
    citation_matches,
    select_citation,
    validate_highlight_identity,
)

TARGET = {
    "sourcefile": "Part 24",
    "sourcepage": "PART 24 - SUMMARY JUDGMENT",
    "subsection_id": "24.2",
    "category": "Civil Procedure Rules and Practice Directions",
}


def citation(sourcefile="Part 24", subsection_id="24.2", sourcepage="PART 24 - SUMMARY JUDGMENT", category=None, citation_path=""):
    return {
        "sourcefile": sourcefile,
        "sourcepage": sourcepage,
        "subsection_id": subsection_id,
        "category": category or TARGET["category"],
        "citation_path": citation_path,
    }


def test_select_citation_requires_canonical_source_and_subsection():
    selected = select_citation(
        TARGET,
        [
            citation(sourcefile="Commercial Court Guide"),
            citation(sourcefile="Part 24", subsection_id="24.2(1)"),
        ],
    )

    assert selected["sourcefile"] == "Part 24"
    assert citation_matches(TARGET, selected)


def test_select_citation_rejects_ambiguous_canonical_candidates():
    with pytest.raises(BrowserGateError, match="found 2"):
        select_citation(TARGET, [citation(), citation()])


def test_select_citation_accepts_citation_path_when_sourcefile_is_absent():
    target = {**TARGET, "sourcefile": "", "citation_path": "/content/Part 24"}
    selected = citation(sourcefile="", citation_path="/content/Part 24/")

    assert select_citation(target, [selected]) == selected


def test_select_citation_rejects_wrong_category_or_source_page():
    assert not citation_matches(TARGET, citation(category="King's Bench Division Guide"))
    assert not citation_matches(TARGET, citation(sourcepage="PART 25 - COSTS"))


def test_select_citation_rejects_unrelated_shared_subsection():
    with pytest.raises(BrowserGateError, match="found 0"):
        select_citation(TARGET, [citation(sourcefile="Commercial Court Guide")])


def test_select_citation_accepts_broader_live_source_page_label():
    selected = citation(sourcepage="PART 24")

    assert select_citation(TARGET, [selected]) == selected


def test_select_citation_normalizes_source_page_dash_punctuation():
    selected = citation(sourcepage="PART 24 - SUMMARY JUDGMENT")

    assert select_citation(TARGET, [selected]) == selected


def test_select_citation_accepts_parent_source_page_for_leaf_oracle_case():
    target = {
        **TARGET,
        "sourcepage": "24.2 The court may give summary judgment",
    }
    selected = citation(sourcepage="PART 24 - SUMMARY JUDGMENT")

    assert select_citation(target, [selected]) == selected


def test_validate_highlight_identity_accepts_heading_in_card_and_subsection_in_mark():
    validate_highlight_identity(
        "24.2 The court may give summary judgment",
        "Part 24, PART 24 - SUMMARY JUDGMENT, Civil Procedure Rules and Practice Directions 24.2 The court may give summary judgment",
        "PART 24 - SUMMARY JUDGMENT",
        "24.2",
    )


def test_validate_highlight_identity_accepts_leaf_subsection_for_parent_heading():
    validate_highlight_identity(
        "24.2 The court may give summary judgment",
        "Part 24, PART 24 - SUMMARY JUDGMENT, 24.2 The court may give summary judgment",
        "PART 24 - SUMMARY JUDGMENT",
        "PART 24",
    )


def test_validate_highlight_identity_accepts_canonical_source_labels_when_card_omits_parent_heading():
    validate_highlight_identity(
        "24.2 The court may give summary judgment-",
        "24.2 The court may give summary judgment-",
        "PART 24",
        "24.2",
        "PART 24 - SUMMARY JUDGMENT",
        "Part 24",
    )


def test_validate_highlight_identity_rejects_heading_from_another_card():
    with pytest.raises(BrowserGateError, match="canonical target heading"):
        validate_highlight_identity(
            "24.2 The court may give summary judgment",
            "Part 25, PART 25 - COSTS",
            "PART 24 - SUMMARY JUDGMENT",
            "24.2",
        )


def test_choose_case_prefers_part_24_leaf_case_when_sourcepage_is_leaf_heading():
    parent_case = {
        "sourcefile": "Part 24",
        "sourcepage": "PART 24 - SUMMARY JUDGMENT",
        "subsection_id": "PART 24",
        "body_text": "Contents of this Part",
    }
    leaf_case = {
        "sourcefile": "Part 24",
        "sourcepage": "24.2 The court may give summary judgment-",
        "subsection_id": "24.2",
        "identity": "civil procedure rules and practice directions::part 24",
        "body_text": "24.2 The court may give summary judgment- (a) against a claimant",
    }

    assert choose_case({"cases": [parent_case, leaf_case]}) is leaf_case


def test_exhaustive_browser_gate_reuses_one_browser_per_shard(monkeypatch, tmp_path):
    cases = [
        {"case_id": "case-1", "subsection_id": "24.1", "sourcepage": "Part 24"},
        {"case_id": "case-2", "subsection_id": "24.2", "sourcepage": "Part 24"},
    ]
    oracle = {"cases": cases}
    browser_instances = []

    class FakeBrowser:
        def close(self):
            pass

    class FakePlaywright:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(launch=self.launch))

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def launch(self, headless):
            browser = FakeBrowser()
            browser_instances.append(browser)
            return browser

    def fake_run_browser_gate(*args, **kwargs):
        assert kwargs["browser"] is browser_instances[0]
        case = kwargs["target_case"]
        return {
            "browser": {
                "selected_citation": {
                    "document_id": case["case_id"],
                    "source_revision": "rev-1",
                    "source_id": case["case_id"],
                    "canonical_text_sha256": f"hash-{case['case_id']}",
                },
                "highlighted_text_sha256": f"highlight-{case['case_id']}",
                "primary_source_identity": {},
            }
        }

    monkeypatch.setattr(browser_gate, "sync_playwright", lambda: FakePlaywright())
    monkeypatch.setattr(browser_gate, "run_browser_gate", fake_run_browser_gate)

    report = browser_gate.run_exhaustive_browser_gate(
        "https://candidate.example",
        oracle,
        tmp_path / "coverage.json",
        shard_count=1,
    )

    assert report["status"] == "PASS"
    assert len(browser_instances) == 1


def test_success_diagnostics_are_compact_and_do_not_write_browser_bundle(tmp_path):
    class FakeTracing:
        def __init__(self):
            self.stop_paths = []

        def stop(self, path=None):
            self.stop_paths.append(path)

    tracing = FakeTracing()
    page = SimpleNamespace(
        url="https://candidate.example",
        title=lambda: "Candidate",
        context=SimpleNamespace(tracing=tracing),
    )

    browser_gate._write_browser_diagnostics(page, {"responses": []}, tmp_path, retain_full=False)

    payload = (tmp_path / "browser-diagnostics.json").read_text()
    assert '"retention": "compact"' in payload
    assert tracing.stop_paths == [None]
    assert not (tmp_path / "browser-final.png").exists()
    assert not (tmp_path / "browser-trace.zip").exists()


def test_exhaustive_browser_gate_writes_global_progress_checkpoint(monkeypatch, tmp_path):
    cases = [
        {"case_id": "case-0", "subsection_id": "24.1", "sourcepage": "Part 24"},
        {"case_id": "case-1", "subsection_id": "24.2", "sourcepage": "Part 24"},
        {"case_id": "case-2", "subsection_id": "24.3", "sourcepage": "Part 24"},
        {"case_id": "case-3", "subsection_id": "24.4", "sourcepage": "Part 24"},
    ]
    oracle = {"cases": cases}

    class FakeBrowser:
        def close(self):
            pass

    class FakePlaywright:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(launch=lambda headless: FakeBrowser()))

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_run_browser_gate(*args, **kwargs):
        case = kwargs["target_case"]
        return {
            "browser": {
                "selected_citation": {
                    "document_id": case["case_id"],
                    "source_revision": "rev-1",
                    "source_id": case["case_id"],
                    "canonical_text_sha256": f"hash-{case['case_id']}",
                },
                "highlighted_text_sha256": f"highlight-{case['case_id']}",
                "primary_source_identity": {},
            }
        }

    monkeypatch.setattr(browser_gate, "sync_playwright", lambda: FakePlaywright())
    monkeypatch.setattr(browser_gate, "run_browser_gate", fake_run_browser_gate)
    diagnostics_dir = tmp_path / "diagnostics"

    browser_gate.run_exhaustive_browser_gate(
        "https://candidate.example",
        oracle,
        tmp_path / "coverage.json",
        diagnostics_dir=diagnostics_dir,
        shard_index=1,
        shard_count=2,
    )

    progress = __import__("json").loads((diagnostics_dir / "browser-progress.json").read_text())
    assert progress["completed_case_ids"] == ["case-1", "case-3"]
    assert not (diagnostics_dir / "browser-progress.json.tmp").exists()
