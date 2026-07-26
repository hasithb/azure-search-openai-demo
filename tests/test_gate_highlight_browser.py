import pytest

from scripts.gate_highlight_browser import BrowserGateError, citation_matches, select_citation, validate_highlight_identity


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


def test_validate_highlight_identity_accepts_heading_in_card_and_subsection_in_mark():
    validate_highlight_identity(
        "24.2 The court may give summary judgment",
        "Part 24, PART 24 - SUMMARY JUDGMENT, Civil Procedure Rules and Practice Directions 24.2 The court may give summary judgment",
        "PART 24 - SUMMARY JUDGMENT",
        "24.2",
    )


def test_validate_highlight_identity_rejects_heading_from_another_card():
    with pytest.raises(BrowserGateError, match="canonical target heading"):
        validate_highlight_identity(
            "24.2 The court may give summary judgment",
            "Part 25, PART 25 - COSTS",
            "PART 24 - SUMMARY JUDGMENT",
            "24.2",
        )
