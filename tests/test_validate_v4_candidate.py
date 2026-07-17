import json

from scripts.validate_v4_candidate import validate_documents


def test_candidate_validation_accepts_complete_documents():
    result = validate_documents([
        {"id": "doc-1", "content": "Rule text", "category": "CPR", "sourcefile": "Part 1", "sourcepage": "Part 1"}
    ])

    assert result["status"] == "PASS"
    assert result["document_count"] == 1


def test_candidate_validation_rejects_empty_index():
    result = validate_documents([])

    assert result["status"] == "FAIL"
    assert result["document_count"] == 0
    assert result["empty_index_count"] == 1


def test_candidate_validation_rejects_duplicate_and_incomplete_documents():
    result = validate_documents([
        {"id": "doc-1", "content": "Rule text", "category": "CPR", "sourcefile": "Part 1", "sourcepage": "Part 1"},
        {"id": "doc-1", "content": "", "category": "", "sourcefile": "Part 1", "sourcepage": ""},
    ])

    assert result["status"] == "FAIL"
    assert result["duplicate_id_count"] == 1
    assert result["missing_field_count"] == 3
    assert result["uncategorized_count"] == 1
    assert result["empty_content_count"] == 1


def test_candidate_validation_rejects_subsection_metadata_not_in_content():
    result = validate_documents([
        {
            "id": "doc-1",
            "content": "35.1 The rule text",
            "category": "CPR",
            "sourcefile": "Part 35",
            "sourcepage": "Part 35",
            "subsection_id": "35.2",
            "subsections": ["35.1", "35.2"],
        }
    ])

    assert result["status"] == "FAIL"
    assert result["subsection_mismatch_count"] == 2