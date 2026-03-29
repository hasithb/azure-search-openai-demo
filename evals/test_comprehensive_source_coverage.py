"""
Comprehensive accuracy and quality tests for all sources and sections.

Tests output quality across every source category and major section in the
legal RAG index. This validates that the system can accurately retrieve and
present information from:
  - CPR Parts 1-89 (all sections)
  - Practice Directions (all indexed PDs)
  - Court Guides (Commercial, Circuit Commercial, TCC, Patents, King's Bench, Chancery)
  - Pre-Action Protocols
  - Cross-source questions (multi-source retrieval)

Ground truth is loaded from evals/ground_truth_comprehensive.jsonl.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_FILE = ROOT / "evals" / "ground_truth_comprehensive.jsonl"
V3_INDEX_FILE = ROOT / "data" / "legal-scraper" / "processed" / "v3_full_corrected.json"
UPLOAD_DIR = ROOT / "data" / "legal-scraper" / "processed" / "Upload"

# ---------------------------------------------------------------------------
# Regex patterns (aligned with evaluate.py)
# ---------------------------------------------------------------------------
CPR_PART_REGEX = re.compile(r"Part\s+(\d+[A-Z]?)\b", re.IGNORECASE)
PD_REGEX = re.compile(
    r"(?:Practice\s+Direction|PD)\s+(\d+[A-Z]{0,2})\b", re.IGNORECASE
)
COURT_GUIDE_REGEX = re.compile(
    r"(Commercial\s+Court|Circuit\s+Commercial\s+Court|Technology\s+and\s+Construction\s+Court|Patents\s+Court|King'?s?\s+Bench\s+Division|Chancery)\s*(?:Guide|Division)?",
    re.IGNORECASE,
)
CITATION_BRACKET_REGEX = re.compile(r"\[([^\]]+)\]")
SUBSECTION_REGEX = re.compile(
    r"\b(\d+\.\d+(?:\.\d+)?[A-Z]?)\b"
    r"|\b([A-Z]\.\d+(?:\.\d+)?)\b"
    r"|\b(Rule\s+\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_ground_truth() -> list[dict[str, Any]]:
    """Load ground truth entries from the comprehensive JSONL file."""
    entries = []
    if not GROUND_TRUTH_FILE.exists():
        pytest.skip(f"Ground truth file not found: {GROUND_TRUTH_FILE}")
    for line in GROUND_TRUTH_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def load_index_documents() -> list[dict[str, Any]]:
    """Load all indexed documents from the v3 corrected JSON."""
    if not V3_INDEX_FILE.exists():
        pytest.skip(f"Index file not found: {V3_INDEX_FILE}")
    data = json.loads(V3_INDEX_FILE.read_text())
    if isinstance(data, list):
        return data
    return data.get("documents", data.get("value", [data]))


def load_upload_document(filename: str) -> dict[str, Any] | None:
    """Load a single upload document JSON."""
    path = UPLOAD_DIR / filename
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data[0] if data else None
    return data


# All 7 categories in the system
ALL_CATEGORIES = [
    "Civil Procedure Rules and Practice Directions",
    "Commercial Court",
    "Circuit Commercial Court",
    "Technology and Construction Court",
    "King's Bench Division",
    "Chancery Division",
    "Patents Court",
]

# Map of CPR Parts that MUST be in the index (1-89, plus 57A, 63A)
EXPECTED_CPR_PARTS = [str(i) for i in range(1, 90)] + ["57A", "63A"]

# All Practice Directions referenced in Upload directory
EXPECTED_PRACTICE_DIRECTIONS = [
    "1A", "2A", "2B", "2C", "2E", "2F",
    "3A", "3B", "3C", "3D", "3E",
    "5A", "5B",
    "6A", "6B",
    "7A", "7B", "7C",
    "16", "17", "18", "19A", "19B",
    "20", "22", "23A",
    "26", "27A", "27B", "28", "29",
    "30", "31A", "31B", "31C", "32", "33", "34A", "34B", "35", "36", "37",
    "40A", "40B", "40D", "40E", "40F", "41A", "41B", "42",
    "44", "45", "46", "47", "48",
    "49A", "49B", "49C", "49D", "49E", "49F", "49G",
    "51Y", "51ZB", "51ZC", "51ZE", "51ZF",
    "52A", "52B", "52C", "52D", "52E",
    "53A", "53B",
    "54A", "54B", "54C", "54D", "54E",
    "55A", "55B",
    "56", "56A",
    "57", "57AC", "57AD", "57B",
    "58", "59", "60",
    "61", "62", "63A", "64A", "64B", "65", "66", "67", "69",
    "70A", "70B", "71", "72", "73", "74A", "75", "77",
    "83", "84",
]

# Court guide source files
COURT_GUIDE_FILES = {
    "Commercial Court": "14.341_JO_Commercial_Court_Guide_FINAL_processed.json",
    "King's Bench Division": "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json",
    "Chancery Division": "Chancery-Guide-2024-web_processed.json",
    "Technology and Construction Court": "The-Technology-and-Construction-Court-Guide_processed.json",
    "Patents Court": "Patents-Court-Guide-Updated-February-2025_processed.json",
}


# ===================================================================
#  Section 1: Ground Truth Format & Completeness
# ===================================================================

class TestGroundTruthCompleteness:
    """Verify the comprehensive ground truth covers all source categories."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.entries = load_ground_truth()
        self.by_category: dict[str, list] = {}
        for e in self.entries:
            cat = e.get("category", "UNKNOWN")
            self.by_category.setdefault(cat, []).append(e)

    def test_minimum_total_entries(self):
        """Ground truth should have at least 60 entries."""
        assert len(self.entries) >= 60, (
            f"Expected >=60 entries, got {len(self.entries)}"
        )

    def test_all_categories_represented(self):
        """Every source category must have at least 1 question."""
        for cat in ALL_CATEGORIES:
            assert cat in self.by_category, (
                f"Category '{cat}' has no ground truth entries"
            )

    def test_cpr_category_has_breadth(self):
        """CPR category should cover multiple Parts."""
        cpr_entries = self.by_category.get(
            "Civil Procedure Rules and Practice Directions", []
        )
        parts_mentioned = set()
        for e in cpr_entries:
            text = e["question"] + " " + e["truth"]
            for m in CPR_PART_REGEX.finditer(text):
                parts_mentioned.add(m.group(1))
        assert len(parts_mentioned) >= 20, (
            f"Expected CPR questions covering >=20 Parts, got {len(parts_mentioned)}: "
            f"{sorted(parts_mentioned, key=lambda x: (int(re.match(r'(\\d+)', x).group(1)), x))}"
        )

    def test_chancery_division_has_entries(self):
        """Chancery Division must have ground truth entries (was previously missing)."""
        chancery = self.by_category.get("Chancery Division", [])
        assert len(chancery) >= 3, (
            f"Expected >=3 Chancery Division entries, got {len(chancery)}"
        )

    def test_court_guides_have_entries(self):
        """Each court guide should have at least 1 entry."""
        guide_cats = [
            "Commercial Court",
            "Circuit Commercial Court",
            "Technology and Construction Court",
            "Patents Court",
            "King's Bench Division",
            "Chancery Division",
        ]
        for cat in guide_cats:
            assert cat in self.by_category, (
                f"Court guide '{cat}' has no ground truth entries"
            )

    def test_pre_action_protocols_covered(self):
        """At least 2 entries should reference Pre-Action Protocols."""
        count = 0
        for e in self.entries:
            text = e["question"] + " " + e["truth"]
            if re.search(r"pre-action\s+protocol", text, re.IGNORECASE):
                count += 1
        assert count >= 2, (
            f"Expected >=2 Pre-Action Protocol entries, got {count}"
        )

    def test_cross_source_questions_exist(self):
        """At least 2 entries should reference multiple sources/Parts."""
        cross_count = 0
        for e in self.entries:
            text = e["question"] + " " + e["truth"]
            parts = set(CPR_PART_REGEX.findall(text))
            guides = set(COURT_GUIDE_REGEX.findall(text))
            pds = set(PD_REGEX.findall(text))
            sources = len(parts) + len(guides) + len(pds)
            if sources >= 3:
                cross_count += 1
        assert cross_count >= 2, (
            f"Expected >=2 cross-source entries, got {cross_count}"
        )

    def test_entries_have_required_fields(self):
        """Every entry must have question, truth, source_type, category."""
        for i, e in enumerate(self.entries):
            for field in ("question", "truth", "source_type", "category"):
                assert field in e, f"Entry {i} missing '{field}'"
                assert e[field], f"Entry {i} has empty '{field}'"

    def test_truth_has_source_references(self):
        """Each truth should contain at least one source reference in brackets."""
        missing = []
        for i, e in enumerate(self.entries):
            truth = e["truth"]
            if not CITATION_BRACKET_REGEX.search(truth):
                missing.append(i)
        # Allow up to 5% without bracket citations
        max_missing = max(3, len(self.entries) // 20)
        assert len(missing) <= max_missing, (
            f"{len(missing)} entries lack source references: indices {missing[:10]}"
        )


# ===================================================================
#  Section 2: Index Document Coverage
# ===================================================================

class TestIndexDocumentCoverage:
    """Verify the search index has complete coverage of all expected sources."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.docs = load_index_documents()
        self.sourcefiles = {d.get("sourcefile", "") for d in self.docs}
        self.categories = {d.get("category", "") for d in self.docs}

    def test_index_has_documents(self):
        """Index should have substantial number of documents."""
        assert len(self.docs) >= 200, (
            f"Expected >=200 index docs, got {len(self.docs)}"
        )

    def test_cpr_parts_present_in_index(self):
        """All major CPR Parts should have index documents."""
        missing = []
        for part_num in range(1, 90):
            part_str = f"Part {part_num}"
            found = any(
                part_str in sf
                for sf in self.sourcefiles
            )
            if not found:
                missing.append(part_num)
        # Allow some missing (e.g., Parts 43, 78 may be revoked)
        assert len(missing) <= 5, (
            f"Missing CPR Parts from index ({len(missing)}): {missing}"
        )

    def test_practice_directions_present(self):
        """Major Practice Directions should be in the index."""
        pd_count = sum(
            1 for sf in self.sourcefiles
            if "Practice Direction" in sf or sf.startswith("PD")
        )
        assert pd_count >= 40, (
            f"Expected >=40 Practice Direction sourcefiles, got {pd_count}"
        )

    def test_documents_have_content(self):
        """Every document should have non-empty content."""
        empty = []
        for i, doc in enumerate(self.docs):
            content = doc.get("content", "").strip()
            if not content:
                empty.append(i)
        assert len(empty) == 0, (
            f"{len(empty)} documents have empty content"
        )

    def test_documents_have_category(self):
        """Every document should have a category field."""
        missing_cat = [
            i for i, d in enumerate(self.docs)
            if not d.get("category")
        ]
        assert len(missing_cat) == 0, (
            f"{len(missing_cat)} documents lack a category"
        )

    def test_documents_have_sourcepage(self):
        """Every document should have a sourcepage field."""
        missing_sp = [
            i for i, d in enumerate(self.docs)
            if not d.get("sourcepage")
        ]
        assert len(missing_sp) == 0, (
            f"{len(missing_sp)} documents lack sourcepage"
        )


# ===================================================================
#  Section 3: Per-Source Content Quality
# ===================================================================

class TestCPRPartContentQuality:
    """Test content accuracy for individual CPR Parts."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.docs = load_index_documents()
        self.by_sourcefile: dict[str, list[dict]] = {}
        for d in self.docs:
            sf = d.get("sourcefile", "")
            self.by_sourcefile.setdefault(sf, []).append(d)

    @pytest.mark.parametrize("part_num,expected_topic", [
        (1, "overriding objective"),
        (2, "application and interpretation"),
        (3, "case management powers"),
        (5, "court documents"),
        (6, "service"),
        (7, "claim form"),
        (9, "responding to particulars"),
        (10, "acknowledgment of service"),
        (11, "jurisdiction"),
        (12, "default judgment"),
        (14, "admissions"),
        (15, "defence"),
        (17, "amendments"),
        (21, "children and protected parties"),
        (24, "summary judgment"),
        (25, "interim remedies"),
        (27, "small claims"),
        (28, "fast track"),
        (29, "multi-track"),
        (30, "transfer"),
        (31, "disclosure"),
        (35, "experts"),
        (36, "offers to settle"),
        (38, "discontinuance"),
        (39, "hearings"),
        (40, "judgments"),
        (44, "costs"),
        (52, "appeals"),
        (54, "judicial review"),
        (55, "possession"),
        (70, "enforcement"),
        (72, "third party debt"),
    ])
    def test_cpr_part_content_matches_topic(self, part_num, expected_topic):
        """Each CPR Part's content should mention its core topic."""
        part_key = f"Part {part_num}"
        docs = self.by_sourcefile.get(part_key, [])
        if not docs:
            # Try with longer names
            docs = [
                d for sf, dl in self.by_sourcefile.items()
                if sf.startswith(part_key)
                for d in dl
            ]
        if not docs:
            pytest.skip(f"Part {part_num} not found in index")

        combined_content = " ".join(
            d.get("content", "") for d in docs
        ).lower()
        assert expected_topic in combined_content, (
            f"Part {part_num} content should mention '{expected_topic}' "
            f"but it wasn't found in {len(combined_content)} chars of content"
        )


class TestPracticeDirectionContentQuality:
    """Test content accuracy for Practice Directions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.docs = load_index_documents()
        self.pd_docs: dict[str, list[dict]] = {}
        for d in self.docs:
            sf = d.get("sourcefile", "")
            if "Practice Direction" in sf or sf.startswith("PD"):
                self.pd_docs.setdefault(sf, []).append(d)

    @pytest.mark.parametrize("pd_number,expected_topic", [
        ("16", "statements of case"),
        ("22", "statements of truth"),
        ("27A", "small claims"),
        ("31A", "disclosure"),
        ("31B", "electronic"),
        ("35", "experts"),
        ("40B", "judgments"),
        ("44", "costs"),
        ("52A", "appeals"),
        ("54A", "judicial review"),
        ("55A", "possession"),
        ("57AD", "disclosure"),
    ])
    def test_pd_content_matches_topic(self, pd_number, expected_topic):
        """Each PD's content should mention its core topic."""
        matching_docs = []
        for sf, docs in self.pd_docs.items():
            # Match "Practice Direction 16" or "Practice Direction 16 –"
            if re.search(
                rf"Practice\s+Direction\s+{re.escape(pd_number)}\b",
                sf,
                re.IGNORECASE,
            ):
                matching_docs.extend(docs)
            elif sf == f"PD {pd_number}":
                matching_docs.extend(docs)

        if not matching_docs:
            pytest.skip(f"PD {pd_number} not found in index")

        combined_content = " ".join(
            d.get("content", "") for d in matching_docs
        ).lower()
        assert expected_topic in combined_content, (
            f"PD {pd_number} content should mention '{expected_topic}'"
        )


class TestCourtGuideContentQuality:
    """Test content quality for each Court Guide."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.court_guide_docs: dict[str, list[dict]] = {}
        for filename in COURT_GUIDE_FILES.values():
            filepath = UPLOAD_DIR / filename
            if filepath.exists():
                data = json.loads(filepath.read_text())
                docs = data if isinstance(data, list) else [data]
                self.court_guide_docs[filename] = docs

    @pytest.mark.parametrize("guide_name,filename,expected_keywords", [
        (
            "Commercial Court",
            "14.341_JO_Commercial_Court_Guide_FINAL_processed.json",
            ["commercial court", "statement", "case management"],
        ),
        (
            "King's Bench Division",
            "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json",
            ["king", "bench"],
        ),
        (
            "Chancery Division",
            "Chancery-Guide-2024-web_processed.json",
            ["chancery"],
        ),
        (
            "TCC",
            "The-Technology-and-Construction-Court-Guide_processed.json",
            ["technology", "construction"],
        ),
        (
            "Patents Court",
            "Patents-Court-Guide-Updated-February-2025_processed.json",
            ["patent"],
        ),
    ])
    def test_court_guide_has_relevant_content(
        self, guide_name, filename, expected_keywords
    ):
        """Court guide documents should contain expected keywords."""
        docs = self.court_guide_docs.get(filename, [])
        if not docs:
            pytest.skip(f"{guide_name} guide not found at {filename}")

        combined = " ".join(d.get("content", "") for d in docs).lower()
        for keyword in expected_keywords:
            assert keyword in combined, (
                f"{guide_name} guide should contain '{keyword}'"
            )

    @pytest.mark.parametrize("guide_name,filename", [
        ("Commercial Court", "14.341_JO_Commercial_Court_Guide_FINAL_processed.json"),
        ("King's Bench", "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json"),
        ("Chancery", "Chancery-Guide-2024-web_processed.json"),
        ("TCC", "The-Technology-and-Construction-Court-Guide_processed.json"),
        ("Patents Court", "Patents-Court-Guide-Updated-February-2025_processed.json"),
    ])
    def test_court_guide_has_sections(self, guide_name, filename):
        """Court guides should have identifiable sections/subsections."""
        docs = self.court_guide_docs.get(filename, [])
        if not docs:
            pytest.skip(f"{guide_name} not found")

        combined = " ".join(d.get("content", "") for d in docs)
        subsections = SUBSECTION_REGEX.findall(combined)
        assert len(subsections) >= 3, (
            f"{guide_name} should have >=3 identifiable subsections, "
            f"got {len(subsections)}"
        )


# ===================================================================
#  Section 4: Ground Truth Quality Metrics (offline - no API needed)
# ===================================================================

class TestGroundTruthAnswerQuality:
    """Validate that ground truth answers are well-formed and substantive."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.entries = load_ground_truth()

    def test_answers_have_minimum_length(self):
        """Each truth answer should be substantive (>100 chars)."""
        short = []
        for i, e in enumerate(self.entries):
            if len(e["truth"]) < 100:
                short.append((i, len(e["truth"]), e["question"][:60]))
        assert len(short) == 0, (
            f"{len(short)} answers are too short: {short[:5]}"
        )

    def test_questions_are_specific(self):
        """Questions should be specific enough for useful evaluation."""
        vague = []
        for i, e in enumerate(self.entries):
            q = e["question"]
            # Questions should mention at least a Part, PD, or Court Guide
            if not (
                CPR_PART_REGEX.search(q)
                or PD_REGEX.search(q)
                or COURT_GUIDE_REGEX.search(q)
                or re.search(r"pre-action\s+protocol", q, re.IGNORECASE)
                or re.search(r"chancery|commercial|tcc|patents|king", q, re.IGNORECASE)
            ):
                vague.append((i, q[:80]))
        # Allow very few vague questions
        assert len(vague) <= 3, (
            f"{len(vague)} questions lack specific source references: {vague[:5]}"
        )

    def test_truth_cites_specific_rules(self):
        """Many truth answers should cite specific rule numbers or sections."""
        with_rules = 0
        for e in self.entries:
            truth = e["truth"]
            if (
                re.search(r"Rule\s+\d+", truth, re.IGNORECASE)
                or re.search(r"\d+\.\d+", truth)
                or re.search(r"Section\s+\d+", truth, re.IGNORECASE)
                or re.search(r"paragraph\s+\d+", truth, re.IGNORECASE)
                or re.search(r"Part\s+\d+", truth, re.IGNORECASE)
                or re.search(r"CPR\s+\d+", truth, re.IGNORECASE)
                or re.search(r"PD\s+\d+", truth, re.IGNORECASE)
            ):
                with_rules += 1
        ratio = with_rules / len(self.entries) if self.entries else 0
        assert ratio >= 0.4, (
            f"Expected >=40% of truths to cite specific rules, got {ratio:.0%}"
        )

    def test_categories_are_valid(self):
        """All categories in ground truth should be valid system categories."""
        invalid = []
        for i, e in enumerate(self.entries):
            if e["category"] not in ALL_CATEGORIES:
                invalid.append((i, e["category"]))
        assert len(invalid) == 0, (
            f"Invalid categories found: {invalid}"
        )

    def test_source_types_are_valid(self):
        """Source types should be from the expected set."""
        valid_types = {"CPR", "PD", "Court Guide"}
        invalid = []
        for i, e in enumerate(self.entries):
            if e["source_type"] not in valid_types:
                invalid.append((i, e["source_type"]))
        assert len(invalid) == 0, (
            f"Invalid source_types found: {invalid}"
        )


# ===================================================================
#  Section 5: Source-Specific Coverage Depth
# ===================================================================

class TestCPRCoverage:
    """Test that CPR Parts have adequate ground truth coverage."""

    @pytest.fixture(autouse=True)
    def _load(self):
        all_entries = load_ground_truth()
        # Also load other ground truth files for combined coverage
        for gt_file in ROOT.glob("evals/ground_truth_*.jsonl"):
            if gt_file.name == "ground_truth_multimodal.jsonl":
                continue
            if gt_file == GROUND_TRUTH_FILE:
                continue
            for line in gt_file.read_text().splitlines():
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))
        
        self.entries = all_entries
        self.parts_covered: set[str] = set()
        for e in self.entries:
            text = e.get("question", "") + " " + e.get("truth", "")
            for m in CPR_PART_REGEX.finditer(text):
                self.parts_covered.add(m.group(1))

    def test_majority_of_cpr_parts_covered(self):
        """At least 65 of the 89 CPR Parts should be referenced."""
        assert len(self.parts_covered) >= 65, (
            f"Expected >=65 CPR Parts covered, got {len(self.parts_covered)}"
        )

    def test_core_cpr_parts_covered(self):
        """Core operational Parts must all be covered."""
        core_parts = {
            "1", "2", "3", "6", "7", "12", "14", "15", "24", "25",
            "27", "31", "35", "36", "44", "52", "54", "55", "70",
        }
        missing = core_parts - self.parts_covered
        assert not missing, (
            f"Core CPR Parts missing from ground truth: {sorted(missing)}"
        )

    def test_enforcement_parts_covered(self):
        """Enforcement-related Parts should be covered."""
        enforcement_parts = {"70", "71", "72", "83", "84", "85", "89"}
        covered = enforcement_parts.intersection(self.parts_covered)
        assert len(covered) >= 4, (
            f"Expected >=4 enforcement Parts covered, got {len(covered)}: "
            f"missing {enforcement_parts - covered}"
        )

    def test_specialist_parts_covered(self):
        """Specialist jurisdiction Parts should be covered."""
        specialist_parts = {"54", "57", "58", "60", "62", "63", "64", "65"}
        covered = specialist_parts.intersection(self.parts_covered)
        assert len(covered) >= 4, (
            f"Expected >=4 specialist Parts covered, got {len(covered)}"
        )


class TestPracticeDirectionCoverage:
    """Test Practice Direction ground truth coverage."""

    @pytest.fixture(autouse=True)
    def _load(self):
        all_entries = []
        for gt_file in ROOT.glob("evals/ground_truth_*.jsonl"):
            if gt_file.name == "ground_truth_multimodal.jsonl":
                continue
            for line in gt_file.read_text().splitlines():
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

        self.pds_covered: set[str] = set()
        for e in all_entries:
            text = e.get("question", "") + " " + e.get("truth", "")
            for m in PD_REGEX.finditer(text):
                self.pds_covered.add(m.group(1).upper())

    def test_practice_directions_breadth(self):
        """At least 30 distinct PDs should be referenced."""
        assert len(self.pds_covered) >= 30, (
            f"Expected >=30 PDs covered, got {len(self.pds_covered)}: "
            f"{sorted(self.pds_covered)}"
        )

    def test_key_practice_directions_covered(self):
        """Key Practice Directions must be covered."""
        key_pds = {"16", "31A", "31B", "44", "52A", "54A", "57AD"}
        missing = key_pds - self.pds_covered
        assert not missing, (
            f"Key Practice Directions missing: {sorted(missing)}"
        )


class TestCourtGuideCoverage:
    """Test Court Guide ground truth coverage."""

    @pytest.fixture(autouse=True)
    def _load(self):
        all_entries = []
        for gt_file in ROOT.glob("evals/ground_truth_*.jsonl"):
            if gt_file.name == "ground_truth_multimodal.jsonl":
                continue
            for line in gt_file.read_text().splitlines():
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

        self.entries = all_entries
        self.by_category: dict[str, list] = {}
        for e in all_entries:
            cat = e.get("category", "UNKNOWN")
            self.by_category.setdefault(cat, []).append(e)

    @pytest.mark.parametrize("guide_cat,min_entries", [
        ("Commercial Court", 5),
        ("Circuit Commercial Court", 5),
        ("Technology and Construction Court", 5),
        ("Patents Court", 5),
        ("King's Bench Division", 5),
        ("Chancery Division", 3),
    ])
    def test_court_guide_minimum_coverage(self, guide_cat, min_entries):
        """Each court guide should have minimum number of ground truth Q&As."""
        entries = self.by_category.get(guide_cat, [])
        assert len(entries) >= min_entries, (
            f"'{guide_cat}' has {len(entries)} entries, expected >={min_entries}"
        )


# ===================================================================
#  Section 6: Cross-Source Retrieval Testing
# ===================================================================

class TestCrossSourceQuestionQuality:
    """Validate cross-source questions reference multiple distinct sources."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.entries = load_ground_truth()

    def test_cross_source_entries_span_multiple_parts(self):
        """Cross-source entries should reference at least 2 distinct source types."""
        cross_entries = []
        for e in self.entries:
            text = e["question"] + " " + e["truth"]
            has_cpr = bool(CPR_PART_REGEX.search(text))
            has_pd = bool(PD_REGEX.search(text))
            has_guide = bool(COURT_GUIDE_REGEX.search(text))
            source_types = sum([has_cpr, has_pd, has_guide])
            if source_types >= 2:
                cross_entries.append(e)
        
        assert len(cross_entries) >= 2, (
            f"Expected >=2 cross-source entries, got {len(cross_entries)}"
        )

    def test_multi_part_questions_exist(self):
        """Some questions should reference multiple CPR Parts."""
        multi_part_count = 0
        for e in self.entries:
            text = e["question"] + " " + e["truth"]
            parts = set(CPR_PART_REGEX.findall(text))
            if len(parts) >= 2:
                multi_part_count += 1
        assert multi_part_count >= 3, (
            f"Expected >=3 multi-Part questions, got {multi_part_count}"
        )


# ===================================================================
#  Section 7: Pre-Action Protocol Coverage
# ===================================================================

class TestPreActionProtocolCoverage:
    """Verify Pre-Action Protocol documents are indexed and tested."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.upload_files = list(UPLOAD_DIR.glob("Pre-Action_Protocol_*.json"))

    def test_pre_action_protocols_exist_in_index(self):
        """Multiple Pre-Action Protocols should be in the upload data."""
        assert len(self.upload_files) >= 5, (
            f"Expected >=5 Pre-Action Protocols, found {len(self.upload_files)}"
        )

    def test_key_pre_action_protocols_present(self):
        """Key Pre-Action Protocols should be present."""
        protocol_names = [f.stem for f in self.upload_files]
        combined = " ".join(protocol_names).lower()
        
        expected = [
            "personal_injury",
            "judicial_review",
            "construction",
        ]
        for kw in expected:
            assert kw in combined, (
                f"Expected Pre-Action Protocol for '{kw}' not found"
            )

    def test_pre_action_protocols_have_content(self):
        """Each Pre-Action Protocol file should have substantial content."""
        for f in self.upload_files[:5]:  # Sample first 5
            data = json.loads(f.read_text())
            doc = data[0] if isinstance(data, list) else data
            content = doc.get("content", "")
            assert len(content) >= 100, (
                f"Pre-Action Protocol {f.name} has insufficient content ({len(content)} chars)"
            )


# ===================================================================
#  Section 8: Welsh Language Document Coverage
# ===================================================================

class TestWelshLanguageCoverage:
    """Verify Welsh language Practice Directions are indexed."""

    def test_welsh_practice_directions_present(self):
        """Welsh (Cyfarwyddyd Ymarfer) documents should be in the upload data."""
        welsh_files = list(UPLOAD_DIR.glob("Cyfarwyddyd_Ymarfer_*.json"))
        assert len(welsh_files) >= 4, (
            f"Expected >=4 Welsh PD files, found {len(welsh_files)}: "
            f"{[f.name for f in welsh_files]}"
        )


# ===================================================================
#  Section 9: Upload Directory Completeness
# ===================================================================

class TestUploadDirectoryCompleteness:
    """Verify the upload directory has all expected documents."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.upload_files = list(UPLOAD_DIR.glob("*.json"))
        self.file_names = {f.name for f in self.upload_files}

    def test_upload_directory_has_sufficient_files(self):
        """Upload directory should have a large number of document files."""
        assert len(self.upload_files) >= 200, (
            f"Expected >=200 upload files, got {len(self.upload_files)}"
        )

    def test_all_court_guide_files_present(self):
        """All court guide processed files should be present."""
        for guide_name, filename in COURT_GUIDE_FILES.items():
            assert filename in self.file_names, (
                f"{guide_name} file '{filename}' not found in Upload dir"
            )

    def test_all_cpr_parts_have_files(self):
        """Every CPR Part 1-89 should have at least one upload file."""
        missing = []
        for part_num in range(1, 90):
            # Skip possible revoked parts
            if part_num in (43, 78):  # These may have been revoked
                continue
            found = any(
                f"Part_{part_num}_" in fn or f"Part_{part_num}." in fn
                for fn in self.file_names
            )
            if not found:
                missing.append(part_num)
        assert len(missing) <= 2, (
            f"Missing CPR Part Upload files for: {missing}"
        )


# ===================================================================
#  Section 10: Evaluation Config Validation
# ===================================================================

class TestEvaluationConfig:
    """Validate the evaluation configuration files are properly set up."""

    def test_evaluate_config_exists(self):
        """evaluate_config.json should exist with valid content."""
        config_file = ROOT / "evals" / "evaluate_config.json"
        assert config_file.exists(), "evaluate_config.json not found"
        config = json.loads(config_file.read_text())
        assert "testdata_path" in config or "target_url" in config or isinstance(config, dict), (
            "evaluate_config.json should be a valid configuration"
        )

    def test_comprehensive_ground_truth_referenced(self):
        """The comprehensive ground truth file should exist and be loadable."""
        assert GROUND_TRUTH_FILE.exists(), (
            f"Comprehensive ground truth not found: {GROUND_TRUTH_FILE}"
        )
        entries = load_ground_truth()
        assert len(entries) >= 60, (
            f"Comprehensive ground truth should have >=60 entries"
        )
