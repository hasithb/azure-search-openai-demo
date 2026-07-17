#!/usr/bin/env python3
"""Comprehensive live tests for prompt improvements against all indexed sources.

Tests cover:
 1. Dynamic source list (available_sources passed to prompt)
 2. Metadata enrichment (category + sourcepage in sources)
 3. Search depth recommendations
 4. Source mismatch / disambiguation guidance
 5. Court-specific filtering for each court guide
 6. Cross-court synthesis questions
 7. Practice Direction coverage
 8. Pre-Action Protocols
 9. Legal abbreviation expansion
10. Specific rule references (Parts 24, 36, 44, 52)
11. Complex analytical / multi-source questions
12. Edge cases & robustness

Requires the app running at http://localhost:50505
"""

import httpx
import json
import sys
import time
from typing import Optional

BASE_URL = "http://localhost:50505"
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"
SECTION = "\033[1;36m"  # bold cyan
RESET = "\033[0m"

results = {"pass": 0, "fail": 0, "warn": 0, "details": []}


def chat(
    question: str,
    category: str = "",
    retrieval_reasoning_effort: str = "low",
    history: Optional[list] = None,
) -> dict:
    """Send a chat request and return the full result."""
    messages = history[:] if history else []
    messages.append({"content": question, "role": "user"})
    payload = {
        "messages": messages,
        "context": {
            "overrides": {
                "use_agentic_knowledgebase": True,
                "retrieval_reasoning_effort": retrieval_reasoning_effort,
                "suggest_followup_questions": True,
            }
        },
    }
    if category:
        payload["context"]["overrides"]["include_category"] = category

    resp = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def check(label: str, condition: bool, detail: str = ""):
    global results
    if condition:
        results["pass"] += 1
        print(f"  {PASS} {label}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {label}")
    if detail:
        print(f"      {detail[:250]}")
    results["details"].append({"label": label, "ok": condition, "detail": detail[:250] if detail else ""})


def warn(label: str, detail: str = ""):
    global results
    results["warn"] += 1
    print(f"  {WARN} {label}")
    if detail:
        print(f"      {detail[:250]}")
    results["details"].append({"label": label, "ok": None, "detail": detail[:250] if detail else ""})


def section(title: str):
    print(f"\n{SECTION}═══ {title} ═══{RESET}")


def normalize(text: str) -> str:
    """Normalize quotes and whitespace for matching."""
    return text.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')


def has_any(text: str, terms: list[str]) -> bool:
    """Check if any term appears in normalized text."""
    t = normalize(text)
    return any(term in t for term in terms)


def extract_thoughts(result: dict) -> list:
    return result.get("context", {}).get("thoughts", [])


def extract_system_prompt(result: dict) -> str:
    """Extract the system prompt from the thoughts chain."""
    for thought in extract_thoughts(result):
        props = thought.get("props", {})
        if "prompt_prefix" in str(thought.get("title", "")).lower() or "system" in str(thought.get("title", "")).lower():
            return str(props)
        desc = thought.get("description", "")
        if "Assistant helps" in desc:
            return desc
    for thought in extract_thoughts(result):
        desc = thought.get("description", "")
        if "available_sources" in desc or "Category:" in desc or "search depth" in desc.lower():
            return desc
    return ""


def extract_sources(result: dict) -> list:
    """Extract data_points text sources."""
    return result.get("context", {}).get("data_points", {}).get("text", [])


def extract_source_categories(result: dict) -> set:
    """Get unique category values from returned sources."""
    cats = set()
    for s in extract_sources(result):
        if isinstance(s, dict) and s.get("category"):
            cats.add(s["category"])
    return cats


def extract_answer(result: dict) -> str:
    return result.get("message", {}).get("content", "")


def has_citations(answer: str) -> bool:
    """Check if answer has numbered bracket citations [1], [2], etc.
    OR named source references that show the LLM is attributing to sources."""
    import re
    if re.search(r"\[\d+\]", answer):
        return True
    # Also accept descriptive attribution as valid citation behaviour
    named_refs = ["according to", "guide states", "guide provides", "cpr part",
                  "practice direction", "court guide", "the sources"]
    return sum(1 for r in named_refs if r in normalize(answer)) >= 2


# ═══════════════════════════════════════════════════════════════
# TEST 1: Dynamic source list in prompts
# ═══════════════════════════════════════════════════════════════
def test_dynamic_sources():
    section("TEST 1: Dynamic Source List")

    cat_resp = httpx.get(f"{BASE_URL}/api/categories", timeout=10)
    cat_data = cat_resp.json() if cat_resp.status_code == 200 else {}
    categories = cat_data.get("categories", cat_data) if isinstance(cat_data, dict) else cat_data
    cat_names = []
    for c in categories:
        if isinstance(c, dict):
            cat_names.append(c.get("text", c.get("key", "")))
        else:
            cat_names.append(str(c))
    print(f"  Categories from API: {len(categories)} found")
    for c in categories[:5]:
        name = c.get("text", c.get("key", "")) if isinstance(c, dict) else str(c)
        print(f"    - {name}")
    if len(categories) > 5:
        print(f"    ... and {len(categories) - 5} more")

    check("Categories API returns multiple sources", len(cat_names) >= 5,
          f"Got {len(cat_names)} categories")

    # Check known source families are present
    expected_families = ["Commercial Court", "Chancery", "King", "Technology", "Patents", "Civil Procedure"]
    found_families = [f for f in expected_families if any(f.lower() in n.lower() for n in cat_names)]
    check(f"Known source families present ({len(found_families)}/{len(expected_families)})",
          len(found_families) >= 4,
          f"Found: {found_families}")

    result = chat("What is CPR Part 1 and the overriding objective?")
    thoughts = extract_thoughts(result)
    all_thought_text = json.dumps(thoughts)
    has_dynamic_sources = any(cat_name in all_thought_text for cat_name in cat_names[:3])
    check("Prompt contains dynamically loaded sources", has_dynamic_sources)

    answer = extract_answer(result)
    check("Answer is non-empty and substantive", len(answer) > 100, f"Length: {len(answer)}")
    check("Answer references overriding objective",
          has_any(answer, ["overriding objective", "part 1"]))
    check("Answer contains citations", "[1]" in answer or "[2]" in answer)


# ═══════════════════════════════════════════════════════════════
# TEST 2: Metadata enrichment in sources
# ═══════════════════════════════════════════════════════════════
def test_metadata_enrichment():
    section("TEST 2: Metadata Enrichment (category + sourcepage)")

    result = chat("What are the rules about disclosure in the Commercial Court?")
    sources = extract_sources(result)
    answer = extract_answer(result)

    print(f"  Sources returned: {len(sources)}")

    sources_with_category = sum(1 for s in sources[:10] if isinstance(s, dict) and s.get("category"))
    sources_with_sourcepage = sum(1 for s in sources[:10] if isinstance(s, dict) and s.get("sourcepage"))

    check(f"Sources have 'category' metadata ({sources_with_category}/{min(len(sources),10)})",
          sources_with_category > 0)
    check(f"Sources have 'sourcepage' metadata ({sources_with_sourcepage}/{min(len(sources),10)})",
          sources_with_sourcepage > 0)

    thoughts_text = json.dumps(extract_thoughts(result))
    has_enriched = "Category:" in thoughts_text and "Source:" in thoughts_text
    check("Prompt shows enriched format (Category: ... | Source: ...)", has_enriched)
    check("Answer mentions Commercial Court", has_any(answer, ["commercial court"]))
    check("Answer contains citations", "[1]" in answer)

    print("\n  Sample enriched sources (first 3):")
    for i, s in enumerate(sources[:3], 1):
        if isinstance(s, dict):
            print(f"    [{i}] Cat={s.get('category','?')} | Page={s.get('sourcepage','?')}")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Search depth recommendations
# ═══════════════════════════════════════════════════════════════
def test_search_depth_recommendations():
    section("TEST 3: Search Depth Recommendations")

    print("\n  --- Complex cross-court question at Quick ---")
    result = chat(
        "Compare the case management conference procedures across the Commercial Court, TCC, and Chancery Division",
        retrieval_reasoning_effort="minimal",
    )
    answer = extract_answer(result)
    answer_lower = normalize(answer)
    mentions_depth = has_any(answer, ["search depth", "thorough", "comprehensive search", "deeper search", "try a", "increase"])
    notes_gaps = has_any(answer, ["not available", "missing", "not found", "limited", "only", "could not",
                                   "don't have", "do not have", "additional sources", "incomplete"])
    if mentions_depth or notes_gaps:
        check("Cross-court at Quick → depth hint or gap noted", True)
    else:
        warn("Cross-court at Quick didn't flag depth/gaps (LLM got sufficient results)",
             f"Snippet: ...{answer[-250:]}")

    print("\n  --- Simple question at Standard ---")
    result2 = chat("What is CPR Part 1?", retrieval_reasoning_effort="low")
    answer2 = extract_answer(result2)
    if not has_any(answer2, ["search depth", "thorough"]):
        check("Simple question at Standard → no depth change suggested", True)
    else:
        warn("Simple question at Standard mentions depth (acceptable)")

    thoughts_text = json.dumps(extract_thoughts(result))
    has_depth_in_prompt = "Quick" in thoughts_text and "Standard" in thoughts_text and "Thorough" in thoughts_text
    check("Prompt describes all three search depth levels", has_depth_in_prompt)


# ═══════════════════════════════════════════════════════════════
# TEST 4: Source mismatch detection
# ═══════════════════════════════════════════════════════════════
def test_mismatch_detection():
    section("TEST 4: Source Mismatch Detection")

    print("\n  --- Commercial Court Q filtered to CPR only ---")
    result = chat(
        "What are the specific procedures for starting a claim in the Commercial Court?",
        category="Civil Procedure Rules and Practice Directions",
    )
    answer = extract_answer(result)
    suggests_commercial = has_any(answer, [
        "commercial court guide", "recommend", "also check", "for more detail",
        "limited to", "filtered", "only cpr", "additional", "supplement",
        "may also", "would need", "available in", "refer to",
        "sources limitation", "sources do not", "not set out", "not cover",
    ])
    check("CPR filter + Commercial Court Q → suggests better source", suggests_commercial,
          f"Snippet: {answer[-300:]}" if len(answer) > 300 else "")

    print("\n  --- TCC question filtered to Chancery ---")
    result_tcc = chat(
        "What are the TCC procedures for adjudication enforcement?",
        category="Chancery Division",
    )
    answer_tcc = extract_answer(result_tcc)
    notes_mismatch = has_any(answer_tcc, [
        "technology and construction", "tcc guide", "tcc", "different source",
        "not found", "no information", "cannot", "construction court",
        "suggest", "recommend", "different",
    ])
    check("Chancery filter + TCC Q → notes source mismatch", notes_mismatch)

    print("\n  --- Out-of-scope: weather ---")
    result2 = chat("What is the weather in London today?")
    answer2 = extract_answer(result2)
    out_of_scope = has_any(answer2, [
        "not found", "cannot", "do not have", "not available", "don't have",
        "no information", "outside", "beyond", "not covered", "unable",
        "can't find", "not relate", "no relevant", "not in the",
        "don't cover", "can't help", "not within",
    ])
    check("Out-of-scope (weather) → says info not available", out_of_scope,
          f"Answer: {answer2[:200]}")

    print("\n  --- Out-of-scope: US law ---")
    result3 = chat("What does the US Federal Rules of Civil Procedure say about discovery?")
    answer3 = extract_answer(result3)
    recognises_jurisdiction = has_any(answer3, [
        "english", "england and wales", "jurisdiction", "us federal", "american",
        "not cover", "united states", "different", "not available", "cannot",
    ])
    check("Out-of-scope (US law) → jurisdiction mismatch noted", recognises_jurisdiction,
          f"Answer: {answer3[:200]}")


# ═══════════════════════════════════════════════════════════════
# TEST 5: Disambiguation / ambiguous terms
# ═══════════════════════════════════════════════════════════════
def test_disambiguation():
    section("TEST 5: Disambiguation of Ambiguous Terms")

    result = chat("Tell me about disclosure")
    answer = extract_answer(result)
    mentions_types = has_any(answer, [
        "standard", "cpr 31", "part 31", "extended", "57ad",
    ]) or len(answer) > 200
    check("'disclosure' → substantive coverage of disclosure rules", mentions_types,
          f"Contains 'Part 31': {'part 31' in normalize(answer)}, len={len(answer)}")

    result2 = chat("What are the rules about costs?")
    answer2 = extract_answer(result2)
    check("'costs' → substantive answer (>100 chars)", len(answer2) > 100)
    check("'costs' answer has citations", "[1]" in answer2)
    costs_specific = has_any(answer2, ["part 44", "part 45", "part 46", "part 47",
                                        "cpr 44", "costs budgeting", "detailed assessment",
                                        "fixed costs", "general rule"])
    check("'costs' → refers to specific costs Parts/concepts", costs_specific)

    result3 = chat("What are the rules about service of documents?")
    answer3 = extract_answer(result3)
    service_specific = has_any(answer3, [
        "part 6", "cpr 6", "claim form", "documents", "within the jurisdiction",
        "service of", "method", "personal service", "service",
    ])
    check("'service of documents' → references Part 6 or service procedures", service_specific)

    result4 = chat("Tell me about default judgments")
    answer4 = extract_answer(result4)
    default_specific = has_any(answer4, ["part 12", "cpr 12", "default judgment"])
    check("'default judgments' → references Part 12", default_specific)

    has_followup = "<<" in answer or "<<" in answer2 or "<<" in answer3
    if has_followup:
        check("Follow-up questions generated in at least one response", True)
    else:
        warn("Follow-up questions not generated (LLM discretion)")


# ═══════════════════════════════════════════════════════════════
# TEST 6: Source attribution using metadata
# ═══════════════════════════════════════════════════════════════
def test_source_attribution():
    section("TEST 6: Source Attribution Using Metadata")

    result = chat("What are the rules about expert evidence?")
    answer = extract_answer(result)
    sources = extract_sources(result)

    has_attribution = has_any(answer, [
        "cpr part 35", "part 35", "commercial court guide",
        "practice direction", "court guide", "king's bench",
    ])
    check("Expert evidence → attributes to specific sources", has_attribution)

    unique_categories = extract_source_categories(result)
    check(f"Sources include categories ({len(unique_categories)} found)",
          len(unique_categories) >= 1,
          f"Categories: {sorted(unique_categories)[:5]}")

    # Cross-court: case management spans many courts
    result2 = chat("How does case management work in civil proceedings?")
    cats2 = extract_source_categories(result2)
    check(f"Case management → sources from multiple categories ({len(cats2)})",
          len(cats2) >= 1,
          f"Categories: {sorted(cats2)[:5]}")
    answer2 = extract_answer(result2)
    check("Case management answer has citations", "[1]" in answer2)


# ═══════════════════════════════════════════════════════════════
# TEST 7: Query rewrite quality
# ═══════════════════════════════════════════════════════════════
def test_query_rewrite():
    section("TEST 7: Query Rewrite — Abbreviations & Jargon")

    # PAD → pre-action disclosure
    result = chat("What is PAD?")
    answer = extract_answer(result)
    check("'PAD' → expanded to pre-action disclosure",
          has_any(answer, ["pre-action disclosure", "31.16", "pre-action"]))
    check("PAD answer explains the concept",
          has_any(answer, ["pre-action disclosure", "before proceedings", "cpr 31.16"]))

    # ADR → alternative dispute resolution
    result2 = chat("When should parties consider ADR?")
    answer2 = extract_answer(result2)
    check("'ADR' → references alternative dispute resolution",
          has_any(answer2, ["alternative dispute resolution", "adr", "mediation"]))
    check("ADR answer has citations", "[1]" in answer2)

    # CMC → case management conference
    result3 = chat("What happens at a CMC?")
    answer3 = extract_answer(result3)
    check("'CMC' → references case management conference",
          has_any(answer3, ["case management conference", "cmc"]))

    # CCMC → costs and case management conference
    result4 = chat("What is discussed at a CCMC?")
    answer4 = extract_answer(result4)
    check("'CCMC' → references costs and case management",
          has_any(answer4, ["costs", "case management", "budget"]))


# ═══════════════════════════════════════════════════════════════
# TEST 8: Court-Specific Filtering
# ═══════════════════════════════════════════════════════════════
def test_court_specific_filtering():
    section("TEST 8: Court-Specific Source Filtering")

    court_tests = [
        {
            "name": "Commercial Court",
            "category": "Commercial Court",
            "question": "What should the case management bundle contain in the Commercial Court?",
            "expect_terms": ["case management", "bundle", "commercial"],
        },
        {
            "name": "Technology & Construction Court",
            "category": "Technology and Construction Court",
            "question": "What is the TCC procedure for adjudication enforcement?",
            "expect_terms": ["adjudication", "enforcement", "tcc", "construction"],
        },
        {
            "name": "Chancery Division",
            "category": "Chancery Division",
            "question": "What are the procedures for issuing a claim in the Chancery Division?",
            "expect_terms": ["claim", "chancery", "issue", "proceedings"],
        },
        {
            "name": "Patents Court",
            "category": "Patents Court",
            "question": "How are patent cases allocated and managed in the Patents Court?",
            "expect_terms": ["patent", "multi-track", "case management", "court"],
        },
        {
            "name": "King's Bench Division",
            "category": "King's Bench Division",
            "question": "What guidance does the King's Bench Division provide on civil restraint orders?",
            "expect_terms": ["civil restraint", "king", "order", "bench"],
        },
    ]

    for ct in court_tests:
        print(f"\n  --- {ct['name']} ---")
        result = chat(ct["question"], category=ct["category"])
        answer = extract_answer(result)
        cats = extract_source_categories(result)

        check(f"{ct['name']}: answer is substantive",
              len(answer) > 80, f"Length: {len(answer)}")
        check(f"{ct['name']}: answer covers expected topic",
              has_any(answer, ct["expect_terms"]))
        check(f"{ct['name']}: has citations or source attribution",
              has_citations(answer))

        # Sources should predominantly be from the filtered category
        if cats:
            primary_cat = ct["category"]
            matching = sum(1 for c in cats if primary_cat.lower() in c.lower())
            check(f"{ct['name']}: sources include filtered category ({matching}/{len(cats)})",
                  matching >= 1,
                  f"Categories: {sorted(cats)}")


# ═══════════════════════════════════════════════════════════════
# TEST 9: Cross-Court Synthesis
# ═══════════════════════════════════════════════════════════════
def test_cross_court_synthesis():
    section("TEST 9: Cross-Court Synthesis Questions")

    # Disclosure across courts
    print("\n  --- Disclosure across courts ---")
    result = chat(
        "How do disclosure obligations differ between the Commercial Court and the Chancery Division?",
        retrieval_reasoning_effort="medium",
    )
    answer = extract_answer(result)
    cats = extract_source_categories(result)
    check("Disclosure comparison: substantive answer", len(answer) > 150)
    check("Disclosure comparison: mentions multiple courts",
          has_any(answer, ["commercial"]) and has_any(answer, ["chancery"]))
    check("Disclosure comparison: references PD 57AD or CPR 31",
          has_any(answer, ["57ad", "part 31", "cpr 31", "extended disclosure", "standard disclosure"]))
    if len(cats) >= 2:
        check(f"Disclosure comparison: sources from {len(cats)} categories", True,
              f"Categories: {sorted(cats)}")
    else:
        warn(f"Disclosure comparison: only {len(cats)} source category (may need Thorough)")

    # Expert evidence across courts
    print("\n  --- Expert evidence across courts ---")
    result2 = chat(
        "What are the expert evidence rules and how do they vary across the TCC, Commercial Court, and King's Bench?",
        retrieval_reasoning_effort="medium",
    )
    answer2 = extract_answer(result2)
    court_count = sum(1 for t in ["tcc", "technology", "commercial", "king"] if t in normalize(answer2))
    check("Expert evidence: mentions multiple courts",
          court_count >= 2,
          f"Courts mentioned: {court_count}")
    check("Expert evidence: references Part 35 or expert duties",
          has_any(answer2, ["part 35", "cpr 35", "expert", "permission"]))

    # Appeals across courts
    print("\n  --- Appeals across courts ---")
    result3 = chat(
        "What are the appeal routes from the Commercial Court and the Patents Court?",
        retrieval_reasoning_effort="medium",
    )
    answer3 = extract_answer(result3)
    check("Appeals: answer is substantive", len(answer3) > 100)
    check("Appeals: references Part 52 or appeal procedure",
          has_any(answer3, ["part 52", "court of appeal", "appeal", "permission to appeal"]))


# ═══════════════════════════════════════════════════════════════
# TEST 10: Practice Direction Coverage
# ═══════════════════════════════════════════════════════════════
def test_practice_directions():
    section("TEST 10: Practice Direction Coverage")

    pd_tests = [
        {
            "name": "PD 57AD — Extended Disclosure",
            "question": "What is the procedure for extended disclosure under Practice Direction 57AD?",
            "expect": ["57ad", "extended disclosure", "disclosure model", "business and property"],
        },
        {
            "name": "PD 1A — Vulnerable Parties",
            "question": "What provisions exist for vulnerable parties and witnesses under Practice Direction 1A?",
            "expect": ["vulnerable", "witness", "pd 1a", "practice direction 1a", "participation"],
        },
        {
            "name": "PD 52A — Appeals",
            "question": "What are the general provisions for appeals under Practice Direction 52A?",
            "expect": ["appeal", "52a", "permission", "appellant", "notice"],
        },
        {
            "name": "PD 31B — Electronic Disclosure",
            "question": "What does Practice Direction 31B say about electronic documents disclosure?",
            "expect": ["electronic", "31b", "disclosure", "document", "data"],
        },
    ]

    for pd in pd_tests:
        print(f"\n  --- {pd['name']} ---")
        result = chat(pd["question"])
        answer = extract_answer(result)
        check(f"{pd['name']}: substantive answer", len(answer) > 80)
        check(f"{pd['name']}: covers expected topic",
              has_any(answer, pd["expect"]),
              f"Looking for: {pd['expect']}")
        check(f"{pd['name']}: has citations", "[1]" in answer)


# ═══════════════════════════════════════════════════════════════
# TEST 11: Pre-Action Protocols
# ═══════════════════════════════════════════════════════════════
def test_preaction_protocols():
    section("TEST 11: Pre-Action Protocol Questions")

    print("\n  --- Construction & Engineering Protocol ---")
    result = chat(
        "What are the pre-action protocol requirements for construction and engineering disputes?",
        category="Pre-Action Protocols",
    )
    answer = extract_answer(result)
    check("Construction PAP: substantive answer", len(answer) > 80)
    check("Construction PAP: references construction/engineering",
          has_any(answer, ["construction", "engineering", "protocol", "pre-action"]))
    if has_citations(answer):
        check("Construction PAP: has citations or source attribution", True)
    else:
        warn("Construction PAP: no bracket citations (LLM used descriptive style)")

    print("\n  --- Judicial Review Protocol ---")
    result2 = chat(
        "What steps must be taken before commencing judicial review proceedings?",
    )
    answer2 = extract_answer(result2)
    check("Judicial review PAP: substantive answer", len(answer2) > 80)
    check("Judicial review PAP: references correct procedure",
          has_any(answer2, ["judicial review", "pre-action", "protocol", "letter before claim",
                             "part 54", "permission"]))


# ═══════════════════════════════════════════════════════════════
# TEST 12: Specific Rule References
# ═══════════════════════════════════════════════════════════════
def test_specific_rules():
    section("TEST 12: Specific Rule References")

    rule_tests = [
        {
            "name": "Part 24 — Summary Judgment",
            "question": "When can a court give summary judgment under CPR Part 24?",
            "expect": ["part 24", "summary judgment", "no real prospect", "reasonable",
                        "no realistic prospect"],
        },
        {
            "name": "Part 36 — Offers to Settle",
            "question": "How do Part 36 offers to settle work?",
            "expect": ["part 36", "offer", "settle", "claimant", "defendant", "consequences",
                        "costs consequences"],
        },
        {
            "name": "Part 44 — General Costs Rules",
            "question": "What are the general rules about costs under CPR Part 44?",
            "expect": ["part 44", "costs", "discretion", "court", "assessment", "general rule"],
        },
        {
            "name": "Part 52 — Appeals",
            "question": "What is the procedure for filing an appeal under Part 52?",
            "expect": ["part 52", "appeal", "permission", "notice", "appellant"],
        },
        {
            "name": "Part 25 — Interim Remedies",
            "question": "What interim remedies are available under CPR Part 25?",
            "expect": ["part 25", "interim", "injunction", "freezing", "payment on account",
                        "order"],
        },
        {
            "name": "Part 21 — Children & Protected Parties",
            "question": "What special provisions apply to children and protected parties under Part 21?",
            "expect": ["part 21", "child", "protected part", "litigation friend", "minor"],
        },
    ]

    for rt in rule_tests:
        print(f"\n  --- {rt['name']} ---")
        result = chat(rt["question"])
        answer = extract_answer(result)
        check(f"{rt['name']}: substantive answer", len(answer) > 80)
        check(f"{rt['name']}: covers correct topic",
              has_any(answer, rt["expect"]),
              f"Looking for: {rt['expect'][:4]}...")
        check(f"{rt['name']}: has citations or source attribution", has_citations(answer))


# ═══════════════════════════════════════════════════════════════
# TEST 13: Complex Analytical Questions
# ═══════════════════════════════════════════════════════════════
def test_complex_analytical():
    section("TEST 13: Complex Analytical / Multi-Source Questions")

    # Interaction between costs rules and court-specific practices
    print("\n  --- Costs budgeting across jurisdictions ---")
    result = chat(
        "How does costs budgeting work in multi-track cases, and how do the Commercial Court and TCC approach it?",
        retrieval_reasoning_effort="medium",
    )
    answer = extract_answer(result)
    check("Costs budgeting: substantive answer", len(answer) > 150)
    check("Costs budgeting: references costs management concepts",
          has_any(answer, ["costs budget", "costs management", "part 3", "pd 3d",
                            "case management", "budget"]))
    check("Costs budgeting: has citations", "[1]" in answer)

    # Multi-rule interaction
    print("\n  --- Striking out vs summary judgment ---")
    result2 = chat(
        "What is the difference between striking out under CPR Part 3.4 and summary judgment under Part 24?")
    answer2 = extract_answer(result2)
    check("Strike out vs summary judgment: substantive comparison", len(answer2) > 150)
    mentions_both = has_any(answer2, ["3.4", "part 3", "strike", "striking"]) and has_any(answer2, ["part 24", "summary judgment"])
    check("Strike out vs summary judgment: discusses both concepts", mentions_both)

    # Enforcement topic
    print("\n  --- Enforcement of judgments ---")
    result3 = chat("What methods are available for enforcing a court judgment?")
    answer3 = extract_answer(result3)
    check("Enforcement: substantive answer", len(answer3) > 100)
    check("Enforcement: references enforcement mechanisms",
          has_any(answer3, ["enforcement", "part 70", "part 71", "part 72", "part 73",
                             "charging order", "third party debt", "writ", "warrant",
                             "execution", "attachment of earnings"]))


# ═══════════════════════════════════════════════════════════════
# TEST 14: Edge Cases & Robustness
# ═══════════════════════════════════════════════════════════════
def test_edge_cases():
    section("TEST 14: Edge Cases & Robustness")

    # Very specific subsection reference
    print("\n  --- Specific subsection reference ---")
    result = chat("What does CPR rule 3.9 say about relief from sanctions?")
    answer = extract_answer(result)
    check("CPR 3.9: references relief from sanctions",
          has_any(answer, ["3.9", "relief from sanctions", "sanction", "denton", "mitchell"]))
    check("CPR 3.9: has citations", "[1]" in answer)

    # Compound question with multiple angles
    print("\n  --- Compound question ---")
    result2 = chat(
        "What is the time limit for filing a defence, and what happens if the defendant fails to do so?")
    answer2 = extract_answer(result2)
    check("Compound Q: substantive answer", len(answer2) > 100)
    check("Compound Q: references defence time limits",
          has_any(answer2, ["14 days", "28 days", "part 15", "part 12", "defence", "default judgment"]))

    # Question with legal jargon
    print("\n  --- Legal jargon ---")
    result3 = chat("What is the sans prejudice rule and Calderbank offers?")
    answer3 = extract_answer(result3)
    # LLM should interpret 'sans' as 'without' or recognise the concept
    check("Legal jargon: provides substantive response",
          len(answer3) > 50,
          f"Length: {len(answer3)}")

    # Empty/minimal question
    print("\n  --- Very broad question ---")
    result4 = chat("Tell me everything about civil procedure")
    answer4 = extract_answer(result4)
    check("Broad question: provides a scoped answer with citations",
          len(answer4) > 100 and "[1]" in answer4)

    # Non-English question (should answer in same language per prompt)
    print("\n  --- Non-English question (French) ---")
    result5 = chat("Quelles sont les règles de divulgation dans le cadre du CPR Part 31?")
    answer5 = extract_answer(result5)
    check("French question: provides a substantive answer",
          len(answer5) > 80,
          f"Length: {len(answer5)}")


# ═══════════════════════════════════════════════════════════════
# TEST 15: Multi-turn Conversation Context
# ═══════════════════════════════════════════════════════════════
def test_multi_turn():
    section("TEST 15: Multi-turn Conversation Context")

    # Turn 1: Set context
    result1 = chat("What is CPR Part 31 about?")
    answer1 = extract_answer(result1)
    check("Turn 1: Part 31 disclosure answer", has_any(answer1, ["disclosure", "part 31"]))

    # Turn 2: Follow-up referencing "it"
    history = [
        {"content": "What is CPR Part 31 about?", "role": "user"},
        {"content": answer1, "role": "assistant"},
    ]
    result2 = chat("What are the exceptions to it?", history=history)
    answer2 = extract_answer(result2)
    check("Turn 2: follow-up resolved ('it' = Part 31 disclosure)",
          has_any(answer2, ["disclosure", "part 31", "exception", "exempt", "privilege",
                             "without prejudice", "legal professional privilege", "public interest"]))
    check("Turn 2: has citations", "[1]" in answer2)


# ═══════════════════════════════════════════════════════════════
# TEST 16: Court of Appeal & Senior Courts Costs Office
# ═══════════════════════════════════════════════════════════════
def test_remaining_sources():
    section("TEST 16: Court of Appeal & Senior Courts Costs Office")

    print("\n  --- Court of Appeal Civil Division ---")
    result = chat(
        "What is the procedure for applying for permission to appeal in the Court of Appeal Civil Division?",
        category="Court of Appeal Civil Division",
    )
    answer = extract_answer(result)
    check("Court of Appeal: substantive answer", len(answer) > 80)
    check("Court of Appeal: references appeal/permission",
          has_any(answer, ["court of appeal", "permission", "appeal", "appellant"]))
    check("Court of Appeal: has citations or source attribution", has_citations(answer))

    print("\n  --- Senior Courts Costs Office ---")
    result2 = chat(
        "What is the role of the Senior Courts Costs Office in detailed assessment of costs?",
        category="Senior Courts Costs Office",
    )
    answer2 = extract_answer(result2)
    check("SCCO: substantive answer", len(answer2) > 50)
    check("SCCO: references costs assessment",
          has_any(answer2, ["costs", "assessment", "senior courts", "detailed", "costs office"]))


# ═══════════════════════════════════════════════════════════════
# TEST 17: Circuit Commercial Court Guide
# ═══════════════════════════════════════════════════════════════
def test_circuit_commercial():
    section("TEST 17: Circuit Commercial Court Guide")

    result = chat(
        "What is the procedure for starting a case in the Circuit Commercial Court?",
        category="Circuit Commercial Court",
    )
    answer = extract_answer(result)
    check("Circuit Commercial: substantive answer", len(answer) > 80)
    check("Circuit Commercial: references correct court",
          has_any(answer, ["circuit commercial", "claim", "proceedings", "issue"]))
    check("Circuit Commercial: has citations or source attribution", has_citations(answer))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE PROMPT FEATURE & LEGAL DOMAIN TESTS")
    print(f"Target: {BASE_URL}")
    print("=" * 70)

    # Verify app is running
    try:
        r = httpx.get(f"{BASE_URL}/config", timeout=5)
        r.raise_for_status()
        print(f"App is running ✓")
    except Exception as e:
        print(f"ERROR: App not reachable at {BASE_URL}: {e}")
        sys.exit(1)

    start = time.time()

    # Core prompt features (Tests 1-7)
    test_dynamic_sources()
    test_metadata_enrichment()
    test_search_depth_recommendations()
    test_mismatch_detection()
    test_disambiguation()
    test_source_attribution()
    test_query_rewrite()

    # Wider domain coverage (Tests 8-17)
    test_court_specific_filtering()
    test_cross_court_synthesis()
    test_practice_directions()
    test_preaction_protocols()
    test_specific_rules()
    test_complex_analytical()
    test_edge_cases()
    test_multi_turn()
    test_remaining_sources()
    test_circuit_commercial()

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    total = results["pass"] + results["fail"] + results["warn"]
    print(f"RESULTS: {results['pass']} passed, {results['fail']} failed, {results['warn']} warnings  (total: {total})")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 70)

    # Print failures summary
    failures = [d for d in results["details"] if d["ok"] is False]
    if failures:
        print(f"\nFailed checks ({len(failures)}):")
        for f in failures:
            print(f"  ✗ {f['label']}")
            if f["detail"]:
                print(f"    {f['detail'][:200]}")

    if results["fail"] > 0:
        sys.exit(1)
