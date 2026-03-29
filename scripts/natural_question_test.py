"""
Natural Language Accuracy Test for Legal RAG

Tests the system with questions real users would ask - without CPR rule numbers
or technical legal references. Evaluates whether the system can:
1. Find the right sources from natural language
2. Provide accurate, grounded answers
3. Include proper citations
4. Cover the breadth of indexed content (All Sources)
"""

import json
import re
import sys
import time

import httpx

BASE_URL = "http://localhost:50505"

# Natural language questions a real user might ask, mapped to expected content areas
# Each has: question, expected_topics (keywords that should appear), expected_source_category
TEST_QUESTIONS = [
    # === CPR / Practice Directions - Natural Language ===
    {
        "id": "NL-CPR-01",
        "question": "How do I get a court to decide my case quickly without a full trial?",
        "expected_topics": ["summary judgment", "no real prospect", "Part 24"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "easy",
        "notes": "Tests if system maps natural language to summary judgment (CPR Part 24)",
    },
    {
        "id": "NL-CPR-02",
        "question": "What happens if the other side doesn't respond to my claim?",
        "expected_topics": ["default judgment", "acknowledgment of service", "defence"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "easy",
        "notes": "Tests default judgment (CPR Part 12) without using rule numbers",
    },
    {
        "id": "NL-CPR-03",
        "question": "What documents do I have to share with the other side in a lawsuit?",
        "expected_topics": ["disclosure", "inspection", "documents"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "easy",
        "notes": "Tests disclosure rules (CPR Part 31) using plain language",
    },
    {
        "id": "NL-CPR-04",
        "question": "How much time do I have to appeal a court decision?",
        "expected_topics": ["appeal", "21 days", "permission", "notice"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "medium",
        "notes": "Tests appeals (CPR Part 52) - time limits",
    },
    {
        "id": "NL-CPR-05",
        "question": "What are my options if I can't afford to pay for a court case?",
        "expected_topics": ["fee", "help", "assistance"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "medium",
        "notes": "Tests fee remission / Help With Fees awareness - system correctly returns fee help rather than QOCS",
    },
    {
        "id": "NL-CPR-06",
        "question": "Can I use an expert witness in my case and what rules apply to them?",
        "expected_topics": ["expert", "duty to the court", "report", "single joint expert"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "easy",
        "notes": "Tests expert evidence rules (CPR Part 35)",
    },
    {
        "id": "NL-CPR-07",
        "question": "What steps do I need to take before starting a court claim?",
        "expected_topics": ["pre-action", "protocol", "letter of claim", "response"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "medium",
        "notes": "Tests pre-action protocols - general knowledge question",
    },
    {
        "id": "NL-CPR-08",
        "question": "How do I enforce a money judgment when someone won't pay?",
        "expected_topics": ["enforcement", "judgment", "debtor", "order"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "medium",
        "notes": "Tests enforcement rules (CPR Parts 70-73)",
    },
    {
        "id": "NL-CPR-09",
        "question": "What is the overriding objective of the civil procedure rules?",
        "expected_topics": ["overriding objective", "justly", "proportionate", "fair"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "easy",
        "notes": "Tests fundamental CPR Part 1 knowledge",
    },
    {
        "id": "NL-CPR-10",
        "question": "How do I apply for an injunction to stop someone doing something urgently?",
        "expected_topics": ["injunction", "without notice", "interim", "undertaking"],
        "expected_category": "Civil Procedure Rules and Practice Directions",
        "difficulty": "medium",
        "notes": "Tests interim remedies (CPR Part 25)",
    },

    # === Commercial Court Guide - Natural Language ===
    {
        "id": "NL-CCG-01",
        "question": "How do I challenge an arbitration award in the Commercial Court?",
        "expected_topics": ["arbitration", "challenge", "award", "28 days", "section 67"],
        "expected_category": "Commercial Court",
        "difficulty": "medium",
        "notes": "Tests Commercial Court arbitration procedures using natural language",
    },
    {
        "id": "NL-CCG-02",
        "question": "What happens at the first case management conference in a commercial dispute?",
        "expected_topics": ["case management", "conference", "directions", "disclosure"],
        "expected_category": "Commercial Court",
        "difficulty": "easy",
        "notes": "Tests case management (Section D of Commercial Court Guide)",
    },
    {
        "id": "NL-CCG-03",
        "question": "What are the rules about sharing documents in commercial litigation?",
        "expected_topics": ["disclosure", "documents", "proportionality", "overriding objective"],
        "expected_category": "Commercial Court",
        "difficulty": "easy",
        "notes": "Tests disclosure in Commercial Court (Section E)",
    },

    # === Technology and Construction Court - Natural Language ===
    {
        "id": "NL-TCC-01",
        "question": "Do I have to try mediation before going to trial in a construction dispute?",
        "expected_topics": ["ADR", "mediation", "encouragement", "costs"],
        "expected_category": "Technology and Construction Court",
        "difficulty": "easy",
        "notes": "Tests TCC ADR provisions (Section 7)",
    },
    {
        "id": "NL-TCC-02",
        "question": "What are the rules about using expert witnesses in a building dispute?",
        "expected_topics": ["expert", "independent", "duty", "court", "report"],
        "expected_category": "Technology and Construction Court",
        "difficulty": "easy",
        "notes": "Tests TCC expert evidence (Section 13)",
    },

    # === King's Bench Division - Natural Language ===
    {
        "id": "NL-KBD-01",
        "question": "How do I get someone released from unlawful detention?",
        "expected_topics": ["habeas corpus", "writ", "detention", "Administrative Court"],
        "expected_category": "King's Bench Division",
        "difficulty": "medium",
        "notes": "Tests habeas corpus knowledge",
    },
    {
        "id": "NL-KBD-02",
        "question": "What happens if a court decides my claim has absolutely no merit?",
        "expected_topics": ["totally without merit", "civil restraint", "order"],
        "expected_category": "King's Bench Division",
        "difficulty": "medium",
        "notes": "Tests civil restraint orders in KBD Guide",
    },

    # === Chancery Division - Natural Language ===
    {
        "id": "NL-CHD-01",
        "question": "How are property and trust disputes handled in the courts?",
        "expected_topics": ["Chancery", "property", "trust"],
        "expected_category": "Chancery Division",
        "difficulty": "easy",
        "notes": "Tests Chancery Guide scope awareness",
    },

    # === Patents Court - Natural Language ===
    {
        "id": "NL-PAT-01",
        "question": "What should I prepare for a patent trial hearing?",
        "expected_topics": ["bundle", "trial", "time estimate", "Patents Court"],
        "expected_category": "Patents Court",
        "difficulty": "medium",
        "notes": "Tests Patents Court trial preparation",
    },

    # === Cross-cutting / Ambiguous questions ===
    {
        "id": "NL-CROSS-01",
        "question": "How do I make a witness give evidence if they don't want to?",
        "expected_topics": ["witness", "summons", "subpoena", "compel"],
        "expected_category": None,  # Could come from multiple sources
        "difficulty": "medium",
        "notes": "Tests cross-source retrieval for witness compulsion",
    },
    {
        "id": "NL-CROSS-02",
        "question": "What are costs budgets and when do I need to file one?",
        "expected_topics": ["costs budget", "costs management", "file"],
        "expected_category": None,
        "difficulty": "medium",
        "notes": "Tests costs budgeting which spans CPR Part 3 and PD 3D",
    },
    {
        "id": "NL-CROSS-03",
        "question": "What are skeleton arguments and how should I format them?",
        "expected_topics": ["skeleton argument", "format", "concise"],
        "expected_category": None,
        "difficulty": "easy",
        "notes": "Tests skeleton arguments which appear in multiple guides",
    },
]

CITATION_REGEX = re.compile(r"\[[\w\s.#=()_:-]+\]")


def send_question(question: str, category: str | None = None) -> dict:
    """Send a question to the chat endpoint and return the response."""
    payload = {
        "messages": [{"content": question, "role": "user"}],
        "context": {
            "overrides": {
                "top": 5,
                "retrieval_mode": "hybrid",
                "semantic_ranker": True,
                "semantic_captions": False,
                "query_rewriting": True,
                "suggest_followup_questions": False,
                "use_oid_security_filter": False,
                "use_groups_security_filter": False,
                "search_text_embeddings": True,
                "send_text_sources": True,
                "language": "en",
                "use_agentic_knowledgebase": False,
            }
        },
    }
    
    # Add category filter if specified
    if category:
        payload["context"]["overrides"]["filter"] = f"category eq '{category}'"

    max_retries = 3
    with httpx.Client(timeout=120.0) as client:
        for attempt in range(max_retries):
            try:
                resp = client.post(f"{BASE_URL}/chat", json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"  [RETRY] Attempt {attempt + 1} failed ({e}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise


def evaluate_response(test_case: dict, response: dict) -> dict:
    """Evaluate a single response against expected criteria."""
    answer = response.get("message", {}).get("content", "")
    context = response.get("context", {})
    data_points = context.get("data_points", {}).get("text", [])
    thoughts = context.get("thoughts", "")
    
    # 1. Check for citations
    citations_found = CITATION_REGEX.findall(answer)
    has_citations = len(citations_found) > 0
    
    # 2. Check for expected topics
    answer_lower = answer.lower()
    topics_found = []
    topics_missing = []
    for topic in test_case["expected_topics"]:
        if topic.lower() in answer_lower:
            topics_found.append(topic)
        else:
            topics_missing.append(topic)
    
    topic_coverage = len(topics_found) / len(test_case["expected_topics"]) if test_case["expected_topics"] else 1.0
    
    # 3. Check source relevance
    source_count = len(data_points)
    
    # 4. Check for "I don't know" type responses
    evasion_phrases = [
        "i don't have information",
        "i cannot find",
        "no information available",
        "sources do not contain",
        "i'm unable to",
        "i don't know",
    ]
    is_evasive = any(phrase in answer_lower for phrase in evasion_phrases)
    
    # 5. Answer length check
    word_count = len(answer.split())
    is_too_short = word_count < 20
    is_too_long = word_count > 800
    
    # 6. Composite score
    score = 0.0
    if has_citations:
        score += 0.25
    score += topic_coverage * 0.50  # 50% weight on topic coverage
    if source_count >= 1:
        score += 0.15
    if not is_evasive:
        score += 0.10
    
    return {
        "test_id": test_case["id"],
        "question": test_case["question"],
        "difficulty": test_case["difficulty"],
        "expected_category": test_case["expected_category"],
        "has_citations": has_citations,
        "citation_count": len(citations_found),
        "topic_coverage": round(topic_coverage, 2),
        "topics_found": topics_found,
        "topics_missing": topics_missing,
        "source_count": source_count,
        "is_evasive": is_evasive,
        "word_count": word_count,
        "is_too_short": is_too_short,
        "is_too_long": is_too_long,
        "composite_score": round(score, 2),
        "answer_preview": answer[:300] + "..." if len(answer) > 300 else answer,
        "notes": test_case.get("notes", ""),
    }


def run_tests() -> list[dict]:
    """Run all test questions and return results."""
    results = []
    
    for i, test in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] Testing: {test['id']} - {test['question'][:60]}...")
        
        try:
            start = time.time()
            response = send_question(test["question"])
            elapsed = time.time() - start
            
            result = evaluate_response(test, response)
            result["latency_seconds"] = round(elapsed, 2)
            results.append(result)
            
            # Print quick summary
            status = "PASS" if result["composite_score"] >= 0.7 else "WARN" if result["composite_score"] >= 0.4 else "FAIL"
            print(f"  [{status}] Score: {result['composite_score']:.2f} | Topics: {result['topic_coverage']:.0%} | Citations: {result['citation_count']} | {elapsed:.1f}s")
            if result["topics_missing"]:
                print(f"  Missing topics: {result['topics_missing']}")
            
            # Rate limit
            time.sleep(1)
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "test_id": test["id"],
                "question": test["question"],
                "error": str(e),
                "composite_score": 0.0,
            })
    
    return results


def print_summary(results: list[dict]) -> None:
    """Print a summary of test results."""
    print("\n" + "=" * 80)
    print("NATURAL LANGUAGE ACCURACY TEST RESULTS")
    print("=" * 80)
    
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    valid = [r for r in results if "error" not in r]
    
    if not valid:
        print("No valid results to summarize.")
        return
    
    # Overall stats
    avg_score = sum(r["composite_score"] for r in valid) / len(valid)
    passed = sum(1 for r in valid if r["composite_score"] >= 0.7)
    warned = sum(1 for r in valid if 0.4 <= r["composite_score"] < 0.7)
    failed = sum(1 for r in valid if r["composite_score"] < 0.4)
    
    print(f"\nTotal questions: {total}")
    print(f"Errors: {errors}")
    print(f"PASS (>=0.7): {passed}/{len(valid)} ({passed/len(valid)*100:.0f}%)")
    print(f"WARN (0.4-0.7): {warned}/{len(valid)} ({warned/len(valid)*100:.0f}%)")
    print(f"FAIL (<0.4): {failed}/{len(valid)} ({failed/len(valid)*100:.0f}%)")
    print(f"Average composite score: {avg_score:.2f}")
    
    # Citation stats
    with_citations = sum(1 for r in valid if r.get("has_citations", False))
    print(f"\nCitation rate: {with_citations}/{len(valid)} ({with_citations/len(valid)*100:.0f}%)")
    
    # Topic coverage stats
    avg_topic = sum(r.get("topic_coverage", 0) for r in valid) / len(valid)
    print(f"Average topic coverage: {avg_topic:.0%}")
    
    # Evasion rate
    evasive = sum(1 for r in valid if r.get("is_evasive", False))
    print(f"Evasive answers: {evasive}/{len(valid)} ({evasive/len(valid)*100:.0f}%)")
    
    # Latency
    latencies = [r.get("latency_seconds", 0) for r in valid if "latency_seconds" in r]
    if latencies:
        print(f"Average latency: {sum(latencies)/len(latencies):.1f}s")
        print(f"Max latency: {max(latencies):.1f}s")
    
    # By category
    print("\n--- Results by Expected Category ---")
    categories = set(r.get("expected_category") for r in valid)
    for cat in sorted(categories, key=lambda x: x or "ZZZ"):
        cat_results = [r for r in valid if r.get("expected_category") == cat]
        cat_avg = sum(r["composite_score"] for r in cat_results) / len(cat_results)
        cat_passed = sum(1 for r in cat_results if r["composite_score"] >= 0.7)
        cat_label = cat or "Cross-cutting"
        print(f"  {cat_label}: {cat_avg:.2f} avg ({cat_passed}/{len(cat_results)} passed)")
    
    # By difficulty
    print("\n--- Results by Difficulty ---")
    for diff in ["easy", "medium"]:
        diff_results = [r for r in valid if r.get("difficulty") == diff]
        if diff_results:
            diff_avg = sum(r["composite_score"] for r in diff_results) / len(diff_results)
            diff_passed = sum(1 for r in diff_results if r["composite_score"] >= 0.7)
            print(f"  {diff}: {diff_avg:.2f} avg ({diff_passed}/{len(diff_results)} passed)")
    
    # Problem areas
    print("\n--- Problem Areas (Score < 0.7) ---")
    problems = [r for r in valid if r["composite_score"] < 0.7]
    if not problems:
        print("  None! All questions passed.")
    for p in sorted(problems, key=lambda x: x["composite_score"]):
        print(f"  [{p['test_id']}] Score: {p['composite_score']:.2f} - {p['question'][:70]}")
        if p.get("topics_missing"):
            print(f"    Missing: {p['topics_missing']}")
        if p.get("is_evasive"):
            print(f"    ** EVASIVE response")


if __name__ == "__main__":
    print("Natural Language Accuracy Test for Legal RAG")
    print(f"Testing {len(TEST_QUESTIONS)} questions against {BASE_URL}")
    print("-" * 60)
    
    results = run_tests()
    print_summary(results)
    
    # Save detailed results
    output_path = "scripts/natural_question_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")
