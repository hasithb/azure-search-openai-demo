#!/usr/bin/env python3
"""A/B compare the current chat flow against an experimental planner-validator-repair flow.

The experimental path does not change app behavior. It bootstraps the same backend
clients/models, uses the existing query rewrite prompt as the planner, runs a
single LLM validation step over the retrieved sources, optionally performs one
repair search, and then answers with the normal answer prompt plus a small
experimental injection.

Run with:
  /Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2/.venv-upgrade/bin/python scripts/test_planner_ab_compare.py

This script loads environment variables from the repo-root .env file, matching
app/start.sh.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "app" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from quart import current_app  # type: ignore

from app import close_clients, create_app, setup_clients  # type: ignore
from config import CONFIG_CHAT_APPROACH  # type: ignore


DEFAULT_OVERRIDES = {
    "retrieval_mode": "hybrid",
    "semantic_ranker": True,
    "semantic_captions": False,
    "top": 5,
    "suggest_followup_questions": False,
    "temperature": 0.0,
    "seed": 42,
}


VALIDATOR_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "assess_retrieval",
            "description": "Assess whether the retrieved legal sources are sufficient and on-point for the user's question, and suggest one improved search query if they are not.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_sufficient": {
                        "type": "boolean",
                        "description": "True when the retrieved sources likely include the primary authoritative material needed for a grounded answer."
                    },
                    "primary_authority_reference": {
                        "type": "string",
                        "description": "The most likely primary authority based on your general legal knowledge, such as 'CPR 31.16', 'Part 24', 'Part 52', or a specific court guide section."
                    },
                    "recommended_category": {
                        "type": "string",
                        "description": "Exact source category name to prefer for a repair search, or an empty string if no category preference is needed."
                    },
                    "repaired_search_query": {
                        "type": "string",
                        "description": "A single, more precise search query to run if the evidence is insufficient, or an empty string if no retry is needed."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short explanation of what is missing or why the current retrieval is sufficient."
                    }
                },
                "required": [
                    "evidence_sufficient",
                    "primary_authority_reference",
                    "recommended_category",
                    "repaired_search_query",
                    "reason"
                ]
            }
        }
    }
]


EXPERIMENTAL_ANSWER_INJECTION = """
Experimental retrieval mode instructions:
- If the retrieved sources do not appear to include the primary authority for the user's question, say so explicitly before giving any partial answer.
- For broad or ambiguous questions, either state the interpretation you are answering or clearly say that the sources support only a partial answer.
- For out-of-scope questions, do not improvise a legal answer; say the available sources do not cover the request.
- Prefer the most authoritative retrieved source and clearly distinguish court-specific guidance from general CPR/Practice Direction rules.
""".strip()


TEST_CASES = [
    {
        "name": "PAD",
        "question": "What is PAD?",
        "checks": [
            {"type": "contains_any", "terms": ["31.16", "cpr 31.16"], "description": "References CPR 31.16"},
            {"type": "contains_any", "terms": ["pre-action disclosure"], "description": "Identifies PAD as pre-action disclosure"},
            {
                "type": "not_contains_any",
                "terms": [
                    "Practice Direction – Pre-Action Conduct and Protocols",
                    "relevant document is the **Practice Direction",
                    "pre-action protocols explain",
                ],
                "description": "Does not answer PAD with protocol-only material",
            },
        ],
    },
    {
        "name": "Standard Disclosure",
        "question": "What is standard disclosure?",
        "checks": [
            {"type": "contains_any", "terms": ["CPR 31", "Part 31", "31.6"], "description": "References Part 31"},
            {"type": "contains_any", "terms": ["documents", "disclose", "disclosure"], "description": "Explains disclosure concept"},
        ],
    },
    {
        "name": "Summary Judgment",
        "question": "How do I apply for summary judgment?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 24", "24.2", "no real prospect"], "description": "References CPR Part 24"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
    },
    {
        "name": "Appeal Time Limits",
        "question": "What are the time limits for filing an appeal?",
        "checks": [
            {"type": "contains_any", "terms": ["21 days", "Part 52", "appellant"], "description": "References Part 52 time limits"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
    },
    {
        "name": "Costs Broad",
        "question": "How do costs work?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 44", "Part 45", "Part 46", "Part 47", "costs budgeting", "costs management"], "description": "Addresses a concrete costs regime"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
    },
    {
        "name": "Broad Time Limits",
        "question": "What are the time limits?",
        "checks": [
            {"type": "contains_any", "terms": ["specific", "depend", "which", "context", "various", "clarify", "time limit", "days", "Part"], "description": "Answers cautiously or clarifies scope"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
    },
    {
        "name": "Service of Claim Form",
        "question": "How do I serve a claim form?",
        "checks": [
            {"type": "contains_any", "terms": ["Part 6", "6.3", "6.4", "6.5", "service"], "description": "References CPR Part 6"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
    },
    {
        "name": "Weather Out of Scope",
        "question": "What is the weather forecast for London?",
        "checks": [
            {"type": "contains_any", "terms": ["not", "cannot", "no information", "available sources", "don't have", "does not"], "description": "Clearly notes the request is unsupported by the sources"},
        ],
    },
    {
        "name": "Commercial Court with CPR Filter",
        "question": "How does the Commercial Court handle case management conferences?",
        "overrides": {"include_category": "Civil Procedure Rules and Practice Directions"},
        "checks": [
            {"type": "contains_any", "terms": ["Commercial Court Guide", "Commercial Court", "Part 58", "do not contain Commercial Court-specific"], "description": "Recognizes Commercial Court context or mismatch"},
            {"type": "has_citation", "description": "Has at least one citation"},
        ],
    },
]


def load_repo_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def merge_overrides(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_OVERRIDES)
    if extra:
        merged.update(extra)
    return merged


def escape_odata(value: str) -> str:
    return value.replace("'", "''")


def extract_tool_args(chat_completion: Any) -> dict[str, Any]:
    message = chat_completion.choices[0].message
    if not getattr(message, "tool_calls", None):
        return {}
    for tool_call in message.tool_calls:
        if getattr(tool_call, "type", "") != "function":
            continue
        payload = tool_call.function.arguments or "{}"
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def summarize_documents(documents: list[Any], limit: int = 5) -> list[dict[str, str]]:
    summary = []
    for doc in documents[:limit]:
        summary.append(
            {
                "category": str(getattr(doc, "category", "") or ""),
                "sourcepage": str(getattr(doc, "sourcepage", "") or ""),
                "sourcefile": str(getattr(doc, "sourcefile", "") or ""),
                "subsection_id": str(getattr(doc, "subsection_id", "") or ""),
                "snippet": str(getattr(doc, "content", "") or "")[:280],
            }
        )
    return summary


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def extract_reference_terms(primary_reference: str) -> list[str]:
    primary_reference = primary_reference.strip()
    if not primary_reference:
        return []

    terms = {normalize_text(primary_reference)}
    cpr_rule_match = re.search(r"\bcpr\s+(\d+)\.(\d+[a-z]?)\b", primary_reference, re.IGNORECASE)
    if cpr_rule_match:
        part_num, rule_num = cpr_rule_match.groups()
        terms.add(normalize_text(f"{part_num}.{rule_num}"))
        terms.add(normalize_text(f"Part {part_num}"))

    cpr_part_match = re.search(r"\bpart\s+(\d+[a-z]?)\b", primary_reference, re.IGNORECASE)
    if cpr_part_match:
        terms.add(normalize_text(f"Part {cpr_part_match.group(1)}"))

    return [term for term in terms if term]


def extract_strict_reference_terms(primary_reference: str) -> list[str]:
    primary_reference = primary_reference.strip()
    if not primary_reference:
        return []

    cpr_rule_match = re.search(r"\bcpr\s+(\d+)\.(\d+[a-z]?)\b", primary_reference, re.IGNORECASE)
    if cpr_rule_match:
        part_num, rule_num = cpr_rule_match.groups()
        return [normalize_text(f"CPR {part_num}.{rule_num}"), normalize_text(f"{part_num}.{rule_num}")]

    cpr_part_match = re.search(r"\bpart\s+(\d+[a-z]?)\b", primary_reference, re.IGNORECASE)
    if cpr_part_match:
        return [normalize_text(f"Part {cpr_part_match.group(1)}")]

    return [normalize_text(primary_reference)]


def score_authority_match(doc: Any, primary_reference: str, recommended_category: str) -> int:
    category = str(getattr(doc, "category", "") or "")
    source_haystack = normalize_text(
        " ".join(
            [
                str(getattr(doc, "subsection_id", "") or ""),
                str(getattr(doc, "sourcepage", "") or ""),
                str(getattr(doc, "sourcefile", "") or ""),
                category,
            ]
        )
    )
    content_haystack = normalize_text(str(getattr(doc, "content", "") or "")[:1200])

    score = 0
    if recommended_category and category == recommended_category:
        score += 3

    for term in extract_reference_terms(primary_reference):
        if term and term in source_haystack:
            score += 10
        elif term and term in content_haystack:
            score += 4

    return score


def has_reference_in_source_metadata(doc: Any, primary_reference: str) -> bool:
    if not primary_reference:
        return False

    source_haystack = normalize_text(
        " ".join(
            [
                str(getattr(doc, "subsection_id", "") or ""),
                str(getattr(doc, "sourcepage", "") or ""),
                str(getattr(doc, "sourcefile", "") or ""),
            ]
        )
    )
    return any(term in source_haystack for term in extract_strict_reference_terms(primary_reference))


def is_primary_authority_hit(doc: Any, primary_reference: str, recommended_category: str) -> bool:
    category = str(getattr(doc, "category", "") or "")
    if recommended_category and category != recommended_category:
        return False
    return has_reference_in_source_metadata(doc, primary_reference)


def build_repair_query_candidates(primary_reference: str, repaired_query: str) -> list[str]:
    candidates: list[str] = []
    primary_reference = primary_reference.strip()
    repaired_query = repaired_query.strip()

    if repaired_query:
        candidates.append(repaired_query)
    if primary_reference and repaired_query:
        candidates.append(f"{primary_reference} {repaired_query}")
    if primary_reference:
        candidates.append(f"{primary_reference} Civil Procedure Rules")

    cpr_rule_match = re.search(r"\bcpr\s+(\d+)\.(\d+[a-z]?)\b", primary_reference, re.IGNORECASE)
    if cpr_rule_match:
        part_num, rule_num = cpr_rule_match.groups()
        candidates.append(f"CPR {part_num}.{rule_num} Part {part_num} {repaired_query}".strip())
        candidates.append(f"Part {part_num} {part_num}.{rule_num} {repaired_query}".strip())
        candidates.append(f"Part {part_num} {part_num}.{rule_num} Civil Procedure Rules".strip())

    if primary_reference and primary_reference not in candidates:
        candidates.append(primary_reference)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate.strip())
    return deduped


def run_check(check: dict[str, Any], answer: str) -> tuple[bool, str]:
    check_type = check["type"]
    description = check["description"]
    lowered = answer.lower()

    if check_type == "contains_any":
        passed = any(term.lower() in lowered for term in check["terms"])
    elif check_type == "not_contains_any":
        passed = all(term.lower() not in lowered for term in check["terms"])
    elif check_type == "has_citation":
        passed = bool(re.search(r"\[\d+\]", answer))
    elif check_type == "not_contains":
        sentences = [sentence.strip() for sentence in answer.split(".") if sentence.strip()]
        if not sentences:
            passed = True
        else:
            passed = True
            for term in check["terms"]:
                hits = sum(1 for sentence in sentences if term.lower() in sentence.lower())
                if hits > len(sentences) * 0.5:
                    passed = False
                    break
    else:
        passed = False

    return passed, description


@asynccontextmanager
async def bootstrapped_approach():
    load_repo_env()
    app = create_app()
    async with app.app_context():
        await setup_clients()
        try:
            yield current_app.config[CONFIG_CHAT_APPROACH]
        finally:
            await close_clients()


async def run_baseline(approach: Any, question: str, overrides: dict[str, Any]) -> dict[str, Any]:
    messages = [{"role": "user", "content": question}]
    result = await approach.run(messages, context={"overrides": overrides, "auth_claims": {}})
    answer = result.get("message", {}).get("content", "")
    thoughts = result.get("context", {}).get("thoughts", [])
    generated_query = ""
    for thought in thoughts:
        title = thought.get("title") if isinstance(thought, dict) else getattr(thought, "title", "")
        description = thought.get("description", "") if isinstance(thought, dict) else getattr(thought, "description", "")
        if title == "Search using generated search query":
            generated_query = str(description)
            break
    return {
        "answer": answer,
        "generated_query": generated_query,
        "context": result.get("context", {}),
    }


async def validate_retrieval_plan(
    approach: Any,
    question: str,
    overrides: dict[str, Any],
    rewrite_args: dict[str, Any],
    query_text: str,
    documents: list[Any],
) -> dict[str, Any]:
    system = (
        "You are validating a legal retrieval plan for English civil procedure. "
        "Use your general legal knowledge plus the retrieved source metadata/snippets to decide whether the likely primary authority has actually been retrieved. "
        "If the retrieval looks incomplete, off-target, or missing the best authority, propose one better search query and at most one preferred source category. "
        "Be strict: for a general CPR/Practice Direction question, a court guide or other secondary source mentioning the rule is NOT sufficient if the primary CPR/PD source itself has not been retrieved. "
        "Only mark evidence_sufficient=true when the retrieved set contains the primary authority itself, unless the user explicitly asked a court-specific question or selected a court-specific filter."
    )
    payload = {
        "question": question,
        "planner_legal_concept_analysis": rewrite_args.get("legal_concept_analysis", ""),
        "planner_search_query": query_text,
        "selected_source_filter": overrides.get("include_category", ""),
        "available_source_categories": approach.available_sources,
        "top_sources": summarize_documents(documents),
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    completion = await approach.create_chat_completion(
        approach.chatgpt_deployment,
        approach.chatgpt_model,
        messages,
        overrides,
        response_token_limit=500,
        should_stream=False,
        tools=VALIDATOR_TOOL,
        tool_choice={"type": "function", "function": {"name": "assess_retrieval"}},
        temperature=0.0,
        reasoning_effort=approach.get_lowest_reasoning_effort(approach.chatgpt_model),
    )
    return extract_tool_args(completion)


def prioritize_authority_documents(documents: list[Any], validator: dict[str, Any]) -> list[Any]:
    primary_reference = str(validator.get("primary_authority_reference", "") or "").strip()
    recommended_category = str(validator.get("recommended_category", "") or "").strip()
    if not primary_reference and not recommended_category:
        return documents

    ranked = sorted(
        documents,
        key=lambda doc: (
            score_authority_match(doc, primary_reference, recommended_category),
            float(getattr(doc, "reranker_score", 0.0) or 0.0),
            float(getattr(doc, "score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ranked


async def answer_with_documents(
    approach: Any,
    question: str,
    overrides: dict[str, Any],
    documents: list[Any],
    validator: dict[str, Any],
    missing_primary_authority: bool,
) -> tuple[str, Any]:
    data_points = await approach.get_sources_content(
        documents,
        use_semantic_captions=bool(overrides.get("semantic_captions")),
        include_text_sources=bool(overrides.get("send_text_sources", True)),
        download_image_sources=bool(overrides.get("send_image_sources", False)),
        user_oid=None,
        query_hint=question,
    )
    search_depth_labels = {"minimal": "Quick", "low": "Standard", "medium": "Thorough"}
    current_search_depth = search_depth_labels.get(
        overrides.get("retrieval_reasoning_effort", approach.retrieval_reasoning_effort or ""),
        "",
    )

    primary_reference = str(validator.get("primary_authority_reference", "") or "").strip()
    answer_injection = EXPERIMENTAL_ANSWER_INJECTION
    if primary_reference:
        answer_injection = (
            answer_injection
            + "\n- The planner identified the likely primary authority as "
            + primary_reference
            + ". Prefer sources that explicitly address that authority over acronym collisions or tangential mentions."
        )
    if missing_primary_authority and primary_reference:
        answer_injection = (
            answer_injection
            + "\n- The retrieved sources still do not contain the primary authority in source metadata."
            + " Do not define the concept from similarly named or tangential sources."
            + " State the gap explicitly, name the likely authority, and recommend refining the search to that authority."
        )

    answer_messages = approach.prompt_manager.build_conversation(
        system_template_path="chat_answer.system.jinja2",
        system_template_variables=approach.get_system_prompt_variables(">>>" + answer_injection)
        | {
            "include_follow_up_questions": False,
            "image_sources": data_points.images,
            "citations": [str(i) for i in range(1, len(data_points.text or []) + 1)],
            "search_depth": current_search_depth,
            "available_sources": approach.available_sources,
            "selected_source_filter": overrides.get("include_category", ""),
            "court_specific_sources_only_hint": approach.shouldFlagCourtSpecificSourcesOnly(
                question,
                overrides.get("include_category", ""),
                data_points.text or [],
            ),
        },
        user_template_path="chat_answer.user.jinja2",
        user_template_variables={
            "user_query": question,
            "text_sources": approach.format_text_sources_for_prompt(data_points.text),
        },
        past_messages=[],
    )

    completion = await approach.create_chat_completion(
        approach.chatgpt_deployment,
        approach.chatgpt_model,
        answer_messages,
        overrides,
        response_token_limit=approach.get_response_token_limit(approach.chatgpt_model, 1024),
        should_stream=False,
    )
    return completion.choices[0].message.content or "", data_points


async def run_experimental(approach: Any, question: str, overrides: dict[str, Any]) -> dict[str, Any]:
    original_user_query = question
    rewrite_result = await approach.rewrite_query(
        prompt_template="query_rewrite.system.jinja2",
        prompt_variables={
            "user_query": original_user_query,
            "past_messages": [],
            "available_sources": approach.available_sources,
        },
        overrides=overrides,
        chatgpt_model=approach.chatgpt_model,
        chatgpt_deployment=approach.chatgpt_deployment,
        user_query=original_user_query,
        response_token_limit=approach.get_response_token_limit(approach.chatgpt_model, 300),
        tools=approach.query_rewrite_tools,
        temperature=0.0,
        no_response_token=approach.NO_RESPONSE,
    )
    rewrite_args = approach.extract_rewrite_function_arguments(rewrite_result.completion)
    query_text = approach._merge_rewritten_query_with_explicit_references(
        rewrite_result.query,
        original_user_query,
        rewrite_result.subsection_hint,
    )

    use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
    use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
    use_semantic_ranker = True if overrides.get("semantic_ranker") else False
    use_semantic_captions = True if overrides.get("semantic_captions") else False
    top = overrides.get("top", 5)
    minimum_search_score = overrides.get("minimum_search_score", 0.0)
    minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
    search_index_filter = approach.build_filter(overrides)
    search_text_embeddings = overrides.get("search_text_embeddings", True)
    search_image_embeddings = (
        overrides.get("search_image_embeddings", approach.multimodal_enabled) and approach.multimodal_enabled
    )

    vectors = []
    if use_vector_search:
        if search_text_embeddings:
            vectors.append(await approach.compute_text_embedding(query_text))
        if search_image_embeddings:
            vectors.append(await approach.compute_multimodal_embedding(query_text))

    raw_results = await approach.search(
        top,
        query_text,
        search_index_filter,
        vectors,
        use_text_search,
        use_vector_search,
        use_semantic_ranker,
        use_semantic_captions,
        minimum_search_score,
        minimum_reranker_score,
        False,
        None,
        semantic_query_text=original_user_query,
    )
    results = approach._merge_documents_by_query_intent(original_user_query, raw_results, limit=top)

    validator = await validate_retrieval_plan(approach, question, overrides, rewrite_args, query_text, results)
    repaired = False
    repaired_query = str(validator.get("repaired_search_query", "") or "").strip()
    recommended_category = str(validator.get("recommended_category", "") or "").strip()
    primary_reference = str(validator.get("primary_authority_reference", "") or "").strip()

    if validator.get("evidence_sufficient") is False and repaired_query:
        repair_filter = search_index_filter
        if recommended_category:
            category_filter = f"category eq '{escape_odata(recommended_category)}'"
            repair_filter = category_filter if not repair_filter else f"({repair_filter}) and ({category_filter})"

        repaired_results: list[Any] = []
        for repair_query_candidate in build_repair_query_candidates(
            str(validator.get("primary_authority_reference", "") or ""),
            repaired_query,
        ):
            exact_results = await approach.search(
                max(top * 2, 10),
                repair_query_candidate,
                repair_filter,
                [],
                True,
                False,
                False,
                False,
                minimum_search_score,
                minimum_reranker_score,
                False,
                None,
                semantic_query_text=repair_query_candidate,
            )
            repaired_results.extend(exact_results)

            if use_vector_search:
                repair_vectors = []
                if search_text_embeddings:
                    repair_vectors.append(await approach.compute_text_embedding(repair_query_candidate))
                if search_image_embeddings:
                    repair_vectors.append(await approach.compute_multimodal_embedding(repair_query_candidate))

                semantic_results = await approach.search(
                    max(top * 2, 10),
                    repair_query_candidate,
                    repair_filter,
                    repair_vectors,
                    use_text_search,
                    use_vector_search,
                    use_semantic_ranker,
                    use_semantic_captions,
                    minimum_search_score,
                    minimum_reranker_score,
                    False,
                    None,
                    semantic_query_text=original_user_query,
                )
                repaired_results.extend(semantic_results)

        repaired_results = prioritize_authority_documents(repaired_results, validator)
        authority_hits = [
            doc
            for doc in repaired_results
            if is_primary_authority_hit(doc, primary_reference, recommended_category)
        ]
        if authority_hits:
            results = approach._merge_documents_by_query_intent(
                original_user_query,
                results + authority_hits,
                limit=max(top * 2, 10),
            )
        repaired = True

    results = prioritize_authority_documents(results, validator)
    missing_primary_authority = bool(primary_reference) and validator.get("evidence_sufficient") is False and not any(
        is_primary_authority_hit(doc, primary_reference, recommended_category) for doc in results
    )
    if missing_primary_authority:
        results = [
            doc for doc in results if score_authority_match(doc, primary_reference, recommended_category) > 0
        ]

    answer, data_points = await answer_with_documents(
        approach,
        question,
        overrides,
        results,
        validator,
        missing_primary_authority,
    )
    return {
        "answer": answer,
        "planner_query": query_text,
        "legal_concept_analysis": rewrite_args.get("legal_concept_analysis", ""),
        "validator": validator,
        "repaired": repaired,
        "missing_primary_authority": missing_primary_authority,
        "final_sources": summarize_documents(results),
        "citations": data_points.citations or [],
    }


async def main() -> int:
    results_log = []
    baseline_checks_passed = 0
    experimental_checks_passed = 0
    total_checks = 0

    async with bootstrapped_approach() as approach:
        for index, test_case in enumerate(TEST_CASES, 1):
            question = test_case["question"]
            overrides = merge_overrides(test_case.get("overrides"))
            print(f"\n{'=' * 88}")
            print(f"[{index}/{len(TEST_CASES)}] {test_case['name']}")
            print(f"Q: {question}")

            baseline = await run_baseline(approach, question, overrides)
            experimental = await run_experimental(approach, question, overrides)

            print(f"Current query:      {baseline['generated_query'][:180]}")
            print(f"Experimental query: {experimental['planner_query'][:180]}")
            if experimental["repaired"]:
                print(f"Repair query:       {experimental['validator'].get('repaired_search_query', '')[:180]}")
            print(f"Repair reason:      {experimental['validator'].get('reason', '')[:180]}")
            if experimental["missing_primary_authority"]:
                print("Primary authority:  missing after repair")

            baseline_case_results = []
            experimental_case_results = []
            for check in test_case["checks"]:
                total_checks += 1
                baseline_ok, desc = run_check(check, baseline["answer"])
                experimental_ok, _ = run_check(check, experimental["answer"])
                baseline_checks_passed += int(baseline_ok)
                experimental_checks_passed += int(experimental_ok)
                baseline_case_results.append({"description": desc, "passed": baseline_ok})
                experimental_case_results.append({"description": desc, "passed": experimental_ok})
                print(
                    f"  {desc}: current={'PASS' if baseline_ok else 'FAIL'} | experimental={'PASS' if experimental_ok else 'FAIL'}"
                )

            print(f"Current answer:      {baseline['answer'][:320].replace(chr(10), ' ')}")
            print(f"Experimental answer: {experimental['answer'][:320].replace(chr(10), ' ')}")

            results_log.append(
                {
                    "name": test_case["name"],
                    "question": question,
                    "overrides": test_case.get("overrides", {}),
                    "current": {
                        "query": baseline["generated_query"],
                        "answer": baseline["answer"],
                        "checks": baseline_case_results,
                    },
                    "experimental": {
                        "planner_query": experimental["planner_query"],
                        "legal_concept_analysis": experimental["legal_concept_analysis"],
                        "validator": experimental["validator"],
                        "repaired": experimental["repaired"],
                        "missing_primary_authority": experimental["missing_primary_authority"],
                        "answer": experimental["answer"],
                        "final_sources": experimental["final_sources"],
                        "checks": experimental_case_results,
                    },
                }
            )

    summary = {
        "current": {
            "passed_checks": baseline_checks_passed,
            "total_checks": total_checks,
            "pass_rate": round(baseline_checks_passed / total_checks, 3) if total_checks else 0,
        },
        "experimental": {
            "passed_checks": experimental_checks_passed,
            "total_checks": total_checks,
            "pass_rate": round(experimental_checks_passed / total_checks, 3) if total_checks else 0,
        },
        "results": results_log,
    }

    output_path = ROOT / "scripts" / "planner_ab_results.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 88}")
    print("SUMMARY")
    print(f"Current:      {baseline_checks_passed}/{total_checks} ({summary['current']['pass_rate']:.1%})")
    print(f"Experimental: {experimental_checks_passed}/{total_checks} ({summary['experimental']['pass_rate']:.1%})")
    print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))