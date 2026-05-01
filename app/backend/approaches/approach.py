from __future__ import annotations

import base64
import json
import re
from abc import ABC
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, TypedDict, cast

from azure.search.documents.aio import SearchClient
from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRemoteSharePointActivityRecord,
    KnowledgeBaseRemoteSharePointReference,
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetrievalResponse,
    KnowledgeBaseSearchIndexActivityRecord,
    KnowledgeBaseSearchIndexReference,
    KnowledgeBaseWebActivityRecord,
    KnowledgeBaseWebReference,
    KnowledgeRetrievalLowReasoningEffort,
    KnowledgeRetrievalMediumReasoningEffort,
    KnowledgeRetrievalMinimalReasoningEffort,
    KnowledgeRetrievalSemanticIntent,
    KnowledgeSourceParams,
    RemoteSharePointKnowledgeSourceParams,
    SearchIndexKnowledgeSourceParams,
    WebKnowledgeSourceParams,
)
from azure.search.documents.models import (
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
from openai import AsyncOpenAI, AsyncStream
from openai.types import CompletionUsage
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseStreamEvent,
    ResponseUsage,
)

from approaches.promptmanager import PromptManager
from prepdocslib.blobmanager import AdlsBlobManager, BlobManager
from prepdocslib.embeddings import ImageEmbeddings

# CUSTOM: Import legal domain customizations
from customizations.approaches import citation_builder, source_processor
from customizations import is_feature_enabled

# Reasoning effort type for models that support it
ReasoningEffort = str | None


@dataclass
class ActivityDetail:
    id: int
    number: int
    type: str
    source: str
    query: str


@dataclass
class Document:
    id: Optional[str] = None
    ref_id: Optional[str] = None  # Reference id from agentic retrieval (if applicable)
    content: Optional[str] = None
    category: Optional[str] = None
    sourcepage: Optional[str] = None
    sourcefile: Optional[str] = None
    oids: Optional[list[str]] = None
    groups: Optional[list[str]] = None
    captions: Optional[list[QueryCaptionResult]] = None
    score: Optional[float] = None
    reranker_score: Optional[float] = None
    activity: Optional[ActivityDetail] = None
    images: Optional[list[dict[str, Any]]] = None
    # CUSTOM: Legal search index fields for SupportingContent display
    storage_url: Optional[str] = None
    updated: Optional[str] = None
    subsection_id: Optional[str] = None

    def serialize_for_results(self) -> dict[str, Any]:
        result_dict = {
            "type": "searchIndex",
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "sourcepage": self.sourcepage,
            "sourcefile": self.sourcefile,
            "oids": self.oids,
            "groups": self.groups,
            "captions": (
                [
                    {
                        "additional_properties": caption.additional_properties,
                        "text": caption.text,
                        "highlights": caption.highlights,
                    }
                    for caption in self.captions
                ]
                if self.captions
                else []
            ),
            "score": self.score,
            "reranker_score": self.reranker_score,
            "activity": asdict(self.activity) if self.activity else None,
            "images": self.images,
        }
        return result_dict


@dataclass
class WebResult:
    id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    activity: Optional[ActivityDetail] = None

    def serialize_for_results(self) -> dict[str, Any]:
        return {
            "type": "web",
            "id": self.id,
            "ref_id": str(self.id),
            "title": self.title,
            "url": self.url,
            "activity": asdict(self.activity) if self.activity else None,
        }


@dataclass
class SharePointResult:
    id: Optional[str] = None
    web_url: Optional[str] = None
    content: Optional[str] = None
    title: Optional[str] = None
    reranker_score: Optional[float] = None
    activity: Optional[ActivityDetail] = None

    def serialize_for_results(self) -> dict[str, Any]:
        return {
            "type": "remoteSharePoint",
            "id": self.id,
            "ref_id": str(self.id),
            "web_url": self.web_url,
            "content": self.content,
            "title": self.title,
            "reranker_score": self.reranker_score,
            "activity": asdict(self.activity) if self.activity else None,
        }


# CUSTOM: subsection_hint and related_aspects are legal-domain extensions
@dataclass
class RewriteQueryResult:
    query: str
    messages: list[EasyInputMessageParam]
    completion: Response
    reasoning_effort: ReasoningEffort
    subsection_hint: Optional[str] = None
    related_aspects: Optional[list[str]] = None


@dataclass
class ThoughtStep:
    title: str
    description: Optional[Any]
    props: Optional[dict[str, Any]] = None

    def update_token_usage(self, usage: CompletionUsage | ResponseUsage) -> None:
        if self.props:
            self.props["token_usage"] = TokenUsageProps.from_usage(usage)


@dataclass
class AgenticRetrievalResults:
    """Results from agentic retrieval including activities, documents, web results, SharePoint results, and optional answer."""

    response: KnowledgeBaseRetrievalResponse
    documents: list[Document]
    web_results: list[WebResult]
    sharepoint_results: list[SharePointResult] = field(default_factory=list)
    answer: Optional[str] = None  # Synthesized answer when web knowledge source is used
    rewrite_result: Optional[RewriteQueryResult] = None
    query_hint: Optional[str] = None
    activity_details_by_id: Optional[dict[int, ActivityDetail]] = None
    thoughts: list[ThoughtStep] = field(default_factory=list)


@dataclass
class DataPoints:
    text: Optional[list[str | "TextSourceItem"]] = None
    images: Optional[list] = None
    citations: Optional[list[str]] = None
    external_results_metadata: Optional[list[dict[str, Any]]] = None
    citation_activity_details: Optional[dict[str, dict[str, Any]]] = None


class TextSourceItem(TypedDict, total=False):
    id: Optional[str]
    citation: str
    content: str
    full_content: str
    sourcepage: str
    sourcefile: str
    category: str
    storageurl: str
    updated: str
    subsection_id: str


@dataclass
class ExtraInfo:
    data_points: DataPoints
    thoughts: list[ThoughtStep] = field(default_factory=list)
    followup_questions: Optional[list[Any]] = None
    answer: Optional[str] = None  # Only when web knowledge source is used
    enhanced_citations: Optional[list[str]] = None
    citation_map: Optional[dict[str, str]] = None


@dataclass
class TokenUsageProps:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: Optional[int]
    total_tokens: int

    @classmethod
    def from_usage(cls, usage: CompletionUsage | ResponseUsage) -> "TokenUsageProps":
        if isinstance(usage, ResponseUsage):
            return cls(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                reasoning_tokens=(
                    usage.output_tokens_details.reasoning_tokens if usage.output_tokens_details else None
                ),
                total_tokens=usage.total_tokens,
            )
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=(
                usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else None
            ),
            total_tokens=usage.total_tokens,
        )


# CUSTOM: GPTReasoningModelSupport dataclass kept from fork — used by
# get_lowest_reasoning_effort and streaming-support checks for legal-domain models.
# Do not remove; gpt-5.4-mini is the active search-agent model.
@dataclass
class GPTReasoningModelSupport:
    streaming: bool
    lowest_effort: Optional[str]  # lowest reasoning_effort value, e.g. "minimal", "none", or None for "low"


class Approach(ABC):
    # CUSTOM: GPT_REASONING_MODELS kept from fork — upstream now uses
    # get_reasoning_effort_options() instead, but GPT_REASONING_MODELS
    # is still used by chatreadretrieveread.py streaming-support checks.
    GPT_REASONING_MODELS = {
        "o1": GPTReasoningModelSupport(streaming=False, lowest_effort=None),
        "o3": GPTReasoningModelSupport(streaming=True, lowest_effort=None),
        "o3-mini": GPTReasoningModelSupport(streaming=True, lowest_effort=None),
        "o4-mini": GPTReasoningModelSupport(streaming=True, lowest_effort=None),
        "gpt-5": GPTReasoningModelSupport(streaming=True, lowest_effort="minimal"),
        "gpt-5-nano": GPTReasoningModelSupport(streaming=True, lowest_effort="minimal"),
        "gpt-5-mini": GPTReasoningModelSupport(streaming=True, lowest_effort="minimal"),
        "gpt-5.4": GPTReasoningModelSupport(streaming=True, lowest_effort="none"),
        "gpt-5.4-pro": GPTReasoningModelSupport(streaming=True, lowest_effort="none"),
        "gpt-5.4-mini": GPTReasoningModelSupport(streaming=True, lowest_effort="none"),
        "gpt-5.4-nano": GPTReasoningModelSupport(streaming=True, lowest_effort="none"),
    }
    # Set a higher token limit for GPT reasoning models
    RESPONSE_DEFAULT_TOKEN_LIMIT = 1024
    RESPONSE_REASONING_DEFAULT_TOKEN_LIMIT = 8192
    QUERY_REWRITE_NO_RESPONSE = "0"

    def __init__(
        self,
        search_client: SearchClient,
        openai_client: AsyncOpenAI,
        knowledgebase_model: Optional[str],
        knowledgebase_deployment: Optional[str],
        query_language: Optional[str],
        query_speller: Optional[str],
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        embedding_field: str,
        openai_host: str,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        prompt_manager: PromptManager,
        reasoning_effort: Optional[str] = None,
        multimodal_enabled: bool = False,
        image_embeddings_client: Optional[ImageEmbeddings] = None,
        global_blob_manager: Optional[BlobManager] = None,
        user_blob_manager: Optional[AdlsBlobManager] = None,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.query_language = query_language
        self.query_speller = query_speller
        self.knowledgebase_model = knowledgebase_model
        self.knowledgebase_deployment = knowledgebase_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_field = embedding_field
        self.openai_host = openai_host
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.prompt_manager = prompt_manager
        self.query_rewrite_tools = self.prompt_manager.load_tools("chat_query_rewrite_tools.json")
        self.reasoning_effort = reasoning_effort
        self.include_token_usage = True
        self.multimodal_enabled = multimodal_enabled
        self.image_embeddings_client = image_embeddings_client
        self.global_blob_manager = global_blob_manager
        self.user_blob_manager = user_blob_manager

    # CUSTOM: Fuzzy search operator for typo tolerance
    def add_fuzzy_operators(self, query_text: str, edit_distance: int = 1) -> str:
        """Add fuzzy operators (~1 or ~2) to search terms for typo tolerance."""
        words = re.findall(r"\b\w+\b|AND|OR|NOT", query_text)
        fuzzy_words = []
        for word in words:
            if word in ("AND", "OR", "NOT") or len(word) <= 2:
                fuzzy_words.append(word)
            else:
                fuzzy_words.append(f"{word}~{edit_distance}")
        return " ".join(fuzzy_words)

    # CUSTOM: Enhanced build_filter with multi-category support
    def build_filter(self, overrides: dict[str, Any]) -> Optional[str]:
        include_category = overrides.get("include_category")
        exclude_category = overrides.get("exclude_category")
        filters = []
        if include_category and include_category not in ("All", ""):
            if "," in include_category:
                cat_filters = [
                    "category eq '{}'".format(p.strip().replace("'", "''"))
                    for p in include_category.split(",")
                    if p.strip()
                ]
                if cat_filters:
                    filters.append(f"({' or '.join(cat_filters)})")
            else:
                filters.append("category eq '{}'".format(include_category.replace("'", "''")))
        if exclude_category:
            filters.append("category ne '{}'".format(exclude_category.replace("'", "''")))
        return None if not filters else " and ".join(filters)

    # CUSTOM: Subsection delegation methods for citation_builder
    def _get_subsection_sort_key(self, subsection_id: str) -> tuple:
        """Generate sort key for subsection ordering - delegates to customizations module"""
        return citation_builder.get_subsection_sort_key(subsection_id)

    def _extract_subsection_from_document(self, doc: Document) -> str:
        """Extract subsection from document - delegates to customizations module"""
        return citation_builder.extract_subsection(doc)

    def _extract_multiple_subsections_from_document(self, doc: Document) -> list[dict[str, str]]:
        """Extract multiple subsections from document - delegates to customizations module"""
        return citation_builder.extract_multiple_subsections(doc)

    def _normalize_intent_text(self, value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (value or "").lower())).strip()

    def _extract_query_reference_terms(self, query: str) -> list[str]:
        patterns = [
            r"\bpractice\s+direction\s+\d+[a-z]*\b",
            r"\bpart\s+\d+[a-z]?\b",
            r"\bpd\s*\d+[a-z]*\b",
            r"\brule\s+\d+(?:\.\d+)?\b",
            r"\bparagraph\s+\d+(?:\.\d+)?\b",
            r"\bpara\.?\s+\d+(?:\.\d+)?\b",
            r"\b[a-z]\d+(?:\.\d+)+\b",
            r"\b[a-z]\d+\b",
            r"\b\d+(?:\.\d+)+\b",
        ]
        references: list[str] = []
        lowered_query = query.lower()
        for pattern in patterns:
            for match in re.finditer(pattern, lowered_query, re.IGNORECASE):
                references.append(self._normalize_intent_text(match.group(0)))
        return list(dict.fromkeys(reference for reference in references if reference))

    def _extract_explicit_legal_references(self, query: str) -> list[str]:
        patterns = [
            r"\bPractice\s+Direction\s+\d+[A-Za-z]*\b",
            r"\bPD\s*\d+[A-Za-z]*\b",
            r"\bPart\s+\d+[A-Za-z]?\b",
            r"\bRule\s+\d+(?:\.\d+)?\b",
            r"\bParagraph\s+\d+(?:\.\d+)?\b",
            r"\bPara\.?\s+\d+(?:\.\d+)?\b",
        ]
        references: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                reference = re.sub(r"\s+", " ", match.group(0).strip())
                normalized = self._normalize_intent_text(reference)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    references.append(reference)
        return references

    def _expand_explicit_legal_reference_queries(self, reference: str) -> list[str]:
        cleaned_reference = re.sub(r"\s+", " ", reference.strip())
        if not cleaned_reference:
            return []

        expanded_queries: list[str] = [cleaned_reference]
        normalized_reference = self._normalize_intent_text(cleaned_reference)

        practice_direction_match = re.fullmatch(r"practice direction\s+(\d+[a-z]*)", normalized_reference)
        if practice_direction_match:
            expanded_queries.append(f"PD{practice_direction_match.group(1).upper()}")

        pd_match = re.fullmatch(r"pd\s*(\d+[a-z]*)", normalized_reference)
        if pd_match:
            expanded_queries.append(f"Practice Direction {pd_match.group(1).upper()}")

        seen: set[str] = set()
        deduped_queries: list[str] = []
        for query in expanded_queries:
            normalized_query = self._normalize_intent_text(query)
            if normalized_query and normalized_query not in seen:
                seen.add(normalized_query)
                deduped_queries.append(query)

        return deduped_queries

    def _merge_rewritten_query_with_explicit_references(
        self,
        rewritten_query: Optional[str],
        original_query: str,
        subsection_hint: Optional[str] = None,
    ) -> str:
        merged_query = (rewritten_query or "").strip() or original_query.strip()
        normalized_query = self._normalize_intent_text(merged_query)
        missing_references: list[str] = []

        for reference in self._extract_explicit_legal_references(original_query):
            if self._normalize_intent_text(reference) not in normalized_query:
                missing_references.append(reference)

        cleaned_subsection_hint = (subsection_hint or "").strip()
        if cleaned_subsection_hint and self._normalize_intent_text(cleaned_subsection_hint) not in normalized_query:
            missing_references.append(cleaned_subsection_hint)

        if missing_references:
            merged_query = f"{merged_query} {' '.join(missing_references)}".strip()

        return merged_query

    def _covered_query_reference_terms(self, documents: list[Document], query: str) -> set[str]:
        reference_terms = self._extract_query_reference_terms(query)
        if not reference_terms:
            return set()

        covered_terms: set[str] = set()
        for doc in documents[:5]:
            haystack = self._normalize_intent_text(
                " ".join(
                    part
                    for part in [
                        doc.subsection_id,
                        doc.sourcepage,
                        doc.sourcefile,
                        doc.category,
                        (doc.content or "")[:1000],
                    ]
                    if part
                )
            )
            if not haystack:
                continue

            for reference_term in reference_terms:
                if reference_term in haystack:
                    covered_terms.add(reference_term)

        return covered_terms

    def _covered_query_intent_terms(self, documents: list[Document], query: str) -> set[str]:
        intent_terms = self._extract_query_intent_terms(query)
        if not intent_terms:
            return set()

        covered_terms: set[str] = set()
        for doc in documents[:5]:
            source_haystack = self._normalize_intent_text(
                " ".join(part for part in [doc.subsection_id, doc.sourcepage, doc.sourcefile, doc.category] if part)
            )
            content_haystack = self._normalize_intent_text((doc.content or "")[:1000])
            haystack = " ".join(part for part in [source_haystack, content_haystack] if part)
            if not haystack:
                continue

            for intent_term in intent_terms:
                if self._term_matches_haystack(haystack, intent_term):
                    covered_terms.add(intent_term)

        return covered_terms

    def _results_include_reference_in_source(self, documents: list[Document], reference: str) -> bool:
        normalized_reference = self._normalize_intent_text(reference)
        if not normalized_reference:
            return False

        for doc in documents[:5]:
            source_haystack = self._normalize_intent_text(
                " ".join(part for part in [doc.subsection_id, doc.sourcepage, doc.sourcefile, doc.category] if part)
            )
            if normalized_reference in source_haystack:
                return True

        return False

    def _extract_cpr_part_references_from_rewrite(self, rewritten_query: str) -> list[tuple[str, str]]:
        """Extract CPR Part references the LLM placed in its rewritten query.

        Returns (part_reference, rewritten_query) tuples where part_reference
        is e.g. "Part 31" derived from "CPR 31.6" or "CPR Part 52".
        The rewritten_query is passed through so it can be used as the
        supplemental search text.
        """
        part_numbers: set[str] = set()
        lowered = rewritten_query.lower()

        # Match "CPR X.Y" or "CPR X" → derive Part X
        for m in re.finditer(r"\bcpr\s+(\d+)(?:\.\d+)?\b", lowered):
            part_numbers.add(m.group(1))

        # Match "Part X" explicitly
        for m in re.finditer(r"\bpart\s+(\d+)\b", lowered):
            part_numbers.add(m.group(1))

        return [(f"Part {num}", rewritten_query) for num in sorted(part_numbers)]

    def _extract_canonical_legal_concept_queries(self, query: str) -> list[tuple[str, str, Optional[str]]]:
        """Return (targeted_query, required_source_reference, category_filter) tuples.

        The category_filter narrows the supplemental search to the correct
        source family so that Court Guide chunks do not drown out the
        authoritative CPR Part in All Sources mode.
        """
        normalized_query = self._normalize_intent_text(query)
        concept_mappings: list[tuple[str, str, str, Optional[str]]] = [
            (
                "summary judgment",
                "24.3 summary judgment no real prospect of succeeding no other compelling reason",
                "Part 24",
                "category eq 'Civil Procedure Rules and Practice Directions'",
            ),
            (
                "pre action disclosure",
                "31.16 disclosure before proceedings have started application for disclosure pre-action",
                "Part 31",
                "category eq 'Civil Procedure Rules and Practice Directions'",
            ),
        ]
        # Short acronyms need word-boundary matching to avoid false positives
        acronym_mappings: list[tuple[str, str, str, Optional[str]]] = [
            (
                r"\bpad\b",
                "31.16 disclosure before proceedings have started application for disclosure pre-action",
                "Part 31",
                "category eq 'Civil Procedure Rules and Practice Directions'",
            ),
        ]

        results: list[tuple[str, str, Optional[str]]] = [
            (targeted_query, required_source_reference, category_filter)
            for concept_phrase, targeted_query, required_source_reference, category_filter in concept_mappings
            if concept_phrase in normalized_query
        ]
        for pattern, targeted_query, required_source_reference, category_filter in acronym_mappings:
            if re.search(pattern, normalized_query):
                results.append((targeted_query, required_source_reference, category_filter))

        return results

    def _extract_query_intent_terms(self, query: str) -> list[str]:
        stopwords = {
            "a",
            "about",
            "all",
            "an",
            "and",
            "are",
            "be",
            "does",
            "for",
            "from",
            "give",
            "guide",
            "how",
            "in",
            "is",
            "of",
            "on",
            "say",
            "section",
            "sections",
            "tell",
            "that",
            "the",
            "their",
            "there",
            "these",
            "this",
            "under",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
        generic_legal_terms = {
            "application",
            "applications",
            "claim",
            "claims",
            "court",
            "courts",
            "division",
            "filing",
            "guide",
            "judge",
            "judges",
            "legal",
            "part",
            "practice",
            "procedure",
            "proceedings",
            "rule",
            "rules",
        }
        tokens = [token for token in self._normalize_intent_text(query).split() if len(token) >= 4]
        filtered = [token for token in tokens if token not in stopwords and token not in generic_legal_terms]
        return list(dict.fromkeys(filtered))

    def _term_matches_haystack(self, haystack: str, term: str) -> bool:
        normalized_term = self._normalize_intent_text(term)
        if not haystack or not normalized_term:
            return False

        candidates = [normalized_term]
        for suffix in ("ments", "ment", "ations", "ation", "ings", "ing", "ied", "ies", "ed", "es", "s"):
            if normalized_term.endswith(suffix) and len(normalized_term) - len(suffix) >= 4:
                candidates.append(normalized_term[: -len(suffix)])

        return any(candidate and candidate in haystack for candidate in dict.fromkeys(candidates))

    def _extract_query_focus_terms(self, query: str) -> list[str]:
        normalized_query = self._normalize_intent_text(query)
        focus_match = re.search(r"\b(?:about|regarding|concerning|on)\s+(.+)$", normalized_query)
        if not focus_match:
            return []

        focus_terms = self._extract_query_intent_terms(focus_match.group(1))
        return focus_terms if len(focus_terms) >= 2 else []

    def _is_tangential_to_query_focus(self, doc: Document, query: str) -> bool:
        focus_terms = self._extract_query_focus_terms(query)
        if not focus_terms:
            return False

        normalized_query = self._normalize_intent_text(query)
        source_haystack = self._normalize_intent_text(
            " ".join(part for part in [doc.subsection_id, doc.sourcepage, doc.sourcefile, doc.category] if part)
        )
        content_haystack = self._normalize_intent_text((doc.content or "")[:1000])

        focus_source_hits = sum(1 for term in focus_terms if self._term_matches_haystack(source_haystack, term))
        focus_content_hits = sum(1 for term in focus_terms if self._term_matches_haystack(content_haystack, term))

        if "annex" in source_haystack and "annex" not in normalized_query:
            return True

        return focus_source_hits >= max(2, len(focus_terms) - 1) and focus_content_hits < 2

    def _score_document_query_intent(self, doc: Document, query: str) -> int:
        source_haystack = self._normalize_intent_text(
            " ".join(part for part in [doc.subsection_id, doc.sourcepage, doc.sourcefile, doc.category] if part)
        )
        content_haystack = self._normalize_intent_text((doc.content or "")[:1000])
        haystack = " ".join(part for part in [source_haystack, content_haystack] if part)
        if not haystack:
            return 0

        score = 0
        normalized_query = self._normalize_intent_text(query)
        reference_terms = self._extract_query_reference_terms(query)
        for reference_term in reference_terms:
            if self._term_matches_haystack(source_haystack, reference_term):
                score += 6
            elif self._term_matches_haystack(content_haystack, reference_term):
                score += 4

        intent_terms = self._extract_query_intent_terms(query)
        for intent_term in intent_terms:
            if self._term_matches_haystack(content_haystack, intent_term):
                score += 2
            elif self._term_matches_haystack(source_haystack, intent_term):
                score += 1

        for first, second in zip(intent_terms, intent_terms[1:]):
            phrase = f"{first} {second}"
            if phrase in content_haystack:
                score += 3
            elif phrase in source_haystack:
                score += 1

        focus_terms = self._extract_query_focus_terms(query)
        if focus_terms:
            focus_source_hits = sum(1 for term in focus_terms if self._term_matches_haystack(source_haystack, term))
            focus_content_hits = sum(1 for term in focus_terms if self._term_matches_haystack(content_haystack, term))

            if focus_content_hits >= 2:
                score += focus_content_hits * 3
            elif focus_source_hits >= 2:
                score += focus_source_hits

            if focus_source_hits >= max(2, len(focus_terms) - 1) and focus_content_hits < 2:
                score -= 6

        if self._is_tangential_to_query_focus(doc, query):
            score -= 4

        return score

    def _should_retry_for_query_intent(self, documents: list[Document], query: Optional[str]) -> bool:
        if not query or not documents:
            return False

        reference_terms = self._extract_query_reference_terms(query)
        intent_terms = self._extract_query_intent_terms(query)
        if not reference_terms and len(intent_terms) < 2:
            return False

        if len(reference_terms) > 1:
            covered_reference_terms = self._covered_query_reference_terms(documents, query)
            if len(covered_reference_terms) < len(reference_terms):
                return True

        if len(intent_terms) >= 4:
            covered_intent_terms = self._covered_query_intent_terms(documents, query)
            if len(covered_intent_terms) < 3:
                return True

        scores = [self._score_document_query_intent(doc, query) for doc in documents[:3]]
        if not scores:
            return False

        strong_match_threshold = 5 if reference_terms else 3
        return max(scores) < strong_match_threshold

    def _merge_documents_by_query_intent(self, query: str, documents: list[Document], limit: int) -> list[Document]:
        best_by_id: dict[str, tuple[tuple[float, float, float], Document]] = {}

        for index, doc in enumerate(documents):
            rank = (
                float(self._score_document_query_intent(doc, query)),
                float(doc.reranker_score or -1),
                float(doc.score or -1),
            )
            doc_id = doc.id or f"__index_{index}"
            existing = best_by_id.get(doc_id)
            if existing is None or rank > existing[0]:
                best_by_id[doc_id] = (rank, doc)

        ordered_documents = [entry[1] for entry in best_by_id.values()]
        ordered_documents.sort(
            key=lambda doc: (
                self._score_document_query_intent(doc, query),
                doc.reranker_score or -1,
                doc.score or -1,
            ),
            reverse=True,
        )

        ranked_documents = [(doc, self._score_document_query_intent(doc, query)) for doc in ordered_documents]
        if ranked_documents:
            best_score = ranked_documents[0][1]
            if best_score >= 6:
                # CUSTOM: Use a lower threshold for broad queries (no specific legal
                # references) to avoid dropping related-but-secondary topic chunks.
                # Focused queries with reference terms keep the stricter 50% cutoff.
                reference_terms = self._extract_query_reference_terms(query)
                if reference_terms:
                    minimum_score = max(2, int(best_score * 0.5))
                else:
                    minimum_score = max(1, int(best_score * 0.35))
                focused_documents = [
                    doc
                    for doc, score in ranked_documents
                    if score >= minimum_score and not self._is_tangential_to_query_focus(doc, query)
                ]
                if not focused_documents:
                    focused_documents = [doc for doc, score in ranked_documents if score >= minimum_score]
                if focused_documents:
                    ordered_documents = focused_documents

        return ordered_documents[:limit]

    async def search(
        self,
        top: int,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: list[VectorQuery],
        use_text_search: bool,
        use_vector_search: bool,
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
        minimum_search_score: Optional[float] = None,
        minimum_reranker_score: Optional[float] = None,
        use_query_rewriting: Optional[bool] = None,
        access_token: Optional[str] = None,
        semantic_query_text: Optional[str] = None,
    ) -> list[Document]:
        search_text = query_text if use_text_search else ""
        search_vectors = vectors if use_vector_search else []
        semantic_query = semantic_query_text or query_text
        if use_semantic_ranker:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                query_rewrites="generative" if use_query_rewriting else None,
                vector_queries=search_vectors,
                query_type=QueryType.SEMANTIC,
                query_language=self.query_language,
                query_speller=self.query_speller,
                semantic_configuration_name="default",
                semantic_query=semantic_query,
                x_ms_query_source_authorization=access_token,
            )
        else:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                vector_queries=search_vectors,
                x_ms_query_source_authorization=access_token,
            )

        documents: list[Document] = []
        async for page in results.by_page():
            async for document in page:
                documents.append(
                    Document(
                        id=document.get("id"),
                        content=document.get("content"),
                        category=document.get("category"),
                        sourcepage=document.get("sourcepage"),
                        sourcefile=document.get("sourcefile"),
                        oids=document.get("oids"),
                        groups=document.get("groups"),
                        captions=cast(list[QueryCaptionResult], document.get("@search.captions")),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                        images=document.get("images"),
                        # CUSTOM: Legal search index fields
                        storage_url=document.get("storageUrl"),
                        updated=document.get("updated"),
                        subsection_id=document.get("subsection_id"),
                    )
                )

            qualified_documents = [
                doc
                for doc in documents
                if (
                    (doc.score or 0) >= (minimum_search_score or 0)
                    and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
                )
            ]

        return qualified_documents

    def extract_rewritten_query(
        self,
        response: Response,
        user_query: str,
        no_response_token: Optional[str] = None,
    ) -> str:
        # Check output items for function calls
        for item in response.output:
            if isinstance(item, ResponseFunctionToolCall):
                try:
                    parsed_arguments = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    continue
                search_query = parsed_arguments.get("search_query")
                if search_query and (no_response_token is None or search_query != no_response_token):
                    return search_query

        # Fall back to text content
        text = response.output_text
        if text:
            candidate = text.strip()
            if candidate and (no_response_token is None or candidate != no_response_token):
                return candidate

        return user_query

    # CUSTOM: Responses API version of argument extraction. Also pulls
    # subsection_hint and related_aspects — the legal-domain extra tool params
    # added by this fork and preserved from chat_query_rewrite_tools.json.
    def extract_rewrite_function_arguments(self, response: Response) -> dict[str, Any]:
        for item in response.output:
            if isinstance(item, ResponseFunctionToolCall):
                try:
                    parsed_arguments = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed_arguments, dict):
                    return parsed_arguments
        return {}

    async def rewrite_query(
        self,
        *,
        prompt_template: str,
        prompt_variables: dict[str, Any],
        overrides: dict[str, Any],
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],
        user_query: str,
        response_token_limit: int,
        tools: Optional[list[FunctionToolParam]] = None,
        temperature: float = 0.0,
        no_response_token: Optional[str] = None,
    ) -> RewriteQueryResult:
        query_messages = self.prompt_manager.build_conversation(
            system_template_path=prompt_template,
            system_template_variables=prompt_variables,
            user_template_path="query_rewrite.user.jinja2",
            user_template_variables={"user_query": user_query},
            past_messages=prompt_variables.get("past_messages"),
        )
        rewrite_reasoning_effort = self.get_lowest_reasoning_effort(self.chatgpt_model)

        response = cast(
            Response,
            await self.create_response(
                chatgpt_deployment,
                chatgpt_model,
                input=query_messages,
                overrides=overrides,
                response_token_limit=response_token_limit,
                temperature=temperature,
                tools=tools,
                reasoning_effort=rewrite_reasoning_effort,
            ),
        )

        # CUSTOM: extract_rewrite_function_arguments now uses Responses API
        rewrite_arguments = self.extract_rewrite_function_arguments(response)
        rewritten_query = self.extract_rewritten_query(
            response,
            user_query,
            no_response_token=no_response_token,
        )
        subsection_hint = rewrite_arguments.get("subsection_hint")
        if not isinstance(subsection_hint, str) or not subsection_hint.strip():
            subsection_hint = None

        # CUSTOM: Extract related_aspects for broad topic supplemental searches
        related_aspects_raw = rewrite_arguments.get("related_aspects")
        related_aspects: Optional[list[str]] = None
        if isinstance(related_aspects_raw, str) and related_aspects_raw.strip():
            related_aspects = [a.strip() for a in related_aspects_raw.split("|") if a.strip()]
            if not related_aspects:
                related_aspects = None

        return RewriteQueryResult(
            query=rewritten_query,
            messages=query_messages,
            completion=response,
            reasoning_effort=rewrite_reasoning_effort,
            subsection_hint=subsection_hint,
            related_aspects=related_aspects,
        )

    async def run_agentic_retrieval(
        self,
        messages: list[EasyInputMessageParam],
        knowledgebase_client: KnowledgeBaseRetrievalClient,
        search_index_name: str,
        filter_add_on: Optional[str] = None,
        minimum_reranker_score: Optional[float] = None,
        access_token: Optional[str] = None,
        use_web_source: bool = False,
        use_sharepoint_source: bool = False,
        retrieval_reasoning_effort: Optional[str] = None,
        should_rewrite_query: bool = True,
    ) -> AgenticRetrievalResults:
        # STEP 1: Invoke agentic retrieval
        thoughts = []

        knowledge_source_params = [
            SearchIndexKnowledgeSourceParams(
                knowledge_source_name=search_index_name,
                filter_add_on=filter_add_on,
                include_references=True,
                include_reference_source_data=True,
                always_query_source=False,
                reranker_threshold=minimum_reranker_score,
            )
        ]
        # Build list as KnowledgeSourceParams for type variance
        knowledge_source_params_list: list[KnowledgeSourceParams] = cast(
            list[KnowledgeSourceParams], knowledge_source_params
        )

        if use_web_source:
            knowledge_source_params_list.append(
                WebKnowledgeSourceParams(
                    knowledge_source_name="web",
                    include_references=True,
                    include_reference_source_data=True,
                    always_query_source=False,
                )
            )

        if use_sharepoint_source:
            knowledge_source_params_list.append(
                RemoteSharePointKnowledgeSourceParams(
                    knowledge_source_name="sharepoint",
                    include_references=True,
                    include_reference_source_data=True,
                    always_query_source=False,
                )
            )

        agentic_retrieval_input: dict[str, Any] = {}
        rewrite_result = None
        latest_message_content = messages[-1]["content"] if messages else ""
        if should_rewrite_query and isinstance(latest_message_content, str) and latest_message_content:
            rewrite_result = await self.rewrite_query(
                prompt_template="query_rewrite.system.jinja2",
                prompt_variables={
                    "user_query": latest_message_content,
                    "past_messages": messages[:-1],
                    "available_sources": self.available_sources,
                },
                overrides={},
                chatgpt_model=self.chatgpt_model,
                chatgpt_deployment=self.chatgpt_deployment,
                user_query=latest_message_content,
                response_token_limit=self.get_response_token_limit(
                    self.chatgpt_model, 300
                ),
                tools=self.query_rewrite_tools,
                temperature=0.0,
                no_response_token=self.QUERY_REWRITE_NO_RESPONSE,
            )

        if rewrite_result is not None:
            thoughts.append(
                self.format_thought_step_for_chatcompletion(
                    title="Prompt to generate search query",
                    messages=rewrite_result.messages,
                    overrides={},
                    model=self.chatgpt_model,
                    deployment=self.chatgpt_deployment,
                    usage=rewrite_result.completion.usage,
                    reasoning_effort=rewrite_result.reasoning_effort,
                )
            )

        if retrieval_reasoning_effort == "minimal" and rewrite_result is not None:
            merged_rewrite_query = self._merge_rewritten_query_with_explicit_references(
                rewrite_result.query,
                latest_message_content,
                rewrite_result.subsection_hint,
            )
            agentic_retrieval_input["intents"] = [KnowledgeRetrievalSemanticIntent(search=merged_rewrite_query)]
        elif retrieval_reasoning_effort == "minimal":
            if not isinstance(latest_message_content, str):
                raise ValueError("The most recent message content must be a string.")
            agentic_retrieval_input["intents"] = [KnowledgeRetrievalSemanticIntent(search=latest_message_content)]
        else:
            kb_messages: list[KnowledgeBaseMessage] = [
                KnowledgeBaseMessage(
                    role=str(msg["role"]), content=[KnowledgeBaseMessageTextContent(text=str(msg["content"]))]
                )
                for msg in messages
                if msg["role"] != "system"
            ]
            agentic_retrieval_input["messages"] = kb_messages
        # When we're not using a web source, set output mode to extractiveData to avoid synthesized answer
        if not use_web_source:
            agentic_retrieval_input["output_mode"] = "extractiveData"

        retrieval_effort: Optional[
            KnowledgeRetrievalMinimalReasoningEffort
            | KnowledgeRetrievalLowReasoningEffort
            | KnowledgeRetrievalMediumReasoningEffort
        ] = None
        if retrieval_reasoning_effort == "minimal":
            retrieval_effort = KnowledgeRetrievalMinimalReasoningEffort()
        elif retrieval_reasoning_effort == "low":
            retrieval_effort = KnowledgeRetrievalLowReasoningEffort()
        elif retrieval_reasoning_effort == "medium":
            retrieval_effort = KnowledgeRetrievalMediumReasoningEffort()

        request_kwargs: dict[str, Any] = {
            "knowledge_source_params": knowledge_source_params_list,
            "include_activity": True,
            "retrieval_reasoning_effort": retrieval_effort,
        }
        request_kwargs.update(agentic_retrieval_input)

        response = await knowledgebase_client.retrieve(
            retrieval_request=KnowledgeBaseRetrievalRequest(**request_kwargs),
            x_ms_query_source_authorization=access_token,
        )

        # Map activity id -> agent's internal search query and citation
        activities = response.activity or []
        activity_details_by_id: dict[int, ActivityDetail] = {}

        for index, activity in enumerate(activities):
            search_query = None
            if isinstance(activity, KnowledgeBaseSearchIndexActivityRecord):
                if activity.search_index_arguments:
                    search_query = activity.search_index_arguments.search
            elif isinstance(activity, KnowledgeBaseWebActivityRecord):
                if activity.web_arguments:
                    search_query = activity.web_arguments.search
            elif isinstance(activity, KnowledgeBaseRemoteSharePointActivityRecord):
                if activity.remote_share_point_arguments:
                    search_query = activity.remote_share_point_arguments.search

            activity_details_by_id[activity.id] = ActivityDetail(
                id=activity.id,
                number=index + 1,
                type=activity.type or "",
                source=getattr(activity, "knowledge_source_name", "")
                or "",  # Not all activity types have knowledge_source_name
                query=search_query or "",
            )

        # Extract references
        references = response.references or []

        document_refs = [
            r for r in references if isinstance(r, KnowledgeBaseSearchIndexReference) or hasattr(r, "doc_key")
        ]
        document_results: list[Document] = []
        # Create documents from reference source data
        for ref in document_refs:
            if ref.source_data and ref.doc_key:
                # Note that ref.doc_key is the same as source_data["id"]
                document_results.append(
                    Document(
                        id=cast(str, ref.doc_key),
                        ref_id=ref.id,
                        content=ref.source_data.get("content"),
                        category=ref.source_data.get("category"),
                        sourcepage=ref.source_data.get("sourcepage"),
                        sourcefile=ref.source_data.get("sourcefile"),
                        oids=ref.source_data.get("oids"),
                        groups=ref.source_data.get("groups"),
                        reranker_score=getattr(ref, "reranker_score", None),
                        images=ref.source_data.get("images"),
                        activity=activity_details_by_id[ref.activity_source],
                        # CUSTOM: Legal search index fields
                        storage_url=ref.source_data.get("storageUrl"),
                        updated=ref.source_data.get("updated"),
                        subsection_id=ref.source_data.get("subsection_id"),
                    )
                )

        # We need to handle KnowledgeBaseWebReference separately if web knowledge source is used
        web_refs = [r for r in references if isinstance(r, KnowledgeBaseWebReference)]
        web_results: list[WebResult] = []
        for ref in web_refs:
            web_result = WebResult(
                id=ref.id, title=ref.title, url=ref.url, activity=activity_details_by_id[ref.activity_source]
            )
            web_results.append(web_result)

        # Handle KnowledgeBaseRemoteSharePointReference if SharePoint knowledge source is used
        sharepoint_refs = [r for r in references if isinstance(r, KnowledgeBaseRemoteSharePointReference)]
        sharepoint_results: list[SharePointResult] = []
        for ref in sharepoint_refs:
            # Extract content from all sourceData.extracts[].text and concatenate
            content = None
            if ref.source_data and "extracts" in ref.source_data and len(ref.source_data["extracts"]) > 0:
                extracts = [extract.get("text", "") for extract in ref.source_data["extracts"]]
                content = "\n\n".join(extracts) if extracts else None

            # Extract title from sourceData.resourceMetadata.title
            title = None
            if ref.source_data and "resourceMetadata" in ref.source_data:
                title = ref.source_data["resourceMetadata"].get("title")

            sharepoint_result = SharePointResult(
                id=ref.id,
                web_url=ref.web_url,
                content=content,
                title=title,
                reranker_score=getattr(ref, "reranker_score", None),
                activity=activity_details_by_id[ref.activity_source],
            )
            sharepoint_results.append(sharepoint_result)

        # Extract answer from response if web knowledge source provided one
        answer: Optional[str] = None
        if (
            use_web_source
            and response.response
            and len(response.response) > 0
            and len(response.response[0].content) > 0
        ):
            message_content = response.response[0].content[0]
            if isinstance(message_content, KnowledgeBaseMessageTextContent):
                raw_answer: Optional[str] = message_content.text
                # Replace all ref_id tokens (web -> URL, documents -> sourcepage, SharePoint -> web_url)
                if raw_answer:
                    answer = self.replace_all_ref_ids(raw_answer, document_results, web_results, sharepoint_results)

        # CUSTOM: If no document references returned, force querying sources
        if not document_results and is_feature_enabled("agentic_force_query_on_empty"):
            import logging

            logging.info("CUSTOM: No agentic results; retrying with always_query_source=True.")
            retry_source_params: list[KnowledgeSourceParams] = [
                SearchIndexKnowledgeSourceParams(
                    knowledge_source_name=search_index_name,
                    filter_add_on=filter_add_on,
                    include_references=True,
                    include_reference_source_data=True,
                    always_query_source=True,
                    reranker_threshold=minimum_reranker_score,
                )
            ]
            retry_kwargs: dict[str, Any] = dict(request_kwargs)
            retry_kwargs["knowledge_source_params"] = retry_source_params
            retry_response = await knowledgebase_client.retrieve(
                retrieval_request=KnowledgeBaseRetrievalRequest(**retry_kwargs),
                x_ms_query_source_authorization=access_token,
            )
            retry_refs = retry_response.references or []
            for ref in retry_refs:
                if isinstance(ref, KnowledgeBaseSearchIndexReference) and ref.source_data and ref.doc_key:
                    document_results.append(
                        Document(
                            id=cast(str, ref.doc_key),
                            ref_id=ref.id,
                            content=ref.source_data.get("content"),
                            category=ref.source_data.get("category"),
                            sourcepage=ref.source_data.get("sourcepage"),
                            sourcefile=ref.source_data.get("sourcefile"),
                            oids=ref.source_data.get("oids"),
                            groups=ref.source_data.get("groups"),
                            reranker_score=getattr(ref, "reranker_score", None),
                            images=ref.source_data.get("images"),
                            # CUSTOM: Legal search index fields
                            storage_url=ref.source_data.get("storageUrl"),
                            updated=ref.source_data.get("updated"),
                            subsection_id=ref.source_data.get("subsection_id"),
                        )
                    )

        latest_user_query = messages[-1]["content"] if messages else ""
        explicit_legal_references = (
            self._extract_explicit_legal_references(latest_user_query) if isinstance(latest_user_query, str) else []
        )
        canonical_legal_concepts = (
            self._extract_canonical_legal_concept_queries(latest_user_query) if isinstance(latest_user_query, str) else []
        )
        # Also extract CPR Part references from any rewritten query for dynamic retrieval
        rewrite_cpr_refs: list[tuple[str, str]] = []
        if rewrite_result and rewrite_result.query:
            rewrite_cpr_refs = self._extract_cpr_part_references_from_rewrite(rewrite_result.query)
        fallback_result_limit = max(3, len(explicit_legal_references) + len(canonical_legal_concepts) + len(rewrite_cpr_refs) + 1)
        query_hint: Optional[str] = None
        if rewrite_result and rewrite_result.query:
            query_hint = self._merge_rewritten_query_with_explicit_references(
                rewrite_result.query,
                latest_user_query if isinstance(latest_user_query, str) else rewrite_result.query,
                rewrite_result.subsection_hint,
            )
        elif canonical_legal_concepts:
            query_hint = canonical_legal_concepts[0][0]
        elif isinstance(latest_user_query, str):
            query_hint = latest_user_query
        weak_document_matches = bool(
            isinstance(latest_user_query, str)
            and document_results
            and is_feature_enabled("agentic_retry_on_weak_matches")
            and self._should_retry_for_query_intent(document_results, latest_user_query)
        )

        merge_scoring_query = query_hint or (latest_user_query if isinstance(latest_user_query, str) else "")

        if isinstance(latest_user_query, str) and latest_user_query and document_results:
            document_results = self._merge_documents_by_query_intent(
                merge_scoring_query,
                document_results,
                limit=len(document_results),
            )

        # CUSTOM: If references are still missing, or they look off-target, fall back to direct search using agentic query plan
        if (not document_results and is_feature_enabled("agentic_fallback_search")) or weak_document_matches:
            import logging

            logging.info(
                "CUSTOM: %s; running fallback search using agentic queries.",
                "Weak agentic references" if weak_document_matches else "No agentic references",
            )
            query_candidates = [ad.query for ad in activity_details_by_id.values() if ad.query]
            if rewrite_result and rewrite_result.query:
                query_candidates.append(
                    self._merge_rewritten_query_with_explicit_references(
                        rewrite_result.query,
                        latest_user_query if isinstance(latest_user_query, str) else rewrite_result.query,
                        rewrite_result.subsection_hint,
                    )
                )
            if not query_candidates and messages:
                last_content = messages[-1]["content"] if messages else ""
                if isinstance(last_content, str) and last_content:
                    query_candidates = [last_content]
            elif isinstance(latest_user_query, str) and latest_user_query:
                query_candidates.append(latest_user_query)

            query_candidates = list(dict.fromkeys(str(q).strip() for q in query_candidates if str(q).strip()))

            seen_ids: set[str] = set()
            if weak_document_matches:
                seen_ids.update(doc.id for doc in document_results if doc.id)
            for q in query_candidates:
                docs = await self.search(
                    top=fallback_result_limit,
                    query_text=str(q),
                    filter=filter_add_on,
                    vectors=[],
                    use_text_search=True,
                    use_vector_search=False,
                    use_semantic_ranker=True,
                    use_semantic_captions=False,
                    minimum_search_score=0,
                    minimum_reranker_score=0,
                    use_query_rewriting=False,
                )
                for doc in docs:
                    if doc.id and doc.id not in seen_ids:
                        seen_ids.add(doc.id)
                        document_results.append(doc)

            if isinstance(latest_user_query, str) and latest_user_query:
                document_results = self._merge_documents_by_query_intent(
                    merge_scoring_query,
                    document_results,
                    limit=fallback_result_limit,
                )

        supplemental_reference_queries: list[str] = []
        if isinstance(latest_user_query, str) and latest_user_query and document_results:
            covered_reference_terms = self._covered_query_reference_terms(document_results, latest_user_query)
            missing_reference_queries = [
                reference
                for reference in explicit_legal_references
                if self._normalize_intent_text(reference) not in covered_reference_terms
            ]

            for reference_query in missing_reference_queries:
                for targeted_reference_query in self._expand_explicit_legal_reference_queries(reference_query):
                    docs = await self.search(
                        top=min(fallback_result_limit, 3),
                        query_text=targeted_reference_query,
                        filter=filter_add_on,
                        vectors=[],
                        use_text_search=True,
                        use_vector_search=False,
                        use_semantic_ranker=False,
                        use_semantic_captions=False,
                        minimum_search_score=0,
                        minimum_reranker_score=0,
                        use_query_rewriting=False,
                        access_token=access_token,
                        semantic_query_text=targeted_reference_query,
                    )
                    if docs:
                        supplemental_reference_queries.append(targeted_reference_query)
                        document_results = self._merge_documents_by_query_intent(
                            merge_scoring_query,
                            document_results + docs,
                            limit=fallback_result_limit,
                        )
                        break

            # Dynamic CPR Part retrieval from LLM rewrite + canonical concepts
            cpr_category_filter = "category eq 'Civil Procedure Rules and Practice Directions'"
            all_part_refs: list[tuple[str, str]] = list(rewrite_cpr_refs)
            for concept_query, required_source_reference, _ in canonical_legal_concepts:
                if not any(ref == required_source_reference for ref, _ in all_part_refs):
                    all_part_refs.append((required_source_reference, concept_query))

            for part_reference, search_text in all_part_refs:
                if self._results_include_reference_in_source(document_results, part_reference):
                    continue

                docs = await self.search(
                    top=min(fallback_result_limit, 3),
                    query_text=search_text,
                    filter=cpr_category_filter,
                    vectors=[],
                    use_text_search=True,
                    use_vector_search=False,
                    use_semantic_ranker=False,
                    use_semantic_captions=False,
                    minimum_search_score=0,
                    minimum_reranker_score=0,
                    use_query_rewriting=False,
                    access_token=access_token,
                    semantic_query_text=search_text,
                )
                if docs:
                    supplemental_reference_queries.append(f"{part_reference} (from rewrite)")
                    document_results = self._merge_documents_by_query_intent(
                        merge_scoring_query,
                        document_results + docs,
                        limit=fallback_result_limit,
                    )
                    part_doc = next(
                        (
                            candidate
                            for candidate in document_results
                            if candidate.category == "Civil Procedure Rules and Practice Directions"
                            and self._results_include_reference_in_source([candidate], part_reference)
                        ),
                        None,
                    )
                    if part_doc is None:
                        part_doc = next(
                            (candidate for candidate in docs if self._results_include_reference_in_source([candidate], part_reference)),
                            docs[0],
                        )
                        document_results.append(part_doc)

                    if part_doc is not None:
                        remaining_docs = [doc for doc in document_results if doc.id != part_doc.id]
                        document_results = [part_doc, *remaining_docs]

            for part_reference, _search_text in reversed(all_part_refs):
                matched_part_doc = next(
                    (
                        candidate
                        for candidate in document_results
                        if candidate.category == "Civil Procedure Rules and Practice Directions"
                        and self._results_include_reference_in_source([candidate], part_reference)
                    ),
                    None,
                )
                if matched_part_doc is not None:
                    remaining_docs = [doc for doc in document_results if doc.id != matched_part_doc.id]
                    document_results = [matched_part_doc, *remaining_docs]

        # CUSTOM: Agentic retrieval source_data may not include all index fields.
        # Supplement missing fields (updated, storageUrl, subsection_id) via direct lookup.
        docs_needing_supplement = [
            doc for doc in document_results if doc.id and not doc.updated and not doc.storage_url and not doc.subsection_id
        ]
        if docs_needing_supplement:
            supplemental_fields = ["updated", "storageUrl", "subsection_id"]
            for doc in docs_needing_supplement:
                try:
                    index_doc = await self.search_client.get_document(
                        key=doc.id, selected_fields=supplemental_fields
                    )
                    if index_doc:
                        if not doc.updated and index_doc.get("updated"):
                            doc.updated = index_doc["updated"]
                        if not doc.storage_url and index_doc.get("storageUrl"):
                            doc.storage_url = index_doc["storageUrl"]
                        if not doc.subsection_id and index_doc.get("subsection_id"):
                            doc.subsection_id = index_doc["subsection_id"]
                except Exception:
                    pass

        thoughts.append(
            ThoughtStep(
                "Agentic retrieval response",
                [result.serialize_for_results() for result in document_results + web_results + sharepoint_results],
                {
                    "query_plan": (
                        [activity.as_dict() for activity in response.activity] if response.activity else None
                    ),
                    "model": self.knowledgebase_model,
                    "deployment": self.knowledgebase_deployment,
                    "reranker_threshold": minimum_reranker_score,
                    "filter": filter_add_on,
                    "targeted_reference_queries": supplemental_reference_queries,
                },
            )
        )

        return AgenticRetrievalResults(
            response=response,
            documents=document_results,
            web_results=web_results,
            sharepoint_results=sharepoint_results,
            answer=answer,
            rewrite_result=rewrite_result,
            query_hint=query_hint,
            activity_details_by_id=activity_details_by_id,
            thoughts=thoughts,
        )

    def replace_all_ref_ids(
        self,
        answer: str,
        documents: list[Document],
        web_results: list[WebResult],
        sharepoint_results: Optional[list[SharePointResult]] = None,
    ) -> str:
        """Replace [ref_id:<id>] tokens with document sourcepage, web URL, or SharePoint web_url.

        Priority: web result -> SharePoint result -> document.
        Unknown ids left untouched.
        """
        doc_map = {d.ref_id: d.sourcepage for d in documents if d.ref_id and d.sourcepage}
        web_map = {str(w.id): w.url for w in web_results if w.id and w.url}
        sharepoint_entries = sharepoint_results or []
        sharepoint_map = {str(sp.id): sp.web_url.split("/")[-1] for sp in sharepoint_entries if sp.id and sp.web_url}

        def _sub(match: re.Match) -> str:
            ref_id = match.group(1)
            if ref_id in web_map and web_map[ref_id]:
                return f"[{web_map[ref_id]}]"
            if ref_id in sharepoint_map and sharepoint_map[ref_id]:
                return f"[{sharepoint_map[ref_id]}]"
            if ref_id in doc_map and doc_map[ref_id]:
                return f"[{doc_map[ref_id]}]"
            return match.group(0)

        return re.sub(r"\[ref_id:([^\]]+)\]", _sub, answer)

    async def get_sources_content(
        self,
        results: list[Document],
        use_semantic_captions: bool,
        include_text_sources: bool,
        download_image_sources: bool,
        user_oid: Optional[str] = None,
        web_results: Optional[list[WebResult]] = None,
        sharepoint_results: Optional[list[SharePointResult]] = None,
        query_hint: Optional[str] = None,
    ) -> DataPoints:
        """Extract text/image sources & citations from documents.

        Args:
            results: List of retrieved Document objects.
            use_semantic_captions: Whether to use semantic captions instead of full content text.
            download_image_sources: Whether to attempt downloading & base64 encoding referenced images.
            user_oid: Optional user object id for per-user storage access (ADLS scenarios).
            web_results: Optional list of web retrieval results to expose to clients.
            sharepoint_results: Optional list of SharePoint retrieval results to expose to clients.

        Returns:
            DataPoints: with text (structured source objects and legacy strings), images (list[str - base64 data URI]), citations (list[str]).
        """

        def clean_source(s: str, preserve_linebreaks: bool = False) -> str:
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = s.replace(":::", "&#58;&#58;&#58;")  # escape DocFX/markdown triple colons
            # Remove inline metadata prefixes that can leak into supporting content text
            # Example: "SOURCE: ... SOURCEPAGE: ... CATEGORY: ... SECTION: ..."
            s = re.sub(r"\bSOURCE:\s*.*?\bSOURCEPAGE:\s*.*?\bCATEGORY:\s*.*?\bSECTION:\s*", "", s, flags=re.IGNORECASE)

            if preserve_linebreaks:
                # Preserve paragraph and subsection boundaries so frontend highlighting
                # can extract the cited subsection reliably from full_content.
                s = re.sub(r"[ \t]+", " ", s)
                s = re.sub(r" *\n *", "\n", s)
                s = re.sub(r"\n{3,}", "\n\n", s).strip()
                return s

            s = s.replace("\n", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        citations = []
        text_sources: list[str | TextSourceItem] = []
        image_sources = []
        seen_urls = set()
        external_results_metadata: list[dict[str, Any]] = []
        citation_activity_details: dict[str, dict[str, Any]] = {}
        # CUSTOM: Track text_source citations to differentiate chunks from same document
        text_source_citation_counts: dict[str, int] = {}

        if include_text_sources:
            processed_sources = source_processor.process_documents(
                results,
                use_semantic_captions=use_semantic_captions,
                focus_on_indexed_subsection=True,
                adjacent_subsections=1,
                max_unfocused_subsections=4,
                query_hint=query_hint,
            )
            documents_by_id = {str(doc.id): doc for doc in results if doc.id}

            for processed_source in processed_sources:
                original_doc_id = str(processed_source.get("original_doc_id") or processed_source.get("id") or "")
                original_doc = documents_by_id.get(original_doc_id)

                raw_prompt_content = (
                    processed_source.get("caption_summary")
                    if use_semantic_captions and processed_source.get("caption_summary")
                    else processed_source.get("content", "")
                )
                raw_full_content = processed_source.get("full_content") or processed_source.get("content", "")
                cleaned = clean_source(str(raw_prompt_content or ""))
                full_content = clean_source(str(raw_full_content or ""), preserve_linebreaks=True)

                citation = str(processed_source.get("citation") or "")
                if not citation and original_doc:
                    citation = citation_builder.build_enhanced_citation(original_doc, len(citations) + 1)
                if not citation:
                    citation = str(processed_source.get("sourcepage") or "")

                text_source_citation = citation
                if citation in text_source_citation_counts:
                    text_source_citation_counts[citation] += 1
                    count = text_source_citation_counts[citation]

                    if count == 2:
                        for prev_src in text_sources:
                            if isinstance(prev_src, dict) and prev_src.get("citation") == citation:
                                first_heading = self.extract_content_heading(prev_src.get("full_content", "") or prev_src.get("content", ""))
                                if first_heading:
                                    prev_src["citation"] = f"{first_heading}, {citation}"
                                else:
                                    prev_src["citation"] = f"{citation} (Part 1)"
                                try:
                                    first_index = citations.index(citation)
                                    citations[first_index] = prev_src["citation"]
                                except ValueError:
                                    pass
                                break

                    content_heading = self.extract_content_heading(raw_full_content or cleaned)
                    if content_heading:
                        text_source_citation = f"{content_heading}, {citation}"
                    else:
                        text_source_citation = f"{citation} (Part {count})"
                else:
                    text_source_citation_counts[citation] = 1

                resolved_subsection_id = str(
                    processed_source.get("subsection_id")
                    or (self._extract_subsection_from_document(original_doc) if original_doc else "")
                    or ""
                )

                text_sources.append(
                    {
                        "id": str(processed_source.get("id") or original_doc_id or "") or None,
                        "citation": text_source_citation,
                        "content": cleaned,
                        "full_content": full_content,
                        "sourcepage": str(processed_source.get("sourcepage") or getattr(original_doc, "sourcepage", "") or ""),
                        "sourcefile": str(processed_source.get("sourcefile") or getattr(original_doc, "sourcefile", "") or ""),
                        "category": str(processed_source.get("category") or getattr(original_doc, "category", "") or ""),
                        "storageurl": str(
                            processed_source.get("storageurl")
                            or processed_source.get("storageUrl")
                            or getattr(original_doc, "storage_url", "")
                            or ""
                        ),
                        "updated": str(
                            processed_source.get("updated")
                            or getattr(original_doc, "updated", "")
                            or getattr(original_doc, "last_updated", "")
                            or ""
                        ),
                        "subsection_id": resolved_subsection_id,
                    }
                )

        for doc in results:
            citation = citation_builder.build_enhanced_citation(doc, len(citations) + 1)
            if not citation:
                citation = self.get_citation(doc.sourcepage)
            if citation not in citations:
                citations.append(citation)
                if doc.activity:
                    citation_activity_details[citation] = asdict(doc.activity)

            if download_image_sources and hasattr(doc, "images") and doc.images:
                for img in doc.images:
                    # Skip if we've already processed this URL
                    if img["url"] in seen_urls or not img["url"]:
                        continue
                    seen_urls.add(img["url"])
                    url = await self.download_blob_as_base64(img["url"], user_oid=user_oid)
                    if url:
                        image_sources.append(url)
                    image_citation = self.get_image_citation(doc.sourcepage or "", img["url"])
                    citations.append(image_citation)
        if web_results:
            for web in web_results:
                citation = self.get_citation(web.url)
                if citation and citation not in citations:
                    citations.append(citation)
                    # Add activity details if available
                    if web.activity:
                        citation_activity_details[citation] = asdict(web.activity)
                external_results_metadata.append(
                    {
                        "id": web.id,
                        "title": web.title,
                        "url": web.url,
                        "activity": asdict(web.activity) if web.activity else None,
                    }
                )
        if sharepoint_results:
            for sp in sharepoint_results:
                # Extract filename from web_url for citation
                filename = sp.web_url.split("/")[-1] if sp.web_url else ""
                citation = self.get_citation(filename)
                if citation and citation not in citations:
                    citations.append(citation)
                    # Add activity details if available
                    if sp.activity:
                        citation_activity_details[citation] = asdict(sp.activity)
                if include_text_sources and sp.content:
                    text_sources.append(f"[{citation}]: {clean_source(sp.content)}")
                external_results_metadata.append(
                    {
                        "id": sp.id,
                        "title": sp.title or "",
                        "url": sp.web_url or "",
                        "snippet": clean_source(sp.content or ""),
                        "activity": asdict(sp.activity) if sp.activity else None,
                    }
                )

        return DataPoints(
            text=text_sources,
            images=image_sources,
            citations=citations,
            external_results_metadata=external_results_metadata,
            citation_activity_details=citation_activity_details if citation_activity_details else None,
        )

    def get_citation(self, sourcepage: Optional[str]):
        return sourcepage or ""

    def get_image_citation(self, sourcepage: Optional[str], image_url: str):
        sourcepage_citation = self.get_citation(sourcepage)
        image_filename = image_url.split("/")[-1]
        return f"{sourcepage_citation}({image_filename})"

    @staticmethod
    def extract_content_heading(content: str) -> str:
        """Extract a short heading from the first meaningful line of content.

        Used to differentiate citations for chunks from the same document by
        deriving a human-readable label from the chunk text (e.g.,
        "Case Management Conference", "Pre-action Protocol").

        Returns an empty string if no suitable heading is found.
        """
        if not content:
            return ""
        for raw_line in content.split("\n")[:10]:
            line = raw_line.strip()
            if not line or line == "---":
                continue
            # Strip markdown heading markers
            line = re.sub(r"^#+\s*", "", line)
            # Strip breadcrumb markers
            line = re.sub(r"^\[.*?\]\s*", "", line)
            # Strip bold markers
            line = re.sub(r"^\*\*([^*]+)\*\*", r"\1", line)
            line = re.sub(r"^__([^_]+)__", r"\1", line)
            line = line.strip()
            if not line:
                continue
            # Skip lines that are just numbers or very short
            if re.match(r"^\d+\.?\d*$", line) or len(line) < 4:
                continue
            # Truncate long headings
            if len(line) > 60:
                line = line[:57] + "..."
            return line
        return ""

    async def download_blob_as_base64(self, blob_url: str, user_oid: Optional[str] = None) -> Optional[str]:
        """
        Downloads a blob from either Azure Blob Storage or Azure Data Lake Storage and returns it as a base64 encoded string.

        Args:
            blob_url: The URL or path to the blob to download
            user_oid: The user's object ID, required for Data Lake Storage operations and access control

        Returns:
            Optional[str]: The base64 encoded image data with data URI scheme prefix, or None if the blob cannot be downloaded
        """

        # Handle full URLs for both Blob Storage and Data Lake Storage
        container: Optional[str] = None
        if blob_url.startswith("http"):
            url_parts = blob_url.split("/")
            # Extract container name from URL
            # For blob: https://{account}.blob.core.windows.net/{container}/{blob_path}
            # For dfs: https://{account}.dfs.core.windows.net/{filesystem}/{path}
            container = url_parts[3]
            # Extract the blob path portion (everything after the container/filesystem segment)
            blob_path = "/".join(url_parts[4:])
            # If %20 in URL, replace it with a space
            blob_path = blob_path.replace("%20", " ")
        else:
            # Treat as a direct blob path
            blob_path = blob_url

        # Download the blob using the appropriate client
        result = None
        if ".dfs.core.windows.net" in blob_url and self.user_blob_manager:
            result = await self.user_blob_manager.download_blob(blob_path, user_oid=user_oid, container=container)
        elif self.global_blob_manager:
            result = await self.global_blob_manager.download_blob(blob_path, container=container)

        if result:
            content, _ = result  # Unpack the tuple, ignoring properties
            img = base64.b64encode(content).decode("utf-8")
            return f"data:image/png;base64,{img}"
        return None

    async def compute_text_embedding(self, q: str):
        SUPPORTED_DIMENSIONS_MODEL = {
            "text-embedding-ada-002": False,
            "text-embedding-3-small": True,
            "text-embedding-3-large": True,
        }

        class ExtraArgs(TypedDict, total=False):
            dimensions: int

        dimensions_args: ExtraArgs = (
            {"dimensions": self.embedding_dimensions} if SUPPORTED_DIMENSIONS_MODEL[self.embedding_model] else {}
        )
        embedding = await self.openai_client.embeddings.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.embedding_deployment if self.embedding_deployment else self.embedding_model,
            input=q,
            **dimensions_args,
        )
        query_vector = embedding.data[0].embedding
        # This performs an oversampling due to how the search index was setup,
        # so we do not need to explicitly pass in an oversampling parameter here
        return VectorizedQuery(vector=query_vector, k=50, fields=self.embedding_field)

    async def compute_multimodal_embedding(self, q: str):
        if not self.image_embeddings_client:
            raise ValueError("Approach is missing an image embeddings client for multimodal queries")
        multimodal_query_vector = await self.image_embeddings_client.create_embedding_for_text(q)
        return VectorizedQuery(vector=multimodal_query_vector, k=50, fields="images/embedding")

    def get_system_prompt_variables(self, override_prompt: Optional[str]) -> dict[str, str]:
        # Allows client to replace the entire prompt, or to inject into the existing prompt using >>>
        if override_prompt is None:
            return {}
        elif override_prompt.startswith(">>>"):
            return {"injected_prompt": override_prompt[3:]}
        else:
            return {"override_prompt": override_prompt}

    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        """Returns true if the model is a GPT-5 family reasoning model.
        This project no longer supports o-series models as they are rarely used now,
        and they have a slightly different programmatic interface that complicates the code.
        """
        return model.startswith("gpt-5")

    def get_response_token_limit(self, model: str, default_limit: int) -> int:
        if self.is_reasoning_model(model):
            return self.RESPONSE_REASONING_DEFAULT_TOKEN_LIMIT

        return default_limit

    def get_lowest_reasoning_effort(self, model: str) -> ReasoningEffort:
        """Return the lowest valid reasoning_effort for the given model."""
        options = self.get_reasoning_effort_options(model)
        return options[0] if options else None

    @staticmethod
    def get_reasoning_effort_options(model: str) -> list[str]:
        """Return the valid reasoning_effort values for the given model.
        Based off Responses API reference: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        if not Approach.is_reasoning_model(model):
            return []
        # gpt-5.1+ supports "none". Earlier gpt-5 models start at "minimal".
        if model.startswith("gpt-5."):
            minor = int(model[6:].split("-")[0])  # e.g. 4 from "gpt-5.4-pro"
            options = ["none", "low", "medium", "high"]
            # gpt-5.4+ supports "xhigh"
            if minor >= 4:
                options.append("xhigh")
            return options
        return ["minimal", "low", "medium", "high"]

    def create_response(
        self,
        chatgpt_deployment: Optional[str],
        chatgpt_model: str,
        input: list[EasyInputMessageParam],
        overrides: dict[str, Any],
        response_token_limit: int,
        should_stream: bool = False,
        tools: Optional[list[FunctionToolParam]] = None,
        temperature: Optional[float] = None,
        reasoning_effort: ReasoningEffort = None,
    ) -> Awaitable[Response] | Awaitable[AsyncStream[ResponseStreamEvent]]:
        params: dict[str, Any] = {
            "max_output_tokens": response_token_limit,
            "store": False,
        }

        if self.is_reasoning_model(chatgpt_model):
            effort = reasoning_effort or overrides.get("reasoning_effort") or self.reasoning_effort
            if effort:
                params["reasoning"] = {"effort": effort}
        else:
            params["temperature"] = temperature if temperature is not None else overrides.get("temperature", 0.3)

        if should_stream:
            params["stream"] = True

        if tools is not None:
            params["tools"] = tools

        # Azure OpenAI takes the deployment name as the model name
        return self.openai_client.responses.create(  # type: ignore[no-matching-overload]
            model=chatgpt_deployment if chatgpt_deployment else chatgpt_model,
            input=input,
            **params,
        )

    def format_thought_step_for_chatcompletion(
        self,
        title: str,
        messages: list[EasyInputMessageParam],
        overrides: dict[str, Any],
        model: str,
        deployment: Optional[str],
        usage: Optional[CompletionUsage | ResponseUsage] = None,
        reasoning_effort: ReasoningEffort = None,
    ) -> ThoughtStep:
        properties: dict[str, Any] = {"model": model}
        if deployment:
            properties["deployment"] = deployment
        # Only add reasoning_effort setting if the model supports it
        if self.is_reasoning_model(model):
            properties["reasoning_effort"] = reasoning_effort or overrides.get(
                "reasoning_effort", self.reasoning_effort
            )
        if usage:
            properties["token_usage"] = TokenUsageProps.from_usage(usage)
        return ThoughtStep(title, messages, properties)

    async def run(
        self,
        messages: list[EasyInputMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def run_stream(
        self,
        messages: list[EasyInputMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError
