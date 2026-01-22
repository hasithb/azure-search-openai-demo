import logging
import os
from abc import ABC
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict, Union, cast
from urllib.parse import urljoin
import re

import aiohttp
from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseSearchIndexReference,
    SearchIndexKnowledgeSourceParams,
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRetrievalRequest,
    KnowledgeBaseRetrievalResponse,
    KnowledgeBaseSearchIndexActivityRecord,
    KnowledgeRetrievalMinimalReasoningEffort,
    KnowledgeRetrievalLowReasoningEffort,
    KnowledgeRetrievalMediumReasoningEffort,
    KnowledgeRetrievalSemanticIntent,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
from openai import AsyncOpenAI, AsyncStream
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolParam,
)

from approaches.promptmanager import PromptManager
from core.authentication import AuthenticationHelper

# Import legal domain customizations
from customizations.approaches import citation_builder, source_processor
# CUSTOM: Feature flags for category fallback behavior
from customizations import is_feature_enabled

from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class Citation:
    content: str
    id: str
    title: str
    filepath: str
    url: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class Document:
    id: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    sourcepage: Optional[str] = None
    sourcefile: Optional[str] = None
    storage_url: Optional[str] = None
    oids: Optional[list[str]] = None
    groups: Optional[list[str]] = None
    captions: Optional[list[QueryCaptionResult]] = None
    score: Optional[float] = None
    reranker_score: Optional[float] = None
    search_agent_query: Optional[str] = None
    updated: Optional[str] = None  # Add updated field
    subsection_id: Optional[str] = None  # Primary subsection for citation label
    subsections: Optional[list[str]] = None  # All subsections in chunk for frontend matching

    def serialize_for_results(self) -> dict[str, Any]:
        result_dict = {
            "id": str(self.id) if self.id is not None else "",
            "content": str(self.content) if self.content is not None else "",
            "category": str(self.category) if self.category is not None else "",
            "sourcepage": str(self.sourcepage) if self.sourcepage is not None else "",
            "sourcefile": str(self.sourcefile) if self.sourcefile is not None else "",
            "storageUrl": str(self.storage_url) if self.storage_url is not None else "",
            "oids": self.oids if self.oids is not None else [],
            "groups": self.groups if self.groups is not None else [],
            "captions": (
                [
                    {
                        "additional_properties": caption.additional_properties if caption.additional_properties is not None else {},
                        "text": str(caption.text) if caption.text is not None else "",
                        "highlights": str(caption.highlights) if caption.highlights is not None else "",
                    }
                    for caption in self.captions
                ]
                if self.captions
                else []
            ),
            "score": float(self.score) if self.score is not None else 0.0,
            "reranker_score": float(self.reranker_score) if self.reranker_score is not None else 0.0,
            "subsection_id": str(self.subsection_id) if self.subsection_id is not None else "",
            "subsections": self.subsections if self.subsections is not None else [],
            "search_agent_query": str(self.search_agent_query) if self.search_agent_query is not None else "",
            "updated": str(self.updated) if self.updated is not None else "",  # Include updated field
        }
        return result_dict


@dataclass
class ThoughtStep:
    title: str
    description: Optional[Any]
    props: Optional[dict[str, Any]] = None

    def update_token_usage(self, usage: CompletionUsage) -> None:
        if self.props:
            self.props["token_usage"] = TokenUsageProps.from_completion_usage(usage)


@dataclass
class DataPoints:
    text: Optional[list[str]] = None
    images: Optional[list] = None


@dataclass
class ExtraInfo:
    data_points: DataPoints
    thoughts: Optional[list[ThoughtStep]] = None
    followup_questions: Optional[list[Any]] = None
    enhanced_citations: Optional[list[str]] = None  # Add enhanced citations
    citation_map: Optional[dict[str, str]] = None   # Add citation mapping


@dataclass
class TokenUsageProps:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: Optional[int]
    total_tokens: int

    @classmethod
    def from_completion_usage(cls, usage: CompletionUsage) -> "TokenUsageProps":
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=(
                usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else None
            ),
            total_tokens=usage.total_tokens,
        )


# GPT reasoning models don't support the same set of parameters as other models
# https://learn.microsoft.com/azure/ai-services/openai/how-to/reasoning
@dataclass
class GPTReasoningModelSupport:
    streaming: bool


class Approach(ABC):
    # List of GPT reasoning models support
    GPT_REASONING_MODELS = {
        "o1": GPTReasoningModelSupport(streaming=False),
        "o3": GPTReasoningModelSupport(streaming=True),
        "o3-mini": GPTReasoningModelSupport(streaming=True),
        "o4-mini": GPTReasoningModelSupport(streaming=True),
        "gpt-5": GPTReasoningModelSupport(streaming=True),
        "gpt-5-nano": GPTReasoningModelSupport(streaming=True),
        "gpt-5-mini": GPTReasoningModelSupport(streaming=True),
    }
    # Set a higher token limit for GPT reasoning models
    RESPONSE_DEFAULT_TOKEN_LIMIT = 1024
    RESPONSE_REASONING_DEFAULT_TOKEN_LIMIT = 8192

    def __init__(
        self,
        search_client: SearchClient,
        openai_client: AsyncOpenAI,
        auth_helper: AuthenticationHelper,
        query_language: Optional[str],
        query_speller: Optional[str],
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        embedding_field: str,
        openai_host: str,
        vision_endpoint: str,
        vision_token_provider: Callable[[], Awaitable[str]],
        prompt_manager: PromptManager,
        reasoning_effort: Optional[str] = None,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.query_language = query_language
        self.query_speller = query_speller
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_field = embedding_field
        self.openai_host = openai_host
        self.vision_endpoint = vision_endpoint
        self.vision_token_provider = vision_token_provider
        self.prompt_manager = prompt_manager
        self.reasoning_effort = reasoning_effort
        self.include_token_usage = True

    # NEW: safe loaders and fallback prompt renderer to avoid NotImplementedError during startup/use
    def try_load_prompt(self, name: str):
        try:
            return self.prompt_manager.load_prompt(name)
        except NotImplementedError:
            return None
        except Exception:
            return None

    def try_load_tools(self, name: str):
        try:
            return self.prompt_manager.load_tools(name)
        except NotImplementedError:
            return None
        except Exception:
            return None

    def render_prompt(self, prompt: Any, variables: dict[str, Any]) -> list[ChatCompletionMessageParam]:
        """
        Wrapper around prompt_manager.render_prompt with a minimal, robust fallback.
        Produces a simple system + sources + user message when PromptManager isn't implemented.
        """
        try:
            return self.prompt_manager.render_prompt(prompt, variables)
        except NotImplementedError:
            pass
        except Exception:
            pass

        # Fallback: simple messages
        messages: list[ChatCompletionMessageParam] = []
        system_content = (
            "You are an assistant. Answer using only the provided sources. "
            "Use simple numeric citations like [1], [2]. If the answer is unknown, say you don't know."
        )
        messages.append({"role": "system", "content": system_content})

        text_sources = variables.get("text_sources")
        if text_sources:
            # Ensure each source is a string; keep as-is (already numbered or labeled by caller)
            joined_sources = "\n".join(str(s) for s in text_sources)
            messages.append({"role": "system", "content": "Sources:\n" + joined_sources})

        past = variables.get("past_messages")
        if isinstance(past, list) and past:
            for m in past:
                # Keep minimal mapping; ignore malformed entries
                role = (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) or "user"
                content = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
                messages.append({"role": role, "content": str(content)})

        user_query = variables.get("user_query", "")
        messages.append({"role": "user", "content": str(user_query)})

        return messages

    def add_fuzzy_operators(self, query_text: str, edit_distance: int = 1) -> str:
        """Add fuzzy operators (~1 or ~2) to search terms for typo tolerance.
        
        Args:
            query_text: The search query
            edit_distance: Edit distance for fuzzy matching (1 or 2)
        
        Returns:
            Query with fuzzy operators applied to words longer than 2 characters
        """
        import re
        # Match words and preserve operators like AND, OR, NOT
        words = re.findall(r'\b\w+\b|AND|OR|NOT', query_text)
        fuzzy_words = []
        
        for word in words:
            # Skip boolean operators and short words
            if word in ('AND', 'OR', 'NOT') or len(word) <= 2:
                fuzzy_words.append(word)
            else:
                # Add fuzzy operator ~1 for typo tolerance (edit distance 1)
                fuzzy_words.append(f"{word}~{edit_distance}")
        
        return " ".join(fuzzy_words)

    def should_use_vector_search(self, overrides: dict[str, Any]) -> bool:
        """
        Determine if vector search should be used based on overrides and configuration.
        """
        # If embedding deployment is not configured, we can't do vector search
        if not self.embedding_deployment:
            return False
            
        retrieval_mode = overrides.get("retrieval_mode")
        return retrieval_mode in ["vectors", "hybrid", None]

    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        include_category = overrides.get("include_category")
        exclude_category = overrides.get("exclude_category")
        security_filter = self.auth_helper.build_security_filters(overrides, auth_claims)
        filters = []

        if include_category and include_category not in ("All", ""):
            if "," in include_category:
                cats = []
                for c in include_category.split(","):
                    c = c.strip()
                    if c:
                        escaped = c.replace("'", "''")
                        cats.append(f"category eq '{escaped}'")
                if cats:
                    filters.append(f"({' or '.join(cats)})")
            else:
                escaped = include_category.replace("'", "''")
                filters.append(f"category eq '{escaped}'")

        if exclude_category:
            escaped = exclude_category.replace("'", "''")
            filters.append(f"category ne '{excluded}'")
        if security_filter:
            filters.append(security_filter)
        return None if len(filters) == 0 else " and ".join(filters)

    async def search(
        self,
        top: int,
        query_text: str,
        filter: Optional[str],
        vectors: list[VectorQuery],
        use_text_search: bool,
        use_vector_search: bool,
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
        minimum_search_score: float,
        minimum_reranker_score: float,
        use_query_rewriting: bool = False,
    ) -> list[Document]:
        # Add fuzzy operators for typo tolerance if doing text search
        search_text = self.add_fuzzy_operators(query_text) if use_text_search else ""
        vector_queries = vectors if use_vector_search else []

        select_fields = [
            "id", "content", "category", "sourcepage", "sourcefile",
            "storageUrl", "updated", "oids", "groups"
        ]

        if use_semantic_ranker:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                select=select_fields,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                query_rewrites="generative" if use_query_rewriting else None,
                vector_queries=vector_queries,
                query_type=QueryType.SEMANTIC,
                query_language=self.query_language,
                query_speller=self.query_speller,
                semantic_configuration_name="default",
                semantic_query=query_text,
            )
        else:
            # For non-semantic search, use FULL query type to support fuzzy operators
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                select=select_fields,
                vector_queries=vector_queries,
                query_type=QueryType.FULL,
            )

        documents: list[Document] = []
        async for page in results.by_page():
            async for d in page:
                documents.append(
                    Document(
                        id=d.get("id"),
                        content=d.get("content"),
                        category=d.get("category"),
                        sourcepage=d.get("sourcepage"),
                        sourcefile=d.get("sourcefile"),
                        storage_url=d.get("storageUrl"),
                        oids=d.get("oids"),
                        groups=d.get("groups"),
                        captions=cast(list[QueryCaptionResult], d.get("@search.captions")),
                        score=d.get("@search.score"),
                        reranker_score=d.get("@search.reranker_score"),
                        search_agent_query=d.get("@search.query"),
                        updated=d.get("updated"),
                    )
                )

        qualified = [
            doc for doc in documents
            if (doc.score or 0) >= (minimum_search_score or 0)
            and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
        ]
        return qualified[:top]

    async def run_agentic_retrieval(
        self,
        messages: list[ChatCompletionMessageParam],
        agent_client: KnowledgeBaseRetrievalClient,
        search_index_name: str,
        top: Optional[int] = None,
        filter_add_on: Optional[str] = None,
        minimum_reranker_score: Optional[float] = None,
        max_docs_for_reranker: Optional[int] = None,
        results_merge_strategy: Optional[str] = None,
        retrieval_reasoning_effort: Optional[str] = None,  # NEW: Added reasoning effort parameter
    ) -> tuple[KnowledgeBaseRetrievalResponse, list[Document]]:
        # STEP 1: Invoke agentic retrieval
        # Treat 0.0 as None for reranker threshold to avoid aggressive filtering
        threshold = minimum_reranker_score if minimum_reranker_score is not None and minimum_reranker_score > 0 else None
        
        # Build retrieval request kwargs
        agentic_retrieval_input: dict[str, Any] = {}
        
        # Handle reasoning effort - use proper classes based on effort level
        # For minimal reasoning, use semantic intents instead of messages
        if retrieval_reasoning_effort == "minimal":
            last_content = messages[-1]["content"] if messages else ""
            if not isinstance(last_content, str):
                raise ValueError("The most recent message content must be a string for minimal reasoning effort.")
            agentic_retrieval_input["intents"] = [KnowledgeRetrievalSemanticIntent(search=last_content)]
        else:
            # For low/medium/high, use messages format
            kb_messages = [
                KnowledgeBaseMessage(
                    role=str(msg["role"]), content=[KnowledgeBaseMessageTextContent(text=str(msg["content"]))]
                )
                for msg in messages
                if msg["role"] != "system"
            ]
            agentic_retrieval_input["messages"] = kb_messages
        
        # Set output mode to extractiveData to get raw content without answer synthesis
        agentic_retrieval_input["output_mode"] = "extractiveData"
        
        # Determine the reasoning effort class
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
        # If None or unrecognized, don't set reasoning effort (use API default)
        
        def build_knowledge_source_params(filter_value: Optional[str], always_query: bool) -> SearchIndexKnowledgeSourceParams:
            return SearchIndexKnowledgeSourceParams(
                knowledge_source_name=search_index_name,
                reranker_threshold=threshold,
                filter_add_on=filter_value,
                include_references=True,
                include_reference_source_data=True,
                always_query_source=always_query,
            )

        request_kwargs: dict[str, Any] = {
            "knowledge_source_params": [
                build_knowledge_source_params(filter_add_on, False),
            ],
            "include_activity": True,
        }
        if retrieval_effort is not None:
            request_kwargs["retrieval_reasoning_effort"] = retrieval_effort
        request_kwargs.update(agentic_retrieval_input)
        
        # Make the API call
        response = await agent_client.retrieve(
            retrieval_request=KnowledgeBaseRetrievalRequest(**request_kwargs)
        )

        def build_results(resp: KnowledgeBaseRetrievalResponse) -> tuple[list[Document], dict[str, str]]:
            activities = resp.activity
            activity_mapping = (
                {
                    activity.id: activity.search_index_arguments.search if activity.search_index_arguments else ""
                    for activity in activities
                    if isinstance(activity, KnowledgeBaseSearchIndexActivityRecord)
                }
                if activities
                else {}
            )

            results: list[Document] = []
            if resp and resp.references:
                if results_merge_strategy == "interleaved":
                    # Use interleaved reference order
                    references = sorted(resp.references, key=lambda reference: int(reference.id))
                else:
                    # Default to descending strategy
                    references = resp.references
                for reference in references:
                    if isinstance(reference, KnowledgeBaseSearchIndexReference) and reference.source_data:
                        source_data = reference.source_data
                        results.append(
                            Document(
                                id=reference.doc_key,
                                content=source_data.get("content", ""),
                                sourcepage=source_data.get("sourcepage", ""),
                                sourcefile=source_data.get("sourcefile", ""),
                                category=source_data.get("category", ""),
                                storage_url=source_data.get("storageUrl", source_data.get("storage_url", "")),
                                updated=source_data.get("updated", ""),
                                search_agent_query=activity_mapping.get(reference.activity_source, ""),
                            )
                        )
                    if top and len(results) == top:
                        break

            return results, activity_mapping

        results, activity_mapping = build_results(response)

        # CUSTOM: If no references returned, force querying sources
        if not results and is_feature_enabled("agentic_force_query_on_empty"):
            logging.info("CUSTOM: No agentic results; retrying with always_query_source=True.")
            retry_kwargs = dict(request_kwargs)
            retry_kwargs["knowledge_source_params"] = [
                build_knowledge_source_params(filter_add_on, True),
            ]
            response = await agent_client.retrieve(
                retrieval_request=KnowledgeBaseRetrievalRequest(**retry_kwargs)
            )
            results, activity_mapping = build_results(response)

        # CUSTOM: If references are still missing, fall back to direct search using agentic query plan
        if not results and is_feature_enabled("agentic_fallback_search"):
            logging.info("CUSTOM: No agentic references; running fallback search using agentic queries.")
            query_candidates = [q for q in activity_mapping.values() if q]
            if not query_candidates and messages:
                last_content = messages[-1]["content"] if messages else ""
                if isinstance(last_content, str) and last_content:
                    query_candidates = [last_content]

            seen_ids = set()
            for q in query_candidates:
                docs = await self.search(
                    top=top or 3,
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
                    if doc.id in seen_ids:
                        continue
                    seen_ids.add(doc.id)
                    results.append(doc)
                if top and len(results) >= top:
                    results = results[:top]
                    break

        return response, results

    def get_sources_content(
        self, results: list[Document], use_semantic_captions: bool, use_image_citation: bool
    ) -> list[str]:
        """
        Return formatted strings with enhanced subsection support.
        Delegates to customizations.approaches.source_processor for structured processing,
        then formats as strings for prompt inclusion.
        """
        import logging
        
        # Use source_processor for structured data
        structured_sources = source_processor.process_documents(results, use_semantic_captions, use_image_citation)
        
        # Convert structured data to formatted strings for prompt
        formatted_results = []
        for source in structured_sources:
            citation = source.get("citation", "Unknown source")
            content = source.get("content", "")
            
            # If using semantic captions, append them
            if use_semantic_captions and source.get("caption_summary"):
                content = f"{content}\n\nSemantic Captions: {source['caption_summary']}"
            
            formatted_source = f"[{citation}]: {content}"
            formatted_results.append(formatted_source)
        
        logging.info(f"🏁 DEBUG: Returning {len(formatted_results)} total formatted sources")
        return formatted_results

    def _get_subsection_sort_key(self, subsection_id: str) -> tuple:
        """Generate sort key for subsection ordering - delegates to customizations module"""
        return citation_builder.get_subsection_sort_key(subsection_id)

    def _extract_subsection_from_document(self, doc: Document) -> str:
        """Extract subsection from document - delegates to customizations module"""
        return citation_builder.extract_subsection(doc)

    def _extract_multiple_subsections_from_document(self, doc: Document) -> list[dict[str, str]]:
        """Extract multiple subsections from document - delegates to customizations module"""
        return citation_builder.extract_multiple_subsections(doc)

    def get_system_prompt_variables(self, prompt_template: Optional[str] = None) -> dict[str, Any]:
        """Get system prompt variables"""
        return {
            "prompt_template": prompt_template or "Default system prompt",
        }

    async def compute_text_embedding(self, query_text: str) -> VectorQuery:
        """Compute text embedding for vector search"""
        if not query_text:
            raise ValueError("Query text cannot be empty")
        
        # Create embedding using OpenAI client
        embedding_response = await self.openai_client.embeddings.create(
            model=self.embedding_deployment if self.embedding_deployment else self.embedding_model,
            input=query_text
        )
        
        # Extract the embedding vector
        embedding_vector = embedding_response.data[0].embedding
        
        # Create and return VectorQuery
        return VectorQuery(
            vector=embedding_vector,
            k_nearest_neighbors=50,
            fields=self.embedding_field
        )

# Add missing run methods to satisfy ABC requirements
async def run(
    self,
    messages: list[ChatCompletionMessageParam],
    session_state: Any = None,
    context: dict[str, Any] = {},
) -> dict[str, Any]:
    raise NotImplementedError

async def run_stream(
    self,
    messages: list[ChatCompletionMessageParam],
    session_state: Any = None,
    context: dict[str, Any] = {},
) -> AsyncGenerator[dict[str, Any], None]:
    raise NotImplementedError
