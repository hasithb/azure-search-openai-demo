import os
from abc import ABC
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict, Union, cast
from urllib.parse import urljoin
import re

import aiohttp
from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentAzureSearchDocReference,
    KnowledgeAgentIndexParams,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentRetrievalResponse,
    KnowledgeAgentSearchActivityRecord,
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
        "o3-mini": GPTReasoningModelSupport(streaming=True),
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

    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        include_category = overrides.get("include_category")
        exclude_category = overrides.get("exclude_category")
        security_filter = self.auth_helper.build_security_filters(overrides, auth_claims)
        filters = []
        if include_category:
            filters.append("category eq '{}'".format(include_category.replace("'", "''")))
        if exclude_category:
            filters.append("category ne '{}'".format(exclude_category.replace("'", "''")))
        if security_filter:
            filters.append(security_filter)
        return None if len(filters) == 0 else " and ".join(filters)

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
    ) -> list[Document]:
        search_text = query_text if use_text_search else ""
        search_vectors = vectors if use_vector_search else []
        
        # Explicitly request all fields including full content without truncation
        # Azure Cognitive Search should return full field values when explicitly selected
        select_fields = ["id", "content", "category", "sourcepage", "sourcefile", "storageUrl", "oids", "groups", "updated"]
        
        # Add search_mode to ensure we get comprehensive results
        search_kwargs = {
            "search_text": search_text,
            "filter": filter,
            "top": top * 3,  # Request more results initially to account for filtering
            "select": select_fields,
            "search_mode": "all",  # Use "all" mode for more comprehensive matching
            "include_total_count": True,  # Include total count for debugging
        }
        
        if use_semantic_ranker:
            search_kwargs.update({
                "query_caption": "extractive|highlight-false" if use_semantic_captions else None,
                "query_rewrites": "generative" if use_query_rewriting else None,
                "vector_queries": search_vectors,
                "query_type": QueryType.SEMANTIC,
                "query_language": self.query_language,
                "query_speller": self.query_speller,
                "semantic_configuration_name": "default",
                "semantic_query": query_text,
            })
        else:
            search_kwargs["vector_queries"] = search_vectors
            
        results = await self.search_client.search(**search_kwargs)

        documents = []
        async for page in results.by_page():
            async for document in page:
                # Get the full content from the document
                full_content = document.get("content", "")
                
                # Log if content appears truncated
                if full_content and len(full_content) < 100 and full_content.endswith("..."):
                    import logging
                    logging.warning(f"Potentially truncated content for document {document.get('id')}: {full_content}")
                
                documents.append(
                    Document(
                        id=document.get("id"),
                        content=full_content,  # Use the full content
                        category=document.get("category"),
                        sourcepage=document.get("sourcepage"),
                        sourcefile=document.get("sourcefile"),
                        storage_url=document.get("storageUrl"),
                        oids=document.get("oids"),
                        groups=document.get("groups"),
                        captions=cast(list[QueryCaptionResult], document.get("@search.captions")),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                        updated=document.get("updated"),
                    )
                )

        # Apply filtering and return only requested number of results
        qualified_documents = [
            doc
            for doc in documents
            if (
                (doc.score or 0) >= (minimum_search_score or 0)
                and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
            )
        ]
        
        # Return only the top N results as requested
        return qualified_documents[:top]

    async def run_agentic_retrieval(
        self,
        messages: list[ChatCompletionMessageParam],
        agent_client: KnowledgeAgentRetrievalClient,
        search_index_name: str,
        top: Optional[int] = None,
        filter_add_on: Optional[str] = None,
        minimum_reranker_score: Optional[float] = None,
        max_docs_for_reranker: Optional[int] = None,
        results_merge_strategy: Optional[str] = None,
    ) -> tuple[KnowledgeAgentRetrievalResponse, list[Document]]:
        # STEP 1: Invoke agentic retrieval
        response = await agent_client.retrieve(
            retrieval_request=KnowledgeAgentRetrievalRequest(
                messages=[
                    KnowledgeAgentMessage(
                        role=str(msg["role"]), content=[KnowledgeAgentMessageTextContent(text=str(msg["content"]))]
                    )
                    for msg in messages
                    if msg["role"] != "system"
                ],
                target_index_params=[
                    KnowledgeAgentIndexParams(
                        index_name=search_index_name,
                        reranker_threshold=minimum_reranker_score,
                        max_docs_for_reranker=max_docs_for_reranker,
                        filter_add_on=filter_add_on,
                        include_reference_source_data=True,
                    )
                ],
            )
        )

        # STEP 2: Generate a contextual and content specific answer using the search results and chat history
        activities = response.activity
        activity_mapping = (
            {
                activity.id: activity.query.search if activity.query else ""
                for activity in activities
                if isinstance(activity, KnowledgeAgentSearchActivityRecord)
            }
            if activities
            else {}
        )

        results = []
        if response and response.references:
            if results_merge_strategy == "interleaved":
                # Use interleaved reference order
                references = sorted(response.references, key=lambda reference: int(reference.id))
            else:
                # Default to descending strategy
                references = response.references
            for reference in references:
                if isinstance(reference, KnowledgeAgentAzureSearchDocReference) and reference.source_data:
                    results.append(
                        Document(
                            id=reference.doc_key,
                            content=reference.source_data["content"],
                            sourcepage=reference.source_data["sourcepage"],
                            search_agent_query=activity_mapping[reference.activity_source],
                        )
                    )
                if top and len(results) == top:
                    break

        return response, results

    def get_sources_content(
        self, results: list[Document], use_semantic_captions: bool, use_image_citation: bool
    ) -> list[str]:
        """Return formatted strings with enhanced subsection support for three-part citation format"""

        import logging
        logging.info(f"🔍 DEBUG: Processing {len(results)} documents for sources content")
        
        formatted_results = []
        
        # First pass: process each document and collect subsections
        document_groups = []  # List of (original_doc, [subsections]) tuples
        
        for i, doc in enumerate(results):
            logging.info(f"🔍 DEBUG: Processing document {i+1}/{len(results)}: {doc.id}")
            
            # Check if document contains multiple subsections
            subsections = self._extract_multiple_subsections_from_document(doc)
            
            if len(subsections) > 1:
                logging.info(f"🎯 DEBUG: Document {doc.id} will be split into {len(subsections)} sources")
                # Group this document with its subsections
                document_groups.append((doc, subsections))
            else:
                logging.info(f"❌ DEBUG: Document {doc.id} not split - using original logic")
                # Single subsection or no subsections - treat as single group
                document_groups.append((doc, []))
        
        # Second pass: process groups to create formatted results
        # This ensures subsections from the same document appear together
        for doc, subsections in document_groups:
            if len(subsections) > 1:
                # Sort subsections by their natural order
                subsections.sort(key=lambda x: self._get_subsection_sort_key(x['subsection']))
                
                # Split document into multiple sources, one for each subsection
                for j, subsection_info in enumerate(subsections):
                    subsection_id = subsection_info['subsection']
                    subsection_content = subsection_info['content']
                    
                    # Build three-part citation for this specific subsection
                    sourcepage = doc.sourcepage or "Unknown Page"
                    sourcefile = doc.sourcefile or "Unknown Document"
                    base_citation = f"{subsection_id}, {sourcepage}, {sourcefile}"
                    
                    # Format the source with the subsection-specific content
                    formatted_source = f"[{base_citation}]: {subsection_content}"
                    formatted_results.append(formatted_source)
                    logging.info(f"🎯 DEBUG: Added subsection source {j+1}: {subsection_id}")
            else:
                logging.info(f"❌ DEBUG: Document {doc.id} not split - using original logic")
                # Single subsection or no subsections - use existing logic
                subsection = self._extract_subsection_from_document(doc)
                
                # Build three-part citation
                sourcepage = doc.sourcepage or "Unknown Page"
                sourcefile = doc.sourcefile or "Unknown Document"
                
                if subsection:
                    base_citation = f"{subsection}, {sourcepage}, {sourcefile}"
                else:
                    base_citation = f"{sourcepage}, {sourcefile}"
                
                # Get FULL content without any truncation
                content = doc.content or ""
                
                # If using semantic captions, APPEND them to the full content
                if use_semantic_captions and doc.captions:
                    caption_text = " . ".join([str(c.text) for c in doc.captions])
                    if caption_text:
                        content = f"{content}\n\nSemantic Captions: {caption_text}"
                
                # Format the source with the three-part citation followed by full content
                formatted_source = f"[{base_citation}]: {content}"
                formatted_results.append(formatted_source)

        logging.info(f"🏁 DEBUG: Returning {len(formatted_results)} total formatted sources")
        return formatted_results

    def _get_subsection_sort_key(self, subsection_id: str) -> tuple:
        """Generate sort key for subsection ordering (e.g., D5.1 < D5.2 < D5.10)"""
        import re
        
        # Extract components for natural sorting
        # Handle patterns like: D5.1, A4.2, 1.1, Rule 31.1, Para 5.2
        pattern = r'^([A-Z]*)(\d*)\.?(\d+)(?:\.(\d+))?'
        match = re.match(pattern, subsection_id)
        
        if match:
            prefix = match.group(1) or ""  # D, A, Rule, Para, etc.
            major = int(match.group(2)) if match.group(2) else 0  # 5, 4, 31, etc.
            minor = int(match.group(3)) if match.group(3) else 0  # 1, 2, etc.
            patch = int(match.group(4)) if match.group(4) else 0  # For x.y.z format
            
            return (prefix, major, minor, patch)
        
        # Fallback: alphabetical sort
        return (subsection_id, 0, 0, 0)

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