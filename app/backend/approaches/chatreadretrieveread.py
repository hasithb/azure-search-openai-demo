from collections.abc import Awaitable
from typing import Any, Optional, Union, cast
import re
import logging
import json

from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery, VectorizedQuery
from approaches.approach import DataPoints, ExtraInfo, ThoughtStep, Document, TokenUsageProps
from openai import AsyncOpenAI, AsyncStream
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from approaches.approach import DataPoints, ExtraInfo, ThoughtStep
from approaches.chatapproach import ChatApproach
from approaches.promptmanager import PromptManager
from core.authentication import AuthenticationHelper

# Import legal domain customizations
from customizations.approaches import citation_builder, source_processor

def nonewlines(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ")


class ChatReadRetrieveReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        search_index_name: str,
        agent_model: Optional[str],
        agent_deployment: Optional[str],
        agent_client: KnowledgeBaseRetrievalClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        embedding_field: str,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        prompt_manager: PromptManager,
        reasoning_effort: Optional[str] = None,
    ):
        self.search_client = search_client
        self.search_index_name = search_index_name
        self.agent_model = agent_model
        self.agent_deployment = agent_deployment
        self.agent_client = agent_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_field = embedding_field
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.prompt_manager = prompt_manager
        self.query_rewrite_prompt = self.try_load_prompt("chat_query_rewrite.prompty")
        self.query_rewrite_tools = self.try_load_tools("chat_query_rewrite_tools.json")
        self.answer_prompt = self.try_load_prompt("chat_answer_question.prompty")
        self.reasoning_effort = reasoning_effort
        self.include_token_usage = True
        # Add citation mapping storage
        self.citation_map = {}

    def build_enhanced_citation_from_document(self, doc: Document, source_index: int) -> str:
        """Build enhanced citation from document - delegates to customizations module"""
        return citation_builder.build_enhanced_citation(doc, source_index)

    def _extract_subsection_from_document(self, doc: Document) -> str:
        """Extract subsection from document - delegates to customizations module"""
        return citation_builder.extract_subsection(doc)

    def _extract_multiple_subsections_from_document(self, doc: Document) -> list[dict[str, str]]:
        """Extract multiple subsections from document - delegates to customizations module"""
        return citation_builder.extract_multiple_subsections(doc)

    def get_citation_from_document(self, doc: Document) -> str:
        """
        Backwards-compatible citation helper that leverages enhanced citation logic.
        """
        try:
            return self.build_enhanced_citation_from_document(doc, 1)
        except Exception:
            sourcepage = (doc.sourcepage or "").strip()
            sourcefile = (doc.sourcefile or "").strip()
            if sourcepage and sourcefile:
                return f"{sourcepage}, {sourcefile}"
            return sourcefile or sourcepage or "Source"

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[ExtraInfo, Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]]]:
        logging.info("🔍 DIAGNOSTIC_LOG_START: run_until_final_call entered")
        use_agentic_retrieval = True if overrides.get("use_agentic_retrieval") else False
        original_user_query = messages[-1]["content"]
        logging.info(f"🔍 DIAGNOSTIC: use_agentic_retrieval={use_agentic_retrieval}, query='{original_user_query[:100]}...'")

        reasoning_model_support = self.GPT_REASONING_MODELS.get(self.chatgpt_model)
        if reasoning_model_support and (not reasoning_model_support.streaming and should_stream):
            raise Exception(
                f"{self.chatgpt_model} does not support streaming. Please use a different model or disable streaming."
            )
        if use_agentic_retrieval:
            logging.info("🔍 DIAGNOSTIC: Calling run_agentic_retrieval_approach...")
            extra_info = await self.run_agentic_retrieval_approach(messages, overrides, auth_claims)
            logging.info("🔍 DIAGNOSTIC: run_agentic_retrieval_approach completed")
        else:
            logging.info("🔍 DIAGNOSTIC: Calling run_search_approach...")
            extra_info = await self.run_search_approach(messages, overrides, auth_claims)
            logging.info("🔍 DIAGNOSTIC: run_search_approach completed")

        # Pre-build enhanced citations from search results
        self.citation_map = {}
        enhanced_citations = []
        
        logging.info(f"🔍 DEBUG: Building citations from {len(extra_info.data_points.text) if extra_info.data_points.text else 0} data points")
        
        for i, source in enumerate(extra_info.data_points.text, 1):
            logging.info(f"🔍 DEBUG: Source {i} type={type(source).__name__}, is_dict={isinstance(source, dict)}")
            if isinstance(source, dict):
                logging.info(f"🔍 DEBUG: Source {i} keys={list(source.keys())}")
                logging.info(f"🔍 DEBUG: Source {i} sourcepage='{source.get('sourcepage')}' sourcefile='{source.get('sourcefile')}'")
                
                # Create Document object from dict for consistent processing
                doc = Document(
                    id=source.get("id"),
                    content=source.get("content"),
                    sourcepage=source.get("sourcepage"),
                    sourcefile=source.get("sourcefile"),
                    category=source.get("category"),
                    storage_url=source.get("storageUrl"),
                    updated=source.get("updated")
                )
                enhanced_citation = self.build_enhanced_citation_from_document(doc, i)
            else:
                # Handle legacy string format
                logging.info(f"🔍 DEBUG: Source {i} is NOT dict, using fallback. Value: {str(source)[:100]}")
                enhanced_citation = f"Source {i}"
            
            # Store mapping - ensure uniqueness
            citation_key = str(i)
            self.citation_map[citation_key] = enhanced_citation
            enhanced_citations.append(enhanced_citation)
            
            logging.info(f"Citation mapping [{citation_key}] = '{enhanced_citation}'")

        # Format sources for prompt with simple numbering
        text_sources_for_prompt = []
        for i, source in enumerate(extra_info.data_points.text, 1):
            if isinstance(source, dict):
                content = source.get('content', '')
            else:
                # Handle string format (legacy)
                content = str(source)
            
            # Format source with simple numbering for AI
            source_text = f"[{i}]: {content}"
            text_sources_for_prompt.append(source_text)

        messages = self.render_prompt(
             self.answer_prompt,
             self.get_system_prompt_variables(overrides.get("prompt_template"))
             | {
                 "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
                 "past_messages": messages[:-1],
                 "user_query": original_user_query,
                 "text_sources": text_sources_for_prompt,
             },
         )

        # Increase token limit to accommodate full content
        response_token_limit = self.get_response_token_limit(self.chatgpt_model, 8192)  # Increased from 4096
        
        logging.info(f"🔍 DIAGNOSTIC: STEP 3 - Creating final chat completion (max_tokens={response_token_limit})...")
        chat_coroutine = cast(
            Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]],
            self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages,
                overrides,
                response_token_limit,
                should_stream,
            ),
        )

        # Add a readable ThoughtStep for the final answer prompt so it renders in the UI
        extra_info.thoughts.append(
            self.format_thought_step_for_chatcompletion(
                title="Prompt to generate answer",
                messages=messages,
                overrides=overrides,
                model=self.chatgpt_model,
                deployment=self.chatgpt_deployment,
                usage=None,
                reasoning_effort=overrides.get("reasoning_effort", self.reasoning_effort),
            )
        )
        
        logging.info("🔍 DIAGNOSTIC_LOG_END: run_until_final_call returning")
        # Store enhanced citations in extra_info for frontend access
        extra_info.enhanced_citations = enhanced_citations
        extra_info.citation_map = self.citation_map
        
        # Ensure data_points.text contains properly structured data
        if hasattr(extra_info.data_points, 'text') and extra_info.data_points.text:
            # Make sure each item has required fields
            for i, item in enumerate(extra_info.data_points.text):
                if isinstance(item, dict):
                    # Ensure all required fields exist
                    item.setdefault('id', str(i))
                    item.setdefault('sourcepage', '')
                    item.setdefault('sourcefile', '')
                    item.setdefault('content', '')
                    item.setdefault('storageUrl', '')
        
        return (extra_info, chat_coroutine)

    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        """Build search filter with multi-category support - no automatic court detection"""
        filters: list[str] = []

        # Exclude category
        if ex := overrides.get("exclude_category"):
            escaped = str(ex).replace("'", "''")
            filters.append(f"category ne '{escaped}'")

        # Include categories (CSV) - user must explicitly select
        inc = overrides.get("include_category")
        if inc and inc not in ("All", ""):
            parts = [p.strip() for p in inc.split(",")]
            cat_filters = ["category eq '{}'".format(p.replace("'", "''")) for p in parts if p]
            if cat_filters:
                filters.append(f"({' or '.join(cat_filters)})")
        # If no category selected or "All" selected, show everything (no category filter)
        # This removes the automatic fallback to CPR

        # Security filters
        if overrides.get("use_oid_security_filter"):
            oid = auth_claims.get("oid")
            if oid:
                escaped = str(oid).replace("'", "''")
                filters.append(f"oids/any(g:g eq '{escaped}')")
        if overrides.get("use_groups_security_filter"):
            groups = auth_claims.get("groups", []) or []
            if groups:
                group_conditions = []
                for g in groups:
                    g_escaped = str(g).replace("'", "''")
                    group_conditions.append(f"g eq '{g_escaped}'")
                if group_conditions:
                    filters.append(f"groups/any(g:{' or '.join(group_conditions)})")

        return " and ".join(filters) if filters else None

    async def run_search_approach(
        self, messages: list[ChatCompletionMessageParam], overrides: dict[str, Any], auth_claims: dict[str, Any]
    ):
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        use_query_rewriting = True if overrides.get("query_rewriting") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        
        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        
        # Remove automatic court detection - just build filter with user selection
        search_index_filter = self.build_filter(overrides, auth_claims)
        
        query_messages = self.render_prompt(
            self.query_rewrite_prompt, {"user_query": original_user_query, "past_messages": messages[:-1]}
        )
        tools: list[ChatCompletionToolParam] = self.query_rewrite_tools

        # DEBUG: Log the query messages to understand what's being sent
        logging.info(f"🔍 DEBUG: Original user query: '{original_user_query}'")
        logging.info(f"🔍 DEBUG: Past messages count: {len(messages[:-1])}")
        for i, qm in enumerate(query_messages[-3:]):  # Last 3 messages
            role = qm.get("role") if isinstance(qm, dict) else "unknown"
            content = qm.get("content") if isinstance(qm, dict) else str(qm)
            content_preview = str(content)[:200] if content else "empty"
            logging.info(f"🔍 DEBUG: Query message {i}: role={role}, content={content_preview}...")

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question

        logging.info("🔍 DIAGNOSTIC: STEP 1 - Calling OpenAI for query rewrite...")
        chat_completion = cast(
            ChatCompletion,
            await self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages=query_messages,
                overrides=overrides,
                response_token_limit=self.get_response_token_limit(
                    self.chatgpt_model, 100
                ),  # Setting too low risks malformed JSON, setting too high may affect performance
                temperature=0.0,  # Minimize creativity for search query generation
                tools=tools,
                reasoning_effort="low",  # Minimize reasoning for search query generation
            ),
        )

        query_text = self.get_search_query(chat_completion, original_user_query)
        logging.info(f"🔍 DIAGNOSTIC: STEP 1 completed - optimized query: '{query_text}'")

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(query_text))

        # Log the search parameters for debugging (remove court detection logging)
        logging.info(f"Searching with query: {query_text}, top: {top}, filter: {search_index_filter}")
        
        logging.info("🔍 DIAGNOSTIC: STEP 2 - Calling Azure Search...")
        results = await self.search(
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
            use_query_rewriting,
        )

        # Log the search results for debugging
        logging.info("🔍 DIAGNOSTIC: STEP 2 completed")
        logging.info(f"Search returned {len(results)} results")
        for i, result in enumerate(results[:3]):  # Log first 3 results
            content_preview = result.content[:200] if result.content else "No content"
            logging.info(f"Result {i}: id={result.id}, content_length={len(result.content or '')}, preview={content_preview}")

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        structured_sources = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Ensure each source has all required fields with proper field mapping
        for source in structured_sources:
            if isinstance(source, dict):
                # Map common field variations and ensure all required fields are present
                source.setdefault("sourcepage", source.get("source_page", ""))
                source.setdefault("sourcefile", source.get("source_file", ""))
                source.setdefault("category", "")
                source.setdefault("updated", source.get("last_updated", source.get("date_updated", "")))
                source.setdefault("storageurl", source.get("storage_url", source.get("url", "")))
                source.setdefault("url", source.get("storageurl", source.get("storage_url", "")))
                
                # Extract fields from the actual search result if available
                result_idx = structured_sources.index(source)
                if result_idx < len(results):
                    search_result = results[result_idx]
                    
                    # Get fields directly from search result
                    if hasattr(search_result, 'get'):
                        source["sourcepage"] = search_result.get("sourcepage", source.get("sourcepage", ""))
                        source["sourcefile"] = search_result.get("sourcefile", source.get("sourcefile", ""))
                        source["category"] = search_result.get("category", source.get("category", ""))
                        source["updated"] = search_result.get("updated", search_result.get("last_updated", source.get("updated", "")))
                        source["storageurl"] = search_result.get("storageurl", search_result.get("storage_url", source.get("storageurl", "")))
                    else:
                        # Handle Document object attributes
                        source["sourcepage"] = getattr(search_result, "sourcepage", source.get("sourcepage", ""))
                        source["sourcefile"] = getattr(search_result, "sourcefile", source.get("sourcefile", ""))
                        source["category"] = getattr(search_result, "category", source.get("category", ""))
                        source["updated"] = getattr(search_result, "updated", getattr(search_result, "last_updated", source.get("updated", "")))
                        source["storageurl"] = getattr(search_result, "storageurl", getattr(search_result, "storage_url", source.get("storageurl", "")))
                
                # Ensure url field matches storageurl
                if source.get("storageurl") and not source.get("url"):
                    source["url"] = source["storageurl"]
                
                # Log for debugging
                logging.info(f"Final structured source fields: {list(source.keys())}")
                logging.info(f"Final source data: sourcepage='{source.get('sourcepage')}', sourcefile='{source.get('sourcefile')}', category='{source.get('category')}', updated='{source.get('updated')}', storageurl='{source.get('storageurl')}'")

        extra_info = ExtraInfo(
            DataPoints(text=structured_sources),  # Pass structured data to frontend
            thoughts=[
                self.format_thought_step_for_chatcompletion(
                    title="Prompt to generate search query",
                    messages=query_messages,
                    overrides=overrides,
                    model=self.chatgpt_model,
                    deployment=self.chatgpt_deployment,
                    usage=chat_completion.usage,
                    reasoning_effort="low",
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "use_query_rewriting": use_query_rewriting,
                        "top": top,
                        "filter": search_index_filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
            ],
        )
        return extra_info

    def _normalize_agent_results_to_documents(self, response, results) -> list[Document]:
        """
        Convert agentic retrieval results into Document objects for consistent processing
        with the existing citation pipeline.
        """
        documents = []
        for i, result in enumerate(results):
            try:
                source_data = getattr(result, "source_data", None) if hasattr(result, "source_data") else None
                if source_data is None and isinstance(result, dict):
                    source_data = result
                if source_data is None:
                    source_data = {
                        "id": getattr(result, "id", f"agent_result_{i}"),
                        "content": getattr(result, "content", ""),
                        "category": getattr(result, "category", ""),
                        "sourcepage": getattr(result, "sourcepage", ""),
                        "sourcefile": getattr(result, "sourcefile", ""),
                        "storageUrl": getattr(result, "storage_url", getattr(result, "storageUrl", "")),
                    }

                # Prefer parent_id for hydration if present (agent chunks vs full doc)
                parent_id = (
                    source_data.get("parent_id")
                    or source_data.get("parentId")
                    or source_data.get("parentid")
                )
                raw_id = source_data.get("id", f"agent_result_{i}")
                effective_id = parent_id or raw_id

                # Broaden key mapping for robustness
                src_page = (
                    source_data.get("sourcepage")
                    or source_data.get("source_page")
                    or source_data.get("page")
                    or source_data.get("filepath")
                    or ""
                )
                src_file = (
                    source_data.get("sourcefile")
                    or source_data.get("source_file")
                    or source_data.get("document")
                    or source_data.get("title")
                    or ""
                )
                storage_url = (
                    source_data.get("storageUrl")
                    or source_data.get("storage_url")
                    or source_data.get("document_url")
                    or source_data.get("documentUrl")
                    or source_data.get("url")
                    or ""
                )
                updated = (
                    source_data.get("updated")
                    or source_data.get("last_updated")
                    or source_data.get("date_updated")
                    or ""
                )

                doc = Document(
                    id=effective_id,
                    content=source_data.get("content", ""),
                    category=source_data.get("category", ""),
                    sourcepage=src_page,
                    sourcefile=src_file,
                    storage_url=storage_url,
                    updated=updated,
                    oids=source_data.get("oids", []),
                    groups=source_data.get("groups", []),
                    score=getattr(result, "score", source_data.get("score", 0.0)),
                    reranker_score=getattr(result, "reranker_score", source_data.get("reranker_score", 0.0)),
                    captions=getattr(result, "captions", []),
                )

                logging.info(
                    f"Normalized agent result {i} -> id='{doc.id}' "
                    f"(parent_id='{parent_id}', raw_id='{raw_id}'), "
                    f"sourcepage='{doc.sourcepage}', sourcefile='{doc.sourcefile}', storage_url set={bool(doc.storage_url)}"
                )
                documents.append(doc)
            except Exception as e:
                logging.warning(f"Failed to normalize agent result {i}: {e}")
                documents.append(
                    Document(
                        id=f"agent_result_{i}",
                        content=str(getattr(result, "content", "")),
                        category="",
                        sourcepage="",
                        sourcefile="",
                        storage_url="",
                        updated="",
                        oids=[],
                        groups=[],
                        score=0.0,
                        reranker_score=0.0,
                        captions=[],
                    )
                )

        logging.info(f"Normalized {len(results)} agent results to {len(documents)} Document objects")
        return documents

    async def _hydrate_agent_documents_metadata(self, docs: list[Document]) -> list[Document]:
        """
        For agentic retrieval docs that lack full metadata, fetch the full document
        from the search index by id and fill in missing fields (sourcefile, storageUrl, updated, etc.).
        This does not affect the normal search path.
        """
        hydrated: list[Document] = []
        for doc in docs:
            try:
                # Decide if hydration is needed
                needs_hydration = any(
                    [
                        not doc.sourcefile,
                        not doc.storage_url,
                        not doc.updated,
                        not doc.category,
                        not doc.sourcepage,
                    ]
                )
                
                if needs_hydration and doc.id:
                    try:
                        # Fetch full document from the index
                        raw = await self.search_client.get_document(key=str(doc.id))
                        
                        # Safely map known fields (prefer existing values)
                        doc.sourcepage = doc.sourcepage or raw.get(self.sourcepage_field, raw.get("sourcepage", ""))
                        doc.sourcefile = doc.sourcefile or raw.get("sourcefile", raw.get("source_file", ""))
                        doc.category = doc.category or raw.get("category", "")
                        # storage URL appears in different casings
                        doc.storage_url = doc.storage_url or raw.get("storageUrl", raw.get("storage_url", raw.get("url", "")))
                        # updated could be named differently
                        doc.updated = doc.updated or raw.get("updated", raw.get("last_updated", raw.get("date_updated", "")))
                        # Subsection fields for accurate citation navigation
                        doc.subsection_id = doc.subsection_id or raw.get("subsection_id", "")
                        doc.subsections = doc.subsections or raw.get("subsections", [])
                        # Content: keep agent content if present; otherwise hydrate
                        if not doc.content:
                            doc.content = raw.get(self.content_field, raw.get("content", ""))
                    except Exception:
                        pass  # Hydration failed, continue with original doc
                hydrated.append(doc)
            except Exception:
                hydrated.append(doc)
        return hydrated

    async def run_agentic_retrieval_approach(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
    ):
        """
        Run the agentic retrieval approach.
        
        NOTE: Legal terminology expansion (e.g., "pre-action disclosure" -> "disclosure before proceedings")
        is now handled by Azure AI Search synonym maps, configured via scripts/manage_synonym_map.py.
        This is more scalable and doesn't require code changes when adding new terminology.
        """
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0)
        search_index_filter = self.build_filter(overrides, auth_claims)
        top = overrides.get("top", 3)
        max_subqueries = overrides.get("max_subqueries", 10)
        results_merge_strategy = overrides.get("results_merge_strategy", "interleaved")
        # Get retrieval reasoning effort from overrides, default to "low" for best reference support
        retrieval_reasoning_effort = overrides.get("retrieval_reasoning_effort", "low")
        # 50 is the amount of documents that the reranker can process per query
        max_docs_for_reranker = max_subqueries * 50

        response, results = await self.run_agentic_retrieval(
            messages=messages,
            agent_client=self.agent_client,
            search_index_name=self.search_index_name,
            top=top,
            filter_add_on=search_index_filter,
            minimum_reranker_score=minimum_reranker_score,
            max_docs_for_reranker=max_docs_for_reranker,
            results_merge_strategy=results_merge_strategy,
            retrieval_reasoning_effort=retrieval_reasoning_effort,
        )

        # Normalize agent results to Document objects for consistent processing
        normalized_documents = self._normalize_agent_results_to_documents(response, results)

        # NEW: Hydrate missing metadata from the search index (sourcefile, storageUrl, updated, category, etc.)
        hydrated_documents = await self._hydrate_agent_documents_metadata(normalized_documents)
        
        # Reuse existing pipeline for subsection splitting and citation building
        structured_sources = self.get_sources_content(hydrated_documents, use_semantic_captions=False, use_image_citation=False)

        extra_info = ExtraInfo(
            DataPoints(text=structured_sources),  # Pass structured data to frontend
            thoughts=[
                ThoughtStep(
                    "Use agentic retrieval",
                    # Convert messages to readable format
                    self.prompt_manager.messages_to_readable(messages) if hasattr(self.prompt_manager, 'messages_to_readable') else str(messages),
                    {
                        "reranker_threshold": minimum_reranker_score,
                        "max_docs_for_reranker": max_docs_for_reranker,
                        "results_merge_strategy": results_merge_strategy,
                        "filter": search_index_filter,
                    },
                ),
                ThoughtStep(
                    f"Agentic retrieval results (top {top})",
                    [result.serialize_for_results() for result in results],
                    {
                        "query_plan": (
                            [activity.as_dict() for activity in response.activity] if response.activity else None
                        ),
                        "model": self.agent_model,
                        "deployment": self.agent_deployment,
                    },
                ),
            ],
        )
        return extra_info

    def get_citation_from_document_dict(self, doc_dict: dict[str, Any]) -> str:
        """Get citation string from document dict in three-part format"""
        sourcepage = doc_dict.get("sourcepage", "")
        sourcefile = doc_dict.get("sourcefile", "")
        content = doc_dict.get("content", "")
        
        # Extract subsection using citation_builder
        subsection = citation_builder.extract_subsection_from_content(content)
        if not subsection:
            subsection = citation_builder.extract_simple_subsection(sourcepage)
        
        # Format citation
        page_ref = sourcepage if sourcepage else "Unknown Page"
        document = sourcefile if sourcefile else "Unknown Document"
        
        if subsection:
            return f"{subsection}, {page_ref}, {document}"
        else:
            return f"{page_ref}, {document}"
    
    def _extract_subsection_from_content(self, content: str) -> str:
        """Extract subsection from content - delegates to customizations module"""
        return citation_builder.extract_subsection_from_content(content)
    
    def _extract_simple_subsection(self, sourcepage: str) -> str:
        """Extract simple subsection from sourcepage - delegates to customizations module"""
        return citation_builder.extract_simple_subsection(sourcepage)

    def create_chat_completion(
        self,
        chatgpt_deployment: Optional[str],
        chatgpt_model: str,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        response_token_limit: int,
        should_stream: bool = False,
        tools: Optional[list[ChatCompletionToolParam]] = None,
        temperature: Optional[float] = None,
        n: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]]:
        # GPT5_TEMPERATURE_FIX_APPLIED: GPT-5 models only support temperature=1 (default)
        if chatgpt_model in self.GPT_REASONING_MODELS:
            params: dict[str, Any] = {
                # Increase max_completion_tokens to handle full content
                "max_completion_tokens": max(response_token_limit, 16384)  # Increased from 8192
            }

            # Adjust parameters for reasoning models
            supported_features = self.GPT_REASONING_MODELS[chatgpt_model]
            if supported_features.streaming and should_stream:
                params["stream"] = True
                params["stream_options"] = {"include_usage": True}
            params["reasoning_effort"] = reasoning_effort or overrides.get("reasoning_effort") or self.reasoning_effort
            
            # GPT-5 models only support temperature=1, don't override it
            # Ignore any temperature parameter passed in

        else:
            # Include parameters that may not be supported for reasoning models
            params = {
                "max_tokens": max(response_token_limit, 8192),  # Increased from 4096
                "temperature": temperature or overrides.get("temperature", 0.3),
            }
            if should_stream:
                params["stream"] = True
                params["stream_options"] = {"include_usage": True}

        params["tools"] = tools
        # Azure OpenAI takes the deployment name as the model name

        return self.openai_client.chat.completions.create(
            model=chatgpt_deployment if chatgpt_deployment else chatgpt_model,
            messages=messages,
            seed=overrides.get("seed", None),
            n=n or 1,
            **params,
        )

    def format_thought_step_for_chatcompletion(
        self,
        title: str,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        model: str,
        deployment: Optional[str],
        usage: Optional[CompletionUsage] = None,
        reasoning_effort: Optional[str] = None,
    ) -> ThoughtStep:
        properties: dict[str, Any] = {"model": model}
        if deployment:
            properties["deployment"] = deployment
        # Only add reasoning_effort setting if the model supports it
        if model in self.GPT_REASONING_MODELS:
            properties["reasoning_effort"] = reasoning_effort or overrides.get(
                "reasoning_effort", self.reasoning_effort
            )
        if usage:
            properties["token_usage"] = TokenUsageProps.from_completion_usage(usage)

        # Build a readable array of strings, one entry per message, with no truncation
        try:
            readable = self.prompt_manager.messages_to_readable(messages)
            description_segments = [seg for seg in readable.split("\n\n") if seg.strip()]
        except Exception:
            # Fallback to minimal per-message rendering (no trimming)
            description_segments: list[str] = []
            for m in messages:
                if isinstance(m, dict):
                    role = m.get("role", "unknown") or "unknown"
                    content = m.get("content", "")
                else:
                    role = getattr(m, "role", "unknown") or "unknown"
                    content = getattr(m, "content", "")
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                description_segments.append(f"{role.upper()}:\n{str(content)}")

        # Keep raw messages only if explicitly requested, to avoid the JSON “box” in UI
        if overrides.get("include_raw_messages"):
            properties["raw_messages"] = messages

        return ThoughtStep(title, description_segments, properties)

    def _messages_to_description_segments(self, messages: list[ChatCompletionMessageParam]) -> list[str]:
        """Return an array of concise, role-prefixed strings for each message."""
        if not messages:
            return ["No messages"]

        def _as_str(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # multimodal content – indicate count instead of dumping JSON
                return f"[content parts: {len(content)}]"
            return str(content)

        def _trim(s: str, limit: int = 12000) -> str:  # Increased limit to show full prompts without truncation
            s = s.strip()
            return s if len(s) <= limit else s[:limit] + " …"

        segments: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = _as_str(msg.get("content", ""))
            else:
                role = getattr(msg, "role", "unknown")
                content = _as_str(getattr(msg, "content", ""))
            segments.append(f"{role.upper()}:\n{_trim(content)}")
        return segments

    def format_search_results_for_prompt(self, search_results):
        """Format search results to include all key fields in the prompt"""
        formatted_sources = []
        
        for result in search_results:
            # Include all key fields from the index
            source_info = []
            
            # Add category if available
            if result.get("category"):
                source_info.append(f"Category: {result['category']}")
            
            # Add source page
            if result.get("sourcepage"):
                source_info.append(f"Source Page: {result['sourcepage']}")
            
            # Add source file
            if result.get("sourcefile"):
                source_info.append(f"Source File: {result['sourcefile']}")
            
            # Add updated date if available
            if result.get("updated"):
                source_info.append(f"Updated: {result['updated']}")
            
            # Add storage URL
            if result.get("storageUrl"):
                source_info.append(f"URL: {result['storageUrl']}")
            
            # Format the complete source entry
            formatted_source = "\n".join(source_info)
            formatted_source += f"\nContent: {result.get('content', '')}"
            
            formatted_sources.append(formatted_source)
        
        return "\n\n---\n\n".join(formatted_sources)

    async def format_response_with_citations(self, response_text, search_results):
        """Format the response with enhanced citations including all fields"""
        # Store search results for citation generation
        self._current_search_results = search_results
        
        # Extract citations from response
        citations = []
        citation_lookup = {}
        
        for idx, result in enumerate(search_results):
            citation_key = f"[{idx + 1}]"
            citation = {
                "id": result.get("id", f"doc_{idx}"),
                "content": result.get("content", ""),
                "category": result.get("category", ""),
                "sourcepage": result.get("sourcepage", ""),
                "sourcefile": result.get("sourcefile", ""),
                "url": result.get("storageUrl", ""),
                "updated": result.get("updated", ""),
                "title": f"{result.get('sourcefile', '')} - {result.get('sourcepage', '')}",
                "filepath": result.get("sourcepage", ""),
                "metadata": {
                    "category": result.get("category", ""),
                    "sourcefile": result.get("sourcefile", ""),
                    "updated": result.get("updated", "")
                }
            }
            citations.append(citation)
            citation_lookup[citation_key] = citation
        
        return {
            "answer": response_text,
            "citations": citations,
            "thoughts": f"Searched {len(search_results)} sources with all metadata fields included."
        }

    def get_sources_content(self, results: list[Document], use_semantic_captions: bool, use_image_citation: bool) -> list[dict[str, Any]]:
        """
        Return structured data for consistent processing with full content preserved.
        Delegates to customizations.approaches.source_processor for core logic.
        """
        return source_processor.process_documents(results, use_semantic_captions, use_image_citation)

    def _get_subsection_sort_key(self, subsection_id: str) -> tuple:
        """Generate sort key for subsection ordering - delegates to customizations module"""
        return citation_builder.get_subsection_sort_key(subsection_id)

    def get_response_token_limit(self, model_name: str = None, default_limit: int = None) -> int:
        """
        Get the response token limit for the current model.
        
        Args:
            model_name: The model name to get limits for (defaults to self.chatgpt_model)
            default_limit: Default limit to use if provided (overrides model-based calculation)
        
        Returns:
            Token limit for the model
        """
        if default_limit is not None:
            return default_limit
            
        if model_name is None:
            model_name = self.chatgpt_model
        
        # Token limits for different models
        model_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 16384,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "o1": 100000,
            "o1-mini": 65536,
            "o3-mini": 65536,
        }
        
        # Check if it's a reasoning model first
        if model_name in self.GPT_REASONING_MODELS:
            return self.RESPONSE_REASONING_DEFAULT_TOKEN_LIMIT
        
        # Extract base model name (handle versioned models like gpt-4-0613)
        base_model = model_name.lower()
        for model_key in model_limits.keys():
            if model_key in base_model:
                return model_limits[model_key]
        
        # Default to conservative limit if model not found
        return self.RESPONSE_DEFAULT_TOKEN_LIMIT

    async def compute_text_embedding(self, query_text: str) -> VectorizedQuery:
        """
        Compute text embedding for vector search.
        
        Args:
            query_text: The text to compute embedding for
            
        Returns:
            VectorizedQuery object for use in search
        """
        if not query_text:
            raise ValueError("Query text cannot be empty")
        
        # Create embedding using OpenAI client
        embedding_response = await self.openai_client.embeddings.create(
            model=self.embedding_deployment if self.embedding_deployment else self.embedding_model,
            input=query_text
        )
        
        # Extract the embedding vector
        embedding_vector = embedding_response.data[0].embedding
        
        # Create and return VectorizedQuery
        return VectorizedQuery(
            vector=embedding_vector,
            k_nearest_neighbors=50,  # Default number of nearest neighbors
            fields=self.embedding_field
        )