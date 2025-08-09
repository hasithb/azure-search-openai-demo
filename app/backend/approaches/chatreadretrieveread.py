from collections.abc import Awaitable
from typing import Any, Optional, Union, cast
import re
import logging

from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
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
        agent_client: KnowledgeAgentRetrievalClient,
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
        self.query_rewrite_prompt = self.prompt_manager.load_prompt("chat_query_rewrite.prompty")
        self.query_rewrite_tools = self.prompt_manager.load_tools("chat_query_rewrite_tools.json")
        self.answer_prompt = self.prompt_manager.load_prompt("chat_answer_question.prompty")
        self.reasoning_effort = reasoning_effort
        self.include_token_usage = True
        # Add citation mapping storage
        self.citation_map = {}

    def build_enhanced_citation_from_document(self, doc: Document, source_index: int) -> str:
        """Build enhanced citation from document with improved logic for proper formatting"""
        sourcepage = doc.sourcepage or ""
        sourcefile = doc.sourcefile or ""
        
        # Clean up sourcepage and sourcefile
        sourcepage = sourcepage.strip()
        sourcefile = sourcefile.strip()
        
        # Extract subsection from content with enhanced logic
        subsection = self._extract_subsection_from_document(doc)
        
        # Special handling to avoid duplication and encoded formats
        final_sourcepage = sourcepage
        
        if subsection and sourcepage:
            # Check if sourcepage is just the subsection (avoid duplication)
            if sourcepage == subsection:
                final_sourcepage = ""
            # Check if sourcepage contains encoded format like "PD3E-1.1" where we extracted "1.1"
            elif re.search(r'[A-Z]+\d*[A-Z]*-' + re.escape(subsection), sourcepage):
                # Skip encoded sourcepage - use two-part citation instead
                final_sourcepage = ""
            # If subsection is a simple number like "1.1" but sourcepage is descriptive, keep both
            elif re.match(r'^[A-Z]?\d+(?:\.\d+)?$', subsection) and len(sourcepage) > len(subsection):
                # Keep both - this creates the desired three-part format
                pass
        
        # Build citation based on what we have
        if subsection and final_sourcepage and sourcefile:
            # Full three-part format: subsection, sourcepage, sourcefile
            citation = f"{subsection}, {final_sourcepage}, {sourcefile}"
        elif subsection and sourcefile:
            # Subsection + document (no sourcepage or sourcepage was duplicate)
            citation = f"{subsection}, {sourcefile}"
        elif final_sourcepage and sourcefile:
            # Traditional two-part: sourcepage, sourcefile
            citation = f"{final_sourcepage}, {sourcefile}"
        elif sourcefile:
            # Just document
            citation = sourcefile
        else:
            # Fallback
            citation = f"Source {source_index}"
        
        logging.info(f"Built citation: {citation} from doc with sourcepage='{doc.sourcepage}', sourcefile='{doc.sourcefile}', subsection='{subsection}'")
        return citation

    def _extract_subsection_from_document(self, doc: Document) -> str:
        """Extract subsection from document content with improved logic"""
        if not doc:
            return ""
        
        # Priority 1: Check content first for specific subsection numbers (like 1.1, A4.1)
        if doc.content:
            lines = doc.content.split('\n')[:20]  # Check first 20 lines
            
            for line in lines:
                line = line.strip()
                if not line or line == "---":  # Skip empty lines and dividers
                    continue
                
                # Look for specific subsection patterns at start of line OR standalone on line
                subsection_patterns = [
                    r'^([A-Z]\d+\.\d+)\b',           # A4.1, B2.3, etc.
                    r'^(\d+\.\d+)\b',                # 1.1, 2.3, etc.
                    r'^([A-Z]\d+)\b',                # A1, B2, etc.
                    r'^(Rule \d+(?:\.\d+)?)\b',      # Rule 1, Rule 1.1
                    r'^(Para \d+(?:\.\d+)?)\b',      # Para 1.1
                    r'^(\d+\.\d+)$',                 # Standalone subsection number (exact match)
                ]
                
                for pattern in subsection_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group(1)
                        
                # Special case: check if this line is JUST a subsection number (like "1.1" on its own line)
                if re.match(r'^\d+\.\d+$', line):
                    return line
                    
        # Priority 2: Check sourcepage for encoded subsections (like PD3E-1.1)
        if doc.sourcepage:
            # Handle encoded formats like "PD3E-1.1" -> extract "1.1"
            encoded_patterns = [
                r'[A-Z]+\d*[A-Z]*-([A-Z]\d+\.\d+)',  # PD3E-A4.1 -> A4.1
                r'[A-Z]+\d*[A-Z]*-(\d+\.\d+)',       # PD3E-1.1 -> 1.1
                r'[A-Z]+\d*[A-Z]*-([A-Z]\d+)',       # PD3E-A4 -> A4
            ]
            
            for pattern in encoded_patterns:
                match = re.search(pattern, doc.sourcepage)
                if match:
                    return match.group(1)
        
        # Priority 3: Check sourcepage for direct subsection patterns
        if doc.sourcepage:
            direct_patterns = [
                r'^([A-Z]\d+\.\d+)\b',           # A4.1, B2.3, etc.
                r'^(\d+\.\d+)\b',                # 1.1, 2.3, etc.
                r'^([A-Z]\d+)\b',                # A1, B2, etc.
                r'^(Rule \d+(?:\.\d+)?)\b',      # Rule 1, Rule 1.1
                r'^(Part \d+)\b',                # Part 1, Part 2, etc.
            ]
            
            for pattern in direct_patterns:
                match = re.match(pattern, doc.sourcepage, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        return ""

    # NEW: split a document into multiple subsection chunks with identifiers
    def _extract_multiple_subsections_from_document(self, doc: Document) -> list[dict[str, str]]:
        """
        Scan document content and split into multiple subsections when clear subsection
        markers are found (e.g., '1.1', 'A4.1', 'Rule 31.1', 'Para 5.2'). Returns a list
        of dicts with keys: 'subsection' and 'content'. Returns [] if no reliable split.
        """
        if not doc or not doc.content:
            return []

        lines = doc.content.splitlines()
        # Pattern covers: Rule 1.1, CPR 1.1, Para/Paragraph 1.1, A4.1, 1.1, A4
        pattern = re.compile(
            r'^(?:'
            r'(?P<rule>Rule\s+\d+(?:\.\d+)?)|'
            r'(?P<cpr>CPR\s+\d+(?:\.\d+)?)|'
            r'(?P<para>Para(?:graph)?\s+\d+(?:\.\d+)?)|'
            r'(?P<alpha_num_dotted>[A-Z]\d+\.\d+)|'
            r'(?P<num_dotted>\d+\.\d+)|'
            r'(?P<alpha_num>[A-Z]\d+)'
            r')\b',
            re.IGNORECASE,
        )

        # Collect candidate subsection headings with their start line index
        headings: list[tuple[str, int]] = []
        for i, raw in enumerate(lines):
            line = raw.strip()
            if not line or line == "---":
                continue
            m = pattern.match(line)
            if not m:
                continue
            label = m.group(0)
            # Normalize common prefixes for consistency
            if m.lastgroup == "rule":
                label = "Rule " + re.search(r'\d+(?:\.\d+)?', label).group(0)
            elif m.lastgroup == "cpr":
                label = "CPR " + re.search(r'\d+(?:\.\d+)?', label, re.IGNORECASE).group(0)
            elif m.lastgroup and m.lastgroup.startswith("para"):
                label = "Para " + re.search(r'\d+(?:\.\d+)?', label, re.IGNORECASE).group(0)
            else:
                # Ensure uppercase for alpha prefix like 'a4.1' -> 'A4.1'
                label = label.upper()
            headings.append((label, i))

        # If we can’t confidently detect multiple subsections, don’t split
        if len(headings) <= 1:
            return []

        # Build subsection chunks
        subsections: list[dict[str, str]] = []
        for idx, (label, start_line) in enumerate(headings):
            end_line = headings[idx + 1][1] if idx + 1 < len(headings) else len(lines)
            chunk_lines = lines[start_line:end_line]
            content = "\n".join(chunk_lines).strip()
            # Skip pathological tiny chunks
            if not content or len(content) < 10:
                continue
            subsections.append({"subsection": label, "content": content})

        return subsections

    # NEW: lightweight wrapper to keep existing callsites working
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
        use_agentic_retrieval = True if overrides.get("use_agentic_retrieval") else False
        original_user_query = messages[-1]["content"]

        reasoning_model_support = self.GPT_REASONING_MODELS.get(self.chatgpt_model)
        if reasoning_model_support and (not reasoning_model_support.streaming and should_stream):
            raise Exception(
                f"{self.chatgpt_model} does not support streaming. Please use a different model or disable streaming."
            )
        if use_agentic_retrieval:
            extra_info = await self.run_agentic_retrieval_approach(messages, overrides, auth_claims)
        else:
            extra_info = await self.run_search_approach(messages, overrides, auth_claims)

        # Pre-build enhanced citations from search results
        self.citation_map = {}
        enhanced_citations = []
        
        for i, source in enumerate(extra_info.data_points.text, 1):
            if isinstance(source, dict):
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

        messages = self.prompt_manager.render_prompt(
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

    def detect_court_in_query(self, query: str) -> Optional[str]:
        """
        Detect if a specific court is mentioned in the user query.
        Returns the court name if found, None otherwise.
        """
        # Common court patterns to look for
        court_patterns = [
            r'\b(?:county\s+court|high\s+court|crown\s+court|magistrates?\s+court|circuit\s+commercial\s+court|commercial\s+court|family\s+court|employment\s+tribunal|court\s+of\s+appeal|supreme\s+court)\b',
            r'\b(?:CC|HC|QBD|ChD|FD|CCC)\b',  # Common abbreviations
        ]
        
        query_lower = query.lower()
        for pattern in court_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None

    def normalize_court_to_category(self, court_name: str) -> Optional[str]:
        """
        Normalize court name to match category format.
        Returns the normalized category name if the court should be treated as a category.
        """
        if not court_name:
            return None
            
        # Map of court names to their category format
        court_category_map = {
            'circuit commercial court': 'Circuit Commercial Court',
            'commercial court': 'Commercial Court',
            'high court': 'High Court',
            'county court': 'County Court',
            'crown court': 'Crown Court',
            'magistrates court': 'Magistrates Court',
            'family court': 'Family Court',
            'employment tribunal': 'Employment Tribunal',
            'court of appeal': 'Court of Appeal',
            'supreme court': 'Supreme Court',
            'ccc': 'Circuit Commercial Court',
            'hc': 'High Court',
            'qbd': "Queen's Bench Division",
            'chd': 'Chancery Division',
            'fd': 'Family Division'
        }
        
        court_lower = court_name.lower().strip()
        return court_category_map.get(court_lower)

    async def check_if_court_is_category(self, court_name: str) -> bool:
        """
        Check if the detected court name exists as a category in the search index.
        """
        if not court_name:
            return False
            
        normalized_court = self.normalize_court_to_category(court_name)
        if not normalized_court:
            return False
            
        try:
            # Search for documents with this category
            filter_query = f"category eq '{normalized_court}'"
            results = await self.search_client.search(
                search_text="",
                filter=filter_query,
                top=1,
                select=["category"]
            )
            
            # If we get any results, the court exists as a category
            async for _ in results:
                return True
                
        except Exception as e:
            import logging
            logging.warning(f"Error checking if court is category: {e}")
            
        return False

    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        """Build search filter with enhanced court/category detection"""
        filters = []
        
        # First, check if exclude_category is specified
        exclude_category = overrides.get("exclude_category", None)
        if exclude_category:
            filters.append(f"category ne '{exclude_category}'")
        
        # Check if include_category is specified
        include_category = overrides.get("include_category", None)
        if include_category and include_category != "All":
            filters.append(f"category eq '{include_category}'")
        else:
            # If no specific category is included, check if court is mentioned in the query
            original_user_query = overrides.get("original_user_query", "")
            detected_court = self.detect_court_in_query(original_user_query)
            
            if detected_court:
                # Check if this court is actually a category
                normalized_court = self.normalize_court_to_category(detected_court)
                if normalized_court:
                    # Store this for async check later
                    overrides["detected_court_category"] = normalized_court
                    # For now, we'll build the filter assuming it is a category
                    # The actual check will be done in run_search_approach
                    filters.append(f"(category eq '{normalized_court}' or category eq 'Civil Procedure Rules and Practice Directions' or category eq null or category eq '')")
                else:
                    # Court detected but not a category, use default
                    filters.append("(category eq 'Civil Procedure Rules and Practice Directions' or category eq null or category eq '')")
            elif not include_category:
                # No court detected and no category specified, use default
                filters.append("(category eq 'Civil Procedure Rules and Practice Directions' or category eq null or category eq '')")
        
        # Add security filters if needed
        if overrides.get("use_oid_security_filter"):
            oid = auth_claims.get("oid")
            if oid:
                filters.append(f"oids/any(g:search.in(g, '{oid}'))")
        
        if overrides.get("use_groups_security_filter"):
            groups = auth_claims.get("groups", [])
            if groups:
                group_str = ", ".join([f"'{g}'" for g in groups])
                filters.append(f"groups/any(g:search.in(g, '{group_str}'))")
        
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
        
        # Store the original query in overrides for the filter building
        overrides["original_user_query"] = original_user_query
        
        # Build filter with category logic
        search_index_filter = self.build_filter(overrides, auth_claims)
        
        # Check if we need to verify court-as-category
        if overrides.get("detected_court_category"):
            court_category = overrides["detected_court_category"]
            is_category = await self.check_if_court_is_category(court_category)
            
            if not is_category:
                # Court is not a category, rebuild filter with default
                import logging
                logging.info(f"Court '{court_category}' is not a category, using default filter")
                overrides.pop("detected_court_category", None)
                search_index_filter = self.build_filter(overrides, auth_claims)

        query_messages = self.prompt_manager.render_prompt(
            self.query_rewrite_prompt, {"user_query": original_user_query, "past_messages": messages[:-1]}
        )
        tools: list[ChatCompletionToolParam] = self.query_rewrite_tools

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question

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

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(query_text))

        # Log the search parameters for debugging
        import logging
        logging.info(f"Searching with query: {query_text}, top: {top}, filter: {search_index_filter}")
        logging.info(f"Detected court in query: {self.detect_court_in_query(original_user_query)}")

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
                import logging
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

    async def run_agentic_retrieval_approach(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
    ):
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0)
        search_index_filter = self.build_filter(overrides, auth_claims)
        top = overrides.get("top", 3)
        max_subqueries = overrides.get("max_subqueries", 10)
        results_merge_strategy = overrides.get("results_merge_strategy", "interleaved")
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
        )

        structured_sources = self.get_sources_content(results, use_semantic_captions=False, use_image_citation=False)

        extra_info = ExtraInfo(
            DataPoints(text=structured_sources),  # Pass structured data to frontend
            thoughts=[
                ThoughtStep(
                    "Use agentic retrieval",
                    messages,
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
        
        # Extract subsection from content if available
        content = doc_dict.get("content", "")
        subsection = self._extract_subsection_from_content(content)
        
        if not subsection:
            # Fallback to simple subsection extraction from sourcepage
            subsection = self._extract_simple_subsection(sourcepage)
        
        # Format page reference
        page_ref = sourcepage if sourcepage else "Unknown Page"
        
        # Ensure we have a document name
        document = sourcefile if sourcefile else "Unknown Document"
        
        # Return three-part citation format
        if subsection:
            return f"{subsection}, {page_ref}, {document}"
        else:
            return f"{page_ref}, {document}"
    
    def _extract_subsection_from_content(self, content: str) -> str:
        """Extract subsection identifier from content"""
        if not content:
            return ""
        
        # Patterns to match subsection identifiers
        patterns = [
            r'^(\d+\.\d+(?:\.\d+)?)',           # 1.1, 1.2.3, etc.
            r'^([A-Z]\d+\.\d+)',                # A4.1, D2.1, etc.
            r'^(Rule\s+\d+\.\d+)',              # Rule 1.1, etc.
            r'^(CPR\s+\d+\.\d+)',               # CPR 1.1, etc.
            r'^(Para(?:graph)?\s+\d+\.\d+)',    # Para 1.1, Paragraph 1.1
        ]
        
        # Check first few lines for subsection patterns
        lines = content.split('\n')[:5]
        for line in lines:
            line = line.strip()
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        return ""
    
    def _extract_simple_subsection(self, sourcepage: str) -> str:
        """Extract simple subsection from sourcepage"""
        if not sourcepage:
            return ""
        
        # Look for patterns like "1.1" in sourcepage
        patterns = [
            r'(\d+\.\d+)',
            r'([A-Z]\d+\.\d+)',
            r'(Rule\s+\d+\.\d+)',
            r'(Para\s+\d+\.\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sourcepage, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""

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
        from openai.types import CompletionUsage
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
        return ThoughtStep(title, messages, properties)

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
        """Return structured data for consistent processing with full content preserved and multiple subsections support"""
        
        import logging
        logging.info(f"🔍 DEBUG: ChatReadRetrieveRead processing {len(results)} documents for sources content")
        
        structured_results = []
        
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
        
        # Second pass: process groups to create structured results
        # This ensures subsections from the same document appear together
        for doc, subsections in document_groups:
            if len(subsections) > 1:
                # Split document into multiple sources, one for each subsection
                # Sort subsections by their natural order (e.g., D5.1, D5.2, D5.3)
                subsections.sort(key=lambda x: self._get_subsection_sort_key(x['subsection']))
                
                for j, subsection_info in enumerate(subsections):
                    subsection_id = subsection_info['subsection']
                    subsection_content = subsection_info['content']
                    
                    # Build three-part citation for this specific subsection
                    sourcepage = doc.sourcepage or "Unknown Page"
                    sourcefile = doc.sourcefile or "Unknown Document"
                    base_citation = f"{subsection_id}, {sourcepage}, {sourcefile}"
                    
                    # Create structured object for this subsection
                    result_obj = {
                        "id": f"{doc.id}_subsection_{j}",  # Unique ID for each subsection
                        "content": subsection_content,  # Subsection-specific content
                        "sourcepage": sourcepage,
                        "sourcefile": sourcefile,
                        "category": str(doc.category) if doc.category is not None else "",
                        "storageUrl": str(doc.storage_url) if doc.storage_url is not None else "",
                        "oids": doc.oids if doc.oids is not None else [],
                        "groups": doc.groups if doc.groups is not None else [],
                        "score": doc.score if doc.score is not None else 0.0,
                        "reranker_score": doc.reranker_score if doc.reranker_score is not None else 0.0,
                        "updated": str(doc.updated) if doc.updated is not None else "",
                        # Add citation information for frontend compatibility
                        "citation": base_citation,
                        "filepath": sourcepage,
                        "url": str(doc.storage_url) if doc.storage_url is not None else "",
                        # Add metadata to help frontend group related subsections
                        "original_doc_id": str(doc.id),
                        "subsection_index": j,
                        "total_subsections": len(subsections),
                        "is_subsection": True,
                        "subsection_id": subsection_id,  # Add for easier identification
                    }
                    
                    structured_results.append(result_obj)
                    logging.info(f"🎯 DEBUG: Added subsection source {j+1}: {subsection_id}")
            else:
                # Single subsection or no subsections - use existing logic
                result_obj = {
                    "id": str(doc.id) if doc.id is not None else "",
                    "content": str(doc.content) if doc.content is not None else "",  # Full content, no truncation
                    "sourcepage": str(doc.sourcepage) if doc.sourcepage is not None else "",
                    "sourcefile": str(doc.sourcefile) if doc.sourcefile is not None else "",
                    "category": str(doc.category) if doc.category is not None else "",
                    "storageUrl": str(doc.storage_url) if doc.storage_url is not None else "",
                    "oids": doc.oids if doc.oids is not None else [],
                    "groups": doc.groups if doc.groups is not None else [],
                    "score": doc.score if doc.score is not None else 0.0,
                    "reranker_score": doc.reranker_score if doc.reranker_score is not None else 0.0,
                    "updated": str(doc.updated) if doc.updated is not None else "",
                    # Add citation information for frontend compatibility
                    "citation": self.get_citation_from_document(doc),
                    "filepath": str(doc.sourcepage) if doc.sourcepage is not None else "",
                    "url": str(doc.storage_url) if doc.storage_url is not None else "",
                    # Add metadata for consistency
                    "original_doc_id": str(doc.id),
                    "is_subsection": False,
                }

                # Add captions as supplementary information if using semantic captions
                if use_semantic_captions and doc.captions:
                    result_obj["captions"] = [
                        {
                            "text": str(caption.text) if caption.text is not None else "",
                            "highlights": caption.highlights if caption.highlights is not None else "",
                            "additional_properties": caption.additional_properties if caption.additional_properties is not None else {},
                        }
                        for caption in doc.captions
                    ]
                    # Store caption summary separately - don't replace content
                    caption_texts = [str(c.text) for c in doc.captions if c.text is not None]
                    if caption_texts:
                        result_obj["caption_summary"] = " . ".join(caption_texts)

                structured_results.append(result_obj)

        logging.info(f"🔍 DEBUG: Returning {len(structured_results)} structured sources (original: {len(results)})")
        logging.info(f"🔍 DEBUG: Subsection grouping - documents processed: {len(document_groups)}")
        
        # Log the structure for debugging
        for i, result in enumerate(structured_results):
            if result.get("is_subsection"):
                logging.info(f"🔍 DEBUG: Result {i}: subsection {result.get('subsection_id', 'Unknown')} ({result.get('subsection_index', 0)+1}/{result.get('total_subsections', 1)}) of {result.get('original_doc_id')}")
            else:
                logging.info(f"🔍 DEBUG: Result {i}: single document {result.get('original_doc_id')}")
        
        return structured_results

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
