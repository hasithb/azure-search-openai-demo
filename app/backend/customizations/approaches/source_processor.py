"""
Legal Domain Source Processor

This module provides structured source content processing for legal documents.
It handles subsection splitting, document metadata enrichment, and structured
output formatting for the frontend.

Usage:
    from customizations.approaches import SourceProcessor
    
    processor = SourceProcessor()
    structured_sources = processor.process_documents(documents, use_semantic_captions=False)
"""

import hashlib
import logging
import os
import re
from typing import Any, Optional

from .citation_builder import CitationBuilder


def source_revision() -> str:
    return str(os.environ.get("V4_SEARCH_SNAPSHOT_SHA256") or os.environ.get("GIT_SHA") or "local").strip()


def canonical_text_sha256(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SourceProcessor:
    """
    Processes search results into structured source content for the frontend.
    
    Handles:
    - Subsection extraction and splitting
    - Metadata enrichment (storageUrl, category, updated, etc.)
    - Structured output formatting
    - Citation generation using CitationBuilder
    """

    def __init__(self, citation_builder: Optional[CitationBuilder] = None):
        """
        Initialize the source processor.
        
        Args:
            citation_builder: CitationBuilder instance for citation generation.
                            If None, creates a new instance.
        """
        self.citation_builder = citation_builder or CitationBuilder()

    def process_documents(
        self,
        documents: list[Any],
        use_semantic_captions: bool = False,
        use_image_citation: bool = False,
        focus_on_indexed_subsection: bool = False,
        adjacent_subsections: int = 0,
        max_unfocused_subsections: Optional[int] = None,
        query_hint: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Process documents into structured source content.
        
        This is the main entry point that replaces the custom `get_sources_content`
        method in ChatReadRetrieveReadApproach.
        
        Args:
            documents: List of Document objects from search results
            use_semantic_captions: Whether to use semantic captions
            use_image_citation: Whether to use image citations
            focus_on_indexed_subsection: When true, narrow multi-subsection expansion
                to the retrieved subsection and nearby subsections when possible.
            adjacent_subsections: Number of neighboring subsections to include on each
                side of a focused subsection.
            max_unfocused_subsections: When focusing is enabled but no subsection anchor
                can be resolved, cap the number of expanded subsections per document.
            query_hint: Optional query text used to extract subsection references
                (e.g., "CPR 31.16") when the indexed subsection_id does not match.
            
        Returns:
            List of structured source dictionaries for frontend consumption
        """
        logging.debug(f"SourceProcessor processing {len(documents)} documents")
        
        structured_results = []
        
        # First pass: process each document and collect subsections
        document_groups = []  # List of (original_doc, [subsections]) tuples
        
        for i, doc in enumerate(documents):
            doc_id = getattr(doc, 'id', 'unknown')
            doc_sourcepage = getattr(doc, 'sourcepage', '')
            doc_sourcefile = getattr(doc, 'sourcefile', '')
            doc_content_preview = (getattr(doc, 'content', '') or '')[:100]
            logging.debug(f"SourceProcessor doc {i+1}: id='{doc_id}', sourcepage='{doc_sourcepage}', "
                         f"sourcefile='{doc_sourcefile}', content_preview='{doc_content_preview}...'")
            
            # Check if document contains multiple subsections
            subsections = self.citation_builder.extract_multiple_subsections(doc)

            if len(subsections) > 1 and focus_on_indexed_subsection:
                subsections = self._focus_subsections(doc, subsections, adjacent_subsections, query_hint=query_hint)
                if max_unfocused_subsections is not None and len(subsections) > max_unfocused_subsections:
                    subsections = subsections[:max_unfocused_subsections]
            
            if len(subsections) > 1:
                logging.debug(f"Document {getattr(doc, 'id', 'unknown')} will be split into {len(subsections)} sources")
                document_groups.append((doc, subsections))
            else:
                logging.debug(f"Document {getattr(doc, 'id', 'unknown')} not split - using original logic")
                document_groups.append((doc, []))
        
        # Second pass: process groups to create structured results
        for doc, subsections in document_groups:
            if len(subsections) > 1:
                # Split document into multiple sources
                self._process_multi_subsection_document(doc, subsections, structured_results, use_semantic_captions)
            else:
                # Single document
                self._process_single_document(doc, structured_results, use_semantic_captions)
        
        logging.debug(f"Returning {len(structured_results)} structured sources (original: {len(documents)})")
        return structured_results

    def _focus_subsections(
        self,
        doc: Any,
        subsections: list[dict[str, str]],
        adjacent_subsections: int,
        query_hint: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Narrow subsection expansion around the indexed subsection when available.

        Falls back to extracting subsection references from *query_hint*
        when the indexed ``subsection_id`` does not match any extracted
        subsection header.
        """

        focus_subsection = self.citation_builder.extract_subsection(doc)

        # Try the indexed subsection_id first
        focus_index = self._find_subsection_index(focus_subsection, subsections) if focus_subsection else None

        # Fallback: derive a subsection reference from the query
        if focus_index is None and query_hint:
            query_focus = self._extract_subsection_from_query(query_hint, doc)
            if query_focus:
                focus_index = self._find_subsection_index(query_focus, subsections)

        if focus_index is None:
            return subsections

        window = max(adjacent_subsections, 0)
        start = max(0, focus_index - window)
        end = min(len(subsections), focus_index + window + 1)
        return subsections[start:end]

    def _find_subsection_index(
        self, focus_key: str, subsections: list[dict[str, str]]
    ) -> Optional[int]:
        """Return the index of *focus_key* within *subsections*, or ``None``."""
        normalized_focus = self._normalize_subsection_key(focus_key)
        for index, subsection in enumerate(subsections):
            if self._normalize_subsection_key(subsection.get("subsection", "")) == normalized_focus:
                return index
        return None

    def _extract_subsection_from_query(
        self, query: str, doc: Any
    ) -> Optional[str]:
        """Extract a subsection reference from *query* that belongs to *doc*.

        For example, if *doc* is Part 31 and *query* mentions "31.16",
        return "31.16" so the focus logic can window around it.
        """
        sourcefile = str(getattr(doc, "sourcefile", "") or "")
        sourcepage = str(getattr(doc, "sourcepage", "") or "")

        # Identify the Part number from the document metadata
        part_match = re.search(r"Part\s+(\d+)", sourcefile + " " + sourcepage, re.IGNORECASE)
        if not part_match:
            return None
        part_number = part_match.group(1)

        # Look for "<part_number>.<rule>" in the query (e.g., "31.16")
        for m in re.finditer(rf"\b{re.escape(part_number)}\.(\d+[A-Za-z]?)\b", query):
            return f"{part_number}.{m.group(1)}"

        return None

    def _normalize_subsection_key(self, subsection_id: str) -> str:
        """Normalize subsection identifiers so indexed metadata matches split labels."""

        normalized = subsection_id.strip().upper()
        normalized = re.sub(r"^(RULE|CPR|PARA)\s+", "", normalized)
        return normalized

    def _process_multi_subsection_document(
        self,
        doc: Any,
        subsections: list[dict[str, str]],
        results: list[dict[str, Any]],
        use_semantic_captions: bool,
    ) -> None:
        """
        Process a document with multiple subsections into separate sources.
        
        Args:
            doc: The original Document object
            subsections: List of subsection dicts with 'subsection' and 'content' keys
            results: Output list to append structured results to
            use_semantic_captions: Whether to include semantic captions
        """
        # Sort subsections by natural order
        subsections.sort(key=lambda x: self.citation_builder.get_subsection_sort_key(x['subsection']))
        subsections = self._deduplicate_subsections(subsections)
        
        for j, subsection_info in enumerate(subsections):
            subsection_id = subsection_info['subsection']
            subsection_content = subsection_info['content']
            
            # Build three-part citation for this specific subsection
            sourcepage = getattr(doc, 'sourcepage', '') or "Unknown Page"
            sourcefile = getattr(doc, 'sourcefile', '') or "Unknown Document"
            base_citation = f"{subsection_id}, {sourcepage}, {sourcefile}"
            
            # Create structured object for this subsection
            # Include the full original document content so the frontend can
            # display the complete section and highlight the cited subsection.
            full_doc_content = str(getattr(doc, 'content', '') or '')
            result_obj = {
                "id": f"{getattr(doc, 'id', '')}_{j}",  # Unique ID
                "content": subsection_content,
                "full_content": full_doc_content,
                "sourcepage": sourcepage,
                "sourcefile": sourcefile,
                "category": str(getattr(doc, 'category', '')) if getattr(doc, 'category', None) else "",
                "storageUrl": str(getattr(doc, 'storage_url', '')) if getattr(doc, 'storage_url', None) else "",
                "oids": getattr(doc, 'oids', []) or [],
                "groups": getattr(doc, 'groups', []) or [],
                "score": getattr(doc, 'score', 0.0) or 0.0,
                "reranker_score": getattr(doc, 'reranker_score', 0.0) or 0.0,
                "updated": str(getattr(doc, 'updated', '')) if getattr(doc, 'updated', None) else "",
                # Citation info for frontend
                "citation": base_citation,
                "filepath": sourcepage,
                "url": str(getattr(doc, 'storage_url', '')) if getattr(doc, 'storage_url', None) else "",
                # Subsection metadata
                "original_doc_id": str(getattr(doc, 'id', '')),
                "subsection_index": j,
                "total_subsections": len(subsections),
                "is_subsection": True,
                "subsection_id": subsection_id,
                "source_revision": source_revision(),
                "document_id": str(getattr(doc, "id", "")),
                "source_id": str(getattr(doc, "sourcefile", "") or getattr(doc, "id", "")),
                "canonical_text_sha256": canonical_text_sha256(subsection_content),
            }
            
            results.append(result_obj)
            logging.debug(f"Added subsection source {j+1}: {subsection_id}")

    def _deduplicate_subsections(self, subsections: list[dict[str, str]]) -> list[dict[str, str]]:
        """Collapse repeated subsection labels from the same document into a single best chunk."""

        deduplicated: list[dict[str, str]] = []
        positions_by_key: dict[str, int] = {}

        for subsection in subsections:
            subsection_id = subsection.get("subsection", "")
            normalized_key = self._normalize_subsection_key(subsection_id)
            if normalized_key in positions_by_key:
                existing_index = positions_by_key[normalized_key]
                existing = deduplicated[existing_index]
                if len(subsection.get("content", "")) > len(existing.get("content", "")):
                    deduplicated[existing_index] = subsection
                continue

            positions_by_key[normalized_key] = len(deduplicated)
            deduplicated.append(subsection)

        return deduplicated

    def _process_single_document(
        self,
        doc: Any,
        results: list[dict[str, Any]],
        use_semantic_captions: bool,
    ) -> None:
        """
        Process a single document (no subsection splitting).
        
        Args:
            doc: The Document object
            results: Output list to append structured result to
            use_semantic_captions: Whether to include semantic captions
        """
        doc_id = str(getattr(doc, 'id', '')) if getattr(doc, 'id', None) else ""
        content = str(getattr(doc, 'content', '')) if getattr(doc, 'content', None) else ""
        sourcepage = str(getattr(doc, 'sourcepage', '')) if getattr(doc, 'sourcepage', None) else ""
        sourcefile = str(getattr(doc, 'sourcefile', '')) if getattr(doc, 'sourcefile', None) else ""
        category = str(getattr(doc, 'category', '')) if getattr(doc, 'category', None) else ""
        storage_url = str(getattr(doc, 'storage_url', '')) if getattr(doc, 'storage_url', None) else ""
        updated = str(getattr(doc, 'updated', '')) if getattr(doc, 'updated', None) else ""
        
        logging.debug(f"_process_single_document - id='{doc_id}', sourcepage='{sourcepage}', sourcefile='{sourcefile}'")
        
        # Generate citation
        citation = self.citation_builder.build_enhanced_citation(doc, len(results) + 1)
        
        result_obj = {
            "id": doc_id,
            "content": content,
            "full_content": content,
            "sourcepage": sourcepage,
            "sourcefile": sourcefile,
            "category": category,
            "storageUrl": storage_url,
            "oids": getattr(doc, 'oids', []) or [],
            "groups": getattr(doc, 'groups', []) or [],
            "score": getattr(doc, 'score', 0.0) or 0.0,
            "reranker_score": getattr(doc, 'reranker_score', 0.0) or 0.0,
            "updated": updated,
            # Citation info for frontend
            "citation": citation,
            "filepath": sourcepage,
            "url": storage_url,
            # Metadata for consistency
            "original_doc_id": doc_id,
            "is_subsection": False,
            "source_revision": source_revision(),
            "document_id": doc_id,
            "source_id": sourcefile or doc_id,
            "canonical_text_sha256": canonical_text_sha256(content),
        }
        
        # Add captions if using semantic captions
        if use_semantic_captions:
            captions = getattr(doc, 'captions', None)
            if captions:
                result_obj["captions"] = [
                    {
                        "text": str(getattr(caption, 'text', '')) if getattr(caption, 'text', None) else "",
                        "highlights": getattr(caption, 'highlights', '') or "",
                        "additional_properties": getattr(caption, 'additional_properties', {}) or {},
                    }
                    for caption in captions
                ]
                # Store caption summary separately
                caption_texts = [str(getattr(c, 'text', '')) for c in captions if getattr(c, 'text', None)]
                if caption_texts:
                    result_obj["caption_summary"] = " . ".join(caption_texts)
        
        results.append(result_obj)

    def enrich_source_metadata(
        self,
        source: dict[str, Any],
        search_result: Any,
    ) -> None:
        """
        Enrich a source dict with metadata from search results.
        
        Handles various field name conventions and ensures all required
        fields are populated.
        
        Args:
            source: The source dict to enrich
            search_result: The original search result object
        """
        # Map common field variations
        source.setdefault("sourcepage", source.get("source_page", ""))
        source.setdefault("sourcefile", source.get("source_file", ""))
        source.setdefault("category", "")
        source.setdefault("updated", source.get("last_updated", source.get("date_updated", "")))
        source.setdefault("storageurl", source.get("storage_url", source.get("url", "")))
        source.setdefault("url", source.get("storageurl", source.get("storage_url", "")))
        
        # Get fields directly from search result
        if hasattr(search_result, 'get'):
            # Dict-like object
            source["sourcepage"] = search_result.get("sourcepage", source.get("sourcepage", ""))
            source["sourcefile"] = search_result.get("sourcefile", source.get("sourcefile", ""))
            source["category"] = search_result.get("category", source.get("category", ""))
            source["updated"] = search_result.get("updated", search_result.get("last_updated", source.get("updated", "")))
            source["storageurl"] = search_result.get("storageurl", search_result.get("storage_url", source.get("storageurl", "")))
        else:
            # Object with attributes
            source["sourcepage"] = getattr(search_result, "sourcepage", source.get("sourcepage", ""))
            source["sourcefile"] = getattr(search_result, "sourcefile", source.get("sourcefile", ""))
            source["category"] = getattr(search_result, "category", source.get("category", ""))
            source["updated"] = getattr(search_result, "updated", getattr(search_result, "last_updated", source.get("updated", "")))
            source["storageurl"] = getattr(search_result, "storageurl", getattr(search_result, "storage_url", source.get("storageurl", "")))
        
        # Ensure url field matches storageurl
        if source.get("storageurl") and not source.get("url"):
            source["url"] = source["storageurl"]


# Singleton instance for convenience
source_processor = SourceProcessor()
