"""
Field mapping and intelligent filtering for custom Azure AI Search indexes
Follows Azure development best practices for environment-based configuration
"""
import os
import re
from typing import Dict, Any, List, Optional


class FieldMappingConfig:
    """Configuration class for mapping custom index fields to expected RAG application fields"""
    
    def __init__(self):
        # Use environment variables following Azure best practices
        self.content_field = os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content")
        self.embedding_field = os.getenv("AZURE_SEARCH_FIELD_NAME_EMBEDDING", "embedding")
        self.sourcepage_field = os.getenv("AZURE_SEARCH_SOURCEPAGE_FIELD", "sourcepage")
        self.sourcefile_field = os.getenv("AZURE_SEARCH_SOURCEFILE_FIELD", "sourcefile")
        self.category_field = os.getenv("AZURE_SEARCH_CATEGORY_FIELD", "category")
        self.title_field = os.getenv("AZURE_SEARCH_TITLE_FIELD", "title")
        
        # Add custom fields for legal documents
        self.url_field = os.getenv("AZURE_SEARCH_URL_FIELD", "url")
        self.document_url_field = os.getenv("AZURE_SEARCH_DOCUMENT_URL_FIELD", "document_url")
        self.section_reference_field = os.getenv("AZURE_SEARCH_SECTION_REFERENCE_FIELD", "section_reference")
        self.subsection_reference_field = os.getenv("AZURE_SEARCH_SUBSECTION_REFERENCE_FIELD", "subsection_reference")
        self.section_title_field = os.getenv("AZURE_SEARCH_SECTION_TITLE_FIELD", "section_title")
        self.court_name_field = os.getenv("AZURE_SEARCH_COURT_NAME_FIELD", "court_name")
        self.description_field = os.getenv("AZURE_SEARCH_DESCRIPTION_FIELD", "description")
        self.last_updated_field = os.getenv("AZURE_SEARCH_LAST_UPDATED_FIELD", "last_updated")
        self.reading_time_field = os.getenv("AZURE_SEARCH_READING_TIME_FIELD", "reading_time_minutes")
    
    def map_document_fields(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Map document from custom schema to expected schema with enhanced metadata"""
        # Core fields for RAG compatibility
        mapped_doc = {
            "id": doc.get("id"),
            "content": doc.get(self.content_field, "") or doc.get("chunk_content", ""),
            "embedding": doc.get(self.embedding_field, []) or doc.get("chunk_vector", []),
            "sourcepage": doc.get(self.sourcepage_field, ""),
            "sourcefile": doc.get(self.sourcefile_field, "") or doc.get(self.title_field, ""),
            "category": doc.get(self.category_field, "") or doc.get("document_type", ""),
            "title": doc.get(self.title_field, ""),
        }
        
        # Enhanced metadata for citations and filtering
        mapped_doc["metadata"] = {
            "url": doc.get(self.url_field),
            "document_url": doc.get(self.document_url_field),
            "section_reference": doc.get(self.section_reference_field),
            "subsection_reference": doc.get(self.subsection_reference_field),
            "section_title": doc.get(self.section_title_field),
            "court_name": doc.get(self.court_name_field),
            "description": doc.get(self.description_field),
            "last_updated": doc.get(self.last_updated_field),
            "reading_time_minutes": doc.get(self.reading_time_field),
        }
        
        # Filter out None values from metadata
        mapped_doc["metadata"] = {k: v for k, v in mapped_doc["metadata"].items() if v is not None}
        
        return mapped_doc
    
    def generate_citation(self, doc: Dict[str, Any], use_image_citation: bool = False) -> str:
        """Generate citation using URL and section reference fields following Azure best practices"""
        metadata = doc.get("metadata", {})
        
        # Priority 1: URL with section reference
        url = metadata.get("url") or metadata.get("document_url")
        section_ref = metadata.get("section_reference")
        
        if url and section_ref:
            return f"{url}#{section_ref}"
        elif url:
            return url
        
        # Priority 2: Title with section reference for fallback
        title = metadata.get("title") or doc.get("title")
        if title and section_ref:
            return f"{title} - Section {section_ref}"
        
        # Priority 3: Standard sourcepage fallback
        sourcepage = doc.get("sourcepage")
        if sourcepage:
            return self._get_standard_citation(sourcepage, use_image_citation)
        
        return "Unknown source"
    
    def _get_standard_citation(self, sourcepage: str, use_image_citation: bool) -> str:
        """Standard citation logic for backward compatibility"""
        if use_image_citation:
            return sourcepage
        else:
            import os
            path, ext = os.path.splitext(sourcepage)
            if ext.lower() == ".png":
                page_idx = path.rfind("-")
                if page_idx > 0:
                    page_number = int(path[page_idx + 1:])
                    return f"{path[:page_idx]}.pdf#page={page_number}"
            return sourcepage


class CourtFilteringLogic:
    """Intelligent court filtering logic for legal document searches"""
    
    def __init__(self, category_field_name: str = "category"):
        self.category_field = category_field_name
        self.court_keywords = [
            'circuit commercial court',
            'high court',
            'court of appeal',
            'supreme court',
            'magistrate court',
            'tribunal'
        ]
    
    def detect_court_in_query(self, query: str) -> bool:
        """Detect if query mentions court-related terms"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.court_keywords)
    
    def build_court_filter(self, query: str, available_courts: List[str]) -> Optional[str]:
        """
        Build intelligent court filter based on query analysis
        - If no court mentioned: show only N/A court documents
        - If specific court mentioned: filter to that court
        - If court keywords but no specific court: exclude N/A
        """
        if not self.detect_court_in_query(query):
            # No court mentioned, show only general documents
            return f"{self.category_field} eq 'N/A'"
        
        # Check for specific court names in query
        query_lower = query.lower()
        mentioned_courts = []
        for court in available_courts:
            if court.lower() != "n/a" and court.lower() in query_lower:
                mentioned_courts.append(court)
        
        if mentioned_courts:
            # Filter for specific mentioned courts
            court_filters = [f"{self.category_field} eq '{court}'" for court in mentioned_courts]
            return f"({' or '.join(court_filters)})"
        else:
            # Court keywords found but no specific court, exclude N/A
            return f"{self.category_field} ne 'N/A'"
    
    def enhance_filter_with_metadata(self, base_filter: Optional[str], metadata_filters: Dict[str, str]) -> Optional[str]:
        """Enhance existing filters with metadata-based filtering"""
        filters = []
        
        if base_filter:
            filters.append(base_filter)
        
        # Add section-based filtering if needed
        if metadata_filters.get("section_reference"):
            filters.append(f"section_reference eq '{metadata_filters['section_reference']}'")
        
        # Add court-specific filtering
        if metadata_filters.get("court_name"):
            filters.append(f"court_name eq '{metadata_filters['court_name']}'")
        
        # Add document type filtering
        if metadata_filters.get("document_type"):
            filters.append(f"document_type eq '{metadata_filters['document_type']}'")
        
        return " and ".join(filters) if filters else None


# Global instances for easy access
field_config = FieldMappingConfig()
court_filter = CourtFilteringLogic(field_config.category_field)
