"""
Legal Domain Citation Builder

This module provides citation building functionality for legal documents.
It extracts subsections, builds enhanced citations, and handles various
legal document formats (CPR rules, Practice Directions, court guides, etc.)

Usage:
    from customizations.approaches import CitationBuilder
    
    builder = CitationBuilder()
    citation = builder.build_enhanced_citation(doc, source_index=1)
"""

import logging
import re
from typing import Any, Optional

from customizations.subsection_extractor import SubsectionExtractor


class CitationBuilder:
    """
    Builds enhanced citations for legal documents with proper subsection extraction.
    
    Handles citation formats like:
    - Three-part: "1.1, D5 - Filing deadlines (p. 210), Commercial Court Guide"
    - Two-part: "1.1, Commercial Court Guide" 
    - Single: "Commercial Court Guide"
    
    Extracts subsections from:
    - Content text (e.g., "1.1 Filing requirements...")
    - Encoded sourcepage (e.g., "PD3E-1.1" -> "1.1")
    - Direct patterns (e.g., "Rule 31.1", "Para 5.2", "A4.1")
    """

    # Content subsection patterns are defined in SubsectionExtractor (single source of truth)
    CONTENT_SUBSECTION_PATTERNS = SubsectionExtractor.CONTENT_SUBSECTION_PATTERNS

    # Pattern for extracting subsection from encoded sourcepage (e.g., "PD3E-1.1")
    ENCODED_SUBSECTION_PATTERNS = [
        r'[A-Z]+\d*[A-Z]*-([A-Z]\d+\.\d+)',  # PD3E-A4.1 -> A4.1
        r'[A-Z]+\d*[A-Z]*-(\d+\.\d+)',       # PD3E-1.1 -> 1.1
        r'[A-Z]+\d*[A-Z]*-([A-Z]\d+)',       # PD3E-A4 -> A4
    ]

    # Pattern for direct subsection in sourcepage
    DIRECT_SUBSECTION_PATTERNS = [
        r'^([A-Z]\d+\.\d+)\b',           # A4.1, B2.3, etc.
        r'^([A-Z]\.\d+(?:\.\d+)?)\b',   # C.2, F.1, A.1.1, etc.
        r'^(\d+\.\d+)\b',                # 1.1, 2.3, etc.
        r'^([A-Z]\d+)\b',                # A1, B2, etc.
        r'^(Rule \d+(?:\.\d+)?)\b',      # Rule 1, Rule 1.1
        r'^(Part \d+)\b',                # Part 1, Part 2, etc.
    ]

    # Pattern for multiple subsection detection
    MULTI_SUBSECTION_PATTERN = re.compile(
        r'^(?:'
        r'(?P<rule>Rule\s+\d+(?:\.\d+)?)|'
        r'(?P<cpr>CPR\s+\d+(?:\.\d+)?)|'
        r'(?P<para>Para(?:graph)?\s+\d+(?:\.\d+)?)|'
        r'(?P<alpha_dot_num>[A-Z]\.\d+(?:\.\d+)?)|'
        r'(?P<alpha_num_dotted>[A-Z]\d+\.\d+)|'
        r'(?P<num_dotted>\d+\.\d+)|'
        r'(?P<alpha_num>[A-Z]\d+)'
        r')\b',
        re.IGNORECASE,
    )

    def __init__(self):
        self.citation_map: dict[str, str] = {}

    def build_enhanced_citation(self, doc: Any, source_index: int) -> str:
        """
        Build enhanced citation from document with improved logic for proper formatting.
        
        Args:
            doc: Document object with sourcepage, sourcefile, and content attributes
            source_index: Index for fallback citation ("Source 1", "Source 2", etc.)
            
        Returns:
            Formatted citation string
        """
        sourcepage = getattr(doc, 'sourcepage', '') or ''
        sourcefile = getattr(doc, 'sourcefile', '') or ''
        
        # Clean up
        sourcepage = sourcepage.strip()
        sourcefile = sourcefile.strip()
        
        # Extract subsection from content
        subsection = self.extract_subsection(doc)
        
        # Avoid duplication between subsection and sourcepage
        final_sourcepage = sourcepage
        
        if subsection and sourcepage:
            if sourcepage == subsection:
                final_sourcepage = ""
            elif re.search(r'[A-Z]+\d*[A-Z]*-' + re.escape(subsection), sourcepage):
                final_sourcepage = ""
            elif re.match(r'^[A-Z]?\d+(?:\.\d+)?$', subsection) and len(sourcepage) > len(subsection):
                # Keep both for three-part format
                pass
        
        # Build citation
        if subsection and final_sourcepage and sourcefile:
            citation = f"{subsection}, {final_sourcepage}, {sourcefile}"
        elif subsection and sourcefile:
            citation = f"{subsection}, {sourcefile}"
        elif final_sourcepage and sourcefile:
            citation = f"{final_sourcepage}, {sourcefile}"
        elif sourcefile:
            citation = sourcefile
        else:
            citation = f"Source {source_index}"
        
        logging.debug(
            f"Built citation: {citation} from doc with "
            f"sourcepage='{sourcepage}', sourcefile='{sourcefile}', subsection='{subsection}'"
        )
        return citation

    def extract_subsection(self, doc: Any) -> str:
        """
        Extract subsection identifier from document content or sourcepage.
        
        Priority:
        0. Indexed subsection_id field (preferred)
        1. Content text (first 30 lines)
        2. Encoded sourcepage (e.g., "PD3E-1.1")
        3. Direct sourcepage patterns
        
        Handles markdown formatting like:
        - # 1.1 Heading (markdown headings)
        - **1.1** Bold text
        - __1.1__ Underline bold
        
        Returns:
            Subsection string (e.g., "1.1", "A4.1", "Rule 31.1") or empty string
        """
        # Priority 0: Use indexed subsection_id if available
        indexed_subsection = getattr(doc, 'subsection_id', None)
        if isinstance(indexed_subsection, str):
            indexed_subsection = indexed_subsection.strip()
            if indexed_subsection:
                return indexed_subsection
        
        content = getattr(doc, 'content', '') or ''
        sourcepage = getattr(doc, 'sourcepage', '') or ''
        
        # Priority 1: Check content (fallback for backward compatibility)
        if content:
            lines = content.split('\n')[:20]
            deferred_part_heading = ""
            for line in lines:
                line = line.strip()
                if not line or line == "---":
                    continue
                
                # Strip markdown formatting before pattern matching
                # 1. Remove markdown heading prefixes (# ## ### etc.)
                cleaned_line = re.sub(r'^#+\s*', '', line)
                # 2. Remove bold markers (**text** or __text__)
                cleaned_line = re.sub(r'^\*\*([^*]+)\*\*', r'\1', cleaned_line)
                cleaned_line = re.sub(r'^__([^_]+)__', r'\1', cleaned_line)
                # 3. Strip any remaining leading/trailing whitespace
                cleaned_line = cleaned_line.strip()
                
                # Try matching with cleaned line first
                for pattern in self.CONTENT_SUBSECTION_PATTERNS:
                    match = re.match(pattern, cleaned_line, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip()
                        if re.match(r'^Part\s+\d+$', candidate, re.IGNORECASE):
                            deferred_part_heading = deferred_part_heading or candidate
                            break
                        return candidate
                
                # Also try original line in case cleaning removed something important
                if cleaned_line != line:
                    for pattern in self.CONTENT_SUBSECTION_PATTERNS:
                        match = re.match(pattern, line, re.IGNORECASE)
                        if match:
                            candidate = match.group(1).strip()
                            if re.match(r'^Part\s+\d+$', candidate, re.IGNORECASE):
                                deferred_part_heading = deferred_part_heading or candidate
                                break
                            return candidate
                
                if re.match(r'^\d+\.\d+$', cleaned_line):
                    return cleaned_line

            if deferred_part_heading:
                return deferred_part_heading
        
        # Priority 2: Encoded sourcepage
        if sourcepage:
            for pattern in self.ENCODED_SUBSECTION_PATTERNS:
                match = re.search(pattern, sourcepage)
                if match:
                    return match.group(1)
        
        # Priority 3: Direct sourcepage patterns
        if sourcepage:
            for pattern in self.DIRECT_SUBSECTION_PATTERNS:
                match = re.match(pattern, sourcepage, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        return ""

    @staticmethod
    def extract_page_number(sourcepage: str) -> int | None:
        """
        Extract a page number from various sourcepage formats.
        
        Handles:
        - PDF ingested docs: "file.pdf#page=5" -> 5
        - Legal docs with embedded page: "E. Disclosure, E.1 Generally (p. 60)" -> 60
        - Plain page references: "(p. 123)" -> 123
        
        Returns None if no page number found.
        """
        if not sourcepage:
            return None
        
        # PDF ingested format: file.pdf#page=5
        page_hash_match = re.search(r'#page=(\d+)', sourcepage)
        if page_hash_match:
            return int(page_hash_match.group(1))
        
        # Legal format: (p. 60) or (p.60) or (p 60)
        page_paren_match = re.search(r'\(p\.?\s*(\d+)\)', sourcepage)
        if page_paren_match:
            return int(page_paren_match.group(1))
        
        return None

    def extract_multiple_subsections(self, doc: Any) -> list[dict[str, str]]:
        """
        Split document content into multiple subsection chunks.
        
        Used when a document contains multiple distinct subsections that
        should be cited separately.
        
        Returns:
            List of dicts with 'subsection' and 'content' keys, or empty list
        """
        content = getattr(doc, 'content', '') or ''
        if not content:
            return []

        lines = content.splitlines()
        headings: list[tuple[str, int]] = []
        
        for i, raw in enumerate(lines):
            line = raw.strip()
            if not line or line == "---":
                continue
            
            # Clean formatting for pattern matching (Markdown & Breadcrumbs)
            clean_line = re.sub(r'^#+\s*', '', line)  # Strip Markdown headers
            clean_line = re.sub(r'^\[.*?\]\s*', '', clean_line)  # Strip Breadcrumbs
            
            m = self.MULTI_SUBSECTION_PATTERN.match(clean_line)
            if not m:
                continue
            label = m.group(0)
            
            # Normalize labels
            if m.lastgroup == "rule":
                num = re.search(r'\d+(?:\.\d+)?', label)
                label = f"Rule {num.group(0)}" if num else label
            elif m.lastgroup == "cpr":
                num = re.search(r'\d+(?:\.\d+)?', label, re.IGNORECASE)
                label = f"CPR {num.group(0)}" if num else label
            elif m.lastgroup and m.lastgroup.startswith("para"):
                num = re.search(r'\d+(?:\.\d+)?', label, re.IGNORECASE)
                label = f"Para {num.group(0)}" if num else label
            else:
                label = label.upper()
            headings.append((label, i))

        if len(headings) <= 1:
            return []

        subsections: list[dict[str, str]] = []
        for idx, (label, start_line) in enumerate(headings):
            end_line = headings[idx + 1][1] if idx + 1 < len(headings) else len(lines)
            chunk_lines = lines[start_line:end_line]
            chunk_content = "\n".join(chunk_lines).strip()
            if not chunk_content or len(chunk_content) < 10:
                continue
            subsections.append({"subsection": label, "content": chunk_content})

        return subsections

    def extract_simple_subsection(self, sourcepage: str) -> str:
        """
        Extract subsection from sourcepage string directly.
        
        Used for simple cases where only sourcepage is available.
        """
        if not sourcepage:
            return ""
            
        for pattern in self.DIRECT_SUBSECTION_PATTERNS:
            match = re.match(pattern, sourcepage, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def extract_subsection_from_content(self, content: str) -> str:
        """
        Extract subsection from raw content text.
        
        Used when processing content outside of a Document object.
        Delegates to SubsectionExtractor for consistent behavior.
        """
        return SubsectionExtractor.extract_first_subsection(content)

    def get_subsection_sort_key(self, subsection_id: str) -> tuple:
        """
        Generate a sort key for ordering subsections naturally.
        
        All returned tuples have a consistent structure: (type_order: int, prefix: str, major: int, minor: int)
        This ensures Python can compare all tuples without type errors.
        
        Type order:
            0 = numeric only (1.1)
            1 = alphanumeric (A4.1)
            2 = rule format (Rule 31.1)
            3 = fallback
        
        Examples:
            "1.1" -> (0, '', 1, 1)
            "A4.1" -> (1, 'A', 4, 1)
            "Rule 31.1" -> (2, 'RULE', 31, 1)
        """
        subsection_id = subsection_id.strip()
        
        # Handle "Rule X.Y" format
        rule_match = re.match(r'Rule\s+(\d+)(?:\.(\d+))?', subsection_id, re.IGNORECASE)
        if rule_match:
            major = int(rule_match.group(1))
            minor = int(rule_match.group(2)) if rule_match.group(2) else 0
            return (2, 'RULE', major, minor)
        
        # Handle "A4.1" format (letter + number.number)
        alpha_match = re.match(r'([A-Za-z]+)(\d+)(?:\.(\d+))?', subsection_id)
        if alpha_match:
            prefix = alpha_match.group(1).upper()
            major = int(alpha_match.group(2))
            minor = int(alpha_match.group(3)) if alpha_match.group(3) else 0
            return (1, prefix, major, minor)
        
        # Handle "1.1" format (number.number)
        num_match = re.match(r'(\d+)(?:\.(\d+))?', subsection_id)
        if num_match:
            major = int(num_match.group(1))
            minor = int(num_match.group(2)) if num_match.group(2) else 0
            return (0, '', major, minor)
        
        # Fallback - use string sorting within fallback category
        return (3, subsection_id, 0, 0)


# Singleton instance for convenience
citation_builder = CitationBuilder()
