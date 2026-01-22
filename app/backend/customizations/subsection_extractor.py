"""
Subsection Extraction Utility

Shared logic for extracting subsection identifiers from legal document content.
Used by both the indexing pipeline and runtime citation builder.

CUSTOM: This is part of the merge-safe customizations for legal RAG.
"""

import re
from typing import Optional


class SubsectionExtractor:
    """
    Extracts subsection identifiers from legal document content.
    
    Handles various formats:
    - Markdown headings: ## 35.1
    - Breadcrumbs: [PART 35 > 35.1]
    - Bare text: 35.1
    - Legal patterns: Rule 35.1, Para 5.2, Part 35
    """
    
    # Pattern for detecting subsection markers in content
    CONTENT_SUBSECTION_PATTERNS = [
        r'^(Practice\s+Direction\s+\d+[A-Z]{0,3}(?:\s+[A-Z]{1,3})*(?:\s+\d+)?)\b',
        r'^(Part\s+\d+[A-Z]?)\b',
        r'^(Annex\s+\d+[A-Z]?)\b',
        r'^(Appendix\s+(?:[A-Z]|\d+))\b',
        r'^(p\.\s*\d+)\b',
        r'^([A-Z]\.\d+(?:\.\d+)?)\b',    # A.1, B.7, C.2.3, etc.
        r'^([A-Z]\d+\.\d+(?:\.\d+)?)\b', # A4.1, B2.3, B7.1 etc.
        r'^(\d+\.\d+(?:\.\d+)?[A-Z]?)\b', # 1.1, 2.3, 3.1A, 1.2.3 etc.
        r'^([A-Z]\d+)\b',                # A1, B2, etc.
        r'^(Rule \d+(?:\.\d+)?)\b',      # Rule 1, Rule 1.1
        r'^(Para \d+(?:\.\d+)?)\b',      # Para 1.1
    ]
    
    # Pattern for extracting from breadcrumbs [PART X > subsection] or [Court Guides > ... > B.7]
    # Captures the subsection after the last '>' but before ']'
    BREADCRUMB_PATTERN = re.compile(
        r'\[[^\]]*>\s*([A-Z]\.?\d+(?:\.\d+)?[A-Z]?|[A-Z]\d+(?:\.\d+)?[A-Z]?|\d+(?:\.\d+)?[A-Z]?)\s*(?:\]|>)',
        re.IGNORECASE
    )
    
    # Pattern for subsections appearing AFTER breadcrumb closing bracket: ] 2.1 or ] B.7
    BREADCRUMB_SUFFIX_PATTERN = re.compile(
        r'\]\s+([A-Z]\.?\d+(?:\.\d+)?[A-Z]?|[A-Z]\d+(?:\.\d+)?[A-Z]?|\d+(?:\.\d+)?[A-Z]?)\s',
        re.IGNORECASE
    )
    
    # Pattern for markdown headings - handle both numeric and letter-number formats
    # Matches ## 35.1, ## 2.5 Full text, ## B.7 Title
    MARKDOWN_HEADING_PATTERN = re.compile(
        r'^#+\s*([A-Z]\.?\d+(?:\.\d+)?[A-Z]?|[A-Z]\d+(?:\.\d+)?[A-Z]?|\d+(?:\.\d+)?[A-Z]?)\b',
        re.MULTILINE | re.IGNORECASE
    )
    
    # Pattern for subsection tokens anywhere in content
    TOKEN_PATTERN = re.compile(
        r"\b("
        r"Practice\s+Direction\s+\d+[A-Z]{0,3}(?:\s+[A-Z]{1,3})*(?:\s+\d+)?"  # Practice Direction 3D, 3 D, 51 ZG 3
        r"|Part\s+\d+[A-Z]?"  # Part 50, Part 52B
        r"|Annex\s+\d+[A-Z]?"  # Annex 5
        r"|Appendix\s+(?:[A-Z]|\d+)"  # Appendix B, Appendix 1
        r"|p\.\s*\d+"  # p. 1
        r"|[A-Z]\.\d+(?:\.\d+)?[A-Z]?"  # A.1, B.7, C.2.3, 5.1A
        r"|[A-Z]\d+\.\d+(?:\.\d+)?[A-Z]?"  # A4.1, B2.3, A1.1A
        r"|\d+\.\d+(?:\.\d+)?[A-Z]?"  # 1.1, 2.3, 1.2.3, 7.3A
        r"|Rule\s+\d+(?:\.\d+)?"  # Rule 3.1
        r"|Para\s+\d+(?:\.\d+)?"  # Para 5.2
        r")\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_valid_subsection(subsection: str) -> bool:
        """
        Validate that a string is a legal subsection identifier.
        
        Filters out false positives like "Part 1", "PART 44", etc.
        
        Args:
            subsection: Candidate subsection string
            
        Returns:
            True if valid subsection, False otherwise
        """
        if not subsection:
            return False
        
        # Accept valid patterns
        valid_patterns = [
            r'^Practice\s+Direction\s+\d+[A-Z]{0,3}(?:\s+[A-Z]{1,3})*(?:\s+\d+)?$',
            r'^Part\s+\d+[A-Z]?$',
            r'^Annex\s+\d+[A-Z]?$',
            r'^Appendix\s+(?:[A-Z]|\d+)$',
            r'^p\.\s*\d+$',
            r'^[A-Z]\.?\d+(?:\.\d+)?$',  # A.1, B7, A4.1, etc.
            r'^\d+(?:\.\d+)+$',           # 1.1, 35.1, 2.3.4, etc.
            r'^Rule\s+\d+(?:\.\d+)?$',    # Rule 1.1
            r'^Para\s+\d+(?:\.\d+)?$',    # Para 1.1
        ]
        
        for pattern in valid_patterns:
            if re.match(pattern, subsection, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def clean_line(line: str) -> str:
        """
        Remove markdown and formatting from a line before pattern matching.
        
        Args:
            line: Raw line from content
            
        Returns:
            Cleaned line with markdown stripped
        """
        # Remove markdown heading prefixes (# ## ### etc.)
        cleaned = line.replace("\u00a0", " ")
        cleaned = re.sub(r'^#+\s*', '', cleaned.strip())
        cleaned = re.sub(r'^(SOURCEPAGE|SOURCE|CATEGORY)\s*:\s*', '', cleaned, flags=re.IGNORECASE)
        # Remove bold markers (**text** or __text__)
        cleaned = re.sub(r'^\*\*([^*]+)\*\*', r'\1', cleaned)
        cleaned = re.sub(r'^__([^_]+)__', r'\1', cleaned)
        return cleaned.strip()
    
    @staticmethod
    def extract_first_subsection(content: str, max_lines: int = 30) -> str:
        """
        Extract the first/primary subsection identifier from content.
        
        This is used for the subsection_id field (citation label).
        
        Args:
            content: Document content text
            max_lines: Maximum number of lines to scan (for chunked documents)
            
        Returns:
            Subsection identifier (e.g., "35.1", "Rule 3.4") or empty string
        """
        if not content:
            return ""
        
        lines = content.split('\n')[:max_lines]
        
        for line in lines:
            if not line.strip() or line.strip() == "---":
                continue
            
            # Try breadcrumb extraction first (most reliable for v2 format)
            breadcrumb_match = SubsectionExtractor.BREADCRUMB_PATTERN.search(line)
            if breadcrumb_match:
                subsection = breadcrumb_match.group(1)
                if SubsectionExtractor._is_valid_subsection(subsection):
                    return subsection
            
            # Try breadcrumb suffix pattern (] 2.1 format)
            suffix_match = SubsectionExtractor.BREADCRUMB_SUFFIX_PATTERN.search(line)
            if suffix_match:
                subsection = suffix_match.group(1)
                if SubsectionExtractor._is_valid_subsection(subsection):
                    return subsection
            
            # Try markdown heading extraction
            md_match = SubsectionExtractor.MARKDOWN_HEADING_PATTERN.match(line)
            if md_match:
                subsection = md_match.group(1)
                if SubsectionExtractor._is_valid_subsection(subsection):
                    return subsection
            
            # Try pattern matching on cleaned line
            cleaned_line = SubsectionExtractor.clean_line(line)
            if not cleaned_line:
                continue
                
            for pattern in SubsectionExtractor.CONTENT_SUBSECTION_PATTERNS:
                match = re.match(pattern, cleaned_line, re.IGNORECASE)
                if match:
                    return match.group(1)

        token_match = SubsectionExtractor.TOKEN_PATTERN.search(content)
        if token_match:
            token = re.sub(r"\s+", " ", token_match.group(0).strip())
            if SubsectionExtractor._is_valid_subsection(token):
                return token
        
        return ""
    
    @staticmethod
    def extract_all_subsections(content: str) -> list[str]:
        """
        Extract all subsection identifiers from content.
        
        This is used for the subsections array field (for frontend matching).
        
        Args:
            content: Document content text
            
        Returns:
            List of unique subsection identifiers in order of appearance
        """
        if not content:
            return []
        
        normalized_content = content.replace("\u00a0", " ")

        subsections = []
        seen = set()

        def is_plain_letter_digit(token: str) -> bool:
            return re.match(r'^[A-Z]\d+$', token, re.IGNORECASE) is not None
        
        # Extract from breadcrumbs
        for match in SubsectionExtractor.BREADCRUMB_PATTERN.finditer(normalized_content):
            subsection = match.group(1)
            if is_plain_letter_digit(subsection):
                continue
            if SubsectionExtractor._is_valid_subsection(subsection) and subsection not in seen:
                subsections.append(subsection)
                seen.add(subsection)
        
        # Extract from breadcrumb suffix format (] 2.1)
        for match in SubsectionExtractor.BREADCRUMB_SUFFIX_PATTERN.finditer(normalized_content):
            subsection = match.group(1)
            if is_plain_letter_digit(subsection):
                continue
            if SubsectionExtractor._is_valid_subsection(subsection) and subsection not in seen:
                subsections.append(subsection)
                seen.add(subsection)
        
        # Extract from markdown headings
        for match in SubsectionExtractor.MARKDOWN_HEADING_PATTERN.finditer(normalized_content):
            subsection = match.group(1)
            if is_plain_letter_digit(subsection):
                continue
            if SubsectionExtractor._is_valid_subsection(subsection) and subsection not in seen:
                subsections.append(subsection)
                seen.add(subsection)
        
        # Also do line-by-line extraction to catch subsections not in headings (e.g., B.7.1 in content)
        previous_line = ""
        for line in normalized_content.split('\n'):
            if not line.strip() or line.strip() == "---":
                continue
            
            cleaned_line = SubsectionExtractor.clean_line(line)
            if not cleaned_line:
                continue
            
            for pattern in SubsectionExtractor.CONTENT_SUBSECTION_PATTERNS:
                match = re.match(pattern, cleaned_line, re.IGNORECASE)
                if match:
                    subsection = match.group(1)
                    if re.match(r'^[A-Z]\d+$', subsection, re.IGNORECASE):
                        break
                    if previous_line.lower() in ("rule", "para"):
                        combined = f"{previous_line.title()} {subsection}"
                        if combined not in seen:
                            subsections.append(combined)
                            seen.add(combined)
                    elif subsection not in seen:
                        subsections.append(subsection)
                        seen.add(subsection)
                    break  # Only one pattern per line
            
            previous_line = cleaned_line
        
        # Sweep for all subsection-like tokens anywhere in content
        for match in SubsectionExtractor.TOKEN_PATTERN.finditer(normalized_content):
            token = match.group(0).strip()
            if not token:
                continue
            token = re.sub(r"\s+", " ", token)
            if token not in seen:
                subsections.append(token)
                seen.add(token)

        return subsections
    
    @staticmethod
    def extract_subsections_dict(content: str, max_lines: int = 30) -> dict[str, any]:
        """
        Extract both primary and all subsections in one pass.
        
        Args:
            content: Document content text
            max_lines: Maximum lines to scan for primary subsection
            
        Returns:
            Dict with 'primary' and 'all' keys
        """
        return {
            'primary': SubsectionExtractor.extract_first_subsection(content, max_lines),
            'all': SubsectionExtractor.extract_all_subsections(content)
        }
