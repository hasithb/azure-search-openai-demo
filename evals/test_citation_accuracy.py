#!/usr/bin/env python3
"""
Citation Accuracy Testing Script
================================
Comprehensive testing of citation generation and subsection matching.

This script:
1. Generates every possible citation from the search index
2. Tests whether citations correctly match subsections in supporting content
3. Identifies cases where subsection detection fails
4. Produces a detailed report with proposed fixes

CUSTOM: This file is part of the merge-safe customizations for legal RAG.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CitationTestResult:
    """Result of testing a single citation."""
    citation: str
    subsection: str
    sourcepage: str
    sourcefile: str
    content_preview: str
    
    # Matching results
    subsection_found_in_content: bool = False
    subsection_supported: bool = False
    subsection_supported_by_metadata: bool = False
    subsection_matches_index_field: bool = False
    citation_is_three_part: bool = False
    citation_is_two_part: bool = False
    
    # Issues found
    issues: list[str] = field(default_factory=list)
    
    # Debug info
    extracted_subsection: str = ""
    content_first_lines: list[str] = field(default_factory=list)


@dataclass
class CitationTestSummary:
    """Summary of all citation tests."""
    total_documents: int = 0
    total_citations: int = 0
    three_part_citations: int = 0
    two_part_citations: int = 0
    single_part_citations: int = 0
    
    # Success counts
    subsection_in_content: int = 0
    subsection_supported: int = 0
    subsection_matches_index: int = 0
    perfect_matches: int = 0
    
    # Failure counts
    subsection_missing_from_content: int = 0
    subsection_detection_failed: int = 0
    citation_format_issues: int = 0
    
    # Detailed results
    results: list[CitationTestResult] = field(default_factory=list)
    failed_results: list[CitationTestResult] = field(default_factory=list)


# ============================================================================
# CITATION BUILDER (Mirror of backend logic)
# ============================================================================

class CitationBuilder:
    """
    Mirror of app/backend/customizations/approaches/citation_builder.py
    for testing purposes.
    """
    
    # Pattern for detecting subsection markers in content
    CONTENT_SUBSECTION_PATTERNS = [
        r'^([A-Z]\d+\.\d+)\b',           # A4.1, B2.3, etc.
        r'^([A-Z]\.\d+(?:\.\d+)?)\b',   # C.2, F.1, A.1.1, etc.
        r'^(\d+\.\d+)\b',                # 1.1, 2.3, etc.
        r'^([A-Z]\d+)\b',                # A1, B2, etc.
        r'^(Rule \d+(?:\.\d+)?)\b',      # Rule 1, Rule 1.1
        r'^(Para \d+(?:\.\d+)?)\b',      # Para 1.1
        r'^(Part \d+)\b',                # Part 1, Part 2, etc.
        r'^(\d+\.\d+)$',                 # Standalone subsection number
    ]
    
    # Pattern for extracting subsection from encoded sourcepage
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
    
    def extract_subsection(self, doc: dict) -> str:
        """Extract subsection identifier from document content or sourcepage."""
        indexed_subsection = doc.get("subsection_id", "")
        if indexed_subsection:
            return indexed_subsection

        content = doc.get("content", "")
        sourcepage = doc.get("sourcepage", "")

        # Priority 1: Check content
        if content:
            lines = content.split('\n')[:30]
            for line in lines:
                line = line.strip()
                if not line or line == "---":
                    continue

                cleaned_line = re.sub(r'^#+\s*', '', line)
                cleaned_line = re.sub(r'^\*\*([^*]+)\*\*', r'\1', cleaned_line)
                cleaned_line = re.sub(r'^__([^_]+)__', r'\1', cleaned_line)
                cleaned_line = cleaned_line.strip()

                for pattern in self.CONTENT_SUBSECTION_PATTERNS:
                    match = re.match(pattern, cleaned_line, re.IGNORECASE)
                    if match:
                        return match.group(1)

                if cleaned_line != line:
                    for pattern in self.CONTENT_SUBSECTION_PATTERNS:
                        match = re.match(pattern, line, re.IGNORECASE)
                        if match:
                            return match.group(1)

                if re.match(r'^\d+\.\d+$', cleaned_line):
                    return cleaned_line

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
    
    def build_citation(self, doc: dict, index: int) -> str:
        """Build citation from document data."""
        sourcepage = (doc.get("sourcepage", "") or "").strip()
        sourcefile = (doc.get("sourcefile", "") or "").strip()
        
        # Extract subsection from document
        subsection = self.extract_subsection(doc)
        
        # Avoid duplication between subsection and sourcepage
        final_sourcepage = sourcepage
        
        if subsection and sourcepage:
            if sourcepage == subsection:
                final_sourcepage = ""
            elif re.search(r'[A-Z]+\d*[A-Z]*-' + re.escape(subsection), sourcepage):
                final_sourcepage = ""
        
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
            citation = f"Source {index}"
        
        return citation
    
    def check_subsection_in_content(self, subsection: str, content: str) -> bool:
        """Check if subsection appears in content."""
        if not subsection or not content:
            return False
        
        # Escape special regex characters
        escaped = re.escape(subsection)
        
        # Try various patterns
        patterns = [
            rf'(^|\n)\s*{escaped}\b',           # At line start
            rf'\b{escaped}\b',                   # As word boundary
            rf'#{1,3}\s*{escaped}\b',            # As markdown heading
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False


# ============================================================================
# AZURE CLIENTS
# ============================================================================

def load_azd_env() -> dict[str, str]:
    """Load environment variables from azd."""
    try:
        result = subprocess.run(
            ["azd", "env", "get-values"],
            capture_output=True,
            text=True,
            check=True,
        )
        env = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, _, value = line.partition("=")
                env[key] = value.strip('"')
        return env
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to load azd env: {e}[/red]")
        return {}


async def get_search_client():
    """Create Azure Search client."""
    from azure.identity.aio import DefaultAzureCredential
    from azure.search.documents.aio import SearchClient
    
    env = load_azd_env()
    search_service = env.get("AZURE_SEARCH_SERVICE", "cpr-rag")
    search_index = env.get("AZURE_SEARCH_INDEX", "legal-court-rag-index")
    
    endpoint = f"https://{search_service}.search.windows.net"
    
    credential = DefaultAzureCredential()
    
    return SearchClient(
        endpoint=endpoint,
        index_name=search_index,
        credential=credential,
    ), credential


# ============================================================================
# DOCUMENT RETRIEVAL
# ============================================================================

async def retrieve_all_documents(max_docs: int = 1000) -> list[dict]:
    """Retrieve documents from Azure Search index."""
    console.print("[blue]Connecting to Azure Search...[/blue]")
    
    try:
        search_client, credential = await get_search_client()
    except Exception as e:
        console.print(f"[red]Failed to create search client: {e}[/red]")
        return []
    
    documents = []
    
    try:
        async with search_client:
            # Use * to get all documents
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                task = progress.add_task("Retrieving documents...", total=max_docs)
                
                # Search with wildcards to get all documents
                search_results = await search_client.search(
                    search_text="*",
                    select=["id", "content", "sourcepage", "sourcefile", "category"],
                    top=max_docs,
                )
                async for result in search_results:
                    doc = {
                        "id": result.get("id", ""),
                        "content": result.get("content", ""),
                        "sourcepage": result.get("sourcepage", ""),
                        "sourcefile": result.get("sourcefile", ""),
                        "category": result.get("category", ""),
                        "subsection_id": result.get("subsection_id", ""),  # If index has this field
                    }
                    documents.append(doc)
                    progress.update(task, completed=len(documents))
                    
                    if len(documents) >= max_docs:
                        break
    finally:
        await credential.close()
    
    console.print(f"[green]Retrieved {len(documents)} documents[/green]")
    return documents


# ============================================================================
# CITATION TESTING
# ============================================================================

def test_single_document(doc: dict, index: int, builder: CitationBuilder) -> CitationTestResult:
    """Test citation generation and matching for a single document."""
    content = doc.get("content", "")
    sourcepage = doc.get("sourcepage", "")
    sourcefile = doc.get("sourcefile", "")
    index_subsection = doc.get("subsection_id", "")  # From index if available
    
    # Build citation
    citation = builder.build_citation(doc, index)
    
    # Extract subsection
    extracted_subsection = builder.extract_subsection(doc)
    
    # Get first few lines for debugging
    content_first_lines = [l.strip() for l in content.split('\n')[:5] if l.strip()]
    
    # Parse citation format
    citation_parts = [p.strip() for p in citation.split(',')]
    is_three_part = len(citation_parts) >= 3
    is_two_part = len(citation_parts) == 2
    
    # Determine the subsection from citation
    if is_three_part:
        citation_subsection = citation_parts[0]
    elif is_two_part and not citation_parts[0].endswith('.md'):
        # First part might be subsection if not a filename
        if re.match(r'^[\dA-Z]', citation_parts[0]) and not citation_parts[0].startswith('http'):
            citation_subsection = citation_parts[0]
        else:
            citation_subsection = ""
    else:
        citation_subsection = ""
    
    # Test subsection in content
    subsection_found_in_content = builder.check_subsection_in_content(
        citation_subsection or extracted_subsection, 
        content
    )

    # Metadata-backed support (index field or sourcepage)
    candidate_subsection = citation_subsection or extracted_subsection
    normalized_candidate = (candidate_subsection or "").strip().lower().rstrip(".: ")
    normalized_index_subsection = (index_subsection or "").strip().lower().rstrip(".: ")
    sourcepage_lower = (sourcepage or "").strip().lower()

    subsection_supported_by_metadata = False
    if normalized_candidate:
        if normalized_index_subsection and normalized_candidate == normalized_index_subsection:
            subsection_supported_by_metadata = True
        elif sourcepage_lower and normalized_candidate in sourcepage_lower:
            subsection_supported_by_metadata = True

    subsection_supported = subsection_found_in_content or subsection_supported_by_metadata
    
    # Test subsection matches index field
    subsection_matches_index = (
        index_subsection and 
        extracted_subsection and 
        (index_subsection == extracted_subsection or 
         index_subsection.lower() == extracted_subsection.lower())
    )
    
    # Identify issues
    issues = []
    
    if extracted_subsection and not subsection_supported:
        issues.append(f"Extracted subsection '{extracted_subsection}' not found in content or metadata")
    
    if citation_subsection and extracted_subsection and citation_subsection != extracted_subsection:
        issues.append(f"Citation subsection '{citation_subsection}' differs from extracted '{extracted_subsection}'")
    
    if not extracted_subsection and content_first_lines:
        # Check if first line looks like it should be a subsection
        first_line = content_first_lines[0] if content_first_lines else ""
        if re.match(r'^#+\s*\d', first_line) or re.match(r'^#+\s*[A-Z]\d', first_line):
            issues.append(f"First line looks like heading but no subsection extracted: '{first_line[:50]}'")
    
    if is_three_part and not citation_subsection:
        issues.append("Three-part citation expected but no subsection in first position")
    
    # Check for common patterns that should be subsections but weren't detected
    if content:
        first_non_empty_line = ""
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped and stripped != "---":
                first_non_empty_line = stripped
                break
        
        # Common patterns that should be detected
        should_be_subsection_patterns = [
            r'^#+\s*(\d+\.\d+)',           # ## 1.1
            r'^#+\s*([A-Z]\d+\.\d+)',       # ## A4.1  
            r'^#+\s*Rule\s+\d+',            # ## Rule 31
            r'^\*\*(\d+\.\d+)\*\*',         # **1.1**
            r'^(\d+\.\d+)\s+[A-Z]',         # 1.1 Capital letter
        ]
        
        for pattern in should_be_subsection_patterns:
            if re.match(pattern, first_non_empty_line, re.IGNORECASE):
                if not extracted_subsection:
                    issues.append(f"Pattern '{pattern}' matched but no subsection extracted: '{first_non_empty_line[:50]}'")
                break
    
    return CitationTestResult(
        citation=citation,
        subsection=citation_subsection or extracted_subsection,
        sourcepage=sourcepage,
        sourcefile=sourcefile,
        content_preview=content[:200] if content else "",
        subsection_found_in_content=subsection_found_in_content,
        subsection_supported=subsection_supported,
        subsection_supported_by_metadata=subsection_supported_by_metadata,
        subsection_matches_index_field=subsection_matches_index,
        citation_is_three_part=is_three_part,
        citation_is_two_part=is_two_part,
        issues=issues,
        extracted_subsection=extracted_subsection,
        content_first_lines=content_first_lines,
    )


def run_all_tests(documents: list[dict]) -> CitationTestSummary:
    """Run citation tests on all documents."""
    builder = CitationBuilder()
    summary = CitationTestSummary(total_documents=len(documents))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Testing citations...", total=len(documents))
        
        for i, doc in enumerate(documents, 1):
            result = test_single_document(doc, i, builder)
            summary.results.append(result)
            summary.total_citations += 1
            
            # Categorize citation format
            if result.citation_is_three_part:
                summary.three_part_citations += 1
            elif result.citation_is_two_part:
                summary.two_part_citations += 1
            else:
                summary.single_part_citations += 1
            
            # Count successes
            if result.subsection_found_in_content:
                summary.subsection_in_content += 1
            else:
                summary.subsection_missing_from_content += 1

            if result.subsection_supported:
                summary.subsection_supported += 1
            
            if result.subsection_matches_index_field:
                summary.subsection_matches_index += 1
            
            if result.subsection_supported and not result.issues:
                summary.perfect_matches += 1
            
            # Track failures
            if result.issues:
                summary.failed_results.append(result)
                if "not found in content or metadata" in " ".join(result.issues):
                    summary.subsection_detection_failed += 1
                if "Citation subsection" in " ".join(result.issues):
                    summary.citation_format_issues += 1
            
            progress.update(task, completed=i)
    
    return summary


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def analyze_failures(summary: CitationTestSummary) -> dict[str, list[CitationTestResult]]:
    """Categorize failures by type."""
    categories = {
        "subsection_not_in_content": [],
        "subsection_mismatch": [],
        "pattern_not_detected": [],
        "three_part_missing_subsection": [],
        "markdown_heading_undetected": [],
        "other": [],
    }
    
    for result in summary.failed_results:
        categorized = False
        
        for issue in result.issues:
            if "not found in content" in issue:
                categories["subsection_not_in_content"].append(result)
                categorized = True
            elif "differs from extracted" in issue:
                categories["subsection_mismatch"].append(result)
                categorized = True
            elif "Pattern" in issue and "matched but no subsection" in issue:
                categories["pattern_not_detected"].append(result)
                categorized = True
            elif "Three-part citation expected" in issue:
                categories["three_part_missing_subsection"].append(result)
                categorized = True
            elif "looks like heading" in issue:
                categories["markdown_heading_undetected"].append(result)
                categorized = True
        
        if not categorized:
            categories["other"].append(result)
    
    return categories


def generate_report(summary: CitationTestSummary, output_path: Path) -> None:
    """Generate a detailed report."""
    console.print("\n" + "="*80)
    console.print("[bold blue]CITATION ACCURACY REPORT[/bold blue]")
    console.print("="*80 + "\n")
    
    # Summary table
    summary_table = Table(title="Summary Statistics")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Percentage", justify="right")
    
    total = summary.total_citations
    
    summary_table.add_row("Total Documents", str(total), "100%")
    summary_table.add_row("Three-part Citations", str(summary.three_part_citations), 
                         f"{summary.three_part_citations/total*100:.1f}%")
    summary_table.add_row("Two-part Citations", str(summary.two_part_citations),
                         f"{summary.two_part_citations/total*100:.1f}%")
    summary_table.add_row("Single-part Citations", str(summary.single_part_citations),
                         f"{summary.single_part_citations/total*100:.1f}%")
    summary_table.add_row("", "", "")
    summary_table.add_row("[green]Subsection in Content[/green]", 
                         str(summary.subsection_in_content),
                         f"{summary.subsection_in_content/total*100:.1f}%")
    summary_table.add_row("[green]Subsection Supported (Content or Metadata)[/green]", 
                         str(summary.subsection_supported),
                         f"{summary.subsection_supported/total*100:.1f}%")
    summary_table.add_row("[green]Perfect Matches[/green]", 
                         str(summary.perfect_matches),
                         f"{summary.perfect_matches/total*100:.1f}%")
    summary_table.add_row("", "", "")
    summary_table.add_row("[red]Subsection Missing[/red]", 
                         str(summary.subsection_missing_from_content),
                         f"{summary.subsection_missing_from_content/total*100:.1f}%")
    summary_table.add_row("[red]Detection Failed[/red]", 
                         str(summary.subsection_detection_failed),
                         f"{summary.subsection_detection_failed/total*100:.1f}%")
    summary_table.add_row("[red]Format Issues[/red]", 
                         str(summary.citation_format_issues),
                         f"{summary.citation_format_issues/total*100:.1f}%")
    
    console.print(summary_table)
    
    # Analyze failures
    failure_categories = analyze_failures(summary)
    
    console.print("\n" + "="*80)
    console.print("[bold yellow]FAILURE ANALYSIS[/bold yellow]")
    console.print("="*80 + "\n")
    
    for category, results in failure_categories.items():
        if results:
            console.print(f"\n[bold red]{category.upper().replace('_', ' ')}[/bold red]: {len(results)} cases")
            
            # Show first 3 examples
            for i, result in enumerate(results[:3], 1):
                console.print(f"\n  Example {i}:")
                console.print(f"    Citation: {result.citation}")
                console.print(f"    Subsection: {result.subsection or '(none)'}")
                console.print(f"    Extracted: {result.extracted_subsection or '(none)'}")
                console.print(f"    Sourcepage: {result.sourcepage}")
                console.print(f"    First line: {result.content_first_lines[0] if result.content_first_lines else '(empty)'}")
                console.print(f"    Issues: {result.issues}")
    
    # Save detailed results to JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_documents": summary.total_documents,
            "total_citations": summary.total_citations,
            "three_part_citations": summary.three_part_citations,
            "two_part_citations": summary.two_part_citations,
            "single_part_citations": summary.single_part_citations,
            "subsection_in_content": summary.subsection_in_content,
            "subsection_supported": summary.subsection_supported,
            "perfect_matches": summary.perfect_matches,
            "subsection_missing_from_content": summary.subsection_missing_from_content,
            "subsection_detection_failed": summary.subsection_detection_failed,
            "citation_format_issues": summary.citation_format_issues,
        },
        "failure_categories": {
            cat: [
                {
                    "citation": r.citation,
                    "subsection": r.subsection,
                    "extracted_subsection": r.extracted_subsection,
                    "sourcepage": r.sourcepage,
                    "sourcefile": r.sourcefile,
                    "content_preview": r.content_preview[:100],
                    "content_first_lines": r.content_first_lines[:3],
                    "issues": r.issues,
                }
                for r in results
            ]
            for cat, results in failure_categories.items()
        },
        "all_results": [
            {
                "citation": r.citation,
                "subsection": r.subsection,
                "extracted_subsection": r.extracted_subsection,
                "sourcepage": r.sourcepage,
                "sourcefile": r.sourcefile,
                "is_three_part": r.citation_is_three_part,
                "is_two_part": r.citation_is_two_part,
                "subsection_found": r.subsection_found_in_content,
                "subsection_supported": r.subsection_supported,
                "subsection_supported_by_metadata": r.subsection_supported_by_metadata,
                "issues": r.issues,
            }
            for r in summary.results
        ],
    }
    
    output_path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[green]Detailed report saved to: {output_path}[/green]")


def generate_proposed_fixes(summary: CitationTestSummary) -> None:
    """Generate proposed fixes based on failure analysis."""
    console.print("\n" + "="*80)
    console.print("[bold green]PROPOSED FIXES[/bold green]")
    console.print("="*80 + "\n")
    
    failure_categories = analyze_failures(summary)
    
    fixes = []
    
    # Fix 1: Markdown heading detection
    if failure_categories.get("markdown_heading_undetected"):
        fixes.append({
            "issue": "Markdown headings not detected as subsections",
            "count": len(failure_categories["markdown_heading_undetected"]),
            "location": "app/backend/customizations/approaches/citation_builder.py",
            "proposed_fix": """
Add patterns to CONTENT_SUBSECTION_PATTERNS:
    r'^#+\\s*([A-Z]\\d+\\.\\d+)\\b',       # ## A4.1 markdown heading
    r'^#+\\s*(\\d+\\.\\d+)\\b',            # ## 1.1 markdown heading
    r'^\\*\\*([A-Z]?\\d+\\.\\d+)\\*\\*',   # **1.1** bold subsection
""",
            "impact": "Low risk - adds new patterns without changing existing behavior",
        })
    
    # Fix 2: Pattern not detected
    if failure_categories.get("pattern_not_detected"):
        examples = failure_categories["pattern_not_detected"][:3]
        sample_lines = [r.content_first_lines[0] if r.content_first_lines else "" for r in examples]
        fixes.append({
            "issue": "Known patterns not being matched",
            "count": len(failure_categories["pattern_not_detected"]),
            "sample_lines": sample_lines,
            "location": "app/backend/customizations/approaches/citation_builder.py",
            "proposed_fix": """
Modify extract_subsection() to strip markdown formatting before matching:
    
    # Before checking patterns, strip common markdown
    line = line.lstrip('#').strip()
    line = re.sub(r'^\\*\\*(.+?)\\*\\*', r'\\1', line)  # Remove bold
    line = re.sub(r'^__(.+?)__', r'\\1', line)          # Remove underline bold
""",
            "impact": "Low risk - preprocessing step before pattern matching",
        })
    
    # Fix 3: Subsection not in content
    if failure_categories.get("subsection_not_in_content"):
        fixes.append({
            "issue": "Extracted subsection not found when matching in content",
            "count": len(failure_categories["subsection_not_in_content"]),
            "location": "Multiple files",
            "proposed_fix": """
1. In citation_builder.py check_subsection_in_content():
   - Normalize both subsection and content before matching
   - Handle case-insensitive matching for Rule/Part prefixes
   - Consider partial matches for longer subsection IDs

2. In frontend SupportingContent.tsx findMatchingContentIndex():
   - Add fallback matching using sourcepage only
   - Log cases where subsection matching fails for debugging
""",
            "impact": "Medium risk - changes matching behavior",
        })
    
    # Fix 4: Three-part citations missing subsection
    if failure_categories.get("three_part_missing_subsection"):
        fixes.append({
            "issue": "Three-part citations expected but subsection not in first position",
            "count": len(failure_categories["three_part_missing_subsection"]),
            "location": "app/backend/customizations/approaches/citation_builder.py",
            "proposed_fix": """
In build_enhanced_citation():
    - Add validation that three-part citations always have valid subsection
    - Fall back to two-part format if subsection extraction fails
    - Log warning for debugging
""",
            "impact": "Low risk - defensive coding",
        })
    
    # Fix 5: Subsection mismatch
    if failure_categories.get("subsection_mismatch"):
        fixes.append({
            "issue": "Mismatch between citation subsection and extracted subsection",
            "count": len(failure_categories["subsection_mismatch"]),
            "location": "app/backend/customizations/approaches/citation_builder.py",
            "proposed_fix": """
In extract_subsection():
    - Ensure consistent normalization of subsection format
    - Standardize on uppercase for letter prefixes (A4.1 not a4.1)
    - Keep Rule/Part prefixes consistent
""",
            "impact": "Low risk - normalization changes",
        })
    
    # Print fixes
    for i, fix in enumerate(fixes, 1):
        console.print(Panel(
            f"""[bold]Issue:[/bold] {fix['issue']}
[bold]Affected:[/bold] {fix['count']} documents
[bold]Location:[/bold] {fix['location']}
[bold]Impact:[/bold] {fix['impact']}

[bold]Proposed Fix:[/bold]
{fix['proposed_fix']}
""",
            title=f"Fix #{i}",
            border_style="green",
        ))
    
    # Generate patch file suggestions
    console.print("\n[bold]To implement these fixes, update the following files:[/bold]")
    console.print("1. app/backend/customizations/approaches/citation_builder.py")
    console.print("2. app/frontend/src/components/SupportingContent/SupportingContent.tsx")
    console.print("3. app/frontend/src/components/Answer/Answer.tsx")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point."""
    console.print(Panel(
        "[bold blue]Citation Accuracy Testing Script[/bold blue]\n\n"
        "This script tests citation generation and subsection matching\n"
        "for all documents in the Azure Search index.",
        title="Citation Accuracy Test",
    ))
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test citation accuracy")
    parser.add_argument("--max-docs", type=int, default=1000, help="Maximum documents to test")
    parser.add_argument("--output", type=str, default="citation_accuracy_report.json", 
                       help="Output report file")
    args = parser.parse_args()
    
    output_path = Path(__file__).parent / "results" / args.output
    output_path.parent.mkdir(exist_ok=True)
    
    # Retrieve documents
    documents = await retrieve_all_documents(max_docs=args.max_docs)
    
    if not documents:
        console.print("[red]No documents retrieved. Check Azure Search connection.[/red]")
        return
    
    # Run tests
    summary = run_all_tests(documents)
    
    # Generate report
    generate_report(summary, output_path)
    
    # Generate proposed fixes
    generate_proposed_fixes(summary)
    
    console.print("\n[bold green]Testing complete![/bold green]")
    
    # Return exit code based on results
    if summary.perfect_matches < summary.total_citations * 0.8:
        console.print("[yellow]Warning: Less than 80% perfect matches[/yellow]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code or 0)
