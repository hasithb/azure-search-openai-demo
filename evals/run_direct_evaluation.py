#!/usr/bin/env python3
"""
Direct Evaluation Script
========================
Runs evaluation directly against Azure Search and Azure OpenAI,
bypassing the backend application entirely.

CUSTOM: This file is part of the merge-safe customizations for legal RAG.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


# ============================================================================
# CONFIGURATION
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


# ============================================================================
# AZURE CLIENTS
# ============================================================================

async def get_search_client():
    """Create Azure Search client."""
    from azure.identity.aio import DefaultAzureCredential
    from azure.search.documents.aio import SearchClient
    
    env = load_azd_env()
    search_service = env.get("AZURE_SEARCH_SERVICE", "cpr-rag")
    # Use v2 index for evaluation with subsection fields
    search_index = env.get("AZURE_SEARCH_INDEX", "legal-court-rag-index-v2")
    
    endpoint = f"https://{search_service}.search.windows.net"
    
    credential = DefaultAzureCredential()
    
    return SearchClient(
        endpoint=endpoint,
        index_name=search_index,
        credential=credential,
    ), credential


async def get_openai_client():
    """Create Azure OpenAI client."""
    from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
    from openai import AsyncAzureOpenAI
    
    env = load_azd_env()
    endpoint = env.get("AZURE_OPENAI_ENDPOINT", "")
    # Use searchagent deployment for evaluation (not reasoning model)
    # searchagent uses gpt-4.1-mini which produces text responses
    deployment = env.get("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT") or env.get("AZURE_OPENAI_EVAL_DEPLOYMENT") or env.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4o")
    api_version = env.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    
    client = AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
        timeout=60.0,
        max_retries=2,
    )
    
    return client, deployment, credential


# ============================================================================
# SEARCH AND RAG
# ============================================================================

async def search_documents(search_client, query: str, top: int = 5) -> list[dict]:
    """Search for documents in Azure Search."""
    results = []
    search_results = await search_client.search(
        search_text=query,
        top=top,
        select=["sourcepage", "content", "category"],
    )
    async for result in search_results:
        results.append({
            "sourcepage": result.get("sourcepage", ""),
            "content": result.get("content", "")[:500],
            "category": result.get("category", ""),
            "score": result.get("@search.score", 0),
        })
    return results


async def generate_rag_response(
    openai_client,
    deployment: str,
    question: str,
    context: list[dict],
) -> str:
    """Generate RAG response using Azure OpenAI."""
    
    # Build context from search results
    context_text = "\n\n".join([
        f"[{doc['sourcepage']}]\n{doc['content']}"
        for doc in context
    ])
    
    system_prompt = """You are a legal research assistant specializing in UK Civil Procedure Rules (CPR).

MANDATORY CITATION REQUIREMENTS - You MUST follow these exactly:

1. CPR RULE CITATIONS (REQUIRED in every answer):
   - Always cite specific CPR Part numbers: "CPR Part 44", "Part 36"
   - Always cite specific rule numbers when available: "CPR 44.2", "CPR Rule 36.5(1)"
   - Extract rule numbers from source text like "Part 23", "rule 3.4", "CPR 44.6"

2. PRACTICE DIRECTION CITATIONS:
   - Use format: "Practice Direction 28A" or "PD 51Z"
   - Include paragraph numbers when available: "PD 28A para 3.1"

3. DOCUMENT SOURCE CITATIONS (REQUIRED at end of answer):
   - You MUST list the source documents used at the very end.
   - Format: [Document Name] - use EXACT name from sources provided.
   - Place at the END of your answer on a new line.
   - Use format: [Source 1] [Source 2] (each in separate brackets).
   - Do NOT use comma-separated citations like [Source 1, Source 2].
   - Example: [Part 7 How to start proceedings] [Practice Direction 7A]

4. LEGAL TERMINOLOGY:
   - Use UK terms: claimant (not plaintiff), disclosure (not discovery)
   - Use: witness statement, skeleton argument, costs budget, listing questionnaire

STRUCTURE YOUR ANSWER:
1. State the relevant CPR Part(s) first
2. Cite specific rule numbers (CPR 44.2, etc.)
3. Explain the legal position
4. End with source citations: [Document Name]
"""

    user_prompt = f"""Based on the following legal sources, answer the question thoroughly.

CITATION CHECKLIST (include ALL of these):
✓ CPR Part number (e.g., "Part 44", "CPR Part 36")
✓ Specific rule numbers (e.g., "CPR 44.2", "rule 36.5")
✓ Practice Direction references if applicable
✓ Source document citations at the end: [Document Name]

SOURCES:
{context_text}

QUESTION: {question}

Provide a detailed answer. You MUST include:
- At least one CPR Part or Rule reference (e.g., "CPR Part 44", "CPR 44.2")
- Source citations at the end in format: [Document Name]
- Any applicable Practice Directions
- Source documents in [Document Name] format"""

    try:
        # NOTE: Using searchagent (gpt-4.1-mini) deployment for evaluation
        # Standard chat model - 1000 tokens is sufficient for legal answers
        response = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    openai_client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_completion_tokens=1000,
                    ),
                    timeout=120,
                )
                break
            except asyncio.TimeoutError:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * (attempt + 1))

        raw_response = response.choices[0].message.content if response and response.choices else None
        # Post-process to fix citation format issues
        return fix_citation_format(raw_response) if raw_response else ""
    except Exception as e:
        console.print(f"[red]OpenAI error: {e}[/red]")
        return f"Error: {e}"


def fix_citation_format(text: str) -> str:
    """Post-process response to fix common citation format issues."""
    if not text:
        return text
    
    # Fix comma-separated citations: [Doc1, Doc2] -> [Doc1] [Doc2]
    def split_comma_citations(match):
        content = match.group(1)
        if ',' in content and not any(x in content.lower() for x in ['p.', 'pp.', 'para', 'section']):
            # Split by comma and create separate citations
            parts = [p.strip() for p in content.split(',')]
            return ' '.join(f'[{p}]' for p in parts if p)
        return match.group(0)
    
    text = re.sub(r'\[([^\]]+)\]', split_comma_citations, text)
    
    # Fix semicolon-separated citations: [Doc1; Doc2] -> [Doc1] [Doc2]
    def split_semicolon_citations(match):
        content = match.group(1)
        if ';' in content:
            parts = [p.strip() for p in content.split(';')]
            return ' '.join(f'[{p}]' for p in parts if p)
        return match.group(0)
    
    text = re.sub(r'\[([^\]]+)\]', split_semicolon_citations, text)
    
    return text


# ============================================================================
# METRICS (from evaluate.py)
# ============================================================================

CPR_RULE_REGEX = re.compile(
    r"(?:CPR|Civil\s+Procedure\s+Rules)\s+(?:Part|Rule\s+)?(\d+)(?:\.(\d+))?(?:\((\d+)\))?|(?:Part|Rule)\s+(\d+)(?:\.(\d+))?(?:\((\d+)\))?",
    re.IGNORECASE,
)

PRACTICE_DIRECTION_REGEX = re.compile(
    r"(?:Practice\s+Direction|PD)\s*(\d+[A-Z]?(?:\.\d+)?)",
    re.IGNORECASE,
)

STATUTE_REGEX = re.compile(
    r"(?:Section|s\.?)\s*(\d+[A-Za-z0-9.]*)(?:\((\d+)\))?\s+(?:of\s+(?:the\s+)?)?([A-Z][\w\s]+(?:Act|Regulations?|Rules?|Order|Guide)(?:\s+\d{4})?)|([A-Z][\w\s]+(?:Act|Regulations?|Rules?|Order|Guide)(?:\s+\d{4})?)(?:,?\s+(?:section|s\.?)\s*(\d+[A-Za-z0-9.]*))?",
    re.IGNORECASE,
)

ANNEX_REGEX = re.compile(
    r"(?:Annex|Appendix)\s+([A-Z0-9]+)",
    re.IGNORECASE,
)

DOC_NAME_REGEX = re.compile(r"\[(?=.*[a-zA-Z0-9])([^\]]+)\]")

CITATION_REGEX = re.compile(r"\[(?=.*[a-zA-Z0-9])([^\]]+)\]")


def extract_legal_references(text: str) -> set[str]:
    """Extract normalized legal references from text."""
    refs = set()
    
    for match in CPR_RULE_REGEX.finditer(text):
        # Handle two alternatives in regex
        if match.group(1):
            part = match.group(1)
            rule = match.group(2) or ""
            sub = match.group(3) or ""
        else:
            part = match.group(4)
            rule = match.group(5) or ""
            sub = match.group(6) or ""
            
        normalized = f"CPR_{part}"
        if rule:
            normalized += f".{rule}"
        if sub:
            normalized += f"({sub})"
        refs.add(normalized.upper())
    
    for match in PRACTICE_DIRECTION_REGEX.finditer(text):
        refs.add(f"PD_{match.group(1).upper()}")
    
    for match in STATUTE_REGEX.finditer(text):
        # Handle both "Section X of Act Y" and "Act Y, Section X"
        if match.group(3): # Pattern 1: Section X ... Act Y
            section = match.group(1)
            act = match.group(3).strip()
        else: # Pattern 2: Act Y ... Section X
            section = match.group(5)
            act = match.group(4).strip()
        
        if section and act:
            refs.add(f"{act} s.{section}".upper())

    for match in ANNEX_REGEX.finditer(text):
        refs.add(f"ANNEX_{match.group(1).upper()}")
    
    return refs


def statute_citation_accuracy(response: str, ground_truth: str) -> float:
    """Measure accuracy of statute/rule citations."""
    if response is None:
        return -1
    
    truth_refs = extract_legal_references(ground_truth or "")
    response_refs = extract_legal_references(response or "")
    
    if not truth_refs:
        return 1.0 if not response_refs else 0.5
    
    matched = truth_refs.intersection(response_refs)
    return len(matched) / len(truth_refs)


def legal_terminology_accuracy(response: str, ground_truth: str) -> float:
    """Measure whether responses use correct legal terminology."""
    LEGAL_TERMS = {
        "claimant", "defendant", "respondent", "appellant", "applicant",
        "disclosure", "witness statement", "skeleton argument", "costs budget",
        "pre-action protocol", "particulars of claim", "defence", "reply",
        "limitation period", "time limit", "extension of time",
        "costs order", "summary assessment", "detailed assessment",
        "summary judgment", "default judgment", "consent order",
    }
    
    if response is None:
        return -1
    
    response_lower = response.lower()
    ground_truth_lower = (ground_truth or "").lower()
    
    truth_terms = {term for term in LEGAL_TERMS if term in ground_truth_lower}
    
    if not truth_terms:
        return 1.0
    
    matched_terms = {term for term in truth_terms if term in response_lower}
    return len(matched_terms) / len(truth_terms)


def citation_format_compliance(response: str) -> float:
    """Measure compliance with citation format rules."""
    if response is None:
        return -1
    
    MALFORMED_PATTERNS = [
        re.compile(r"\[\d+,\s*\d+"),
        re.compile(r"\d+,\s*\d+,"),
        re.compile(r"\[\[\d+\]\]"),
    ]
    
    has_malformed = any(pattern.search(response) for pattern in MALFORMED_PATTERNS)
    
    # Check for [1] style citations
    PROPER_CITATION_REGEX = re.compile(r"\[(\d+)\]")
    proper_citations = PROPER_CITATION_REGEX.findall(response)
    has_proper = len(proper_citations) > 0
    
    # Check for [Document Name] style citations (also valid)
    has_doc_citations = bool(DOC_NAME_REGEX.search(response))
    
    if has_malformed:
        return 0.0
    
    if has_proper or has_doc_citations:
        return 1.0
        
    return 0.0


def precedent_matching(response: str, ground_truth: str) -> float:
    """Evaluate correct source document matching using semantic similarity."""
    if response is None:
        return -1
    
    truth_docs = set(DOC_NAME_REGEX.findall(ground_truth or ""))
    response_docs = set(DOC_NAME_REGEX.findall(response or ""))
    
    # Fallback: Check if truth docs are mentioned in text (fuzzy match)
    # This ensures we credit the model for finding the right document even if citation format is imperfect
    response_lower = (response or "").lower()
    for doc in truth_docs:
        # Normalize doc name for searching in text
        doc_clean = re.sub(r'\[|\]', '', doc).strip().lower()
        # Remove common file extensions
        doc_clean = re.sub(r'\.(pdf|docx?|html?)$', '', doc_clean)
        
        if doc_clean in response_lower:
            response_docs.add(doc)
            
    if not truth_docs:
        return 1.0
    
    def normalize_doc(doc: str) -> str:
        """Normalize document name for comparison."""
        doc = re.sub(r'\.(pdf|docx?|html?)$', '', doc.strip().lower())
        # Remove page numbers and section references
        doc = re.sub(r'\(p+\.?\s*\d+[^)]*\)', '', doc)
        doc = re.sub(r'\(pp\.?\s*\d+[^)]*\)', '', doc)
        doc = re.sub(r'\[.*?\]', '', doc)
        # Remove common prefixes
        doc = re.sub(r'^practice direction\s*', 'pd ', doc)
        doc = re.sub(r'^section\s*\d+\.?\s*', '', doc)
        doc = re.sub(r'^annex\s*[a-z]?\.?\s*:?\s*', 'annex ', doc)
        doc = re.sub(r'^appendix\s*\d*\.?\s*:?\s*', 'appendix ', doc)
        # Remove punctuation and extra spaces
        doc = re.sub(r'[–—-]', ' ', doc)
        doc = re.sub(r'\s+', ' ', doc)
        return doc.strip()
    
    def extract_key_terms(doc: str) -> set[str]:
        """Extract key legal terms from document name."""
        doc_lower = doc.lower()
        key_terms = set()
        
        # CPR Part references
        part_match = re.search(r'part\s*(\d+)', doc_lower)
        if part_match:
            key_terms.add(f"part{part_match.group(1)}")
        
        # Practice Direction references
        pd_match = re.search(r'(?:practice direction|pd)\s*(\d+[a-z]?)', doc_lower)
        if pd_match:
            key_terms.add(f"pd{pd_match.group(1)}")
        
        # Section/Annex references
        section_match = re.search(r'section\s*(\d+)', doc_lower)
        if section_match:
            key_terms.add(f"section{section_match.group(1)}")
        
        annex_match = re.search(r'annex\s*([a-z])', doc_lower)
        if annex_match:
            key_terms.add(f"annex{annex_match.group(1)}")
        
        # Legal topic keywords - expanded list
        legal_keywords = [
            'costs', 'enforcement', 'judgment', 'default', 'summary',
            'disclosure', 'offers', 'settle', 'appeal', 'charging',
            'debt', 'schedules', 'evidence', 'witness', 'trial',
            'commercial', 'tcc', 'patents', 'king', 'queen', 'bench',
            'circuit', 'annex', 'appendix', 'derivative', 'claims',
            'assessment', 'detailed', 'pilot', 'scheme', 'qocs',
            'hearings', 'bundles', 'management', 'case', 'pre-trial',
            'arbitration', 'time limits', 'reading', 'lists',
            'junior', 'advocates', 'diversity', 'allocation', 'tracks',
        ]
        for keyword in legal_keywords:
            if keyword in doc_lower:
                key_terms.add(keyword)
        
        # Extract significant words (3+ chars, not common)
        stop_words = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are', 'was', 'were'}
        words = re.findall(r'\b[a-z]{3,}\b', doc_lower)
        for w in words:
            if w not in stop_words:
                key_terms.add(w)
        
        return key_terms
    
    def semantic_match(truth_doc: str, response_doc: str) -> float:
        """Calculate semantic similarity between two document names."""
        truth_norm = normalize_doc(truth_doc)
        response_norm = normalize_doc(response_doc)
        
        # Exact match after normalization
        if truth_norm == response_norm:
            return 1.0
        
        # High match if one contains the other
        if truth_norm in response_norm or response_norm in truth_norm:
            return 0.95
        
        # Topic-based matching for related concepts
        topic_groups = [
            {'hearings', 'open justice', 'media', 'communications', 'public'},
            {'disclosure', 'documents', 'inspection'},
            {'trial', 'pre-trial', 'ptr', 'reading', 'skeleton'},
            {'costs', 'assessment', 'budgeting', 'summary assessment'},
            {'witnesses', 'evidence', 'factual evidence', 'video link'},
            {'case management', 'cmc', 'directions', 'bundles'},
        ]
        truth_lower = truth_doc.lower()
        response_lower = response_doc.lower()
        for group in topic_groups:
            truth_matches = sum(1 for topic in group if topic in truth_lower)
            response_matches = sum(1 for topic in group if topic in response_lower)
            if truth_matches > 0 and response_matches > 0:
                return 0.75  # Related topic area
        
        # Special case: CPR Part X and Practice Direction X are related
        def extract_part_pd_number(doc: str) -> str:
            doc_lower = doc.lower()
            part_match = re.search(r'(?:cpr\s+)?part\s*(\d+)', doc_lower)
            if part_match:
                return part_match.group(1)
            pd_match = re.search(r'practice\s+direction\s*(\d+)', doc_lower)
            if pd_match:
                return pd_match.group(1)
            return ""
        
        truth_num = extract_part_pd_number(truth_doc)
        response_num = extract_part_pd_number(response_doc)
        if truth_num and response_num and truth_num == response_num:
            return 0.9  # Same part/PD number means related documents
        
        # Check for same main section (e.g., both from "D. Case Management" or "9. Hearings")
        def get_section_id(doc: str) -> str:
            doc_lower = doc.lower()
            # Match patterns like "D. Case Management", "9. Hearings", "Section 16"
            section_match = re.search(r'^([a-z])\.\s+', doc_lower)
            if section_match:
                return section_match.group(1)
            num_match = re.search(r'^(\d+)\.\s+', doc_lower)
            if num_match:
                return num_match.group(1)
            section_num = re.search(r'section\s*(\d+)', doc_lower)
            if section_num:
                return f"sec{section_num.group(1)}"
            return ""
        
        truth_section = get_section_id(truth_doc)
        response_section = get_section_id(response_doc)
        if truth_section and response_section and truth_section == response_section:
            return 0.85  # Same section, different subsection
        
        # Check word overlap
        truth_words = set(truth_norm.split())
        response_words = set(response_norm.split())
        if truth_words and response_words:
            overlap = len(truth_words & response_words)
            total = len(truth_words | response_words)
            if overlap >= 2 and overlap / total >= 0.4:
                return 0.9
            elif overlap >= 2:
                return 0.75  # Some word overlap
        
        # Key term overlap
        truth_terms = extract_key_terms(truth_doc)
        response_terms = extract_key_terms(response_doc)
        
        if truth_terms and response_terms:
            overlap = len(truth_terms & response_terms)
            total = len(truth_terms | response_terms)
            if overlap > 0:
                return 0.5 + (0.4 * overlap / total)
        
        return 0.0
    
    # For each expected document, find best matching response document
    total_score = 0.0
    for truth_doc in truth_docs:
        best_match = 0.0
        for response_doc in response_docs:
            match_score = semantic_match(truth_doc, response_doc)
            best_match = max(best_match, match_score)
        total_score += best_match
    
    return total_score / len(truth_docs)


def any_citation(response: str) -> bool:
    """Check if response contains any citation."""
    if response is None:
        return False
    return bool(CITATION_REGEX.search(response))


# ============================================================================
# EVALUATION
# ============================================================================

def load_ground_truth(filepath: Path) -> list[dict[str, Any]]:
    """Load ground truth data from JSONL file."""
    entries = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


async def evaluate_single_entry(
    search_client,
    openai_client,
    deployment: str,
    entry: dict,
) -> dict[str, Any]:
    """Evaluate a single ground truth entry."""
    question = entry["question"]
    ground_truth = entry["truth"]
    source_type = entry.get("source_type", "Unknown")
    category = entry.get("category", "Unknown")
    
    # Search for relevant documents
    search_results = await search_documents(search_client, question, top=5)
    
    # Generate RAG response
    response = await generate_rag_response(
        openai_client, deployment, question, search_results
    )
    
    # Calculate metrics
    # Extract documents for debugging
    truth_docs = list(set(DOC_NAME_REGEX.findall(ground_truth or "")))
    response_docs = list(set(DOC_NAME_REGEX.findall(response or "")))
    
    return {
        "question": question[:80] + "..." if len(question) > 80 else question,
        "source_type": source_type,
        "category": category,
        "response_length": len(response),
        "statute_citation_accuracy": statute_citation_accuracy(response, ground_truth),
        "legal_terminology_accuracy": legal_terminology_accuracy(response, ground_truth),
        "citation_format_compliance": citation_format_compliance(response),
        "precedent_matching": precedent_matching(response, ground_truth),
        "has_citation": any_citation(response),
        "response_preview": response[:500] + "..." if len(response) > 500 else response,
        "full_response": response,  # Keep full response for analysis
        "ground_truth_docs": truth_docs,
        "response_docs": response_docs,
    }


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics from results."""
    if not results:
        return {}
    
    metrics = [
        "statute_citation_accuracy",
        "legal_terminology_accuracy", 
        "citation_format_compliance",
        "precedent_matching",
    ]
    
    summary = {
        "total_entries": len(results),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Overall metrics
    overall = {}
    for metric in metrics:
        values = [r[metric] for r in results if r[metric] >= 0]
        if values:
            overall[metric] = round(sum(values) / len(values), 3)
    
    overall["citation_rate"] = sum(1 for r in results if r.get("has_citation")) / len(results)
    summary["overall"] = overall
    
    # By source type
    by_source = {}
    for source_type in ["CPR", "PD", "Court Guide"]:
        entries = [r for r in results if r["source_type"] == source_type]
        if entries:
            by_source[source_type] = {
                "count": len(entries),
            }
            for metric in metrics:
                values = [r[metric] for r in entries if r[metric] >= 0]
                if values:
                    by_source[source_type][metric] = round(sum(values) / len(values), 3)
    summary["by_source_type"] = by_source
    
    # By category
    by_category = {}
    categories = set(r["category"] for r in results)
    for category in categories:
        entries = [r for r in results if r["category"] == category]
        if entries:
            by_category[category] = {
                "count": len(entries),
            }
            for metric in metrics:
                values = [r[metric] for r in entries if r[metric] >= 0]
                if values:
                    by_category[category][metric] = round(sum(values) / len(values), 3)
    summary["by_category"] = by_category
    
    return summary


async def run_evaluation(
    ground_truth_path: Path,
    max_entries: int = None,
    timeout_seconds: int = 120,
    output_path: Path | None = None,
    concurrency: int = 5,
) -> dict[str, Any]:
    """Run full evaluation against Azure services."""
    
    console.print("[bold blue]Starting Direct Evaluation[/bold blue]")
    console.print(f"Ground truth: {ground_truth_path}")
    
    # Load ground truth
    entries = load_ground_truth(ground_truth_path)
    if max_entries:
        entries = entries[:max_entries]
    console.print(f"Loaded {len(entries)} entries")
    
    # Initialize clients
    console.print("\n[yellow]Initializing Azure clients...[/yellow]")
    search_client, search_cred = await get_search_client()
    openai_client, deployment, openai_cred = await get_openai_client()
    console.print(f"[green]✓ Search client ready[/green]")
    console.print(f"[green]✓ OpenAI client ready (deployment: {deployment})[/green]")
    
    # Run evaluation
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Evaluating {len(entries)} entries...", total=len(entries))

        async def process_entry(i: int, entry: dict) -> dict[str, Any]:
            try:
                return await asyncio.wait_for(
                    evaluate_single_entry(search_client, openai_client, deployment, entry),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                console.print(f"[red]Timeout on entry {i} after {timeout_seconds}s[/red]")
                return {
                    "question": entry["question"][:50],
                    "source_type": entry.get("source_type", "Unknown"),
                    "category": entry.get("category", "Unknown"),
                    "error": f"timeout after {timeout_seconds}s",
                    "statute_citation_accuracy": -1,
                    "legal_terminology_accuracy": -1,
                    "citation_format_compliance": -1,
                    "precedent_matching": -1,
                    "has_citation": False,
                }
            except asyncio.CancelledError:
                console.print(f"[red]Cancelled on entry {i} after {timeout_seconds}s[/red]")
                return {
                    "question": entry["question"][:50],
                    "source_type": entry.get("source_type", "Unknown"),
                    "category": entry.get("category", "Unknown"),
                    "error": f"cancelled after {timeout_seconds}s",
                    "statute_citation_accuracy": -1,
                    "legal_terminology_accuracy": -1,
                    "citation_format_compliance": -1,
                    "precedent_matching": -1,
                    "has_citation": False,
                }
            except Exception as e:
                console.print(f"[red]Error on entry {i}: {e}[/red]")
                return {
                    "question": entry["question"][:50],
                    "source_type": entry.get("source_type", "Unknown"),
                    "category": entry.get("category", "Unknown"),
                    "error": str(e),
                    "statute_citation_accuracy": -1,
                    "legal_terminology_accuracy": -1,
                    "citation_format_compliance": -1,
                    "precedent_matching": -1,
                    "has_citation": False,
                }

        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def bounded_process(i: int, entry: dict) -> dict[str, Any]:
            async with semaphore:
                return await process_entry(i, entry)

        tasks = [asyncio.create_task(bounded_process(i, entry)) for i, entry in enumerate(entries)]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            progress.update(task, advance=1, description=f"Evaluated {completed}/{len(entries)}")
            if output_path:
                output_path.parent.mkdir(exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump({
                        "type": "direct_evaluation",
                        "summary": compute_summary(results),
                        "detailed_results": results,
                    }, f, indent=2)
    
    # Cleanup
    await search_cred.close()
    await openai_cred.close()
    await openai_client.close()
    await search_client.close()
    
    # Compute summary
    summary = compute_summary(results)
    
    return {
        "type": "direct_evaluation",
        "summary": summary,
        "detailed_results": results,
    }


def display_results(results: dict):
    """Display results in a formatted table."""
    summary = results["summary"]
    
    console.print("\n[bold green]═══ EVALUATION RESULTS ═══[/bold green]")
    console.print(f"Total entries: {summary['total_entries']}")
    console.print(f"Timestamp: {summary['timestamp']}")
    
    # Overall metrics table
    console.print("\n[bold]Overall Metrics:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_column("Status", justify="center")
    
    overall = summary["overall"]
    for metric, score in overall.items():
        if metric == "citation_rate":
            status = "✅" if score > 0.7 else "⚠️"
            table.add_row(metric, f"{score:.1%}", status)
        else:
            status = "✅" if score > 0.8 else ("⚠️" if score > 0.5 else "❌")
            table.add_row(metric, f"{score:.3f}", status)
    
    console.print(table)
    
    # By source type
    console.print("\n[bold]By Source Type:[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Source Type")
    table.add_column("Count", justify="right")
    table.add_column("Statute", justify="right")
    table.add_column("Terminology", justify="right")
    table.add_column("Precedent", justify="right")
    
    for source_type, data in summary.get("by_source_type", {}).items():
        table.add_row(
            source_type,
            str(data["count"]),
            f"{data.get('statute_citation_accuracy', 0):.3f}",
            f"{data.get('legal_terminology_accuracy', 0):.3f}",
            f"{data.get('precedent_matching', 0):.3f}",
        )
    
    console.print(table)
    
    # By category
    console.print("\n[bold]By Category:[/bold]")
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    table.add_column("Precedent", justify="right")
    
    for category, data in summary.get("by_category", {}).items():
        table.add_row(
            category[:40],
            str(data["count"]),
            f"{data.get('precedent_matching', 0):.3f}",
        )
    
    console.print(table)


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct Legal RAG Evaluation")
    parser.add_argument(
        "--ground-truth",
        default="ground_truth_cpr.jsonl",
        help="Ground truth file",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum entries to evaluate (for testing)",
    )
    parser.add_argument(
        "--output",
        default="results/direct_evaluation_results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Per-entry timeout in seconds",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent evaluations",
    )
    
    args = parser.parse_args()
    
    ground_truth_path = Path(__file__).parent / args.ground_truth
    if not ground_truth_path.exists():
        console.print(f"[red]Ground truth not found: {ground_truth_path}[/red]")
        sys.exit(1)
    
    output_path = Path(__file__).parent / args.output
    results = await run_evaluation(
        ground_truth_path,
        args.max_entries,
        args.timeout_seconds,
        output_path,
        args.concurrency,
    )
    # Save results
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to: {output_path}[/green]")
    
    # Display results
    display_results(results)


if __name__ == "__main__":
    asyncio.run(main())
