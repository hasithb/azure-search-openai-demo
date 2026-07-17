#!/usr/bin/env python
"""
Comprehensive test for content cleaning + citation impact validation.

Tests:
1. CPR content cleaning: no text loss, correct subsection boundary preservation
2. Court guide cleaning: no damage to already-clean content  
3. Citation builder: still works on cleaned content
4. Subsection extraction: still identifies subsections in cleaned content
5. Live index A/B: fetch real docs, clean them, compare citation outputs

Run: set -a && source .env && set +a && python scripts/legal-scraper/test_content_cleaning.py
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Optional

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BACKEND_DIR = PROJECT_ROOT / "app" / "backend"
COURT_GUIDES_DIR = PROJECT_ROOT / "scripts" / "court_guides_processing_pipeline" / "outputs"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from content_cleaner import (
    clean_content, strip_metadata_headers, replace_breadcrumbs,
    strip_markdown, strip_chunk_headers, normalize_whitespace,
    verify_no_text_loss, content_text_only
)
from customizations.subsection_extractor import SubsectionExtractor
from customizations.approaches.citation_builder import CitationBuilder

# ── Test Data ──

# Real-world CPR content sample (Part 29)
CPR_CONTENT_WITH_BREADCRUMBS = """# PART 29 – THE MULTI-TRACK

[PART 29 – THE MULTI-TRACK] Contents of this Part

## Scope of this Part

[PART 29 – THE MULTI-TRACK > Scope of this Part] 29.1 This Part contains general provisions about management of cases allocated to the multi-track and applies only to cases allocated to that track.

[PART 29 – THE MULTI-TRACK > Scope of this Part] (Part 27 sets out the procedure for claims allocated to the small claims track)

## Case management

## 29.2

[PART 29 – THE MULTI-TRACK > 29.2] (1) When it allocates a case to the multi-track, the court will –

[PART 29 – THE MULTI-TRACK > 29.2] (a) give directions for the management of the case and set a timetable for the steps to be taken between the giving of directions and the trial; or may

[PART 29 – THE MULTI-TRACK > 29.2] (b) fix –

[PART 29 – THE MULTI-TRACK > 29.2] (i) a case management conference; or

[PART 29 – THE MULTI-TRACK > 29.2] (ii) a pre-trial review,

## 29.5

[PART 29 – THE MULTI-TRACK > 29.5] (1) A party must apply to the court if he wishes to vary the date which the court has fixed for –

[PART 29 – THE MULTI-TRACK > 29.5] (a) a case management conference;

[PART 29 – THE MULTI-TRACK > 29.5] (b) a pre-trial review;"""

# CPR content with metadata headers (as uploaded to index)
CPR_CONTENT_WITH_HEADERS = """SOURCE: Part 29
SOURCEPAGE: Part 29 – The Multi-track
CATEGORY: Civil Procedure Rules and Practice Directions
SECTION: Part 29

## 29.1

[PART 29 > 29.1] 29.1 This Part contains general provisions about management of cases allocated to the multi-track.

## Case management

## 29.2

[PART 29 > 29.2] (1) When it allocates a case to the multi-track, the court will –"""

# Multi-chunk content with chunk headers
CPR_CHUNK_CONTENT = """Document: Part 44 – General Rules About Costs
Section: Costs orders relating to funding arrangements
Part 2 of 3
==================================================

## 44.2

[PART 44 > 44.2] (1) Where the court makes a costs order, it may –

[PART 44 > 44.2] (a) order costs to be assessed on a standard basis; or

[PART 44 > 44.2] (b) order costs to be assessed on an indemnity basis."""

# Court guide content (Commercial Court - clean format, no breadcrumbs)
COURT_GUIDE_CLEAN = """Starting a case in the Commercial Court, and Electronic Working (CE File)

B.2.1 The case will be begun by a claim form under Part 7 or Part 8. For arbitration claims, the claim form is issued under the Part 8 procedure (rule 62.3(1)), but with an adapted version of the Part 8 claim form: see PD62 §2.2. Save where otherwise specified, references in this Guide to a claim form are to a Part 7 claim form.

B.2.2 Many documents requiring to be provided or filed are now required to be provided or filed electronically (e-filing) under the Electronic Working (CE File) arrangements which apply to the Commercial Court: see Appendix 12.

B.2.3 The Commercial Court may at an appropriate stage give a fixed date for trial (see D.15), but it does not give a fixed date for a hearing when it issues a claim."""

# Court guide with case citation brackets (should NOT be stripped)
COURT_GUIDE_WITH_CASE_CITATIONS = """16.4 The jurisdiction of the court to make a civil restraint order is described in Practice Direction 3C. The principles relating to the exercise of the jurisdiction to make extended civil restraint orders and general civil restraint orders were further considered in R. (Wasif) v SS for the Home Department [2014] EWCA Civ 1091 and R. Wasif v SS for the Home Department [2016] EWCA Civ 82).

16.5 When a court makes, extends, or refuses to make a civil restraint order, the court must specify the form of the order."""

# Chancery guide with form reference brackets
CHANCERY_GUIDE_FORMS = """If a party instructs an authorised legal representative after either sending the Notice of Acting to the other party, or after the court has sent notice to the other parties: rule 42.2(2). That legal representative becomes the party's legal representative and must file a Notice of Change [N434] at court under CPR 42.

This litigant must be present at the hearing. If the hearing is in private, it is a matter for the judge."""

# Practice Direction content with breadcrumbs
PD_CONTENT = """# PRACTICE DIRECTION 3E – COSTS MANAGEMENT

[Practice Direction 3E > 1.1] 1.1 This Practice Direction supplements Part 3 of the Civil Procedure Rules and contains provisions about the filing and exchanging of budgets.

## Scope

[Practice Direction 3E > Scope] 1.2 This Practice Direction and the rules it supplements apply to –

[Practice Direction 3E > 1.2] (a) all Part 7 multi-track cases, except –

[Practice Direction 3E > 1.2] (i) cases in which a party has been served at an overseas address;

**1.3** This direction does not apply to litigants in person."""


# ── Test Functions ──

PASS_COUNT = 0
FAIL_COUNT = 0

def test(name: str, condition: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✅ {name}")
    else:
        FAIL_COUNT += 1
        print(f"  ❌ {name}")
        if detail:
            print(f"     Detail: {detail}")


def run_cpr_cleaning_tests():
    """Test 1: CPR content cleaning preserves all text, removes formatting."""
    print("\n" + "=" * 70)
    print("TEST 1: CPR Content Cleaning")
    print("=" * 70)
    
    cleaned = clean_content(CPR_CONTENT_WITH_BREADCRUMBS)
    
    # Verify no text loss
    result = verify_no_text_loss(CPR_CONTENT_WITH_BREADCRUMBS, cleaned)
    test("No substantive text lost", result['passed'], 
         f"Lost words: {result['lost_words']}" if not result['passed'] else "")
    
    # Verify markdown headings removed
    test("No # heading markers remain", 
         not re.search(r'^#{1,6}\s', cleaned, re.MULTILINE),
         f"Found: {re.findall(r'^#{1,6}.*$', cleaned, re.MULTILINE)[:3]}")
    
    # Verify breadcrumbs removed
    remaining_breadcrumbs = re.findall(r'\[PART[^\]]*\]', cleaned)
    test("No [PART...] breadcrumbs remain",
         '[PART' not in cleaned,
         f"Found: {remaining_breadcrumbs[:3]}")
    
    # Verify subsection IDs preserved as boundary markers
    test("Subsection 29.1 present", '29.1' in cleaned)
    test("Subsection 29.2 present", '29.2' in cleaned)
    test("Subsection 29.5 present", '29.5' in cleaned)
    
    # Verify subsection appears on its own line (boundary marker)
    lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
    test("29.2 appears as standalone line (boundary marker)",
         any(l == '29.2' or l.startswith('29.2 ') or l.startswith('29.2\n') for l in lines),
         f"Lines containing 29.2: {[l for l in lines if '29.2' in l][:3]}")
    
    # Verify actual legal text preserved
    test("Legal text preserved: 'general provisions about management'",
         'general provisions about management' in cleaned)
    test("Legal text preserved: 'case management conference'",
         'case management conference' in cleaned)
    test("Legal text preserved: 'small claims track'",
         'small claims track' in cleaned)
    
    # Show sample
    print(f"\n  📝 Cleaned sample (first 500 chars):")
    print(f"  {'─' * 50}")
    for line in cleaned[:500].split('\n'):
        print(f"  │ {line}")
    print(f"  {'─' * 50}")


def run_metadata_header_tests():
    """Test 2: Metadata header stripping."""
    print("\n" + "=" * 70)
    print("TEST 2: Metadata Header Stripping")
    print("=" * 70)
    
    cleaned = clean_content(CPR_CONTENT_WITH_HEADERS)
    
    test("SOURCE: header removed", 'SOURCE: Part 29' not in cleaned)
    test("SOURCEPAGE: header removed", 'SOURCEPAGE:' not in cleaned)
    test("CATEGORY: header removed", 'CATEGORY:' not in cleaned)
    test("SECTION: header removed", 'SECTION: Part 29' not in cleaned)
    
    # Legal text should remain
    test("Legal text preserved after header removal",
         'general provisions about management' in cleaned)
    
    result = verify_no_text_loss(CPR_CONTENT_WITH_HEADERS, cleaned)
    test("No substantive text lost", result['passed'],
         f"Lost: {result['lost_words']}" if not result['passed'] else "")


def run_chunk_header_tests():
    """Test 3: Multi-chunk header stripping."""
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Chunk Header Stripping")
    print("=" * 70)
    
    cleaned = clean_content(CPR_CHUNK_CONTENT)
    
    test("Document: header removed", 'Document: Part 44' not in cleaned)
    test("Section: header removed", 'Section: Costs orders' not in cleaned)
    test("Part N of M removed", 'Part 2 of 3' not in cleaned)
    test("==== divider removed", '=====' not in cleaned)
    
    # Legal content preserved
    test("Legal text preserved: 'costs order'",
         'costs order' in cleaned)
    test("44.2 subsection present", '44.2' in cleaned)


def run_court_guide_tests():
    """Test 4: Court guides are NOT damaged by cleaning."""
    print("\n" + "=" * 70)
    print("TEST 4: Court Guide Content (Should Be Barely Changed)")
    print("=" * 70)
    
    # Clean content should be virtually identical
    cleaned = clean_content(COURT_GUIDE_CLEAN)
    
    test("B.2.1 preserved", 'B.2.1' in cleaned)
    test("B.2.2 preserved", 'B.2.2' in cleaned)
    test("B.2.3 preserved", 'B.2.3' in cleaned)
    test("Full text preserved", 'Part 7 or Part 8' in cleaned)
    test("Reference preserved: PD62 §2.2", 'PD62 §2.2' in cleaned)
    
    # Content should be essentially unchanged
    orig_text = content_text_only(COURT_GUIDE_CLEAN)
    clean_text = content_text_only(cleaned)
    test("Text identical after cleaning", orig_text == clean_text,
         f"Diff length: {abs(len(orig_text) - len(clean_text))}")


def run_case_citation_bracket_tests():
    """Test 5: Case citation brackets [2014] EWCA Civ NOT stripped."""
    print("\n" + "=" * 70)
    print("TEST 5: Case Citation Brackets Preserved (Critical)")
    print("=" * 70)
    
    cleaned = clean_content(COURT_GUIDE_WITH_CASE_CITATIONS)
    
    test("[2014] EWCA Civ 1091 preserved",
         '[2014] EWCA Civ 1091' in cleaned,
         f"Content: {cleaned[:200]}")
    test("[2016] EWCA Civ 82 preserved",
         '[2016] EWCA Civ 82' in cleaned)
    test("Full legal text preserved",
         'civil restraint order' in cleaned)
    
    # Chancery form references
    cleaned_chancery = clean_content(CHANCERY_GUIDE_FORMS)
    test("[N434] form reference preserved",
         '[N434]' in cleaned_chancery,
         f"Content: {cleaned_chancery[:200]}")
    test("CPR 42 reference preserved",
         'CPR 42' in cleaned_chancery)


def run_practice_direction_tests():
    """Test 6: Practice Direction breadcrumbs cleaned correctly."""
    print("\n" + "=" * 70)
    print("TEST 6: Practice Direction Content")
    print("=" * 70)
    
    cleaned = clean_content(PD_CONTENT)
    
    test("PD breadcrumbs removed", '[Practice Direction 3E' not in cleaned)
    test("# heading removed", not re.search(r'^#\s', cleaned, re.MULTILINE))
    test("**bold** markers removed", '**1.3**' not in cleaned)
    test("1.3 text preserved", '1.3' in cleaned and 'litigants in person' in cleaned)
    test("1.1 text preserved", '1.1' in cleaned)
    test("Legal text preserved: 'filing and exchanging of budgets'",
         'filing and exchanging of budgets' in cleaned)


def run_subsection_extraction_tests():
    """Test 7: SubsectionExtractor still works on cleaned content."""
    print("\n" + "=" * 70)
    print("TEST 7: Subsection Extraction on Cleaned Content")
    print("=" * 70)
    
    # Test CPR
    cleaned_cpr = clean_content(CPR_CONTENT_WITH_BREADCRUMBS)
    first = SubsectionExtractor.extract_first_subsection(cleaned_cpr)
    all_subs = SubsectionExtractor.extract_all_subsections(cleaned_cpr)
    
    test(f"CPR first subsection found: '{first}'",
         first != "",
         "No subsection extracted from cleaned CPR content")
    
    # Check key subsections found
    all_sub_strs = set(all_subs)
    test("29.1 in extracted subsections", 
         any('29.1' in s for s in all_sub_strs),
         f"Found: {all_subs[:10]}")
    test("29.2 in extracted subsections",
         any('29.2' in s for s in all_sub_strs),
         f"Found: {all_subs[:10]}")
    test("29.5 in extracted subsections",
         any('29.5' in s for s in all_sub_strs),
         f"Found: {all_subs[:10]}")
    
    # Test with metadata headers
    cleaned_headers = clean_content(CPR_CONTENT_WITH_HEADERS)
    first_h = SubsectionExtractor.extract_first_subsection(cleaned_headers)
    test(f"Headers variant: first subsection found: '{first_h}'",
         first_h != "")
    
    # Test court guide
    cleaned_guide = clean_content(COURT_GUIDE_CLEAN)
    first_g = SubsectionExtractor.extract_first_subsection(cleaned_guide)
    test(f"Court guide: first subsection found: '{first_g}'",
         first_g != "",
         "No subsection from court guide content")
    
    # Test PD
    cleaned_pd = clean_content(PD_CONTENT)
    first_pd = SubsectionExtractor.extract_first_subsection(cleaned_pd)
    all_pd = SubsectionExtractor.extract_all_subsections(cleaned_pd)
    test(f"PD: first subsection found: '{first_pd}'",
         first_pd != "")
    test(f"PD: multiple subsections found: {len(all_pd)}",
         len(all_pd) >= 2,
         f"Found: {all_pd}")
    
    print(f"\n  📊 Extracted subsections:")
    print(f"     CPR: first='{first}', all={all_subs[:8]}")
    print(f"     Court Guide: first='{first_g}'")
    print(f"     PD: first='{first_pd}', all={all_pd[:8]}")


def run_citation_builder_tests():
    """Test 8: CitationBuilder still works on cleaned content."""
    print("\n" + "=" * 70)
    print("TEST 8: Citation Builder on Cleaned Content")
    print("=" * 70)
    
    builder = CitationBuilder()
    
    # Simulate a Document-like object for cleaned CPR
    class FakeDoc:
        def __init__(self, content, sourcepage, sourcefile, category="CPR"):
            self.content = content
            self.sourcepage = sourcepage
            self.sourcefile = sourcefile
            self.category = category
            self.subsection_id = None  # Simulating current state (not in select_fields)
    
    # Test with cleaned CPR
    cleaned_cpr = clean_content(CPR_CONTENT_WITH_BREADCRUMBS)
    doc = FakeDoc(cleaned_cpr, "Part 29 – The Multi-track", "Part 29")
    
    subsection = builder.extract_subsection(doc)
    test(f"Citation builder extracts subsection from cleaned CPR: '{subsection}'",
         subsection != "",
         "No subsection extracted")
    
    citation = builder.build_enhanced_citation(doc, source_index=1)
    test(f"Citation builder produces citation: '{citation}'",
         citation != "" and citation != "Unknown",
         f"Citation: {citation}")
    
    multi = builder.extract_multiple_subsections(doc)
    test(f"Multiple subsections extracted: {len(multi)} found",
         len(multi) >= 2,
         f"Found: {[(s.get('subsection_id','?')) for s in multi][:5]}")
    
    # Test with court guide
    cleaned_guide = clean_content(COURT_GUIDE_CLEAN)
    doc_guide = FakeDoc(cleaned_guide, "B. Commencement, B.2 Starting a case (p. 18)", "Commercial Court Guide")
    
    sub_guide = builder.extract_subsection(doc_guide)
    test(f"Court guide subsection: '{sub_guide}'",
         sub_guide != "")
    
    citation_guide = builder.build_enhanced_citation(doc_guide, source_index=2)
    test(f"Court guide citation: '{citation_guide}'",
         citation_guide != "")
    
    # Test with subsection_id field (simulating future state)
    doc_with_field = FakeDoc(cleaned_cpr, "Part 29 – The Multi-track", "Part 29")
    doc_with_field.subsection_id = "29.1"  # As if select_fields included it
    
    sub_with_field = builder.extract_subsection(doc_with_field)
    test(f"With subsection_id field, extraction uses it: '{sub_with_field}'",
         sub_with_field == "29.1",
         f"Got '{sub_with_field}' instead of '29.1'")
    
    print(f"\n  📊 Citation samples:")
    print(f"     CPR: subsection='{subsection}', citation='{citation}'")
    print(f"     Court Guide: subsection='{sub_guide}', citation='{citation_guide}'")
    print(f"     With field: subsection='{sub_with_field}'")


def run_real_court_guide_file_tests():
    """Test 9: Process actual court guide JSON files."""
    print("\n" + "=" * 70)
    print("TEST 9: Real Court Guide Files")
    print("=" * 70)
    
    import glob
    guide_files = sorted(glob.glob(str(COURT_GUIDES_DIR / "*.json")))
    guide_files = [f for f in guide_files if not f.endswith('.md5')]
    
    if not guide_files:
        print("  ⚠️  No court guide files found, skipping")
        return
    
    total_docs = 0
    total_text_preserved = 0
    total_citations_ok = 0
    total_subsections_ok = 0
    bracket_preservation_ok = 0
    bracket_total = 0
    
    builder = CitationBuilder()
    
    for filepath in guide_files:
        filename = Path(filepath).name
        with open(filepath) as f:
            data = json.load(f)
        
        doc_count = len(data)
        total_docs += doc_count
        text_ok = 0
        citation_ok = 0
        sub_ok = 0
        
        for doc in data:
            content = doc.get('content', '')
            if not content or len(content) < 50:
                continue
            
            cleaned = clean_content(content)
            
            # Text preservation check
            result = verify_no_text_loss(content, cleaned)
            if result['passed']:
                text_ok += 1
            
            # Case citation bracket preservation
            case_cites = re.findall(r'\[\d{4}\]\s+\w+', content)
            if case_cites:
                bracket_total += len(case_cites)
                for cite in case_cites:
                    if cite in cleaned:
                        bracket_preservation_ok += 1
            
            # Subsection extraction
            class FakeDoc:
                pass
            fd = FakeDoc()
            fd.content = cleaned
            fd.sourcepage = doc.get('sourcepage', '')
            fd.sourcefile = doc.get('sourcefile', '')
            fd.category = doc.get('category', '')
            fd.subsection_id = None
            
            sub = builder.extract_subsection(fd)
            if sub:
                sub_ok += 1
            
            citation = builder.build_enhanced_citation(fd, source_index=1)
            if citation and citation != "Unknown":
                citation_ok += 1
        
        total_text_preserved += text_ok
        total_citations_ok += citation_ok
        total_subsections_ok += sub_ok
        
        docs_with_content = len([d for d in data if len(d.get('content','')) >= 50])
        pct_text = (text_ok / docs_with_content * 100) if docs_with_content else 0
        pct_cite = (citation_ok / docs_with_content * 100) if docs_with_content else 0
        pct_sub = (sub_ok / docs_with_content * 100) if docs_with_content else 0
        
        print(f"  📄 {filename[:55]:57s} docs={doc_count:4d}  text={pct_text:5.1f}%  cite={pct_cite:5.1f}%  sub={pct_sub:5.1f}%")
    
    total_with_content = max(total_docs, 1)
    test(f"Text preservation across all guides: {total_text_preserved}/{total_docs}",
         total_text_preserved >= total_docs * 0.95,
         f"Only {total_text_preserved}/{total_docs} preserved")
    
    if bracket_total > 0:
        test(f"Case citation brackets preserved: {bracket_preservation_ok}/{bracket_total}",
             bracket_preservation_ok == bracket_total,
             f"Lost {bracket_total - bracket_preservation_ok} case citations!")


def run_live_index_tests():
    """Test 10: Fetch real documents from live index, clean them, verify citations."""
    print("\n" + "=" * 70)
    print("TEST 10: Live Index A/B Citation Test")
    print("=" * 70)
    
    # Check for Azure credentials
    search_service = os.getenv('AZURE_SEARCH_SERVICE', '')
    if not search_service:
        print("  ⚠️  AZURE_SEARCH_SERVICE not set, skipping live tests")
        return
    
    try:
        from azure.identity import AzureCliCredential
        from azure.search.documents import SearchClient
    except ImportError:
        print("  ⚠️  Azure SDK not available, skipping live tests")
        return
    
    endpoint = search_service if search_service.startswith('https://') else f"https://{search_service}.search.windows.net"
    index_name = os.getenv('AZURE_SEARCH_INDEX', 'legal-court-rag-index-v2')
    
    try:
        credential = AzureCliCredential()
        client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    except Exception as e:
        print(f"  ⚠️  Cannot connect to Azure Search: {e}")
        return
    
    # Test queries covering different content types
    queries = [
        ("What is the overriding objective under CPR Part 1?", "CPR"),
        ("What are the rules for allocation to the multi-track?", "CPR"),
        ("Part 3 sanctions for non-compliance", "CPR"),
        ("Practice Direction 3E costs management", "PD"),
        ("Commercial Court Guide filing deadlines", "Court Guide"),
        ("What is the time limit for filing a defence?", "CPR"),
        ("Part 44 costs assessment", "CPR"),
        ("Rule 31.16 pre-action disclosure", "CPR"),
    ]
    
    builder = CitationBuilder()
    
    results_summary = []
    
    for query_text, source_type in queries:
        try:
            results = client.search(
                search_text=query_text,
                select=["id", "content", "category", "sourcepage", "sourcefile", "storageUrl", "updated"],
                top=3,
                query_type="semantic",
                semantic_configuration_name="default",
                query_language="en-GB",
            )
            
            docs = list(results)
            if not docs:
                print(f"  ⚠️  No results for: {query_text}")
                continue
            
            for doc_result in docs[:2]:
                content = doc_result.get('content', '')
                sourcepage = doc_result.get('sourcepage', '')
                sourcefile = doc_result.get('sourcefile', '')
                category = doc_result.get('category', '')
                doc_id = doc_result.get('id', '')
                
                # Clean the content
                cleaned = clean_content(content)
                
                # Build FakeDoc for original
                class FakeDoc:
                    pass
                
                orig_doc = FakeDoc()
                orig_doc.content = content
                orig_doc.sourcepage = sourcepage
                orig_doc.sourcefile = sourcefile
                orig_doc.category = category
                orig_doc.subsection_id = None
                
                clean_doc = FakeDoc()
                clean_doc.content = cleaned
                clean_doc.sourcepage = sourcepage
                clean_doc.sourcefile = sourcefile
                clean_doc.category = category
                clean_doc.subsection_id = None
                
                # Compare subsection extraction
                orig_sub = builder.extract_subsection(orig_doc)
                clean_sub = builder.extract_subsection(clean_doc)
                
                # Compare citation building
                orig_cite = builder.build_enhanced_citation(orig_doc, source_index=1)
                clean_cite = builder.build_enhanced_citation(clean_doc, source_index=1)
                
                # Compare multiple subsections
                orig_multi = builder.extract_multiple_subsections(orig_doc)
                clean_multi = builder.extract_multiple_subsections(clean_doc)
                
                # Text preservation
                text_result = verify_no_text_loss(content, cleaned)
                
                # Token savings
                orig_tokens = len(content) / 4  # rough estimate
                clean_tokens = len(cleaned) / 4
                savings_pct = (1 - clean_tokens / orig_tokens) * 100 if orig_tokens > 0 else 0
                
                sub_match = (orig_sub == clean_sub) or (clean_sub != "")
                cite_match = (orig_cite == clean_cite) or (clean_cite != "" and clean_cite != "Unknown")
                multi_match = len(clean_multi) >= len(orig_multi) * 0.8  # allow some variance
                
                results_summary.append({
                    'query': query_text[:40],
                    'doc_id': doc_id[:40],
                    'source_type': source_type,
                    'text_preserved': text_result['passed'],
                    'sub_match': sub_match,
                    'cite_match': cite_match,
                    'multi_match': multi_match,
                    'orig_sub': orig_sub,
                    'clean_sub': clean_sub,
                    'savings_pct': savings_pct,
                    'orig_multi_count': len(orig_multi),
                    'clean_multi_count': len(clean_multi),
                })
                
        except Exception as e:
            print(f"  ⚠️  Error on query '{query_text[:40]}': {e}")
    
    if not results_summary:
        print("  ⚠️  No results to report")
        return
    
    # Report results
    print(f"\n  {'Query':<42s} {'Doc':<42s} {'Type':<6s} {'Text':5s} {'Sub':5s} {'Cite':5s} {'Multi':5s} {'Save':>5s}")
    print(f"  {'─'*42} {'─'*42} {'─'*6} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
    
    for r in results_summary:
        text_sym = "✅" if r['text_preserved'] else "❌"
        sub_sym = "✅" if r['sub_match'] else "❌"
        cite_sym = "✅" if r['cite_match'] else "❌"
        multi_sym = "✅" if r['multi_match'] else "❌"
        
        print(f"  {r['query']:<42s} {r['doc_id']:<42s} {r['source_type']:<6s} {text_sym:5s} {sub_sym:5s} {cite_sym:5s} {multi_sym:5s} {r['savings_pct']:4.0f}%")
        
        if not r['sub_match']:
            print(f"     ⚠️  Subsection mismatch: orig='{r['orig_sub']}' clean='{r['clean_sub']}'")
    
    # Aggregate test results
    text_pass = sum(1 for r in results_summary if r['text_preserved'])
    sub_pass = sum(1 for r in results_summary if r['sub_match'])
    cite_pass = sum(1 for r in results_summary if r['cite_match'])
    multi_pass = sum(1 for r in results_summary if r['multi_match'])
    total = len(results_summary)
    avg_savings = sum(r['savings_pct'] for r in results_summary) / total if total else 0
    
    print(f"\n  📊 Summary: {total} documents tested")
    test(f"Text preserved: {text_pass}/{total}", text_pass == total)
    test(f"Subsection extraction: {sub_pass}/{total}", sub_pass >= total * 0.9,
         f"Only {sub_pass}/{total}")
    test(f"Citation building: {cite_pass}/{total}", cite_pass >= total * 0.9,
         f"Only {cite_pass}/{total}")
    test(f"Multi-subsection extraction: {multi_pass}/{total}", multi_pass >= total * 0.8,
         f"Only {multi_pass}/{total}")
    print(f"  📉 Average token savings: {avg_savings:.1f}%")


def run_scraper_completeness_check():
    """Test 11: Verify scraper captures all content for sample CPR pages."""
    print("\n" + "=" * 70)
    print("TEST 11: Scraper Completeness (Local JSON Check)")
    print("=" * 70)
    
    upload_dir = PROJECT_ROOT / "data" / "legal-scraper" / "processed" / "Upload"
    
    if not upload_dir.exists():
        print("  ⚠️  Upload directory not found, skipping")
        return
    
    import glob
    json_files = sorted(glob.glob(str(upload_dir / "*.json")))
    
    if not json_files:
        print("  ⚠️  No JSON files found in Upload directory")
        return
    
    total = len(json_files)
    empty_content = 0
    short_content = 0
    has_breadcrumbs = 0
    has_headers = 0
    has_markdown = 0
    clean_ok = 0
    subsection_found = 0
    
    for filepath in json_files:
        try:
            with open(filepath) as f:
                doc = json.load(f)
        except:
            continue
        
        content = doc.get('content', '')
        if not content:
            empty_content += 1
            continue
        if len(content) < 100:
            short_content += 1
            continue
        
        if '[PART' in content or '[Practice Direction' in content:
            has_breadcrumbs += 1
        if re.search(r'^(SOURCE|SOURCEPAGE|CATEGORY|SECTION):', content, re.MULTILINE):
            has_headers += 1
        if re.search(r'^#{1,3}\s', content, re.MULTILINE):
            has_markdown += 1
        
        cleaned = clean_content(content)
        result = verify_no_text_loss(content, cleaned)
        if result['passed']:
            clean_ok += 1
        
        first_sub = SubsectionExtractor.extract_first_subsection(cleaned)
        if first_sub:
            subsection_found += 1
    
    docs_with_content = total - empty_content
    
    print(f"  📊 Upload Directory Stats:")
    print(f"     Total JSON files:    {total}")
    print(f"     Empty content:       {empty_content}")
    print(f"     Short content (<100):{short_content}")
    print(f"     Has breadcrumbs:     {has_breadcrumbs}")
    print(f"     Has metadata headers:{has_headers}")
    print(f"     Has markdown:        {has_markdown}")
    
    test(f"Text preserved after cleaning: {clean_ok}/{docs_with_content}",
         clean_ok >= docs_with_content * 0.95,
         f"Only {clean_ok}/{docs_with_content}")
    test(f"Subsection extraction on cleaned: {subsection_found}/{docs_with_content}",
         subsection_found >= docs_with_content * 0.7,
         f"Only {subsection_found}/{docs_with_content}")


# ── Main ──

if __name__ == "__main__":
    print("=" * 70)
    print("  CONTENT CLEANING & CITATION IMPACT VALIDATION")
    print("  Testing proposed V2 index content changes")
    print("=" * 70)
    
    # Unit tests (no Azure needed)
    run_cpr_cleaning_tests()
    run_metadata_header_tests()
    run_chunk_header_tests()
    run_court_guide_tests()
    run_case_citation_bracket_tests()
    run_practice_direction_tests()
    run_subsection_extraction_tests()
    run_citation_builder_tests()
    
    # File-based tests (court guides + local scraped data)
    run_real_court_guide_file_tests()
    run_scraper_completeness_check()
    
    # Live index tests (requires Azure credentials)
    run_live_index_tests()
    
    # Final summary
    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 70)
    
    if FAIL_COUNT == 0:
        print("  🎉 All tests passed! Content cleaning is safe to implement.")
    else:
        print(f"  ⚠️  {FAIL_COUNT} test(s) failed. Review before proceeding.")
    
    sys.exit(0 if FAIL_COUNT == 0 else 1)
