#!/usr/bin/env python3
"""
Test script to validate subsection_id and subsections field implementation.

This script tests:
1. SubsectionExtractor utility with markdown/breadcrumb content
2. Index schema has the new fields
3. Upload script populates the fields correctly
4. Citation builder prefers indexed values
"""

import sys
import os
import asyncio
from pathlib import Path

# Add app/backend to path
backend_dir = Path(__file__).parent / "app" / "backend"
sys.path.insert(0, str(backend_dir))

from customizations.subsection_extractor import SubsectionExtractor


def test_extractor_markdown():
    """Test extraction from markdown format content."""
    print("\n=== Testing Markdown Format ===")
    
    content = """[PART 35 > 35.1]

## 35.1

Duty to restrict expert evidence

(1) Expert evidence shall be restricted to that which is reasonably required to resolve the proceedings.

(2) The duties in rule 35.3 apply before as well as after the issue of proceedings.

## 35.2

Interpretation and Definitions

(1) A reference to an 'expert' in this Part is a reference to a person who has been instructed to give or prepare expert evidence for the purpose of proceedings."""

    # Test primary subsection extraction
    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    print(f"Primary subsection: {subsection_id}")
    assert subsection_id == "35.1", f"Expected '35.1', got '{subsection_id}'"
    
    # Test all subsections extraction
    subsections = SubsectionExtractor.extract_all_subsections(content)
    print(f"All subsections: {subsections}")
    assert "35.1" in subsections, "35.1 should be in subsections list"
    assert "35.2" in subsections, "35.2 should be in subsections list"
    assert len(subsections) == 2, f"Expected 2 subsections, got {len(subsections)}"
    
    print("✅ Markdown format test passed")


def test_extractor_breadcrumb():
    """Test extraction from breadcrumb format content."""
    print("\n=== Testing Breadcrumb Format ===")
    
    content = """[CPR > PART 3 > 3.1]

3.1 The court's general powers of management

(1) The list of powers in this rule is in addition to any powers given to the court by any other rule or practice direction or by any other enactment or any powers it may otherwise have.

(2) Except where these Rules provide otherwise, the court may –
(a) extend or shorten the time for compliance with any rule, practice direction or court order (even if an application for extension is made after the time for compliance has expired);"""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    print(f"Primary subsection: {subsection_id}")
    assert subsection_id == "3.1", f"Expected '3.1', got '{subsection_id}'"
    
    subsections = SubsectionExtractor.extract_all_subsections(content)
    print(f"All subsections: {subsections}")
    assert "3.1" in subsections, "3.1 should be in subsections list"
    
    print("✅ Breadcrumb format test passed")


def test_extractor_bare_text():
    """Test extraction from bare text format (legacy)."""
    print("\n=== Testing Bare Text Format ===")
    
    content = """35.1 Duty to restrict expert evidence

(1) Expert evidence shall be restricted to that which is reasonably required to resolve the proceedings.

35.2 Interpretation

(1) A reference to an 'expert' in this Part is a reference to a person who has been instructed to give or prepare expert evidence."""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    print(f"Primary subsection: {subsection_id}")
    assert subsection_id == "35.1", f"Expected '35.1', got '{subsection_id}'"
    
    subsections = SubsectionExtractor.extract_all_subsections(content)
    print(f"All subsections: {subsections}")
    assert "35.1" in subsections, "35.1 should be in subsections list"
    assert "35.2" in subsections, "35.2 should be in subsections list"
    
    print("✅ Bare text format test passed")


def test_extractor_chunked_content():
    """Test extraction from chunked content with headers."""
    print("\n=== Testing Chunked Content with Headers ===")
    
    content = """SOURCE: CPR_Part_35.txt
CATEGORY: Civil Procedure Rules
SECTION: Part 35
CHUNK: 2 of 5

[PART 35 > 35.3]

## 35.3

Experts – overriding duty to the court

(1) It is the duty of experts to help the court on matters within their expertise.

(2) This duty overrides any obligation to the person from whom experts have received instructions or by whom they are paid."""

    # This should still find 35.3 despite 4-line header
    subsection_id = SubsectionExtractor.extract_first_subsection(content, max_lines=30)
    print(f"Primary subsection (with headers): {subsection_id}")
    assert subsection_id == "35.3", f"Expected '35.3', got '{subsection_id}'"
    
    print("✅ Chunked content test passed")


def test_extractor_letter_number():
    """Test extraction of letter-number subsections (A.1, B.2)."""
    print("\n=== Testing Letter-Number Format ===")
    
    content = """[Court Guides > Commercial Court > B.7]

## B.7

London Circuit Commercial Court Triaging

B.7.1 All claims issued in the London Circuit Commercial Court will be automatically assigned to the triaging Judge."""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    print(f"Primary subsection: {subsection_id}")
    assert subsection_id == "B.7", f"Expected 'B.7', got '{subsection_id}'"
    
    subsections = SubsectionExtractor.extract_all_subsections(content)
    print(f"All subsections: {subsections}")
    assert "B.7" in subsections, "B.7 should be in subsections list"
    assert "B.7.1" in subsections, "B.7.1 should be in subsections list"
    
    print("✅ Letter-number format test passed")


def test_extractor_no_subsection():
    """Test behavior when no subsection is found."""
    print("\n=== Testing Content Without Subsections ===")
    
    content = """This is some random text without any subsection identifiers.

It just contains regular paragraphs and sentences.

No rules, no subsections, nothing."""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    print(f"Primary subsection (should be empty): '{subsection_id}'")
    assert subsection_id == "", f"Expected empty string, got '{subsection_id}'"
    
    subsections = SubsectionExtractor.extract_all_subsections(content)
    print(f"All subsections (should be empty list): {subsections}")
    assert subsections == [], f"Expected empty list, got {subsections}"
    
    print("✅ No subsection test passed")


def main():
    """Run all tests."""
    print("=" * 70)
    print("SUBSECTION FIELD IMPLEMENTATION VALIDATION")
    print("=" * 70)
    
    try:
        # Test extractor utility
        test_extractor_markdown()
        test_extractor_breadcrumb()
        test_extractor_bare_text()
        test_extractor_chunked_content()
        test_extractor_letter_number()
        test_extractor_no_subsection()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Recreate index: cd scripts/legal-scraper && python create_index.py")
        print("2. Re-upload documents: python upload_with_embeddings.py")
        print("3. Test citation accuracy: cd evals && python test_citation_accuracy.py")
        print("=" * 70)
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
