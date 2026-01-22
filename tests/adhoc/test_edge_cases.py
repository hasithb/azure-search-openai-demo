#!/usr/bin/env python3
"""
Edge case tests for subsection extraction based on real data analysis.
"""

import sys
from pathlib import Path

# Add app/backend to path
backend_dir = Path(__file__).parent / "app" / "backend"
sys.path.insert(0, str(backend_dir))

from customizations.subsection_extractor import SubsectionExtractor


def test_practice_direction_23a():
    """Test extraction from Practice Direction 23A with numbered headings."""
    print("\n=== Practice Direction 23A (Numbered Headings) ===")
    
    content = """# PRACTICE DIRECTION 23A – APPLICATIONS

## This Practice Direction supplements CPR Part 23

[PRACTICE DIRECTION 23A – APPLICATIONS > This Practice Direction supplements CPR Part 23] Contents of this Practice Direction

## Referral to a different judge

[PRACTICE DIRECTION 23A – APPLICATIONS > Referral to a different judge] 1. Masters or District Judges may refer to a judge

## Application notices

[PRACTICE DIRECTION 23A – APPLICATIONS > Application notices] 2.1 An application notice must, in addition to the matters

## 2.5 Every application should be made as soon as it appears necessary or desirable to make it.

[PRACTICE DIRECTION 23A – APPLICATIONS > 2.5 Every application should be made] 2.6 Applications should wherever possible"""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    subsections = SubsectionExtractor.extract_all_subsections(content)
    
    print(f"Primary subsection: '{subsection_id}'")
    print(f"All subsections: {subsections}")
    
    # Should extract 2.1 or 2.5 as primary (first actual subsection, not the title)
    assert subsection_id in ["2.1", "2.5"], f"Expected '2.1' or '2.5', got '{subsection_id}'"
    assert "2.1" in subsections, "Should find 2.1"
    assert "2.5" in subsections, "Should find 2.5"
    assert "2.6" in subsections, "Should find 2.6"
    
    print("✅ Practice Direction 23A test passed")


def test_part_44_with_part_reference():
    """Test that we don't extract 'Part 1' as primary subsection."""
    print("\n=== Part 44 (Avoid 'Part X' as Primary) ===")
    
    content = """# PART 44

## This Part supplements Part 1

[PART 44 > Introduction] Part 1 contains the overriding objective.

## 44.1 Scope of this Part

[PART 44 > 44.1] (1) This Part sets out the general rules

## 44.2 Court's powers

[PART 44 > 44.2] The court has discretion"""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    subsections = SubsectionExtractor.extract_all_subsections(content)
    
    print(f"Primary subsection: '{subsection_id}'")
    print(f"All subsections: {subsections}")
    
    # Should extract 44.1, not 'Part 1'
    assert subsection_id == "44.1", f"Expected '44.1', got '{subsection_id}'"
    assert "44.1" in subsections, "Should find 44.1"
    assert "44.2" in subsections, "Should find 44.2"
    # Part 1 might be in the list but shouldn't be primary
    
    print("✅ Part 44 test passed")


def test_practice_direction_40b_no_numbered_sections():
    """Test document with only text headings, no numbered subsections."""
    print("\n=== Practice Direction 40B (Text Headings Only) ===")
    
    content = """# PRACTICE DIRECTION 40B – JUDGMENTS AND ORDERS

## This Practice Direction supplements CPR Part 40

[PRACTICE DIRECTION 40B – JUDGMENTS AND ORDERS > General] General provisions

## Drawing up and filing of judgments and orders

[PRACTICE DIRECTION 40B – JUDGMENTS AND ORDERS > Drawing up] 2.3 Every judgment or order must state"""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    subsections = SubsectionExtractor.extract_all_subsections(content)
    
    print(f"Primary subsection: '{subsection_id}'")
    print(f"All subsections: {subsections}")
    
    # Should extract 2.3 as first numbered subsection
    assert subsection_id == "2.3", f"Expected '2.3', got '{subsection_id}'"
    assert "2.3" in subsections, "Should find 2.3"
    
    print("✅ Practice Direction 40B test passed")


def test_heading_with_full_text():
    """Test markdown heading with full text after number."""
    print("\n=== Markdown Heading with Full Text ===")
    
    content = """## 5.2 A District Judge may—

[SECTION > 5.2] (a) consider the application without a hearing; or

[SECTION > 5.2] (b) direct that the application should be transferred

## 6.5 A case summary and draft order must be filed and served in –

[SECTION > 6.5] (a) multi-track cases; and"""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    subsections = SubsectionExtractor.extract_all_subsections(content)
    
    print(f"Primary subsection: '{subsection_id}'")
    print(f"All subsections: {subsections}")
    
    assert subsection_id == "5.2", f"Expected '5.2', got '{subsection_id}'"
    assert "5.2" in subsections, "Should find 5.2"
    assert "6.5" in subsections, "Should find 6.5"
    
    print("✅ Markdown heading with text test passed")


def test_footnotes_section():
    """Test that footnotes section doesn't extract as primary."""
    print("\n=== Footnotes Section ===")
    
    content = """## 44.13 Final subsection

[PART 44 > 44.13] (1) This is the last rule

## Footnotes

[PART 44 > Footnotes] 1976 c.36. Back to text

[PART 44 > Footnotes] 2002 c.38. Back to text"""

    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    subsections = SubsectionExtractor.extract_all_subsections(content)
    
    print(f"Primary subsection: '{subsection_id}'")
    print(f"All subsections: {subsections}")
    
    # Should extract 44.13, not a year from footnotes
    assert subsection_id == "44.13", f"Expected '44.13', got '{subsection_id}'"
    
    print("✅ Footnotes test passed")


def main():
    """Run all edge case tests."""
    print("=" * 70)
    print("EDGE CASE TESTS (Based on Real Data Issues)")
    print("=" * 70)
    
    try:
        test_practice_direction_23a()
        test_part_44_with_part_reference()
        test_practice_direction_40b_no_numbered_sections()
        test_heading_with_full_text()
        test_footnotes_section()
        
        print("\n" + "=" * 70)
        print("✅ ALL EDGE CASE TESTS PASSED")
        print("=" * 70)
        print("\nSubsection extraction handles all real-world document formats correctly!")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nNeed to improve extraction patterns...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
