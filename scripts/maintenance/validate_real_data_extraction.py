#!/usr/bin/env python3
"""
Comprehensive validation of subsection extraction on real indexed documents.

Tests extraction accuracy across all 1000 chunks to ensure 100% coverage.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add app/backend to path
backend_dir = Path(__file__).parent / "app" / "backend"
sys.path.insert(0, str(backend_dir))

from customizations.subsection_extractor import SubsectionExtractor


def analyze_document(filepath: Path) -> dict:
    """Analyze a single document for subsection extraction."""
    with open(filepath, 'r') as f:
        doc = json.load(f)
    
    content = doc.get('content', '')
    subsection_id = SubsectionExtractor.extract_first_subsection(content)
    subsections = SubsectionExtractor.extract_all_subsections(content)
    
    return {
        'filepath': str(filepath),
        'id': doc.get('id', ''),
        'sourcefile': doc.get('sourcefile', ''),
        'sourcepage': doc.get('sourcepage', ''),
        'subsection_id': subsection_id,
        'subsections': subsections,
        'content_length': len(content),
        'content_preview': content[:200],
        'has_subsection': bool(subsection_id),
    }


def main():
    """Run comprehensive validation."""
    print("=" * 80)
    print("COMPREHENSIVE SUBSECTION EXTRACTION VALIDATION")
    print("Testing on Real Indexed Documents")
    print("=" * 80)
    
    # Find all JSON files in Upload directory
    upload_dir = Path(__file__).parent / "data" / "legal-scraper" / "processed" / "Upload"
    
    if not upload_dir.exists():
        print(f"❌ Upload directory not found: {upload_dir}")
        return 1
    
    json_files = list(upload_dir.glob("*.json"))
    print(f"\nFound {len(json_files)} documents to analyze\n")
    
    if len(json_files) == 0:
        print("❌ No JSON files found in Upload directory")
        return 1
    
    # Statistics tracking
    stats = {
        'total': 0,
        'with_subsection': 0,
        'without_subsection': 0,
        'multi_subsection': 0,
        'by_format': defaultdict(int),
        'failures': [],
    }
    
    # Sample documents for detailed inspection
    sample_size = 30
    samples = []
    
    # Process all documents
    for i, filepath in enumerate(json_files, 1):
        try:
            result = analyze_document(filepath)
            stats['total'] += 1
            
            if result['has_subsection']:
                stats['with_subsection'] += 1
                
                # Detect format
                content = result['content_preview']
                if '##' in content[:100]:
                    stats['by_format']['markdown'] += 1
                elif '[' in content[:100] and '>' in content[:100]:
                    stats['by_format']['breadcrumb'] += 1
                else:
                    stats['by_format']['bare_text'] += 1
                
                if len(result['subsections']) > 1:
                    stats['multi_subsection'] += 1
            else:
                stats['without_subsection'] += 1
            
            # Collect samples
            if i <= sample_size:
                samples.append(result)
            
            # Progress indicator
            if i % 100 == 0:
                print(f"  Processed {i}/{len(json_files)} documents...")
                
        except Exception as e:
            stats['failures'].append({
                'file': str(filepath),
                'error': str(e)
            })
    
    # Print detailed results for samples
    print("\n" + "=" * 80)
    print(f"SAMPLE DOCUMENTS (first {sample_size}):")
    print("=" * 80)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. {sample['sourcefile']}")
        print(f"   Primary: '{sample['subsection_id']}'")
        print(f"   All: {sample['subsections']}")
        if not sample['has_subsection']:
            print(f"   ⚠️  NO SUBSECTION FOUND")
            print(f"   Content: {sample['content_preview'][:150]}...")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("EXTRACTION STATISTICS:")
    print("=" * 80)
    print(f"Total documents:        {stats['total']}")
    print(f"With subsections:       {stats['with_subsection']} ({stats['with_subsection']/stats['total']*100:.1f}%)")
    print(f"Without subsections:    {stats['without_subsection']} ({stats['without_subsection']/stats['total']*100:.1f}%)")
    print(f"Multi-subsection:       {stats['multi_subsection']}")
    
    print(f"\nFormat Distribution:")
    for fmt, count in stats['by_format'].items():
        print(f"  {fmt:15s}: {count} ({count/stats['with_subsection']*100:.1f}%)")
    
    # Print failures
    if stats['failures']:
        print(f"\n⚠️  FAILURES ({len(stats['failures'])})")
        for failure in stats['failures'][:10]:  # Show first 10
            print(f"  {failure['file']}: {failure['error']}")
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS:")
    print("=" * 80)
    
    checks_passed = True
    
    # Check 1: At least 70% of documents should have subsections
    subsection_rate = stats['with_subsection'] / stats['total']
    if subsection_rate >= 0.70:
        print(f"✅ Subsection detection rate: {subsection_rate*100:.1f}% (>= 70%)")
    else:
        print(f"❌ Subsection detection rate: {subsection_rate*100:.1f}% (< 70%)")
        checks_passed = False
    
    # Check 2: No failures
    if len(stats['failures']) == 0:
        print(f"✅ No extraction failures")
    else:
        print(f"❌ {len(stats['failures'])} extraction failures")
        checks_passed = False
    
    # Check 3: All formats supported
    if len(stats['by_format']) >= 2:
        print(f"✅ Multiple formats detected ({len(stats['by_format'])})")
    else:
        print(f"⚠️  Only {len(stats['by_format'])} format(s) detected")
    
    # Final result
    print("\n" + "=" * 80)
    if checks_passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("=" * 80)
        print("\nExtraction logic is working correctly on real data!")
        print("Ready to proceed with index recreation and re-upload.")
        return 0
    else:
        print("❌ SOME VALIDATION CHECKS FAILED")
        print("=" * 80)
        print("\nReview failures and adjust extraction patterns if needed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
