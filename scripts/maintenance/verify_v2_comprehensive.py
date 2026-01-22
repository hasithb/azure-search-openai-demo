"""
Comprehensive V2 Testing for legal-court-rag-index-v2

Tests critical functionality, content presence, and search quality for the new index.
"""
import sys
import os
from collections import Counter
from azure.core.credentials import AzureKeyCredential, AzureSasCredential
from azure.identity import DefaultAzureCredential, AzureDeveloperCliCredential
from azure.search.documents import SearchClient

# Configuration
INDEX_NAME = 'legal-court-rag-index-v2'
SERVICE_NAME = "cpr-rag"
ENDPOINT = f"https://{SERVICE_NAME}.search.windows.net"

def get_client():
    key = os.environ.get("AZURE_SEARCH_KEY")
    if key:
        credential = AzureKeyCredential(key)
        print(f"Using API key authentication")
    else:
        # Try AZD credential first as it's more reliable in this env
        try:
            credential = AzureDeveloperCliCredential()
            print(f"Using AzureDeveloperCliCredential")
        except:
            credential = DefaultAzureCredential()
            print(f"Using DefaultAzureCredential")
            
    return SearchClient(
        endpoint=ENDPOINT,
        index_name=INDEX_NAME,
        credential=credential
    )

class V2Tester:
    def __init__(self):
        self.client = get_client()
        self.results = {}
        
    def test_1_duplicate_check(self):
        """Test 1: Check for duplicate document IDs."""
        print("\n" + "="*80)
        print("TEST 1: Duplicate ID Check")
        print("="*80)
        
        all_ids = []
        # 'id' field is retrievable even if not filterable
        results = self.client.search('*', select=['id'])
        
        for result in results:
            all_ids.append(result['id'])
        
        total = len(all_ids)
        unique = len(set(all_ids))
        
        print(f"Total documents: {total}")
        print(f"Unique IDs: {unique}")
        
        if total > 0 and total == unique:
            print("✅ PASSED: No duplicates found")
            self.results['duplicate_check'] = 'PASS'
            return True
        else:
            duplicates = total - unique
            print(f"❌ FAILED: {duplicates} duplicate documents found (or 0 docs)")
            self.results['duplicate_check'] = f'FAIL - {duplicates} duplicates'
            return False
            
    def test_2_specific_files_presence(self):
        """Test 2: Verify specific critical files are present using sourcefile filter."""
        print("\n" + "="*80)
        print("TEST 2: Critical File Presence (Debt Protocol & PD 53B)")
        print("="*80)
        
        # Debt Protocol - we know this works
        print("Checking Debt Protocol...")
        debt_results = list(self.client.search(
            '*',
            filter="sourcefile eq 'Debt_Pap__PDF_.json'",
            top=1,
            select=['id', 'sourcefile']
        ))
        if debt_results:
            print(f"✅ Found Debt Protocol: {debt_results[0]['sourcefile']}")
        else:
            print(f"❌ Missing Debt Protocol (Debt_Pap__PDF_.json)")

        # PD 53B - Debugging the name mismatch
        print("\nChecking PD 53B...")
        # Search by content title to see what the sourcefile actually is
        pd_results = list(self.client.search(
            "\"Practice Direction 53B\"",
            top=1,
            select=['id', 'sourcefile', 'sourcepage']
        ))
        
        pd_found = False
        target_json = "Practice_Direction_53B___Media_And_Communications_Claims.json"
        
        if pd_results:
            actual_sourcefile = pd_results[0]['sourcefile']
            print(f"Found document matching 'Practice Direction 53B'. Sourcefile: '{actual_sourcefile}'")
            
            if actual_sourcefile == target_json:
                print(f"✅ Exact filename match.")
                pd_found = True
            else:
                print(f"⚠️ Filename mismatch. Expected: '{target_json}'")
                # Check if the snake_case file is present but maybe not the top result for that query?
                # Try explicit filter
                json_results = list(self.client.search(
                    '*',
                    filter=f"sourcefile eq '{target_json}'",
                    top=1
                ))
                if json_results:
                     print(f"✅ Found exact file via filter: {target_json}")
                     pd_found = True
                else:
                     print(f"❌ Could not find file with exact name '{target_json}' via filter.")
                     print("Hypothesis: The document in index might be from a different source or manually renamed?")
        else:
            print("❌ No document found for 'Practice Direction 53B'")
            
        if debt_results and pd_found:
            self.results['critical_files'] = 'PASS'
            return True
        else:
            self.results['critical_files'] = 'FAIL'
            return False

    def test_3_search_quality(self):
        """Test 3: Verify search returns relevant results for key legal concepts."""
        print("\n" + "="*80)
        print("TEST 3: Search Quality")
        print("="*80)
        
        # Tuple: (Query, Expected string in Sourcefile OR Sourcepage)
        queries = [
            ("Pre-Action Protocol for Debt Claims", "Debt_Pap"),
            ("Media and Communications Claims", "Practice"), # Keep loose
            ("Costs management", "Part")
        ]
        
        passed = 0
        
        for query, expected_snippet in queries:
            print(f"\nQuery: '{query}'")
            results = list(self.client.search(
                query,
                top=5,
                select=['sourcefile', 'sourcepage']
            ))
            
            found_relevant = False
            for r in results:
                print(f"  - {r['sourcefile']}")
                if expected_snippet.lower() in r['sourcefile'].lower():
                    found_relevant = True
            
            if found_relevant:
                print(f"✅ Found relevant result containing '{expected_snippet}'")
                passed += 1
            else:
                print(f"❌ Failed to find relevant result containing '{expected_snippet}'")
                
        if passed == len(queries):
            print(f"\n✅ PASSED: All search queries returned relevant results")
            self.results['search_quality'] = 'PASS'
            return True
        else:
            print(f"\n❌ FAILED: {len(queries) - passed} queries failed")
            self.results['search_quality'] = 'FAIL'
            return False

    def test_4_category_check(self):
        """Test 4: Verify category field is populated."""
        print("\n" + "="*80)
        print("TEST 4: Category Distribution")
        print("="*80)
        
        # Use facets to check categories
        results = self.client.search(
            "*",
            facets=["category"]
        )
        
        facets = results.get_facets()
        if facets and 'category' in facets:
            print("Categories found:")
            for cat in facets['category']:
                print(f"  - {cat['value']}: {cat['count']} docs")
            
            if len(facets['category']) > 0:
                print("\n✅ PASSED: Categories exist")
                self.results['categories'] = 'PASS'
                return True
        
        print("\n❌ FAILED: No categories found")
        self.results['categories'] = 'FAIL'
        return False

    def test_5_court_guide_compatibility(self):
        """Test 5: Verify Court Guides are present and structurally compatible."""
        print("\n" + "="*80)
        print("TEST 5: Court Guide Compatibility")
        print("="*80)
        
        # Categories often associated with Court Guides
        # Escape single quotes in category names
        guide_categories = [
            "Commercial Court",
            "King''s Bench Division", # Escape single quote for OData
            "Chancery Division",
            "Patents Court"
        ]
        
        passed_cats = 0
        
        for category in guide_categories:
            print(f"\nChecking Category: '{category}'")
            try:
                results = list(self.client.search(
                    "*",
                    filter=f"category eq '{category}'",
                    top=3,
                    select=['id', 'sourcefile', 'sourcepage', 'content']
                ))
                
                if results:
                    print(f"✅ Found {len(results)} docs in '{category}'")
                    for doc in results:
                        print(f"  - File: {doc['sourcefile']}")
                        print(f"    Page: {doc['sourcepage']}")
                        # Check for potential list serialization issues in content
                        if isinstance(doc['content'], list):
                            print("    ❌ ERROR: Content is a list, expected string!")
                        elif isinstance(doc['content'], str) and doc['content'].strip().startswith("[") and doc['content'].strip().endswith("]") and len(doc['content']) < 200:
                            print("    ⚠️ WARNING: Content looks like a serialized list repr")
                        else:
                            print(f"    Content Start: {doc['content'][:50]}...")
                    passed_cats += 1
                else:
                    print(f"⚠️ No documents found in category '{category}'")
            except Exception as e:
                print(f"❌ Error querying category '{category}': {e}")
        
        if passed_cats > 0:
            print(f"\n✅ PASSED: Found Court Guide content in {passed_cats} categories")
            self.results['court_guides'] = 'PASS'
            return True
        else:
            print(f"\n❌ FAILED: No Court Guide content found")
            self.results['court_guides'] = 'FAIL'
            return False

    def run(self):
        tests = [
            self.test_1_duplicate_check,
            self.test_2_specific_files_presence,
            self.test_3_search_quality,
            self.test_4_category_check,
            self.test_5_court_guide_compatibility
        ]
        
        success_count = 0
        for test in tests:
            try:
                if test():
                    success_count += 1
            except Exception as e:
                print(f"❌ Error running test: {e}")
                
        print("\n" + "="*80)
        print(f"SUMMARY: {success_count}/{len(tests)} tests passed")
        print("="*80)
        return success_count == len(tests)

if __name__ == "__main__":
    tester = V2Tester()
    success = tester.run()
    sys.exit(0 if success else 1)
