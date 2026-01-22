"""
Comprehensive Phase 2 Testing for legal-court-rag-index-v2

Tests all critical functionality before production migration.
NO SHORTCUTS - Full validation of index quality and RAG performance.
"""
import sys
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.azure' / os.environ.get('AZURE_ENV_NAME', 'legal-rag') / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

sys.path.insert(0, 'scripts/legal-scraper')
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from config import Config
from collections import Counter
import json

class Phase2Tester:
    def __init__(self, index_name='legal-court-rag-index-v2'):
        self.index_name = index_name
        endpoint = f'https://{Config.AZURE_SEARCH_SERVICE}.search.windows.net'
        
        # Use same auth logic as upload script
        key = Config.AZURE_SEARCH_KEY
        if key:
            credential = AzureKeyCredential(key)
            print(f"Using API key authentication")
        else:
            credential = DefaultAzureCredential()
            print(f"Using DefaultAzureCredential authentication")
        
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential
        )
        self.results = {}
        
    def test_1_duplicate_check(self):
        """Test 1: Check for duplicate document IDs."""
        print("\n" + "="*80)
        print("TEST 1: Duplicate ID Check")
        print("="*80)
        
        all_ids = []
        results = self.client.search('*', select='id')
        
        for result in results:
            all_ids.append(result['id'])
        
        total = len(all_ids)
        unique = len(set(all_ids))
        
        print(f"Total documents: {total}")
        print(f"Unique IDs: {unique}")
        
        if total == unique:
            print("✅ PASSED: No duplicates found")
            self.results['duplicate_check'] = 'PASS'
            return True
        else:
            duplicates = total - unique
            print(f"❌ FAILED: {duplicates} duplicate documents found")
            
            id_counts = Counter(all_ids)
            dups = {k: v for k, v in id_counts.items() if v > 1}
            print(f"\nFirst 5 duplicates:")
            for id_val, count in list(dups.items())[:5]:
                print(f"  - {id_val}: {count} copies")
            
            self.results['duplicate_check'] = f'FAIL - {duplicates} duplicates'
            return False
    
    def test_2_filterable_id(self):
        """Test 2: Verify id field is filterable."""
        print("\n" + "="*80)
        print("TEST 2: Filterable ID Field")
        print("="*80)
        
        test_ids = [
            'Part_44___General_Rules_about_Costs',
            'Practice_Direction___Pre-Action_Conduct_and_Protocols',
            'Insolvency_Proceedings_chunk_001'
        ]
        
        passed = 0
        failed = 0
        
        for test_id in test_ids:
            try:
                results = list(self.client.search(
                    '*',
                    filter=f"id eq '{test_id}'",
                    top=1,
                    select=['id']
                ))
                if results and results[0]['id'] == test_id:
                    print(f"✅ Filter works: {test_id[:50]}...")
                    passed += 1
                else:
                    print(f"❌ Filter failed: {test_id[:50]}... (not found)")
                    failed += 1
            except Exception as e:
                print(f"❌ Filter error: {test_id[:50]}... - {str(e)[:100]}")
                failed += 1
        
        if failed == 0:
            print(f"\n✅ PASSED: All {passed} filter tests succeeded")
            self.results['filterable_id'] = 'PASS'
            return True
        else:
            print(f"\n❌ FAILED: {failed}/{len(test_ids)} filter tests failed")
            self.results['filterable_id'] = f'FAIL - {failed} failures'
            return False
    
    def test_3_document_structure(self):
        """Test 3: Validate document schema and required fields."""
        print("\n" + "="*80)
        print("TEST 3: Document Structure Validation")
        print("="*80)
        
        # Note: embedding field not included because Azure Search doesn't return vectors in search results
        required_fields = ['id', 'content', 'category', 'sourcepage', 'sourcefile']
        sample_size = 10
        
        results = list(self.client.search('*', top=sample_size))
        
        if len(results) == 0:
            print("❌ FAILED: No documents found in index")
            self.results['document_structure'] = 'FAIL - empty index'
            return False
        
        print(f"Checking {len(results)} sample documents...")
        
        issues = []
        for i, doc in enumerate(results):
            for field in required_fields:
                if field not in doc:
                    issues.append(f"Doc {i}: Missing field '{field}'")
                elif field == 'content' and not doc[field]:
                    issues.append(f"Doc {i}: Empty content")
        
        if not issues:
            print(f"✅ PASSED: All {len(results)} documents have correct structure")
            print(f"  - All required fields present")
            print(f"  - Content non-empty")
            self.results['document_structure'] = 'PASS'
            return True
        else:
            print(f"❌ FAILED: {len(issues)} structural issues found:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
            self.results['document_structure'] = f'FAIL - {len(issues)} issues'
            return False
    
    def test_4_search_quality(self):
        """Test 4: Verify search returns relevant results."""
        print("\n" + "="*80)
        print("TEST 4: Search Quality Test")
        print("="*80)
        
        test_queries = [
            {
                'query': 'Part 44 costs',
                'expected_terms': ['Part 44', 'costs', 'General Rules'],
                'min_results': 1
            },
            {
                'query': 'civil recovery proceedings',
                'expected_terms': ['civil recovery', 'proceedings'],
                'min_results': 1
            },
            {
                'query': 'insolvency practice direction',
                'expected_terms': ['insolvency', 'practice direction'],
                'min_results': 1
            }
        ]
        
        passed = 0
        failed = 0
        
        for test in test_queries:
            results = list(self.client.search(
                test['query'],
                top=5,
                select=['id', 'sourcepage', 'content']
            ))
            
            if len(results) >= test['min_results']:
                # Check if any result contains expected terms
                found_terms = False
                for result in results:
                    content = result.get('content', '').lower()
                    sourcepage = result.get('sourcepage', '').lower()
                    combined = content + ' ' + sourcepage
                    
                    if any(term.lower() in combined for term in test['expected_terms']):
                        found_terms = True
                        break
                
                if found_terms:
                    print(f"✅ Query passed: '{test['query']}' ({len(results)} results)")
                    passed += 1
                else:
                    print(f"⚠️  Query questionable: '{test['query']}' (results don't match expected terms)")
                    print(f"    Top result: {results[0].get('sourcepage', 'N/A')[:60]}")
                    failed += 1
            else:
                print(f"❌ Query failed: '{test['query']}' (only {len(results)} results, expected {test['min_results']}+)")
                failed += 1
        
        if failed == 0:
            print(f"\n✅ PASSED: All {passed} search quality tests succeeded")
            self.results['search_quality'] = 'PASS'
            return True
        else:
            print(f"\n⚠️  PARTIAL: {passed} passed, {failed} failed")
            self.results['search_quality'] = f'PARTIAL - {failed} failures'
            return passed > failed
    
    def test_5_category_distribution(self):
        """Test 5: Verify document categories are properly assigned."""
        print("\n" + "="*80)
        print("TEST 5: Category Distribution")
        print("="*80)
        
        results = self.client.search('*', select='category')
        categories = {}
        
        for result in results:
            cat = result.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"Total categories: {len(categories)}")
        print(f"\nCategory distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} documents")
        
        if len(categories) > 0:
            print(f"\n✅ PASSED: Documents have category assignments")
            self.results['category_distribution'] = 'PASS'
            return True
        else:
            print(f"\n❌ FAILED: No categories found")
            self.results['category_distribution'] = 'FAIL'
            return False
    
    def test_6_new_documents(self):
        """Test 6: Verify new documents from upload are present."""
        print("\n" + "="*80)
        print("TEST 6: New Document Verification")
        print("="*80)
        
        new_doc_ids = [
            'Practice_Direction___Pre-Action_Conduct_and_Protocols',
            'Practice_Direction___Competition_Law___claims_relating_to_the_application_of_chapters_I_and_II_of_pa',
            'Insolvency_Proceedings_chunk_002',
            'Practice_Direction___Civil_Recovery_Proceedings_chunk_001'
        ]
        
        found = 0
        missing = []
        
        for doc_id in new_doc_ids:
            results = list(self.client.search(
                '*',
                filter=f"id eq '{doc_id}'",
                top=1,
                select=['id', 'sourcepage']
            ))
            
            if results:
                print(f"✅ Found: {doc_id[:60]}...")
                found += 1
            else:
                print(f"❌ Missing: {doc_id[:60]}...")
                missing.append(doc_id)
        
        if len(missing) == 0:
            print(f"\n✅ PASSED: All {found} new documents found")
            self.results['new_documents'] = 'PASS'
            return True
        else:
            print(f"\n❌ FAILED: {len(missing)} new documents missing")
            self.results['new_documents'] = f'FAIL - {len(missing)} missing'
            return False
    
    def test_7_content_list_handling(self):
        """Test 7: Verify documents with list content were processed correctly."""
        print("\n" + "="*80)
        print("TEST 7: List Content Handling")
        print("="*80)
        
        # Check documents from civil_procedure_rules_review.json which had list content
        test_ids = [
            'Practice_Direction___Pre-Action_Conduct_and_Protocols',
            'Insolvency_Proceedings_chunk_001',
            'Practice_Direction___Competition_Law___claims_relating_to_the_application_of_chapters_I_and_II_of_pa'
        ]
        
        passed = 0
        failed = 0
        
        for doc_id in test_ids:
            results = list(self.client.search(
                '*',
                filter=f"id eq '{doc_id}'",
                top=1,
                select=['id', 'content']
            ))
            
            if results:
                content = results[0].get('content', '')
                # Check content is string (not list) and non-empty
                if isinstance(content, str) and len(content) > 0:
                    print(f"✅ Correct: {doc_id[:60]}... (string, {len(content)} chars)")
                    passed += 1
                else:
                    print(f"❌ Invalid: {doc_id[:60]}... (type={type(content)}, len={len(content) if isinstance(content, (str, list)) else 'N/A'})")
                    failed += 1
            else:
                print(f"❌ Not found: {doc_id[:60]}...")
                failed += 1
        
        if failed == 0:
            print(f"\n✅ PASSED: All list-content documents processed correctly")
            self.results['list_content'] = 'PASS'
            return True
        else:
            print(f"\n❌ FAILED: {failed} documents have content issues")
            self.results['list_content'] = f'FAIL - {failed} issues'
            return False
    
    def run_all_tests(self):
        """Run all Phase 2 tests."""
        print("\n" + "="*80)
        print("PHASE 2: COMPREHENSIVE INDEX TESTING")
        print(f"Index: {self.index_name}")
        print("="*80)
        
        tests = [
            self.test_1_duplicate_check,
            self.test_2_filterable_id,
            self.test_3_document_structure,
            self.test_4_search_quality,
            self.test_5_category_distribution,
            self.test_6_new_documents,
            self.test_7_content_list_handling
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n❌ TEST ERROR: {test.__name__} - {str(e)}")
                failed += 1
                self.results[test.__name__] = f'ERROR - {str(e)[:100]}'
        
        # Final Summary
        print("\n" + "="*80)
        print("PHASE 2 TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {len(tests)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"\nTest Results:")
        for test_name, result in self.results.items():
            status = "✅" if result == 'PASS' else ("⚠️ " if 'PARTIAL' in result else "❌")
            print(f"  {status} {test_name}: {result}")
        
        print("\n" + "="*80)
        if failed == 0:
            print("✅ ALL TESTS PASSED - Index ready for Phase 3 (GitHub Workflow)")
            print("\nNext Steps:")
            print("1. Update GitHub secret: AZURE_SEARCH_INDEX=legal-court-rag-index-v2")
            print("2. Test workflow with dry_run: true")
            print("3. Run workflow with dry_run: false")
            print("\nSee docs/INDEX_MIGRATION_CHECKLIST.md for details")
        else:
            print(f"❌ {failed} TESTS FAILED - Fix issues before proceeding")
            print("\nRecommended Actions:")
            if 'duplicate_check' in self.results and self.results['duplicate_check'] != 'PASS':
                print("- Remove duplicate documents from index")
            if 'filterable_id' in self.results and self.results['filterable_id'] != 'PASS':
                print("- Verify index schema has filterable=True for id field")
            if 'new_documents' in self.results and self.results['new_documents'] != 'PASS':
                print("- Re-run upload script to add missing documents")
        print("="*80 + "\n")
        
        return failed == 0

if __name__ == "__main__":
    tester = Phase2Tester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
