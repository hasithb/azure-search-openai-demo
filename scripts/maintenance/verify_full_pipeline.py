
import json
import glob
import os
import sys

# Add app/backend to path to import backend modules
sys.path.insert(0, os.path.join(os.getcwd(), "app", "backend"))

try:
    from customizations.approaches.citation_builder import CitationBuilder
    from customizations.approaches.source_processor import SourceProcessor
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)

def verify_pipeline():
    print("--- Starting Pipeline Verification ---\n")
    
    # 1. Inspect Generated Data
    upload_dir = "data/legal-scraper/processed/Upload"
    json_files = sorted(glob.glob(os.path.join(upload_dir, "*.json")))
    
    if not json_files:
        print(f"❌ No JSON files found in {upload_dir}. Did you run the scraper?")
        return

    print(f"✅ Found {len(json_files)} JSON files.")
    
    # Check for chunking
    chunked_files = [f for f in json_files if "_chunk_" in f]
    if chunked_files:
        print(f"✅ Chunking active: Found {len(chunked_files)} chunk files.")
    else:
        print("⚠️ No chunks found. Verify if documents were large enough to split.")

    # 2. Simulate Backend processing on the files
    builder = CitationBuilder()
    processor = SourceProcessor(citation_builder=builder)
    
    print("\n--- Testing Backend Citation Logic on New Data ---\n")
    
    for file_path in json_files[:5]:  # Test first 5 files
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        # Create a mock Document object (as expected by backend)
        class MockDoc:
            def __init__(self, data):
                self.id = data.get('id')
                self.content = data.get('content')
                self.sourcepage = data.get('sourcepage')
                self.sourcefile = data.get('sourcefile')
        
        doc = MockDoc(doc_data)
        
        print(f"Testing Document: {doc.id}")
        
        # Test 1: Extract Subsection
        subsection = builder.extract_subsection(doc)
        print(f"  > Extracted Subsection: '{subsection}'")
        
        # Test 2: Verify Breadcrumbs in content (Visual check)
        if "[Part" in doc.content or "[PRACTICE" in doc.content:
            print("  > ✅ Breadcrumbs detected in content.")
        else:
            print("  > ⚠️ No Breadcrumbs found in content.")
            
        # Test 3: Verify Markdown headers
        if "# " in doc.content:
            print("  > ✅ Markdown headers detected.")
        else:
            print("  > ⚠️ No Markdown headers found.")

        # Test 4: Build full citation
        citation = builder.build_enhanced_citation(doc, 1)
        print(f"  > Generated Citation: '{citation}'")
        print("-" * 30)

if __name__ == "__main__":
    verify_pipeline()
