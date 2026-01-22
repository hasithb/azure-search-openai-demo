import os
import glob
import json
import asyncio
import re
from azure.identity.aio import AzureDeveloperCliCredential
from azure.search.documents.aio import SearchClient
from load_azd_env import load_azd_env
from customizations.subsection_extractor import SubsectionExtractor


def has_existing_header(text: str) -> bool:
    if not text:
        return False
    head = [line.strip() for line in text.splitlines()[:6] if line.strip()]
    if any(line.startswith("SOURCE:") or line.startswith("SOURCEPAGE:") or line.startswith("SECTION:") for line in head):
        return True
    if any(line.startswith("[PART") or (line.startswith("[") and ">" in line) for line in head):
        return True
    return False


def extract_parent_section_from_sourcepage(value: str) -> str:
    if not value:
        return ""
    raw = value.strip()
    first_segment = raw.split(",", 1)[0].strip()
    if re.match(r"^[A-Z]\.", first_segment) or re.match(r"^(Section|Appendix|Part|Practice Direction)\b", first_segment, re.IGNORECASE):
        return first_segment

    patterns = [
        r"\b(Practice Direction\s+[0-9A-Z]+)\b",
        r"\b(Part\s+\d+[A-Z]?)\b",
        r"\b(Section\s+\d+)\b",
        r"\b(Appendix\s+[A-Z])\b",
        r"\b([A-Z]\.\s+[^,]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_subsection_from_sourcepage(value: str) -> str:
    if not value:
        return ""
    raw = value.strip()
    patterns = [
        r"\b([A-Z]\.\d+(?:\.\d+)?)\b",  # C.2, F.1, A.1.1
        r"\b([A-Z]\d+\.\d+(?:\.\d+)?)\b",  # A4.1, B2.3
        r"\b(\d+\.\d+(?:\.\d+)?)\b",  # 8.4, 35.1
        r"\b([A-Z]\d+)\b",  # A1, B2
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""

async def main():
    load_azd_env()
    service_name = os.environ.get("AZURE_SEARCH_SERVICE")
    index_name = os.environ.get("AZURE_SEARCH_INDEX")
    
    if not service_name or not index_name:
        print("Error: AZURE_SEARCH_SERVICE or AZURE_SEARCH_INDEX not set.")
        return

    endpoint = f"https://{service_name}.search.windows.net"
    credential = AzureDeveloperCliCredential()

    print(f"Connecting to {endpoint}, index: {index_name}")

    async with SearchClient(endpoint=endpoint, index_name=index_name, credential=credential) as client:
        # Adjusted path to look for files relative to workspace root (assuming script runs from root or we adjust path)
        # Using absolute path logic or relative to cwd.
        # If running from root: "court_guides_processing_pipeline/outputs/*.json"
        files = glob.glob("court_guides_processing_pipeline/outputs/*.json")
        if not files:
            print("No JSON files found in court_guides_processing_pipeline/outputs/")
            return

        for file_path in files:
            print(f"Processing {file_path}...")
            with open(file_path, "r") as f:
                documents = json.load(f)
            
            to_upload = []
            for doc in documents:
                sourcepage = doc.get("sourcepage", "") or ""
                sourcefile = doc.get("sourcefile", "") or ""
                category = doc.get("category", "") or ""
                content = doc.get("content", "") or ""

                extracted_subsection = SubsectionExtractor.extract_first_subsection(content)
                extracted_subsections = SubsectionExtractor.extract_all_subsections(content)
                derived_subsection = extract_subsection_from_sourcepage(sourcepage)
                parent_section = extract_parent_section_from_sourcepage(sourcepage)

                subsection_id = extracted_subsection or derived_subsection or parent_section or ""
                subsections = list(extracted_subsections)
                if subsection_id and subsection_id not in subsections:
                    subsections.insert(0, subsection_id)

                if content and not has_existing_header(content):
                    header_lines = []
                    if sourcefile:
                        header_lines.append(f"SOURCE: {sourcefile}")
                    if sourcepage:
                        header_lines.append(f"SOURCEPAGE: {sourcepage}")
                    if category:
                        header_lines.append(f"CATEGORY: {category}")
                    if parent_section and parent_section != subsection_id:
                        header_lines.append(f"SECTION: {parent_section}")
                    if subsection_id:
                        header_lines.append(f"## {subsection_id}")

                    if header_lines:
                        content = "\n".join(header_lines) + "\n\n" + content

                # Select only fields that match the index schema
                clean_doc = {
                    "id": doc.get("id"),
                    "content": content,
                    "category": category,
                    "sourcepage": sourcepage,
                    "sourcefile": sourcefile,
                    "storageUrl": doc.get("storageUrl"),
                    "embedding": doc.get("embedding"),
                    "subsection_id": subsection_id,
                    "subsections": subsections,
                }
                
                # Filter out None values just in case, though usually fields are nullable
                # Actually, verify_v2_comprehensive.py showed category is failing, so we must ensure it's here.
                if not clean_doc["category"]:
                    print(f"Warning: Document {clean_doc['id']} is missing category.")
                
                to_upload.append(clean_doc)
            
            # Batch upload
            batch_size = 500
            for i in range(0, len(to_upload), batch_size):
                batch = to_upload[i : i + batch_size]
                try:
                    results = await client.upload_documents(documents=batch)
                    print(f"Uploaded batch {i//batch_size + 1} ({len(batch)} docs) from {file_path}")
                except Exception as e:
                    print(f"Error uploading batch from {file_path}: {e}")

    await credential.close()

if __name__ == "__main__":
    asyncio.run(main())
