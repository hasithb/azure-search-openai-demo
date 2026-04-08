#!/usr/bin/env python
"""
Upload legal documents with embeddings to Azure Search.
Supports dry-run mode for validation and staging index uploads.

Usage:
    python upload_with_embeddings.py --input Upload [--dry-run] [--staging]
"""
import os
import sys
import json
import glob
import argparse
import logging
import hashlib
import time
import re
from pathlib import Path
from openai import AzureOpenAI, RateLimitError, APIConnectionError, APIError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add scripts to path
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(script_dir, '../../app/backend')

# Import from local scripts first
sys.path.insert(0, script_dir)
from config import Config

# Then add backend to path for customizations
sys.path.insert(0, backend_dir)
from customizations.subsection_extractor import SubsectionExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Index field detection
_INDEX_FIELDS: set[str] | None = None


def load_index_fields(endpoint: str, index_name: str):
    """Fetch field names from the deployed index to avoid sending unsupported fields."""
    global _INDEX_FIELDS
    from azure.search.documents.indexes import SearchIndexClient
    from azure.identity import DefaultAzureCredential

    idx_client = SearchIndexClient(endpoint=endpoint, credential=DefaultAzureCredential())
    idx = idx_client.get_index(index_name)
    _INDEX_FIELDS = {f.name for f in idx.fields}
    logger.info("Index fields: %s", sorted(_INDEX_FIELDS))

def load_documents_from_files(input_dir: str) -> list:
    """Load all JSON documents from input directory."""
    documents = []
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle both single document and array of documents
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
                    
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents from {len(json_files)} files")
    return documents

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5)
)
def create_embeddings_with_retry(client, texts: list, model: str):
    """Create embeddings with automatic retry on rate limit errors."""
    return client.embeddings.create(input=texts, model=model)

def normalize_openai_endpoint(raw: str) -> str:
    """Normalize AZURE_OPENAI_SERVICE value to a full https endpoint URL.

    Handles all common formats:
      'myservice'                                  -> 'https://myservice.openai.azure.com'
      'myservice.openai.azure.com'                 -> 'https://myservice.openai.azure.com'
      'https://myservice.openai.azure.com'         -> 'https://myservice.openai.azure.com'
      'https://myservice.openai.azure.com/'        -> 'https://myservice.openai.azure.com'
    """
    value = raw.strip().rstrip("/")
    if value.startswith("https://"):
        return value
    if value.endswith(".openai.azure.com"):
        return f"https://{value}"
    return f"https://{value}.openai.azure.com"

def generate_embeddings(documents: list) -> list:
    """Generate embeddings for documents that are missing them."""
    
    docs_to_embed = [doc for doc in documents if not doc.get("embedding")]
    if not docs_to_embed:
        return documents
        
    logger.info(f"Generating embeddings for {len(docs_to_embed)} documents...")
    
    endpoint = normalize_openai_endpoint(Config.AZURE_OPENAI_SERVICE)
    logger.info(f"OpenAI endpoint: {endpoint}")
    # Log URL shape info that won't be masked by GitHub Actions
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    logger.info(f"Endpoint hostname length: {len(parsed.hostname or '')}, dot-count: {(parsed.hostname or '').count('.')}")
        
    deployment = Config.AZURE_OPENAI_EMB_DEPLOYMENT
    logger.info(f"Embedding deployment: {deployment}")
    
    if Config.AZURE_OPENAI_KEY:
        logger.info("Using API key for OpenAI authentication")
        client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_KEY,
            api_version="2023-05-15",
            azure_endpoint=endpoint,
            max_retries=3,  # Built-in retry mechanism
            timeout=120.0
        )
    else:
        # Use DefaultAzureCredential which supports Environment, WorkloadIdentity (OIDC), ManagedIdentity, and AzureCLI
        logger.info("Using DefaultAzureCredential for OpenAI authentication")
        credential = DefaultAzureCredential()
        # Pre-test token acquisition so failures are logged clearly
        try:
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            logger.info(f"✅ Token acquired for cognitiveservices scope (expires {token.expires_on})")
        except Exception as e:
            logger.error(f"❌ Failed to get token for cognitiveservices scope: {e}")
            raise
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        client = AzureOpenAI(
            azure_ad_token_provider=token_provider,
            api_version="2023-05-15",
            azure_endpoint=endpoint,
            max_retries=3,
            timeout=120.0
        )
    
    # Quick connectivity test with a single short text
    logger.info("Testing OpenAI embedding endpoint connectivity...")
    # Pre-check DNS resolution to give a clear error if hostname is wrong
    import socket
    try:
        hostname = urlparse(endpoint).hostname
        addrs = socket.getaddrinfo(hostname, 443)
        logger.info(f"✅ DNS resolved {hostname} ({len(addrs)} address(es))")
    except socket.gaierror as e:
        logger.error(f"❌ DNS resolution failed for hostname '{hostname}': {e}")
        logger.error(f"   Raw AZURE_OPENAI_SERVICE value length: {len(Config.AZURE_OPENAI_SERVICE)}")
        raise RuntimeError(f"Cannot resolve OpenAI hostname '{hostname}'. Check AZURE_OPENAI_SERVICE secret value.") from e
    try:
        test_resp = client.embeddings.create(input=["test"], model=deployment)
        logger.info(f"✅ Connectivity test passed (got {len(test_resp.data[0].embedding)}-dim vector)")
    except Exception as e:
        logger.error(f"❌ Connectivity test FAILED: {type(e).__name__}: {e}")
        # Log the full exception chain for diagnosis
        cause = e.__cause__ or e.__context__
        while cause:
            logger.error(f"  Caused by: {type(cause).__name__}: {cause}")
            cause = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
        raise
        
    # Process in batches optimized for high rate limits
    # With 12,000 requests/min and 2M tokens/min, we can be aggressive
    batch_size = 100  # Larger batches for faster processing
    success_count = 0
    for i in range(0, len(docs_to_embed), batch_size):
        batch = docs_to_embed[i:i+batch_size]
        # Handle content that might be a list (join with newlines) or string
        texts = []
        for doc in batch:
            content = doc["content"]
            if isinstance(content, list):
                # Join list elements with newlines
                text = "\n".join(content)
            else:
                text = content
            texts.append(text.replace("\n", " ")[:8000])  # Truncate to token limit
        
        try:
            response = create_embeddings_with_retry(client, texts, deployment)
            for j, data in enumerate(response.data):
                batch[j]["embedding"] = data.embedding
            success_count += len(batch)
            logger.info(f"✅ Generated embeddings: {success_count}/{len(docs_to_embed)} ({(success_count/len(docs_to_embed)*100):.1f}%)")
            
            # Minimal delay with high rate limits (12K req/min, 2M tokens/min)
            if i + batch_size < len(docs_to_embed):
                time.sleep(0.5)  # 0.5 second delay between batches
        except Exception as e:
            logger.error(f"❌ Error generating embeddings for batch {i//batch_size + 1}: {type(e).__name__}: {e}")
            cause = e.__cause__ or e.__context__
            while cause:
                logger.error(f"  Caused by: {type(cause).__name__}: {cause}")
                cause = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
            logger.warning(f"Skipping {len(batch)} documents in this batch")
            
    logger.info(f"Embedding generation complete: {success_count}/{len(docs_to_embed)} successful")
    return documents

def sanitize_id(doc_id: str) -> str:
    """Sanitize document ID for Azure Search.
    
    Preserves the case of the original ID to match existing index format.
    Only replaces invalid characters with underscores.
    Azure Search document keys support: letters, numbers, dashes, underscores, and equals signs.
    """
    # Replace invalid chars with underscore (preserve case)
    s = re.sub(r'[^a-zA-Z0-9_\-=]', '_', doc_id)
    # Replace multiple consecutive underscores with triple underscore (matches index convention)
    s = re.sub(r'_{2,}', '___', s)
    # Strip leading/trailing underscores
    s = s.strip('_')
    return s

def map_document_to_schema(doc: dict) -> dict:
    """Map document to Azure Search schema."""
    doc_id = doc.get("id", "")
    sanitized_id = sanitize_id(doc_id)
    
    if doc_id != sanitized_id:
        logger.info(f"Sanitized ID: '{doc_id}' -> '{sanitized_id}'")
    
    # Handle content that might be a list (join with newlines) or string
    content = doc.get("content", "")
    if isinstance(content, list):
        content = "\n".join(content)

    sourcepage = doc.get("sourcepage", "")
    sourcefile = doc.get("sourcefile", "")
    category = doc.get("category", "Legal Document")

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
        # Use first comma-delimited segment if it looks like a section label
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
    
    # Extract subsections for accurate citation navigation
    extracted_subsection = SubsectionExtractor.extract_first_subsection(content)
    extracted_subsections = SubsectionExtractor.extract_all_subsections(content)

    derived_subsection = extract_subsection_from_sourcepage(sourcepage)
    parent_section = extract_parent_section_from_sourcepage(sourcepage)

    subsection_id = extracted_subsection or derived_subsection or parent_section or ""
    subsections = list(extracted_subsections)
    if subsection_id and subsection_id not in subsections:
        subsections.insert(0, subsection_id)

    # Prepend structured header if missing to improve citation highlight reliability
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
        
    result = {
        "id": sanitized_id,
        "content": content,
        "embedding": doc.get("embedding", []),
        "category": category,
        "sourcepage": sourcepage,
        "sourcefile": sourcefile,
        "storageUrl": doc.get("storageUrl", ""),
        "oids": doc.get("oids", []) if doc.get("oids") else [],
        "groups": doc.get("groups", []) if doc.get("groups") else [],
    }
    # Only include extended fields if the target index supports them
    if _INDEX_FIELDS is not None:
        for fname, val in [
            ("parent_id", doc.get("parent_id", "")),
            ("subsection_id", subsection_id),
            ("subsections", subsections),
            ("updated", doc.get("updated", "")),
        ]:
            if fname in _INDEX_FIELDS:
                result[fname] = val
    return result

def validate_documents(documents: list, check_embeddings: bool = False) -> tuple[list, list]:
    """Validate documents before upload. Returns (valid, invalid).
    
    Args:
        documents: List of documents to validate
        check_embeddings: If True, validate embedding dimensions. Set False during dry-run.
    """
    valid = []
    invalid = []
    
    for doc in documents:
        errors = []
        
        if not doc.get("id"):
            errors.append("Missing id")
        if not doc.get("content"):
            errors.append("Missing content")
        # Only check embeddings if explicitly requested (during actual upload)
        if check_embeddings and "embedding" in doc and len(doc.get("embedding", [])) != Config.EMBEDDING_DIMENSIONS:
            errors.append(f"Embedding has wrong dimensions: {len(doc.get('embedding', []))} vs {Config.EMBEDDING_DIMENSIONS}")
        
        if errors:
            invalid.append((doc.get("id", "unknown"), errors))
        else:
            valid.append(map_document_to_schema(doc))
    
    return valid, invalid

def compute_content_hash(doc: dict) -> str:
    """Compute a deterministic hash of the document content.
    
    Includes all fields that represent meaningful content changes:
    - id: Document identifier
    - content: The actual text content
    - sourcefile: Part/section reference
    - sourcepage: Human-readable title
    - category: Document classification
    - storageUrl: Source location
    - updated: Official update date from the government website
    
    Excludes fields that don't represent content changes:
    - embedding: Derived from content
    - oids, groups: Access control metadata
    - parent_id: Usually redundant with id
    """
    id_val = doc.get("id", "") or ""
    content = doc.get("content", "") or ""
    # Handle content that might be a list
    if isinstance(content, list):
        content = "\n".join(content)
    sourcefile = doc.get("sourcefile", "") or ""
    sourcepage = doc.get("sourcepage", "") or ""
    category = doc.get("category", "") or ""
    storage_url = doc.get("storageUrl", "") or ""
    updated = doc.get("updated", "") or ""
    
    # Create a deterministic string to hash
    # Using pipe separator as it's unlikely to appear in the data
    to_hash = f"{id_val}|{sourcefile}|{sourcepage}|{category}|{storage_url}|{updated}|{content}"
    return hashlib.md5(to_hash.encode("utf-8")).hexdigest()

def filter_changed_documents(client, documents: list) -> tuple[list, int, int, int]:
    """
    Filter out documents that haven't changed in the index.
    Uses get_document() (key lookup) instead of filter queries, which works
    regardless of whether the 'id' field is marked as filterable.
    Returns (docs_to_upload, unchanged_count, new_count, changed_count)
    """
    if not documents:
        return [], 0, 0, 0
        
    logger.info("Checking for existing documents to minimize updates...")
    docs_to_upload = []
    unchanged_count = 0
    new_count = 0
    changed_count = 0
    
    select_fields = ["id", "content", "sourcefile", "sourcepage", "category", "storageUrl"]
    if _INDEX_FIELDS is not None and "updated" in _INDEX_FIELDS:
        select_fields.append("updated")
    
    for doc in documents:
        doc_id = doc["id"]
        try:
            existing = client.get_document(
                key=doc_id,
                selected_fields=select_fields
            )
            # Document exists — check if content changed
            remote_hash = compute_content_hash(existing)
            local_hash = compute_content_hash(doc)
            
            if remote_hash != local_hash:
                docs_to_upload.append(doc)
                changed_count += 1
                logger.info(f"📝 content changed: {doc_id}")
            else:
                unchanged_count += 1
        except Exception as e:
            error_str = str(e)
            if "ResourceNotFoundError" in type(e).__name__ or "404" in error_str:
                # Document doesn't exist in index — it's new
                docs_to_upload.append(doc)
                new_count += 1
                logger.info(f"✨ New document: {doc_id}")
            else:
                # Unexpected error — include document to be safe
                logger.warning(f"Error checking doc {doc_id}, will re-upload: {e}")
                docs_to_upload.append(doc)
                new_count += 1

    return docs_to_upload, unchanged_count, new_count, changed_count

def upload_to_azure_search(index_name: str, documents: list, batch_size: int = 100, dry_run: bool = False) -> int:
    """Upload documents to Azure Search."""
    try:
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from azure.core.credentials import AzureKeyCredential
        from azure.identity import DefaultAzureCredential, AzureCliCredential
        from azure.core.exceptions import ResourceNotFoundError
        
        endpoint = Config.AZURE_SEARCH_SERVICE
        # Ensure endpoint is a full URL
        if endpoint and not endpoint.startswith("https://"):
            endpoint = f"https://{endpoint}.search.windows.net"
        
        key = Config.AZURE_SEARCH_KEY
        
        if key:
            credential = AzureKeyCredential(key)
        else:
            # Use DefaultAzureCredential which supports Environment, WorkloadIdentity (OIDC), ManagedIdentity, and AzureCLI
            logger.info("Using DefaultAzureCredential for authentication")
            credential = DefaultAzureCredential()
        
        if not endpoint:
            logger.error("Azure Search endpoint not configured")
            return 0
        
        # Verify index exists
        index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        try:
            index_client.get_index(index_name)
            logger.info(f"✅ Index '{index_name}' found")
        except ResourceNotFoundError:
            logger.error(f"❌ Index '{index_name}' does not exist")
            if dry_run:
                logger.info("Dry run: Would create index (not implemented in this script)")
            return 0

        # Load index field schema to avoid sending unsupported fields
        load_index_fields(endpoint, index_name)

        # Configure Search Client
        client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
        
        # --- DIFFERENTIAL UPDATE LOGIC ---
        # Only upload changed/new documents
        docs_to_upload, unchanged, new_count, changed_count = filter_changed_documents(client, documents)
        
        # Write Upload Plan Report
        reports_dir = os.path.join(Config.SCRAPER_DATA_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, "upload_plan.txt")
        
        total_changes = new_count + changed_count
        
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("        LEGAL DOCUMENT SCRAPER - DIFF REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Target Index: {index_name}\n")
            f.write(f"Timestamp:    {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("                    SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Documents Scraped:    {len(documents)}\n")
            f.write(f"✨ New Documents:           {new_count}\n")
            f.write(f"📝 Changed Documents:       {changed_count}\n")
            f.write(f"⏭️  Unchanged Documents:     {unchanged}\n")
            f.write("-" * 60 + "\n\n")
            
            if total_changes > 0:
                f.write("=" * 60 + "\n")
                f.write("⚠️  ACTION REQUIRED: DIFFERENCES DETECTED\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"There are {total_changes} document(s) that need to be updated.\n")
                f.write("The following sections require embedding generation and upload:\n\n")
                
                # List new documents
                if new_count > 0:
                    f.write("NEW DOCUMENTS (to be added):\n")
                    f.write("-" * 40 + "\n")
                    new_docs = [d for d in docs_to_upload if d.get('_is_new', False)]
                    for i, doc in enumerate(docs_to_upload[:30], 1):
                        f.write(f"  {i}. {doc.get('id')}\n")
                    if len(docs_to_upload) > 30:
                        f.write(f"  ... and {len(docs_to_upload) - 30} more\n")
                    f.write("\n")
                
                # List changed documents  
                if changed_count > 0:
                    f.write("CHANGED DOCUMENTS (to be updated):\n")
                    f.write("-" * 40 + "\n")
                    for i, doc in enumerate(docs_to_upload[:30], 1):
                        f.write(f"  {i}. {doc.get('id')}\n")
                    if len(docs_to_upload) > 30:
                        f.write(f"  ... and {len(docs_to_upload) - 30} more\n")
                    f.write("\n")
                
                f.write("=" * 60 + "\n")
                f.write("NEXT STEPS:\n")
                f.write("  1. Review the changes above\n")
                f.write("  2. Approve the upload job in GitHub Actions\n")
                f.write("  3. Embeddings will be generated for changed docs only\n")
                f.write("  4. Documents will be uploaded to Azure Search\n")
                f.write("=" * 60 + "\n")
            else:
                f.write("=" * 60 + "\n")
                f.write("✅ NO ACTION REQUIRED: INDEX IS UP TO DATE\n")
                f.write("=" * 60 + "\n\n")
                f.write("All scraped documents match the current Azure Search index.\n")
                f.write("No embedding generation or upload is needed.\n")
            
        logger.info(f"Upload plan written to {report_path}")

        # === ENHANCED STATISTICS LOGGING ===
        logger.info("\n" + "=" * 80)
        logger.info("📊 DIFFERENTIAL UPLOAD STATISTICS")
        logger.info("=" * 80)
        
        if total_changes > 0:
            logger.info("⚠️  STATUS: DIFFERENCES DETECTED - UPLOAD REQUIRED")
        else:
            logger.info("✅ STATUS: NO DIFFERENCES - INDEX IS UP TO DATE")
        
        logger.info("\nDocument Analysis:")
        logger.info(f"  Total Input Documents:     {len(documents):>6}")
        logger.info(f"  ✨ New Documents:           {new_count:>6}  ({new_count/len(documents)*100:.1f}%)")
        logger.info(f"  📝 Changed Documents:       {changed_count:>6}  ({changed_count/len(documents)*100:.1f}%)")
        logger.info(f"  ⏭️  Unchanged Documents:    {unchanged:>6}  ({unchanged/len(documents)*100:.1f}%)")
        logger.info(f"\n  🎯 Total Requiring Upload:  {total_changes:>6}")
        logger.info("=" * 80 + "\n")
        
        # Write detailed statistics file for workflow artifacts
        stats_file = os.path.join(Config.PROCESSED_DIR, "upload_statistics.txt")
        with open(stats_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DIFFERENTIAL UPLOAD STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target Index: {index_name}\n")
            f.write(f"Mode: {'DRY RUN' if dry_run else 'PRODUCTION UPLOAD'}\n\n")
            
            f.write("Document Counts:\n")
            f.write(f"  Total Input:        {len(documents):>6}\n")
            f.write(f"  New:                {new_count:>6}  ({new_count/len(documents)*100:>5.1f}%)\n")
            f.write(f"  Changed:            {changed_count:>6}  ({changed_count/len(documents)*100:>5.1f}%)\n")
            f.write(f"  Unchanged:          {unchanged:>6}  ({unchanged/len(documents)*100:>5.1f}%)\n")
            f.write(f"  Total to Upload:    {total_changes:>6}\n\n")
            
            if total_changes > 0:
                f.write("Status: UPLOAD REQUIRED\n")
            else:
                f.write("Status: INDEX UP TO DATE\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"📄 Statistics written to {stats_file}")
        
        if not docs_to_upload:
            logger.info("🎉 No changes detected. Index is up to date!")
            # Write to GITHUB_OUTPUT if present
            if os.getenv('GITHUB_OUTPUT'):
                with open(os.getenv('GITHUB_OUTPUT'), 'a') as fh:
                    fh.write("has_changes=false\n")
            return 0

        # Write to GITHUB_OUTPUT if present
        if os.getenv('GITHUB_OUTPUT'):
            with open(os.getenv('GITHUB_OUTPUT'), 'a') as fh:
                fh.write("has_changes=true\n")

        if dry_run:
            logger.info(f"🔍 DRY-RUN: Would upload {len(docs_to_upload)} documents to {index_name}")
            return 0

        # --- GENERATE EMBEDDINGS LATE ---
        # Only generate embeddings for the docs we are actually going to upload
        logger.info(f"Generating embeddings for {len(docs_to_upload)} updates...")
        docs_to_upload = generate_embeddings(docs_to_upload)

        logger.info(f"Uploading {len(docs_to_upload)} valid updates...")
        
        uploaded = 0
        failed = 0
        total_batches = (len(docs_to_upload) + batch_size - 1) // batch_size
        
        for i in range(0, len(docs_to_upload), batch_size):
            batch = docs_to_upload[i:i+batch_size]
            batch_num = i//batch_size + 1
            try:
                results = client.upload_documents(batch)
                successful = sum(1 for r in results if r.succeeded)
                batch_failed = len(batch) - successful
                uploaded += successful
                failed += batch_failed
                
                logger.info(f"Batch {batch_num}/{total_batches}: uploaded {successful}/{len(batch)} documents")
                if batch_failed > 0:
                    logger.warning(f"  ⚠️  {batch_failed} documents failed in this batch")
                
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Error uploading batch {batch_num}: {e}")
                failed += len(batch)
        
        # Final statistics
        logger.info("\n" + "=" * 80)
        logger.info("📊 UPLOAD COMPLETION STATISTICS")
        logger.info("=" * 80)
        logger.info(f"  Documents to Upload:  {len(docs_to_upload):>6}")
        logger.info(f"  ✅ Successfully Uploaded: {uploaded:>6}")
        if failed > 0:
            logger.info(f"  ❌ Failed:                {failed:>6}")
        logger.info("\n  Breakdown by Change Type:")
        logger.info(f"    ✨ New:                 {new_count:>6}")
        logger.info(f"    📝 Changed:             {changed_count:>6}")
        logger.info("=" * 80 + "\n")
        
        # Update statistics file with upload results
        stats_file = os.path.join(Config.PROCESSED_DIR, "upload_statistics.txt")
        with open(stats_file, 'a') as f:
            f.write("\nUPLOAD RESULTS:\n")
            f.write(f"  Attempted:          {len(docs_to_upload):>6}\n")
            f.write(f"  Successful:         {uploaded:>6}\n")
            if failed > 0:
                f.write(f"  Failed:             {failed:>6}\n")
            f.write(f"\n  Breakdown:\n")
            f.write(f"    New:              {new_count:>6}\n")
            f.write(f"    Changed:          {changed_count:>6}\n")
            f.write(f"\n  Completion Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"✅ Upload complete: {uploaded} documents updated")
        return uploaded
        
    except ImportError as e:
        logger.error(f"Azure SDK not available: {e}")
        return 0
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Upload legal documents to Azure Search")
    parser.add_argument("--input", default="Upload", help="Input directory name (default: Upload)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    parser.add_argument("--staging", action="store_true", help="Upload to staging index instead of production")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for uploads")
    
    args = parser.parse_args()
    
    # Validate config
    is_valid, errors = Config.validate_azure_config()
    if not is_valid:
        logger.error("❌ Azure configuration incomplete:")
        for error in errors:
            logger.error(f"   - {error}")
        return 1
    
    # Load documents
    input_dir = os.path.join(Config.PROCESSED_DIR, args.input)
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    documents = load_documents_from_files(input_dir)
    if not documents:
        logger.error("No documents to upload")
        return 1
    
    # NOTE: We do NOT generate embeddings here anymore. 
    # We delay it until after diff analysis to save costs.
    # documents = generate_embeddings(documents)
    
    # Validate documents (don't check embeddings yet - they'll be generated later for changed docs only)
    logger.info(f"\n📋 Validating {len(documents)} documents (structure only, embeddings checked later)...")
    valid, invalid = validate_documents(documents, check_embeddings=False)
    
    if invalid:
        logger.error(f"❌ {len(invalid)} documents failed validation:")
        for doc_id, errors in invalid[:10]:
            logger.error(f"   {doc_id}: {', '.join(errors)}")
        if len(invalid) > 10:
            logger.error(f"   ... and {len(invalid)-10} more")
    
    logger.info(f"✅ {len(valid)} documents passed validation")
    
    # Select target index
    target_index = Config.AZURE_SEARCH_INDEX_STAGING if args.staging else Config.AZURE_SEARCH_INDEX
    logger.info(f"📍 Target index: {target_index}")
    
    # Run upload (or dry-run) with diff logic
    if args.dry_run:
        logger.info("\n🔍 Starting Dry Run Analysis...")
    else:
        logger.info(f"\n⬆️  Starting Upload to {target_index}...")

    uploaded = upload_to_azure_search(target_index, valid, args.batch_size, dry_run=args.dry_run)
    
    if args.dry_run:
        logger.info("Dry run complete.")
        return 0
    elif uploaded > 0:
        logger.info(f"\n✅ Success! {uploaded} documents uploaded")
        return 0
    else:
        # If 0 uploaded but not dry run, it might mean no changes (success) or failure.
        # upload_to_azure_search returns 0 for both "no changes" and "error".
        # Logging in the function makes it clear.
        logger.info("Process complete.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
