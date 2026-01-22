#!/usr/bin/env python
"""
Scraper for UK Civil Procedure Rules (CPR) - Enhanced for RAG Accuracy.
Uses lxml for robust parsing and injects "Breadcrumb Context" into every paragraph
to ensure high-precision retrieval (e.g., [Part 3 > Rule 3.4] Text...).
"""
import os
import sys
import json
import time
import argparse
import random
import re
import logging
import requests
import io
import pypdf
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Dict, Optional

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from config import Config
except ImportError:
    # Fallback if config not found (e.g. running standalone)
    class Config:
        UPLOAD_DIR = "data/legal-scraper/processed/Upload"
        VERBOSE = True

# Configure logging
logging.basicConfig(
    level=logging.INFO if Config.VERBOSE else logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.justice.gov.uk/courts/procedure-rules/civil/rules"

class CPRScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.output_dir = Config.UPLOAD_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch URL and return BeautifulSoup object using lxml for speed."""
        try:
            # Optimized for speed - reduced sleep time
            # time.sleep(random.uniform(0.1, 0.3)) 
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # PDF Handling
            if url.lower().endswith('.pdf') or 'application/pdf' in response.headers.get('Content-Type', ''):
                logger.info(f"Detected PDF content for {url}")
                try:
                    pdf_file = io.BytesIO(response.content)
                    pdf_reader = pypdf.PdfReader(pdf_file)
                    text_content = []
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_content.append(extracted)
                    
                    full_text = "\n\n".join(text_content)
                    logger.info(f"Extracted {len(full_text)} chars from PDF")
                    
                    # Convert newlines to paragraphs for better scraping compatibility
                    # Split by double newline (paragraphs) and wrap in <p>
                    paragraphs = full_text.split('\n\n')
                    html_paragraphs = "".join([f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()])

                    # Wrap in minimal HTML structure
                    html_wrapper = f"""
                    <html>
                    <body>
                        <article>
                            <h1>{url.split('/')[-1].replace('.pdf', '')}</h1>
                            <div class="pdf-content">
                                {html_paragraphs}
                            </div>
                        </article>
                    </body>
                    </html>
                    """
                    return BeautifulSoup(html_wrapper, 'html.parser')
                except Exception as pdf_err:
                    logger.error(f"Failed to parse PDF {url}: {pdf_err}")
                    return None

            # Use html.parser as fallback if lxml is missing/broken
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def get_cpr_links(self) -> List[Dict[str, str]]:
        """Scrape the main index page for CPR Part links."""
        search_urls = [ BASE_URL, "https://www.justice.gov.uk/courts/procedure-rules/civil/protocol" ]

        links = []

        for target_url in search_urls:
          logger.info(f"Fetching index: {target_url}")
          soup = self.get_soup(target_url)
          if not soup:
              continue

          # Find the main content area
          main_content = soup.find('div', id='content') or soup.find('main') or soup.find('body')
        
          if not main_content:
              logger.error(f"Could not find main content on index page {target_url}")
              continue

          for a in main_content.find_all('a', href=True):
              href = a['href']
              text = a.get_text(strip=True)
            
              # Normalize URL
              full_url = urljoin(BASE_URL, href)
            
              # Filter for likely CPR Part links or protocol
              is_part = "procedure-rules/civil/rules/" in full_url and ("part" in full_url.lower() or "part" in text.lower())
              is_protocol = "procedure-rules/civil/protocol" in full_url or "/protocol/" in full_url or "pd_pre-action_conduct" in full_url or "pd_pre_action_conduct" in full_url.lower()

              if is_part or is_protocol:
                  links.append({
                      "url": full_url,
                      "title": text,
                      "id": self._generate_id_from_url(full_url)
                  })
        
        # Deduplicate by URL
        unique_links = {l['url']: l for l in links}.values()
        return list(unique_links)

    def _generate_id_from_url(self, url: str) -> str:
        """Generate a clean ID from the URL (fallback)."""
        basename = url.strip('/').split('/')[-1]
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', basename)
        return f"cpr_{clean_name}"

    def _generate_id_from_title(self, title: str, content: str) -> str:
        """Generate a consistent ID from Title or Content match."""
        # PRIORITIZE H1 MARKDOWN HEADERS (# ) which represent the true page title
        
        # 1. Check for Practice Direction H1
        pd_match_h1 = re.search(r'^#\s+(PRACTICE\s+DIRECTION\s+\d+[A-Z]*)\s*[-–]\s*([^\n]+)', content, re.MULTILINE | re.IGNORECASE)
        if pd_match_h1:
            pd_num = pd_match_h1.group(1).title()
            pd_title = pd_match_h1.group(2).strip().title()
            return f"{pd_num} – {pd_title}"

        # 2. Check for Part H1
        part_match_h1 = re.search(r'^#\s+(PART\s+\d+[A-Z]?)\s*[-–]\s*([^\n]+)', content, re.MULTILINE)
        if part_match_h1:
            part_num = part_match_h1.group(1).title()
            part_title = part_match_h1.group(2).strip().title()
            return f"{part_num} – {part_title}"
            
        # Fallback to Text search if Markdown missing (backward compatibility)
        
        # 3. Practice Directions (Text fallback)
        pd_match = re.search(r'^(PRACTICE\s+DIRECTION\s+\d+[A-Z]*)\s*[-–]\s*([^\n]+)', content, re.MULTILINE | re.IGNORECASE)
        if pd_match:
            pd_num = pd_match.group(1).title()
            pd_title = pd_match.group(2).strip().title()
            return f"{pd_num} – {pd_title}"

        # 4. PART X (Text fallback)
        part_match = re.search(r'^(PART\s+\d+[A-Z]?)\s*[-–]\s*([^\n]+)', content, re.MULTILINE)
        if part_match:
            part_num = part_match.group(1).title()
            part_title = part_match.group(2).strip().title()
            return f"{part_num} – {part_title}"
        
        # Fallback to link title
        if title:
            clean_title = title.strip()
            if ' - ' in clean_title:
                clean_title = clean_title.replace(' - ', ' – ')
            return clean_title
        
        return None

    def clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove noise elements that confuse legal parsing."""
        # List of noise selectors to remove
        noise_selectors = [
            "script", "style", "nav", "header", "footer", 
            ".tools", ".back-to-top", ".related-items", 
            "#cookie-banner", ".global-cookie-message",
            ".breadcrumb", "#breadcrumb", ".breadcrumbs", ".you-are-here"
        ]
        
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()
                
        # Handle specific text patterns often found in footer boilerplate
        for element in soup.find_all(string=re.compile(r"^Back to top")):
            if element.parent:
                element.parent.decompose()
                
        return soup

    def scrape_rule_page(self, link_info: Dict[str, str]) -> bool:
        """
        Scrape a single rule page with Breadcrumb Context Injection.
        Traverses DOM to identify 'Part' and 'Rule' headers and prepends them
        to paragraph text.
        """
        url = link_info['url']
        fallback_id = link_info['id']
        logger.info(f"Scraping with Context Injection: {url}")
        
        soup = self.get_soup(url)
        if not soup:
            return False

        # Isolate Main Content
        content_div = soup.find('div', class_='article-content') or \
                      soup.find('div', id='content') or \
                      soup.find('main') or \
                      soup.find('body')

        if not content_div:
            logger.warning(f"No content found for {url}")
            return False

        # Clean HTML noise
        content_div = self.clean_html(content_div)

        # Context containers
        context_part = ""
        context_rule = ""
        
        # EXTRACT METADATA: UPDATED DATE
        # Look for <meta name="DC.date.modified" content="YYYY-MM-DD">
        updated_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ") # Default
        meta_date = soup.find('meta', attrs={'name': 'DC.date.modified'})
        if meta_date and meta_date.get('content'):
            try:
                # Convert YYYY-MM-DD to ISO 8601
                d_str = meta_date['content']
                d_obj = datetime.strptime(d_str, "%Y-%m-%d")
                updated_date = d_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                logger.warning(f"Could not parse date {meta_date['content']} for {url}")

        extracted_paragraphs = []
        
        # Recursive traversal or linear scan? 
        # Linear scan of direct children is safer for header tracking.
        # However, text is often nested deep in divs.
        # Strategy: Flatten the DOM into a list of significant elements (headings and paragraphs)
        
        # We will iterate through all distinct elements in document order
        all_elements = content_div.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div', 'li'])
        
        for elem in all_elements:
            # Skip if element no longer strictly exists (was decomposed)
            if not elem.parent:
                continue

            # Skip if element is just a container for other elements we will process
            if elem.name == 'div' and (elem.find('p') or elem.find('h1') or elem.find('h2')):
                continue

            text = elem.get_text(" ", strip=True)
            if not text:
                continue
            
            # Detect Context Changes
            # H1 usually denotes the PART or PRACTICE DIRECTION
            if elem.name == 'h1' or (elem.name == 'p' and re.match(r'^(PART|PRACTICE\s+DIRECTIONS?)\b', text, re.IGNORECASE)):
                context_part = text
                context_rule = "" # Reset rule when Part changes
                # Preserve header in output structure
                extracted_paragraphs.append(f"# {text}")
                continue
                
            # H2/H3 usually denotes the RULE (e.g. "Rule 3.1", "3.1") or Para for PDs
            # Look for "Rule X", "Para X", or digit-dot pattern matching typical rule headers
            # Also catch specific PD 53B / Protocol headers like "DATA PROTECTION"
            is_generic_header = re.match(r'^(DATA PROTECTION|MISUSE OF PRIVATE|HARASSMENT|DEFAMATION|INTRODUCTION|OBJECTIVES|PROPORTIONALITY|EXPERTS|SETTLEMENT|LIMITATION)', text, re.IGNORECASE)
            
            if (elem.name in ['h2', 'h3', 'h4'] or \
               (elem.name == 'p' and (re.match(r'^(Rule|Para\.?|Paragraph)\s*\d+|^\d+(\.\d+)?', text, re.IGNORECASE) or is_generic_header))) \
               and len(text) < 100: # Heuristic: Rules are short titles
                context_rule = text
                extracted_paragraphs.append(f"## {text}")
                continue

            # Process Content Paragraphs
            # Only process leaf nodes or paragraphs to avoid duplication
            # (If a div contains p, find_all returns both. We want the p.)
            if elem.name in ['p', 'li'] and not elem.find_all(['p', 'li']):
                 # BREADCRUMB INJECTION
                 # If we have context, prepend it.
                 # Format: [Part 7 > Rule 7.1] The claim form...
                 
                 breadcrumb = ""
                 if context_part or context_rule:
                     parts = [c for c in [context_part, context_rule] if c]
                     breadcrumb = f"[{' > '.join(parts)}] "
                 
                 full_text = f"{breadcrumb}{text}"
                 extracted_paragraphs.append(full_text)

        # Assemble final document content
        full_content = "\n\n".join(extracted_paragraphs)
        
        # Clean up excessive newlines
        full_content = re.sub(r'\n{3,}', '\n\n', full_content)

        if len(full_content) < 100:
             logger.warning(f"Content too short for {url}")
             return False

        # Use the ID Logic (Backward Compatible)
        title_based_id = self._generate_id_from_title(link_info['title'], full_content)
        doc_id = title_based_id if title_based_id else fallback_id
        
        # Extract sourcefile (Part number) for grouping
        # Better extraction from Title if possible
        clean_title = link_info['title']
        if '–' in clean_title:
             sourcefile = clean_title.split('–')[0].strip()
        elif '-' in clean_title:
             sourcefile = clean_title.split('-')[0].strip()
        else:
             sourcefile = doc_id.split('___')[0].replace('_', ' ').strip() # Fallback for new ID format
        
        # Initialize Chunker
        try:
            from token_chunker import LegalDocumentChunker
            # User Preference: "Per URL" chunking.
            # Maximize token limit to 8000 (near 8191 limit of text-embedding-3-large).
            # Only split if the physical document exceeds the model's capacity.
            chunker = LegalDocumentChunker(max_tokens=8000, overlap_tokens=200)
        except ImportError:
            logger.error("Could not import LegalDocumentChunker. Is token_chunker.py in the same directory?")
            # Fallback to no chunking if missing (risk of truncation)
            chunker = None

        if chunker:
            chunks = chunker.chunk_legal_document(full_content, doc_id, link_info['title'])
        else:
            # Create a single 'chunk' simulating the output
            chunks = [{
                'text': full_content,
                'chunk_index': 0,
                'total_chunks': 1
            }]

        saved_count = 0
        for chunk in chunks:
            chunk_content = chunk['text']
            chunk_index = chunk['chunk_index']
            total_chunks = chunk['total_chunks']
            
            # Generate Chunk ID
            # If only 1 chunk, keep original ID? 
            # Existing system uses _chunk_XXX for large files?
            # To be safe and consistent, if total_chunks > 1 OR we want uniform naming:
            # Let's use _chunk_XXX always if we want consistency, or only if > 1.
            # But earlier grep showed _chunk_001. Let's use uniform if > 1.
            
            if total_chunks > 1:
                chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
            else:
                chunk_id = doc_id

            doc = {
                "id": chunk_id,
                "content": chunk_content, 
                "category": "Civil Procedure Rules and Practice Directions",
                "sourcepage": link_info['title'], 
                "sourcefile": sourcefile,
                "storageUrl": url,
                "oids": ["all"], 
                "groups": ["all", "36094ff3-5c6d-49ef-b385-fa37118527e3"], 
                "parent_id": doc_id,
                "embedding": [], 
                "updated": updated_date
            }

            safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', chunk_id)[:100]
            output_path = os.path.join(self.output_dir, f"{safe_filename}.json")
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save {output_path}: {e}")
        
        if saved_count > 0:
            logger.info(f"Saved {saved_count} chunks for {doc_id}")
            return True
        return False

    def run(self, limit: Optional[int] = None):
        links = self.get_cpr_links()
        logger.info(f"Found {len(links)} potential CPR pages")
        
        if not links:
            return

        if limit:
            links = links[:limit]
        
        import concurrent.futures
        # Increased workers for maximum speed
        logger.info(f"Starting enriched scrape with 15 workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            results = list(executor.map(self.scrape_rule_page, links))
            success_count = sum(1 for r in results if r)
            
        logger.info(f"Scrape complete. Enriched {success_count}/{len(links)} docs.")

def main():
    parser = argparse.ArgumentParser(description="Enriched CPR Scraper")
    parser.add_argument("--test-single", action="store_true", help="Test single page")
    parser.add_argument("--test-few", type=int, help="Test N pages")
    parser.add_argument("--url", action="append", help="Scrape specific URL(s)")
    args = parser.parse_args()

    scraper = CPRScraper()
    
    if args.url:
        # Manually construct links for specific URLs
        specific_links = []
        for url in args.url:
            # Attempt to derive a decent title from the URL
            derived_title = url.split("/")[-1].replace("-", " ").replace("_", " ").title()
            if url.endswith(".pdf"):
                derived_title = derived_title.replace(".Pdf", "") + " (PDF)"
                
            specific_links.append({
                "url": url,
                "title": derived_title,
                "id": scraper._generate_id_from_url(url)
            })
        
        # Scrape specific links sequentially
        for link in specific_links:
            scraper.scrape_rule_page(link)
    else:
        limit = None
        if args.test_single:
            limit = 1
        elif args.test_few:
            limit = args.test_few
            
        scraper.run(limit=limit)

if __name__ == "__main__":
    main()
