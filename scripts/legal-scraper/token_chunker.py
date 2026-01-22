import re
import tiktoken
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class LegalDocumentChunker:
    """
    Intelligent chunker for legal documents that respects legal structure
    and maintains semantic meaning while staying within token limits.
    """
    
    def __init__(self, max_tokens: int = 7000, overlap_tokens: int = 200):
        """
        Initialize the chunker with token limits.
        
        Args:
            max_tokens: Maximum tokens per chunk (leave buffer for embedding model)
            overlap_tokens: Tokens to overlap between chunks for context
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the embedding model's tokenizer."""
        return len(self.encoding.encode(text))
    
    def find_legal_boundaries(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Find logical boundaries in legal text for chunking.
        
        Returns:
            List of (position, boundary_type, header_text) tuples
        """
        boundaries = []
        
        # Legal document boundary patterns (in order of preference)
        # Updated to support Markdown headers (# PART, ## Rule) in addition to raw text
        patterns = [
            # Major sections (highest priority)
            (r'\n\s*([IVX]+)\s+([A-Z][A-Z\s]+)\s*\n', 'major_section'),
            (r'\n\s*(?:#\s*)?(PART\s+\d+\s*[-–]\s*[A-Z][A-Z\s]+)\s*\n', 'part'),
            (r'\n\s*(?:#\s*)?(PRACTICE DIRECTION\s+\d+[A-Z]?\s*[-–]\s*[A-Z][A-Z\s]+)\s*\n', 'practice_direction'),
            (r'\n\s*(DATA PROTECTION|MISUSE OF PRIVATE|HARASSMENT|DEFAMATION|INTRODUCTION|OBJECTIVES|PROPORTIONALITY|EXPERTS|SETTLEMENT|LIMITATION).*\n', 'major_section'),
            
            # Rules and sub-rules
            (r'\n\s*(?:##\s*)?([A-Z][a-z]+\s+\d+(?:\.\d+)*(?:\s*[A-Z]\s*\d*)?)\s*\n', 'rule'),
            (r'\n\s*(\d+\.\d+(?:\.\d+)*)\s+([A-Z][^.]+)\s*\n', 'sub_rule'),
            (r'\n\s*(\d+\.)\s+([A-Z][^.]+)\s*\n', 'numbered_section'),
            
            # Paragraphs with legal structure
            (r'\n\s*\(([a-z])\)\s+([A-Z][^.]+)', 'paragraph'),
            (r'\n\s*\((\d+)\)\s+([A-Z][^.]+)', 'numbered_paragraph'),
            
            # Headers and important markers
            (r'\n\s*(To the top)\s*\n', 'section_end'),
            (r'\n\s*([A-Z][a-z]+(?:\s+[a-z]+)*)\s*\n(?=\d+\.)', 'topic_header'),
        ]
        
        for pattern, boundary_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                start_pos = match.start()
                header_text = match.group(0).strip()
                boundaries.append((start_pos, boundary_type, header_text))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        
        return boundaries
    
    def create_chunk_with_context(self, text: str, start: int, end: int, 
                                 chunk_index: int, total_chunks: int,
                                 rule_title: str, section_context: str = "") -> str:
        """
        Create a chunk with proper context and metadata.
        
        Args:
            text: Full text
            start: Start position in text
            end: End position in text
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks for this document
            rule_title: Title of the rule/document
            section_context: Context about which section this chunk belongs to
        
        Returns:
            Formatted chunk with context
        """
        chunk_text = text[start:end].strip()
        
        # Add context header for multi-chunk documents
        if total_chunks > 1:
            context_header = f"Document: {rule_title}"
            if section_context:
                context_header += f"\nSection: {section_context}"
            context_header += f"\nPart {chunk_index + 1} of {total_chunks}"
            context_header += "\n" + "="*50 + "\n"
            chunk_text = context_header + chunk_text
        
        return chunk_text
    
    def chunk_legal_document(self, text: str, document_id: str, 
                           rule_title: str) -> List[Dict]:
        """
        Chunk a legal document intelligently, respecting legal boundaries.
        Prioritizes keeping sections intact. Only splits internally if a single section exceeds max_tokens.
        """
        token_count = self.count_tokens(text)
        
        # If document is within token limit, return as single chunk
        if token_count <= self.max_tokens:
            return [{
                'text': text,
                'token_count': token_count,
                'chunk_index': 0,
                'total_chunks': 1,
                'needs_chunking': False
            }]
        
        logger.info(f"Document {document_id} has {token_count} tokens, chunking required")
        
        # Find legal boundaries
        boundaries = self.find_legal_boundaries(text)
        
        if not boundaries:
            return self._fallback_sentence_chunking(text, document_id, rule_title)
        
        chunks = []
        current_start = 0
        last_safe_boundary = 0
        
        # Add a synthetic boundary at the end to ensure we process the last section
        boundary_list = boundaries + [(len(text), 'end', '')]
        
        # We also need to be careful: boundaries are Start positions of headers.
        # So the content of "Rule 1" is [boundary[i].pos : boundary[i+1].pos].
        
        # Actually, my proposed algorithm works better if we iterate through segments.
        # Let's iterate through the boundary list.
        
        # State tracking
        current_section_context = ""
        last_section_context = ""

        # Use a while loop to allow manual index manipulation (for retries after splitting)
        i = 0
        # Start looking from the first boundary? 
        # No, we just iterate the identified boundaries. 
        # But `current_start` must advance.
        
        while i < len(boundary_list):
            boundary_pos, boundary_type, header_text = boundary_list[i]
            
            # If boundary is behind current_start (e.g. after a split), skip it
            if boundary_pos <= current_start and boundary_type != 'end':
                # Update context if we are passing a header
                if boundary_type in ['major_section', 'part', 'practice_direction', 'rule']:
                     current_section_context = header_text
                i += 1
                continue

            # Candidate chunk: From current_start to this boundary
            candidate_text = text[current_start:boundary_pos]
            candidate_tokens = self.count_tokens(candidate_text)
            
            if candidate_tokens <= self.max_tokens:
                # It fits! 
                # This boundary is now our "safe" aggregation point.
                last_safe_boundary = boundary_pos
                
                # Update context for the *next* segment
                if boundary_type in ['major_section', 'part', 'practice_direction', 'rule']:
                     # Before we move on, save the context of the segment we just swallowed?
                     # Actually, valid implementation calculates context based on the *start* of the chunk.
                     # We just need to track the latest header we've seen.
                     last_section_context = current_section_context
                     current_section_context = header_text
                
                # Move to try next boundary
                i += 1
            else:
                # Overflow!
                # Decision: Aggregate overflow or Single Section overflow?
                
                # Check if we have aggregated multiple sections (last_safe_boundary > current_start)
                if last_safe_boundary > current_start:
                    # AGGREGATION OVERFLOW:
                    # The chunk [current_start : boundary_pos] is too big.
                    # But [current_start : last_safe_boundary] was safe.
                    # So we cut at last_safe_boundary.
                    
                    chunk_text = text[current_start:last_safe_boundary].strip()
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'token_count': self.count_tokens(chunk_text),
                            'section_context': last_section_context, # Approximate
                            'start_pos': current_start,
                            'end_pos': last_safe_boundary
                        })
                    
                    # Advance
                    current_start = last_safe_boundary
                    # We do NOT increment i. We must re-evaluate the current boundary 
                    # against the new current_start.
                
                else:
                    # SINGLE SECTION OVERFLOW:
                    # We haven't even reached the first safe boundary after current_start, 
                    # and we are already over limit. 
                    # This means [current_start : boundary_pos] (one section) is huge.
                    
                    # We must split strictly.
                    # Estimate end point (current_start + max_tokensish)
                    # We can use the existing _find_safe_break_point logic but constrained.
                    
                    break_point = self._find_safe_break_point(
                        text, current_start, boundary_pos, current_section_context
                    )
                    
                    # If break_point fails to advance (edge case), force advance
                    if break_point <= current_start:
                        break_point = current_start + int(len(candidate_text) * 0.5) 
                        if break_point <= current_start: # Still stuck (1 char?)
                             break_point = boundary_pos
                    
                    chunk_text = text[current_start:break_point].strip()
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'token_count': self.count_tokens(chunk_text),
                            'section_context': current_section_context,
                            'start_pos': current_start,
                            'end_pos': break_point
                        })
                    
                    current_start = break_point
                    last_safe_boundary = current_start # Reset safe boundary
                    # Do not increment i. Re-eval rest of the section.

        # Format chunks
        formatted_chunks = []
        total_chunks = len(chunks)
        
        for i, chunk_data in enumerate(chunks):
            formatted_text = self.create_chunk_with_context(
                chunk_data['text'], 0, len(chunk_data['text']),
                i, total_chunks, rule_title, chunk_data.get('section_context', '')
            )
            
            formatted_chunks.append({
                'text': formatted_text,
                'token_count': self.count_tokens(formatted_text),
                'chunk_index': i,
                'total_chunks': total_chunks,
                'needs_chunking': True,
                'section_context': chunk_data.get('section_context', '')
            })
        
        logger.info(f"Split document {document_id} into {len(formatted_chunks)} chunks")
        return formatted_chunks
    
    def _find_safe_break_point(self, text: str, start: int, end: int, 
                              section_context: str) -> int:
        """Find a safe place to break text while preserving legal meaning."""
        # Look for paragraph breaks, sentence endings, etc.
        search_text = text[start:end]
        
        # Priority breaking points
        break_patterns = [
            r'\n\s*\n',  # Double line breaks
            r'\.\s*\n',  # Sentence ending with newline
            r'\n\s*\([a-z]\)',  # Before paragraph markers
            r'\n\s*\(\d+\)',  # Before numbered items
            r'\.\s+',  # Sentence boundaries
        ]
        
        for pattern in break_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Find the match closest to our target position
                target_pos = len(search_text) * 0.7  # Prefer breaks around 70% through
                best_match = min(matches, key=lambda m: abs(m.end() - target_pos))
                break_point = start + best_match.end()
                
                # Ensure we don't create too small chunks
                if break_point - start > self.max_tokens * 0.3:
                    return break_point
        
        # Fallback to hard limit
        return min(end, start + self.max_tokens * 4)  # Rough character estimate
    
    def _split_large_text(self, text: str, section_context: str) -> List[Dict]:
        """Split text that's still too large after boundary detection."""
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Estimate chunk size (rough character to token ratio)
            estimated_end = current_pos + (self.max_tokens * 4)
            estimated_end = min(estimated_end, len(text))
            
            # Find safe break point
            break_point = self._find_safe_break_point(
                text, current_pos, estimated_end, section_context
            )
            
            chunk_text = text[current_pos:break_point].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'token_count': self.count_tokens(chunk_text),
                    'section_context': section_context,
                    'start_pos': current_pos,
                    'end_pos': break_point
                })
            
            current_pos = break_point
        
        return chunks
    
    def _fallback_sentence_chunking(self, text: str, document_id: str, 
                                   rule_title: str) -> List[Dict]:
        """Fallback chunking method when no legal boundaries are found."""
        logger.warning(f"No legal boundaries found for {document_id}, using sentence chunking")
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if self.count_tokens(potential_chunk) > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Format chunks
        formatted_chunks = []
        total_chunks = len(chunks)
        
        for i, chunk_text in enumerate(chunks):
            formatted_text = self.create_chunk_with_context(
                chunk_text, 0, len(chunk_text), i, total_chunks, rule_title
            )
            
            formatted_chunks.append({
                'text': formatted_text,
                'token_count': self.count_tokens(formatted_text),
                'chunk_index': i,
                'total_chunks': total_chunks,
                'needs_chunking': True
            })
        
        return formatted_chunks
