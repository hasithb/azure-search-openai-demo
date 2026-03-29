#!/usr/bin/env python
"""
Content Cleaner for Legal Documents (V2 → Clean Format)

Transforms indexed content by removing formatting noise while preserving
all substantive text and section/subsection boundary information.

Cleaning rules:
1. Remove SOURCE:/SOURCEPAGE:/CATEGORY:/SECTION: metadata headers (redundant with index fields)
2. Replace [PART X > Y] breadcrumbs with Y on its own line (preserves boundary)
3. Strip markdown # headings — keep text, remove #
4. Strip **bold** and __underline__ markers — keep text
5. Remove multi-chunk context headers (Document:/Section:/Part N of M/====)
6. Normalize whitespace (collapse 3+ newlines to 2)

After cleaning, subsection identifiers appear on their own line (e.g. "29.2\\n")
making them unambiguous boundary markers for both current solution and Foundry agents.
"""

import re
from typing import Optional


def strip_metadata_headers(content: str) -> str:
    """Remove SOURCE:, SOURCEPAGE:, CATEGORY:, SECTION: lines from top of content.
    
    These headers are redundant — the same data is in dedicated index fields.
    Only strips from the first ~6 lines (where upload_with_embeddings.py inserts them).
    """
    if not content:
        return content
    
    lines = content.split('\n')
    # Find how many leading lines are metadata headers
    strip_count = 0
    for i, line in enumerate(lines[:8]):  # only check first 8 lines
        stripped = line.strip()
        if not stripped:
            # Blank line in header area — skip it
            strip_count = i + 1
            continue
        if re.match(r'^(SOURCE|SOURCEPAGE|CATEGORY|SECTION)\s*:', stripped, re.IGNORECASE):
            strip_count = i + 1
            continue
        # Non-header line found — stop
        break
    
    if strip_count > 0:
        return '\n'.join(lines[strip_count:]).lstrip('\n')
    return content


def replace_breadcrumbs(content: str) -> str:
    """Replace [PART X > subsection] breadcrumbs with the subsection ID on its own line.
    
    Transforms: [PART 29 > 29.2] (1) When it allocates...
    Into:       29.2\n(1) When it allocates...
    
    For breadcrumbs without a subsection (just [PART 29]): remove entirely.
    For headings with breadcrumbs like [PART 29 > Scope]: emit "Scope" on own line.
    
    IMPORTANT: Does NOT touch case citation brackets like [2014] EWCA Civ 1091
    or form references like [N434] — these are legal content, not breadcrumbs.
    """
    if not content:
        return content
    
    # Pattern for breadcrumbs: [PART X... > subsection_or_title] possibly followed by text
    # The key distinguishing feature of breadcrumbs vs case citations:
    # - Breadcrumbs start with [PART, [Practice Direction, [CPR, [Court, etc.
    # - Case citations start with [year] like [2014] or [form] like [N434]
    
    def replace_breadcrumb_match(match):
        """Replace a single breadcrumb, extracting the subsection after last '>'."""
        full_match = match.group(0)
        inner = match.group(1)  # content between [ and ]
        after = match.group(2)  # text after the ]
        
        if '>' in inner:
            # Has hierarchy — extract after last >
            parts = inner.rsplit('>', 1)
            subsection = parts[1].strip()
            if subsection:
                # Put subsection on its own line, then the following text
                return f'\n{subsection}\n{after.lstrip()}'
            else:
                return after.lstrip()
        else:
            # No hierarchy — just [PART 29] or similar top-level marker
            # Extract inner text for context
            return f'\n{inner.strip()}\n{after.lstrip()}'
    
    # Match breadcrumb-style brackets:
    # Must start with known breadcrumb prefixes to avoid matching case citations
    breadcrumb_pattern = re.compile(
        r'\[('
        r'(?:PART|Practice\s+Direction|CPR|Court|Section|Appendix|Annex)'  # breadcrumb prefix
        r'[^\]]*'  # rest of breadcrumb content
        r')\]'
        r'([ \t]*)',  # optional whitespace after bracket (NOT newline — preserve structure)
        re.IGNORECASE
    )
    
    result = breadcrumb_pattern.sub(replace_breadcrumb_match, content)
    return result


def strip_markdown(content: str) -> str:
    """Remove markdown formatting while preserving text.
    
    - # Heading → Heading
    - ## Heading → Heading
    - **bold** → bold
    - __underline__ → underline
    
    Only strips leading # from lines (not # in middle of text like "Part #23").
    """
    if not content:
        return content
    
    lines = content.split('\n')
    cleaned = []
    for line in lines:
        # Strip markdown heading prefix (# ## ### etc.)
        stripped = re.sub(r'^#{1,6}\s+', '', line)
        # Strip bold markers (**text** → text)
        stripped = re.sub(r'\*\*([^*]+)\*\*', r'\1', stripped)
        # Strip underline markers (__text__ → text)
        stripped = re.sub(r'__([^_]+)__', r'\1', stripped)
        cleaned.append(stripped)
    
    return '\n'.join(cleaned)


def strip_chunk_headers(content: str) -> str:
    """Remove multi-chunk context headers added by token_chunker.py.
    
    Removes patterns like:
        Document: Part 44 – General Rules About Costs
        Section: Costs orders relating to funding arrangements
        Part 2 of 3
        ==================================================
    """
    if not content:
        return content
    
    # Check first few lines for chunk header pattern
    lines = content.split('\n')
    strip_count = 0
    for i, line in enumerate(lines[:6]):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^Document:\s+', stripped):
            strip_count = i + 1
            continue
        if re.match(r'^Section:\s+', stripped):
            strip_count = i + 1
            continue
        if re.match(r'^Part\s+\d+\s+of\s+\d+', stripped):
            strip_count = i + 1
            continue
        if re.match(r'^={5,}$', stripped):
            strip_count = i + 1
            continue
        # If we've been finding headers and hit non-header, stop
        if strip_count > 0:
            break
    
    if strip_count > 0:
        return '\n'.join(lines[strip_count:]).lstrip('\n')
    return content


def normalize_whitespace(content: str) -> str:
    """Collapse 3+ consecutive newlines to exactly 2 (one blank line)."""
    if not content:
        return content
    return re.sub(r'\n{3,}', '\n\n', content).strip()


def clean_content(content: str) -> str:
    """Apply full cleaning pipeline to content.
    
    Order matters:
    1. Strip chunk headers (must be before metadata since they're at top of chunks)
    2. Strip metadata headers
    3. Replace breadcrumbs (before markdown strip, since breadcrumbs may contain #)
    4. Strip markdown
    5. Normalize whitespace
    """
    if not content:
        return content
    
    result = strip_chunk_headers(content)
    result = strip_metadata_headers(result)
    result = replace_breadcrumbs(result)
    result = strip_markdown(result)
    result = normalize_whitespace(result)
    
    return result


# ── Verification helpers ──

def content_text_only(content: str) -> str:
    """Extract only the substantive text (stripping ALL formatting) for comparison.
    
    Used to verify that cleaning doesn't lose any actual text content.
    Strips breadcrumbs entirely (they are navigational, not substantive)
    but preserves case citations like [2014] EWCA Civ.
    """
    if not content:
        return ""
    text = content
    # Remove breadcrumb brackets entirely (navigational markup, not content)
    # These contain duplicated Part titles already in sourcepage/sourcefile fields
    text = re.sub(
        r'\[(?:PART|Practice\s+Direction|CPR|Court|Section|Appendix|Annex)[^\]]*\]\s*',
        '', text, flags=re.IGNORECASE
    )
    # Remove remaining non-citation brackets — keep inner text
    # (case citations like [2014] EWCA Civ are preserved by this pattern)
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'^(SOURCE|SOURCEPAGE|CATEGORY|SECTION)\s*:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^Document:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Section:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Part\s+\d+\s+of\s+\d+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^={5,}$', '', text, flags=re.MULTILINE)
    # Normalize whitespace for comparison
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def verify_no_text_loss(original: str, cleaned: str) -> dict:
    """Verify that cleaning didn't lose substantive text.
    
    Returns dict with 'passed', 'original_words', 'cleaned_words', 'lost_words'.
    """
    orig_text = content_text_only(original)
    clean_text = content_text_only(cleaned)
    
    orig_words = set(orig_text.lower().split())
    clean_words = set(clean_text.lower().split())
    
    # Words that were in original but not in cleaned (potential loss)
    lost = orig_words - clean_words
    
    # Filter out formatting artifacts that are legitimately removed
    formatting_artifacts = {
        'source:', 'sourcepage:', 'category:', 'section:',
        'document:', 'part', '>', 'of', '=', '==', '===',
        ']', '[', '–',  # stray bracket/dash fragments
    }
    real_loss = lost - formatting_artifacts
    
    return {
        'passed': len(real_loss) == 0,
        'original_word_count': len(orig_words),
        'cleaned_word_count': len(clean_words),
        'lost_words': sorted(real_loss)[:20],  # cap at 20
        'lost_count': len(real_loss),
    }
