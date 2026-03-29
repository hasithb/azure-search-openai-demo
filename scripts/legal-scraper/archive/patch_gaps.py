#!/usr/bin/env python3
"""
Targeted patch script: fix the 6 genuine coverage gaps in v3_full_corrected.json.

Fixes applied:
  1. Part 11             → assign subsections=['11.1']
  2. PD 31B chunk_000   → assign all 15 page sections (replace ['Schedule'])
  3. PAP Dilapidations  → run extractor on newly-fetched HTML, assign sections
  4. PAP Housing ch_001 → assign sections 8.1–12.1
  5. PAP JR single chunk→ assign ['annexa','annexb','annexc']
  6. PAP Clinical ch_000→ assign ['annex a','annex b','annex c']
"""
import json, sys
from pathlib import Path
from bs4 import BeautifulSoup

ROOT  = Path(__file__).parent.parent.parent
V3    = ROOT / "data/legal-scraper/processed/v3_full_corrected.json"
CACHE = ROOT / "data/legal-scraper/processed/html_cache"

# ── helpers ─────────────────────────────────────────────────────────────────

def load_html(url):
    slug = url.rstrip("/").split("/")[-1]
    for f in CACHE.glob("*.html"):
        if slug in f.name:
            return f.read_text(errors="replace")
    return None

def extract_sections_from_html(html_text):
    """
    Return (tier, list_of_section_ids) from an HTML page.
    Dilapidations-style pages: no anchor IDs on headings → use heading text.
    """
    import re as _re
    soup = BeautifulSoup(html_text, "html.parser")

    # Tier-1: anchor IDs on section headings (e.g. <h3><a id="1.1">)
    anchor_ids = []
    for h in soup.find_all(["h3","h4"]):
        a = h.find("a", id=True)
        if a:
            aid = a["id"].strip()
            if _re.match(r'^\d', aid) or _re.match(r'^annex', aid, _re.I):
                anchor_ids.append(aid)
    if anchor_ids:
        return 1, anchor_ids

    # Tier-2: section numbers as the heading text itself (e.g. <h3>1.1</h3>)
    text_ids = []
    seen = set()
    for h in soup.find_all(["h2","h3","h4"]):
        txt = h.get_text().strip()
        if _re.match(r'^\d+\.\d+', txt) or _re.match(r'^Annex [A-Z]$', txt, _re.I):
            norm = txt.split()[0] if _re.match(r'^\d', txt) else txt
            if norm not in seen:
                text_ids.append(norm)
                seen.add(norm)
    if text_ids:
        return 2, text_ids

    return 3, []

# ── load data ───────────────────────────────────────────────────────────────

v3 = json.load(open(V3))
print(f"Loaded {len(v3)} chunks from {V3.name}")

changes = 0

# ── FIX 1: Part 11 ──────────────────────────────────────────────────────────
for c in v3:
    if c.get("id","").startswith("Part 11"):
        print(f"  FIX 1  Part 11: {c['id'][:70]}")
        print(f"    before: subsections={c.get('subsections',[])}")
        c["subsections"] = ["11.1"]
        print(f"    after : subsections={c['subsections']}")
        changes += 1

# ── FIX 2: PD 31B chunk_000 ─────────────────────────────────────────────────
all_31b_secs = ['1.1','6.1','7.1','8.1','10.1','14.1','17.1','20.1',
                '25.1','28.1','30.1','31.1','32.1','36.1','Schedule']
for c in v3:
    if "Direction 31B" in c.get("id","") and "chunk_000" in c.get("id",""):
        print(f"  FIX 2  PD 31B chunk_000: {c['id'][:70]}")
        print(f"    before: subsections={c.get('subsections',[])} ({len(c.get('subsections',[]))} items)")
        c["subsections"] = all_31b_secs
        print(f"    after : subsections={c['subsections']} ({len(c['subsections'])} items)")
        changes += 1

# ── FIX 3: PAP Dilapidations (Commercial Property) ──────────────────────────
for c in v3:
    if "Physical State of Commercial" in c.get("id","") or "Dilapidations Protocol" in c.get("id",""):
        url = c.get("storageUrl","")
        html = load_html(url)
        if html:
            tier, section_ids = extract_sections_from_html(html)
            print(f"  FIX 3  PAP Dilapidations: {c['id'][:70]}")
            print(f"    before: tier={c.get('page_tier')}  subsections={c.get('subsections',[])} page_all_sections={c.get('page_all_sections',[])[:5]}")
            # Determine which sections appear in this chunk's content
            content_lower = c.get("content","").lower()
            chunk_secs = [s for s in section_ids if s.lower() in content_lower]
            c["page_tier"] = tier
            c["page_all_sections"] = section_ids
            c["page_section_count"] = len(section_ids)
            c["subsections"] = chunk_secs if chunk_secs else section_ids
            print(f"    after : tier={c['page_tier']}  subsections={c['subsections'][:6]} ({len(c['subsections'])} items) page_secs={len(section_ids)}")
            changes += 1
        else:
            print(f"  FIX 3  PAP Dilapidations: HTML still not cached! Skipping.")

# ── FIX 4: PAP Housing Conditions chunk_001 ─────────────────────────────────
housing_remainders = ['8.1','9.1','10.1','11.1','12.1']
for c in v3:
    if "Housing Conditions Claims (England)" in c.get("id","") and "chunk_001" in c.get("id",""):
        print(f"  FIX 4  PAP Housing chunk_001: {c['id'][:70]}")
        print(f"    before: subsections={c.get('subsections',[])}")
        c["subsections"] = housing_remainders
        print(f"    after : subsections={c['subsections']}")
        changes += 1

# ── FIX 5: PAP Judicial Review (single chunk) ───────────────────────────────
for c in v3:
    if "Judicial Review" in c.get("id","") and "Pre-Action Protocol" in c.get("id","") and "chunk" not in c.get("id","").split("Protocol")[-1]:
        print(f"  FIX 5  PAP Judicial Review: {c['id'][:70]}")
        print(f"    before: subsections={c.get('subsections',[])}")
        c["subsections"] = ["annexa","annexb","annexc"]
        print(f"    after : subsections={c['subsections']}")
        changes += 1

# ── FIX 6: PAP Clinical Disputes chunk_000 ──────────────────────────────────
for c in v3:
    if "Resolution of Clinical Disputes" in c.get("id","") and "chunk_000" in c.get("id",""):
        print(f"  FIX 6  PAP Clinical Disputes chunk_000: {c['id'][:70]}")
        print(f"    before: subsections={c.get('subsections',[])}")
        c["subsections"] = ["annex a","annex b","annex c"]
        print(f"    after : subsections={c['subsections']}")
        changes += 1

# ── save ────────────────────────────────────────────────────────────────────
print(f"\nTotal changes: {changes}")
if changes == 0:
    print("WARNING: No changes made — check id patterns!")
    sys.exit(1)

with open(V3, "w") as f:
    json.dump(v3, f, indent=2, ensure_ascii=False)
print(f"Written: {V3}")
