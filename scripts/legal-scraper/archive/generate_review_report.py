#!/usr/bin/env python3
"""
Generate an HTML review report for the V3 extraction.

For every unique CPR page (175 URLs) this report shows:
  - Extraction tier and reason
  - Every section found in the HTML (anchors / headings)
  - Which index chunks are mapped to each section
  - A content preview for each chunk so you can verify the rule text is correct

Run from project root:
  python3 scripts/legal-scraper/generate_review_report.py

Output:
  data/legal-scraper/processed/v3_review_report.html   (open in browser)

HTML cache:
  data/legal-scraper/processed/html_cache/             (saved after first run)
"""

import json
import re
import sys
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from html import escape

import requests

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
CORRECTED_JSON = PROJECT_ROOT / "data/legal-scraper/processed/v3_full_corrected.json"
CACHE_DIR = PROJECT_ROOT / "data/legal-scraper/processed/html_cache"
OUTPUT_HTML = PROJECT_ROOT / "data/legal-scraper/processed/v3_review_report.html"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from html_section_extractor import extract_sections

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Safari/537.36"
    )
}

# ── load corrected index ────────────────────────────────────────────────────
print("Loading v3_full_corrected.json ...")
with open(CORRECTED_JSON, encoding="utf-8") as f:
    all_docs = json.load(f)

# group chunks by URL
url_chunks: dict[str, list[dict]] = defaultdict(list)
for d in all_docs:
    url_chunks[d["storageUrl"]].append(d)

urls = sorted(url_chunks.keys())
print(f"  {len(all_docs)} chunks across {len(urls)} unique URLs")


# ── fetch / cache HTML ─────────────────────────────────────────────────────
def cache_path(url: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("/")[-1])[:60]
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    return CACHE_DIR / f"{slug}_{h}.html"


def get_html(url: str) -> str | None:
    cp = cache_path(url)
    if cp.exists():
        return cp.read_text(encoding="utf-8")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        cp.write_text(resp.text, encoding="utf-8")
        return resp.text
    except Exception as e:
        print(f"  FETCH ERROR {url}: {e}")
        return None


print("Fetching / loading HTML pages (cached on disk)...")
url_html: dict[str, str | None] = {}
for i, url in enumerate(urls, 1):
    url_html[url] = get_html(url)
    if not cache_path(url).exists() or i % 25 == 0:
        time.sleep(0.25)
    if i % 50 == 0:
        print(f"  {i}/{len(urls)} ...")

fresh = sum(1 for u in urls if url_html[u] is not None)
print(f"  HTML available: {fresh}/{len(urls)}")


# ── run extract_sections for every page ───────────────────────────────────
print("Extracting sections from HTML ...")
url_sections = {}
for url, html in url_html.items():
    if html:
        try:
            url_sections[url] = extract_sections(html)
        except Exception as e:
            print(f"  EXTRACT ERROR {url}: {e}")


# ── helpers ────────────────────────────────────────────────────────────────
def snippet(text: str, length: int = 280) -> str:
    """Return first `length` chars of chunk content, stripped of breadcrumbs."""
    lines = [l for l in text.split("\n") if l.strip() and not l.startswith("#")
             and not l.startswith("===") and not l.startswith("Document:")
             and not l.startswith("Part ") or len(l) > 30]
    clean = " ".join(lines[:6])
    # strip [BREADCRUMB > X] tokens
    clean = re.sub(r"\[.*?\]", "", clean).strip()
    return clean[:length] + ("…" if len(clean) > length else "")


TIER_COLOUR = {1: "#d4edda", 2: "#d1ecf1", 3: "#fff3cd"}
TIER_LABEL  = {1: "Tier 1 – HTML anchors", 2: "Tier 2 – heading text", 3: "Tier 3 – page title fallback"}


# ── build HTML ──────────────────────────────────────────────────────────────
print("Building HTML report ...")
parts: list[str] = []

STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         font-size: 13px; margin: 0; background: #f8f9fa; color: #212529; }
  h1 { background: #343a40; color: #fff; padding: 14px 20px; margin: 0; font-size: 18px; }
  .summary { background: #e9ecef; padding: 10px 20px; border-bottom: 1px solid #dee2e6;
             display: flex; gap: 24px; font-size: 12px; }
  .summary span { font-weight: 600; }
  .page-block { background: #fff; margin: 12px 16px; border-radius: 6px;
                border: 1px solid #dee2e6; overflow: hidden; }
  .page-header { padding: 8px 14px; cursor: pointer; display: flex;
                 align-items: baseline; gap: 10px; user-select: none; }
  .page-header:hover { filter: brightness(0.96); }
  .page-title { font-weight: 700; font-size: 13px; flex: 1; }
  .tier-badge { padding: 2px 8px; border-radius: 10px; font-size: 11px;
                font-weight: 600; white-space: nowrap; }
  .chunk-count { color: #6c757d; font-size: 11px; }
  .page-body { border-top: 1px solid #dee2e6; padding: 10px 14px;
               display: none; overflow: auto; }
  .page-block.open .page-body { display: block; }
  .page-url { font-size: 11px; color: #6c757d; margin-bottom: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #f1f3f5; text-align: left; padding: 5px 8px;
       border-bottom: 2px solid #dee2e6; }
  td { padding: 5px 8px; border-bottom: 1px solid #f1f3f5; vertical-align: top; }
  tr:hover td { background: #f8f9fa; }
  .sec-id { font-weight: 700; font-family: monospace; color: #0066cc; }
  .sec-heading { color: #495057; font-size: 11px; }
  .chunk-id-cell { font-family: monospace; font-size: 11px; color: #6c757d; }
  .content-preview { color: #495057; font-size: 11px; max-width: 500px; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 8px;
          font-size: 10px; font-weight: 600; margin-right: 3px; }
  .pill-green { background: #c3e6cb; color: #155724; }
  .pill-yellow { background: #ffeeba; color: #856404; }
  .pill-red { background: #f5c6cb; color: #721c24; }
  .no-chunk { color: #adb5bd; font-style: italic; font-size: 11px; }
  details summary { cursor: pointer; font-size: 11px; color: #495057;
                    padding: 2px 0; }
  .filter-bar { background: #e9ecef; padding: 8px 20px; border-bottom: 1px solid #dee2e6;
                display: flex; gap: 12px; align-items: center; }
  .filter-bar input { padding: 4px 8px; border: 1px solid #ced4da; border-radius: 4px;
                      font-size: 12px; width: 300px; }
  .filter-bar label { font-size: 12px; display: flex; align-items: center; gap: 4px; }
  .hidden { display: none !important; }
</style>
"""

JS = """
<script>
document.querySelectorAll('.page-header').forEach(h => {
  h.addEventListener('click', () => h.closest('.page-block').classList.toggle('open'));
});

function applyFilters() {
  const q = document.getElementById('search').value.toLowerCase();
  const showProblems = document.getElementById('toggle-problems').checked;
  document.querySelectorAll('.page-block').forEach(block => {
    const text = block.textContent.toLowerCase();
    const hasProblem = block.dataset.problem === '1';
    const matchSearch = !q || text.includes(q);
    const matchProblem = !showProblems || hasProblem;
    block.classList.toggle('hidden', !(matchSearch && matchProblem));
  });
  const visible = document.querySelectorAll('.page-block:not(.hidden)').length;
  document.getElementById('visible-count').textContent = visible + ' pages shown';
}

document.getElementById('search').addEventListener('input', applyFilters);
document.getElementById('toggle-problems').addEventListener('change', applyFilters);

document.getElementById('expand-all').addEventListener('click', () => {
  document.querySelectorAll('.page-block:not(.hidden)').forEach(b => b.classList.add('open'));
});
document.getElementById('collapse-all').addEventListener('click', () => {
  document.querySelectorAll('.page-block').forEach(b => b.classList.remove('open'));
});
</script>
"""

# stats
tier1_count = sum(1 for ps in url_sections.values() if ps.tier == 1)
tier2_count = sum(1 for ps in url_sections.values() if ps.tier == 2)
tier3_count = sum(1 for ps in url_sections.values() if ps.tier == 3)

parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<title>V3 Extraction Review – CPR Legal RAG</title>
{STYLE}
</head>
<body>
<h1>V3 Extraction Review — CPR Legal RAG</h1>
<div class="summary">
  <div>Total pages: <span>{len(urls)}</span></div>
  <div>Total chunks: <span>{len(all_docs)}</span></div>
  <div style="color:#155724">Tier 1 (HTML anchors): <span>{tier1_count}</span></div>
  <div style="color:#0c5460">Tier 2 (heading text): <span>{tier2_count}</span></div>
  <div style="color:#856404">Tier 3 (title fallback): <span>{tier3_count}</span></div>
  <div style="margin-left:auto" id="visible-count">{len(urls)} pages shown</div>
</div>
<div class="filter-bar">
  <input type="text" id="search" placeholder="Filter by page title, section ID, rule number…">
  <label><input type="checkbox" id="toggle-problems"> Show pages with issues only</label>
  <button id="expand-all" style="font-size:11px;padding:3px 8px">Expand all</button>
  <button id="collapse-all" style="font-size:11px;padding:3px 8px">Collapse all</button>
</div>
""")

problem_pages = 0

for url in urls:
    chunks = sorted(url_chunks[url], key=lambda d: d["id"])
    ps = url_sections.get(url)
    html_available = ps is not None
    tier = ps.tier if ps else 0
    tier_reason = ps.tier_reason if ps else "no_html"
    all_section_ids = ps.all_section_ids if ps else []

    # detect problems:
    # 1. all chunks on this page have the same tier-3 (page title) subsection
    # 2. multi-chunk but all get same subsection_id (no differentiation)
    chunk_sids = [d["subsection_id"] for d in chunks]
    unique_sids = set(chunk_sids)
    tier3_all = (tier == 3 and len(chunks) > 1)
    no_differentiation = (len(chunks) > 1 and len(unique_sids) == 1 and tier < 3)
    is_problem = tier3_all or no_differentiation

    if is_problem:
        problem_pages += 1

    bg = TIER_COLOUR.get(tier, "#ffffff")
    tier_label = TIER_LABEL.get(tier, "unknown")

    page_title = chunks[0]["sourcepage"] if chunks else url.split("/")[-1]

    # build section → chunks mapping using full chunk coverage
    # (section_id → list of chunks whose `subsections` include this section)
    sid_to_chunks: dict[str, list[dict]] = defaultdict(list)
    for d in chunks:
      chunk_section_ids = set(d.get("subsections") or [])
      if d.get("subsection_id"):
        chunk_section_ids.add(d["subsection_id"])
      for section_id in chunk_section_ids:
        sid_to_chunks[section_id].append(d)

    # coverage derived from full chunk subsection lists
    covered_sids = {s for s in all_section_ids if sid_to_chunks.get(s)}
    uncovered_html_sids = [s for s in all_section_ids if s not in covered_sids]

    chunk_all_sids = set()
    for d in chunks:
      chunk_all_sids.update(d.get("subsections") or [])
      if d.get("subsection_id"):
        chunk_all_sids.add(d["subsection_id"])
    orphan_chunk_sids = [s for s in chunk_all_sids if s not in all_section_ids and s not in (page_title.upper(),)]

    problem_attr = ' data-problem="1"' if is_problem else ''

    # ── page header ──
    badges = f'<span class="tier-badge" style="background:{bg}">{tier_label}</span>'
    if is_problem:
        badges += ' <span class="pill pill-yellow">⚠ review</span>'

    parts.append(f"""
<div class="page-block"{problem_attr}>
  <div class="page-header" style="background:{bg}">
    <span class="page-title">{escape(page_title)}</span>
    {badges}
    <span class="chunk-count">{len(chunks)} chunk{'s' if len(chunks) != 1 else ''}</span>
  </div>
  <div class="page-body">
    <div class="page-url">🔗 <a href="{escape(url)}" target="_blank">{escape(url)}</a></div>
    <div style="font-size:11px;color:#6c757d;margin-bottom:8px">
      Tier reason: <code>{escape(tier_reason)}</code> &nbsp;|&nbsp;
      Sections found in HTML: <strong>{len(all_section_ids)}</strong>
      {(' &nbsp;|&nbsp; <span class="pill pill-yellow">⚠ '+ str(len(uncovered_html_sids)) +' HTML sections not in any chunk</span>') if uncovered_html_sids else ''}
    </div>
""")

    # ── table: HTML sections with their chunk mappings ──
    if all_section_ids:
        parts.append("""
    <table>
      <thead>
        <tr>
          <th style="width:100px">Section ID</th>
          <th style="width:220px">HTML heading text</th>
          <th>Chunk(s) mapped here</th>
          <th>Content preview</th>
        </tr>
      </thead>
      <tbody>
""")
        for si in ps.sections:
            mapped = sid_to_chunks.get(si.anchor_id, [])
            heading_text = escape(si.heading_text or "")[:80] if si else ""
            sid_html = f'<span class="sec-id">{escape(si.anchor_id)}</span>'
            heading_html = f'<br><span class="sec-heading">{heading_text}</span>' if heading_text else ""

            if mapped:
                chunk_cells = []
                for d in mapped:
                    prev = escape(snippet(d["content"]))
                    chunk_cells.append(
                        f'<div class="chunk-id-cell">{escape(d["id"])}</div>'
                        f'<div class="content-preview">{prev}</div>'
                    )
                chunk_html = "".join(chunk_cells)
                preview_html = ""  # already inline above
            else:
                chunk_html = '<span class="no-chunk">— no chunk mapped to this section</span>'
                preview_html = ""

            parts.append(f"""        <tr>
          <td>{sid_html}{heading_html}</td>
          <td></td>
          <td colspan="2">{chunk_html}</td>
        </tr>
""")

        # show chunks with no overlap against HTML section IDs (true orphans)
        for d in chunks:
          chunk_sids = set(d.get("subsections") or [])
          if d.get("subsection_id"):
            chunk_sids.add(d["subsection_id"])
          if not (chunk_sids & set(all_section_ids)):
                prev = escape(snippet(d["content"]))
                sid_html = f'<span class="sec-id" style="color:#856404">{escape(d["subsection_id"])}</span>'
                parts.append(f"""        <tr style="background:#fff8e1">
          <td>{sid_html}<br><span class="sec-heading" style="color:#856404">(not in HTML sections)</span></td>
          <td></td>
          <td class="chunk-id-cell">{escape(d["id"])}</td>
          <td class="content-preview">{prev}</td>
        </tr>
""")
        parts.append("      </tbody></table>\n")

    else:
        # tier 3 — no HTML sections, just show chunks
        parts.append('<p style="font-size:11px;color:#856404;margin:6px 0">⚠ No sections extracted from HTML (tier 3 fallback — page title used as subsection_id)</p>')
        parts.append("""<table>
  <thead><tr><th>Chunk</th><th>subsection_id</th><th>Content preview</th></tr></thead>
  <tbody>
""")
        for d in chunks:
            prev = escape(snippet(d["content"]))
            parts.append(f"""  <tr>
    <td class="chunk-id-cell">{escape(d["id"])}</td>
    <td class="sec-id">{escape(d["subsection_id"])}</td>
    <td class="content-preview">{prev}</td>
  </tr>
""")
        parts.append("  </tbody></table>\n")

    parts.append("  </div>\n</div>\n")

# ── footer / JS ────────────────────────────────────────────────────────────
parts.append(f"""
<div style="padding:16px 20px;font-size:11px;color:#6c757d;border-top:1px solid #dee2e6">
  Generated from <code>v3_full_corrected.json</code> — {len(all_docs)} chunks, {len(urls)} pages.
  Pages with ⚠ review flag: {problem_pages}
</div>
{JS}
</body></html>
""")

OUTPUT_HTML.write_text("".join(parts), encoding="utf-8")
print(f"\nDone! Report written to:\n  {OUTPUT_HTML}")
print(f"  Size: {OUTPUT_HTML.stat().st_size / 1024:.1f} KB")
print(f"  Pages flagged for review: {problem_pages}")
print(f"\nOpen with:  open '{OUTPUT_HTML}'")
