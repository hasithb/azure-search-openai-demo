#!/usr/bin/env python3
"""
Produce a human-readable line-by-line review of every record in final_all_sections.json.
Checks each record for: ID, URL, tier, sections, content first line, anomalies.
Output: data/legal-scraper/processed/manual_review.txt
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SRC = ROOT / "data/legal-scraper/processed/final_all_sections.json"
OUT = ROOT / "data/legal-scraper/processed/manual_review.txt"

with open(SRC) as f:
    docs = json.load(f)

lines = []
anomalies = []

lines.append("=" * 100)
lines.append(f"MANUAL LINE-BY-LINE REVIEW: final_all_sections.json  ({len(docs)} records)")
lines.append("=" * 100)
lines.append("")

for i, d in enumerate(docs, 1):
    doc_id        = d.get("id", "MISSING")
    url           = d.get("storageUrl", "MISSING")
    slug          = url.split("/")[-1] if url else "?"
    category      = d.get("category", "")
    subsection_id = d.get("subsection_id", "-")
    subsections   = d.get("subsections", [])
    tier          = d.get("page_tier")
    page_secs     = d.get("page_all_sections", [])
    page_sec_cnt  = d.get("page_section_count", 0)
    content_lines = d.get("content_lines", [])
    line_count    = d.get("content_line_count", 0)

    # --- Anomaly detection ---
    issues = []

    if not doc_id or doc_id == "MISSING":
        issues.append("MISSING id")
    if not url or url == "MISSING":
        issues.append("MISSING storageUrl")
    if subsection_id in (None, "", "-"):
        issues.append("blank subsection_id")
    if tier == 3 and subsections:
        issues.append(f"tier-3 but has {len(subsections)} subsections (expected empty)")
    if tier in (1, 2) and page_sec_cnt > 0 and not subsections:
        issues.append("tier-1/2 page with sections but chunk claims none (coverage gap)")
    for s in subsections:
        if s not in page_secs:
            issues.append(f"claimed section '{s}' not in page_all_sections")
    if not content_lines:
        issues.append("empty content_lines")
    if line_count != len(content_lines):
        issues.append(f"content_line_count {line_count} != actual {len(content_lines)}")

    first_line = content_lines[0][:100] if content_lines else "(empty)"
    last_line  = content_lines[-1][:80] if content_lines else "(empty)"

    status = "⚠ ISSUE" if issues else "✓ OK"

    lines.append(f"[{i:03d}] {status}  tier={tier}")
    lines.append(f"       ID         : {doc_id[:110]}")
    lines.append(f"       URL        : {url}")
    lines.append(f"       subsec_id  : {subsection_id}")
    lines.append(f"       subsections: {subsections[:10]}")
    lines.append(f"       page_secs  : {page_secs[:12]}")
    lines.append(f"       content    : {line_count} lines | first: {first_line}")
    if issues:
        for iss in issues:
            lines.append(f"       *** {iss}")
        anomalies.append({"idx": i, "id": doc_id[:80], "issues": issues})
    lines.append("")

# Summary
lines.append("=" * 100)
lines.append(f"ANOMALY SUMMARY  ({len(anomalies)} records with issues)")
lines.append("=" * 100)
if anomalies:
    for a in anomalies:
        lines.append(f"  [{a['idx']:03d}] {a['id']}")
        for iss in a["issues"]:
            lines.append(f"          -> {iss}")
else:
    lines.append("  None — all 314 records are clean.")
lines.append("")

# Stats
tiers = {}
for d in docs:
    t = d.get("page_tier")
    tiers[t] = tiers.get(t, 0) + 1
lines.append("TIER DISTRIBUTION:")
for t, c in sorted(tiers.items(), key=lambda x: (x[0] is None, x[0])):
    lines.append(f"  tier {t}: {c} chunks")

text = "\n".join(lines)
OUT.write_text(text, encoding="utf-8")
print(f"Written: {OUT}  ({len(lines)} lines, {len(anomalies)} anomalies)")
