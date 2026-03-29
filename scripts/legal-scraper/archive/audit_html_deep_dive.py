#!/usr/bin/env python3
"""
Deep dive into pages with standalone anchors but no heading anchors.
Check if standalone anchors are:
1. Legacy auto-generated IDs (useless)
2. Rule numbers placed OUTSIDE headings (adjacent to them)
3. Something else exploitable
"""
import json
import re
import time
import requests
from bs4 import BeautifulSoup


def main():
    with open("data/legal-scraper/processed/html_anchor_audit.json") as f:
        audit = json.load(f)

    # Find pages with standalone anchors but NO heading anchors
    interesting = [r for r in audit
                   if r.get("status") == "ok"
                   and r.get("total_anchors_in_headings", 0) == 0
                   and r.get("standalone_anchor_count", 0) > 0]

    print(f"Pages with standalone-only anchors: {len(interesting)}")
    print()

    headers = {"User-Agent": "Mozilla/5.0"}

    # Sample representative pages
    sample_urls = [
        "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part35",
        "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part58",
        "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part61",
        "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part41",
        "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part14",
        "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part15",
    ]

    for url in sample_urls:
        short = url.replace("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/", "")
        print(f"\n{'='*70}")
        print(f"PAGE: {short}")
        print(f"{'='*70}")

        resp = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")

        content = (
            soup.find("div", class_="article-content")
            or soup.find("div", class_="content")
            or soup.find("main")
            or soup.find("body")
        )

        all_a = content.find_all("a", id=True)

        # Show first 10 anchors with their parent context
        for a in all_a[:10]:
            parent = a.parent
            grandparent = parent.parent if parent else None
            parent_tag = parent.name if parent else "none"
            gp_tag = grandparent.name if grandparent else "none"
            parent_class = parent.get("class", []) if parent else []
            anchor_text = a.get_text(strip=True)[:50]
            parent_text = parent.get_text(strip=True)[:80] if parent else ""

            print(f"  <a id='{a['id']}'> text='{anchor_text}'")
            print(f"    parent: <{parent_tag} class={parent_class}> '{parent_text}'")
            print(f"    grandparent: <{gp_tag}>")
            print()

        if len(all_a) > 10:
            print(f"  ... and {len(all_a) - 10} more anchors")

        # Check heading structure
        print(f"\n  HEADINGS on this page:")
        for tag in ["h2", "h3"]:
            for h in content.find_all(tag)[:5]:
                has_anchor = bool(h.find("a", id=True))
                text = h.get_text(strip=True)[:80]
                print(f"    <{tag}> anchor_inside={has_anchor} '{text}'")

        # Check if anchors are immediately BEFORE headings (siblings)
        print(f"\n  ANCHORS ADJACENT TO HEADINGS:")
        found_adjacent = 0
        for a in all_a[:20]:
            next_sib = a.find_next_sibling()
            if next_sib and next_sib.name in ["h1", "h2", "h3", "h4"]:
                print(f"    <a id='{a['id']}'> FOLLOWED BY <{next_sib.name}> '{next_sib.get_text(strip=True)[:60]}'")
                found_adjacent += 1
            parent = a.parent
            if parent:
                next_of_parent = parent.find_next_sibling()
                if next_of_parent and next_of_parent.name in ["h1", "h2", "h3", "h4"]:
                    print(f"    <a id='{a['id']}'> (via parent <{parent.name}>) BEFORE <{next_of_parent.name}> '{next_of_parent.get_text(strip=True)[:60]}'")
                    found_adjacent += 1
        if found_adjacent == 0:
            print("    (none found)")

        time.sleep(0.3)

    # Classification of all 62 pages
    print(f"\n\n{'='*70}")
    print("CLASSIFICATION OF 62 PAGES WITHOUT HEADING ANCHORS")
    print(f"{'='*70}")

    no_heading = [r for r in audit if r.get("status") == "ok" and r.get("total_anchors_in_headings", 0) == 0]

    zero_anchors_zero_headings = []
    zero_anchors_with_headings = []
    standalone_with_headings = []
    standalone_no_headings = []

    for r in no_heading:
        h_count = sum(r.get("heading_counts", {}).get(t, 0) for t in ["h2", "h3", "h4"])
        sa_count = r.get("standalone_anchor_count", 0)
        short = r["url"].replace("https://www.justice.gov.uk/courts/procedure-rules/civil/rules/", "")

        if sa_count == 0 and h_count <= 1:
            zero_anchors_zero_headings.append(f"  {short} (headings: {r.get('heading_counts')})")
        elif sa_count == 0 and h_count > 1:
            zero_anchors_with_headings.append(f"  {short} (h2={r['heading_counts'].get('h2',0)}, h3={r['heading_counts'].get('h3',0)})")
        elif sa_count > 0 and h_count > 1:
            standalone_with_headings.append(f"  {short} (standalone={sa_count}, h2={r['heading_counts'].get('h2',0)}, h3={r['heading_counts'].get('h3',0)})")
        else:
            standalone_no_headings.append(f"  {short} (standalone={sa_count}, headings={r.get('heading_counts')})")

    print(f"\nA) No anchors + minimal headings (simple/short pages): {len(zero_anchors_zero_headings)}")
    for x in zero_anchors_zero_headings:
        print(x)

    print(f"\nB) No anchors + HAS headings (structured but no IDs): {len(zero_anchors_with_headings)}")
    for x in zero_anchors_with_headings:
        print(x)

    print(f"\nC) Standalone anchors + HAS headings (anchors outside headings): {len(standalone_with_headings)}")
    for x in standalone_with_headings:
        print(x)

    print(f"\nD) Other: {len(standalone_no_headings)}")
    for x in standalone_no_headings:
        print(x)


if __name__ == "__main__":
    main()
