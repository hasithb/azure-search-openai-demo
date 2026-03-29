#!/usr/bin/env python3
"""Compare OLD court guide data (in Upload/) vs NEW data (in outputs_azure_di/).

Shows differences in doc counts, content, fields, and sourcepages before uploading.
"""

import json
import os

OLD_DIR = "data/legal-scraper/processed/Upload"
NEW_DIR = "scripts/court_guides_processing_pipeline/outputs_azure_di"

# Map: guide name -> (old_filename, new_filename)
GUIDES = {
    "Commercial Court": (
        "14.341_JO_Commercial_Court_Guide_FINAL_processed.json",
        "14.341_JO_Commercial_Court_Guide_FINAL_processed.json",
    ),
    "Kings Bench": (
        "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json",
        "35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json",
    ),
    "Court of Appeal": (
        None,
        "35.67_JO_Court-of-Appeal-Civil-Division-Guide_FINAL_WEB_processed.json",
    ),
    "Chancery": (
        "Chancery-Guide-2024-web_processed.json",
        "Chancery-Guide-2024-web_processed.json",
    ),
    "Patents Court": (
        "Patents-Court-Guide-Updated-February-2025_processed.json",
        "Patents-Court-Guide-Updated-February-2025_processed.json",
    ),
    "Senior Courts Costs Office": (
        None,
        "Senior-Courts-Costs-Office-Guide_processed.json",
    ),
    "TCC": (
        "The-Technology-and-Construction-Court-Guide_processed.json",
        "The-Technology-and-Construction-Court-Guide_processed.json",
    ),
}


def load_docs(path):
    with open(path) as f:
        return json.load(f)


def avg_len(docs):
    lengths = [len(d.get("content", "")) for d in docs]
    return sum(lengths) / len(lengths) if lengths else 0


def total_len(docs):
    return sum(len(d.get("content", "")) for d in docs)


def get_fields(docs):
    fields = set()
    for d in docs:
        fields.update(d.keys())
    return fields


def get_categories(docs):
    return set(d.get("category", "") for d in docs if d.get("category"))


def main():
    print("=" * 100)
    print("COURT GUIDE COMPARISON: OLD (Upload/) vs NEW (outputs_azure_di/)")
    print("=" * 100)

    total_old_docs = 0
    total_new_docs = 0
    total_old_chars = 0
    total_new_chars = 0
    new_guides = []

    for name, (old_file, new_file) in GUIDES.items():
        print()
        print("-" * 100)
        print(f"  {name}")
        print("-" * 100)

        old_path = os.path.join(OLD_DIR, old_file) if old_file else None
        new_path = os.path.join(NEW_DIR, new_file) if new_file else None

        old_docs = load_docs(old_path) if old_path and os.path.exists(old_path) else None
        new_docs = load_docs(new_path) if new_path and os.path.exists(new_path) else None

        if old_docs is None:
            print("  NEW GUIDE (not currently in index)")
            nd = len(new_docs)
            nc = total_len(new_docs)
            print(f"  New: {nd} docs, {nc:,} chars total, avg {avg_len(new_docs):.0f} chars/doc")
            print(f"  Fields: {sorted(get_fields(new_docs))}")
            print(f"  Categories: {get_categories(new_docs)}")
            total_new_docs += nd
            total_new_chars += nc
            new_guides.append(name)
            continue

        if new_docs is None:
            print("  NO NEW DATA")
            continue

        total_old_docs += len(old_docs)
        total_new_docs += len(new_docs)
        ot = total_len(old_docs)
        nt = total_len(new_docs)
        total_old_chars += ot
        total_new_chars += nt

        diff_docs = len(new_docs) - len(old_docs)
        diff_chars = nt - ot
        pct_chars = (diff_chars / ot * 100) if ot else 0

        print(f"  Doc count:     OLD={len(old_docs):>4}   NEW={len(new_docs):>4}   diff={diff_docs:+d}")
        print(f"  Total chars:   OLD={ot:>8,}   NEW={nt:>8,}   diff={diff_chars:+,} ({pct_chars:+.1f}%)")
        print(f"  Avg chars/doc: OLD={avg_len(old_docs):>8.0f}   NEW={avg_len(new_docs):>8.0f}")

        # Field comparison
        old_fields = get_fields(old_docs)
        new_fields = get_fields(new_docs)
        if old_fields != new_fields:
            only_old_f = old_fields - new_fields
            only_new_f = new_fields - old_fields
            print("  FIELD DIFF:")
            if only_old_f:
                print(f"    Only in OLD: {sorted(only_old_f)}")
            if only_new_f:
                print(f"    Only in NEW: {sorted(only_new_f)}")
        else:
            print(f"  Fields:        MATCH ({len(old_fields)} fields)")

        # Category comparison
        old_cats = get_categories(old_docs)
        new_cats = get_categories(new_docs)
        if old_cats != new_cats:
            print(f"  CATEGORY DIFF: OLD={old_cats}  NEW={new_cats}")
        else:
            print(f"  Categories:    MATCH {old_cats}")

        # Sourcefile comparison
        old_sf = set(d.get("sourcefile", "") for d in old_docs)
        new_sf = set(d.get("sourcefile", "") for d in new_docs)
        if old_sf != new_sf:
            print(f"  SOURCEFILE DIFF: OLD={old_sf}  NEW={new_sf}")
        else:
            print(f"  Sourcefiles:   MATCH {old_sf}")

        # Sourcepage comparison
        old_sp_set = set(d.get("sourcepage", "") for d in old_docs)
        new_sp_set = set(d.get("sourcepage", "") for d in new_docs)
        common = old_sp_set & new_sp_set
        only_old = old_sp_set - new_sp_set
        only_new = new_sp_set - old_sp_set
        print(f"  Sourcepages:   common={len(common)}  only_old={len(only_old)}  only_new={len(only_new)}")

        if only_old:
            print("    Removed sourcepages:")
            for sp in sorted(only_old)[:10]:
                print(f"      - {sp}")
            if len(only_old) > 10:
                print(f"      ... and {len(only_old)-10} more")

        if only_new:
            print("    Added sourcepages:")
            for sp in sorted(only_new)[:10]:
                print(f"      + {sp}")
            if len(only_new) > 10:
                print(f"      ... and {len(only_new)-10} more")

        # Content comparison for matching sourcepages
        old_by_sp = {}
        for d in old_docs:
            sp = d.get("sourcepage", "")
            if sp:
                old_by_sp[sp] = d
        new_by_sp = {}
        for d in new_docs:
            sp = d.get("sourcepage", "")
            if sp:
                new_by_sp[sp] = d

        matching = 0
        content_diffs = []
        for sp in sorted(common):
            oc = old_by_sp.get(sp, {}).get("content", "")
            nc = new_by_sp.get(sp, {}).get("content", "")
            if oc == nc:
                matching += 1
            else:
                diff_pct = abs(len(nc) - len(oc)) / max(len(oc), 1) * 100
                content_diffs.append((sp, len(oc), len(nc), diff_pct))

        print(f"  Content match: {matching}/{len(common)} identical sourcepages")
        if content_diffs:
            print(f"  Content diffs: {len(content_diffs)} sourcepages differ")
            content_diffs.sort(key=lambda x: -x[3])
            for sp, ol, nl, pct in content_diffs[:8]:
                label = sp[:65]
                print(f"    {label:<65s}  OLD={ol:>5} NEW={nl:>5} ({pct:+.0f}%)")
            if len(content_diffs) > 8:
                print(f"    ... and {len(content_diffs)-8} more")

        # Show a sample content diff for the largest change
        if content_diffs:
            sp_sample = content_diffs[0][0]
            old_content = old_by_sp.get(sp_sample, {}).get("content", "")
            new_content = new_by_sp.get(sp_sample, {}).get("content", "")
            print(f"\n  Sample diff for: {sp_sample}")
            print(f"    OLD first 200 chars: {old_content[:200]!r}")
            print(f"    NEW first 200 chars: {new_content[:200]!r}")

    # Summary
    print()
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"  New guides being added: {len(new_guides)} ({', '.join(new_guides)})")
    print(f"  Updated guides: {len(GUIDES) - len(new_guides)}")
    print(f"  Total docs:  OLD={total_old_docs}  NEW={total_new_docs}  diff={total_new_docs-total_old_docs:+d}")
    print(f"  Total chars: OLD={total_old_chars:,}  NEW={total_new_chars:,}  diff={total_new_chars-total_old_chars:+,}")

    # Check for Circuit Commercial Court Guide (in old but not in new)
    ccg_path = os.path.join(OLD_DIR, "Circuit-Commercial-Court-Guide-2023-web_processed.json")
    if os.path.exists(ccg_path):
        ccg = load_docs(ccg_path)
        print(f"\n  NOTE: Circuit Commercial Court Guide ({len(ccg)} docs) is in OLD but NOT in NEW pipeline.")
        print("  It will NOT be replaced by this upload. Decide whether to keep or remove it separately.")


if __name__ == "__main__":
    main()
