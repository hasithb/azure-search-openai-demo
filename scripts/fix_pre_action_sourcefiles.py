#!/usr/bin/env python3
"""Fix truncated sourcefile values for Pre-Action Protocol documents.

All Pre-Action Protocol documents have sourcefile="Pre" (truncated).
This script derives the correct sourcefile from the sourcepage field
and fixes both the v3 index data and the Upload directory files.
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3_FILE = ROOT / "data" / "legal-scraper" / "processed" / "v3_full_corrected.json"
UPLOAD_DIR = ROOT / "data" / "legal-scraper" / "processed" / "Upload"


def derive_sourcefile_from_sourcepage(sourcepage: str) -> str:
    """Extract a clean sourcefile label from the sourcepage value.

    E.g. "Pre-Action Protocol for Personal Injury Claims" -> "Pre-Action Protocol for Personal Injury Claims"
    """
    # The sourcepage already has the full name, use it directly
    # Strip any trailing section info after a dash-separated section marker
    # But Pre-Action Protocol names often have " - " in the middle,
    # so just use the whole thing
    return sourcepage.strip()


def fix_v3_index():
    """Fix sourcefile in v3_full_corrected.json."""
    print(f"Loading {V3_FILE}...")
    with open(V3_FILE) as f:
        docs = json.load(f)

    fixed = 0
    for doc in docs:
        if doc.get("sourcefile") == "Pre" and doc.get("sourcepage", "").startswith("Pre"):
            new_sourcefile = derive_sourcefile_from_sourcepage(doc["sourcepage"])
            doc["sourcefile"] = new_sourcefile
            fixed += 1

    print(f"Fixed {fixed} documents in v3 index")
    with open(V3_FILE, "w") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Saved {V3_FILE}")


def fix_upload_files():
    """Fix sourcefile in Upload directory JSON files."""
    pre_files = sorted(UPLOAD_DIR.glob("Pre-Action_Protocol_*.json")) + sorted(UPLOAD_DIR.glob("Pre-action_Protocol_*.json"))
    print(f"\nFound {len(pre_files)} Pre-Action Protocol files in Upload/")

    fixed_files = 0
    fixed_docs = 0
    for filepath in pre_files:
        with open(filepath) as f:
            data = json.load(f)

        docs = data if isinstance(data, list) else [data]
        file_changed = False

        for doc in docs:
            if doc.get("sourcefile") == "Pre" and doc.get("sourcepage", "").startswith("Pre"):
                new_sourcefile = derive_sourcefile_from_sourcepage(doc["sourcepage"])
                doc["sourcefile"] = new_sourcefile
                fixed_docs += 1
                file_changed = True

        if file_changed:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            fixed_files += 1

    print(f"Fixed {fixed_docs} documents across {fixed_files} files")


def verify_fix():
    """Verify no more truncated sourcefiles remain."""
    print("\nVerifying fixes...")
    with open(V3_FILE) as f:
        docs = json.load(f)

    remaining = [d for d in docs if d.get("sourcefile") == "Pre"]
    if remaining:
        print(f"WARNING: {len(remaining)} documents still have sourcefile='Pre'")
        for d in remaining[:5]:
            print(f"  sourcepage: {d.get('sourcepage', '')}")
    else:
        print("All Pre-Action Protocol sourcefiles fixed in v3 index")

    # Check Upload files
    upload_remaining = 0
    for filepath in sorted(UPLOAD_DIR.glob("Pre-Action_Protocol_*.json")) + sorted(UPLOAD_DIR.glob("Pre-action_Protocol_*.json")):
        with open(filepath) as f:
            data = json.load(f)
        docs = data if isinstance(data, list) else [data]
        for doc in docs:
            if doc.get("sourcefile") == "Pre":
                upload_remaining += 1

    if upload_remaining:
        print(f"WARNING: {upload_remaining} Upload documents still have sourcefile='Pre'")
    else:
        print("All Pre-Action Protocol sourcefiles fixed in Upload/")


if __name__ == "__main__":
    fix_v3_index()
    fix_upload_files()
    verify_fix()
