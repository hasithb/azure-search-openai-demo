#!/usr/bin/env python3
"""Validate ground truth JSONL files."""
import json
import sys
from pathlib import Path

def validate(filepath):
    errors = 0
    cats = {}
    total = 0
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                j = json.loads(line)
                cat = j.get("category", "NONE")
                cats[cat] = cats.get(cat, 0) + 1
                for field in ["question", "truth", "source_type", "category"]:
                    if field not in j:
                        print(f"Line {i}: missing {field}")
                        errors += 1
            except json.JSONDecodeError as e:
                print(f"Line {i}: JSON error: {e}")
                errors += 1

    print(f"File: {filepath}")
    print(f"Total entries: {total}")
    print(f"Errors: {errors}")
    print()
    print("=== COVERAGE BY CATEGORY ===")
    for k, v in sorted(cats.items()):
        print(f"  {v:3d} {k}")
    print(f"  TOTAL: {sum(cats.values())}")
    return errors

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    filepath = sys.argv[1] if len(sys.argv) > 1 else str(root / "evals" / "ground_truth_comprehensive.jsonl")
    sys.exit(validate(filepath))
