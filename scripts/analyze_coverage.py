#!/usr/bin/env python3
"""Analyze ground truth coverage across all source categories."""
import json
import re
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Collect all questions across all ground truth files
all_questions = []
for f in ROOT.glob("evals/ground_truth_*.jsonl"):
    if f.name == "ground_truth_multimodal.jsonl":
        continue
    for line in f.open():
        line = line.strip()
        if line:
            j = json.loads(line)
            all_questions.append(j)

# Count by category
cats = collections.Counter(j.get("category", "NONE") for j in all_questions)
print("=== TOTAL QUESTIONS BY CATEGORY ===")
for k, v in sorted(cats.items()):
    print(f"  {v:3d} {k}")
print(f"  TOTAL: {len(all_questions)}")
print()

# Extract CPR parts mentioned
cpr_parts = set()
pd_numbers = set()
for j in all_questions:
    q = j.get("question", "") + " " + j.get("truth", "")
    for m in re.finditer(r"Part\s+(\d+[A-Z]?)\b", q):
        cpr_parts.add(m.group(1))
    for m in re.finditer(r"Practice\s+Direction\s+(\d+[A-Z]?)\b", q):
        pd_numbers.add(m.group(1))

print("=== CPR PARTS REFERENCED ===")
parts_sorted = sorted(cpr_parts, key=lambda x: (int(re.match(r"(\d+)", x).group(1)), x))
print(parts_sorted)
print()
print("=== PRACTICE DIRECTIONS REFERENCED ===")
pds_sorted = sorted(pd_numbers, key=lambda x: (int(re.match(r"(\d+)", x).group(1)), x))
print(pds_sorted)
print()

# Missing CPR parts (1-89)
all_parts = set(str(i) for i in range(1, 90))
missing_parts = sorted(all_parts - cpr_parts, key=lambda x: int(x))
print(f"=== CPR PARTS NOT IN GROUND TRUTH ({len(missing_parts)}) ===")
print(missing_parts)
print()

# Check Chancery Division coverage 
chancery_qs = [j for j in all_questions if "Chancery" in j.get("category", "")]
print(f"=== CHANCERY DIVISION QUESTIONS: {len(chancery_qs)} ===")
for q in chancery_qs:
    print(f"  - {q['question'][:80]}...")
