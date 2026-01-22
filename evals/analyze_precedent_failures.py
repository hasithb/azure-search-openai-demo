#!/usr/bin/env python3
"""Analyze precedent matching failures."""

import json

with open('/Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2/evals/results/direct_evaluation_results.json') as f:
    data = json.load(f)

print('=== PRECEDENT MATCHING ANALYSIS ===\n')

failures = [r for r in data['detailed_results'] if r['precedent_matching'] < 1.0]
print(f'Total failures: {len(failures)} out of {len(data["detailed_results"])}\n')

print('=== FIRST 5 FAILURES ===\n')
for i, result in enumerate(failures[:5], 1):
    print(f'{i}. Question: {result["question"][:70]}')
    print(f'   Source Type: {result["source_type"]}')
    print(f'   Category: {result["category"]}')
    print(f'   Score: {result["precedent_matching"]:.2f}')
    print(f'   Ground Truth Docs: {result["ground_truth_docs"]}')
    print(f'   Response Docs: {result["response_docs"]}')
    print()

print('\n=== BY SOURCE TYPE ===\n')
for source_type in ['CPR', 'PD', 'Court Guide']:
    type_results = [r for r in data['detailed_results'] if r['source_type'] == source_type]
    type_failures = [r for r in type_results if r['precedent_matching'] < 1.0]
    if type_results:
        pct = (1 - len(type_failures)/len(type_results)) * 100
        print(f'{source_type}: {len(type_failures)} failures out of {len(type_results)} ({pct:.1f}% success)')

print('\n=== COURT GUIDE FAILURE EXAMPLES ===\n')
court_guide_failures = [r for r in failures if r['source_type'] == 'Court Guide']
for i, result in enumerate(court_guide_failures[:3], 1):
    print(f'{i}. Q: {result["question"][:80]}')
    print(f'   Expected: {result["ground_truth_docs"]}')
    print(f'   Got: {result["response_docs"]}')
    print(f'   Score: {result["precedent_matching"]:.2f}')
    print()
