# V2 Index Evaluation Results
**Date:** 2026-01-18  
**Index:** legal-court-rag-index-v2  
**Documents:** 267 CPR documents  
**Model:** searchagent (gpt-4.1-mini)  
**Ground Truth Entries:** 62

## Summary

The v2 index with pre-indexed subsection fields has **dramatically improved citation accuracy** compared to the baseline (38.6% precedent matching).

### Overall Metrics

| Metric | Score | Status | vs Baseline |
|--------|-------|--------|-------------|
| **precedent_matching** | **94.0%** | ✅ | **+144% improvement** |
| citation_format_compliance | 100% | ✅ | +100% |
| legal_terminology_accuracy | 100% | ✅ | +100% |
| statute_citation_accuracy | 65.1% | ⚠️ | N/A |
| citation_rate | 100% | ✅ | +100% |

### By Source Type

| Source Type | Count | Statute | Terminology | Precedent |
|-------------|-------|---------|-------------|-----------|
| **CPR** | 11 | 100% | 100% | **100%** |
| **PD** | 9 | 88.9% | 100% | 97.8% |
| **Court Guide** | 42 | 50.8% | 100% | 91.6% |

### By Category

| Category | Count | Precedent |
|----------|-------|-----------|
| Civil Procedure Rules and Practice Directions | 20 | 99.0% |
| King's Bench Division | 13 | 99.2% |
| Patents Court | 8 | 98.8% |
| Circuit Commercial Court | 6 | 90.8% |
| Technology and Construction Court | 7 | 81.9% |
| Commercial Court | 8 | 81.2% |

## Key Findings

### ✅ What's Working

1. **Subsection Fields Effective**: Pre-indexed `subsection_id` and `subsections[]` fields enable accurate citation navigation
   - CPR documents achieve **100% precedent matching**
   - Practice Directions achieve **97.8% precedent matching**

2. **Citation Format Perfect**: 100% compliance with `[Document Name]` format
   - No comma-separated citations
   - Proper source document attribution

3. **Legal Terminology**: 100% accuracy using UK terms (claimant, disclosure, etc.)

4. **High Overall Accuracy**: 94% precedent matching across all document types

### ⚠️ Areas for Improvement

1. **Statute Citation Accuracy (65.1%)**:
   - Court Guide documents (50.8%) don't always contain explicit CPR Part/Rule numbers
   - Ground truth data may not include all relevant statutory references
   - Model sometimes cites related CPR rules not in ground truth

2. **Court Guide Documents**:
   - More narrative/procedural content vs. statutory rules
   - Requires different citation strategy (guide sections vs. CPR rules)

## Technical Details

### V2 Index Schema Enhancements

Added three new fields to enable accurate subsection lookups:

1. **subsection_id** (String, filterable, facetable)
   - First subsection ID in chunk (e.g., "35.1")
   - Enables direct filtering for specific rules

2. **subsections** (Collection(String), filterable)
   - All subsection IDs in chunk (e.g., ["35.1", "35.2", "35.3"])
   - Enables comprehensive rule coverage lookup

3. **updated** (String, filterable, sortable)
   - CPR page last modified date from Ministry of Justice
   - Enables filtering by document currency

### Subsection Extraction

- **Extraction Rate**: 84.6% (from prior testing)
- **Methods**:
  - Markdown headers: `## 35.1 Expert's right to ask court for directions`
  - Breadcrumbs: `[PART 35 > 35.1 Expert's right to ask court for directions]`
  - Bare text: `35.1 Expert's right to ask court for directions`

### Model Configuration

- **Issue**: gpt-5-nano (reasoning model) used all tokens for internal reasoning, returned empty responses
- **Solution**: Switched to `searchagent` deployment (gpt-4.1-mini)
- **Tokens**: 1000 max_completion_tokens (sufficient for legal answers)

## Comparison to Baseline

| Metric | Baseline | V2 Index | Improvement |
|--------|----------|----------|-------------|
| Precedent Matching | 38.6% | **94.0%** | **+144%** |
| CPR Precedent | Unknown | **100%** | - |
| PD Precedent | Unknown | 97.8% | - |
| Court Guide Precedent | Unknown | 91.6% | - |

## Next Steps

### Recommended Improvements

1. **Refine Statute Citation Metric**:
   - Adjust scoring for Court Guide documents (less emphasis on CPR rule numbers)
   - Consider alternative metrics for procedural vs. statutory content

2. **Enhance Court Guide Documents**:
   - Extract section numbers and references for better citation
   - Map guide sections to related CPR rules

3. **Increase Subsection Extraction**:
   - Current 84.6% extraction rate
   - Target 95%+ by improving regex patterns

4. **Consider Hybrid Approach**:
   - Use searchagent for standard queries
   - Use gpt-5-nano (reasoning model) for complex legal analysis (with higher token limits)

## Conclusion

The v2 index with pre-indexed subsection fields has **achieved the goal** of dramatically improving citation accuracy:

- ✅ **94% precedent matching** (vs 38.6% baseline) - **+144% improvement**
- ✅ **100% CPR precedent matching** - perfect accuracy for core legal rules
- ✅ **100% citation format compliance** - correct document attribution
- ✅ **100% legal terminology** - proper UK legal terms

The subsection field strategy is **highly effective** and should be retained for production use.

### Files Generated

- `evals/results/direct_evaluation_results.json`: Full detailed results with all 62 evaluations
- `evals/V2_EVALUATION_SUMMARY.md`: This summary document

### Next Evaluation

To re-run evaluation:

```bash
cd evals
python run_direct_evaluation.py                    # All 62 entries
python run_direct_evaluation.py --max-entries 10  # Quick test
```
