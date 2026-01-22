# Court Guides Processing Pipeline — Implementation Plan

**Status:** Planning (Not yet implemented)
**Last Updated:** January 4, 2026
**Priority:** Low (Court guides update ~annually)

***

## Executive Summary

This document outlines a strategic plan to recover, consolidate, and automate the PDF-to-JSON conversion pipeline for UK court guides. Currently, the pipeline is **partially documented but missing critical extraction scripts**, and lacks automated change detection/monitoring capabilities.

### Current State

- ✅ **5 processed JSON files** ready for Azure AI Search
- ✅ **Documentation** of pipeline steps and field mappings
- ⚠️ **Missing extraction scripts** (8+ referenced but not version-controlled)
- ⚠️ **No automated monitoring** for source document changes
- ⚠️ **Manual process** for PDF extraction and validation

### Goal State

- 🎯 **All scripts version-controlled** and reproducible
- 🎯 **Single automated pipeline** for PDF → JSON → Embedding → Azure Search
- 🎯 **Change monitoring system** to detect when UK Judiciary updates guides
- 🎯 **Integrated validation** in CI/CD to ensure quality before indexing
- 🎯 **Clear handoff documentation** for future maintenance

***

## Phase 1: Script Recovery & Documentation (2-3 weeks)

### 1.1 Recover Missing Extraction Scripts

**Missing scripts referenced in `SCRIPTS_USED.md`:**

- `scripts/extract_court_guides.py` — Docling-based PDF extraction
- `scripts/process_court_guides.py` — JSON processing and grouping
- `scripts/reconstruct_and_compare_all.py` — Content validation
- `scripts/rebuild_kings_bench_from_md.py` — Special case handling
- `scripts/add_kings_bench_annexes.py` — Annex stitching
- `scripts/fix_content_join.py` — Post-processing fixes
- `scripts/fix_kings_bench_structure.py` — Hierarchy fixes
- `scripts/compare_content.py` — TOC/content comparison
- `scripts/validate_missing_is_in_titles.py` — Coverage validation
- `scripts/validate_against_markdown.py` — Markdown comparison

**Action Items:**

1. Search codebase or archived repos for these scripts
1. If unavailable, recreate based on `SCRIPTS_USED.md` specifications
1. Add to `court_guides_processing_pipeline/scripts/` with documentation
1. Test on existing court guides to validate correctness

**Success Criteria:**

- All extraction scripts produce identical JSON outputs to current `outputs/` directory
- Scripts have clear docstrings and argument documentation
- Each script has a corresponding test/validation

***

## Phase 2: Evaluate PDF Parsing Strategy (1-2 weeks)

### 2.1 Docling vs Azure Document Intelligence

**Current approach:** IBM Docling (open-source, local processing)
**Alternative approach:** Azure Document Intelligence (existing in `prepdocslib`)

| Aspect | Docling | Azure DI |
|--------|---------|----------|
| **Cost** | Free (open-source) | Pay-per-call (~$2-6/doc) |
| **Speed** | Local, ~30-60s per guide | API calls, ~2-5s per page |
| **OCR Quality** | Good for scanned docs | Excellent, Microsoft-grade |
| **Integration** | New dependency | Already in `prepdocslib` |
| **Legal Domain** | Generic document parser | Multi-domain support |
| **Maintenance** | Community-maintained | Microsoft-supported |

**Recommendation:**

- **Short-term (2026):** Keep Docling; it's proven and cost-effective for ~5 guides/year
- **Long-term (2027+):** Consider Azure DI if:
  - Court guides become harder to parse (format changes)
  - Integration with main pipeline is needed
  - Budget allows for API costs

**Action Items:**

1. Run both Docling and Azure DI on 1-2 complex guides (Chancery, Patents)
1. Compare output quality, speed, and content retention
1. Document findings in `docs/PARSER_COMPARISON.md`
1. Make final decision based on results

***

## Phase 3: Consolidate Processing Pipeline (3-4 weeks)

### 3.1 Create Unified Pipeline Script

**Output:** `scripts/legal-scraper/process_court_guides_pipeline.py`

**Features:**

- Single entry point for entire PDF → Azure Search workflow
- Modular design (extraction → processing → validation → upload)
- Support for dry-run and staging modes
- Clear progress reporting and error handling
- Reuses existing `legal-scraper/` utilities where possible

**Integration Points:**

```text
process_court_guides_pipeline.py
├── extract_stage()
│   └── Uses: Docling (or Azure DI)
│   └── Outputs: JSON to staging directory
├── process_stage()
│   └── Uses: scripts/process_court_guides.py logic
│   └── Outputs: processed/*.json + review/*.json
├── validate_stage()
│   └── Uses: existing validation scripts from evals/
│   └── Checks: content retention, field mappings, embeddings
├── embed_stage()
│   └── Uses: existing embeddings.py from prepdocslib
│   └── Generates: 3072-dim vectors via Azure OpenAI
└── upload_stage()
    └── Uses: existing Azure Search utilities
    └── Supports: --dry-run, --staging, --production
```

**Success Criteria:**

- Single command to process all court guides
- Produces identical outputs to current manual process
- Can be scheduled as a cron job or triggered manually
- Full audit trail and error logging

***

## Phase 4: Add Change Monitoring (2-3 weeks)

### 4.1 Automated Court Guide Update Detection

**Problem:** Court guides are manually checked; changes might be missed.

**Solution:** Multi-layered monitoring approach:

#### Layer 1: Scheduled Download & Hash Check

```python
# scripts/legal-scraper/monitor_court_guides.py

monitor_court_guides()
├── Schedule: Weekly (Monday 9 AM)
├── Actions:
│   ├── Download latest PDFs from justice.gov.uk
│   ├── Compare file hashes with last known versions
│   ├── If changed:
│   │   ├── Log alert to Application Insights
│   │   ├── Trigger email notification to maintainers
│   │   ├── (Optionally) auto-trigger pipeline
│   └── Store checksums in Azure Blob Storage
└── Outputs: change_detection_report.json
```

#### Layer 2: Content Diff Analysis

```python
# If PDF changed, compare extracted content

compare_extracted_content()
├── Extract old and new PDFs
├── Compare sections using fuzzy matching
├── Identify:
│   ├── New sections added
│   ├── Sections removed
│   ├── Sections with >5% content changes
└── Generate: content_diff_report.md
```

#### Layer 3: Integration with CI/CD

- Add GitHub Action to run `monitor_court_guides.py` weekly
- Auto-open PR if changes detected with diff summary
- Require manual approval before re-indexing (legal content = critical)

**Implementation:**

1. `scripts/legal-scraper/monitor_court_guides.py` — Download & hash check
1. `scripts/legal-scraper/compare_guide_versions.py` — Content diffing
1. `.github/workflows/monitor-court-guides.yml` — Weekly schedule
1. Update `docs/monitoring.md` with runbook

**Success Criteria:**

- Detects new guide versions within 24 hours
- Low false positive rate
- Clear diff reports for manual review

***

## Phase 5: Validation & Integration (2-3 weeks)

### 5.1 Add to CI/CD Pipeline

**New tests in `tests/`:**

```python
test_court_guides_pipeline.py
├── test_extraction_completeness()
│   └── Verify all sections extracted
├── test_json_schema_validation()
│   └── Verify outputs match Azure Search schema
├── test_embedding_generation()
│   └── Verify 3072-dim embeddings created
└── test_azure_search_upload()
    └── Verify documents queryable in index
```

**Integration with existing workflows:**

- Add to `test_integration.sh`
- Run on every commit to `scripts/legal-scraper/`
- Run weekly for monitoring updates

### 5.2 Documentation for Operators

Create `docs/OPERATOR_RUNBOOK.md`:

- How to manually run pipeline
- How to monitor guide changes
- Troubleshooting common failures
- How to roll back if issues detected
- Emergency contacts

***

## Phase 6: Knowledge Transfer & Maintenance (1 week)

### 6.1 Documentation Package

Create `docs/ARCHITECTURE.md`:

- High-level pipeline architecture diagram
- Docling configuration reference
- Azure Search field mappings
- Error handling and recovery procedures

Update main `docs/` folder:

- Add reference to court guides pipeline in `data_ingestion.md`
- Link to new runbook from `monitoring.md`
- Add court guides to architecture documentation

### 6.2 Handoff

- Document any known limitations or workarounds
- Identify future improvement opportunities
- Create low-priority tech debt items

***

## Implementation Timeline

| Phase | Duration | Start | End | Dependencies |
|-------|----------|-------|-----|--------------|
| 1. Script Recovery | 2-3 weeks | W1 | W3 | None |
| 2. Parser Evaluation | 1-2 weeks | W2 | W3 | Phase 1 (scripts) |
| 3. Pipeline Consolidation | 3-4 weeks | W3 | W7 | Phase 1, 2 |
| 4. Change Monitoring | 2-3 weeks | W6 | W8 | Phase 3 |
| 5. CI/CD Integration | 2-3 weeks | W7 | W9 | Phase 3, 4 |
| 6. Documentation & Handoff | 1 week | W9 | W10 | All previous |

**Total Duration:** ~10 weeks (if done sequentially)
**With Parallelization:** ~7-8 weeks

***

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                  COURT GUIDES PROCESSING PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐   ┌────────────────────┐   ┌──────────────────┐ │
│  │ UK Judiciary     │   │  Monitor Script    │   │   GitHub Action  │ │
│  │ Website (justice │   │  (weekly check)    │   │   (weekly cron)  │ │
│  │  .gov.uk)        │   │                    │   │                  │ │
│  └────────┬─────────┘   └────────┬───────────┘   └──────────┬───────┘ │
│           │                      │                           │          │
│           │ Download PDFs        │ Detect changes            │ Trigger  │
│           │ if changed           │                           │ on change│
│           └──────────────────────┴───────────────┬───────────┘          │
│                                                  │                      │
│                                                  ▼                      │
│                        ┌─────────────────────────────────────┐         │
│                        │  UNIFIED PIPELINE SCRIPT            │         │
│                        │ (process_court_guides_pipeline.py)  │         │
│                        └────────────────┬────────────────────┘         │
│                                         │                              │
│                   ┌─────────────────────┼─────────────────────┐        │
│                   │                     │                     │        │
│                   ▼                     ▼                     ▼        │
│         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│         │  Extract Stage   │  │ Process Stage    │  │ Validate     │ │
│         │  (Docling)       │  │ (JSON grouping,  │  │ Stage        │ │
│         │                  │  │  hierarchy)      │  │ (schema,     │ │
│         │ Extract PDFs →   │  │                  │  │  content)    │ │
│         │ Raw JSON         │  │ Process → JSON   │  │              │ │
│         └────────┬─────────┘  │ + review files   │  └────────┬─────┘ │
│                  │             └────────┬────────┘            │        │
│                  │                      │                     │        │
│                  └──────────────────────┴─────────────────────┘        │
│                                         │                              │
│                                         ▼                              │
│                        ┌──────────────────────────────┐               │
│                        │  Embed Stage (Azure OpenAI)  │               │
│                        │  → 3072-dim embeddings       │               │
│                        └────────────┬─────────────────┘               │
│                                     │                                 │
│                                     ▼                                 │
│                        ┌──────────────────────────────┐               │
│                        │  Upload Stage (Azure Search) │               │
│                        │  [--dry-run|--staging|--prod]│               │
│                        └────────────┬─────────────────┘               │
│                                     │                                 │
│                                     ▼                                 │
│                        ┌──────────────────────────────┐               │
│                        │  Azure AI Search Index       │               │
│                        │  (legal-court-rag-index)    │               │
│                        │  ✓ 200+ documents indexed    │               │
│                        │  ✓ Full-text + semantic      │               │
│                        │  ✓ Category filtering        │               │
│                        └──────────────────────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

***

## Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Missing scripts unrecoverable | Medium | High | Search archived repos, check cloud storage backups |
| Parser output differs from current | Low | Medium | Run side-by-side tests before deployment |
| Guide updates break pipeline | Low | High | Add comprehensive error handling + alerting |
| Schema changes incompatible with Azure Search | Low | Medium | Test with staging index first |
| Performance issues with automation | Low | Medium | Profile with production-size data |

***

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| All scripts recovered | 100% | All 8+ referenced scripts available and tested |
| Pipeline automation | Single command | `python process_court_guides_pipeline.py --production` |
| Change detection latency | <24 hours | Timestamp of detection vs actual change date |
| CI/CD integration | Pass rate 100% | All tests pass on each run |
| Content retention | ≥95% | Compare character/word counts before/after |
| Documentation completeness | 100% | All phases documented with runbooks |

***

## Future Improvements (Post-Implementation)

1. **ML-based change detection** — Train model to predict document importance changes
1. **Webhook integration** — If Justice.gov.uk ever provides document change webhooks
1. **Multi-language support** — Expand to Welsh Court Guide translations
1. **Real-time indexing** — Stream updates to Azure Search instead of batch uploads
1. **Version control for JSONs** — Git-track processed outputs for audit trail

***

## Appendix: Current Folder Structure

```text
court_guides_processing_pipeline/
├── docs/
│   ├── README.md                 # Original documentation
│   ├── SCRIPTS_USED.md          # Script reference
│   ├── CHECKLIST.md             # Validation checklist
│   ├── mapping.json             # Field mappings
│   └── IMPLEMENTATION_PLAN.md   # This document
├── sources/
│   ├── 14.341_JO_Commercial_Court_Guide_FINAL.pdf
│   ├── 35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf
│   ├── Chancery-Guide-2024-web.pdf
│   ├── Patents-Court-Guide-Updated-February-2025.pdf
│   └── The-Technology-and-Construction-Court-Guide.pdf
├── outputs/
│   ├── 14.341_JO_Commercial_Court_Guide_FINAL_processed.json
│   ├── 35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json
│   ├── Chancery-Guide-2024-web_processed.json
│   ├── Patents-Court-Guide-Updated-February-2025_processed.json
│   └── The-Technology-and-Construction-Court-Guide_processed.json
├── scripts/                     # To be populated with recovered scripts
│   ├── extract_court_guides.py
│   ├── process_court_guides.py
│   ├── validate_output.py
│   └── ...
└── validation/                  # Test results and validation reports
    └── (To be populated as phase progresses)
```

***

## Questions for Stakeholders

1. **Priority:** Is maintaining court guides important enough to automate, or is annual manual review acceptable?
1. **Budget:** Are we open to Azure Document Intelligence costs (~$2-6/guide/year) for better quality?
1. **Ownership:** Who will maintain this after implementation?
1. **SLA:** What's the acceptable delay for court guide updates (24h, 1 week, monthly)?

***

**Document Owner:** [To be assigned]
**Last Reviewed:** January 4, 2026
**Next Review:** [When phase 1 completes]
