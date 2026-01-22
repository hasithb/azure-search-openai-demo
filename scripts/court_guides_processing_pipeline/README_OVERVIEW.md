# Court Guides Processing Pipeline — Overview

**Folder Status:** Organized and documented (Planning phase)
**Last Updated:** January 4, 2026

***

## What Is This Folder?

This folder contains the **complete documentation, source files, and outputs** for the UK court guides processing pipeline — a multi-stage workflow that converts court guide PDFs to structured JSON data optimized for Azure AI Search semantic search.

### At a Glance

| Aspect | Details |
|--------|---------|
| **Purpose** | Convert UK Judiciary court guide PDFs → JSON → Azure Search index |
| **Current Status** | ✅ Outputs complete; ⚠️ Scripts partially recovered |
| **Update Frequency** | ~Annually (court guides rarely change) |
| **Integration** | Feeds into `legal-court-rag-index` in Azure AI Search |
| **Maintenance** | Low effort; can be fully automated |

***

## Folder Structure

```text
court_guides_processing_pipeline/
│
├── docs/                          [📚 Documentation]
│   ├── IMPLEMENTATION_PLAN.md     ← Full strategic plan for automation
│   ├── README.md                  ← Original process documentation
│   ├── SCRIPTS_USED.md            ← Reference for pipeline scripts
│   ├── CHECKLIST.md               ← Validation checklist
│   └── mapping.json               ← Azure Search field mappings
│
├── sources/                        [📄 Input PDFs]
│   ├── 14.341_JO_Commercial_Court_Guide_FINAL.pdf
│   ├── 35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf
│   ├── Chancery-Guide-2024-web.pdf
│   ├── Patents-Court-Guide-Updated-February-2025.pdf
│   └── The-Technology-and-Construction-Court-Guide.pdf
│
├── outputs/                        [✅ Output JSONs (Azure Search ready)]
│   ├── 14.341_JO_Commercial_Court_Guide_FINAL_processed.json
│   ├── 35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json
│   ├── Chancery-Guide-2024-web_processed.json
│   ├── Patents-Court-Guide-Updated-February-2025_processed.json
│   └── The-Technology-and-Construction-Court-Guide_processed.json
│
├── scripts/                        [🔧 Pipeline scripts (to be populated)]
│   └── [Recovery in progress - see IMPLEMENTATION_PLAN.md Phase 1]
│
└── validation/                     [🧪 Test results (as we progress)]
    └── [To be populated during implementation]
```

***

## The Pipeline at a Glance

```text
PDF (sources/)
    ↓
[Docling Extraction] - Extract structured content with OCR
    ↓
Raw JSON (temporary)
    ↓
[JSON Processing] - Group headings, merge content, preserve hierarchy
    ↓
Processed JSON (outputs/)
    ↓
[Azure OpenAI Embeddings] - Generate 3072-dimensional vectors
    ↓
[Azure Search Upload] - Index for semantic search
    ↓
Azure AI Search Index (legal-court-rag-index)
```

### Current Court Guides in Index

| Guide | Category | Processed Date | Size |
|-------|----------|---|---|
| Commercial Court Guide | Commercial | Jan 2025 | 12.4 MB |
| Kings Bench Division Guide | Kings Bench | Jan 2025 | 3.8 MB |
| Chancery Guide | Chancery | Jan 2025 | 24.5 MB |
| Patents Court Guide | Patents | Feb 2025 | 2.5 MB |
| Technology & Construction | TCC | Jan 2025 | 5.9 MB |

**Total:** 5 guides, 698 indexed sections, 100% searchable

***

## Key Documents to Read

1. **[IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** (Start here!)
   - Complete roadmap for automation
   - 6 phases with timelines and dependencies
   - Risk assessment and metrics

1. **[README.md](docs/README.md)**
   - Original process documentation
   - What worked well ✅ and known issues ⚠️

1. **[SCRIPTS_USED.md](docs/SCRIPTS_USED.md)**
   - Reference for each pipeline script
   - Troubleshooting guidance

1. **[CHECKLIST.md](docs/CHECKLIST.md)**
   - Step-by-step validation checklist
   - Run after any updates

1. **[mapping.json](docs/mapping.json)**
   - Field mappings: PDF → Processed JSON → Azure Search schema

***

## Current Status: Not Yet Automated ⚠️

### What Works ✅

- Docling extraction pipeline (proven, produces good results)
- JSON processing and transformation
- Azure OpenAI embedding generation
- Azure Search indexing
- 5 court guides fully processed and indexed

### What's Missing ⚠️

- **No version control for scripts** — extraction/processing scripts exist but not in repo
- **No automated change detection** — manual check needed for guide updates
- **No scheduled pipeline** — currently manual, one-off process
- **No CI/CD integration** — not part of deployment workflow

***

## When You Need to Use This

### Scenario 1: Court Guide Updated

1. Check the [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) Phase 4 for monitoring approach
1. Follow [CHECKLIST.md](docs/CHECKLIST.md) to validate changes
1. Re-run pipeline (instructions pending Phase 3 completion)

### Scenario 2: New Court Guide Added

1. Add PDF to `sources/`
1. Update [docs/mapping.json](docs/mapping.json) with guide metadata
1. Re-run unified pipeline (pending development)

### Scenario 3: Troubleshooting Failed Update

1. Check [docs/SCRIPTS_USED.md](docs/SCRIPTS_USED.md) for debugging steps
1. Review logs in `validation/` directory
1. Contact maintainer (see IMPLEMENTATION_PLAN.md for escalation)

***

## Integration with Main Application

The processed outputs feed directly into:

- **Backend:** `app/backend/approaches/` uses court guide content in RAG prompts
- **Frontend:** Users see court guide citations in answer sources
- **Search Index:** `legal-court-rag-index` in Azure AI Search

### Field Mappings

| Output Field | Azure Search Field | Frontend Use |
|--------------|-------------------|--------------|
| `id` | `id` (key) | Unique document reference |
| `content` | `content` | Full-text search + semantic search |
| `sourcepage` | `sourcepage` | Citation in answer sources |
| `sourcefile` | `sourcefile` | Document type filter |
| `category` | `category` | Category-based filtering |
| `embedding` | `embedding` | Semantic similarity search |

***

## Maintenance Philosophy

**Low-touch by design:**

- Court guides update ~1 time per year
- Docling extraction is proven and stable
- Once automated (Phase 3-5), can be fully hands-off
- Monitoring (Phase 4) detects changes automatically

**When changes happen:**

1. Automated detection alerts maintainer
1. Manual review of diff to approve
1. Re-run pipeline (1-2 hours)
1. Validate output in staging index
1. Promote to production

***

## Next Steps

### For Immediate Action (This Week)

✅ **Done:** Organized folder structure
✅ **Done:** Documented implementation plan
⏭️ **Next:** Review [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) with stakeholders

### For Implementation (Pending Decision)

See [IMPLEMENTATION_PLAN.md - Implementation Timeline](docs/IMPLEMENTATION_PLAN.md#implementation-timeline)

***

## Questions & Resources

| Question | Answer Location |
|----------|-----------------|
| How does the pipeline work? | [docs/README.md](docs/README.md) |
| What are the scripts? | [docs/SCRIPTS_USED.md](docs/SCRIPTS_USED.md) |
| How to validate changes? | [docs/CHECKLIST.md](docs/CHECKLIST.md) |
| What's the full plan? | [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) |
| What fields are in the JSON? | [docs/mapping.json](docs/mapping.json) |

***

## Related Documentation in Main Repo

- 📊 **Data Ingestion:** `docs/data_ingestion.md` (general document processing)
- 🔍 **Search Quality:** `evals/ground_truth_cpr.jsonl` (legal RAG evaluation)
- 📈 **Monitoring:** `docs/monitoring.md` (Application Insights setup)
- 🏗️ **Architecture:** `docs/architecture.md` (system overview)

***

**Folder Owner:** [To be assigned]
**Last Reviewed:** January 4, 2026
**Next Review:** When Phase 1 of implementation plan begins
