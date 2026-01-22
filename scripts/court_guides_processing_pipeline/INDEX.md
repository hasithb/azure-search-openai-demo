# 📋 Court Guides Processing Pipeline — Documentation Index

**Folder Name:** `court_guides_processing_pipeline/`
**Status:** ✅ Organized (January 4, 2026)
**Ready For:** Stakeholder review and Phase 1 implementation

***

## 📖 Start Here Based on Your Role

### 👔 Project Manager / Decision Maker

**Start with:** `QUICK_REFERENCE.md`
→ 5-minute summary of what, why, and next steps
→ Then: `docs/IMPLEMENTATION_PLAN.md` (Executive Summary section)

### 👨‍💻 Engineer (Implementing Phase 1-3)

**Start with:** `docs/IMPLEMENTATION_PLAN.md`
→ Phase 1: Script Recovery (what's missing, how to recover)
→ Phase 2-3: Pipeline consolidation approach
→ Reference: `docs/SCRIPTS_USED.md` for individual script details

### 🚀 DevOps / SRE (Implementing Phase 4-5)

**Start with:** `docs/IMPLEMENTATION_PLAN.md`
→ Phase 4: Change Monitoring (automated detection)
→ Phase 5: CI/CD Integration (testing & automation)
→ Reference: `COMPLETION_SUMMARY.md` for current status

### 🔍 QA / Validator

**Start with:** `docs/CHECKLIST.md`
→ Step-by-step validation process
→ Then: `README_OVERVIEW.md` (Testing & Quality section)

***

## 📚 Documentation Files

### At Root Level (Quick Access)

| File | Purpose | Read Time | Best For |
|------|---------|-----------|----------|
| **README_OVERVIEW.md** | Comprehensive guide to the entire folder | 10 mins | Getting oriented, understanding architecture |
| **QUICK_REFERENCE.md** | Quick 5-minute summary | 5 mins | Executives, quick lookup |
| **COMPLETION_SUMMARY.md** | What was done on Jan 4, 2026 | 5 mins | Understanding recent changes |
| **INDEX.md** | This file | 3 mins | Navigation help |

### In `docs/` Folder (Detailed Reference)

| File | Purpose | Size | Best For |
|------|---------|------|----------|
| **IMPLEMENTATION_PLAN.md** | 6-phase strategic plan with timelines & risk assessment | 19 KB | Implementation planning, architecture |
| **README.md** | Original process documentation & what worked well | 3.7 KB | Understanding current approach, known issues |
| **SCRIPTS_USED.md** | Reference guide for each pipeline script | 1.9 KB | Troubleshooting, understanding individual steps |
| **CHECKLIST.md** | Step-by-step validation checklist | 1.8 KB | Validating changes, ensuring quality |
| **mapping.json** | Azure Search field mappings | 3.8 KB | Understanding JSON structure, schema details |

***

## 🎯 How to Use This Documentation

### Scenario 1: "I Need to Understand the Current State"

1. Read: `README_OVERVIEW.md` (10 mins)
1. Skim: `docs/README.md` (5 mins)
1. Understand: `docs/mapping.json` (structure)
1. Check: `COMPLETION_SUMMARY.md` (what changed recently)

### Scenario 2: "I Need to Implement the Plan"

1. Read: `docs/IMPLEMENTATION_PLAN.md` → Full Strategic Plan (20 mins)
1. Focus on: Your assigned phase (Phases 1-6)
1. Reference: `docs/SCRIPTS_USED.md` for technical details
1. Validate: `docs/CHECKLIST.md` for quality gates

### Scenario 3: "Court Guides Need to Be Updated"

1. Check: `docs/CHECKLIST.md` (validation steps)
1. Run: Scripts in `validation/` folder (after Phase 1)
1. Reference: `docs/SCRIPTS_USED.md` (troubleshooting)
1. Upload: Updated outputs to Azure Search

### Scenario 4: "Something Broke - Help!"

1. Check: `docs/SCRIPTS_USED.md` → "If a script fails" section
1. Review: `validation/` folder for error logs (after Phase 1)
1. Consult: `docs/IMPLEMENTATION_PLAN.md` → Risk Assessment section
1. Escalate: Contact folder owner

***

## 🗂️ Folder Structure Overview

```text
court_guides_processing_pipeline/
│
├── 📖 ROOT DOCUMENTATION (Quick Access)
│   ├── README_OVERVIEW.md          ← Full guide to this folder
│   ├── QUICK_REFERENCE.md          ← 5-minute summary
│   ├── COMPLETION_SUMMARY.md       ← What was done on Jan 4
│   └── INDEX.md                    ← You are here!
│
├── 📚 docs/ (Detailed Reference)
│   ├── IMPLEMENTATION_PLAN.md      ← 6-phase roadmap (⭐ KEY DOCUMENT)
│   ├── README.md                   ← Original process documentation
│   ├── SCRIPTS_USED.md             ← Script reference guide
│   ├── CHECKLIST.md                ← Validation checklist
│   └── mapping.json                ← Field mappings
│
├── 📄 sources/ (Input PDFs - 5 guides)
│   ├── 14.341_JO_Commercial_Court_Guide_FINAL.pdf
│   ├── 35.16_JO_Kings_Bench_Division_Guide_2025_WEB4.pdf
│   ├── Chancery-Guide-2024-web.pdf
│   ├── Patents-Court-Guide-Updated-February-2025.pdf
│   └── The-Technology-and-Construction-Court-Guide.pdf
│
├── ✅ outputs/ (Output JSONs - Azure Search ready)
│   ├── 14.341_JO_Commercial_Court_Guide_FINAL_processed.json
│   ├── 35.16_JO_Kings_Bench_Division_Guide_2025_WEB4_processed.json
│   ├── Chancery-Guide-2024-web_processed.json
│   ├── Patents-Court-Guide-Updated-February-2025_processed.json
│   └── The-Technology-and-Construction-Court-Guide_processed.json
│
├── 🔧 scripts/ (To be populated in Phase 1)
│   └── [Missing extraction/processing scripts - see IMPLEMENTATION_PLAN.md]
│
└── 🧪 validation/ (To be populated as implementation progresses)
    └── [Test results, validation reports, error logs]
```

***

## ⏭️ Next Steps

### This Week

1. **You (Decision Maker):** Review `QUICK_REFERENCE.md` (5 mins)
1. **Team Lead:** Review `docs/IMPLEMENTATION_PLAN.md` (20 mins)
1. **Meeting:** Discuss 5-6 key decisions (see IMPLEMENTATION_PLAN.md "Questions for Stakeholders")

### Next Week

1. **Assign owners** for each phase
1. **Choose parser strategy** (Phase 2)
1. **Approve budget** (if using Azure DI)
1. **Schedule Phase 1 kickoff**

### Implementation (Timeline)

- See `docs/IMPLEMENTATION_PLAN.md` → Implementation Timeline section
- Expected: 7-10 weeks total (depending on parallelization)

***

## 🔗 Related Documentation in Main Repo

These documents reference or are related to the court guides pipeline:

| Main Repo Document | Relevance |
|-------------------|-----------|
| `docs/data_ingestion.md` | Mentions legal domain processing; links here |
| `docs/monitoring.md` | Covers data update monitoring; references our Phase 4 |
| `AGENTS.md` | Mentions court guides in context of legal RAG |
| `docs/legal_evaluation.md` | Uses court guides data for RAG evaluation |
| `evals/` | Ground truth and evaluation for legal RAG |

***

## 📞 Questions & Support

| Question | Answer |
|----------|--------|
| Where do I start? | See "Start Here Based on Your Role" section above |
| How long will this take? | 7-10 weeks; see IMPLEMENTATION_PLAN.md timeline |
| What are the key decisions? | See IMPLEMENTATION_PLAN.md → "Decisions for Stakeholders" |
| Can we parallelize phases? | Yes; see IMPLEMENTATION_PLAN.md → "Implementation Timeline" |
| What if guides change during implementation? | See docs/CHECKLIST.md for validation process |
| Where are the scripts? | Being recovered in Phase 1; see IMPLEMENTATION_PLAN.md Phase 1 |

***

## 📊 Document Statistics

| Metric | Value |
|--------|-------|
| Total documents | 13 files |
| Total documentation | 40 KB |
| Total data (PDFs + JSONs) | 56 MB |
| Court guides indexed | 5 |
| Sections in index | 698 |
| Implementation phases | 6 |
| Estimated duration | 7-10 weeks |

***

## ✅ Document Checklist (What's Complete)

- ✅ Folder renamed (clear, descriptive name)
- ✅ Files organized (sources, outputs, docs, scripts, validation)
- ✅ Implementation plan documented (6 phases, 19 KB)
- ✅ Overview created (README_OVERVIEW.md)
- ✅ Quick reference created (QUICK_REFERENCE.md)
- ✅ Completion summary created (COMPLETION_SUMMARY.md)
- ✅ Navigation index created (This file)
- ✅ Main repo docs updated (data_ingestion.md, monitoring.md)
- ✅ Original docs preserved (README.md, CHECKLIST.md, etc.)

***

## 📝 Version & Maintenance

| Field | Value |
|-------|-------|
| Created | January 4, 2026 |
| Last Updated | January 4, 2026 |
| Status | ✅ Ready for stakeholder review |
| Owner | [To be assigned] |
| Next Review | When Phase 1 begins |

***

**End of Index.** [Back to README_OVERVIEW.md](README_OVERVIEW.md) or [To Main IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
