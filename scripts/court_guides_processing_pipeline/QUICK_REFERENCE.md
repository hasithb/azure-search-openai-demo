# Court Guides Processing — Quick Reference

**Folder:** `court_guides_processing_pipeline/`
**Status:** Organized (Jan 2026) — Ready for implementation
**Key Doc:** `docs/IMPLEMENTATION_PLAN.md`

***

## The 5-Minute Summary

### What?

UK court guide PDFs are converted to JSON and indexed in Azure AI Search for legal RAG search.

### Current State

- ✅ 5 guides processed and indexed (698 sections total)
- ⚠️ No automated pipeline (manual process)
- ⚠️ Scripts not version-controlled

### Plan

6-phase implementation to fully automate:

1. Recover missing scripts (2-3 weeks)
1. Evaluate parsing strategy (1-2 weeks)
1. Build unified pipeline (3-4 weeks)
1. Add change monitoring (2-3 weeks)
1. CI/CD integration (2-3 weeks)
1. Documentation & handoff (1 week)

**Total:** ~10 weeks (7-8 weeks with parallelization)

### Why?

- Court guides update annually → automatable
- Manual process error-prone → standardize
- Monitoring gaps → miss important updates
- Scripts missing → can't reproduce

***

## Folder Organization

```text
court_guides_processing_pipeline/
├── docs/                          # Documentation
│   ├── IMPLEMENTATION_PLAN.md     # ← START HERE
│   ├── README.md                  # Original process docs
│   ├── SCRIPTS_USED.md            # Script reference
│   ├── CHECKLIST.md               # Validation steps
│   └── mapping.json               # Field mappings
├── sources/                        # 5 PDF files (input)
├── outputs/                        # 5 JSON files (output)
├── scripts/                        # (To be populated)
└── validation/                     # (For test results)
```

***

## Key Decisions Pending

| Decision | Options | Impact |
|----------|---------|--------|
| **Parser** | Keep Docling OR switch to Azure DI? | Cost vs. quality |
| **Automation Priority** | Phase 3 first OR Phase 4 first? | Speed to value vs. monitoring |
| **Ownership** | Who maintains after implementation? | Support model |
| **SLA** | Same-day updates OR weekly batch? | Operational load |

***

## Implementation Phases at a Glance

| Phase | Focus | Duration | Owner | Status |
|-------|-------|----------|-------|--------|
| 1 | Recover scripts | 2-3w | Engineer | 🚧 Pending |
| 2 | Evaluate parsers | 1-2w | Engineer | 🚧 Pending |
| 3 | Build pipeline | 3-4w | Engineer | 🚧 Pending |
| 4 | Add monitoring | 2-3w | DevOps | 🚧 Pending |
| 5 | CI/CD + tests | 2-3w | DevOps | 🚧 Pending |
| 6 | Handoff + docs | 1w | Team | 🚧 Pending |

***

## Success Metrics

✅ All scripts recovered and tested
✅ Single command to process all guides
✅ Automated detection of guide changes
✅ 95%+ content retention
✅ Full CI/CD integration
✅ Complete operator runbook

***

## Next Steps

1. **Review** `docs/IMPLEMENTATION_PLAN.md` (detailed read: 20 mins)
1. **Decide** on parser strategy (Phase 2)
1. **Assign** owners for each phase
1. **Schedule** Phase 1 kickoff
1. **Monitor** progress against timeline

***

## Contact Points

| Question | Document |
|----------|----------|
| "How does pipeline work?" | `docs/README.md` |
| "What's the full plan?" | `docs/IMPLEMENTATION_PLAN.md` |
| "What do I do if a guide updates?" | `docs/CHECKLIST.md` |
| "Why is something missing?" | `docs/SCRIPTS_USED.md` |

***

**Last Updated:** January 4, 2026
**Reviewed By:** [TBD]
**Next Review:** When Phase 1 begins
