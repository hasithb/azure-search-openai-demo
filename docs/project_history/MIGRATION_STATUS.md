# Index Migration Complete - Next Steps

## ✅ Phase 2 Complete: Index Validated

All tests passed successfully on **legal-court-rag-index-v2**:

| Test | Status | Details |
|------|--------|---------|
| Duplicate Check | ✅ PASS | 1,287 unique documents |
| Filterable ID | ✅ PASS | ID field filter queries working |
| Document Structure | ✅ PASS | All required fields present |
| Search Quality | ✅ PASS | Relevant results for legal queries |
| Category Distribution | ✅ PASS | 7 categories, 427 CPR docs |
| New Documents | ✅ PASS | All 4 new docs found |
| List Content Handling | ✅ PASS | Properly processed as strings |

**Index is production-ready!** 🎉

## 📋 What Was Accomplished

### Migration Summary
- **Old Index:** `legal-court-rag-index` (1,127 docs) - Still active in production
- **New Index:** `legal-court-rag-index-v2` (1,287 docs) - Validated, ready for use
- **Changes:** +160 documents (835 uploaded total, some updated existing)
- **Key Improvement:** ID field now filterable for efficient differential uploads

### Performance Improvements
- **Upload Speed:** 60x faster (90 min → 1.5 min for 835 docs)
- **Batch Processing:** 100 docs/batch (up from 3)
- **Retry Logic:** Optimized exponential backoff (2-20s)
- **Rate Limits:** Configured for 12K req/min, 2M tokens/min

### Code Quality
- Fixed list content handling in 3 locations
- Improved endpoint URL formatting
- Added comprehensive test suite
- Created migration scripts and documentation

## 🚀 Ready for Phase 3: GitHub Workflow Update

### Quick Start

1. **Update GitHub Secret:**
   ```
   Repository Secret: AZURE_SEARCH_INDEX = legal-court-rag-index-v2
   ```

2. **Test with Dry Run:**
   - Go to Actions → Legal Document Scraper Pipeline
   - Run workflow with `dry_run: true`
   - Verify differential upload logic works

3. **Production Run:**
   - Run workflow with `dry_run: false`
   - Verify weekly scraping populates v2 index

**Detailed instructions:** See [PHASE3_GITHUB_WORKFLOW_UPDATE.md](PHASE3_GITHUB_WORKFLOW_UPDATE.md)

## 🧪 Optional: Local Testing

Test the new index with your local development environment:

```bash
# Switch to v2 index
./switch_to_v2_index.sh

# Start the app
cd app && ./start.sh

# Test in browser at http://localhost:50505
```

**Test scenarios:**
- Search for "Part 44 costs" → Should return CPR Part 44 documents
- Search for "civil recovery proceedings" → Should return practice directions
- Search for "insolvency practice direction" → Should return insolvency docs
- Verify citations render correctly
- Check category filtering works

**Switch back:**
```bash
azd env set AZURE_SEARCH_INDEX "legal-court-rag-index"
```

## 📅 Migration Timeline

### ✅ Phase 1: Index Creation (COMPLETE)
- Created `legal-court-rag-index-v2` with filterable ID
- Migrated 1,127 existing documents
- Status: **DONE**

### ✅ Phase 2: Validation (COMPLETE)
- Uploaded 835 new/changed documents
- Ran comprehensive test suite (7 tests)
- All tests passed
- Status: **DONE**

### 🔜 Phase 3: GitHub Workflow (READY TO START)
- Update repository secret
- Test with dry run
- Run production workflow
- Estimated time: 30 minutes

### 🔜 Phase 4: Production App Migration (PENDING)
- Update deployed app environment variable
- Monitor performance for 7-14 days
- Keep old index as safety net
- Estimated time: 1 hour setup + monitoring period

### 🔜 Phase 5: Cleanup (PENDING)
- Delete old index after confidence period
- Update documentation
- Archive migration scripts
- Estimated time: 30 minutes

## 📁 Key Files

### Test Results
- `phase2_test_results.log` - Comprehensive test output
- `phase2_comprehensive_test.py` - Test suite (reusable)

### Migration Scripts
- `scripts/migrate_index_filterable_id.py` - One-time migration (already run)
- `scripts/legal-scraper/upload_with_embeddings.py` - Optimized upload script
- `switch_to_v2_index.sh` - Local environment switcher

### Documentation
- `PHASE3_GITHUB_WORKFLOW_UPDATE.md` - Next steps guide
- `docs/INDEX_MIGRATION_CHECKLIST.md` - Full migration plan
- `docs/MAINTENANCE_GUIDE.md` - Ongoing operations

### Configuration
- `scripts/legal-scraper/config.py` - Updated to v2 index
- `.github/workflows/legal-scraper.yml` - Uses `secrets.AZURE_SEARCH_INDEX`

## ⚠️ Important Notes

1. **DO NOT DELETE old index** until Phase 5 (after 7-14 day safety period)
2. **Production app still uses old index** - update in Phase 4
3. **GitHub workflow needs secret update** - see Phase 3 guide
4. **Test locally first** (optional but recommended)
5. **Monitor workflow runs** for first few cycles

## 🆘 Rollback Plan

If issues occur at any phase:

1. **GitHub Workflow:** Update secret back to `legal-court-rag-index`
2. **Local Testing:** Run `azd env set AZURE_SEARCH_INDEX "legal-court-rag-index"`
3. **Production App:** No change needed (still on old index)
4. **Investigate and fix** issues in v2 index
5. **Retry** when ready

## 📊 Index Comparison

| Metric | Old Index | New Index | Change |
|--------|-----------|-----------|--------|
| Documents | 1,127 | 1,287 | +160 |
| ID Filterable | ❌ No | ✅ Yes | Required |
| Categories | 7 | 7 | Same |
| CPR Docs | ~400 | 427 | +27 |
| Upload Method | Full replace | Differential | Better |

## 🎯 Success Criteria

Before proceeding to Phase 4:

- ✅ All Phase 2 tests pass (DONE)
- ⬜ GitHub workflow dry run succeeds
- ⬜ GitHub workflow production run succeeds
- ⬜ At least 1-2 weekly runs complete successfully
- ⬜ Local testing validates search quality (optional)

## 📞 Support

If you encounter issues:

1. Check test results: `cat phase2_test_results.log`
2. Review workflow logs in GitHub Actions
3. Run local tests: `python phase2_comprehensive_test.py`
4. Check migration checklist: `docs/INDEX_MIGRATION_CHECKLIST.md`

---

**Ready to proceed with Phase 3!** See [PHASE3_GITHUB_WORKFLOW_UPDATE.md](PHASE3_GITHUB_WORKFLOW_UPDATE.md) for detailed instructions.
