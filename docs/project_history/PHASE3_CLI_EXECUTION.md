# Phase 3 CLI Execution Summary

## ✅ Actions Completed

### 1. GitHub Secret Updated
```bash
gh secret set AZURE_SEARCH_INDEX -R adalex-ai/azure-search-openai-demo
```
**Value:** `legal-court-rag-index-v2`  
**Status:** ✅ Complete

### 2. Dry Run Workflow Triggered
```bash
gh workflow run "Legal Document Scraper Pipeline" \
  -R adalex-ai/azure-search-openai-demo \
  -f dry_run=true \
  -f force_upload=false
```
**Status:** ⏳ Running (ID: 21046...)

---

## 📊 Monitor Dry Run

### Quick Status Check
```bash
gh run list -R adalex-ai/azure-search-openai-demo --workflow="legal-scraper.yml" --limit 1
```

### Watch Live (Real-time Updates)
```bash
gh run watch $(gh run list -R adalex-ai/azure-search-openai-demo --workflow=legal-scraper.yml --limit 1 --json databaseId -q '.[0].databaseId') -R adalex-ai/azure-search-openai-demo
```

### View in Browser
```bash
gh run view $(gh run list -R adalex-ai/azure-search-openai-demo --workflow=legal-scraper.yml --limit 1 --json databaseId -q '.[0].databaseId') -R adalex-ai/azure-search-openai-demo --web
```

### View Logs
```bash
gh run view $(gh run list -R adalex-ai/azure-search-openai-demo --workflow=legal-scraper.yml --limit 1 --json databaseId -q '.[0].databaseId') -R adalex-ai/azure-search-openai-demo --log
```

---

## 🚀 After Dry Run Succeeds

### Step 3: Trigger Production Run
```bash
gh workflow run "Legal Document Scraper Pipeline" \
  -R adalex-ai/azure-search-openai-demo \
  -f dry_run=false \
  -f force_upload=false
```

### Then Watch Production Run
```bash
# Check status
gh run list -R adalex-ai/azure-search-openai-demo --workflow="legal-scraper.yml" --limit 1

# Watch live
gh run watch $(gh run list -R adalex-ai/azure-search-openai-demo --workflow=legal-scraper.yml --limit 1 --json databaseId -q '.[0].databaseId') -R adalex-ai/azure-search-openai-demo
```

---

## 🛠️ Helper Tools

### Interactive Workflow Manager
```bash
./phase3_workflow_manager.sh
```
Provides menu-driven interface for:
- Watching current run
- Viewing logs
- Triggering production run
- Opening in browser
- Checking secrets

### Check All Secrets
```bash
gh secret list -R adalex-ai/azure-search-openai-demo | grep AZURE
```

---

## ✅ Success Criteria

Before proceeding to Phase 4:

- [ ] Dry run completes successfully (no errors)
- [ ] Dry run shows differential upload logic working
- [ ] Dry run validates documents in v2 index
- [ ] Production run uploads documents successfully
- [ ] No errors in workflow logs

---

## 🔄 Rollback (If Needed)

If issues occur, revert the secret:

```bash
echo "legal-court-rag-index" | gh secret set AZURE_SEARCH_INDEX -R adalex-ai/azure-search-openai-demo
```

Then re-run workflow with old index to verify rollback.

---

## 📋 Phase Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Index Creation | ✅ Complete | v2 index created with filterable ID |
| Phase 2: Validation | ✅ Complete | All 7 tests passed (1,287 docs) |
| Phase 3: GitHub Workflow | ⏳ In Progress | Secret updated, dry run running |
| Phase 4: Production App | 🔜 Pending | Update app after workflow validated |
| Phase 5: Cleanup | 🔜 Pending | Delete old index after 7-14 days |

---

## 📞 Next Steps

1. **Wait for dry run to complete** (~5-10 minutes)
2. **Verify dry run success** - Check logs for:
   - Scraper completed successfully
   - Validation passed
   - Upload plan generated (differential logic)
   - No actual upload (dry run mode)
3. **Trigger production run** (if dry run succeeds)
4. **Verify production run** - Check:
   - Documents uploaded to v2 index
   - Only changed documents uploaded
   - No errors
5. **Monitor 1-2 weekly runs** before Phase 4

---

**Repository:** https://github.com/adalex-ai/azure-search-openai-demo  
**Workflow:** https://github.com/adalex-ai/azure-search-openai-demo/actions/workflows/legal-scraper.yml
