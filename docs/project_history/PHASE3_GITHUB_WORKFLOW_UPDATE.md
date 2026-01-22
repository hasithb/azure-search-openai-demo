# Phase 3: GitHub Workflow Update Guide

## Status: Ready to Execute ✅

All Phase 2 tests passed successfully. The index `legal-court-rag-index-v2` is validated and ready for production use.

## Prerequisites Completed

- ✅ 1,287 documents in `legal-court-rag-index-v2`
- ✅ No duplicates found
- ✅ ID field is filterable
- ✅ All documents have correct structure
- ✅ Search quality validated
- ✅ New documents confirmed present
- ✅ List content handling verified

## Step-by-Step Instructions

### Step 1: Update GitHub Secret

The workflow uses `secrets.AZURE_SEARCH_INDEX` (not `vars.AZURE_SEARCH_INDEX`), so you need to update the **Repository Secret**.

**Instructions:**

1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions
2. Find the secret named `AZURE_SEARCH_INDEX`
3. Click **Update** (or **New repository secret** if it doesn't exist)
4. Set value to: `legal-court-rag-index-v2`
5. Click **Update secret**

### Step 2: Test with Dry Run

Run the workflow in dry-run mode to validate everything works:

1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/legal-scraper.yml
2. Click **Run workflow** dropdown
3. Set parameters:
   - `dry_run`: ✅ **true** (checked)
   - `force_upload`: ❌ **false** (unchecked)
4. Click **Run workflow**
5. Monitor the run and verify:
   - ✅ Scraper completes successfully
   - ✅ Validation passes
   - ✅ Upload plan shows differential logic working (only changed docs)
   - ✅ No actual upload occurs (dry run)

### Step 3: Production Run

Once dry run succeeds:

1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/legal-scraper.yml
2. Click **Run workflow** dropdown
3. Set parameters:
   - `dry_run`: ❌ **false** (unchecked)
   - `force_upload`: ❌ **false** (unchecked)
4. Click **Run workflow**
5. Monitor the run and verify:
   - ✅ Documents uploaded to `legal-court-rag-index-v2`
   - ✅ Only changed documents uploaded (differential logic)
   - ✅ No errors in logs

### Step 4: Verify Workflow Output

After successful run, check:

1. **GitHub Actions Summary** - Should show:
   - Number of documents scraped
   - Number of documents requiring upload
   - Upload success/failure

2. **Azure Search Index** - Verify document count matches expectations

## Current Workflow Configuration

The workflow uses these environment variables:

```yaml
env:
  AZURE_SEARCH_SERVICE: ${{ secrets.AZURE_SEARCH_SERVICE }}
  AZURE_SEARCH_INDEX: ${{ secrets.AZURE_SEARCH_INDEX }}  # ← UPDATE THIS SECRET
  AZURE_SEARCH_KEY: ${{ secrets.AZURE_SEARCH_KEY }}
  AZURE_OPENAI_SERVICE: ${{ secrets.AZURE_OPENAI_SERVICE }}
  AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}
  AZURE_OPENAI_EMB_DEPLOYMENT: ${{ secrets.AZURE_OPENAI_EMB_DEPLOYMENT }}
```

**Only `AZURE_SEARCH_INDEX` needs to be updated.**

## Rollback Plan

If issues occur:

1. **Immediate**: Update `AZURE_SEARCH_INDEX` secret back to `legal-court-rag-index`
2. **Re-run workflow** with dry run to verify rollback
3. **Investigate** issue in v2 index
4. **Fix** and re-test locally
5. **Retry** migration when ready

## Notes

- Old index `legal-court-rag-index` remains **ACTIVE** in deployed app
- Do NOT delete old index until Phase 5 (7-14 days after full migration)
- Workflow will now populate v2 index with weekly updates
- Deployed app still uses old index (until Phase 4)

## Next Phase: Phase 4 - Production App Migration

After workflow runs successfully for 1-2 cycles:

1. Update deployed app environment variable: `AZURE_SEARCH_INDEX=legal-court-rag-index-v2`
2. Monitor app performance
3. Keep old index for 7-14 days as safety net
4. Phase 5: Delete old index after confidence period

---

**See also:** [docs/INDEX_MIGRATION_CHECKLIST.md](docs/INDEX_MIGRATION_CHECKLIST.md)
