# Azure Search Index Migration Checklist

## Overview
Migration from `legal-court-rag-index` → `legal-court-rag-index-v2`

**Why:** Add filterable `id` field for efficient differential uploads

**Status:** 🟡 In Progress

---

## Phase 1: Populate New Index ✅

- [x] Create new index with filterable id field
- [x] Migrate existing 1,127 documents from old index
- [ ] Upload 835 new/changed documents from latest scrape
- [ ] Verify total document count (~1,962 expected)
- [ ] Spot-check random documents for data integrity

**Verification Commands:**
```bash
# Check document count
python -c "
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from scripts.legal-scraper.config import Config

client = SearchClient(
    endpoint=f'https://{Config.AZURE_SEARCH_SERVICE}.search.windows.net',
    index_name='legal-court-rag-index-v2',
    credential=AzureKeyCredential(Config.AZURE_SEARCH_KEY)
)
results = client.search('*', select='id', include_total_count=True)
print(f'Total documents: {results.get_count()}')
"

# Test filterable id
python validate_migration.py
```

---

## Phase 2: Local Testing ⏳

- [ ] Update local `.env` or `local.env`:
  ```bash
  AZURE_SEARCH_INDEX=legal-court-rag-index-v2
  ```
- [ ] Start local app: `./app/start.sh` or task "Development"
- [ ] Test search queries across multiple categories
- [ ] Verify citations render correctly
- [ ] Test category filtering works
- [ ] Check answer quality with legal precedents
- [ ] Test new documents appear in search results

**Test Queries:**
- "What is Part 44 about?" (existing doc)
- "What are the rules for civil recovery proceedings?" (new doc)
- "Explain enforcement by taking control of goods" (test chunked docs)

---

## Phase 3: GitHub Workflow Setup ⏳

- [ ] Update GitHub Actions secret:
  - Go to: Settings → Secrets and variables → Actions
  - Update `AZURE_SEARCH_INDEX` = `legal-court-rag-index-v2`
  
- [ ] Test workflow with dry-run:
  - Go to: Actions → Legal Document Scraper Pipeline
  - Click "Run workflow"
  - Set `dry_run: true`
  - Verify diff report shows mostly "unchanged" documents

- [ ] Run actual workflow (after dry-run success):
  - Run workflow with `dry_run: false`
  - Monitor for successful completion
  - Check diff report for new/changed/unchanged counts

---

## Phase 4: Production App Migration ⏳

**Current State:**
- Old index: `legal-court-rag-index` (1,127 docs)
- New index: `legal-court-rag-index-v2` (1,962 docs)
- Deployed app uses: `legal-court-rag-index`

**Migration Strategy:** Blue/Green Deployment (Recommended)

### Option A: Zero-Downtime Update via Azure Portal

1. **Backup Current Config:**
   ```bash
   az webapp config appsettings list \
     --name <your-app-name> \
     --resource-group <your-rg> \
     > backup_appsettings.json
   ```

2. **Update App Setting:**
   ```bash
   az webapp config appsettings set \
     --name <your-app-name> \
     --resource-group <your-rg> \
     --settings AZURE_SEARCH_INDEX=legal-court-rag-index-v2
   ```

3. **Monitor Application Insights:**
   - Check for errors in first 30 minutes
   - Verify successful query logs
   - Monitor response times

4. **Rollback if needed:**
   ```bash
   az webapp config appsettings set \
     --name <your-app-name> \
     --resource-group <your-rg> \
     --settings AZURE_SEARCH_INDEX=legal-court-rag-index
   ```

### Option B: Deploy New Version with azd

If using `azd`:
```bash
# Update infra/main.bicep or use environment variable
azd env set AZURE_SEARCH_INDEX legal-court-rag-index-v2
azd deploy
```

**Post-Migration Checklist:**
- [ ] Verify app starts without errors
- [ ] Test 5-10 queries via web UI
- [ ] Check Application Insights for exceptions
- [ ] Verify category filtering still works
- [ ] Test both Ask and Chat pages
- [ ] Monitor for 24-48 hours

---

## Phase 5: Cleanup (After Stability Confirmed) ⏳

**Wait 7-14 days before deletion** to ensure no rollback needed.

- [ ] Confirm production app stable on new index (7+ days)
- [ ] Verify no errors in Application Insights
- [ ] Get approval from stakeholders
- [ ] Delete old index:
  ```bash
  az search index delete \
    --name legal-court-rag-index \
    --service-name cpr-rag \
    --resource-group <your-rg>
  ```
- [ ] Update documentation to reflect new index name

---

## Rollback Plan

If issues occur at any phase:

1. **During Local Testing:**
   - Revert `.env` to `AZURE_SEARCH_INDEX=legal-court-rag-index`
   - Investigate and fix issues before retrying

2. **During GitHub Workflow:**
   - Revert GitHub secret to `legal-court-rag-index`
   - Old index still receives updates

3. **During Production Migration:**
   - Use Azure Portal or CLI to revert app setting
   - Old index remains unchanged and functional
   - Investigate issues with new index offline

---

## Risk Mitigation

✅ **Low Risk:**
- New index is separate - old index untouched
- Can test thoroughly before production switch
- Easy rollback at any phase
- No data loss - both indexes coexist

⚠️ **Potential Issues:**
- Query behavior differences (unlikely - same schema)
- Performance differences (monitor with App Insights)
- GitHub workflow needs secret update

---

## Success Criteria

- [ ] New index has all documents (old + new)
- [ ] Local testing passes all test queries
- [ ] GitHub workflow correctly identifies unchanged docs
- [ ] Production app runs for 7+ days without errors
- [ ] Performance metrics match or exceed old index
- [ ] Stakeholder approval for old index deletion

---

## Notes

- **Do NOT delete old index until Phase 5 is complete**
- Keep both indexes for 7-14 days minimum
- Monitor Application Insights closely during Phase 4
- Document any issues encountered for future reference
