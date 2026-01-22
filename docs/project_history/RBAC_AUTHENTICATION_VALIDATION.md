# RBAC Authentication Validation Report

**Date:** January 17, 2026  
**Issue:** GitHub Actions workflow failing with "Forbidden" (403) errors  
**Root Cause:** API Keys passed to workflow despite `disableLocalAuth=true` on Azure resources  
**Resolution:** Remove API Keys from workflow to force RBAC authentication  

---

## ✅ Local CLI Validation Results

### 1. Azure Search Authentication (RBAC)
**Status:** ✅ **PASSED**

```bash
# Test command
python scripts/legal-scraper/upload_with_embeddings.py --input Upload --dry-run

# Results
✅ Using DefaultAzureCredential for authentication
✅ Index 'legal-court-rag-index-v2' found
✅ Response status: 200 (all differential check queries)
✅ Successfully queried existing documents for diff analysis
```

**Evidence:**
- No API keys in environment
- Successfully authenticated using Service Principal (Client ID: `1d382868-51d6-4200-a4ba-3a7b94ecb2d3`)
- Queried index and retrieved 835 existing documents
- Generated differential statistics showing 498 changed documents (59.6%)

### 2. Azure OpenAI Authentication (RBAC)
**Status:** ✅ **PASSED**

```bash
# Test: Generate embedding using RBAC
python -c "from openai import AzureOpenAI; ..."

# Results
✅ Generated 3072 dimension embedding
✅ Model: text-embedding-3-large
✅ Usage: 10 tokens
```

**Evidence:**
- No API keys in environment
- Successfully generated embeddings using `DefaultAzureCredential`
- Token provider using `https://cognitiveservices.azure.com/.default` scope

### 3. Differential Upload Statistics
**Status:** ✅ **PASSED** (Option A Implementation)

```text
Document Counts:
  Total Input:           835
  New:                     0  (  0.0%)
  Changed:               498  ( 59.6%)
  Unchanged:             337  ( 40.4%)
  Total to Upload:       498

Status: UPLOAD REQUIRED
```

**File:** `data/legal-scraper/processed/upload_statistics.txt`

---

## 🔧 Fix Implemented

### Changes to `.github/workflows/legal-scraper.yml`

**Removed API Key Environment Variables:**

```diff
- AZURE_SEARCH_KEY: ${{ secrets.AZURE_SEARCH_KEY }}
- AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}
```

**Added Identity Variables (for upload job):**

```diff
+ AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
+ AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
```

**Why This Works:**
1. `upload_with_embeddings.py` checks for API keys first: `if key: ...`
2. If no keys are present, it falls back to `DefaultAzureCredential()`
3. `DefaultAzureCredential` uses the Service Principal authentication from GitHub Actions OIDC
4. The Service Principal has the required RBAC roles:
   - `Search Index Data Contributor` (Azure Search)
   - `Cognitive Services OpenAI User` (Azure OpenAI)
   - `Search Service Contributor` (Resource Group scope)

---

## 🧪 Test Script

A reusable test script has been created: **`test_rbac_auth.sh`**

```bash
# Run the test
./test_rbac_auth.sh

# What it tests:
✅ Azure Search authentication (index access)
✅ Azure OpenAI authentication (embedding generation)
✅ Differential update statistics generation
```

---

## 📊 GitHub Workflow Run Results

### Run #21080521998 (January 16, 2026)
**Status:** ✅ **SUCCESS**

| Job | Status | Details |
|-----|--------|---------|
| scrape-and-validate | ✅ Success | Scraper completed, validation passed |
| upload-production | ✅ Success | RBAC authentication worked, no errors |

**Key Log Evidence:**
```
2026-01-16 22:04:21 - INFO - Using DefaultAzureCredential for authentication
2026-01-16 22:04:22 - INFO - Response status: 200
2026-01-16 22:06:26 - INFO - Using DefaultAzureCredential for authentication
2026-01-16 22:06:28 - INFO - Response status: 200
```

**No "Forbidden" errors** - all authentication succeeded using RBAC.

### Artifact Generated
- **upload-statistics** artifact contains the new Option A statistics
- **Retention:** 90 days
- **Format:** Detailed breakdown with percentages

---

## 🔐 Azure RBAC Configuration

### Service Principal Details
- **App ID:** `1d382868-51d6-4200-a4ba-3a7b94ecb2d3`
- **Object ID:** `26aa9f4b-068b-40b3-8794-846e52f266ac`
- **Display Name:** Azure Search OpenAI Chat Client App 72638

### Role Assignments
| Role | Scope | Purpose |
|------|-------|---------|
| Search Index Data Contributor | `/providers/Microsoft.Search/searchServices/cpr-rag` | Read/write index documents |
| Cognitive Services OpenAI User | `/providers/Microsoft.CognitiveServices/accounts/cog-gz2m4s637t5me` | Generate embeddings |
| Search Service Contributor | `/subscriptions/.../resourceGroups/rg-cpr-rag` | Manage search service |

### Azure Resource Configuration
| Resource | Setting | Value |
|----------|---------|-------|
| cpr-rag (Search) | `disableLocalAuth` | `true` |
| cpr-rag (Search) | `publicNetworkAccess` | `Enabled` |
| cog-gz2m4s637t5me (OpenAI) | `disableLocalAuth` | `true` |

---

## 🎯 Validation Checklist

- [x] Local CLI test passes without API keys
- [x] Azure Search authentication works with RBAC
- [x] Azure OpenAI authentication works with RBAC
- [x] Differential update logic queries existing documents
- [x] Statistics file generated with new format (Option A)
- [x] GitHub workflow completes without "Forbidden" errors
- [x] Service Principal has required RBAC roles
- [x] Workflow uses OIDC authentication (no secrets in logs)

---

## 🚀 Next Steps

### For Future Workflows
1. **Monitor Statistics Artifact:** Download `upload-statistics` from each run to track changes
2. **Review Deployment:** The upload job ran but reported 0 changes (needs investigation)
3. **Consider Alerts:** Add failure notifications if differential upload detects issues

### For Investigation
The workflow reported "INDEX UP TO DATE" (0 changes) despite local test showing 498 changes. Possible causes:
- Scraper generated different data in GitHub Actions vs local
- Index was updated between local test and workflow run
- Data source files differ between local and GitHub Actions environment

---

## 📝 Summary

**Problem:** GitHub Actions could not authenticate to Azure resources due to `disableLocalAuth=true` blocking API key access.

**Solution:** Removed API keys from workflow environment variables to force RBAC authentication via `DefaultAzureCredential`.

**Validation:** Local CLI tests and GitHub workflow runs confirm RBAC authentication works correctly for both Azure Search and Azure OpenAI.

**Statistics Feature:** "Option A" implementation successfully generates detailed upload statistics with percentages for New/Changed/Unchanged documents.
