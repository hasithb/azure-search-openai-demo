# Enhanced Feedback System - Integration Points Quick Reference

This document shows the exact lines where customizations integrate with upstream code.

## Backend Integration Points

### 1. App Configuration (app/backend/app.py)

**Already in place** - Just verify these exist:

```python
# Around line 20-30: Import statements
from customizations.routes import categories_bp, feedback_bp
```

```python
# Around line 850-860: Blueprint registration
app.register_blueprint(categories_bp)
app.register_blueprint(feedback_bp)
```

### 2. Chat Approach (app/backend/approaches/chatapproach.py)

### 3. Infrastructure (infra/main.bicep)

**Location** - In `appEnvVariables` object (around line 472-560):

Add these three lines:

```bicep
  // CUSTOM: Deployment metadata for feedback tracking
  DEPLOYMENT_ID: environmentName
  APP_VERSION: 'v1.0.0'
  GIT_SHA: deployment().properties.template.metadata.version
```

Example placement:

```bicep
var appEnvVariables = {
  AZURE_STORAGE_ACCOUNT: storage.outputs.name
  AZURE_STORAGE_CONTAINER: storageContainerName
  // ... other variables ...
  RUNNING_IN_PRODUCTION: 'true'
  // CUSTOM: Deployment metadata for feedback tracking
  DEPLOYMENT_ID: environmentName
  APP_VERSION: 'v1.0.0'
  GIT_SHA: deployment().properties.template.metadata.version
  // RAG Configuration
  RAG_SEARCH_TEXT_EMBEDDINGS: ragSearchTextEmbeddings
  // ... rest of variables ...
}
```

## Frontend Integration Points

### No Frontend Code Changes Required!

✅ The frontend LegalFeedback component already:

- Submits feedback to `/api/feedback` endpoint
- Includes thoughts in the payload
- Respects user consent for context sharing

✅ The backend filters thoughts before they reach the frontend

✅ The feedback route filters thoughts again before storing

This is a **defense-in-depth** approach:

1. Backend filters on response
1. User receives only user-safe thoughts
1. User submits feedback with safe thoughts
1. Backend filters again on storage (safety layer)

## Custom Files (No Upstream Changes Needed)

These files are entirely in `/customizations/` and require NO changes to upstream:

✅ `app/backend/customizations/thought_filter.py`
✅ `app/backend/customizations/config.py` (enhanced)
✅ `app/backend/customizations/__init__.py` (enhanced)
✅ `app/backend/customizations/routes/feedback.py` (enhanced)

## Testing the Integration

### Verify Thought Filtering Works

```bash
# Run backend tests
pytest tests/test_thought_filter.py -v

# Run feedback tests
pytest tests/test_feedback.py -v

# Run both with coverage
pytest tests/test_thought_filter.py tests/test_feedback.py --cov=customizations -v
```

### Test Manually

1. **Start local server:**

```bash
   cd app/backend
   python -m quart --app=app:app --debug run
   ```

1. **Send feedback with thoughts:**

```bash
   curl -X POST http://localhost:5000/api/feedback \
     -H "Content-Type: application/json" \
     -d '{
       "message_id": "test-msg-123",
       "rating": "unhelpful",
       "issues": ["wrong_citation"],
       "comment": "Test feedback",
       "context_shared": true,
       "user_prompt": "What is X?",
       "ai_response": "X is...",
       "thoughts": [
         {"title": "Search Query", "description": "Searching for X"},
         {"title": "Prompt to generate answer", "description": "Admin content"}
       ]
     }'
   ```

1. **Check feedback file was created:**

```bash
   ls -la feedback_data/local/
   # Should see: 2026-01-10t*_test-msg-123.json (user-visible)
   #            2026-01-10t*_test-msg-123_admin.json (admin-only)
   ```

1. **Verify filtering:**

```bash
   cat feedback_data/local/2026-01-10t*_test-msg-123.json | jq '.context.thoughts'
   # Should show: Only "Search Query" thought (Prompt to generate answer filtered out)

   cat feedback_data/local/2026-01-10t*_test-msg-123_admin.json | jq '.admin_only_thoughts'
   # Should show: "Prompt to generate answer" with full system instructions
   ```

## Rollback Instructions

If you need to remove the feedback enhancement:

1. **Remove from app.py:**

```bash
   # Remove or comment out:
   from customizations.routes import feedback_bp
   app.register_blueprint(feedback_bp)
   ```

1. **Remove from chatapproach.py:**

```bash
   # Remove these lines (they're optional):
   from customizations import filter_thoughts_for_user
   extra_info.thoughts = filter_thoughts_for_user(extra_info.thoughts)
   ```

1. **Remove from main.bicep:**

```bash
   # Remove these lines:
   DEPLOYMENT_ID: environmentName
   APP_VERSION: 'v1.0.0'
   GIT_SHA: deployment().properties.template.metadata.version
   ```

1. **Keep customizations folder intact:**

```bash
   # These are safe to keep, they won't cause issues:
   - app/backend/customizations/thought_filter.py
   - app/backend/customizations/config.py
   - tests/test_feedback.py
   - tests/test_thought_filter.py
   ```

## Troubleshooting

### Issue: Feedback endpoint not found (404)

**Solution:** Verify `app.py` has the feedback blueprint imported and registered:

```python
from customizations.routes import feedback_bp
app.register_blueprint(feedback_bp)
```

### Issue: System prompts still visible in API response

**Solution:** Verify filtering is in place in `chatapproach.py`:

```python
from customizations import filter_thoughts_for_user
extra_info.thoughts = filter_thoughts_for_user(extra_info.thoughts)
```

### Issue: Feedback files not created

**Solution:** Check that `/api/feedback` endpoint is accessible and the `feedback_data` directory has write permissions:

```bash
mkdir -p feedback_data/local
chmod 755 feedback_data
```

### Issue: Admin files not created

**Solution:** Admin files are only created if there are admin-only thoughts to save. If you're seeing all user-safe thoughts, this is expected behavior.

## Version Tracking

The system captures these version markers:

- **DEPLOYMENT_ID**: From environment name or env var - typically: `prod`, `staging`, `dev`
- **APP_VERSION**: Semantic version - set in Bicep as `v1.0.0`
- **GIT_SHA**: Git commit hash - automatically captured by Bicep
- **MODEL_NAME**: Auto-detected from AZURE_OPENAI_CHATGPT_MODEL
- **ENVIRONMENT**: Auto-detected from RUNNING_IN_PRODUCTION flag

Example feedback metadata:

```json
{
  "deployment_id": "prod-v1",
  "app_version": "1.0.0",
  "git_sha": "abc123def456789",
  "model_name": "gpt-4",
  "environment": "production"
}
```

This allows you to:

- Query feedback by version: `SELECT * WHERE git_sha = 'abc123...'`
- Track issues across model upgrades: `SELECT * WHERE model_name = 'gpt-4-turbo'`
- Identify production vs development feedback: `WHERE environment = 'production'`
