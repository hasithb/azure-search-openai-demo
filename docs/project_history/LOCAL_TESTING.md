# Local Testing Guide - Enhanced Feedback System

Before deploying to Azure with `azd up`, test the feedback system locally with these scripts.

## Quick Start

### Step 1: Start the Backend Server

In **Terminal 1**, run:

```bash
./run_local_test.sh
```

You should see:

```bash
✓ Virtual environment activated
✓ Environment loaded
🏃 Starting Quart backend server...
Backend will be available at: http://127.0.0.1:50505
```

The server will start with:

- **URL**: http://127.0.0.1:50505
- **Hot reload**: Enabled (changes auto-reload)
- **Authentication**: Disabled (for testing)
- **Blob storage**: Enabled for feedback

Keep this terminal running.

### Step 2: Test the Feedback Endpoint

In **Terminal 2**, run:

```bash
./test_feedback_local.sh
```

You should see:

```bash
✅ Test 1 PASSED: Feedback accepted
✅ Test 2 PASSED: Feedback with thoughts accepted
✅ Test 3 PASSED: Verification local storage
✅ ALL TESTS PASSED
```

### Step 3: Check Saved Feedback Files

The feedback is automatically saved to:

```bash
feedback_data/local/
├── 2026-01-10t14-32-00_test-001.json          # User-visible feedback
└── 2026-01-10t14-32-00_test-002.json          # User-visible feedback
```

Check the contents:

```bash
cat feedback_data/local/*.json | jq
```

Expected output shows:

- ✅ User prompt and AI response
- ✅ Safe thoughts only (no system prompts)
- ✅ Deployment metadata
- ✅ User rating and comment

## Manual Testing

### Test 1: Simple Feedback

```bash
curl -X POST http://127.0.0.1:50505/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "message_id": "manual-test-001",
    "rating": "helpful",
    "comment": "Great answer!"
  }'
```

Expected response:

```json
{"status": "received"}
```

### Test 2: Feedback with Context

```bash
curl -X POST http://127.0.0.1:50505/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "message_id": "manual-test-002",
    "rating": "helpful",
    "comment": "Good citations",
    "context_shared": true,
    "user_prompt": "What is CPR Part 52?",
    "ai_response": "CPR Part 52 covers appeals...",
    "thoughts": [
      {"title": "Search Query", "description": "CPR Part 52", "props": {}},
      {"title": "Retrieved Documents", "description": "[docs]", "props": {}}
    ]
  }'
```

### Test 3: With System Prompt (Should Be Filtered)

```bash
curl -X POST http://127.0.0.1:50505/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "message_id": "manual-test-003",
    "rating": "unhelpful",
    "context_shared": true,
    "user_prompt": "How do appeals work?",
    "ai_response": "Appeals work through...",
    "thoughts": [
      {"title": "Prompt to generate answer", "description": "You are a legal expert...", "props": {}}
    ]
  }'
```

Then verify the system prompt was filtered:

```bash
cat feedback_data/local/2026-01-10t*_manual-test-003.json | jq '.context.thoughts'
```

Should show **empty array** (system prompt removed).

## Verification Checklist

✅ Backend starts without errors
✅ Feedback endpoint accepts POST requests
✅ Response returns `{"status": "received"}`
✅ Files created in `feedback_data/local/`
✅ System prompts removed from user files
✅ Deployment metadata included
✅ Thoughts are filtered correctly

## Troubleshooting

### Backend won't start

```bash
# Check if port 50505 is in use
lsof -i :50505

# Kill if needed
kill -9 <PID>
```

### Azure credentials not loaded

```bash
# Manually load environment
azd auth login
azd env select cpr-rag
```

### Import errors

```bash
# Reinstall dependencies
source .venv-upgrade/bin/activate
pip install -r app/backend/requirements.txt
```

### No feedback files created

```bash
# Check if directory was created
ls -la feedback_data/

# Check backend logs for errors
```

## Performance Notes

- First request may be slower (initialization)
- Hot reload works for backend Python files
- Each feedback creates 1-2 JSON files (user + admin)
- Files are gzipped when uploaded to blob storage

## Next Steps

Once local testing passes:

```bash
# Deploy to Azure
azd up

# This will:
# • Provision Azure resources (if first time)
# • Deploy backend and frontend
# • Enable blob storage for production feedback
# • Configure deployment metadata
```

## Files

| File | Purpose |
|------|---------|
| `run_local_test.sh` | Starts the backend with proper environment |
| `test_feedback_local.sh` | Automated tests for feedback endpoint |
| `verify_feedback_blob.py` | Validates blob storage logic (no pytest needed) |

## Support

If issues arise:

1. Check backend logs in Terminal 1
1. Review `feedback_data/local/` for saved files
1. Verify Azure environment: `azd env get-values | grep AZURE_`
1. Check thought filtering: `python verify_feedback_blob.py`
