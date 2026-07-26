# Quick Start: Running Customization Tests

Complete guide for running all new test suites for customization features.

## Pre-Workflow Local Triage Prompt

Use this prompt with `scripts/computer_use_test.py --task` before dispatching an
Azure or GitHub Actions release workflow. It is deliberately fail-closed: a
missing citation, stale page, ambiguous source, failed click, or incomplete
check is an issue and must not be reported as a pass.

```text
You are performing a local pre-workflow regression triage for the Legal RAG
application. Test the running localhost app only. Do not call Azure APIs, do
not upload or delete Search documents, do not modify production or rollback
indexes, do not promote a candidate, and do not change any validation oracle.

Your goal is to find the first concrete defect that would make a staging v4
release unsafe to start. Use the browser UI and visible application behavior;
do not infer success from the answer text alone.

Run these checks in order and stop investigating new areas after the first
reproducible failure, while still recording all checks already completed:

1. Confirm the page is the local target at `http://localhost:50505`. If it is
  not reachable, report `BLOCKED` with the exact navigation or startup error.
2. Ask exactly:
  `What is CPR Part 24 rule 24.2 and the test for summary judgment?`
3. Confirm the answer discusses rule `24.2` and the summary judgment test.
  This text check is not sufficient by itself.
4. Inspect every rendered citation and record its visible label plus these DOM
  attributes when available: `data-subsection-id`, `data-sourcepage`,
  `data-sourcefile`, `data-category`, and `data-citation-path`.
5. Require at least one CPR citation whose subsection identity is exactly
  `24.2`. A citation for `24.1` is a failure even if the visible answer says
  `24.2`.
6. Click the canonical `24.2` citation. Confirm Supporting Content opens,
  remains associated with Part 24, and does not silently select another
  source or subsection.
7. Confirm the highlighted block is non-empty, starts at the `24.2` heading or
  its exact canonical body, includes evidence for rule `24.2`, and excludes
  the preceding `24.1` heading and the next subsection heading. Do not accept
  a highlight merely because some Part 24 text is visible.
8. Return to the chat and test the source filter for
  `Civil Procedure Rules and Practice Directions`. Ask the same question and
  confirm every displayed citation belongs to that category.
9. Reset to `All Sources`. Ask:
  `What is the Chancery Guide guidance on summary judgment or strike-out?`
  Confirm the answer has a Chancery Guide citation and that its source
  metadata is not mislabeled as CPR.
10. Reload the page and repeat one citation click. A reload, empty result,
   stale Supporting Content panel, or citation that cannot be reopened is a
   failure.
11. Resize or use a mobile viewport if available. Confirm the source filter,
   answer, citation, and Supporting Content controls remain readable and do
   not overlap or become unreachable.

For every issue, record:

- the exact question and source filter;
- the visible answer and citation label;
- all available citation metadata attributes;
- the browser action that exposed the issue;
- the expected behavior;
- the observed behavior;
- whether the issue is `BLOCKING`, `WARNING`, or `UNAVAILABLE`;
- a minimal reproduction that a developer can run locally.

Do not repair code, retry indefinitely, reinterpret `24.1` as `24.2`, or mark
an unavailable check as passed. Do not report `READY FOR HUMAN APPROVAL`.
Return exactly one final JSON object with this shape, followed by a concise
plain-text diagnosis:

{
  "status": "PASS" | "BLOCKED",
  "target": "http://localhost:50505",
  "checks": [
   {"name": "...", "status": "PASS" | "FAIL" | "UNAVAILABLE", "evidence": "..."}
  ],
  "issues": [
   {"severity": "BLOCKING" | "WARNING" | "UNAVAILABLE", "title": "...", "reproduction": "...", "evidence": "..."}
  ],
  "firstConcreteFailure": "..." | null,
  "releaseStatus": "BLOCKED: DO NOT PROMOTE"
}
```

Run the prompt locally with:

```bash
# Terminal 1: start the local app
./app/start.sh

# Terminal 2: run the deterministic checks first
npm --prefix app/frontend test -- --run src/components/Answer/__tests__/AnswerParser.test.ts
npm --prefix app/frontend run build
source .venv/bin/activate
python scripts/test_live_citation_click.py

# Terminal 2: run the browser triage prompt
python scripts/computer_use_test.py --headless --task "$(sed -n '/^You are performing a local pre-workflow/,/^```$/p' docs/customizations/RUNNING_TESTS.md | sed '1d;$d')"

# Existing broad regressions
python scripts/computer_use_test.py --headless --suite highlight-regression
python scripts/computer_use_test.py --headless --suite detailed-citations
```

Do not dispatch the GitHub workflow until the local result is complete and all
required checks pass. A local pass is necessary but does not authorize
promotion; the release remains `BLOCKED: DO NOT PROMOTE` until fresh staging
evidence satisfies the repository release process.

## Summary of Test Files

### Backend Tests (6 files, 109 tests)
- `tests/test_customizations_config.py` - Feature flags and metadata
- `tests/test_customizations_prompts.py` - Prompt extensions
- `tests/test_customizations_citation_builder.py` - Citation building
- `tests/test_customizations_routes.py` - API endpoints
- `tests/test_feature_flags.py` - Feature flag toggles and degradation
- `tests/e2e_customizations.py` - End-to-end scenarios

### Frontend Tests (3 files, 48 tests)
- `app/frontend/src/customizations/__tests__/config.test.ts`
- `app/frontend/src/customizations/__tests__/useCategories.test.ts`
- `app/frontend/src/customizations/__tests__/useMobile.test.ts`

---

## Running Tests

### Option 1: Run All Tests

#### Backend
```bash
cd /Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2

# Activate virtual environment
source .venv/bin/activate

# Run all customization tests
pytest tests/test_customizations*.py tests/test_feature_flags.py tests/e2e_customizations.py -v

# With coverage
pytest tests/test_customizations*.py tests/test_feature_flags.py --cov=customizations --cov-report=html -v
```

#### Frontend
```bash
cd app/frontend

# Run all customization tests
npm test -- customizations/__tests__ --coverage

# Or with watch mode
npm test -- customizations/__tests__ --watch
```

---

### Option 2: Run Specific Test Categories

#### Backend Feature Flag Tests
```bash
pytest tests/test_feature_flags.py -v
```

#### Backend Route Tests
```bash
pytest tests/test_customizations_routes.py -v
```

#### Backend Citation Builder Tests
```bash
pytest tests/test_customizations_citation_builder.py -v
```

#### Frontend Config Tests
```bash
npm test -- config.test.ts --coverage
```

#### Frontend Hook Tests
```bash
npm test -- useCategories.test.ts useMobile.test.ts --coverage
```

#### E2E Tests (Slower, Playwright required)
```bash
pytest tests/e2e_customizations.py -v

# Run specific E2E test class
pytest tests/e2e_customizations.py::TestAdminModeFeatures -v

# With headed browser for debugging
pytest tests/e2e_customizations.py -v --headed
```

---

### Option 3: Run by Feature

#### Category Filter Feature
**Backend:**
```bash
pytest tests/test_customizations_routes.py::TestCategoriesEndpoint -v
```

**Frontend:**
```bash
npm test -- useCategories.test.ts --coverage
```

**E2E:**
```bash
pytest tests/e2e_customizations.py::TestCategoryFilterFeature -v
```

#### Admin Mode Feature
**Frontend Config:**
```bash
npm test -- config.test.ts -t "isAdminMode" --coverage
```

**E2E:**
```bash
pytest tests/e2e_customizations.py::TestAdminModeFeatures -v
```

#### Feedback Feature
**Backend:**
```bash
pytest tests/test_customizations_routes.py::TestFeedbackEndpoint -v
```

**E2E:**
```bash
pytest tests/e2e_customizations.py::TestFeedbackFeature -v
```

#### Citations Feature
**Backend:**
```bash
pytest tests/test_customizations_citation_builder.py -v
```

**E2E:**
```bash
pytest tests/e2e_customizations.py::TestCitationSanitization -v
```

---

## Quick Check: Did Tests Work?

### After Backend Tests
```bash
# Should see output like:
# tests/test_customizations_config.py ........................... PASSED
# tests/test_customizations_prompts.py .......................... PASSED
# tests/test_customizations_citation_builder.py ................. PASSED
# tests/test_customizations_routes.py ........................... PASSED
# tests/test_feature_flags.py ................................... PASSED

# Total: 109 tests passed
```

### After Frontend Tests
```bash
# Should see output like:
# PASS app/frontend/src/customizations/__tests__/config.test.ts
# PASS app/frontend/src/customizations/__tests__/useCategories.test.ts
# PASS app/frontend/src/customizations/__tests__/useMobile.test.ts

# Tests: 48 passed, 48 total
```

---

## Test Output Interpretation

### Successful Test Run
```
============================= test session starts ==============================
platform darwin -- Python 3.10.x, pytest-x.x.x, py-x.x.x, pluggy-x.x.x
collected 109 items

tests/test_customizations_config.py ...................... PASSED [ 22%]
tests/test_customizations_prompts.py .................... PASSED [ 35%]
tests/test_customizations_citation_builder.py ........... PASSED [ 62%]
tests/test_customizations_routes.py ..................... PASSED [ 80%]
tests/test_feature_flags.py ............................. PASSED [ 99%]
tests/e2e_customizations.py .............................. PASSED [100%]

============================== 109 passed in 15.32s ==============================
```

### Test Failure Example
```
FAILED tests/test_customizations_config.py::test_is_feature_enabled_returns_true_for_enabled_feature

AssertionError: assert False == True
```
→ Check if feature flag was changed, or if config file was modified

---

## Coverage Reports

### Backend Coverage
```bash
pytest tests/test_customizations*.py tests/test_feature_flags.py \
  --cov=customizations \
  --cov-report=html \
  --cov-report=term-missing

# View: htmlcov/index.html
```

### Frontend Coverage
```bash
npm test -- customizations/__tests__ --coverage

# View: coverage/lcov-report/index.html
```

---

## Continuous Integration

### For GitHub Actions

Add to `.github/workflows/`:

```yaml
name: Test Customizations

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: |
          pip install -r requirements-dev.txt
          pytest tests/test_customizations*.py tests/test_feature_flags.py --cov=customizations
  
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: |
          cd app/frontend
          npm install
          npm test -- customizations/__tests__ --coverage
```

---

## Troubleshooting

### Backend Tests Not Finding Modules
```bash
# Make sure you're in the right directory
cd /Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2

# Activate venv
source .venv/bin/activate

# Try again
pytest tests/test_customizations_config.py -v
```

### Frontend Tests Failing
```bash
cd app/frontend

# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Then run tests
npm test -- customizations/__tests__
```

### E2E Tests Timing Out
```bash
# E2E tests may need more time
pytest tests/e2e_customizations.py -v --timeout=30

# Or run headless (no UI):
pytest tests/e2e_customizations.py -v --browser=firefox
```

### Import Errors in Backend Tests
```bash
# Make sure conftest.py is in tests/ directory
ls -la tests/conftest.py

# And customizations module is importable
python -c "from customizations.config import is_feature_enabled; print('OK')"
```

---

## Feature Flag Testing Quick Reference

### Test All Flags Enabled
```bash
pytest tests/test_feature_flags.py::TestBackendFeatureFlagToggles::test_enable_all_features -v
```

### Test All Flags Disabled
```bash
pytest tests/test_feature_flags.py::TestBackendFeatureFlagToggles::test_disable_all_features -v
```

### Test Graceful Degradation
```bash
pytest tests/test_feature_flags.py::TestGracefulDegradationWithFlags -v
```

### Test Feature Combinations
```bash
pytest tests/test_feature_flags.py::TestFeatureFlagCombinations -v
```

---

## Document Reference

- **Detailed Test Documentation**: [docs/customizations/TEST_SUITE.md](docs/customizations/TEST_SUITE.md)
- **Customization Guidelines**: [docs/customizations/README.md](docs/customizations/README.md)
- **Architecture Overview**: [AGENTS.md](AGENTS.md)

---

## Summary

✅ **171 total tests** covering all customization features  
✅ **Visible features**: Category filter, Feedback, Citations, Legal prompts  
✅ **Hidden features**: Admin mode, Ask page, Splash screen  
✅ **Feature flags**: Toggle tests, degradation, combinations  
✅ **CI/CD ready**: No external dependencies, can run in parallel  

**Run all tests in < 2 minutes**
