# Instructions for Coding Agents

This file contains instructions for developers working on the Azure Search and OpenAI demo application. It covers the overall code layout, how to add new data, how to add new azd environment variables, how to add new developer settings, how to add tests for new features, and how to run computer use testing with GPT-5.4.

Always keep this file up to date with any changes to the codebase or development process.
If necessary, edit this file to ensure it accurately reflects the current state of the project.

## Overall code layout

* app: Contains the main application code, including frontend and backend.
  * app/backend: Contains the Python backend code, written with Quart framework.
    * app/backend/approaches: Contains the different approaches
      * app/backend/approaches/approach.py: Base class for all approaches
      * app/backend/approaches/chatreadretrieveread.py: Chat approach, includes query rewriting step first
      * app/backend/approaches/promptmanager.py: Manages loading and rendering of Jinja2 prompt templates
      * app/backend/approaches/prompts/query_rewrite.system.jinja2: Jinja2 template used to rewrite the query based off search history into a better search query
      * app/backend/approaches/prompts/chat_query_rewrite_tools.json: Tools used by the query rewriting prompt
      * app/backend/approaches/prompts/chat_answer.system.jinja2: Jinja2 template for the system message used by the Chat approach to answer questions
      * app/backend/approaches/prompts/chat_answer.user.jinja2: Jinja2 template for the user message used by the Chat approach, including sources
    * app/backend/prepdocslib: Contains the document ingestion library used by both local and cloud ingestion
      * app/backend/prepdocslib/blobmanager.py: Manages uploads to Azure Blob Storage
      * app/backend/prepdocslib/cloudingestionstrategy.py: Builds the Azure AI Search indexer and skillset for the cloud ingestion pipeline
      * app/backend/prepdocslib/csvparser.py: Parses CSV files
      * app/backend/prepdocslib/embeddings.py: Generates embeddings for text and images using Azure OpenAI
      * app/backend/prepdocslib/figureprocessor.py: Generates figure descriptions for both local ingestion and the cloud figure-processor skill
      * app/backend/prepdocslib/fileprocessor.py: Orchestrates parsing and chunking of individual files
      * app/backend/prepdocslib/filestrategy.py: Strategy for uploading and indexing files (local ingestion)
      * app/backend/prepdocslib/htmlparser.py: Parses HTML files
      * app/backend/prepdocslib/integratedvectorizerstrategy.py: Strategy using Azure AI Search integrated vectorization
      * app/backend/prepdocslib/jsonparser.py: Parses JSON files
      * app/backend/prepdocslib/listfilestrategy.py: Lists files from local filesystem or Azure Data Lake
      * app/backend/prepdocslib/mediadescriber.py: Interfaces for describing images (Azure OpenAI GPT-4o, Content Understanding)
      * app/backend/prepdocslib/page.py: Data classes for pages, images, and chunks
      * app/backend/prepdocslib/parser.py: Base parser interface
      * app/backend/prepdocslib/pdfparser.py: Parses PDFs using Azure Document Intelligence or local parser
      * app/backend/prepdocslib/searchmanager.py: Manages Azure AI Search index creation and updates
      * app/backend/prepdocslib/servicesetup.py: Shared service setup helpers for OpenAI, embeddings, blob storage, etc.
      * app/backend/prepdocslib/strategy.py: Base strategy interface for document ingestion
      * app/backend/prepdocslib/textparser.py: Parses plain text and markdown files
      * app/backend/prepdocslib/textprocessor.py: Processes text chunks for cloud ingestion (merges figures, generates embeddings)
      * app/backend/prepdocslib/textsplitter.py: Splits text into chunks using different strategies
    * app/backend/app.py: The main entry point for the backend application.
  * app/functions: Azure Functions used for cloud ingestion custom skills (document extraction, figure processing, text processing). Each function bundles a synchronized copy of `prepdocslib`; run `python scripts/copy_prepdocslib.py` to refresh the local copies if you modify the library.
  * app/frontend: Contains the React frontend code, built with TypeScript, built with vite.
    * app/frontend/src/api: Contains the API client code for communicating with the backend.
    * app/frontend/src/components: Contains the React components for the frontend.
    * app/frontend/src/locales: Contains the translation files for internationalization.
      * app/frontend/src/locales/da/translation.json: Danish translations
      * app/frontend/src/locales/en/translation.json: English translations
      * app/frontend/src/locales/es/translation.json: Spanish translations
      * app/frontend/src/locales/fr/translation.json: French translations
      * app/frontend/src/locales/it/translation.json: Italian translations
      * app/frontend/src/locales/ja/translation.json: Japanese translations
      * app/frontend/src/locales/nl/translation.json: Dutch translations
      * app/frontend/src/locales/ptBR/translation.json: Portuguese translations
      * app/frontend/src/locales/tr/translation.json: Turkish translations
    * app/frontend/src/pages: Contains the main pages of the application
* infra: Contains the Bicep templates for provisioning Azure resources.
* tests: Contains the test code, including e2e tests, app integration tests, and unit tests.

## Adding new data

New files should be added to the `data` folder, and then either run scripts/prepdocs.sh or scripts/prepdocs.ps1 to ingest the data.

**IMPORTANT:** When adding or removing documents from the search index, also update the "Available Documents" list in `app/frontend/src/customizations/HelpAboutPanel.tsx` (look for the `INDEX-SOURCES` comment). This keeps the user-facing help panel in sync with the actual index contents.

## Adding a new azd environment variable

An azd environment variable is stored by the azd CLI for each environment. It is passed to the "azd up" command and can configure both provisioning options and application settings.
When adding new azd environment variables, update:

1. infra/main.parameters.json : Add the new parameter with a Bicep-friendly variable name and map to the new environment variable
1. infra/main.bicep: Add the new Bicep parameter at the top, and add it to the `appEnvVariables` object
1. .azdo/pipelines/azure-dev.yml: Add the new environment variable under `env` section
1. .github/workflows/azure-dev.yml: Add the new environment variable under `env` section

You may also need to update:

1. app/backend/prepdocs.py: If the variable is used in the ingestion script, retrieve it from environment variables here. Not always needed.
1. app/backend/app.py: If the variable is used in the backend application, retrieve it from environment variables in setup_clients() function. Not always needed.

## Adding a new setting to "Developer Settings" in RAG app

When adding a new developer setting, update:

* frontend:
  * app/frontend/src/api/models.ts : Add to ChatAppRequestOverrides
  * app/frontend/src/components/Settings.tsx : Add a UI element for the setting
  * app/frontend/src/locales/*/translations.json: Add a translation for the setting label/tooltip for all languages
  * app/frontend/src/pages/chat/Chat.tsx: Add the setting to the component, pass it to Settings

* backend:
  * app/backend/approaches/chatreadretrieveread.py :  Retrieve from overrides parameter
  * app/backend/app.py: Some settings may need to be sent down in the /config route.

## When adding tests for a new feature

All tests are in the `tests` folder and use the pytest framework.
There are three styles of tests:

* e2e tests: These use playwright to run the app in a browser and test the UI end-to-end. They are in e2e.py and they mock the backend using the snapshots from the app tests. (Before running e2e tests, make sure to run `npm run build` in app/frontend first to build the frontend code.)
* app integration tests: Mostly in test_app.py, these test the app's API endpoints and use mocks for services like Azure OpenAI and Azure Search.
* unit tests: The rest of the tests are unit tests that test individual functions and methods. They are in test_*.py files.

When adding a new feature, add tests for it in the appropriate file.
If the feature is a UI element, add an e2e test for it.
If it is an API endpoint, add an app integration test for it.
If it is a function or method, add a unit test for it.
Use mocks from tests/conftest.py to mock external services. Prefer mocking at the HTTP/requests level when possible.

For source-filter UI changes, prefer Playwright tests in `tests/e2e.py` that intercept `/api/categories` and the outgoing chat request, then assert the `include_category` override sent by the browser rather than relying on live retrieval behaviour.

For the built localhost app, use `scripts/test_live_citation_click.py` as a smoke test for the citation badge click flow. It verifies that clicking a rendered citation opens Supporting Content and creates a highlighted subsection anchor against the real built frontend served at `http://localhost:50505`.

For A/B evaluation of the current chat flow against the experimental planner-validator-repair retrieval prototype, use `scripts/test_planner_ab_compare.py`. It bootstraps the same backend clients locally, runs the current solution and the experimental path side by side, and writes a JSON summary to `scripts/planner_ab_results.json`.

For prompt-only evaluation of low-risk answer-prompt guardrails against the current chat flow, use `scripts/test_prompt_injection_compare.py`. It compares the current prompt behavior against a small injected prompt override and writes a JSON summary to `scripts/prompt_injection_results.json`.

For testing whether giving the query rewrite tool more knowledge about the search index structure improves retrieval, use `scripts/test_rewrite_index_awareness.py`. It injects index-awareness instructions (field names, category values, sourcepage patterns, search strategy guidance) into the rewrite prompt and compares answer quality across 12 test cases. Results are written to `scripts/rewrite_index_awareness_results.json`.

For testing whether pre-filtering by category improves court-specific queries, use `scripts/test_category_filter_impact.py`. It compares unfiltered search against category-filtered search for court-specific and CPR queries across 7 test cases. Results are written to `scripts/category_filter_results.json`. Key finding: category filtering can hurt broad CPR queries by over-narrowing the result set.

When you're running tests, make sure you activate the .venv virtual environment first:

```shell
source .venv/bin/activate
```

To check for coverage, run the following command:

```shell
pytest --cov --cov-report=annotate:cov_annotate
```

Open the cov_annotate directory to view the annotated source code. There will be one file per source file. If a file has 100% source coverage, it means all lines are covered by tests, so you do not need to open the file.

For each file that has less than 100% test coverage, find the matching file in cov_annotate and review the file.

If a line starts with a ! (exclamation mark), it means that the line is not covered by tests. Add tests to cover the missing lines.

## Source hierarchy regression workflow

The live source-hierarchy regression script is `scripts/test_source_hierarchy.py`.

Run it locally against a running app with:

```shell
APP_URL=http://localhost:50505 python scripts/test_source_hierarchy.py
```

There is also an opt-in GitHub Actions workflow at `.github/workflows/source-hierarchy-regression.yml`.
Use `workflow_dispatch` or `workflow_call` and pass the deployed app URL as `app_url`.

## Sending pull requests

When sending pull requests, make sure to follow the PULL_REQUEST_TEMPLATE.md format.

## Upgrading dependencies

### Python backend dependencies

To upgrade a particular package in the backend, use the following command, replacing `<package-name>` with the name of the package you want to upgrade:

```shell
cd app/backend && uv pip compile requirements.in -o requirements.txt --python-version 3.10 --upgrade-package <package-name>
```

After upgrading, run tests to verify compatibility:

```shell
source .venv/bin/activate
pytest tests/
```

### npm frontend dependencies

To upgrade a particular package in the frontend:

1. **Navigate to the frontend directory**:

   ```shell
   cd app/frontend
   ```

2. **Upgrade the package** (replace `<package-name>` with the package you want to upgrade):

   ```shell
   npm install <package-name>@latest
   ```

3. **Build the frontend** to verify the upgrade works:

   ```shell
   npm run build
   ```

4. **Run all tests** to ensure nothing broke:

   ```shell
   # Run e2e tests from the root directory
   cd ../..
   source .venv/bin/activate
   pytest tests/e2e.py
   ```

5. **Commit changes** if the upgrade is successful:

   ```shell
   git add package.json package-lock.json
   git commit -m "chore: upgrade <package-name> to <version>"
   ```

**Important notes for frontend upgrades**:

* When upgrading React or related core packages, you may need to upgrade multiple packages together (e.g., `react`, `react-dom`, `@types/react`, `@types/react-dom`)
* Some upgrades may require code changes for API compatibility - check the package's changelog
* For major version upgrades of UI libraries like Fluent UI or MSAL, review breaking changes carefully. Manual tests are required for any MSAL changes since the E2E tests do not cover authentication flows.
* If npm reports peer dependency conflicts, the `.npmrc` file has `legacy-peer-deps=true` which allows the install to proceed. This is currently needed because `react-helmet-async` declares peer dependencies on React 17/18, but works fine with React 19.

## Checking Python type hints

To check Python type hints, use the following command:

```shell
ty check
```

Note that we do not currently enforce type hints in the tests folder, as it would require adding a lot of `# type: ignore` comments to the existing tests.
We only enforce type hints in the main application code and scripts.

## Python code style

Do not use single underscores in front of "private" methods or variables in Python code. We do not follow that convention in this codebase, since this is an application and not a library.

## Computer Use Testing with GPT-5.4

The script `scripts/computer_use_test.py` uses Azure OpenAI GPT-5.4 computer use to drive a Playwright browser against the locally-running app. The model sees screenshots, decides on actions (clicks, typing, scrolling), and the script executes them in a real Chromium window.

### Prerequisites

1. App running locally: `./app/start.sh` (serves at `http://localhost:50505`)
2. Azure OpenAI deployment of `gpt-5.4` (or `computer-use-preview`). Register at <https://aka.ms/OAI/gpt54access>
3. Azure CLI logged in: `az login`
4. Playwright browsers installed: `playwright install chromium`

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPUTER_USE_AZURE_OPENAI_ENDPOINT` | *(required)* | Azure OpenAI resource endpoint, e.g. `https://myresource.openai.azure.com` |
| `COMPUTER_USE_MODEL` | `gpt-5.4` | Deployment name |
| `COMPUTER_USE_TARGET_URL` | `http://localhost:50505` | URL the browser navigates to |
| `COMPUTER_USE_MAX_ITERATIONS` | `12` | Hard cap on action-screenshot loops |
| `COMPUTER_USE_DISPLAY_WIDTH` | `1440` | Browser viewport width |
| `COMPUTER_USE_DISPLAY_HEIGHT` | `900` | Browser viewport height |

If `COMPUTER_USE_AZURE_OPENAI_ENDPOINT` is not set explicitly, the harness falls back to the repo's standard `.env` value for `AZURE_OPENAI_ENDPOINT`.

If `COMPUTER_USE_MODEL` is not set explicitly, the harness prefers a compatible deployment from the repo's standard `.env` values and otherwise falls back to `gpt-5.4`. It intentionally avoids reusing `mini` or `nano` deployments for computer use.

### Running

```shell
source .venv/bin/activate

# Interactive mode (type tasks at the prompt):
python scripts/computer_use_test.py

# Single-task mode:
python scripts/computer_use_test.py --task "Ask 'What is the dental plan?' and report the answer"

# Headless mode (no visible browser window):
python scripts/computer_use_test.py --headless --task "Search for eye exam coverage"

# Detailed citation regression suite:
python scripts/computer_use_test.py --suite detailed-citations

# Cross-source highlight regression suite:
python scripts/computer_use_test.py --suite highlight-regression
```

### Detailed citation suite

The built-in `detailed-citations` suite runs multiple source-filtered scenarios across:

* All Sources
* Civil Procedure Rules and Practice Directions
* Commercial Court Guide
* Chancery Guide
* King's Bench Division Guide
* Technology and Construction Court Guide
* Patents Court Guide

Each scenario asks a grounded legal question, checks in-text and bottom citation rendering, clicks citations to verify the supporting-content panel, and records issues in the saved JSON log.

The scenario catalog is grounded in the v3 index patterns already used by the repo's citation tests, including:

* CPR parts such as Part 1, Part 3, Part 24 and Part 52
* Practice Directions such as PD44 and PD19A
* Pre-Action Protocol sources that may collapse to short labels like `Pre`
* Court Guide citations with hierarchical sourcepages, annexes, and PDF-style sourcefile names

### Highlight regression suite

The built-in `highlight-regression` suite focuses specifically on supporting-content highlighting across source families. It uses screenshot-driven computer use checks to verify that the supporting-content panel keeps the full section visible while the highlighted block starts at the cited subsection itself, excludes the adjacent heading immediately above it, and does not spill into the next subsection.

The suite currently covers:

* All Sources mixed retrieval
* Civil Procedure Rules and Practice Directions
* Commercial Court Guide
* Chancery Guide
* King's Bench Division Guide
* Technology and Construction Court Guide
* Patents Court Guide
* Circuit Commercial Court Guide

### Example tasks

These are good first tasks to verify the harness works:

* `"Ask 'What is the dental plan?' and report the answer"`
* `"Open the first citation and describe what you see"`
* `"Open Developer Settings and change the retrieval mode to Vectors"`
* `"Search for annual eye exam coverage and list the sources"`

### Session logs

Each run writes a JSON log to `scripts/computer_use_logs/` containing all screenshots (base64), model responses, and actions. The directory has a `.gitignore` to exclude generated logs.

## Deploying the application

To deploy the application, use the `azd` CLI tool. Make sure you have the latest version of the `azd` CLI installed. Then, run the following command from the root of the repository:

```shell
azd up
```

That command will BOTH provision the Azure resources AND deploy the application code.

If you only changed the Bicep templates and want to re-provision the Azure resources, run:

```shell
azd provision
```

If you only changed the application code and want to re-deploy the code, run:

```shell
azd deploy
```

If you are using cloud ingestion and only want to deploy individual functions, run the necessary deploy commands, for example:

```shell
azd deploy document-extractor
azd deploy figure-processor
azd deploy text-processor
```
