# GitHub Copilot Instructions for Azure Search OpenAI Demo (Legal RAG)

This repository is a customized fork of `Azure-Samples/azure-search-openai-demo` with legal-domain behavior layered on top of the upstream app.

## Architecture overview

This codebase follows a merge-safe strategy:

- Put fork-specific code in `/customizations/` folders whenever possible.
- Keep changes to upstream-owned files small and clearly marked with `CUSTOM:` comments.
- Treat prompt files and a handful of core integration files as intentional exceptions that must be reviewed during upstream merges.

The active application surface in this fork is chat-first. Do not assume the legacy `Ask` page is still a customization target.

## Key directories

| Directory | Purpose |
|-----------|---------|
| `app/backend/customizations/` | Backend feature flags, routes, metadata helpers, subsection extraction |
| `app/frontend/src/customizations/` | Frontend feature flags, source controls, citation helpers, UX components |
| `app/backend/approaches/prompts/` | Legal-domain prompt customizations that must be reviewed during upgrades |

## Custom code locations

### Backend customizations (`app/backend/customizations/`)

```text
customizations/
|-- __init__.py
|-- config.py                   # Feature flags, source display names, deployment metadata
|-- subsection_extractor.py     # Shared subsection parsing
|-- approaches/
|   |-- citation_builder.py     # Enhanced legal citation generation
|   `-- source_processor.py     # Source shaping and metadata preservation
`-- routes/
    |-- categories.py           # GET /api/categories
    `-- feedback.py             # POST /api/feedback
```

### Frontend customizations (`app/frontend/src/customizations/`)

```text
customizations/
|-- index.ts                    # Barrel exports
|-- config.ts                   # Frontend feature flags and admin mode
|-- answerParagraphs.ts         # Readability formatting
|-- chunkDeduplicator.ts        # Subsection-aware supporting-content dedupe
|-- citationMetadata.ts         # Structured citation metadata + path building
|-- citationSanitizer.ts        # Citation cleanup and fuzzy matching
|-- externalSourceHandler.ts    # Iframe-blocked source handling
|-- useCategories.ts            # Dynamic source loading
|-- useMobile.ts                # Mobile breakpoint and label helpers
|-- ChatInputControls.tsx       # Source filter + search depth controls
|-- CitationMetadataDisplay.tsx # Metadata badges for supporting content
|-- DataPrivacyNotice.tsx
|-- HelpAboutPanel.tsx
|-- LegalFeedback.tsx
|-- SplashScreen.tsx
|-- mobile.css
`-- __tests__/
```

## Integration points

When modifying upstream files, these are the main places where the fork is connected back into the app.

### Backend app setup (`app/backend/app.py`)

```python
from customizations.config import fetch_available_sources, is_deployed_ui_compat_enabled, is_feature_enabled
from customizations.routes import categories_bp, feedback_bp

app.register_blueprint(categories_bp)
app.register_blueprint(feedback_bp)

available_sources = await fetch_available_sources(search_client)
```

### Backend approaches (`app/backend/approaches/approach.py`, `chatreadretrieveread.py`)

These files carry legal-specific retrieval, citation, and source-metadata hooks. Review their `CUSTOM:` blocks after every upstream merge.

### Frontend chat flow (`app/frontend/src/pages/chat/Chat.tsx`)

```typescript
import { HelpAboutPanel, buildCitationLabel } from "../../customizations";
import { ChatInputControls, MobileDropdownPanel } from "../../customizations/ChatInputControls";
import { useCategories } from "../../customizations/useCategories";
import { isFeatureEnabled, isAdminMode } from "../../customizations/config";
import { useIsMobile } from "../../customizations/useMobile";
import { isIframeBlocked } from "../../customizations/externalSourceHandler";
```

### Frontend answer and supporting-content flow

Review these files together because the citation metadata pipeline crosses all of them:

- `app/frontend/src/components/Answer/Answer.tsx`
- `app/frontend/src/components/Answer/AnswerParser.tsx`
- `app/frontend/src/components/AnalysisPanel/AnalysisPanel.tsx`
- `app/frontend/src/components/SupportingContent/SupportingContent.tsx`
- `app/frontend/src/components/QuestionInput/QuestionInput.tsx`

### Frontend shell and local dev

```typescript
// app/frontend/src/pages/layout/Layout.tsx
import { HelpAboutPanel } from "../../customizations/HelpAboutPanel";
import { SplashScreen } from "../../customizations/SplashScreen";

// app/frontend/src/index.tsx
import "./customizations/mobile.css";

// app/frontend/vite.config.ts
"/api/categories": "http://localhost:50505",
"/api/feedback": "http://localhost:50505"
```

## Coding guidelines

### When adding new features

1. Add new fork-specific code to `/customizations/` folders whenever practical.
2. Use feature flags in `app/backend/customizations/config.py` or `app/frontend/src/customizations/config.ts`.
3. Minimize edits to upstream-owned files and mark them with `CUSTOM:` comments.
4. Export public customization APIs through `__init__.py` or `index.ts`.
5. Add or update tests under `app/frontend/src/customizations/__tests__/` or the relevant backend/test area.

### When modifying prompts

The prompts in `app/backend/approaches/prompts/` are intentionally outside `/customizations/` because they are core business logic. When upstream updates them:

- Review changes manually.
- Preserve legal citation rules and corpus-specific guidance.
- Recheck how prompt wording interacts with numbered citation output and source-hierarchy behavior.

### Feature flag pattern

Backend:

```python
from customizations.config import is_feature_enabled

if is_feature_enabled("category_filter"):
    ...
```

Frontend:

```typescript
import { isFeatureEnabled } from "../../customizations";

if (isFeatureEnabled("structuredCitationMatching")) {
    ...
}
```

## Upgrading from upstream

When pulling updates from `Azure-Samples/azure-search-openai-demo`:

1. Safe files with low merge risk:
   - Everything under `app/backend/customizations/`
   - Everything under `app/frontend/src/customizations/`

2. Review and usually reapply these integration points:
   - `app/backend/app.py`
   - `app/backend/approaches/approach.py`
   - `app/backend/approaches/chatreadretrieveread.py`
   - `app/frontend/src/pages/chat/Chat.tsx`
   - `app/frontend/src/pages/layout/Layout.tsx`
   - `app/frontend/src/components/QuestionInput/QuestionInput.tsx`
   - `app/frontend/src/components/Answer/Answer.tsx`
   - `app/frontend/src/components/Answer/AnswerParser.tsx`
   - `app/frontend/src/components/AnalysisPanel/AnalysisPanel.tsx`
   - `app/frontend/src/components/SupportingContent/SupportingContent.tsx`
   - `app/frontend/src/index.tsx`
   - `app/frontend/vite.config.ts`

3. Prompt files to merge carefully:
   - `chat_answer.system.jinja2`
   - `chat_answer.user.jinja2`
   - `query_rewrite.system.jinja2`
   - `query_rewrite.user.jinja2`
   - Vision prompt files when relevant

Do not carry forward stale assumptions from older docs such as `Ask.tsx`, `CategoryDropdown/`, `SearchBoxWithCategories/`, `prompt_extensions.py`, or `thought_filter.py` unless those files are reintroduced.

## Testing

- Frontend customization tests live under `app/frontend/src/customizations/__tests__/`.
- Backend and integration tests live under `tests/`.
- Additional operational docs live under `docs/customizations/`.

## Important notes

1. Keep citation output in `[1][2][3]` format.
2. If you add a fork-specific integration outside `/customizations/`, update `docs/customizations/README.md` in the same change.
3. Keep `CUSTOM:` comments intact when refactoring upstream-owned files.
4. Treat the customization docs as part of the merge surface, not optional documentation.