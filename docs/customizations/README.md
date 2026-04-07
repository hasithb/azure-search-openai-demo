# Customizations Guide: Legal RAG System

This document describes the fork-specific customizations that currently differentiate this repository from the upstream `Azure-Samples/azure-search-openai-demo` codebase.

The goal is twofold:

1. Keep merge-safe code isolated in `/customizations/` folders wherever practical.
2. Make the unavoidable upstream integration points explicit so they can be reviewed after upstream pulls.

This guide reflects the current chat-first application state. The legacy `Ask` page is no longer part of the live customization surface in this fork.

## Overview

This fork turns the sample app into a legal-domain assistant focused on UK Civil Procedure Rules, Practice Directions, and court guides. Compared with upstream `main`, the active customizations currently include:

- Dynamic source filtering driven from the search index
- Legal citation cleanup and numbered citation handling
- Structured citation metadata for precise supporting-content matching
- Subsection-aware supporting-content grouping and deduplication
- Consent-based legal feedback capture with deployment metadata
- Mobile-specific chat controls and responsive layout changes
- End-user help/about content and splash-screen UX
- Prompt awareness of available document sources and legal retrieval behavior

## Merge-safe layout

Most fork-specific code lives in dedicated `/customizations/` folders.

### Backend customizations

Location: `/app/backend/customizations/`

```text
customizations/
|-- __init__.py                 # Shared exports for backend customization helpers
|-- config.py                   # Feature flags, source display names, deployment metadata
|-- subsection_extractor.py     # Shared subsection parsing used across legal citation flows
|-- approaches/
|   |-- __init__.py
|   |-- citation_builder.py     # Enhanced legal citation and subsection labeling
|   `-- source_processor.py     # Source shaping for frontend/supporting content
`-- routes/
    |-- __init__.py
    |-- categories.py           # GET /api/categories
    `-- feedback.py             # POST /api/feedback
```

### Frontend customizations

Location: `/app/frontend/src/customizations/`

```text
customizations/
|-- index.ts                    # Barrel exports for live customization surface
|-- config.ts                   # Frontend feature flags and admin mode helper
|-- answerParagraphs.ts         # Readability formatting for long answers
|-- chunkDeduplicator.ts        # Subsection-aware deduplication for supporting content
|-- citationMetadata.ts         # Structured citation metadata extraction and path building
|-- citationSanitizer.ts        # Citation cleanup and fuzzy citation matching helpers
|-- externalSourceHandler.ts    # Non-embeddable source handling
|-- useCategories.ts            # Dynamic source/category loading
|-- useMobile.ts                # Mobile breakpoint logic and label abbreviations
|-- ChatInputControls.tsx       # Desktop/mobile source and search-depth controls
|-- CitationMetadataDisplay.tsx # Supporting-content metadata badges
|-- DataPrivacyNotice.tsx       # Privacy UI helper
|-- HelpAboutPanel.tsx          # End-user help/about drawer
|-- LegalFeedback.tsx           # Feedback UI for answer-level reporting
|-- SplashScreen.tsx            # Intro splash screen used by the layout shell
|-- mobile.css                  # Mobile-specific layout overrides
`-- __tests__/                  # Frontend tests for customization modules
```

## Core non-customizations that still require review

Some legal-domain behavior intentionally lives outside `/customizations/` because it modifies core application flow or prompts.

### Prompt files

Location: `/app/backend/approaches/prompts/`

Current legal prompt work is primarily in these files:

- `chat_answer.system.jinja2`
- `chat_answer.user.jinja2`
- `query_rewrite.system.jinja2`
- `query_rewrite.user.jinja2`
- `ask_answer_question_vision.prompty`
- `chat_answer_question_vision.prompty`

These files are not merge-safe by design. They must be reviewed during upstream updates.

### Upstream files with required integration points

The following upstream-owned files currently carry active customization hooks and should be reviewed after every upstream merge:

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

## Current feature inventory

### 1. Dynamic source filtering

Backend:

- `/api/categories` is served from `app/backend/customizations/routes/categories.py`
- `customizations.config.SOURCE_DISPLAY_NAMES` normalizes raw search-index categories into user-facing names
- `fetch_available_sources()` also feeds source names into prompt context at app startup

Frontend:

- `useCategories.ts` loads the available sources dynamically
- `ChatInputControls.tsx` renders the source picker for desktop and mobile
- `Chat.tsx` enforces source selection before searching when the filter is enabled
- `Settings.tsx` also accepts dynamic categories for admin/developer settings

Important difference from older documentation:

- The active UI no longer uses `CategoryDropdown/` or `SearchBoxWithCategories/`
- The active integration is through `ChatInputControls.tsx` and `QuestionInput.leftOfSend`

### 2. Legal citation pipeline

Backend:

- `citation_builder.py` produces richer legal citation labels
- `source_processor.py` preserves source metadata used by the frontend
- `subsection_extractor.py` provides shared subsection parsing logic
- `chatreadretrieveread.py` and `approach.py` carry citation metadata into the response data points

Frontend:

- `citationSanitizer.ts` repairs malformed numeric citations, strips legal-text artifacts, and supports fuzzy citation matching
- `citationMetadata.ts` extracts structured citation metadata from data points and builds stable in-app citation paths
- `AnswerParser.tsx` resolves numbered citations like `[1]` back to the correct source metadata
- `Answer.tsx` forwards structured citation metadata when the user opens supporting content

### 3. Supporting-content precision and subsection awareness

This fork goes beyond the upstream string-matching behavior.

- `SupportingContent.tsx` groups source cards by logical section, not just document filename
- `subsectionMatcher.ts` performs structured subsection matching when metadata is available
- `chunkDeduplicator.ts` preserves distinct subsections from the same source document
- `CitationMetadataDisplay.tsx` can render subsection/source badges for debugging or admin workflows

These pieces work together to keep supporting-content highlighting focused on the cited subsection rather than collapsing unrelated sections from the same guide.

### 4. Feedback, privacy, and deployment traceability

Backend:

- `routes/feedback.py` exposes `POST /api/feedback`
- Feedback records can include deployment metadata via `get_deployment_metadata()`
- Context sharing is consent-based: prompts, answer text, conversation history, and thoughts are only included when the user opts in

Frontend:

- `LegalFeedback.tsx` provides per-answer helpful/unhelpful controls
- `HelpAboutPanel.tsx` and `DataPrivacyNotice.tsx` provide end-user help/privacy guidance
- `vite.config.ts` proxies both `/api/categories` and `/api/feedback` in local development

Important difference from older documentation:

- The current feedback route does not rely on the older `thought_filter.py` documentation flow
- The live route stores a single feedback record with optional shared context and deployment metadata

See `docs/customizations/FEEDBACK_SYSTEM.md` for deeper operational detail.

### 5. Mobile and end-user UX customizations

The current frontend divergence from upstream is broader than just legal logic.

- `useMobile.ts` and `mobile.css` drive responsive behavior and short labels on smaller screens
- `ChatInputControls.tsx` swaps desktop dropdowns for a mobile settings button and dropdown panel
- `Chat.tsx` shows the analysis panel as a modal overlay on mobile
- `SplashScreen.tsx` is integrated into `Layout.tsx`
- `HelpAboutPanel.tsx` is mounted in the layout shell rather than being chat-page only
- `index.tsx` imports `mobile.css` globally

### 6. Admin-mode and debugging affordances

Frontend feature flags in `config.ts` govern:

- `adminMode`
- `answerParagraphs`
- `structuredCitationMatching`
- `citationMetadataDisplay`
- `preserveSubsectionBoundaries`

`isAdminMode()` also supports a `?admin=true` URL override, which gates developer-facing controls such as settings and thought-process visibility.

### 7. Retrieval behavior and prompt awareness

Backend customization flags in `app/backend/customizations/config.py` currently include:

- `category_filter`
- `legal_domain_prompts`
- `citation_sanitizer`
- `custom_evals`
- `enhanced_feedback`
- `agentic_force_query_on_empty`
- `agentic_fallback_search`
- `adaptive_search_retry`
- `agentic_retry_on_weak_matches`

In addition, `app/backend/app.py` now loads available search-index sources at startup and passes them into `ChatReadRetrieveReadApproach`, so prompts can reason about the current corpus instead of using a hard-coded source list.

## Upgrade checklist

When updating from upstream `main`, review these areas in order:

1. Reapply backend route imports and blueprint registration for `categories_bp` and `feedback_bp` in `app/backend/app.py`.
2. Recheck the `fetch_available_sources()` startup flow in `app/backend/app.py`.
3. Reconcile legal retrieval and citation metadata hooks in `app/backend/approaches/approach.py` and `app/backend/approaches/chatreadretrieveread.py`.
4. Reapply frontend imports from `app/frontend/src/customizations/` in `Chat.tsx`, `Layout.tsx`, `Answer.tsx`, `AnswerParser.tsx`, `AnalysisPanel.tsx`, and `SupportingContent.tsx`.
5. Recheck `QuestionInput.tsx` support for `leftOfSend`, which the custom chat controls depend on.
6. Reapply Vite dev-server proxies for `/api/categories` and `/api/feedback`.
7. Review prompt files under `app/backend/approaches/prompts/` manually.

Also remove any outdated assumptions copied forward from earlier docs. In particular, this fork no longer relies on:

- `ask/Ask.tsx` as a live customization target
- `CategoryDropdown/` or `SearchBoxWithCategories/` custom folders
- `prompt_extensions.py` in backend customizations
- `thought_filter.py` as the documented feedback security mechanism

## Testing references

Relevant current tests include:

- Frontend customization tests under `app/frontend/src/customizations/__tests__/`
- Chat/supporting-content behavior tests under `tests/`
- Additional fork-specific documentation in `docs/customizations/`

Useful companion docs:

- `docs/customizations/CITATION_SYSTEM.md`
- `docs/customizations/FEEDBACK_SYSTEM.md`
- `docs/customizations/SYSTEM_ARCHITECTURE.md`
- `docs/customizations/RUNNING_TESTS.md`

## Maintenance rule

If a fork-specific feature is added outside `/customizations/`, update this guide and `.github/copilot-instructions.md` in the same change. That is the only reliable way to keep future upstream merges sane.
