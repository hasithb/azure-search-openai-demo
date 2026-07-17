# Handoff: Load the live primary source in-panel and highlight the cited passage

Audience: the implementing agent (sonnet 4.66). This is an implementation-ready plan. No code has been written yet.

## Goal

In the **Primary Source** tab, for citations whose primary source is a live external
web page (Civil Procedure Rules on `justice.gov.uk`, statutes on `legislation.gov.uk`),
**load the actual live website inside the panel and auto-highlight + scroll to the cited
passage.** Today these citations hit a dead-end fallback ("This source can't be embedded")
because the pages block iframing.

## Why the iframe is blocked (root cause)

`justice.gov.uk` / `legislation.gov.uk` send `X-Frame-Options: SAMEORIGIN` and/or
`Content-Security-Policy: frame-ancestors ...`. Browsers refuse to render them inside our
iframe **because the response comes from a different origin**. You cannot un-block an iframe
from the client. The only way a browser will embed the page is if the bytes are served from
**our own origin**. Therefore we need a **same-origin server-side reverse proxy**.

## Architecture (3 parts + a degradation chain)

### Part 1 — Backend reverse-proxy route (new, merge-safe)

Create `app/backend/customizations/routes/proxy_source.py` exposing:

```
GET /api/proxy-source?url=<encoded live url>&q=<encoded passage phrase>
```

Register the blueprint in `app/backend/app.py` next to the existing customization
blueprints (`categories_bp`, `feedback_bp`), and export it from
`app/backend/customizations/routes/__init__.py`.

Route logic, in order:

1. **Allowlist guard (SECURITY — do this first, fail closed).**
   - Parse `url` with `urllib.parse`. Require `https`.
   - Hostname must match the SAME allowlist used by the frontend `isIframeBlocked`
     (`justice.gov.uk`, `www.justice.gov.uk`, `legislation.gov.uk`,
     `www.legislation.gov.uk`). Keep this list in ONE place server-side; consider also
     allowing the other official UK legal domains already referenced in `AGENTS.md`
     (gov.uk, judiciary.gov.uk, bailii.org) ONLY if a real citation needs them — start
     minimal.
   - Reject anything else with `abort(400)`. Reject private/loopback/link-local IPs
     (resolve and check) to prevent SSRF. Reject non-allowlisted redirects (see step 2).
2. **Fetch server-side** with `aiohttp` (already a backend dependency; confirm in
   `app/backend/requirements.txt`). Use:
   - a short connect/read timeout (e.g. 8s total),
   - a normal browser `User-Agent`,
   - `allow_redirects=True` BUT re-validate the final URL host against the allowlist
     after redirects (or follow manually and re-check each hop),
   - a response size cap (e.g. abort if `Content-Length` or streamed bytes exceed ~5 MB),
   - **do NOT forward user cookies or auth headers** (these pages are public).
3. **Strip framing protections from OUR response** (not the upstream fetch):
   - Do not forward `X-Frame-Options`.
   - Replace any upstream CSP with our own: `Content-Security-Policy: frame-ancestors 'self'`.
4. **Rewrite the HTML** so the framed page works from our origin without proxying assets:
   - Inject `<base href="<origin of the fetched url>">` right after `<head>` so the page's
     own relative CSS/JS/images load directly from the real origin (we only proxy the HTML
     document, not its sub-resources).
   - Inject, just before `</body>`, a highlight script + the target phrase:
     ```html
     <script>window.__HIGHLIGHT_PHRASE__ = "<js-escaped q value>";</script>
     <script>/* highlight routine, see below */</script>
     ```
   - The highlight routine: normalize whitespace, walk text nodes (TreeWalker), find the
     first node whose normalized text contains the normalized phrase, wrap the match in
     `<mark style="background:#fde68a;scroll-margin:120px">`, then
     `mark.scrollIntoView({block:'center'})`. Wrap in `try/catch`; do nothing on failure.
   - **Neutralize frame-busters**: before returning, remove/disable inline scripts containing
     `top.location`, `self !== top`, `top != self`, `framebuster`, `frameBreaker`. Most UK
     gov pages don't frame-bust, but guard anyway.
5. Return `quart.Response(rewritten_html, content_type="text/html")` with the CSP header set.
6. **Optional**: tiny in-memory TTL cache keyed by `url` (e.g. 5 min) to avoid re-fetching on
   every open. Not required for correctness.

Security review checklist (call this out in the PR):
- [ ] Strict hostname allowlist, https-only, redirects re-validated.
- [ ] Private/loopback IP rejection (SSRF).
- [ ] Response size + timeout caps.
- [ ] No user cookies/headers forwarded.
- [ ] Only HTML document proxied; sub-resources go direct via `<base>`.

### Part 2 — Frontend points the iframe at the proxy

In `app/frontend/src/customizations/PrimarySourceViewer.tsx`, replace the current
iframe-blocked dead-end branch (the `externalSource && isIframeBlocked(externalSource)`
block that renders the "open in new tab" message) with a proxied live iframe:

```tsx
const phrase = pickDistinctivePhrase(metadata?.content); // first ~8 distinctive words
const proxySrc = `/api/proxy-source?url=${encodeURIComponent(externalSource)}&q=${encodeURIComponent(phrase)}`;
// render <iframe src={proxySrc} className={styles.frame} style={{ height }} title="Primary source" />
// plus a secondary "Open in new tab" button beneath it (text-fragment deep link).
```

Add two helpers in `app/frontend/src/customizations/externalSourceHandler.ts`:
- `pickDistinctivePhrase(content: string): string` — strip leading bracketed breadcrumb
  prefixes like `[PART 29 – ... > ...]`, take the first ~8 words of the actual passage,
  collapse whitespace. This becomes `q`.
- `buildTextFragmentUrl(url: string, content: string): string` — append
  `#:~:text=<encoded phrase>` for the secondary "Open in new tab" button so the real browser
  tab also scrolls+highlights (Scroll-To-Text-Fragment, supported in Chromium/Edge).

Keep the existing React #185 fix intact: do NOT add `onVerified` to any effect deps; keep the
`onVerifiedRef` pattern. Call `onVerifiedRef.current?.("none")` for proxied external sources
(we can't programmatically confirm the highlight landed across the iframe boundary — the
upstream page is same-origin-from-our-perspective only for framing, but the proxied document's
`<base>` points at the real origin, so cross-frame DOM reads are still blocked; treat as "none").

### Part 3 — Degradation chain (must never regress)

Wire the proxied iframe with a `load` listener + a fallback timer (e.g. 8s). If the iframe
errors or never loads:
1. Fall back to the **reader view** of `metadata.fullContent` with the cited subsection
   highlighted (see the separate "reader view" plan below — thread `full_content` through
   metadata and reuse `extractSubsectionContent` + `resolveTargetSubsection`).
2. If `fullContent` is empty, fall back to today's passage banner.

This guarantees the panel always shows the passage even if the live host is down or changes
its markup.

### Reader-view fallback prerequisites (for Part 3 and for court guides with no live URL)

- Add `fullContent: string` to `StructuredCitationMetadata` in
  `app/frontend/src/customizations/citationMetadata.ts` and populate from `dp.full_content`
  in `extractMetadataFromDataPoint`.
- Reuse `extractSubsectionContent(fullContent, target)` from
  `app/frontend/src/components/SupportingContent/SupportingContentParser.ts` and
  `resolveTargetSubsection(...)` from `SupportingContent.tsx` to compute highlight bounds —
  same logic that already powers the Supporting Content tab, so boundaries match.

## Per-branch result

| Branch | Behavior |
|--------|----------|
| CPR (`justice.gov.uk`) | Live page renders in-panel via proxy; cited passage highlighted + scrolled |
| `legislation.gov.uk` | Same — live page, highlighted |
| Court guides (no live URL, local dev) | Reader-view highlight of `full_content` (no live page exists) |
| In-app PDFs (prod court guides) | Unchanged PDF.js text-layer highlight |

## Dev wiring

- Add `"/api/proxy-source": "http://localhost:50505"` to the proxy block in
  `app/frontend/vite.config.ts` so `npm run dev` reaches the backend.
- Backend is served at `http://localhost:50505`; rebuild frontend
  (`cd app/frontend && npm run build`) before browser validation since the app serves the
  built static files.

## Verification plan

1. Backend unit test for the allowlist guard: assert `abort(400)` for non-allowlisted hosts,
   http (non-https), private IPs, and post-redirect host changes. Put it under `tests/`.
2. Manual/e2e: ask a CPR question, click a citation → Show supporting content →
   Show in primary source. Confirm the live justice.gov.uk page renders in the panel with the
   cited subsection highlighted and scrolled into view, no React error #185, no 404.
3. Confirm the "Open in new tab" secondary button lands on the passage (text fragment).
4. Confirm court-guide citations (no live URL) fall back to the reader-view highlight.

## Tradeoffs / constraints to accept

- Security is the #1 risk: the route is an outbound fetcher → strict allowlist + SSRF guards
  are mandatory, or it becomes an open proxy.
- Bandwidth/latency: each open re-fetches server-side; add the optional TTL cache if needed.
- Markup fragility: gov.uk HTML changes can break the highlight → reader-view fallback covers it.
- Licensing: justice.gov.uk and legislation.gov.uk are Crown copyright under the Open
  Government Licence (permits reproduction/re-use). Keep the allowlist to official OGL sources.

## Files touched

- New: `app/backend/customizations/routes/proxy_source.py` (+ export in `routes/__init__.py`,
  register in `app/backend/app.py`).
- Edit: `app/frontend/src/customizations/PrimarySourceViewer.tsx` (proxy iframe + fallbacks).
- Edit: `app/frontend/src/customizations/externalSourceHandler.ts`
  (`pickDistinctivePhrase`, `buildTextFragmentUrl`; reuse the existing allowlist).
- Edit: `app/frontend/src/customizations/citationMetadata.ts` (add `fullContent`).
- Edit: `app/frontend/vite.config.ts` (dev proxy entry).
- Tests: `tests/` (allowlist guard) and an e2e/browser smoke for the live highlight.

The feature is flag-gated (`primarySourceTab` in
`app/frontend/src/customizations/config.ts`); base rollback branch is
`upgrade/upstream-sync-2026-04-30`.
