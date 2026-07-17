# Legal Index Completeness Problem Prompt

## Focused Planner-Agent Fix Prompt

Use this prompt for the planner agent to resolve the current implementation blocker:

```text
You are the planner agent for the legal Azure AI Search v4 release pipeline in
this repository. Fix the source-fidelity audit workflow at its root cause.

## Current failure

The candidate snapshot is:

   reports/search_snapshot_20260715_r4.json

The command below consumes a full CPU core for many minutes and is terminated
before it writes a new report:

   source .venv/bin/activate && python scripts/audit_source_documents.py \
    --index-snapshot reports/search_snapshot_20260715_r4.json \
    --html-fidelity \
    --json-output reports/source_document_accuracy.json \
    --markdown-output reports/source_document_accuracy.md

The existing report is stale and must not be treated as evidence. A PDF-only
run also failed to complete within the available execution window. Determine
whether the controlling cause is unbounded source parsing, repeated expensive
matching, duplicate canonical-source expansion, network/cache behavior, or
another local bottleneck. Instrument the audit sufficiently to identify the
slow phase and source, then fix the controlling code path.

## Non-negotiable safety rules

1. Do not raise fidelity thresholds.
2. Do not convert FAIL, UNKNOWN, UNAVAILABLE, AMBIGUOUS, INDEX_ONLY, or
  MISSING_FROM_INDEX findings into PASS.
3. Do not skip canonical sources or quietly add exclusions. Any exclusion
  requires a versioned, reviewed policy and explicit evidence.
4. Do not use the stale July 12 report as fresh r4 evidence.
5. Do not mutate production v3, promote the candidate, set approved=true, or
  write to Azure Search as part of this task.
6. Preserve fail-closed behavior: a timeout, interrupted source, missing
  output, incomplete source list, or unresolved classification must fail the
  audit and evidence gate.
7. Keep aggregate n-gram and page coverage diagnostic-only. Substantive
  occurrence/block matching remains authoritative.

## Required investigation

Start from these files and their focused tests:

- scripts/audit_source_documents.py
- scripts/build_v4_evidence_bundle.py
- tests/test_audit_source_documents.py
- tests/test_promote_v4_candidate.py
- reports/search_snapshot_20260715_r4.json

Trace the complete path from snapshot loading through canonical manifest
loading, reconciliation, PDF/HTML extraction, matching, and report writing.
Do not perform broad repository refactoring. Identify one falsifiable root-
cause hypothesis and a cheap discriminating check before editing.

The audit must:

- have bounded work per source and per matching operation;
- avoid recomputing normalized text, token indexes, or similarity structures;
- use a persistent HTTP cache for HTML requests and deterministic cache keys;
- report progress or checkpoints identifying the current phase and source;
- write reports atomically only after complete reconciliation;
- return a non-zero exit code when a run is incomplete or interrupted;
- make timeout and retry behavior explicit and testable;
- preserve exact source identity and category matching;
- retain source-level evidence, substantive block counts, and explicit
  remediation dispositions in the report.

If the complete audit cannot finish locally because an external canonical site
is unavailable, prove that with a bounded, reproducible result and classify the
affected sources as UNAVAILABLE or another accurate non-PASS disposition. Do
not claim coverage from a partial run.

## Required implementation deliverables

1. Fix the controlling performance or lifecycle defect in the smallest
  coherent change.
2. Add focused regression tests for the discovered defect, including timeout,
  interruption, cache, or duplicate-work behavior as applicable.
3. Ensure the evidence builder rejects stale, partial, missing, non-unique, or
  source-incomplete reports.
4. Add concise operator documentation describing how to run the audit, resume
  from checkpoints, identify the slow source, and distinguish fresh evidence
  from stale output.

## Validation

### Current implementation notes

The audit writes the final JSON and Markdown reports atomically. Use a separate
checkpoint path for each run, for example:

```shell
source .venv/bin/activate
python scripts/audit_source_documents.py \
  --index-snapshot reports/search_snapshot_20260715_r4.json \
  --family pdf \
  --source "Chancery Guide" \
  --checkpoint /tmp/index-v4-r4-checkpoint.json \
  --json-output /tmp/index-v4-r4.json \
  --markdown-output /tmp/index-v4-r4.md
```

The checkpoint is an operational progress marker for reconciliation and PDF or
HTML fidelity phases. It is written atomically and includes the run ID, Search
snapshot provenance, source identity digest, phase, and processed count. The
HTML response cache is a separate per-URL cache and may be kept between runs.
Neither cache nor checkpoint is release evidence: only a report with
`complete: true`, fresh UTC timestamps, a verified snapshot envelope, complete
source-level results, occurrence ledgers, and explicit remediation dispositions
can be consumed by the evidence builder. A process interrupted before final
publication leaves the prior report untouched and cannot produce promotable
evidence.

The authoritative gate uses substantive occurrence/block counts and the
occurrence ledger. Aggregate n-gram and page coverage remain diagnostic and
cannot clear a missing, ambiguous, unavailable, duplicate, or unresolved
source. The current matcher avoids document-level rescans for unique whole-index
matches while retaining document scoping for duplicate occurrences.

Run:

   source .venv/bin/activate
   pytest -q tests/test_audit_source_documents.py \
    tests/test_promote_v4_candidate.py \
    tests/test_run_v4_application_gates.py
   python -m py_compile scripts/audit_source_documents.py \
    scripts/build_v4_evidence_bundle.py
   git diff --check

Then run a bounded smoke audit against the r4 snapshot and a fresh full audit
using a persistent HTML cache. Verify all of the following:

- the output modification time changes only after a completed run;
- the report records the snapshot identity and generation timestamp;
- source count matches the canonical reconciliation set;
- every source has an explicit status and remediation disposition;
- no partial report is mistaken for final evidence;
- substantive block metrics, not diagnostic n-grams, control fidelity;
- the evidence builder accepts only complete, fresh source-level evidence.

Report the exact root cause, files changed, tests and commands run, elapsed
time/phase metrics, fresh report summary, unresolved source identities, and
whether the release is BLOCKED or READY FOR HUMAN APPROVAL. Given the current
state, do not report READY unless a fresh complete audit and every other
release gate pass. Never claim that stale July 12 counts are current.
```

Use the following prompt when asking an engineer or coding agent to complete the
legal index v4 release methodology.

```text
You are working on a legal-domain fork of the Azure Search OpenAI Demo. The
application indexes canonical UK legal sources, including Civil Procedure Rules,
Practice Directions, court guides, forms, appendices, tables, footnotes, and
other operative HTML and PDF content.

## Problem

We need a fail-closed, repeatable release process that proves the actual legal
content from every canonical source is present in the candidate Azure AI Search
index and usable by the application. A high aggregate coverage percentage is
not sufficient: it can hide missing paragraphs, ambiguous matches, duplicate
source identities, stale snapshots, parser loss, or content that exists in the
artifact but was never uploaded or cannot be retrieved by the application.

The process must prove the complete chain:

canonical source bytes -> extraction -> occurrence ledger -> chunk artifact ->
embeddings -> candidate Search index -> verified Search snapshot -> application
retrieval -> citations/supporting content -> source filters and ACL behavior.

Every result must be tied to the same immutable release and candidate targets.
The process must never silently fall back to the current production v3 index,
a stale local snapshot, or an unavailable application.

## Required contract

For every canonical source, recapture the source during the release and assign
each potentially operative occurrence a stable identity containing:

- canonical source identity;
- structural locator such as page, heading, rule, paragraph, or subsection;
- occurrence ordinal for repeated text;
- normalized-content hash;
- extraction and matching status.

Include operative paragraphs, rules, subrules, headings, list items, tables,
forms, checklists, footnotes, appendices, and other legal content by default.
Repeated identical text at different locations must remain separate occurrences.
No normalized-hash deduplication may erase a legal occurrence.

Each occurrence must be proven present in both the generated artifact and the
candidate Search index, or be explicitly excluded by a versioned and reviewed
policy with a recorded reason. The release must fail when an occurrence is
unknown, unavailable, parse-failed, unclassified, unmatched, ambiguous,
index-only, excluded without approval, or absent from the final candidate.

Aggregate n-gram coverage, page coverage, overlap with another source, and
similarity scores are diagnostic only. They cannot clear a release blocker.

## Existing implementation to preserve and complete

The repository already contains or is expected to contain:

- occurrence-level source auditing and an occurrence ledger;
- fresh HTML and PDF oracle capture;
- an HTML transition audit;
- immutable v4 staging artifact generation;
- 3,072-dimensional `text-embedding-3-large` embeddings;
- verified Search snapshot envelopes with provenance and hashes;
- exact selected-field artifact/Search equality checks;
- strict candidate document validation, including subsection/content checks;
- evidence-bundle construction;
- promotion checks requiring candidate validation `status == PASS`;
- v3 index and knowledge-base rollback targets.

Preserve these fail-closed behaviors. Do not weaken them to make a partial run
pass. Keep fork-specific logic in `customizations/` where practical, preserve
`CUSTOM:` integration markers, and add focused tests for every new gate.

## Remaining work

Complete the application-level release gates. They must run against the exact
candidate v4 Search index, paired knowledge base, and candidate application
deployment or test target, not against production v3 by accident. The gates
must explicitly and independently verify:

1. Retrieval accuracy for representative queries across every source family.
2. Expected source category and subsection/rule retrieval.
3. Citation rendering and citation click behavior.
4. Supporting-content opening and subsection highlighting boundaries.
5. Source/category filtering, including the `include_category` request override.
6. ACL behavior and rejection of unauthorized source access.
7. Source hierarchy and precedence behavior.
8. Candidate application availability and correct index/knowledge-base
   configuration.

Create one strict machine-readable application-gate report or aggregator. It
must reject skipped, unavailable, partial, diagnostic-only, malformed, or
unproven results. Every required gate must have an explicit `PASS` result and
candidate provenance. A missing candidate app is a failure, not a skip.

Wire that report into the evidence bundle and promotion workflow. Promotion
must require all of the following:

- clean HTML/PDF source capture and transition audit;
- clean occurrence-level fidelity audit;
- exact artifact/Search equality;
- candidate structural validation with matching snapshot provenance;
- clean application-level gate report for the same candidate;
- explicit GitHub `Production` environment approval;
- paired v4 index and knowledge base;
- unchanged v3 rollback pair.

## Known blockers

Do not claim the release is ready while any of these remain unresolved:

- Practice Direction 27B currently has same-source ambiguous occurrences and
  unmatched form clauses;
- canonical URL/source manifest drift and duplicate source families remain;
- no fresh r3 Search upload, verified r3 snapshot, complete fidelity audit, and
  candidate-targeted application run has yet cleared the release;
- application retrieval, citation, category, ACL, and source-hierarchy checks
  are not valid if they silently target v3, stale data, or an unavailable app.

## Deliverables

Implement the smallest coherent set of code, workflow, documentation, and
focused tests needed to satisfy the contract. Then run the relevant tests,
Python compilation, workflow YAML parsing, and whitespace checks.

Report:

1. Files changed and the behavior each change controls.
2. Tests and static checks run, with their results.
3. The exact candidate provenance used by each gate.
4. Any remaining blockers, with counts and source identities.
5. Whether promotion is `READY FOR HUMAN APPROVAL` or
   `BLOCKED: DO NOT PROMOTE`.

Never report production readiness from local unit tests alone. Never perform a
production Search write or application cutover without the complete evidence
bundle and explicit Production approval.
```