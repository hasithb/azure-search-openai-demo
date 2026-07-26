# Legal Index Completeness Problem Prompt

## Focused Planner-Agent Fix Prompt

Use this prompt for the planner agent to resolve the current implementation blocker:

````text
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

````

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

## Current Release Handoff Prompt

Use this prompt for the planner agent to continue the release from the current
staging-only workflow state:

```text
You are the planner agent responsible for completing the legal Azure AI Search
v4 release in this repository. Continue from the exact current state below.

## Objective

Advance the release through all staging validation phases and stop only at:

   READY FOR HUMAN APPROVAL

The release must remain fail-closed. Do not perform production Search writes,
production configuration changes, cutover, `approved=true`, or promotion.

## Current state

- Repository: `/Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2`
- Branch: `feature/primary-source-validation`
- Latest fix commit: `3018bc75`
- Staging workflow: `.github/workflows/update-index-v4.yml`
- Latest dispatched release: `20260718-r21`
- Latest workflow run: `29646775945`
- Dispatch input: `promote=false`
- Verified embedding service:
  `cog-gz2m4s637t5me`
- Verified embedding endpoint:
  `https://cog-gz2m4s637t5me.openai.azure.com/`
- Verified embedding deployment: `text-embedding-3-large`
- Required embedding dimensions: `3072`
- OIDC principal object ID:
  `6743c9fc-61d0-44df-9bfe-a63f9d07a9a3`
- Required role: `Cognitive Services OpenAI User`
- Required data action:
  `Microsoft.CognitiveServices/accounts/OpenAI/deployments/embeddings/action`

The root cause of the earlier embedding 401 was configuration mismatch: the
repository service secret targeted a different Azure OpenAI account than the
account carrying the exact-scope role assignment. Commit `3018bc75` makes the
staging embedding step use the verified service explicitly. The deployment
secret remains `text-embedding-3-large`. Do not restore the incorrect service
secret dependency or broaden RBAC.

The r21 run was accepted and preflight completed successfully. Its candidate
job was still running when this handoff was prepared. Do not dispatch a second
run until run `29646775945` has a terminal result.

The older run `29643610606` (`20260718-r12`, commit `2b22c89b`) is stale failure
evidence. That revision passed an empty `AZURE_OPENAI_ENDPOINT` to
`generate_v4_embeddings.py` and failed before making an embedding request.
Do not infer an Azure OpenAI RBAC failure from that run, and do not use its
candidate environment values as evidence for r21. The current branch commit
`3018bc75` supplies the verified endpoint explicitly with `--endpoint`.

## First actions: establish facts

1. Inspect run `29646775945` through the GitHub Actions API or a non-interactive
  equivalent. Do not rely on a stale local log or the interactive `gh run
  watch` renderer.
2. Confirm the run commit is `3018bc75`, promotion is disabled, and no
  production job has started.
3. Inspect the candidate job steps. Confirm the managed-identity embedding
  step completed successfully before diagnosing later phases.
4. If r21 is still running, monitor it without dispatching another run.
5. If r21 failed, capture the exact failed step and error first. Fix only the
  controlling defect, then rerun with a new release ID and `promote=false`.
6. If r21 succeeded, use its generated release ID, artifact, snapshot, and
  provenance as the only inputs to downstream validation.

When inspecting logs, always record the run ID, head SHA, release ID, and
embedding command arguments together. A failure from an older SHA is not a
failure of the current workflow, and a missing endpoint is a configuration
failure distinct from a 401/403 authorization failure.

Never treat a queued, cancelled, interrupted, partial, or stale report as
evidence. A workflow failure is a release blocker, not permission to skip a
phase.

## Required staging sequence

After embeddings pass, verify that the same release proceeds through every
phase below:

1. Candidate document validation, including non-empty 3072-dimensional vectors,
  selected-field equality, source identity, category, subsection/content,
  and duplicate-occurrence checks.
2. Immutable artifact hashing and artifact upload.
3. Staging Azure AI Search index provisioning and upload only. Confirm the
  target is the v4 staging index, never `legal-court-rag-index-v3`.
4. Verified Search snapshot capture. Confirm the snapshot envelope contains
  the exact candidate service, index, knowledge base, release ID, Git SHA,
  artifact hash, and snapshot hash.
5. Fresh occurrence-level HTML/PDF fidelity audit and transition audit. Any
  `FAIL`, `UNKNOWN`, `UNAVAILABLE`, `AMBIGUOUS`, `INDEX_ONLY`,
  `MISSING_FROM_INDEX`, incomplete, duplicate, or unresolved finding blocks
  the release.
6. Candidate application deployment or candidate test target. Verify its
  runtime configuration points to the paired v4 index and knowledge base.
7. Run all required application gates against that candidate target:
  `retrieval`, `category`, `source_hierarchy`, `citation`, `acl`, and
  `highlight`.
8. Build the evidence bundle only after all preceding reports are complete and
  provenance-compatible.

Each application gate must be an explicit machine-readable `PASS` with the
same candidate provenance. A missing, unavailable, skipped, diagnostic-only,
malformed, partial, or production-targeted gate is a failure.

## Provenance contract

Before reporting readiness, collect and compare these fields across the source
audit, artifact validation, Search snapshot, candidate validation, application
gate report, and evidence bundle:

- `release_id`
- `git_sha`
- `deployment_id`
- `artifact_sha256`
- `search_snapshot_sha256`
- `search_service`
- `search_index`
- `knowledge_base`

Reject any missing, duplicated, stale, mismatched, or locally fabricated
provenance. The v3 Search index may remain only as the documented rollback
target; it must not be used as the candidate or as an application fallback.

## Handling failures

- Do not repeat an unchanged workflow run after an authentication or resource
  targeting failure; identify the exact target and principal first.
- Do not fix a failed fidelity audit by raising thresholds, deduplicating legal
  occurrences, adding silent exclusions, or converting statuses to `PASS`.
- Do not approve unresolved Practice Direction 27B ambiguity or unmatched form
  clauses. Preserve source identities and remediation dispositions.
- If an external canonical source is unavailable, produce a bounded,
  reproducible non-PASS result with the source identity and classification.
- Keep temporary diagnostic logging only until the target is verified; remove
  diagnostics that are no longer needed without weakening the workflow gate.
- Do not modify production resources or invoke promotion while investigating.

## Validation commands

Run focused checks before and after any code change:

   source .venv/bin/activate
   pytest -q tests/test_audit_source_documents.py \
    tests/test_promote_v4_candidate.py \
    tests/test_run_v4_application_gates.py
   python -m py_compile scripts/audit_source_documents.py \
    scripts/build_v4_evidence_bundle.py
   git diff --check

For workflow-only changes, additionally validate YAML syntax using the
repository's available YAML checker. Prefer API-based GitHub run inspection
when the terminal's interactive alternate buffer hides `gh` output.

## Final report

Report:

1. The exact run IDs, commit, release ID, and candidate targets used.
2. The embedding result and the exact failed step for any unsuccessful retry.
3. Every required gate and its machine-readable status.
4. The complete provenance comparison.
5. Unresolved source identities, statuses, remediation dispositions, and
  counts.
6. Tests and validation commands with results.
7. One of exactly these conclusions:

     READY FOR HUMAN APPROVAL

  or

     BLOCKED: DO NOT PROMOTE

Only use `READY FOR HUMAN APPROVAL` when the fresh source audit, artifact and
Search checks, candidate validation, all six application gates, and evidence
bundle pass for the same candidate. Human approval and promotion are outside
this task and must remain unperformed.
```

## Current End-to-End Test-and-Fix Planner Prompt

Use this prompt when the goal is to test every v4 staging gate, repair any
failure found, and rerun until the candidate either passes completely or is
blocked by a genuine unresolved source or infrastructure issue.

````text
You are the release planner and coding agent for the legal Azure AI Search v4
workflow in this repository. Your job is to test the complete staging release,
diagnose failures from evidence, implement root-cause fixes, and rerun the
affected checks until every required gate has an explicit PASS or the release
is accurately BLOCKED.

## Non-negotiable boundary

This is a staging-only exercise. Never promote, cut over, approve, or mutate
production. Every workflow dispatch must use `promote=false`. Do not write to,
replace, or reconfigure `legal-court-rag-index-v3` or its paired knowledge
base. Preserve v3 as the rollback target. Do not weaken a gate, raise a
threshold, add a silent exclusion, accept a skipped test, or treat partial or
stale output as evidence.

## Known release context

- Repository: `/Users/HasithB/Downloads/PROJECTS/azure-search-openai-demo-2`
- Branch: `feature/primary-source-validation`
- Workflow: `.github/workflows/update-index-v4.yml`
- Latest known failed release: `20260718-r17`
- Latest known workflow run: `29656368165`
- Latest known failing commit: `b8c37c8bb19d9280a02e4a84e44925b05d3feca2`
- Failed step: browser highlight gate
- Candidate app:
  `https://capps-v4-candidate-20260717-r5.blackflower-4aba1fd4.uksouth.azurecontainerapps.io`
- Candidate index: `legal-court-rag-v4-staging-20260718-r17`
- Candidate knowledge base:
  `legal-court-rag-v4-staging-20260718-r17-agent-upgrade`
- Rollback index: `legal-court-rag-index-v3`
- Rollback knowledge base: `legal-court-rag-index-v3-agent-upgrade`
- Embedding model: `text-embedding-3-large`, exactly 3072 dimensions

Treat these values as starting context, not assumed truth. First inspect the
actual run, commit, release ID, candidate URL, index, knowledge base, report
envelopes, and workflow inputs using non-interactive GitHub/API commands. If a
newer terminal run exists, use its exact provenance instead. Never dispatch a
replacement while the current run is still active.

## Required operating method

Before each edit:

1. Identify the smallest owning code path and state one falsifiable root-cause
   hypothesis.
2. Name one cheap check that could disconfirm it.
3. Inspect the nearest implementation and focused test.
4. Make the smallest coherent fix, preserving fail-closed behavior.
5. Immediately run the narrowest executable validation for the changed slice.

Do not patch a symptom merely to make a report pass. Record the failed gate,
its exact evidence, the root cause, files changed, validation result, and the
new candidate provenance before moving on.

## Gate sequence to execute completely

For a fresh staging release, verify these gates in order. A failed gate blocks
the release, but after repairing it you must rerun that gate and every later
gate because its outputs may have changed.

1. Preflight and configuration: verify OIDC/managed identity, exact Azure
   OpenAI embedding endpoint and deployment, 3072 dimensions, permissions,
   source manifest, and `promote=false`.
2. Artifact generation: verify source identity, category, subsection metadata,
   occurrence preservation, non-empty embeddings, vector dimensions, hashes,
   and deterministic manifest contents.
3. Staging Search upload: provision and populate only the immutable v4 staging
   index. Verify selected-field equality and reject duplicate or missing
   documents.
4. Search snapshot: capture a verified snapshot envelope containing release ID,
   Git SHA, artifact hash, Search service, index, knowledge base, and snapshot
   hash.
5. HTML/PDF oracle and transition audit: recapture fresh canonical sources.
   Reject mixed oracle versions, stale snapshots, parser loss, unavailable or
   ambiguous sources, unmatched substantive blocks, duplicate occurrences,
   `INDEX_ONLY`, `MISSING_FROM_INDEX`, `UNKNOWN`, and incomplete output.
6. Candidate deployment: verify the live candidate app is available and its
   runtime configuration points to the exact paired staging index and
   knowledge base, never v3 or an implicit fallback.
7. Application gates: run every required gate independently and retain its
   machine-readable report:
   - retrieval accuracy across CPR, Practice Directions, court guides, forms,
     appendices, tables, and other source families;
   - category/source filtering and the `include_category` request override;
   - source hierarchy and precedence behavior;
   - citation rendering, citation metadata, and citation click behavior;
   - supporting-content opening and subsection highlight boundaries;
   - ACL authorization and rejection of unauthorized source access.
8. Strict application aggregation: run
   `scripts/run_v4_application_gates.py` and reject missing, skipped,
   unavailable, malformed, partial, diagnostic-only, or provenance-mismatched
   gate reports.
9. Evidence bundle: run the evidence builder only after all preceding reports
   are complete and mutually bound to the same candidate.
10. Promotion guard: run promotion validation in dry-run or validation mode
    only. Confirm it would reject any missing gate and that no production job,
    Search write, or cutover occurred.

## Browser-highlight investigation

The known r17 failure is:

`No live citation matched subsection "PART 24" or source page "PART 24 - SUMMARY JUDGMENT"`

Diagnose this with evidence before changing matching. Capture all visible
`.supContainer` metadata after the candidate answer is generated, including
`data-subsection-id`, `data-sourcepage`, `data-sourcefile`,
`data-citation-path`, `data-category`, title, and citation text. Compare it to
the selected oracle case and the candidate Search document. A safe match may
use an exact canonical sourcefile/path plus a normalized subsection or source
page, including a nested subsection such as `24.2`, but must not accept any
arbitrary citation. The gate must still prove that the clicked citation maps to
the canonical source and that highlighted content has the expected body hash,
length, boundaries, and subsection identity. Add a focused test for any
extracted matcher or selector helper.

## Failure repair rules

- Authentication or authorization failures: verify endpoint, deployment,
  principal, scope, and role before changing code; never broaden permissions
  or hardcode credentials.
- Embedding failures: preserve bounded concurrency, single tokenization,
  retry/backoff, append-only checkpoints, and no restart-after-100-documents
  behavior. Validate dimensions and request payloads.
- Oracle or fidelity failures: repair source capture, parsing, identity, or
  matching. Never lower substantive fidelity requirements.
- Candidate targeting failures: repair provenance/configuration and redeploy
  the candidate; do not fall back to v3.
- Application failures: inspect the live request, response, rendered metadata,
  and gate fixture before changing the gate. Keep the test deterministic and
  candidate-bound.
- External source outages: produce a bounded, reproducible non-PASS result
  classified as `UNAVAILABLE` or another accurate remediation status. Do not
  call the release complete.

## Required implementation and validation

Add or update only focused code, tests, workflow wiring, and documentation
needed to fix observed defects. Preserve existing `CUSTOM:` markers and
fail-closed checks. At minimum, run the focused tests for every touched area,
then:

```shell
source .venv/bin/activate
python -m py_compile scripts/gate_highlight_browser.py \
  scripts/run_v4_application_gates.py \
  scripts/build_v4_evidence_bundle.py
pytest -q tests/test_validate_highlight_oracle.py \
  tests/test_run_v4_application_gates.py \
  tests/test_application_gate.py \
  tests/test_audit_source_documents.py \
  tests/test_promote_v4_candidate.py
git diff --check
```

For workflow changes, validate YAML syntax and inspect the rendered workflow
steps. Dispatch a new run only after local validation succeeds, always with a
new release ID and `promote=false`. Monitor it through a terminal result,
inspect every job step, download every relevant artifact, and verify that no
later gate was skipped. If the workflow remains fail-fast, explicitly report
all skipped gates as untested rather than passing them.

## Completion contract

Return a concise release record containing:

- run ID, release ID, commit, candidate URL, Search index, and knowledge base;
- every required gate with `PASS`, `FAIL`, or `UNTESTED` and its report path;
- the exact root cause and fix for each failure;
- provenance comparison across artifact, Search snapshot, candidate app,
  application gates, and evidence bundle;
- unresolved source identities and remediation statuses;
- commands and tests run with results.

Use exactly one final conclusion:

`READY FOR HUMAN APPROVAL`

only when every gate above has a fresh explicit PASS for the same candidate,
the evidence bundle validates, the promotion guard passes validation, and v3
remains unchanged. Otherwise use:

`BLOCKED: DO NOT PROMOTE`

Never claim readiness because infrastructure succeeded, because one gate
passed, or because later gates were skipped after an earlier failure.

````
