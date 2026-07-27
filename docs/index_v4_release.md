# Index v4 Release Operations

Index v4 is released separately from the legacy v3 updater. Each run creates
an immutable staging Search index and a paired agentic knowledge base. The
production application is switched only after the complete evidence bundle is
approved by the GitHub `Production` environment.

## Required Production configuration

Configure these GitHub Actions secrets for the repository and `Production`
environment:

- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- `AZURE_SEARCH_SERVICE`
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMB_DEPLOYMENT`
- `AZURE_OPENAI_SERVICE`, `AZURE_OPENAI_KNOWLEDGEBASE_DEPLOYMENT`,
  `AZURE_OPENAI_KNOWLEDGEBASE_MODEL`
- `AZURE_RESOURCE_GROUP`, `AZURE_BACKEND_SERVICE`

Configure these repository variables for the dedicated candidate Container App
used by application validation:

- `V4_CANDIDATE_RESOURCE_GROUP`
- `V4_CANDIDATE_APP`
- `V4_CANDIDATE_URL` (HTTPS URL only; it must not point at localhost)

The workflow derives the release-specific staging Search index and paired
knowledge-base names from the required manual `release_id` input. Do not set
or reuse a shared `V4_SEARCH_INDEX` or `V4_SEARCH_KNOWLEDGEBASE` value: each
run must target its own immutable names, such as
`legal-court-rag-v4-staging-20260715-r4` and
`legal-court-rag-v4-staging-20260715-r4-agent-upgrade`.

The audit job generates retrieval, category, source-hierarchy, citation, ACL,
and highlight reports against the candidate. Each gate file must be a JSON
object with `{"status": "PASS"}`. The strict wrapper
`scripts/run_v4_application_gates.py` rejects missing, duplicate, malformed,
unknown, or non-passing reports and binds the combined report to the release,
deployment, candidate Search targets, artifact hash, and snapshot hash. The
workflow does not substitute diagnostic scripts or skip a report when a
variable is unset.

The workflow also generates `reports/highlight_oracle.json` from the canonical
HTML and PDF snapshots and validates it with
`scripts/validate_highlight_oracle.py`. This is a required `highlight` gate,
not a diagnostic report: it must contain unique section cases covering every
canonical source identity, a consistent oracle version, and the SHA-256 of the
snapshot manifest. Each case also contains the normalized canonical body span,
its length, and a case-folded SHA-256 body fingerprint. Validation recomputes
that span from the bound snapshot and rejects changed or missing body evidence;
the heading and next-heading fields alone are not sufficient. Promotion rejects
an application-gate report that omits a passing, non-empty highlight gate.
The evidence artifact also retains `candidate_provenance.json`,
`highlight_gate.json`, and `acl_gate.json` alongside the combined application
gate report.

The eight court-guide artifacts are sourced from the checked-in Azure DI output
directory `scripts/court_guides_processing_pipeline/outputs_azure_di`. The v4
generator accepts `--court-guides-dir` so CI and local release runs use the same
explicit, reproducible input. It fails closed when any configured processed JSON
is missing or empty; it does not fabricate guide content or silently fall back
to an older report directory.

Set the `AZURE_DEPLOYMENT_TARGET` repository/environment variable to either
`appservice` or `containerapps`. The workflow refuses any other value.

The identity used by the workflow needs permission to create and upload the
staging Search resources, invoke the embedding deployment, and update the
application configuration. Prefer OIDC and managed identity over stored
client secrets.

Before building or provisioning any release resources, the workflow runs a
read-only candidate preflight. It requires the dedicated candidate resource
group, app name, and HTTPS origin variables; verifies that the Container App is
successfully provisioned; confirms that `V4_CANDIDATE_URL` matches its ingress
FQDN; and rejects the blocked r4 candidate app. A missing candidate or URL
mismatch stops the run before any immutable Search index or knowledge base is
created.

## Release gates

### Authoritative completeness contract

The release target is normalized canonical legal text plus its legal structure.
For every canonical source, the workflow must recapture the source bytes during
the release and assign every potentially operative occurrence a stable identity
(source identity, structural locator, ordinal, and normalized-content hash).
Every occurrence must then be either:

- present in the generated artifact and the candidate Search index; or
- explicitly excluded by a versioned, reviewed policy with a recorded reason.

All operative paragraphs, rules, subrules, headings, list items, tables,
forms, checklists, footnotes, and appendices are included by default. Automatic
boilerplate or contents filtering is not an approval decision: any exclusion
must be represented in the occurrence ledger and reviewed. Repeated identical
text remains repeated content when it occurs at different structural locators;
deduplication must never erase an occurrence.

The release is blocked when any occurrence is unknown, unavailable, unmatched,
ambiguous within its source, index-only, excluded without approval, or absent
from the final candidate. Aggregate percentages, n-gram coverage, and overlap
with another source are diagnostic metrics only and cannot clear a blocker.

Before artifact generation, the workflow deletes prior oracle snapshots,
recaptures every canonical HTML source with `--refresh`, captures every
canonical PDF, and runs `audit_html_transition.py`. A capture failure or a
transition failure stops the workflow. The resulting snapshots and transition
report are uploaded with the immutable candidate artifact so later gates use
the same release evidence rather than a stale local crawl.

Run `.github/workflows/update-index-v4.yml` manually with an immutable
`release_id`. The workflow must complete all of these gates before promotion:

1. Generate the release-specific source artifact and non-empty 3,072-dimensional
   `text-embedding-3-large` vectors.
2. Create and upload only to the `v4`/`staging` Search target.
3. Capture a verified Search snapshot and run the complete PDF and HTML source
   fidelity audit.
4. Reject every `FAIL`, `WARN`, `INDEX_ONLY`, `MISSING_FROM_INDEX`, and
   `UNAVAILABLE` source result.
5. Build an evidence bundle with exact artifact and snapshot hashes.
6. Require a clean HTML transition report and exact equality of the artifact's
   selected Search fields against the verified candidate snapshot. Repeated
   source occurrences must remain independently represented in the audit.
7. Deploy the dedicated candidate app revision, expose its complete provenance
   through `/api/provenance`, and run retrieval, citation, category, ACL,
   source-hierarchy, and highlight-oracle checks against that revision before
   approving `Production`.
8. Include the passing application-gate and highlight-oracle reports in the evidence bundle and
   revalidate it during the Production approval job.

The bundle also requires `reports/v4_release_index_uniqueness.json`, produced by a read-only
Azure AI Search `/indexes` inventory check before staging creation. The check rejects a reused
release-specific index name and never creates, updates, or deletes Search resources. It also
requires `reports/v4_citation_coverage.json`, generated from
`reports/v4_citation_coverage_input.json`. Every canonical citation must reconcile to exactly
one Search document, rendered citation, click, Supporting Content result, and Primary Source
result using `source revision + source ID + document ID + subsection ID + canonical text SHA-256`.
Missing, duplicate, ambiguous, drifted, or unavailable records block the release.

The candidate revision is configured with `V4_RELEASE_ID`, `GIT_SHA`,
`DEPLOYMENT_ID`, `V4_ARTIFACT_SHA256`, and
`V4_SEARCH_SNAPSHOT_SHA256`. The last value is updated after Search snapshot
capture and before application-gate execution. A stale or incomplete
`/api/provenance` response is a hard failure.

### Snapshot provenance

The fidelity report and every derived diagnostic must use the same
provenance-bearing snapshot captured from the candidate index. The snapshot
must be an envelope containing a non-empty service, index, capture timestamp,
selected-field list, document count, and document hash. Legacy JSON arrays and
envelopes with null provenance are unverified evidence and must not be used to
approve or reject a release.

The audit CLI refuses unverified snapshots by default. Use
`--allow-unverified-snapshot` only for explicitly diagnostic, non-release
analysis.

When an older report disagrees with a newer staging inventory, rerun the
focused source audit against the verified snapshot before changing scraping or
parsing code. A low coverage result from a snapshot containing only one chunk
of a multi-chunk source is evidence-provenance drift, not proof of corpus loss.

The July 13, 2026 verified r2 audit now scopes repeated substantive matches to
individual Search documents. For Practice Direction 27B this reduced the
ambiguous-block count from 82 to 38 and confirmed all 13 expected chunks are
present. The focused audit recorded 1,253 substantive blocks: 1,213 matched,
38 ambiguous within the same `Practice Direction 27B` source family, and 2
unmatched form clauses. The source remains blocked; cross-document overlap is
diagnostic evidence only and does not waive same-source ambiguity or missing
content.

The verified HTML crawl also found two distinct classes of manifest evidence:

- The live Part 48 page removes punctuation from a long slug during the
   justice.gov.uk redirect. The verifier now compares redirect slugs after
   removing punctuation while still rejecting substantive slug changes.
- The former Part 48 and Practice Direction 48 alias URLs return 404 and are
   no longer tried as fallback sources. They were manifest aliases, not parser
   failures.

The crawl reported many duplicate canonical URLs with paired sourcefile names
and duplicate Search document families. These are `MANIFEST_DRIFT` findings,
not permission to relax the ambiguity gate. They must be reconciled to one
canonical source identity and one intended document family before a release can be
approved. The crawl is not a passing release audit while those duplicates, the
PD27B unmatched clauses, or same-source ambiguities remain.

The offline v4 generator now groups web sources by normalized canonical URL
before scraping and chunking. When a short label and a descriptive label point
to the same URL, the descriptive sourcefile is retained; the alias snapshot is
not transformed into a second document family. Missing snapshots for selected
canonical identities remain a hard failure. The regenerated local artifact has
241 canonical source/URL families, 241 source snapshots, 1,857 documents, and
zero collision-hash IDs. This generator correction does not by itself clear the
underlying manifest drift in the live corpus or the remaining fidelity gates.

The local r3 artifact has been regenerated and its generator, audit, and
staging-safety tests pass. A fresh r3 Search upload, verified r3 snapshot,
complete fidelity audit, and agentic retrieval run are still required before
promotion; the available verified snapshot is r2 and contains 1,929 documents.

The candidate index and knowledge base must be promoted as a pair. The v3
index and `legal-court-rag-index-v3-agent-upgrade` knowledge base are retained
and are never deleted by this process.

## Local preflight simulator

### Local application smoke

The local application smoke attaches to an already-running local app by
default. It checks the bounded readiness endpoint, compares `/api/provenance`
with the release provenance file, and then runs the real Playwright citation,
Supporting Content, and subsection-highlight path. It does not enable fixture
chat behavior and it never makes Azure writes.

```shell
source .venv-upgrade/bin/activate
python scripts/preflight_v4_local.py \
   --mode local-smoke \
   --candidate-url http://127.0.0.1:50505 \
   --provenance reports/candidate_provenance.json \
   --oracle reports/highlight_oracle.json \
   --snapshot-dir reports/html_oracle_snapshots \
   --output reports/v4-local-smoke
```

To let the smoke runner own the local process, provide an explicit command:

```shell
python scripts/preflight_v4_local.py \
   --mode local-smoke \
   --candidate-url http://127.0.0.1:50505 \
   --startup-command './app/start.sh' \
   --startup-timeout 90 \
   --provenance reports/candidate_provenance.json \
   --oracle reports/highlight_oracle.json \
   --snapshot-dir reports/html_oracle_snapshots
```

The runner terminates only a process it started itself. Readiness polling is
bounded and reports the last HTTP status or connection error. Browser failures
retain `browser-diagnostics.json`, `browser-final.png`, and
`browser-trace.zip` under `highlight-browser-diagnostics/` when the diagnostics
directory is supplied. The release workflow uploads this directory with
`if: always()` so failures remain inspectable.

Before dispatching a candidate workflow, run the non-mutating simulator against
captured or reconstructed observations:

```shell
source .venv-upgrade/bin/activate
python scripts/preflight_v4_release.py \
   --input tests/fixtures/v4/ready/preflight.json
```

The command must report `"status": "PASS"` and
`"promotion_eligible": false`. It performs no Azure, GitHub, Search, ACR, or
Container App writes and never grants Production approval. The simulator
checks the same fail-closed contracts used by the workflow: HTTPS/FQDN
binding, v4 staging Search pair, snapshot hash, immutable ACR digest, exact
candidate revision, `latestReadyRevisionName`, healthy state, 100% traffic,
Search environment binding, and provenance.

The reconstructed r7 race can be checked explicitly and is expected to fail:

```shell
python scripts/preflight_v4_release.py \
   --input tests/fixtures/v4/r7-reconstructed/preflight.json
```

That fixture captures an empty image output and an `Activating` revision while
the previous revision remains latest-ready. The readiness poller retries only
those transient propagation states within its bounded timeout; immutable
contradictions fail immediately. Do not dispatch a new GitHub Actions run
until the local simulator and focused test suite pass. The first candidate run
after this preflight must use `promote=false`.

## Cutover and rollback

The approved workflow updates both application settings in one deployment API
request:

```text
AZURE_SEARCH_INDEX=<candidate v4 index>
AZURE_SEARCH_KNOWLEDGEBASE_NAME=<candidate v4 knowledge base>
```

For App Service this uses `az webapp config appsettings set`; for Container
Apps it uses `az containerapp update --set-env-vars`. The target resource and
resource group are explicit Production environment values, and the job fails
closed when they are absent.

Rollback restores the unchanged pair:

```text
AZURE_SEARCH_INDEX=legal-court-rag-index-v3
AZURE_SEARCH_KNOWLEDGEBASE_NAME=legal-court-rag-index-v3-agent-upgrade
```

Use the same deployment-specific Azure CLI command with those two values,
then run the canary workflow. Candidate resources remain available for
investigation; they are not automatically deleted.

## Monitoring

The daily `index-canary-monitor.yml` workflow reads `AZURE_SEARCH_INDEX` from
the GitHub environment, defaults to the v3 rollback index, and authenticates
with OIDC. After promotion, update the Production environment variables to
the active pair so the canary validates the promoted index. On rollback,
restore the v3 pair and rerun the monitor manually.
