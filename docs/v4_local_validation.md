# Local v4 validation

Run the deterministic application-gate harness before dispatching the v4 release workflow:

```shell
source .venv-upgrade/bin/activate
python scripts/preflight_v4_local.py \
	--mode offline \
	--require-runtime-contract \
	--output reports/v4-local
```

The harness starts a loopback fixture origin, exercises the same request and response adapters used by the retrieval, category, source-hierarchy, citation, and ACL gates, writes all six gate reports, and validates them with the strict application-gate aggregator. It also proves the deterministic Container App readiness/runtime-identity contract: the expected revision is ready, running, healthy, receives 100% traffic, uses an immutable image digest, and has the expected Search index and knowledge-base environment values.

The preflight includes fail-closed chat probes for the two remote failure classes that a happy-path fixture can miss: a request timeout and a non-JSON response. Both must be converted into gate failures rather than being treated as usable answers. The command always reports `promotion_eligible: false` and performs no Azure or production mutation.

The offline ACL and highlight results are contract fixtures. They prove that report shape and deterministic gate logic are wired correctly; they do not prove Azure Search permission filtering or real browser rendering. The live ACL and browser gates remain required for a candidate deployment. The local runtime contract likewise does not replace managed identity, Azure Search permissions, Container Apps provisioning, or live `/chat` latency checks.

Use the explicit smoke mode only when a deployed candidate workflow has supplied real candidate inputs. The offline fixture does not silently become a live check:

```shell
python scripts/preflight_v4_local.py --mode live-smoke
```

The current command intentionally stops with an explanatory error until the candidate URL, provenance, oracle, and Azure-only inputs are provided by the release workflow. Keep workflow dispatches at `promote=false` until the local result, candidate runtime checks, six live reports, fidelity evidence, and application provenance all pass.