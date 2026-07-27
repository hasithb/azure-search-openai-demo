from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "update-index-v4.yml"


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_candidate_image_output_is_owned_by_deploy_job():
    workflow = workflow_text()

    assert "needs.build-candidate.outputs.candidate_image" not in workflow
    deploy_job = workflow.split("  deploy-candidate-app:", 1)[1].split("  audit-candidate:", 1)[0]
    assert "candidate_image: ${{ steps.image.outputs.candidate_image }}" in deploy_job
    assert 'candidate_image="${registry_server}/v4-candidate@${digest}"' in deploy_job
    assert "@sha256:" not in deploy_job.split('candidate_image="', 1)[1].split('"', 1)[0]


def test_readiness_precedes_runtime_validation_and_exports_uppercase_inputs():
    workflow = workflow_text()
    readiness_position = workflow.index("python scripts/wait_v4_candidate_readiness.py")
    validation_position = workflow.index("python scripts/validate_v4_runtime_identity.py")

    assert readiness_position < validation_position
    assert "CANDIDATE_IMAGE: ${{ needs.deploy-candidate-app.outputs.candidate_image }}" in workflow
    assert "CANDIDATE_REVISION: ${{ needs.deploy-candidate-app.outputs.candidate_revision }}" in workflow


def test_evidence_bundle_requires_runtime_identity_in_each_build():
    workflow = workflow_text()

    assert workflow.count("--candidate-runtime-identity reports/candidate_runtime_identity.json") == 2
    assert workflow.count("reports/candidate_runtime_identity.json") >= 3


def test_local_validation_runs_before_remote_preflight():
    workflow = workflow_text()

    assert workflow.index("  local-validation:") < workflow.index("  preflight:")
    local_validation = workflow.split("  local-validation:", 1)[1].split("  preflight:", 1)[0]
    assert "python scripts/preflight_v4_local.py --mode offline --require-runtime-contract --output reports/v4-local" in local_validation
    assert "python scripts/preflight_v4_release.py" in local_validation
    assert "tests/fixtures/v4/ready/preflight.json" in local_validation


def test_preflight_checks_release_index_uniqueness_read_only():
    workflow = workflow_text()
    preflight = workflow.split("  preflight:", 1)[1].split("  build-candidate:", 1)[0]

    assert "actions/checkout@v4" in preflight
    assert "ref: ${{ github.sha }}" in preflight
    assert "indexes?api-version=2024-07-01" in preflight
    assert "validate_v4_release_index_uniqueness.py" in preflight
    assert "reports/v4_release_index_uniqueness.json" in preflight


def test_workflow_requires_exhaustive_citation_evidence_before_bundle():
    workflow = workflow_text()

    assert "Require exhaustive citation evidence" in workflow
    assert "v4_citation_coverage_input.json" in workflow
    assert "validate_v4_citation_coverage.py" in workflow
    assert "--release-index-uniqueness reports/v4_release_index_uniqueness.json" in workflow


def test_workflow_passes_replay_bound_coverage_to_downstream_gates():
    workflow = workflow_text()

    browser_step = workflow.split("Generate provenance-bound browser highlight gate", 1)[1].split("Generate provenance-bound ACL gate", 1)[0]
    assert "--exhaustive-coverage-input \"reports/v4_browser_shard_${shard_index}.json\"" in browser_step
    assert workflow.index("Run strict application gates") < workflow.index("Require exhaustive citation evidence")
    assert "if: inputs.promote == true" in workflow


def test_workflow_runs_and_merges_deterministic_browser_shards_before_coverage_validation():
    workflow = workflow_text()
    browser_step = workflow.split("Generate provenance-bound browser highlight gate", 1)[1].split("Generate provenance-bound ACL gate", 1)[0]

    assert "--shard-index" in browser_step
    assert "--shard-count" in browser_step
    assert "--merge-report" in browser_step
    assert "reports/v4_citation_coverage_input.json" in browser_step
    assert browser_step.index("--merge-report") > browser_step.index("--shard-count")
    assert "V4_BROWSER_SHARD_COUNT" in browser_step