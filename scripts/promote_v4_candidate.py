"""Validate an approved v4 candidate before a blue/green application cutover.

This command is deliberately non-destructive. It validates the release evidence
bundle and emits the exact target pair that a separately approved deployment may
switch to; it never mutates Search, deletes an index, or changes application
configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LEGACY_INDEX = "legal-court-rag-index-v3"


class PromotionError(ValueError):
    """Raised when a candidate is not eligible for production promotion."""


def _require_string(bundle: dict[str, Any], field: str) -> str:
    value = str(bundle.get(field) or "").strip()
    if not value:
        raise PromotionError(f"Evidence bundle is missing {field}")
    return value


def validate_evidence_bundle(bundle: dict[str, Any]) -> dict[str, str]:
    if bundle.get("approved") is not True:
        raise PromotionError("Evidence bundle is not approved")
    if str(bundle.get("approval_environment") or "") != "Production":
        raise PromotionError("Evidence bundle requires Production approval")

    index_name = _require_string(bundle, "candidate_index")
    knowledgebase_name = _require_string(bundle, "candidate_knowledgebase")
    artifact_sha256 = _require_string(bundle, "artifact_sha256")
    snapshot_sha256 = _require_string(bundle, "search_snapshot_sha256")
    rollback_index = _require_string(bundle, "rollback_index")
    rollback_knowledgebase = _require_string(bundle, "rollback_knowledgebase")
    fidelity = bundle.get("fidelity")
    if not isinstance(fidelity, dict):
        raise PromotionError("Evidence bundle is missing fidelity results")
    if any(int(fidelity.get(field, 0) or 0) for field in ("unmatched", "ambiguous", "unavailable", "unclassified")):
        raise PromotionError("Fidelity gate is not clean")
    if fidelity.get("substantive_coverage") != 1.0:
        raise PromotionError("Fidelity gate does not report 100% substantive coverage")
    artifact_search = bundle.get("artifact_search")
    if not isinstance(artifact_search, dict) or any(
        int(artifact_search.get(field, 0) or 0) for field in ("missing_count", "extra_count", "mismatched_count")
    ):
        raise PromotionError("Artifact/Search equality gate is not clean")
    candidate_validation = bundle.get("candidate_validation")
    if not isinstance(candidate_validation, dict) or candidate_validation.get("status") != "PASS":
        raise PromotionError("Candidate Search validation gate is not clean")
    if any(LEGACY_INDEX.casefold() in target.casefold() for target in (index_name, knowledgebase_name)):
        raise PromotionError("Refusing to promote or mutate the legacy v3 target")
    if "v4" not in index_name.casefold() or "v4" not in knowledgebase_name.casefold():
        raise PromotionError("Candidate targets must contain v4")
    if index_name.casefold() not in knowledgebase_name.casefold():
        raise PromotionError("Knowledge-base target must identify the candidate index")
    application_gates = bundle.get("application_gates")
    if not isinstance(application_gates, dict) or application_gates.get("schema_version") != 1 or application_gates.get("status") != "PASS":
        raise PromotionError("Application-gate validation is not clean")
    gates = application_gates.get("gates")
    required_gates = {"retrieval", "category", "source_hierarchy", "citation", "acl", "highlight"}
    if not isinstance(gates, dict) or set(gates) != required_gates:
        raise PromotionError("Application-gate evidence must contain all six required gates")
    for gate_name, gate in gates.items():
        if not isinstance(gate, dict) or gate.get("status") != "PASS" or gate.get("gate") != gate_name:
            raise PromotionError(f"Application-gate evidence is missing a passing {gate_name} gate")
    highlight_gate = gates["highlight"]
    if int(highlight_gate.get("case_count", 0) or 0) <= 0 or int(highlight_gate.get("source_count", 0) or 0) <= 0:
        raise PromotionError("Application-gate highlight evidence is empty")
    application_provenance = application_gates.get("provenance")
    if not isinstance(application_provenance, dict):
        raise PromotionError("Application-gate evidence is missing provenance")
    for field, expected in {
        "search_index": index_name,
        "knowledge_base": knowledgebase_name,
        "artifact_sha256": artifact_sha256,
        "search_snapshot_sha256": snapshot_sha256,
    }.items():
        if application_provenance.get(field) != expected:
            raise PromotionError(f"Application-gate provenance does not match {field}")
    if rollback_index.casefold() != LEGACY_INDEX.casefold():
        raise PromotionError("Rollback index must be the unchanged v3 production index")
    if not rollback_knowledgebase or "v3" not in rollback_knowledgebase.casefold():
        raise PromotionError("Rollback knowledge base must identify the v3 production pair")

    return {
        "candidate_index": index_name,
        "candidate_knowledgebase": knowledgebase_name,
        "artifact_sha256": artifact_sha256,
        "search_snapshot_sha256": snapshot_sha256,
        "rollback_index": rollback_index,
        "rollback_knowledgebase": rollback_knowledgebase,
    }


def load_and_validate(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PromotionError("Evidence bundle must be a JSON object")
    return validate_evidence_bundle(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(load_and_validate(args.evidence), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
