"""Validate canonical section-boundary evidence for the highlight release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .generate_highlight_oracle import PDF_SECTION_RE, body_evidence, normalize_text
except ImportError:
    from generate_highlight_oracle import PDF_SECTION_RE, body_evidence, normalize_text


class OracleValidationError(ValueError):
    """Raised when highlight oracle evidence is incomplete or stale."""


def validate(
    report: dict[str, Any],
    snapshot_dir: Path,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    if report.get("schema_version") != 1:
        raise OracleValidationError("Highlight oracle must use schema version 1")
    oracle_version = str(report.get("oracle_version") or "").strip()
    if not oracle_version:
        raise OracleValidationError("Highlight oracle is missing oracle_version")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise OracleValidationError("Highlight oracle has no cases")
    identities = report.get("source_identities")
    if not isinstance(identities, list) or not identities:
        raise OracleValidationError("Highlight oracle has no source identities")
    if report.get("case_count") != len(cases) or report.get("source_count") != len(identities):
        raise OracleValidationError("Highlight oracle counts do not match its evidence")
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise OracleValidationError(f"Canonical snapshot manifest is missing: {manifest_path}")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if report.get("snapshot_manifest_sha256") != manifest_hash:
        raise OracleValidationError("Highlight oracle is not bound to the canonical snapshot manifest")

    case_ids: set[str] = set()
    case_identities: set[str] = set()
    snapshots: dict[str, dict[str, Any]] = {}
    for path in snapshot_dir.glob("*.json"):
        if path.name != "manifest.json":
            snapshots[path.name] = json.loads(path.read_text(encoding="utf-8"))

    for case in cases:
        if not isinstance(case, dict):
            raise OracleValidationError("Highlight oracle contains a non-object case")
        required = ("case_id", "identity", "sourcefile", "sourcepage", "subsection_id", "expected_heading", "heading_locator", "body_sha256", "body_length")
        if any(not str(case.get(field) or "").strip() for field in required):
            raise OracleValidationError("Highlight oracle contains an incomplete case")
        case_id = str(case["case_id"])
        if case_id in case_ids:
            raise OracleValidationError(f"Duplicate highlight oracle case: {case_id}")
        case_ids.add(case_id)
        case_identities.add(str(case["identity"]))
        if case.get("oracle_version") != oracle_version:
            raise OracleValidationError("Highlight oracle cases use inconsistent oracle versions")
        if case.get("identity") not in identities:
            raise OracleValidationError("Highlight oracle case references an unknown source identity")
        snapshot_name = Path(str(case.get("snapshot_file") or "")).name
        snapshot = snapshots.get(snapshot_name)
        if snapshot is None:
            raise OracleValidationError(f"Highlight oracle snapshot is missing: {snapshot_name}")
        blocks = snapshot.get("schema_census", {}).get("blocks", [])
        if not blocks and snapshot.get("source_type") == "pdf":
            blocks = [
                {"kind": "heading" if PDF_SECTION_RE.match(line) else "body", "locator": f"pdf-line[{index}]", "text": normalize_text(line)}
                for index, line in enumerate(str(snapshot.get("extracted_text") or "").splitlines(), start=1)
                if normalize_text(line)
            ]
        heading = next((block for block in blocks if block.get("locator") == case.get("heading_locator")), None)
        next_heading = next((block for block in blocks if block.get("locator") == case.get("next_heading_locator")), None)
        if heading is None:
            raise OracleValidationError("Highlight oracle heading locator is not present in its snapshot")
        body_text, body_sha256, preceding_text = body_evidence(blocks, heading, next_heading)
        if body_sha256 != case.get("body_sha256") or len(body_text) != case.get("body_length") or body_text != case.get("body_text"):
            raise OracleValidationError("Highlight oracle body evidence does not match its canonical snapshot")
        if case.get("preceding_text", preceding_text) != preceding_text:
            raise OracleValidationError("Highlight oracle preceding evidence does not match its canonical snapshot")
        next_locator = str(case.get("next_heading_locator") or "")
        if next_locator and not str(case.get("next_heading") or "").strip():
            raise OracleValidationError("Highlight oracle next-heading locator has no text")
    if case_identities != set(identities):
        raise OracleValidationError("Highlight oracle source identity list is incomplete")
    gate = {
        "gate": "highlight",
        "schema_version": 1,
        "status": "PASS",
        "oracle_version": oracle_version,
        "case_count": len(cases),
        "source_count": len(identities),
        "snapshot_manifest_sha256": manifest_hash,
    }
    if provenance is not None:
        gate["provenance"] = provenance
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.oracle.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise OracleValidationError("Highlight oracle must be a JSON object")
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict):
            raise OracleValidationError("Candidate provenance must be a JSON object")
        gate = validate(report, args.snapshot_dir, provenance=provenance)
    except (OSError, json.JSONDecodeError, OracleValidationError) as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema_version": 1, "status": "FAIL", "error": str(error)}, indent=2) + "\n", encoding="utf-8")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())