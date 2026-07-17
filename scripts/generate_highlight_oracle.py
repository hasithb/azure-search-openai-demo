"""Build deterministic section cases from canonical HTML oracle snapshots.

The cases describe source identity and exact heading boundaries. They are an
independent input for citation and supporting-content validation; they do not
reuse the production scraper or frontend matcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SECTION_RE = re.compile(
    r"^(?P<id>(?:rule\s+\d+(?:\.\d+)*|para\s+\d+(?:\.\d+)*|"
    r"(?:part|chapter|section|appendix)\s+[0-9A-Za-z]+|"
    r"[A-Z](?:\.?\d)+(?:\.\d+)*|\d+(?:\.\d+)+))\b",
    re.IGNORECASE,
)
PDF_SECTION_RE = re.compile(
    r"^\s*(?P<id>(?:rule\s+\d+(?:\.\d+)*|para\.?\s+\d+(?:\.\d+)*|"
    r"(?:part|chapter|section|appendix)\s+[0-9A-Za-z]+|"
    r"[A-Z](?:\.?\d)+(?:\.\d+)*|\d+(?:\.\d+)+))\b(?P<title>.*)$",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def body_evidence(blocks: list[dict[str, Any]], heading: dict[str, Any], next_heading: dict[str, Any] | None) -> tuple[str, str]:
    """Fingerprint the complete canonical span from heading through its body."""
    start = blocks.index(heading)
    end = blocks.index(next_heading) if next_heading is not None else len(blocks)
    body_text = normalize_text(" ".join(str(block.get("text") or "") for block in blocks[start:end]))
    return body_text, hashlib.sha256(body_text.casefold().encode("utf-8")).hexdigest()


def case_id(identity: str, locator: str) -> str:
    return hashlib.sha256(f"{identity}|{locator}".encode("utf-8")).hexdigest()[:20]


def load_snapshot_cases(snapshot_dir: Path) -> list[dict[str, Any]]:
    snapshot_dir = snapshot_dir.resolve()
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(snapshot_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("status") != "ok":
            continue
        identity = str(snapshot.get("identity") or "").strip()
        content_hash = str(
            snapshot.get("content_sha256") or snapshot.get("source_sha256") or ""
        ).strip()
        blocks = snapshot.get("schema_census", {}).get("blocks", [])
        if not identity or not content_hash:
            raise ValueError(f"Incomplete canonical snapshot: {path}")

        if not isinstance(blocks, list):
            raise ValueError(f"Invalid HTML block census: {path}")
        if not blocks and snapshot.get("source_type") == "pdf":
            extracted_text = str(snapshot.get("extracted_text") or "")
            blocks = []
            for line_number, line in enumerate(extracted_text.splitlines(), start=1):
                text = normalize_text(line)
                if text:
                    blocks.append({
                        "kind": "heading" if PDF_SECTION_RE.match(line) else "body",
                        "locator": f"pdf-line[{line_number}]",
                        "text": text,
                    })
            if not blocks:
                raise ValueError(f"PDF snapshot produced no section headings: {path}")

        headings = [
            block
            for block in blocks
            if isinstance(block, dict) and block.get("kind") == "heading" and normalize_text(str(block.get("text") or ""))
        ]
        for index, heading in enumerate(headings):
            text = normalize_text(str(heading["text"]))
            match = SECTION_RE.match(text)
            if not match:
                continue
            locator = str(heading.get("locator") or "").strip()
            if not locator:
                raise ValueError(f"Section heading has no locator: {path}")
            key = (identity, locator)
            if key in seen:
                raise ValueError(f"Duplicate section heading: {identity} {locator}")
            seen.add(key)
            next_heading = headings[index + 1] if index + 1 < len(headings) else None
            body_text, body_sha256 = body_evidence(blocks, heading, next_heading)
            cases.append(
                {
                    "case_id": case_id(identity, locator),
                    "oracle_version": str(snapshot.get("oracle_version") or ""),
                    "snapshot_file": str(path.relative_to(ROOT)),
                    "snapshot_content_sha256": content_hash,
                    "identity": identity,
                    "category": str(snapshot.get("category") or ""),
                    "sourcefile": str(snapshot.get("sourcefile") or ""),
                    "sourcepage": text,
                    "subsection_id": match.group("id"),
                    "expected_heading": text,
                    "heading_locator": locator,
                    "next_heading": normalize_text(str(next_heading.get("text") or "")) if next_heading else None,
                    "next_heading_locator": str(next_heading.get("locator") or "") if next_heading else None,
                    "body_text": body_text,
                    "body_sha256": body_sha256,
                    "body_length": len(body_text),
                }
            )
    if not cases:
        raise ValueError("Canonical snapshots produced no section cases")
    return cases


def build_report(snapshot_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Canonical snapshot manifest is missing: {manifest_path}")
    cases = load_snapshot_cases(snapshot_dir)
    identities = sorted({case["identity"] for case in cases})
    categories = sorted({case["category"] for case in cases})
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Oracle case IDs are not unique")
    return {
        "schema_version": 1,
        "oracle_version": cases[0]["oracle_version"],
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "source_count": len(identities),
        "categories": categories,
        "source_identities": identities,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "reports" / "html_oracle_snapshots")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.snapshot_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())