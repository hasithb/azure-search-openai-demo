"""Summarize the independent raw HTML oracle snapshot corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DIR = ROOT / "reports" / "html_oracle_snapshots"
DEFAULT_OUTPUT = ROOT / "reports" / "html_oracle_census.json"


def load_snapshots(snapshot_dir: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for path in sorted(snapshot_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["snapshot_path"] = str(path)
        snapshots.append(payload)
    return snapshots


def build_report(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    source_status = Counter(str(snapshot.get("status", "unknown")) for snapshot in snapshots)
    kind_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    unavailable: list[dict[str, Any]] = []
    redirects = 0
    block_count = 0
    for snapshot in snapshots:
        if snapshot.get("status") == "not_applicable":
            continue
        if snapshot.get("status") != "ok":
            unavailable.append(
                {
                    "identity": snapshot.get("identity", ""),
                    "sourcefile": snapshot.get("sourcefile", ""),
                    "requested_url": snapshot.get("requested_url", ""),
                    "error": snapshot.get("error", ""),
                    "error_type": snapshot.get("error_type", ""),
                    "snapshot_path": snapshot.get("snapshot_path", ""),
                }
            )
            continue
        census = snapshot.get("schema_census", {})
        block_count += int(census.get("block_count", 0))
        kind_counts.update({str(key): int(value) for key, value in census.get("kind_counts", {}).items()})
        schema_counts.update({str(key): int(value) for key, value in census.get("schema_counts", {}).items()})
        redirects += int(snapshot.get("redirect_count", 0))

    return {
        "schema_version": 1,
        "oracle_version": snapshots[0].get("oracle_version") if snapshots else None,
        "snapshot_count": len(snapshots),
        "source_status": dict(sorted(source_status.items())),
        "successful_source_count": source_status.get("ok", 0),
        "unavailable_source_count": len(unavailable),
        "redirect_count": redirects,
        "legal_block_count": block_count,
        "kind_counts": dict(sorted(kind_counts.items())),
        "schema_counts": dict(sorted(schema_counts.items())),
        "release_blockers": unavailable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(load_snapshots(args.snapshot_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "release_blockers"}, sort_keys=True))
    return 0 if report["unavailable_source_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())