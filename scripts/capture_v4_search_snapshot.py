"""Capture a verified Search snapshot for a v4 candidate index."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from audit_source_documents import fetch_live_index_documents, write_index_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default=os.environ.get("AZURE_SEARCH_SERVICE", ""), required=False)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.service:
        raise ValueError("--service or AZURE_SEARCH_SERVICE is required")
    if "v4" not in args.index.casefold() or "staging" not in args.index.casefold():
        raise ValueError("Snapshot target must be a v4 staging index")
    documents = fetch_live_index_documents(args.service, args.index)
    provenance = write_index_snapshot(args.output, documents, args.service, args.index)
    print(provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
