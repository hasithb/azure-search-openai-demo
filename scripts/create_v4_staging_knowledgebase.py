"""Create the agentic-retrieval knowledge base for a disposable v4 index."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.upload_v4_staging import validate_staging_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--knowledgebase", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--openai-service", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tenant-id", default=os.environ.get("AZURE_TENANT_ID", ""))
    parser.add_argument("--execute", action="store_true", help="Create the knowledge base; default is validation-only")
    args = parser.parse_args()

    validate_staging_target(args.index)
    validate_staging_target(args.knowledgebase)
    if args.index.casefold() not in args.knowledgebase.casefold():
        raise ValueError("Knowledge-base name must include the staging index name")

    if not args.execute:
        print(
            {
                "index": args.index,
                "knowledgebase": args.knowledgebase,
                "status": "validated",
                "execute": False,
            }
        )
        return 0

    os.environ.update(
        {
            "AZURE_SEARCH_SERVICE": args.service,
            "AZURE_SEARCH_INDEX": args.index,
            "AZURE_SEARCH_KNOWLEDGEBASE_NAME": args.knowledgebase,
            "AZURE_OPENAI_SERVICE": args.openai_service,
            "AZURE_OPENAI_KNOWLEDGEBASE_DEPLOYMENT": args.deployment,
            "AZURE_OPENAI_KNOWLEDGEBASE_MODEL": args.model,
        }
    )
    if args.tenant_id:
        os.environ["AZURE_TENANT_ID"] = args.tenant_id

    from scripts.create_knowledgebase import create_knowledgebase

    asyncio.run(create_knowledgebase())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())