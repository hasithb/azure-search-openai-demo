"""Run fail-closed Azure AI Search ACL behavior checks."""

from __future__ import annotations

import os
from typing import Any

import httpx
from azure.identity.aio import AzureCliCredential
from azure.search.documents.aio import SearchClient

try:
    from .gate_common import (  # type: ignore[unresolved-import]
        GateFailure,
        gate_parser,
        passing_report,
        run_gate,
    )
except ImportError:
    from gate_common import GateFailure, gate_parser, passing_report, run_gate


async def search_counts(search_service: str, index_name: str, tenant_id: str) -> dict[str, int]:
    endpoint = f"https://{search_service}.search.windows.net"
    credential = AzureCliCredential(tenant_id=tenant_id)
    try:
        token = await credential.get_token("https://search.azure.com/.default")
        async with SearchClient(endpoint=endpoint, index_name=index_name, credential=credential) as client:
            authorized = await client.search(
                search_text="*",
                top=5,
                select=["id", "sourcefile"],
                include_total_count=True,
                x_ms_query_source_authorization=token.token,
            )
            authorized_docs = [document async for document in authorized]
            elevated = await client.search(
                search_text="*",
                top=5,
                select=["id", "sourcefile", "oids", "groups"],
                include_total_count=True,
                x_ms_enable_elevated_read=True,
            )
            elevated_docs = [document async for document in elevated]
    finally:
        await credential.close()

    return {
        "authorized_count": len(authorized_docs),
        "elevated_count": len(elevated_docs),
        "authorized_total": await authorized.get_count(),
        "elevated_total": await elevated.get_count(),
    }


async def run(candidate: str, provenance: dict[str, str]) -> dict[str, Any]:
    if os.environ.get("V4_LOCAL_FIXTURE") == "1":
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{candidate}/api/acl")
            response.raise_for_status()
            counts = response.json()
        if not isinstance(counts, dict):
            raise GateFailure("Fixture ACL response must be a JSON object")
        if counts.get("authorized_total", 0) <= 0 or counts.get("elevated_total", 0) <= 0:
            raise GateFailure("Fixture ACL response contains no documents")
        if counts["authorized_total"] > counts["elevated_total"]:
            raise GateFailure("Fixture ACL result set is broader than elevated read")
        return passing_report("acl", [{"id": "acl_filtering_fixture", **counts, "status": "PASS"}], provenance=provenance)

    del candidate
    search_service = os.environ.get("AZURE_SEARCH_SERVICE", "").strip()
    index_name = os.environ.get("AZURE_SEARCH_INDEX", "").strip()
    tenant_id = os.environ.get("AZURE_TENANT_ID", "").strip()
    if not search_service or not index_name or not tenant_id:
        raise GateFailure("ACL gate requires AZURE_SEARCH_SERVICE, AZURE_SEARCH_INDEX, and AZURE_TENANT_ID")

    counts = await search_counts(search_service, index_name, tenant_id)
    if counts["elevated_count"] <= 0 or counts["elevated_total"] <= 0:
        raise GateFailure("Elevated ACL query returned no documents")
    if counts["authorized_count"] <= 0 or counts["authorized_total"] <= 0:
        raise GateFailure("Authorized ACL query returned no documents")
    if counts["authorized_total"] > counts["elevated_total"]:
        raise GateFailure("Authorized ACL result set is broader than elevated read")

    return passing_report("acl", [{"id": "acl_filtering", **counts, "status": "PASS"}], provenance=provenance)


def main() -> int:
    args = gate_parser(__doc__ or "Run the ACL gate").parse_args()
    return run_gate("acl", args.output, run, args.candidate_url, args.provenance)


if __name__ == "__main__":
    raise SystemExit(main())