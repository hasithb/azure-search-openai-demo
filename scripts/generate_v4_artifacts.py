"""Generate validated local v4 index artifacts from HTML oracle snapshots.

This command is deliberately offline. It never creates an Azure client and never
writes to Azure AI Search or Azure OpenAI. The input is the refreshed oracle
snapshot set; transformation uses the production CPR scraper and chunk builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_source_documents import CanonicalSource, load_web_sources, normalize_url  # noqa: E402
import audit_html_transition as transition  # noqa: E402
import update_cpr_index_v3 as updater  # noqa: E402
from upload_court_guides_v3 import GUIDE_FILES, map_doc  # noqa: E402


def content_hash(document: dict[str, Any]) -> str:
    content = document.get("content", "")
    if isinstance(content, list):
        content = "\n".join(content)
    value = "|".join(
        str(document.get(field, "") or "")
        for field in ("id", "sourcefile", "sourcepage", "category", "storageUrl", "updated")
    ) + f"|{content}|{document.get('embedding_text', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot_hash(snapshot: dict[str, Any]) -> str:
    """Hash the immutable source snapshot without depending on JSON key order."""
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_embedding_text(document: dict[str, Any], content: str) -> str:
    sourcefile = str(document.get("sourcefile") or "")
    category = str(document.get("category") or "")
    hierarchy_path = str(document.get("hierarchy_path") or "")
    section_title = str(document.get("section_title") or "")
    legal_references = document.get("legal_references") or []
    embedding_header = "\n".join(
        line
        for line in (
            f"SOURCE FAMILY: {category}",
            f"SOURCE: {sourcefile}",
            f"HIERARCHY: {hierarchy_path}",
            f"LEGAL REFERENCES: {', '.join(legal_references)}" if legal_references else "",
            f"SECTION: {section_title}",
        )
        if line
    )
    return f"{embedding_header}\n\n{content}".strip()


def enrich_retrieval_metadata(document: dict[str, Any], content_override: str | None = None) -> dict[str, Any]:
    """Add stable hierarchy and reference text without changing display content."""
    content = str(document.get("content") or "")
    sourcepage = str(document.get("sourcepage") or "")
    sourcefile = str(document.get("sourcefile") or "")
    category = str(document.get("category") or "")
    subsection_id = str(document.get("subsection_id") or "")
    section_title = subsection_id or sourcepage or sourcefile
    hierarchy_parts = [part for part in (sourcefile, sourcepage, subsection_id) if part]
    hierarchy_path = " > ".join(dict.fromkeys(hierarchy_parts))
    legal_references = sorted(
        set(
            re.findall(
                r"\b(?:CPR\s+)?(?:Part\s+\d+[A-Z]?|PD\s*\d+[A-Z]*|\d+[A-Z]?\.\d+(?:\.\d+)*)\b",
                " ".join((sourcepage, subsection_id, content)),
                flags=re.IGNORECASE,
            )
        )
    )
    document["section_title"] = section_title
    document["hierarchy_path"] = hierarchy_path
    document["legal_references"] = legal_references
    document["embedding_text"] = _build_embedding_text(document, content_override if content_override is not None else content)
    return document


def expand_oversized_embedding_windows(
    documents: list[dict[str, Any]], max_embedding_tokens: int = 8100
) -> list[dict[str, Any]]:
    """Create bounded retrieval children without changing canonical content."""
    expanded: list[dict[str, Any]] = []
    chunker = updater.LegalDocumentChunker(max_tokens=6500, overlap_tokens=200)

    for document in documents:
        if chunker.count_tokens(document.get("embedding_text", "")) <= max_embedding_tokens:
            expanded.append(document)
            continue

        original_id = str(document.get("id") or "")
        chunks = chunker.chunk_legal_document(
            str(document.get("content") or ""),
            original_id,
            str(document.get("section_title") or document.get("sourcefile") or original_id),
        )
        children: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            child = dict(document)
            child["id"] = f"{original_id}__window_{index}"
            child["parent_id"] = original_id
            child["child_window"] = index
            child["child_window_count"] = len(chunks)
            child["embedding_text"] = _build_embedding_text(child, str(chunk["text"]))
            if chunker.count_tokens(child["embedding_text"]) > max_embedding_tokens:
                raise ValueError(f"Child embedding window exceeds {max_embedding_tokens} tokens: {child['id']}")
            children.append(child)
        if len(children) < 2:
            raise ValueError(f"Unable to split oversized embedding input: {original_id}")
        expanded.extend(children)

    return expanded


def deduplicate_sources_by_url(sources: list[CanonicalSource]) -> dict[str, CanonicalSource]:
    """Choose one descriptive source identity for each canonical HTML URL."""
    selected: dict[str, CanonicalSource] = {}
    for source in sources:
        normalized_url = normalize_url(source.url)
        if not normalized_url:
            selected[source.identity] = source
            continue
        current = selected.get(normalized_url)
        if current is None or (len(source.sourcefile), source.sourcefile) > (
            len(current.sourcefile), current.sourcefile
        ):
            selected[normalized_url] = source
    return {source.identity: source for source in selected.values()}


def validate_source_snapshot(snapshot: dict[str, Any], source: Any, path: Path) -> None:
    """Reject incomplete oracle snapshots before production transformations run."""
    if snapshot.get("status") != "ok":
        raise ValueError(f"Source snapshot is not ok: {path.name}")
    if snapshot.get("source_type") != source.source_type:
        raise ValueError(f"Source snapshot type mismatch: {path.name}")
    if source.source_type == "pdf":
        content_type = str(snapshot.get("content_type") or "").casefold()
        if "application/pdf" not in content_type:
            raise ValueError(f"PDF snapshot has no PDF content type: {path.name}")
        if not snapshot.get("source_sha256"):
            raise ValueError(f"PDF snapshot has no source byte hash: {path.name}")
        if not snapshot.get("extracted_text"):
            raise ValueError(f"PDF snapshot has no extracted text: {path.name}")
    if not snapshot.get("html"):
        raise ValueError(f"Source snapshot has no transformed HTML: {path.name}")


def generate(
    snapshot_dir: Path,
    court_guides_dir: Path | None = None,
    release_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    release_id = release_id or os.environ.get("V4_RELEASE_ID", "")
    if not release_id:
        raise ValueError("release_id is required for a release-bound artifact")
    all_sources = {source.identity: source for source in load_web_sources()}
    sources = deduplicate_sources_by_url(list(all_sources.values()))
    actions = {entry["sourcefile"]: entry for entry in updater.ACTION_LIST}
    actions_by_url = {entry["url"].rstrip("/"): entry for entry in updater.ACTION_LIST}
    documents: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    source_snapshot_hashes: dict[str, str] = {}
    used_ids: set[str] = set()
    snapshot_count = 0

    for path in sorted(snapshot_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("status") != "ok":
            continue
        source = sources.get(snapshot.get("identity"))
        if source is None:
            if snapshot.get("identity") in all_sources:
                continue
            raise ValueError(f"Snapshot has no canonical source: {path.name}")
        snapshot_count += 1
        validate_source_snapshot(snapshot, source, path)
        if source.identity in source_snapshot_hashes:
            raise ValueError(f"Duplicate source snapshot: {source.identity}")
        action = actions.get(snapshot.get("sourcefile"))
        requested_url = str(snapshot.get("requested_url") or "").rstrip("/")
        action = action or actions_by_url.get(requested_url)
        if action is None:
            action = {
                "sourcefile": source.sourcefile,
                "azure_id": None,
                "url": str(snapshot.get("final_url") or requested_url or source.url).rstrip("/"),
                "section": "ORACLE",
            }
        action = {
            **action,
            "sourcefile": source.sourcefile,
            "url": str(snapshot.get("final_url") or action.get("url") or source.url).rstrip("/"),
        }
        soup = BeautifulSoup(str(snapshot.get("html") or ""), "html.parser")
        scraped = updater.scrape_page(
            updater.requests.Session(),
            action,
            prefetched_result=(soup, snapshot.get("final_url", requested_url), snapshot.get("redirect_count", 0)),
        )
        if scraped is None:
            raise ValueError(f"Production scraper returned no content: {source.identity}")
        built = updater.build_index_docs(action, scraped)
        if not built:
            raise ValueError(f"No index documents generated: {source.identity}")
        source_suffix = hashlib.sha256(source.identity.encode("utf-8")).hexdigest()[:10]
        for document in built:
            enrich_retrieval_metadata(document)
            if document["id"] in used_ids:
                document["id"] = f"{document['id']}_{source_suffix}"
                document["parent_id"] = f"{document['parent_id']}_{source_suffix}"
            used_ids.add(document["id"])
        source_counts[source.sourcefile] = source_counts.get(source.sourcefile, 0) + len(built)
        source_snapshot_hashes[source.identity] = snapshot_hash(snapshot)
        documents.extend(built)

    missing_snapshot_identities = sorted(set(sources) - set(source_snapshot_hashes))
    if missing_snapshot_identities:
        raise ValueError(
            "Missing canonical source snapshots: " + ", ".join(missing_snapshot_identities)
        )

    court_guides_dir = court_guides_dir or ROOT / "scripts" / "court_guides_processing_pipeline" / "outputs_azure_di"
    extraction_manifest_path = court_guides_dir / "court_guides_extraction_manifest.json"
    if not extraction_manifest_path.exists():
        raise ValueError(f"Court-guide extraction manifest is missing: {extraction_manifest_path}")
    extraction_manifest = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    if extraction_manifest.get("schema_version") != 1:
        raise ValueError("Court-guide extraction manifest has an unsupported schema")
    for guide_name, guide in GUIDE_FILES.items():
        guide_path = court_guides_dir / guide["file"]
        if not guide_path.exists():
            raise ValueError(f"Fresh court-guide artifact is missing: {guide_path}")
        raw_documents = json.loads(guide_path.read_text(encoding="utf-8"))
        extraction_entry = next(
            (
                entry
                for entry in extraction_manifest.get("guides", {}).values()
                if entry.get("processed_json") == guide["file"]
            ),
            None,
        )
        if not extraction_entry or extraction_entry.get("processed_json_sha256") != hashlib.sha256(guide_path.read_bytes()).hexdigest():
            raise ValueError(f"Court-guide artifact provenance does not match extraction manifest: {guide_path}")
        if not isinstance(raw_documents, list) or not raw_documents:
            raise ValueError(f"Fresh court-guide artifact is empty: {guide_path}")
        for raw_document in raw_documents:
            document = map_doc(raw_document, id_prefix=guide_name.replace(" ", "_"))
            document.setdefault("parent_id", "")
            document.setdefault("subsection_id", "")
            document.setdefault("subsections", [])
            enrich_retrieval_metadata(document)
            if document["id"] in used_ids:
                raise ValueError(f"Duplicate court-guide document id: {document['id']}")
            used_ids.add(document["id"])
            document["artifact_content_sha256"] = content_hash(document)
            documents.append(document)
        source_counts[guide["sourcefile"]] = len(raw_documents)
        snapshot_count += 1

    documents = expand_oversized_embedding_windows(documents)
    ids = [str(document.get("id") or "") for document in documents]
    duplicate_ids = sorted({document_id for document_id in ids if ids.count(document_id) > 1})
    missing_fields = []
    oversized = []
    for document in documents:
        for field in ("id", "content", "sourcefile", "sourcepage", "parent_id", "subsection_id", "subsections"):
            if field not in document:
                missing_fields.append(f"{document.get('id', '<unknown>')}: {field}")
        token_count = updater.LegalDocumentChunker(max_tokens=8000).count_tokens(document.get("embedding_text", ""))
        if token_count > 8100:
            oversized.append({"id": document.get("id", ""), "token_count": token_count})
        document["artifact_content_sha256"] = content_hash(document)

    if duplicate_ids or missing_fields or oversized:
        raise ValueError(json.dumps({"duplicate_ids": duplicate_ids, "missing_fields": missing_fields, "oversized": oversized}, indent=2))

    manifest = {
        "release_id": release_id,
        "artifact_version": f"v4-{release_id}",
        "court_guides_extraction_manifest_sha256": hashlib.sha256(extraction_manifest_path.read_bytes()).hexdigest(),
        "snapshot_count": snapshot_count,
        "document_count": len(documents),
        "source_count": len(source_counts),
        "source_counts": source_counts,
        "source_snapshot_hashes": source_snapshot_hashes,
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 3072,
        "embedding_token_limit": 8100,
        "document_ids_unique": True,
        "metadata_complete": True,
        "oversized_document_count": 0,
    }
    return documents, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate offline v4 CPR/PD index artifacts")
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "reports" / "html_oracle_snapshots")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "index_v4_artifacts")
    parser.add_argument("--release-id", default=None, help="Immutable release identifier, or V4_RELEASE_ID")
    parser.add_argument(
        "--court-guides-dir",
        type=Path,
        default=ROOT / "scripts" / "court_guides_processing_pipeline" / "outputs_azure_di",
        help="Directory containing the eight processed court-guide JSON artifacts",
    )
    args = parser.parse_args()

    documents, manifest = generate(args.snapshot_dir, args.court_guides_dir, args.release_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "documents.jsonl").write_text(
        "".join(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n" for document in documents),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())