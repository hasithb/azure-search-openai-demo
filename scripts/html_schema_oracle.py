"""Independent DOM inventory for legal HTML fidelity checks.

This module intentionally does not import the production scraper. It records
legal-bearing DOM blocks before markdown or table flattening takes place.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ORACLE_VERSION = "2"


_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_CONTENT_ROOTS = {"article", "main"}
_BLOCK_TAGS = {"p", "li", "td", "th", "form", "aside"}
_SKIP_TAGS = {"head", "nav", "script", "style", "template", "noscript"}
_FOOTNOTE_RE = re.compile(r"footnote|foot-note|note", re.IGNORECASE)
_SPECIAL_HEADING_RE = re.compile(r"^(schedule|annex|appendix|form)\b", re.IGNORECASE)


def normalize_block_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


@dataclass(eq=False)
class DomNode:
    tag: str
    attributes: dict[str, str]
    parent: "DomNode | None" = None
    children: list["DomNode"] | None = None
    text_parts: list[str] | None = None
    content: list[str | "DomNode"] | None = None
    ordinal: int = 0

    def __post_init__(self) -> None:
        self.children = self.children or []
        self.text_parts = self.text_parts or []
        self.content = self.content or []

    @property
    def text(self) -> str:
        parts: list[str] = []
        pending = list(reversed(self.content))
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                parts.append(item)
            else:
                pending.extend(reversed(item.content))
        return normalize_block_text(" ".join(parts))

    @property
    def direct_text(self) -> str:
        return normalize_block_text(" ".join(self.text_parts))

    def path(self) -> str:
        parts: list[str] = []
        node: DomNode | None = self
        while node is not None:
            siblings = [child for child in (node.parent.children if node.parent else []) if child.tag == node.tag]
            position = siblings.index(node) + 1 if node in siblings else node.ordinal
            parts.append(f"{node.tag}[{position}]")
            node = node.parent
        return "/" + "/".join(reversed(parts))


class _DomParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = DomNode("document", {}, children=[])
        self.stack = [self.root]
        self.counts: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        attributes = {key.casefold(): value or "" for key, value in attrs}
        self.counts[normalized_tag] = self.counts.get(normalized_tag, 0) + 1
        node = DomNode(normalized_tag, attributes, parent=self.stack[-1], ordinal=self.counts[normalized_tag])
        self.stack[-1].children.append(node)
        self.stack[-1].content.append(node)
        if normalized_tag not in {"br", "hr", "img", "input", "meta", "link", "source", "area", "base", "col", "embed", "param", "track", "wbr"}:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag.casefold():
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == normalized_tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.stack[-1].text_parts.append(data)
            self.stack[-1].content.append(data)


@dataclass(frozen=True)
class LegalBlock:
    kind: str
    locator: str
    text: str
    normalized_hash: str
    attributes: dict[str, str]
    schema: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _content_root(root: DomNode) -> DomNode:
    candidates: list[DomNode] = []
    pending = [root]
    while pending:
        node = pending.pop()
        if node.tag in _CONTENT_ROOTS or node.attributes.get("role") == "main":
            candidates.append(node)
        pending.extend(node.children)
    return max(candidates, key=lambda node: len(node.text), default=root)


def _schema_for(node: DomNode, kind: str) -> str:
    classes = f"{node.attributes.get('class', '')} {node.attributes.get('id', '')}"
    if kind == "footnote" or _FOOTNOTE_RE.search(classes):
        return "footnote"
    if node.tag in {"td", "th"}:
        return "table-cell"
    if node.tag == "li":
        return "ordered-list" if node.parent and node.parent.tag == "ol" else "unordered-list"
    if kind == "heading" and _SPECIAL_HEADING_RE.search(node.text):
        return "schedule-annex-form"
    return kind


def _kind_for(node: DomNode) -> str | None:
    classes = f"{node.attributes.get('class', '')} {node.attributes.get('id', '')}"
    if _FOOTNOTE_RE.search(classes) or node.attributes.get("role") == "note":
        return "footnote"
    if node.tag in _HEADING_TAGS:
        return "heading"
    if node.tag in _BLOCK_TAGS:
        return "table_cell" if node.tag in {"td", "th"} else node.tag
    return None


def extract_legal_blocks(source_html: str) -> list[LegalBlock]:
    parser = _DomParser()
    parser.feed(source_html)
    root = _content_root(parser.root)
    blocks: list[LegalBlock] = []

    pending = [root]
    while pending:
        node = pending.pop()
        if node.tag in _SKIP_TAGS:
            continue
        kind = _kind_for(node)
        text = node.direct_text if node.tag == "li" else node.text
        if kind and text:
            normalized = normalize_block_text(text)
            digest = hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()
            blocks.append(
                LegalBlock(
                    kind=kind,
                    locator=node.path(),
                    text=normalized,
                    normalized_hash=digest,
                    attributes=dict(node.attributes),
                    schema=_schema_for(node, kind),
                )
            )
        pending.extend(reversed(node.children))
    return blocks


def compare_legal_blocks(expected: list[LegalBlock], observed: list[LegalBlock]) -> dict[str, Any]:
    """Compare inventories without allowing duplicate text to hide omissions."""
    expected_counts: dict[tuple[str, str], int] = {}
    observed_counts: dict[tuple[str, str], int] = {}
    for block in expected:
        key = (block.kind, block.normalized_hash)
        expected_counts[key] = expected_counts.get(key, 0) + 1
    for block in observed:
        key = (block.kind, block.normalized_hash)
        observed_counts[key] = observed_counts.get(key, 0) + 1

    missing: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    matched_counts = {
        key: min(expected_count, observed_counts.get(key, 0))
        for key, expected_count in expected_counts.items()
    }
    for block in expected:
        key = (block.kind, block.normalized_hash)
        if matched_counts.get(key, 0) > 0:
            matched_counts[key] -= 1
        else:
            missing.append(block.to_dict())
    remaining_observed = dict(observed_counts)
    for key, matched_count in {
        key: min(expected_counts.get(key, 0), observed_count)
        for key, observed_count in observed_counts.items()
    }.items():
        remaining_observed[key] -= matched_count
    for block in observed:
        key = (block.kind, block.normalized_hash)
        if remaining_observed.get(key, 0) > 0:
            remaining_observed[key] -= 1
            unexpected.append(block.to_dict())

    return {
        "expected_count": len(expected),
        "observed_count": len(observed),
        "matched_count": len(expected) - len(missing),
        "missing_blocks": missing,
        "unexpected_blocks": unexpected,
        "ambiguous_hashes": sorted(
            f"{kind}:{digest}" for (kind, digest), count in observed_counts.items() if count > 1
        ),
        "exact_match": not missing and not unexpected,
    }


def build_schema_census(source_html: str) -> dict[str, Any]:
    blocks = extract_legal_blocks(source_html)
    counts: dict[str, int] = {}
    schemas: dict[str, int] = {}
    for block in blocks:
        counts[block.kind] = counts.get(block.kind, 0) + 1
        schemas[block.schema] = schemas.get(block.schema, 0) + 1
    return {
        "block_count": len(blocks),
        "kind_counts": dict(sorted(counts.items())),
        "schema_counts": dict(sorted(schemas.items())),
        "blocks": [block.to_dict() for block in blocks],
    }


def capture_html_snapshot(session: Any, url: str, *, timeout: int = 30) -> dict[str, Any]:
    """Fetch raw HTML and return immutable response metadata plus oracle census."""
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    body = response.content
    if isinstance(body, str):
        body = body.encode("utf-8")
    source_html = body.decode(response.encoding or "utf-8", errors="replace")
    return {
        "requested_url": url,
        "final_url": str(response.url),
        "redirect_count": len(getattr(response, "history", [])),
        "status_code": int(response.status_code),
        "retrieved_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "content_sha256": hashlib.sha256(body).hexdigest(),
        "content_type": str(response.headers.get("Content-Type", "")),
        "oracle_version": ORACLE_VERSION,
        "schema_census": build_schema_census(source_html),
        "html": source_html,
    }


def write_html_snapshot(snapshot: dict[str, Any], path: Path) -> None:
    """Write one raw response atomically so interrupted crawls never leave JSON half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)