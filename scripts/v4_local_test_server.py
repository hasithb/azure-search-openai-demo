"""Deterministic HTTP fixture origin for local v4 application-gate checks."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PROVENANCE: dict[str, str] = {
    "release_id": "local-r1",
    "git_sha": "local-git",
    "deployment_id": "local-deployment",
    "artifact_sha256": "a" * 64,
    "search_snapshot_sha256": "b" * 64,
    "search_service": "local-search",
    "search_index": "legal-court-rag-v4-staging-local-r1",
    "knowledge_base": "legal-court-rag-v4-staging-local-r1-agent-upgrade",
}

CPR_CATEGORY = "Civil Procedure Rules and Practice Directions"
CATEGORIES = [
    {"key": CPR_CATEGORY, "display_name": CPR_CATEGORY},
    {"key": "Commercial Court", "display_name": "Commercial Court"},
    {"key": "Chancery Division", "display_name": "Chancery Division"},
]


def source(category: str, sourcepage: str, subsection_id: str, sourcefile: str = "cpr.pdf") -> dict[str, str]:
    return {
        "category": category,
        "sourcefile": sourcefile,
        "sourcepage": sourcepage,
        "subsection_id": subsection_id,
        "title": sourcepage,
    }


class FixtureHandler(BaseHTTPRequestHandler):
    server_version = "v4-local-fixture/1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/categories":
            self.write_json({"categories": CATEGORIES})
        elif self.path == "/api/provenance":
            self.write_json({"schema_version": 1, **PROVENANCE})
        elif self.path == "/api/acl":
            self.write_json({"authorized_count": 2, "elevated_count": 3, "authorized_total": 2, "elevated_total": 3})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/chat":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        context = payload.get("context", {}) if isinstance(payload, dict) else {}
        overrides = context.get("overrides", {}) if isinstance(context, dict) else {}
        category = str(overrides.get("include_category") or "")
        question = str((payload.get("messages") or [{}])[-1].get("content") or "").casefold()
        if category == "Commercial Court" or "commercial court" in question:
            selected = source(category, "Commercial Court Guide / Case management", "Case management")
            selected["category"] = "Commercial Court"
            answer = "The Commercial Court uses structured case management for conferences [1]."
        elif category == "Chancery Division" or "chancery" in question:
            selected = source(category, "Chancery Guide / Case management", "Case management")
            selected["category"] = "Chancery Division"
            answer = "The Chancery Division handles case management conferences [1]."
        elif "part 52" in question or "appeal" in question:
            selected = source(CPR_CATEGORY, "CPR Part 52", "Part 52")
            answer = "CPR Part 52 governs appeals and appeal time limits [1]."
        elif "31.16" in question or "pre-action" in question:
            selected = source(CPR_CATEGORY, "CPR Part 31", "31.16")
            answer = "CPR 31.16 provides for pre-action disclosure [1]."
        elif "part 31" in question or "disclosure" in question:
            selected = source(CPR_CATEGORY, "CPR Part 31", "Part 31")
            answer = "CPR Part 31 describes the disclosure process [1]."
        else:
            selected = source(CPR_CATEGORY, "CPR Part 1", "Part 1")
            answer = "CPR Part 1 states the overriding objective and case management principles [1]."
        if category and selected["category"] != category:
            selected["category"] = category
        self.write_json({"output_text": answer, "message": {"content": answer}, "context": {"data_points": {"text": [selected]}}})

    def write_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def start_fixture_server() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    return server, f"http://127.0.0.1:{server.server_port}"