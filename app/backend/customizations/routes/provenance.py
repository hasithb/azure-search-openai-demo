"""Candidate release provenance endpoint."""

from __future__ import annotations

import os

from quart import Blueprint, jsonify

PROVENANCE_ENV_FIELDS = {
    "release_id": "V4_RELEASE_ID",
    "git_sha": "GIT_SHA",
    "deployment_id": "DEPLOYMENT_ID",
    "artifact_sha256": "V4_ARTIFACT_SHA256",
    "search_snapshot_sha256": "V4_SEARCH_SNAPSHOT_SHA256",
    "search_service": "AZURE_SEARCH_SERVICE",
    "search_index": "AZURE_SEARCH_INDEX",
    "knowledge_base": "AZURE_SEARCH_KNOWLEDGEBASE_NAME",
}

provenance_bp = Blueprint("provenance", __name__, url_prefix="/api")


@provenance_bp.route("/provenance", methods=["GET"])
async def get_provenance():
    values = {field: os.getenv(variable, "").strip() for field, variable in PROVENANCE_ENV_FIELDS.items()}
    missing = [field for field, value in values.items() if not value]
    if missing:
        return jsonify({"error": "Candidate provenance is incomplete", "missing": missing}), 503
    return jsonify({"schema_version": 1, **values}), 200