"""Validate immutable Git and release inputs before building v4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


class ReleaseSafetyError(ValueError):
    """Raised when release identity is not safe to evidence or promote."""


GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
RELEASE_ID = re.compile(r"^(?:[0-9]+|20[0-9]{6}-r[1-9][0-9]*)$")


def _git(repository: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseSafetyError(f"Git command failed: {' '.join(args)}") from error
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_release_safety(
    repository: Path,
    *,
    expected_sha: str,
    expected_ref: str,
    release_id: str,
    artifact_path: Path,
    expected_artifact_sha256: str,
    require_clean: bool = True,
    allowed_dirty_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    expected_sha = expected_sha.strip().lower()
    if not GIT_SHA.fullmatch(expected_sha):
        raise ReleaseSafetyError("Expected Git SHA must be a 40-character hexadecimal commit")
    if not RELEASE_ID.fullmatch(release_id.strip()):
        raise ReleaseSafetyError(f"Release id has an invalid format: {release_id!r}")
    actual_sha = _git(repository, "rev-parse", "HEAD").lower()
    if actual_sha != expected_sha:
        raise ReleaseSafetyError(f"Checked-out Git SHA does not match expected SHA: {actual_sha} != {expected_sha}")
    actual_ref = _git(repository, "symbolic-ref", "--short", "HEAD")
    if expected_ref.strip() and actual_ref != expected_ref.strip():
        raise ReleaseSafetyError(f"Checked-out Git ref does not match expected ref: {actual_ref} != {expected_ref}")
    status = _git(repository, "status", "--porcelain")
    unexpected_status = "\n".join(
        line for line in status.splitlines() if not any(line[3:].startswith(prefix) for prefix in allowed_dirty_prefixes)
    )
    if require_clean and unexpected_status:
        raise ReleaseSafetyError("Git worktree is not clean")
    if not artifact_path.is_file():
        raise ReleaseSafetyError(f"Release artifact is missing: {artifact_path}")
    actual_artifact_sha256 = sha256_file(artifact_path)
    if actual_artifact_sha256 != expected_artifact_sha256.strip().lower():
        raise ReleaseSafetyError("Release artifact SHA does not match expected artifact identity")
    return {
        "schema_version": 1,
        "status": "PASS",
        "read_only": True,
        "git_sha": actual_sha,
        "git_ref": actual_ref,
        "release_id": release_id.strip(),
        "artifact_path": str(artifact_path),
        "artifact_sha256": actual_artifact_sha256,
        "clean_tree": not bool(unexpected_status),
        "ignored_generated_changes": bool(status) and not bool(unexpected_status),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--allow-dirty-prefix", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_release_safety(
            args.repository,
            expected_sha=args.git_sha,
            expected_ref=args.ref,
            release_id=args.release_id,
            artifact_path=args.artifact,
            expected_artifact_sha256=args.artifact_sha256,
            allowed_dirty_prefixes=tuple(args.allow_dirty_prefix),
        )
    except ReleaseSafetyError as error:
        result = {"schema_version": 1, "status": "FAIL", "read_only": True, "error": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())