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


def _git_status_entries(repository: Path) -> list[tuple[str, tuple[str, ...]]]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain=v1", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseSafetyError("Git command failed: status --porcelain=v1 -z") from error

    raw_entries = result.stdout.split(b"\0")
    entries: list[tuple[str, tuple[str, ...]]] = []
    index = 0
    while index < len(raw_entries) - 1:
        entry = raw_entries[index]
        index += 1
        if not entry:
            continue
        status = entry[:2].decode("ascii")
        path = entry[3:].decode("utf-8", errors="surrogateescape")
        paths = [path]
        if status[0] in {"R", "C"} or status[1] in {"R", "C"}:
            if index >= len(raw_entries) - 1:
                raise ReleaseSafetyError("Git status returned an incomplete rename entry")
            paths.append(raw_entries[index].decode("utf-8", errors="surrogateescape"))
            index += 1
        entries.append((status, tuple(paths)))
    return entries


def _path_is_allowed(path: str, allowed_dirty_prefixes: tuple[str, ...]) -> bool:
    return any(
        path == prefix or (prefix.endswith("/") and path.startswith(prefix))
        for prefix in allowed_dirty_prefixes
    )


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
    allow_detached: bool = False,
) -> dict[str, Any]:
    expected_sha = expected_sha.strip().lower()
    if not GIT_SHA.fullmatch(expected_sha):
        raise ReleaseSafetyError("Expected Git SHA must be a 40-character hexadecimal commit")
    if not RELEASE_ID.fullmatch(release_id.strip()):
        raise ReleaseSafetyError(f"Release id has an invalid format: {release_id!r}")
    actual_sha = _git(repository, "rev-parse", "HEAD").lower()
    if actual_sha != expected_sha:
        raise ReleaseSafetyError(f"Checked-out Git SHA does not match expected SHA: {actual_sha} != {expected_sha}")
    symbolic_ref = subprocess.run(
        ["git", "-C", str(repository), "symbolic-ref", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if symbolic_ref.returncode == 0:
        actual_ref = symbolic_ref.stdout.strip()
    elif allow_detached:
        actual_ref = "DETACHED"
    else:
        raise ReleaseSafetyError("Checked-out Git HEAD is detached; pass allow_detached=True for immutable CI checkouts")
    if expected_ref.strip() and actual_ref != expected_ref.strip() and actual_ref != "DETACHED":
        raise ReleaseSafetyError(f"Checked-out Git ref does not match expected ref: {actual_ref} != {expected_ref}")
    status_entries = _git_status_entries(repository)
    unexpected_entries = [
        {"status": status, "paths": list(paths)}
        for status, paths in status_entries
        if not all(_path_is_allowed(path, allowed_dirty_prefixes) for path in paths)
    ]
    if require_clean and unexpected_entries:
        raise ReleaseSafetyError(f"Git worktree is not clean: {json.dumps(unexpected_entries, sort_keys=True)}")
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
        "clean_tree": not bool(unexpected_entries),
        "ignored_generated_changes": bool(status_entries) and not bool(unexpected_entries),
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
    parser.add_argument("--allow-detached", action="store_true")
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
            allow_detached=args.allow_detached,
        )
    except ReleaseSafetyError as error:
        result = {"schema_version": 1, "status": "FAIL", "read_only": True, "error": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        args.output.write_text(payload, encoding="utf-8")
        print(payload, end="", flush=True)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())