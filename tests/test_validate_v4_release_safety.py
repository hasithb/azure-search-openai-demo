import hashlib
import subprocess
import sys

import pytest

from scripts.validate_v4_release_safety import (
    ReleaseSafetyError,
    validate_release_safety,
)


def test_release_safety_accepts_matching_clean_repository(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    artifact = repository / "artifact.jsonl"
    artifact.write_text('{"id":"one"}\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "artifact.jsonl"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-m", "artifact"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()

    result = validate_release_safety(
        repository,
        expected_sha=sha,
        expected_ref="main",
        release_id="20260726-r1",
        artifact_path=artifact,
        expected_artifact_sha256=artifact_sha,
    )

    assert result["status"] == "PASS"
    assert result["read_only"] is True


def test_release_safety_accepts_matching_detached_repository_when_explicit(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    artifact = repository / "artifact.jsonl"
    artifact.write_text('{"id":"one"}\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "artifact.jsonl"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-m", "artifact"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repository), "checkout", "--detach", sha], check=True, capture_output=True)

    result = validate_release_safety(
        repository,
        expected_sha=sha,
        expected_ref="main",
        release_id="20260726-r1",
        artifact_path=artifact,
        expected_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        allow_detached=True,
    )

    assert result["git_ref"] == "DETACHED"


def test_release_safety_rejects_detached_repository_without_explicit_opt_in(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    artifact = repository / "artifact.jsonl"
    artifact.write_text("artifact\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "artifact.jsonl"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-m", "artifact"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repository), "checkout", "--detach", sha], check=True, capture_output=True)

    with pytest.raises(ReleaseSafetyError, match="detached"):
        validate_release_safety(
            repository,
            expected_sha=sha,
            expected_ref="main",
            release_id="20260726-r1",
            artifact_path=artifact,
            expected_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("field", ["sha", "ref", "release"])
def test_release_safety_rejects_identity_mismatch(tmp_path, field):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    artifact = repository / "artifact.jsonl"
    artifact.write_text("artifact\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "artifact.jsonl"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-m", "artifact"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(ReleaseSafetyError):
        validate_release_safety(
            repository,
            expected_sha="0" * 40 if field == "sha" else sha,
            expected_ref="release" if field == "ref" else "main",
            release_id="invalid" if field == "release" else "20260726-r1",
            artifact_path=artifact,
            expected_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )


def _committed_repository(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "-C", str(repository), "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-m", "tracked"], check=True, capture_output=True)
    sha = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    return repository, sha


def _validate(repository, sha, artifact, prefixes):
    return validate_release_safety(
        repository,
        expected_sha=sha,
        expected_ref="main",
        release_id="20260726-r1",
        artifact_path=artifact,
        expected_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        allowed_dirty_prefixes=tuple(prefixes),
    )


def _commit_artifact(repository, artifact):
    subprocess.run(["git", "-C", str(repository), "add", str(artifact.relative_to(repository))], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-m", "artifact"], check=True, capture_output=True)
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def test_release_safety_accepts_first_unstaged_allowed_entry(tmp_path):
    repository, sha = _committed_repository(tmp_path)
    artifact = repository / "reports" / "release" / "artifact.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    sha = _commit_artifact(repository, artifact)
    artifact.write_text("changed\n", encoding="utf-8")

    result = _validate(repository, sha, artifact, ["reports/release/"])
    assert result["ignored_generated_changes"] is True


def test_release_safety_accepts_nested_untracked_allowed_entry(tmp_path):
    repository, sha = _committed_repository(tmp_path)
    artifact = repository / "artifact.jsonl"
    artifact.write_text("artifact\n", encoding="utf-8")
    sha = _commit_artifact(repository, artifact)
    generated = repository / "reports" / "nested" / "release.json"
    generated.parent.mkdir(parents=True)
    generated.write_text("generated\n", encoding="utf-8")

    result = _validate(repository, sha, artifact, ["reports/"])
    assert result["ignored_generated_changes"] is True


def test_release_safety_does_not_treat_file_suffix_as_allowed(tmp_path):
    repository, sha = _committed_repository(tmp_path)
    artifact = repository / "reports" / "release.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    sha = _commit_artifact(repository, artifact)
    unexpected = repository / "reports" / "release.json.backup"
    unexpected.write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ReleaseSafetyError, match="reports/release.json.backup"):
        _validate(repository, sha, artifact, ["reports/release.json"])


def test_release_safety_requires_both_paths_of_rename_to_be_allowed(tmp_path):
    repository, sha = _committed_repository(tmp_path)
    artifact = repository / "reports" / "release" / "artifact.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    sha = _commit_artifact(repository, artifact)
    subprocess.run(
        ["git", "-C", str(repository), "mv", "tracked.txt", "reports/release/renamed.txt"],
        check=True,
    )

    with pytest.raises(ReleaseSafetyError, match="tracked.txt"):
        _validate(repository, sha, artifact, ["reports/release/"])


def test_release_safety_cli_prints_and_writes_structured_failure(tmp_path):
    repository, sha = _committed_repository(tmp_path)
    artifact = repository / "artifact.jsonl"
    artifact.write_text("artifact\n", encoding="utf-8")
    sha = _commit_artifact(repository, artifact)
    output = tmp_path / "reports" / "v4_release_safety.json"
    command = [
        sys.executable,
        "scripts/validate_v4_release_safety.py",
        "--repository",
        str(repository),
        "--git-sha",
        sha,
        "--ref",
        "main",
        "--release-id",
        "20260726-r1",
        "--artifact",
        str(artifact),
        "--artifact-sha256",
        "0" * 64,
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 1
    assert '"status": "FAIL"' in completed.stdout
    assert output.exists()
    assert output.read_text(encoding="utf-8") == completed.stdout


@pytest.mark.parametrize("failure", ["missing", "hash"])
def test_release_safety_rejects_invalid_artifact_identity(tmp_path, failure):
    repository, sha = _committed_repository(tmp_path)
    artifact = repository / "artifact.jsonl"
    artifact.write_text("artifact\n", encoding="utf-8")
    sha = _commit_artifact(repository, artifact)
    artifact_path = artifact if failure == "hash" else repository / "missing.jsonl"
    expected_sha = "0" * 64 if failure == "hash" else hashlib.sha256(artifact.read_bytes()).hexdigest()

    with pytest.raises(ReleaseSafetyError, match="missing|SHA"):
        validate_release_safety(
            repository,
            expected_sha=sha,
            expected_ref="main",
            release_id="20260726-r1",
            artifact_path=artifact_path,
            expected_artifact_sha256=expected_sha,
        )