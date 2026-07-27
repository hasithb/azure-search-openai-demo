import hashlib
import subprocess

import pytest

from scripts.validate_v4_release_safety import ReleaseSafetyError, validate_release_safety


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