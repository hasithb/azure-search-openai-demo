import json
import sys
from pathlib import Path

from scripts.run_v4_browser_shards import run_shards


def write_fake_shard(path: Path) -> None:
    path.write_text(
        """
import json
import pathlib
import sys

shard = sys.argv[sys.argv.index('--shard-index') + 1]
output_dir = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])
if shard == '1':
    print('case-0042 failed: synthetic assertion', flush=True)
    raise SystemExit(7)
for name in (f'v4_browser_shard_{shard}.json', f'highlight_gate_shard_{shard}.json'):
    (output_dir / name).write_text(json.dumps({'status': 'PASS'}))
""",
        encoding="utf-8",
    )


def test_run_shards_records_exact_failure_and_log_tail(tmp_path: Path):
    fake_shard = tmp_path / "fake_shard.py"
    write_fake_shard(fake_shard)
    summary_path = tmp_path / "summary.json"

    result = run_shards(
        command=[sys.executable, str(fake_shard), "--output-dir", str(tmp_path / "reports")],
        shard_count=2,
        log_dir=tmp_path / "logs",
        coverage_dir=tmp_path / "reports",
        gate_report_dir=tmp_path / "reports",
        diagnostics_dir=tmp_path / "diagnostics",
        summary_path=summary_path,
        timeout_seconds=5,
        grace_seconds=1,
        tail_lines=10,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result == 1
    assert summary["status"] == "FAIL"
    assert summary["shards"][0]["return_code"] == 0
    assert summary["shards"][0]["coverage_status"] == "PASS"
    assert summary["shards"][1]["return_code"] == 7
    assert "case-0042 failed" in summary["shards"][1]["log_tail"][-1]


def test_run_shards_rejects_invalid_count(tmp_path: Path):
    try:
        run_shards(
            command=[sys.executable, "-c", "pass"],
            shard_count=0,
            log_dir=tmp_path / "logs",
            coverage_dir=tmp_path / "reports",
            gate_report_dir=tmp_path / "reports",
            diagnostics_dir=None,
            summary_path=tmp_path / "summary.json",
            timeout_seconds=1,
            grace_seconds=1,
            tail_lines=10,
        )
    except ValueError as error:
        assert "between 1 and 32" in str(error)
    else:
        raise AssertionError("invalid shard count was accepted")