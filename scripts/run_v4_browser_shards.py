"""Run v4 browser shards with fail-closed process accounting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_log_tail(path: Path, line_count: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]


def report_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    return str(payload.get("status") or "") or "INVALID"


def terminate_process(process: subprocess.Popen[str], grace_seconds: float) -> str:
    if process.poll() is not None:
        return "exited"
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return "terminated"
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return "killed"


def run_shards(
    *,
    command: list[str],
    shard_count: int,
    log_dir: Path,
    coverage_dir: Path,
    gate_report_dir: Path,
    diagnostics_dir: Path | None,
    summary_path: Path,
    timeout_seconds: float,
    grace_seconds: float,
    tail_lines: int,
) -> int:
    if shard_count < 1 or shard_count > 32:
        raise ValueError("shard-count must be between 1 and 32")

    processes: list[dict[str, Any]] = []
    started_at = time.monotonic()
    coverage_dir.mkdir(parents=True, exist_ok=True)
    gate_report_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for shard_index in range(shard_count):
        log_path = log_dir / f"highlight_gate_shard_{shard_index}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        shard_command = [
            *command,
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
            "--exhaustive-coverage-input",
            str(coverage_dir / f"v4_browser_shard_{shard_index}.json"),
            "--output",
            str(gate_report_dir / f"highlight_gate_shard_{shard_index}.json"),
        ]
        if diagnostics_dir is not None:
            shard_command.extend(["--diagnostics-dir", str(diagnostics_dir / f"shard-{shard_index}")])
        process = subprocess.Popen(shard_command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        processes.append(
            {
                "shard_index": shard_index,
                "pid": process.pid,
                "command": shard_command,
                "process": process,
                "log_file": log_file,
                "log_path": str(log_path),
                "started_at": time.time(),
                "coverage_report": str(coverage_dir / f"v4_browser_shard_{shard_index}.json"),
                "gate_report": str(gate_report_dir / f"highlight_gate_shard_{shard_index}.json"),
            }
        )

    deadline = started_at + timeout_seconds
    while any(item["process"].poll() is None for item in processes):
        if time.monotonic() >= deadline:
            for item in processes:
                item["timeout"] = True
                item["termination"] = terminate_process(item["process"], grace_seconds)
        else:
            time.sleep(0.25)

    results: list[dict[str, Any]] = []
    for item in processes:
        process = item["process"]
        item["log_file"].close()
        return_code = process.returncode
        coverage_path = Path(item["coverage_report"])
        gate_path = Path(item["gate_report"])
        result = {
            "shard_index": item["shard_index"],
            "pid": item["pid"],
            "command": item["command"],
            "started_at": item["started_at"],
            "finished_at": time.time(),
            "elapsed_seconds": round(time.time() - item["started_at"], 3),
            "return_code": return_code,
            "timed_out": bool(item.get("timeout", False)),
            "termination": item.get("termination"),
            "log_path": item["log_path"],
            "coverage_report": item["coverage_report"],
            "gate_report": item["gate_report"],
            "coverage_status": report_status(coverage_path),
            "gate_status": report_status(gate_path),
            "log_tail": read_log_tail(Path(item["log_path"]), tail_lines),
        }
        results.append(result)
        print(
            f"Browser shard {item['shard_index']}: return_code={return_code} "
            f"timed_out={result['timed_out']} elapsed_seconds={result['elapsed_seconds']} "
            f"coverage_status={result['coverage_status']} gate_status={result['gate_status']}"
        )
        if return_code != 0 or result["timed_out"] or result["coverage_status"] != "PASS" or result["gate_status"] != "PASS":
            print(f"Browser shard {item['shard_index']} log tail:")
            print("\n".join(result["log_tail"]))

    summary = {
        "schema_version": 1,
        "status": "PASS" if all(
            result["return_code"] == 0
            and not result["timed_out"]
            and result["coverage_status"] == "PASS"
            and result["gate_status"] == "PASS"
            for result in results
        ) else "FAIL",
        "shard_count": shard_count,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "shards": results,
    }
    atomic_write_json(summary_path, summary)
    return 0 if summary["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20_700)
    parser.add_argument("--grace-seconds", type=float, default=30)
    parser.add_argument("--tail-lines", type=int, default=100)
    parser.add_argument("--log-dir", type=Path, default=Path("reports"))
    parser.add_argument("--coverage-dir", type=Path, default=Path("reports"))
    parser.add_argument("--gate-report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--summary-path", type=Path, default=Path("reports/browser_shard_process_summary.json"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a shard command is required")
    try:
        return run_shards(
            command=command,
            shard_count=args.shard_count,
            log_dir=args.log_dir,
            coverage_dir=args.coverage_dir,
            gate_report_dir=args.gate_report_dir,
            diagnostics_dir=args.diagnostics_dir,
            summary_path=args.summary_path,
            timeout_seconds=args.timeout_seconds,
            grace_seconds=args.grace_seconds,
            tail_lines=args.tail_lines,
        )
    except (OSError, ValueError) as error:
        print(f"Browser shard orchestration failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())