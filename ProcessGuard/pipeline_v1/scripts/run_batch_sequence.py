from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROCESS_GUARD_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESS_GUARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_GUARD_ROOT))

from pipeline_v1.output_utils import OUTPUT_ROOT  # noqa: E402


def _resolve_report_path(*, report: Path | None, latest_match: str | None) -> Path | None:
    if report is not None:
        return report
    if not latest_match:
        return None
    candidates = sorted(
        (
            path / "report.json"
            for path in OUTPUT_ROOT.iterdir()
            if path.is_dir() and latest_match in path.name
        ),
        key=lambda item: item.parent.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _process_running(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    ignored = {os.getpid(), os.getppid()}
    pids = [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]
    for pid in pids:
        if pid in ignored:
            continue
        cmd = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not cmd:
            continue
        if "run_batch_sequence.py" in cmd:
            continue
        return True
    return False


def _load_summary(report_path: Path | None) -> dict[str, object]:
    if report_path is None or not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def _summary_line(summary: dict[str, object]) -> str:
    if not summary:
        return "summary=unavailable"
    accepted = int(summary.get("accepted", 0))
    attempted = int(summary.get("attempted", 0))
    rejected = int(summary.get("rejected", 0))
    acceptance_rate = float(summary.get("acceptance_rate", 0.0))
    unmet = len(summary.get("unmet_cells", [])) if isinstance(summary.get("unmet_cells", []), list) else 0
    return (
        f"accepted={accepted} attempted={attempted} rejected={rejected} "
        f"acceptance_rate={acceptance_rate:.3f} unmet_cells={unmet}"
    )


def _run_command(command: str) -> int:
    print(f"[sequence] starting: {command}", flush=True)
    completed = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        cwd=PROCESS_GUARD_ROOT.parent,
    )
    print(f"[sequence] finished rc={completed.returncode}: {command}", flush=True)
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for one batch run to finish, then launch the next batch command(s).")
    parser.add_argument("--wait-process-pattern", type=str, required=True, help="Substring matched by `pgrep -f` for the currently running batch.")
    parser.add_argument("--wait-report", type=Path, default=None, help="Explicit report.json path for the current batch.")
    parser.add_argument("--wait-latest-match", type=str, default=None, help="Resolve the current batch report.json from the newest output directory containing this substring.")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval while waiting for the current batch to finish.")
    parser.add_argument("--allow-unmet-cells", action="store_true", help="Continue even if the finished batch report still contains unmet_cells.")
    parser.add_argument("--dry-run", action="store_true", help="Print the queued commands after the wait step without executing them.")
    parser.add_argument("--command", action="append", required=True, help="Next command to execute. Repeatable for multiple sequential rounds.")
    args = parser.parse_args()

    report_path = _resolve_report_path(report=args.wait_report, latest_match=args.wait_latest_match)
    print(f"[sequence] waiting on process pattern: {args.wait_process_pattern}", flush=True)
    if report_path is not None:
        print(f"[sequence] tracking report: {report_path}", flush=True)

    last_status_line = ""
    while True:
        running = _process_running(args.wait_process_pattern)
        summary = _load_summary(report_path)
        status_line = f"running={running} {_summary_line(summary)}"
        if status_line != last_status_line:
            print(f"[sequence] {status_line}", flush=True)
            last_status_line = status_line
        if not running:
            break
        time.sleep(max(int(args.poll_seconds), 1))

    final_summary = _load_summary(report_path)
    print(f"[sequence] current batch finished: {_summary_line(final_summary)}", flush=True)
    unmet_cells = final_summary.get("unmet_cells", [])
    if not args.allow_unmet_cells and isinstance(unmet_cells, list) and unmet_cells:
        print("[sequence] refusing to start next round because unmet_cells is non-empty", flush=True)
        sys.exit(2)

    if args.dry_run:
        for index, command in enumerate(args.command, start=1):
            print(f"[sequence] dry-run next[{index}]: {command}", flush=True)
        return

    for command in args.command:
        rc = _run_command(command)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
