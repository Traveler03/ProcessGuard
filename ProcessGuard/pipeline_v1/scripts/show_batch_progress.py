from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROCESS_GUARD_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESS_GUARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_GUARD_ROOT))

from pipeline_v1.output_utils import OUTPUT_ROOT  # noqa: E402


def _resolve_report_path(*, report: Path | None, latest_match: str | None) -> Path:
    if report is not None:
        return report
    if latest_match:
        candidates = sorted(
            (
                path / "report.json"
                for path in OUTPUT_ROOT.iterdir()
                if path.is_dir() and latest_match in path.name and (path / "report.json").exists()
            ),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"no report.json found under {OUTPUT_ROOT} matching {latest_match!r}")
    raise ValueError("either --report or --latest-match is required")


def _print_counter_block(title: str, counter: dict[str, Any], *, limit: int) -> None:
    print(f"{title}:")
    if not counter:
        print("  (empty)")
        return
    rows = sorted(
        ((str(key), int(value)) for key, value in counter.items()),
        key=lambda item: (-item[1], item[0]),
    )
    for key, value in rows[:limit]:
        print(f"  {value:>4}  {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show progress for one pipeline_v1 batch run.")
    parser.add_argument("--report", type=Path, default=None, help="Explicit report.json path.")
    parser.add_argument("--latest-match", type=str, default=None, help="Pick the newest output run whose directory name contains this substring.")
    parser.add_argument("--show-cells", type=int, default=8, help="How many cell rows to print from accepted/attempted counters.")
    args = parser.parse_args()

    report_path = _resolve_report_path(report=args.report, latest_match=args.latest_match)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})

    accepted = int(summary.get("accepted", 0))
    attempted = int(summary.get("attempted", 0))
    rejected = int(summary.get("rejected", 0))
    acceptance_rate = float(summary.get("acceptance_rate", 0.0))
    target_per_cell = summary.get("target_per_cell")
    max_attempts_per_cell = summary.get("max_attempts_per_cell")
    unmet_cells = [str(item) for item in summary.get("unmet_cells", [])]
    failed_cells = payload.get("failed_cells", [])

    print(f"report: {report_path}")
    print(
        "summary: "
        f"accepted={accepted} attempted={attempted} rejected={rejected} "
        f"acceptance_rate={acceptance_rate:.3f} "
        f"target_per_cell={target_per_cell} max_attempts_per_cell={max_attempts_per_cell}"
    )
    _print_counter_block("judge_score_distribution", dict(summary.get("judge_score_distribution", {})), limit=args.show_cells)
    _print_counter_block("accepted_by_cell", dict(summary.get("accepted_by_cell", {})), limit=args.show_cells)
    _print_counter_block("attempts_by_cell", dict(summary.get("attempts_by_cell", {})), limit=args.show_cells)
    _print_counter_block("rejection_reason_distribution", dict(summary.get("rejection_reason_distribution", {})), limit=args.show_cells)
    print("unmet_cells:")
    if unmet_cells:
        for cell in unmet_cells[: args.show_cells]:
            print(f"  {cell}")
    else:
        print("  (none)")
    print("failed_cells:")
    if failed_cells:
        for item in failed_cells[: args.show_cells]:
            cell_id = str(item.get("cell_id", ""))
            attempted = int(item.get("attempted", 0))
            reasons = [str(reason) for reason in item.get("last_failure_reasons", []) if str(reason).strip()]
            print(f"  {cell_id} attempted={attempted}")
            if reasons:
                print(f"    last_failure={reasons[0]}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
