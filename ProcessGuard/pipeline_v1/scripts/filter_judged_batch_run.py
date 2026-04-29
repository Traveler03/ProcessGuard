from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PROCESS_GUARD_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESS_GUARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_GUARD_ROOT))

from pipeline_v1.output_utils import create_run_dir, output_file_in_run_dir  # noqa: E402
from pipeline_v1.risk_taxonomy import canonical_risk_type_and_subtype  # noqa: E402
from pipeline_v1.scripts.generate_batch import _build_report  # noqa: E402
from pipeline_v1.scripts.repair_batch_run import (  # noqa: E402
    _build_cells_metadata,
    _build_review_payloads,
    _load_json,
    _normalize_run_dir,
    _ordered_unique,
    _record_cell_id,
    _record_risk_type,
)


def _judge_score(record: dict[str, Any]) -> int:
    try:
        return int(record.get("quality_gate", {}).get("llm_judge", {}).get("score", 0))
    except Exception:
        return 0


def _canonical_record(record: dict[str, Any]) -> dict[str, Any]:
    execution_plan = record.get("execution_plan", {})
    label_combo = execution_plan.get("label_combo", {})
    risk_type, _ = canonical_risk_type_and_subtype(execution_plan.get("risk_setup", {}))
    validation = record.get("validation", {})
    evidence_indexes = validation.get("evidence_indexes", {}) if isinstance(validation, dict) else {}
    trajectory = record.get("trajectory", {})
    messages = trajectory.get("messages", []) if isinstance(trajectory, dict) else []
    content = [
        [
            {
                "role": str(message.get("role", "")).strip(),
                "content": str(message.get("content", "")),
            }
        ]
        for message in messages
        if isinstance(message, dict)
    ]
    return {
        "conv_id": _record_cell_id(record),
        "scenario_name": str(execution_plan.get("scenario_id", "")).strip(),
        "risk_source_category": risk_type,
        "binary_label": str(label_combo.get("binary_label", "")).strip(),
        "action_state": str(label_combo.get("action_state", "")).strip(),
        "realization_state": str(label_combo.get("realization_state", "")).strip(),
        "tool_used": deepcopy(execution_plan.get("tool_subset", [])),
        "content": content,
        "action_state_evidence_indexes": [
            int(item)
            for item in evidence_indexes.get("action_state_evidence_indexes", [])
            if isinstance(item, int) or str(item).strip().isdigit()
        ],
        "realization_state_evidence_indexes": [
            int(item)
            for item in evidence_indexes.get("realization_state_evidence_indexes", [])
            if isinstance(item, int) or str(item).strip().isdigit()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a judged pipeline_v1 batch run by LLM-judge score and export canonical eval JSON."
    )
    parser.add_argument("--run", type=Path, required=True, help="Input run directory or a file inside it.")
    parser.add_argument("--min-judge-score", type=int, required=True, help="Keep accepted records whose judge score is at least this value.")
    parser.add_argument("--seed", type=int, default=0, help="Optional seed used only for the filtered run directory name.")
    parser.add_argument("--run-name", type=str, default="", help="Optional output run suffix.")
    args = parser.parse_args()

    source_run_dir = _normalize_run_dir(args.run)
    batch_path = source_run_dir / "batch.json"
    report_path = source_run_dir / "report.json"
    config_snapshot = source_run_dir / "config.snapshot.yaml"

    records = _load_json(batch_path)
    report = _load_json(report_path)
    if not isinstance(records, list):
        raise ValueError(f"{batch_path} does not contain a JSON list")

    kept_records: list[dict[str, Any]] = []
    dropped_records: list[dict[str, Any]] = []
    score_distribution: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        score = _judge_score(record)
        score_distribution[str(score)] += 1
        if score >= args.min_judge_score:
            kept_records.append(deepcopy(record))
        else:
            dropped_records.append(deepcopy(record))

    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    attempted_count = int(summary.get("attempted", 0))
    original_rejected = int(summary.get("rejected", 0))
    attempts_by_cell = Counter({str(key): int(value) for key, value in dict(summary.get("attempts_by_cell", {})).items()})
    rejection_reason_counter = Counter(
        {str(key): int(value) for key, value in dict(summary.get("rejection_reason_distribution", {})).items()}
    )
    dropped_by_score = Counter(str(_judge_score(record)) for record in dropped_records)
    for score, count in dropped_by_score.items():
        rejection_reason_counter[f"postfilter_judge_score_below_{args.min_judge_score}:score_{score}"] += count

    accepted_by_cell = Counter(_record_cell_id(record) for record in kept_records)
    last_failure_reasons_by_cell: dict[str, list[str]] = {}
    for item in report.get("failed_cells", []):
        if not isinstance(item, dict):
            continue
        cell_id = str(item.get("cell_id", "")).strip()
        reasons = [str(reason).strip() for reason in item.get("last_failure_reasons", []) if str(reason).strip()]
        if cell_id and reasons:
            last_failure_reasons_by_cell[cell_id] = reasons
    for record in dropped_records:
        cell_id = _record_cell_id(record)
        last_failure_reasons_by_cell[cell_id] = [f"judge_score_below_keep_threshold:{_judge_score(record)}"]

    cells = _build_cells_metadata(records=records, reports=[report])
    filtered_report = _build_report(
        cells=cells,
        accepted_records=kept_records,
        attempted_count=attempted_count,
        rejected_count=original_rejected + len(dropped_records),
        rejection_reason_counter=rejection_reason_counter,
        accepted_by_cell=accepted_by_cell,
        attempts_by_cell=attempts_by_cell,
        last_failure_reasons_by_cell=last_failure_reasons_by_cell,
        target_per_cell=summary.get("target_per_cell"),
        max_attempts_per_cell=summary.get("max_attempts_per_cell"),
    )

    risk_types = _ordered_unique([_record_risk_type(record) for record in kept_records if _record_risk_type(record)])
    run_name = args.run_name.strip() or f"keep_judge_score_ge_{args.min_judge_score}"
    run_dir = create_run_dir(
        kind="batch",
        scenario="multi",
        label_combo="multi",
        seed=args.seed,
        run_name=run_name,
    )
    if config_snapshot.exists():
        shutil.copyfile(config_snapshot, run_dir / "config.snapshot.yaml")

    filtered_batch_path = output_file_in_run_dir(run_dir, "", default_name="batch.json")
    filtered_report_path = run_dir / "report.json"
    review_success_path = run_dir / "review_success_by_risk_type.json"
    review_unsafe_path = run_dir / "review_unsafe_by_risk_type.json"
    filter_summary_path = run_dir / "filter_summary.json"
    canonical_path = run_dir / "canonical_eval.json"

    success_review, unsafe_review = _build_review_payloads(
        merged_records=kept_records,
        run_dir=run_dir,
        batch_path=filtered_batch_path,
        risk_types=risk_types,
    )
    canonical_records = [_canonical_record(record) for record in kept_records]

    filter_summary = {
        "source_run_dir": str(source_run_dir),
        "min_judge_score": int(args.min_judge_score),
        "input_record_count": len(records),
        "kept_record_count": len(kept_records),
        "dropped_record_count": len(dropped_records),
        "score_distribution": dict(score_distribution),
        "dropped_score_distribution": dict(dropped_by_score),
        "kept_cell_ids": [_record_cell_id(record) for record in kept_records],
        "dropped_cell_ids": [_record_cell_id(record) for record in dropped_records],
        "final_unmet_count": len(filtered_report.get("summary", {}).get("unmet_cells", [])),
        "canonical_eval_path": str(canonical_path),
    }

    filtered_batch_path.write_text(json.dumps(kept_records, ensure_ascii=False, indent=2), encoding="utf-8")
    filtered_report_path.write_text(json.dumps(filtered_report, ensure_ascii=False, indent=2), encoding="utf-8")
    review_success_path.write_text(json.dumps(success_review, ensure_ascii=False, indent=2), encoding="utf-8")
    review_unsafe_path.write_text(json.dumps(unsafe_review, ensure_ascii=False, indent=2), encoding="utf-8")
    filter_summary_path.write_text(json.dumps(filter_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    canonical_path.write_text(json.dumps(canonical_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[filter] run_dir={run_dir}", flush=True)
    print(
        "[filter] kept "
        f"accepted={filtered_report['summary']['accepted']} attempted={filtered_report['summary']['attempted']} "
        f"rejected={filtered_report['summary']['rejected']} unmet={len(filtered_report['summary']['unmet_cells'])}",
        flush=True,
    )
    print(f"[filter] canonical_eval={canonical_path}", flush=True)
    print(str(filtered_batch_path))


if __name__ == "__main__":
    main()
