from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import sys
from typing import Any


PROCESS_GUARD_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESS_GUARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_GUARD_ROOT))

from pipeline_v1.config import DEFAULT_CONFIG_PATH  # noqa: E402
from pipeline_v1.judge_adapter import apply_llm_judge  # noqa: E402
from pipeline_v1.output_utils import create_run_dir, output_file_in_run_dir, snapshot_config  # noqa: E402
from pipeline_v1.risk_taxonomy import canonical_risk_type_and_subtype  # noqa: E402
from pipeline_v1.scripts.generate_batch import _build_report, _effective_judge_settings  # noqa: E402


UNSAFE_COMBO_PRIORITY = [
    "unsafe_executed_realized",
    "unsafe_executed_not_realized",
    "unsafe_attempted_realized",
    "unsafe_attempted_not_realized",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_run_dir(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if not candidate.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")
    batch_path = candidate / "batch.json"
    report_path = candidate / "report.json"
    if not batch_path.exists():
        raise FileNotFoundError(f"missing batch.json under {candidate}")
    if not report_path.exists():
        raise FileNotFoundError(f"missing report.json under {candidate}")
    return candidate


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _record_sampling(record: dict[str, Any]) -> dict[str, Any]:
    sampling = record.get("sampling", {})
    if not isinstance(sampling, dict):
        raise ValueError("record is missing sampling metadata")
    return sampling


def _record_cell_id(record: dict[str, Any]) -> str:
    cell_id = str(_record_sampling(record).get("cell_id", "")).strip()
    if not cell_id:
        raise ValueError("record is missing sampling.cell_id")
    return cell_id


def _record_risk_type(record: dict[str, Any]) -> str:
    risk_type, _ = canonical_risk_type_and_subtype(record.get("execution_plan", {}).get("risk_setup", {}))
    return risk_type


def _record_scenario_id(record: dict[str, Any]) -> str:
    return str(record.get("execution_plan", {}).get("scenario_id", "")).strip()


def _record_label_combo(record: dict[str, Any]) -> str:
    return str(record.get("execution_plan", {}).get("label_combo", {}).get("slug", "")).strip()


def _record_binary_label(record: dict[str, Any]) -> str:
    return str(record.get("execution_plan", {}).get("label_combo", {}).get("binary_label", "")).strip()


def _record_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    combo = _record_label_combo(item["record"])
    combo_rank = UNSAFE_COMBO_PRIORITY.index(combo) if combo in UNSAFE_COMBO_PRIORITY else len(UNSAFE_COMBO_PRIORITY)
    return (
        combo_rank,
        int(item["source_batch_index"]),
        str(item["cell_id"]),
    )


def _summarize_record(source_batch_index: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_batch_index": int(source_batch_index),
        "risk_type": _record_risk_type(record),
        "label_combo": _record_label_combo(record),
        "scenario_id": _record_scenario_id(record),
        "cell_id": _record_cell_id(record),
        "record": record,
    }


def _load_run_payload(run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "batch_path": run_dir / "batch.json",
        "report_path": run_dir / "report.json",
        "records": _load_json(run_dir / "batch.json"),
        "report": _load_json(run_dir / "report.json"),
    }


def _merge_records(
    *,
    base_records: list[dict[str, Any]],
    supplement_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    merged_records = [deepcopy(record) for record in base_records]
    existing_by_cell = {
        _record_cell_id(record): index
        for index, record in enumerate(merged_records)
    }
    added_cells: list[str] = []
    skipped_cells: list[str] = []
    for record in supplement_records:
        cell_id = _record_cell_id(record)
        if cell_id in existing_by_cell:
            skipped_cells.append(cell_id)
            continue
        existing_by_cell[cell_id] = len(merged_records)
        merged_records.append(deepcopy(record))
        added_cells.append(cell_id)
    return merged_records, added_cells, skipped_cells


def _normalize_record_risk_fields(record: dict[str, Any]) -> dict[str, Any]:
    execution_plan = record.get("execution_plan", {})
    if isinstance(execution_plan, dict):
        risk_setup = execution_plan.get("risk_setup", {})
        if isinstance(risk_setup, dict):
            risk_setup.pop("risk_subtype", None)
    sampling = record.get("sampling", {})
    if isinstance(sampling, dict):
        sampling.pop("risk_subtype", None)
    return record


def _build_cells_metadata(
    *,
    records: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> list[dict[str, str | None]]:
    cells_by_id: dict[str, dict[str, str | None]] = {}
    for record in records:
        sampling = _record_sampling(record)
        cell_id = _record_cell_id(record)
        sampling_risk_type, _ = canonical_risk_type_and_subtype(
            {
                "risk_type": str(sampling.get("risk_type", "")).strip(),
                "risk_subtype": str(sampling.get("risk_subtype", "")).strip(),
            }
        )
        cells_by_id[cell_id] = {
            "scenario_name": str(sampling.get("scenario_name", "")).strip() or _record_scenario_id(record),
            "label_combo": str(sampling.get("label_combo", "")).strip() or _record_label_combo(record),
            "risk_type": sampling_risk_type or _record_risk_type(record),
            "cell_id": cell_id,
        }
    for report in reports:
        for item in report.get("failed_cells", []):
            if not isinstance(item, dict):
                continue
            cell_id = str(item.get("cell_id", "")).strip()
            if not cell_id or cell_id in cells_by_id:
                continue
            report_risk_type, _ = canonical_risk_type_and_subtype(
                {
                    "risk_type": str(item.get("risk_type", "")).strip(),
                    "risk_subtype": str(item.get("risk_subtype", "")).strip(),
                }
            )
            cells_by_id[cell_id] = {
                "scenario_name": str(item.get("scenario_name", "")).strip(),
                "label_combo": str(item.get("label_combo", "")).strip(),
                "risk_type": report_risk_type,
                "cell_id": cell_id,
            }
    return list(cells_by_id.values())


def _merge_report_counters(reports: list[dict[str, Any]]) -> tuple[int, int, Counter[str], Counter[str], dict[str, list[str]]]:
    attempted_count = 0
    rejected_count = 0
    attempts_by_cell: Counter[str] = Counter()
    rejection_reason_counter: Counter[str] = Counter()
    last_failure_reasons_by_cell: dict[str, list[str]] = {}
    for report in reports:
        summary = report.get("summary", {})
        attempted_count += int(summary.get("attempted", 0))
        rejected_count += int(summary.get("rejected", 0))
        attempts_by_cell.update({str(key): int(value) for key, value in dict(summary.get("attempts_by_cell", {})).items()})
        rejection_reason_counter.update({str(key): int(value) for key, value in dict(summary.get("rejection_reason_distribution", {})).items()})
        for item in report.get("failed_cells", []):
            if not isinstance(item, dict):
                continue
            cell_id = str(item.get("cell_id", "")).strip()
            if not cell_id:
                continue
            reasons = [str(reason).strip() for reason in item.get("last_failure_reasons", []) if str(reason).strip()]
            if reasons:
                last_failure_reasons_by_cell[cell_id] = reasons
    return attempted_count, rejected_count, attempts_by_cell, rejection_reason_counter, last_failure_reasons_by_cell


def _apply_judge_annotations(
    *,
    records: list[dict[str, Any]],
    judge_settings: Any,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    total = len(records)
    for index, record in enumerate(records):
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print(f"[repair] judge_progress {index + 1}/{total}", flush=True)
        reasons = apply_llm_judge(record, judge_settings)
        if not reasons:
            continue
        failures.append(
            {
                "source_batch_index": index,
                "cell_id": _record_cell_id(record),
                "risk_type": _record_risk_type(record),
                "label_combo": _record_label_combo(record),
                "scenario_id": _record_scenario_id(record),
                "reasons": list(reasons),
                "quality_gate": deepcopy(record.get("quality_gate", {})),
            }
        )
    return failures


def _choose_one_per_risk_type(
    *,
    records: list[dict[str, Any]],
    risk_types: list[str],
    unsafe_only: bool,
) -> list[dict[str, Any]]:
    used_scenarios: set[str] = set()
    selected: list[dict[str, Any]] = []
    for risk_type in risk_types:
        candidates = [
            item
            for item in records
            if item["risk_type"] == risk_type and (not unsafe_only or _record_binary_label(item["record"]) == "unsafe")
        ]
        if not candidates:
            continue
        ordered_candidates = sorted(candidates, key=_record_sort_key if unsafe_only else lambda item: (int(item["source_batch_index"]), str(item["cell_id"])))
        preferred = next((item for item in ordered_candidates if item["scenario_id"] not in used_scenarios), ordered_candidates[0])
        used_scenarios.add(preferred["scenario_id"])
        selected.append(preferred)
    return selected


def _build_review_payloads(
    *,
    merged_records: list[dict[str, Any]],
    run_dir: Path,
    batch_path: Path,
    risk_types: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    indexed_records = [
        _summarize_record(index, record)
        for index, record in enumerate(merged_records)
    ]
    success_records = _choose_one_per_risk_type(
        records=indexed_records,
        risk_types=risk_types,
        unsafe_only=False,
    )
    unsafe_records = _choose_one_per_risk_type(
        records=indexed_records,
        risk_types=risk_types,
        unsafe_only=True,
    )
    available_unsafe_counts_by_risk_type: dict[str, dict[str, int]] = {}
    for item in indexed_records:
        if _record_binary_label(item["record"]) != "unsafe":
            continue
        risk_type = item["risk_type"]
        label_combo = item["label_combo"]
        available_unsafe_counts_by_risk_type.setdefault(risk_type, {})
        available_unsafe_counts_by_risk_type[risk_type][label_combo] = available_unsafe_counts_by_risk_type[risk_type].get(label_combo, 0) + 1
    success_payload = {
        "run_dir": str(run_dir),
        "batch_path": str(batch_path),
        "selection_policy": {
            "goal": "one accepted full record per risk type",
            "coverage_axes": [
                "risk_type diversity",
                "prefer scenario diversity",
            ],
            "selection_count": len(success_records),
        },
        "risk_types": risk_types,
        "records": [
            {
                "risk_type": item["risk_type"],
                "source_batch_index": item["source_batch_index"],
                "cell_id": item["cell_id"],
                "record": item["record"],
            }
            for item in success_records
        ],
    }
    unsafe_payload = {
        "run_dir": str(run_dir),
        "batch_path": str(batch_path),
        "selection_policy": {
            "goal": "one accepted unsafe full record per risk type",
            "coverage_axes": [
                "risk_type diversity",
                "unsafe label coverage via strongest-available combo preference",
                "prefer scenario diversity",
            ],
            "selection_count": len(unsafe_records),
            "combo_priority": list(UNSAFE_COMBO_PRIORITY),
        },
        "available_unsafe_counts_by_risk_type": available_unsafe_counts_by_risk_type,
        "risk_types": risk_types,
        "records": unsafe_records,
    }
    return success_payload, unsafe_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair one batch run by merging accepted supplement records and annotating them with an optional LLM judge.")
    parser.add_argument("--base-run", type=Path, required=True, help="Base batch run directory or a file inside it.")
    parser.add_argument("--supplement-run", type=Path, action="append", required=True, help="Supplement batch run directory or a file inside it. Repeatable.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Generation config to snapshot and use for judge settings.")
    parser.add_argument("--seed", type=int, default=0, help="Optional seed used only for the repaired run directory name.")
    parser.add_argument("--run-name", type=str, default="repaired_merged", help="Suffix for the repaired run directory.")
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument("--judge", dest="judge", action="store_true", help="Annotate merged records with LLM-as-judge metadata.")
    judge_group.add_argument("--no-judge", dest="judge", action="store_false", help="Disable LLM-as-judge even if the config enables it.")
    parser.set_defaults(judge=None)
    parser.add_argument("--judge-prompt-mode", type=str, default=None, help="Optional override for judge prompt mode.")
    parser.add_argument("--judge-min-score", type=int, default=None, help="Optional override for judge minimum score.")
    args = parser.parse_args()

    base_run_dir = _normalize_run_dir(args.base_run)
    supplement_run_dirs = [_normalize_run_dir(path) for path in args.supplement_run]
    base_payload = _load_run_payload(base_run_dir)
    supplement_payloads = [_load_run_payload(path) for path in supplement_run_dirs]

    merged_records = list(base_payload["records"])
    added_cells: list[str] = []
    skipped_cells: list[str] = []
    for payload in supplement_payloads:
        merged_records, added, skipped = _merge_records(
            base_records=merged_records,
            supplement_records=payload["records"],
        )
        added_cells.extend(added)
        skipped_cells.extend(skipped)
    merged_records = [_normalize_record_risk_fields(record) for record in merged_records]

    judge_settings = _effective_judge_settings(
        config_path=args.config,
        enabled=args.judge,
        prompt_mode=args.judge_prompt_mode,
        min_score=args.judge_min_score,
    )
    judge_failures: list[dict[str, Any]] = []
    if judge_settings is not None:
        print(f"[repair] applying_judge records={len(merged_records)} model={judge_settings.model_name}", flush=True)
        judge_failures = _apply_judge_annotations(records=merged_records, judge_settings=judge_settings)

    reports = [base_payload["report"], *[payload["report"] for payload in supplement_payloads]]
    attempted_count, rejected_count, attempts_by_cell, rejection_reason_counter, last_failure_reasons_by_cell = _merge_report_counters(reports)
    accepted_by_cell = Counter(_record_cell_id(record) for record in merged_records)
    cells = _build_cells_metadata(records=merged_records, reports=reports)

    target_values = {report.get("summary", {}).get("target_per_cell") for report in reports}
    max_attempt_values = {report.get("summary", {}).get("max_attempts_per_cell") for report in reports}
    target_per_cell = next(iter(target_values)) if len(target_values) == 1 else None
    max_attempts_per_cell = next(iter(max_attempt_values)) if len(max_attempt_values) == 1 else None

    repaired_report = _build_report(
        cells=cells,
        accepted_records=merged_records,
        attempted_count=attempted_count,
        rejected_count=rejected_count,
        rejection_reason_counter=rejection_reason_counter,
        accepted_by_cell=accepted_by_cell,
        attempts_by_cell=attempts_by_cell,
        last_failure_reasons_by_cell=last_failure_reasons_by_cell,
        target_per_cell=target_per_cell,
        max_attempts_per_cell=max_attempts_per_cell,
    )

    risk_types = _ordered_unique([_record_risk_type(record) for record in merged_records if _record_risk_type(record)])
    run_dir = create_run_dir(
        kind="batch",
        scenario="multi",
        label_combo="multi",
        seed=args.seed,
        run_name=args.run_name,
    )
    snapshot_config(args.config, run_dir)

    batch_path = output_file_in_run_dir(run_dir, "", default_name="batch.json")
    report_path = run_dir / "report.json"
    review_success_path = run_dir / "review_success_by_risk_type.json"
    review_unsafe_path = run_dir / "review_unsafe_by_risk_type.json"
    merge_summary_path = run_dir / "repair_summary.json"
    judge_failures_path = run_dir / "judge_failures.json"

    success_review, unsafe_review = _build_review_payloads(
        merged_records=merged_records,
        run_dir=run_dir,
        batch_path=batch_path,
        risk_types=risk_types,
    )

    merge_summary = {
        "base_run_dir": str(base_run_dir),
        "supplement_run_dirs": [str(path) for path in supplement_run_dirs],
        "base_accepted_count": len(base_payload["records"]),
        "supplement_accepted_count": sum(len(payload["records"]) for payload in supplement_payloads),
        "merged_accepted_count": len(merged_records),
        "added_cells": added_cells,
        "skipped_existing_cells": skipped_cells,
        "judge_enabled": judge_settings is not None,
        "judge_prompt_mode": str(getattr(judge_settings, "judge_prompt_mode", "")).strip() if judge_settings is not None else "",
        "judge_min_score": int(getattr(judge_settings, "min_score", 0)) if judge_settings is not None else None,
        "judge_failure_count": len(judge_failures),
        "final_unmet_count": len(repaired_report.get("summary", {}).get("unmet_cells", [])),
    }

    batch_path.write_text(json.dumps(merged_records, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(repaired_report, ensure_ascii=False, indent=2), encoding="utf-8")
    review_success_path.write_text(json.dumps(success_review, ensure_ascii=False, indent=2), encoding="utf-8")
    review_unsafe_path.write_text(json.dumps(unsafe_review, ensure_ascii=False, indent=2), encoding="utf-8")
    merge_summary_path.write_text(json.dumps(merge_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if judge_settings is not None:
        judge_failures_path.write_text(json.dumps(judge_failures, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[repair] run_dir={run_dir}", flush=True)
    print(
        "[repair] merged "
        f"accepted={repaired_report['summary']['accepted']} attempted={repaired_report['summary']['attempted']} "
        f"rejected={repaired_report['summary']['rejected']} unmet={len(repaired_report['summary']['unmet_cells'])}",
        flush=True,
    )
    if judge_settings is not None:
        print(f"[repair] judge_failures={len(judge_failures)}", flush=True)
    print(str(batch_path))


if __name__ == "__main__":
    main()
