from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Callable


PROCESS_GUARD_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESS_GUARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_GUARD_ROOT))

from pipeline_v1.scripts.repair_batch_run import _load_json, _normalize_run_dir  # noqa: E402
from pipeline_v1.risk_taxonomy import canonical_risk_type_and_subtype  # noqa: E402


def _judge_score(record: dict[str, Any]) -> int:
    try:
        return int(record.get("quality_gate", {}).get("llm_judge", {}).get("score", 0))
    except Exception:
        return 0


def _scenario_id(record: dict[str, Any]) -> str:
    return str(record.get("execution_plan", {}).get("scenario_id", "")).strip()


def _label_combo(record: dict[str, Any]) -> str:
    return str(record.get("execution_plan", {}).get("label_combo", {}).get("slug", "")).strip()


def _risk_type(record: dict[str, Any]) -> str:
    risk_type, _ = canonical_risk_type_and_subtype(record.get("execution_plan", {}).get("risk_setup", {}))
    return risk_type


def _cell_id(record: dict[str, Any]) -> str:
    return str(record.get("sampling", {}).get("cell_id", "")).strip()


def _secondary_bucket_sort_key(records: list[dict[str, Any]], bucket_value: str) -> tuple[int, str]:
    return (-max(_judge_score(record) for record in records), bucket_value)


def _sort_bucket_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            -_judge_score(record),
            _scenario_id(record),
            _cell_id(record),
        ),
    )


def _pick_one_from_bucket(
    records: list[dict[str, Any]],
    *,
    used_cells: set[str],
    used_scenarios: set[str],
) -> dict[str, Any] | None:
    unseen_scenario = next(
        (
            record
            for record in records
            if _cell_id(record) not in used_cells and _scenario_id(record) not in used_scenarios
        ),
        None,
    )
    if unseen_scenario is not None:
        return unseen_scenario
    return next((record for record in records if _cell_id(record) not in used_cells), None)


def _select_diverse_samples(
    records: list[dict[str, Any]],
    *,
    secondary_key_fn: Callable[[dict[str, Any]], str],
    limit: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[secondary_key_fn(record)].append(record)
    ordered_bucket_names = [
        bucket_name
        for bucket_name, _ in sorted(
            buckets.items(),
            key=lambda item: _secondary_bucket_sort_key(item[1], item[0]),
        )
    ]
    sorted_buckets = {
        bucket_name: _sort_bucket_records(bucket_records)
        for bucket_name, bucket_records in buckets.items()
    }

    selected: list[dict[str, Any]] = []
    used_cells: set[str] = set()
    used_scenarios: set[str] = set()

    while len(selected) < limit:
        progress = False
        for bucket_name in ordered_bucket_names:
            candidate = _pick_one_from_bucket(
                sorted_buckets[bucket_name],
                used_cells=used_cells,
                used_scenarios=used_scenarios,
            )
            if candidate is None:
                continue
            selected.append(candidate)
            used_cells.add(_cell_id(candidate))
            used_scenarios.add(_scenario_id(candidate))
            progress = True
            if len(selected) >= limit:
                break
        if not progress:
            break
    return selected[:limit]


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    execution_plan = record.get("execution_plan", {})
    risk_setup = execution_plan.get("risk_setup", {})
    risk_type, _ = canonical_risk_type_and_subtype(risk_setup)
    trajectory = record.get("trajectory", {})
    sampling = record.get("sampling", {})
    sampling = sampling if isinstance(sampling, dict) else {}
    return {
        "sampling": {
            key: value
            for key, value in sampling.items()
            if str(key).strip() != "risk_subtype"
        },
        "scenario_id": execution_plan.get("scenario_id"),
        "label_combo": execution_plan.get("label_combo"),
        "risk_setup": {
            "risk_type": risk_type,
            "trigger_channel": risk_setup.get("trigger_channel"),
            "risk_asset": risk_setup.get("risk_asset"),
            "harm_target": risk_setup.get("harm_target"),
            "target_kind": risk_setup.get("target_kind"),
            "risk_payload_facts": risk_setup.get("risk_payload_facts", []),
        },
        "branch_operator": execution_plan.get("branch_operator"),
        "rollout_contract": execution_plan.get("rollout_contract", {}),
        "quality_gate": record.get("quality_gate", {}),
        "trajectory": {
            "query": trajectory.get("query"),
            "messages": trajectory.get("messages", []),
            "tool_calls": trajectory.get("tool_calls", []),
            "events": trajectory.get("events", []),
            "outcome": trajectory.get("outcome", {}),
            "status": trajectory.get("status", {}),
            "step_tags": trajectory.get("step_tags", []),
        },
        "validation": record.get("validation", {}),
    }


def _write_grouped_samples(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    group_by: str,
    group_key_fn: Callable[[dict[str, Any]], str],
    secondary_key_fn: Callable[[dict[str, Any]], str],
    limit: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[group_key_fn(record)].append(record)

    manifest_items: list[dict[str, Any]] = []
    for group_name in sorted(grouped):
        selected = _select_diverse_samples(
            grouped[group_name],
            secondary_key_fn=secondary_key_fn,
            limit=limit,
        )
        payload = {
            "group_by": group_by,
            "group_value": group_name,
            "selection_policy": {
                "target_count": limit,
                "primary_order": "judge_score_desc",
                "secondary_diversity_axis": "risk_type" if group_by == "label_combo" else "label_combo",
                "scenario_diversity": True,
            },
            "record_count": len(selected),
            "records": [_compact_record(record) for record in selected],
        }
        file_path = output_dir / f"{group_name}.json"
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_items.append(
            {
                "group_value": group_name,
                "record_count": len(selected),
                "file_path": str(file_path),
            }
        )
    return manifest_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inspection-friendly ProcessGuard sample slices.")
    parser.add_argument("--run", type=Path, required=True, help="Input run directory or a file inside it.")
    parser.add_argument("--per-file", type=int, default=10, help="Number of samples per output file.")
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="inspection_samples",
        help="Directory name to create inside the run directory.",
    )
    args = parser.parse_args()

    run_dir = _normalize_run_dir(args.run)
    batch_records = _load_json(run_dir / "batch.json")
    if not isinstance(batch_records, list):
        raise ValueError("batch.json must contain a JSON list")

    output_root = run_dir / args.output_dir_name.strip()
    output_root.mkdir(parents=True, exist_ok=True)

    by_combo_manifest = _write_grouped_samples(
        records=batch_records,
        output_dir=output_root / "by_label_combo",
        group_by="label_combo",
        group_key_fn=_label_combo,
        secondary_key_fn=_risk_type,
        limit=max(int(args.per_file), 1),
    )
    by_risk_manifest = _write_grouped_samples(
        records=batch_records,
        output_dir=output_root / "by_risk_type",
        group_by="risk_type",
        group_key_fn=_risk_type,
        secondary_key_fn=_label_combo,
        limit=max(int(args.per_file), 1),
    )

    manifest = {
        "source_run_dir": str(run_dir),
        "source_batch_path": str(run_dir / "batch.json"),
        "per_file": max(int(args.per_file), 1),
        "output_root": str(output_root),
        "by_label_combo": by_combo_manifest,
        "by_risk_type": by_risk_manifest,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[samples] output_root={output_root}", flush=True)
    print(f"[samples] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
