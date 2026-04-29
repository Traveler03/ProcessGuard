from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any


PROCESS_GUARD_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESS_GUARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_GUARD_ROOT))

from pipeline_v1.llm_client import llm_available  # noqa: E402

from pipeline_v1.config import DEFAULT_CONFIG_PATH, load_judge_settings  # noqa: E402
from pipeline_v1.judge_adapter import apply_llm_judge  # noqa: E402
from pipeline_v1.lightweight_export import lightweight_records  # noqa: E402
from pipeline_v1.output_utils import clear_output_root, create_run_dir, lightweight_output_path, output_file_in_run_dir, snapshot_config  # noqa: E402
from pipeline_v1.pipeline import generate_sample  # noqa: E402
from pipeline_v1.planner import risk_type_candidates, scenario_workflow_constraints, simplify_tools  # noqa: E402
from pipeline_v1.scenarios import load_scenarios  # noqa: E402
from pipeline_v1.schema import LABEL_COMBO_SPECS  # noqa: E402


def _reason_key(reason: str) -> str:
    return str(reason).split(":", 1)[0].strip() or "unknown"


def _sampling_metadata(
    *,
    scenario_name: str,
    label_combo: str,
    risk_type: str | None,
    seed: int,
    cell_id: str,
    attempt_index: int,
) -> dict[str, Any]:
    return {
        "scenario_name": scenario_name,
        "label_combo": label_combo,
        "risk_type": risk_type,
        "seed": seed,
        "cell_id": cell_id,
        "attempt_index": attempt_index,
    }


def _resolved_sampling_risk_type(record: dict[str, Any], requested_risk_type: str | None) -> str | None:
    normalized_requested = str(requested_risk_type or "").strip()
    if normalized_requested:
        return normalized_requested
    execution_plan = record.get("execution_plan", {})
    if not isinstance(execution_plan, dict):
        return None
    risk_setup = execution_plan.get("risk_setup", {})
    if not isinstance(risk_setup, dict):
        return None
    resolved = str(risk_setup.get("risk_type", "")).strip()
    return resolved or None


def _record_rejection_reasons(record: dict[str, Any]) -> list[str]:
    validation = record.get("validation", {})
    if bool(validation.get("accepted", False)):
        return []
    return [
        f"validation:{str(reason).strip()}"
        for reason in validation.get("reasons", [])
        if str(reason).strip()
    ] or ["validation:unknown_failure"]


def _effective_judge_settings(
    *,
    config_path: Path,
    enabled: bool | None,
    prompt_mode: str | None,
    min_score: int | None,
) -> Any:
    base_settings = load_judge_settings(config_path)
    effective_enabled = base_settings.enabled if enabled is None else bool(enabled)
    if not effective_enabled:
        return None
    judge_settings = replace(base_settings, enabled=True)
    if prompt_mode:
        judge_settings = replace(judge_settings, judge_prompt_mode=prompt_mode.strip())
    if min_score is not None:
        judge_settings = replace(judge_settings, min_score=int(min_score))
    if not llm_available(judge_settings):
        raise RuntimeError(
            "pipeline_v1 LLM judge endpoint is unavailable: "
            f"base_url={judge_settings.api_base_url} model={judge_settings.model_name}"
        )
    return judge_settings


def _selected_scenarios(allowed_scenarios: set[str]) -> list[str]:
    scenario_names = [
        scenario["scenario_name"]
        for scenario in load_scenarios()
        if not allowed_scenarios or scenario["scenario_name"] in allowed_scenarios
    ]
    if allowed_scenarios and len(scenario_names) != len(allowed_scenarios):
        missing = sorted(allowed_scenarios - set(scenario_names))
        raise KeyError(f"Unknown scenarios requested: {','.join(missing)}")
    if not scenario_names:
        raise RuntimeError("No scenarios selected for batch generation")
    return scenario_names


def _scenario_lookup() -> dict[str, dict[str, Any]]:
    return {
        scenario["scenario_name"]: scenario
        for scenario in load_scenarios()
    }


def _candidate_risky_tools_for_scenario(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    tools = simplify_tools(scenario)
    workflow_constraints = scenario_workflow_constraints(scenario, tools)
    tools_by_name = {str(tool.get("name", "")).strip(): tool for tool in tools}
    risky_tool_names = [
        tool_name
        for tool_name in workflow_constraints["completion_tools"]
        if tool_name in workflow_constraints["branchable_tools"]
    ]
    if not risky_tool_names:
        raise RuntimeError(
            f"Scenario {scenario['scenario_name']} has no workflow tool shared by completion_tools and branchable_tools"
        )
    return [tools_by_name[tool_name] for tool_name in risky_tool_names]


def _compatible_risk_types_for_scenario(
    *,
    scenario: dict[str, Any],
    label_combo: str,
    config_path: Path,
    requested_risk_types: list[str],
) -> list[str]:
    compatible_sets: list[set[str]] = []
    for risky_tool in _candidate_risky_tools_for_scenario(scenario):
        compatible_sets.append(
            {
                str(candidate.get("risk_type", "")).strip()
                for candidate in risk_type_candidates(
                    scenario,
                    risky_tool,
                    combo=label_combo,
                    config_path=config_path,
                )
                if str(candidate.get("risk_type", "")).strip()
            }
        )
    if not compatible_sets:
        raise RuntimeError(f"Scenario {scenario['scenario_name']} has no compatible risk types")
    compatible = set.intersection(*compatible_sets)
    if requested_risk_types:
        compatible &= {item.strip() for item in requested_risk_types if item.strip()}
    if not compatible:
        raise RuntimeError(
            f"Scenario {scenario['scenario_name']} has no compatible risk types for the requested selection"
        )
    return sorted(compatible)


def _cell_id(scenario_name: str, label_combo: str, risk_type: str | None) -> str:
    parts = [scenario_name, label_combo]
    if risk_type:
        parts.append(risk_type)
    return "::".join(parts)


def _build_cells(
    *,
    scenario_names: list[str],
    label_combos: list[str],
    risk_types: list[str],
    config_path: Path,
    compatible_risk_types: bool,
) -> list[dict[str, str | None]]:
    scenarios_by_name = _scenario_lookup()
    cells: list[dict[str, str | None]] = []
    for scenario_name in scenario_names:
        scenario = scenarios_by_name[scenario_name]
        if compatible_risk_types:
            compatible_by_combo = {
                combo: _compatible_risk_types_for_scenario(
                    scenario=scenario,
                    label_combo=combo,
                    config_path=config_path,
                    requested_risk_types=risk_types,
                )
                for combo in label_combos
            }
        for combo in label_combos:
            if compatible_risk_types:
                risk_axis = compatible_by_combo[combo]
            else:
                risk_axis = risk_types or [None]
            for risk_type in risk_axis:
                cells.append(
                    {
                        "scenario_name": scenario_name,
                        "label_combo": combo,
                        "risk_type": risk_type,
                        "cell_id": _cell_id(scenario_name, combo, risk_type),
                    }
                )
    if not cells:
        raise RuntimeError("No batch cells selected")
    return cells


def _build_report(
    *,
    cells: list[dict[str, str | None]],
    accepted_records: list[dict[str, Any]],
    attempted_count: int,
    rejected_count: int,
    rejection_reason_counter: Counter[str],
    accepted_by_cell: Counter[str],
    attempts_by_cell: Counter[str],
    last_failure_reasons_by_cell: dict[str, list[str]],
    target_per_cell: int | None,
    max_attempts_per_cell: int | None,
) -> dict[str, Any]:
    judge_score_distribution = Counter()
    for record in accepted_records:
        score = (
            record.get("quality_gate", {})
            .get("llm_judge", {})
            .get("score")
        )
        if isinstance(score, int):
            judge_score_distribution[str(score)] += 1
    unmet_cells = sorted(
        cell_id
        for cell_id, attempts in attempts_by_cell.items()
        if target_per_cell is not None
        and accepted_by_cell.get(cell_id, 0) < target_per_cell
        and (max_attempts_per_cell is None or attempts >= max_attempts_per_cell)
    )
    cells_by_id = {str(cell["cell_id"]): cell for cell in cells}
    failed_cells: list[dict[str, Any]] = []
    for cell_id in unmet_cells:
        cell = cells_by_id.get(cell_id, {})
        failed_cells.append(
            {
                "cell_id": cell_id,
                "scenario_name": cell.get("scenario_name"),
                "label_combo": cell.get("label_combo"),
                "risk_type": cell.get("risk_type"),
                "accepted": int(accepted_by_cell.get(cell_id, 0)),
                "attempted": int(attempts_by_cell.get(cell_id, 0)),
                "last_failure_reasons": list(last_failure_reasons_by_cell.get(cell_id, [])),
            }
        )
    return {
        "summary": {
            "accepted": len(accepted_records),
            "attempted": attempted_count,
            "rejected": rejected_count,
            "acceptance_rate": (len(accepted_records) / attempted_count) if attempted_count else 0.0,
            "target_per_cell": target_per_cell,
            "max_attempts_per_cell": max_attempts_per_cell,
            "judge_score_distribution": dict(judge_score_distribution),
            "rejection_reason_distribution": dict(rejection_reason_counter),
            "accepted_by_cell": dict(accepted_by_cell),
            "attempts_by_cell": dict(attempts_by_cell),
            "unmet_cells": unmet_cells,
        },
        "failed_cells": failed_cells,
    }


def _write_partial_outputs(
    *,
    output_path: Path | None,
    light_output_path: Path | None,
    report_path: Path | None,
    accepted_records: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    if output_path is not None:
        output_path.write_text(json.dumps(accepted_records, ensure_ascii=False, indent=2), encoding="utf-8")
    if light_output_path is not None:
        light_output_path.write_text(
            json.dumps(lightweight_records(accepted_records), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if report_path is not None:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_generation_exception(
    *,
    rejection_reason_counter: Counter[str],
    rejected_count_ref: list[int],
    reason: str,
) -> None:
    rejected_count_ref[0] += 1
    rejection_reason_counter[_reason_key(reason)] += 1


def _is_incompatible_risk_type_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "Incompatible risk_type_override=" in str(exc)


def _batch_progress_callback(*, cell_id: str, attempt_index: int | None = None) -> Any:
    def _emit(message: str) -> None:
        attempt_text = f" try={attempt_index}" if attempt_index is not None else ""
        print(f"[batch] progress cell={cell_id}{attempt_text} {message}", flush=True)

    return _emit


def _generate_single_pass(
    *,
    cells: list[dict[str, str | None]],
    seed: int,
    config_path: Path,
    judge_settings: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        cell_seed = seed + (cell_index * 10000)
        print(
            f"[batch] cell_start cell={cell['cell_id']} try=1 seed={cell_seed}",
            flush=True,
        )
        record = generate_sample(
            scenario_name=str(cell["scenario_name"]),
            label_combo=str(cell["label_combo"]),
            seed=cell_seed,
            config_path=config_path,
            risk_type_override=cell["risk_type"],
            progress_callback=_batch_progress_callback(cell_id=str(cell["cell_id"]), attempt_index=1),
        )
        record["sampling"] = _sampling_metadata(
            scenario_name=str(cell["scenario_name"]),
            label_combo=str(cell["label_combo"]),
            risk_type=_resolved_sampling_risk_type(record, cell["risk_type"]),
            seed=cell_seed,
            cell_id=str(cell["cell_id"]),
            attempt_index=1,
        )
        if judge_settings is not None:
            apply_llm_judge(record, judge_settings)
        records.append(record)
        print(
            f"[batch] cell_done cell={cell['cell_id']} try=1 accepted={bool(record.get('validation', {}).get('accepted', False))}",
            flush=True,
        )
    return records


def _generate_until_target(
    *,
    cells: list[dict[str, str | None]],
    seed: int,
    config_path: Path,
    judge_settings: Any,
    target_per_cell: int,
    max_attempts_per_cell: int | None,
    output_path: Path | None = None,
    light_output_path: Path | None = None,
    report_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted_records: list[dict[str, Any]] = []
    accepted_by_cell: Counter[str] = Counter()
    attempts_by_cell: Counter[str] = Counter()
    rejection_reason_counter: Counter[str] = Counter()
    last_failure_reasons_by_cell: dict[str, list[str]] = {}
    attempted_count = 0
    rejected_count_box = [0]

    for cell_index, cell in enumerate(cells):
        print(
            f"[batch] cell_start cell={cell['cell_id']} target={target_per_cell} max_attempts={max_attempts_per_cell}",
            flush=True,
        )
        while accepted_by_cell[cell["cell_id"]] < target_per_cell:
            attempt_index = attempts_by_cell[cell["cell_id"]] + 1
            if max_attempts_per_cell is not None and attempt_index > max_attempts_per_cell:
                print(
                    f"[batch] cell_exhausted cell={cell['cell_id']} accepted={accepted_by_cell[cell['cell_id']]}/{target_per_cell}",
                    flush=True,
                )
                partial_report = _build_report(
                    cells=cells,
                    accepted_records=accepted_records,
                    attempted_count=attempted_count,
                    rejected_count=rejected_count_box[0],
                    rejection_reason_counter=rejection_reason_counter,
                    accepted_by_cell=accepted_by_cell,
                    attempts_by_cell=attempts_by_cell,
                    last_failure_reasons_by_cell=last_failure_reasons_by_cell,
                    target_per_cell=target_per_cell,
                    max_attempts_per_cell=max_attempts_per_cell,
                )
                _write_partial_outputs(
                    output_path=output_path,
                    light_output_path=light_output_path,
                    report_path=report_path,
                    accepted_records=accepted_records,
                    report=partial_report,
                )
                break
            attempts_by_cell[cell["cell_id"]] += 1
            attempted_count += 1
            cell_seed = seed + (cell_index * 10000) + (attempt_index - 1)
            print(
                f"[batch] attempt cell={cell['cell_id']} try={attempt_index} seed={cell_seed}",
                flush=True,
            )
            try:
                record = generate_sample(
                    scenario_name=cell["scenario_name"],
                    label_combo=cell["label_combo"],
                    seed=cell_seed,
                    config_path=config_path,
                    risk_type_override=cell["risk_type"],
                    progress_callback=_batch_progress_callback(
                        cell_id=str(cell["cell_id"]),
                        attempt_index=attempt_index,
                    ),
                )
            except Exception as exc:
                if _is_incompatible_risk_type_error(exc):
                    raise RuntimeError(
                        f"Invalid batch cell selection for {cell['cell_id']}: {exc}"
                    ) from exc
                _record_generation_exception(
                    rejection_reason_counter=rejection_reason_counter,
                    rejected_count_ref=rejected_count_box,
                    reason=f"generation_exception:{type(exc).__name__}:{exc}",
                )
                reason_text = f"generation_exception:{type(exc).__name__}:{exc}"
                last_failure_reasons_by_cell[cell["cell_id"]] = [reason_text]
                print(
                    f"[batch] reject cell={cell['cell_id']} try={attempt_index} reason={reason_text}",
                    flush=True,
                )
                partial_report = _build_report(
                    cells=cells,
                    accepted_records=accepted_records,
                    attempted_count=attempted_count,
                    rejected_count=rejected_count_box[0],
                    rejection_reason_counter=rejection_reason_counter,
                    accepted_by_cell=accepted_by_cell,
                    attempts_by_cell=attempts_by_cell,
                    last_failure_reasons_by_cell=last_failure_reasons_by_cell,
                    target_per_cell=target_per_cell,
                    max_attempts_per_cell=max_attempts_per_cell,
                )
                _write_partial_outputs(
                    output_path=output_path,
                    light_output_path=light_output_path,
                    report_path=report_path,
                    accepted_records=accepted_records,
                    report=partial_report,
                )
                continue
            record["sampling"] = _sampling_metadata(
                scenario_name=cell["scenario_name"],
                label_combo=cell["label_combo"],
                risk_type=_resolved_sampling_risk_type(record, cell["risk_type"]),
                seed=cell_seed,
                cell_id=cell["cell_id"],
                attempt_index=attempt_index,
            )
            reasons = _record_rejection_reasons(record)
            if not reasons and judge_settings is not None:
                reasons.extend(apply_llm_judge(record, judge_settings))
            if reasons:
                rejected_count_box[0] += 1
                for reason in reasons:
                    rejection_reason_counter[_reason_key(reason)] += 1
                last_failure_reasons_by_cell[cell["cell_id"]] = list(reasons)
                print(
                    f"[batch] reject cell={cell['cell_id']} try={attempt_index} reasons={','.join(sorted({_reason_key(reason) for reason in reasons}))}",
                    flush=True,
                )
                partial_report = _build_report(
                    cells=cells,
                    accepted_records=accepted_records,
                    attempted_count=attempted_count,
                    rejected_count=rejected_count_box[0],
                    rejection_reason_counter=rejection_reason_counter,
                    accepted_by_cell=accepted_by_cell,
                    attempts_by_cell=attempts_by_cell,
                    last_failure_reasons_by_cell=last_failure_reasons_by_cell,
                    target_per_cell=target_per_cell,
                    max_attempts_per_cell=max_attempts_per_cell,
                )
                _write_partial_outputs(
                    output_path=output_path,
                    light_output_path=light_output_path,
                    report_path=report_path,
                    accepted_records=accepted_records,
                    report=partial_report,
                )
                continue
            accepted_records.append(record)
            accepted_by_cell[cell["cell_id"]] += 1
            print(
                f"[batch] accept cell={cell['cell_id']} accepted={accepted_by_cell[cell['cell_id']]}/{target_per_cell}",
                flush=True,
            )
            partial_report = _build_report(
                cells=cells,
                accepted_records=accepted_records,
                attempted_count=attempted_count,
                rejected_count=rejected_count_box[0],
                rejection_reason_counter=rejection_reason_counter,
                accepted_by_cell=accepted_by_cell,
                attempts_by_cell=attempts_by_cell,
                last_failure_reasons_by_cell=last_failure_reasons_by_cell,
                target_per_cell=target_per_cell,
                max_attempts_per_cell=max_attempts_per_cell,
            )
            _write_partial_outputs(
                output_path=output_path,
                light_output_path=light_output_path,
                report_path=report_path,
                accepted_records=accepted_records,
                report=partial_report,
            )

    report = _build_report(
        cells=cells,
        accepted_records=accepted_records,
        attempted_count=attempted_count,
        rejected_count=rejected_count_box[0],
        rejection_reason_counter=rejection_reason_counter,
        accepted_by_cell=accepted_by_cell,
        attempts_by_cell=attempts_by_cell,
        last_failure_reasons_by_cell=last_failure_reasons_by_cell,
        target_per_cell=target_per_cell,
        max_attempts_per_cell=max_attempts_per_cell,
    )
    return accepted_records, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a batch of ProcessGuard pipeline_v1 samples.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--config", type=Path, default=None, help="Optional pipeline_v1 generation config path.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional suffix for the auto-created run directory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional file name inside the auto-created run directory.")
    parser.add_argument("--scenario", action="append", default=None, help="Optional scenario allowlist. Repeatable.")
    parser.add_argument("--label-combo", action="append", default=None, help="Optional combo allowlist. Repeatable.")
    parser.add_argument("--risk-type", action="append", default=None, help="Optional explicit risk-type category allowlist. Repeatable.")
    parser.add_argument(
        "--compatible-risk-types",
        action="store_true",
        help="Expand the risk-type category axis per scenario using only globally compatible categories for that scenario's risky completion surface.",
    )
    parser.add_argument("--target-per-cell", type=int, default=None, help="If set, keep generating until each selected cell reaches this many accepted records.")
    parser.add_argument(
        "--max-attempts-per-cell",
        type=int,
        default=5,
        help="Hard cap for rejection-sampling attempts per selected cell. Use 0 for no cap.",
    )
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument("--judge", dest="judge", action="store_true", help="Enable LLM-as-judge quality filtering for batch generation.")
    judge_group.add_argument("--no-judge", dest="judge", action="store_false", help="Disable LLM-as-judge even if the config enables it.")
    parser.set_defaults(judge=None)
    parser.add_argument("--judge-prompt-mode", type=str, default=None, help="Optional override for judge prompt mode.")
    parser.add_argument("--judge-min-score", type=int, default=None, help="Optional override for minimum accepted judge score.")
    parser.add_argument("--clear-output", action="store_true", help="Delete existing pipeline_v1 output contents before creating the new run directory.")
    args = parser.parse_args()

    allowed_scenarios = {item.strip() for item in (args.scenario or []) if item.strip()}
    label_combos = [item.strip() for item in (args.label_combo or LABEL_COMBO_SPECS.keys()) if item.strip()]
    risk_types = [item.strip() for item in (args.risk_type or []) if item.strip()]
    config_path = args.config or DEFAULT_CONFIG_PATH
    scenario_names = _selected_scenarios(allowed_scenarios)
    cells = _build_cells(
        scenario_names=scenario_names,
        label_combos=label_combos,
        risk_types=risk_types,
        config_path=config_path,
        compatible_risk_types=bool(args.compatible_risk_types),
    )
    judge_settings = _effective_judge_settings(
        config_path=config_path,
        enabled=args.judge,
        prompt_mode=args.judge_prompt_mode,
        min_score=args.judge_min_score,
    )

    if args.clear_output:
        clear_output_root()

    scenario_suffix = "multi" if len(allowed_scenarios) != 1 else next(iter(allowed_scenarios))
    combo_suffix = "multi" if len(label_combos) != 1 else label_combos[0]
    run_dir = create_run_dir(
        kind="batch",
        scenario=scenario_suffix,
        label_combo=combo_suffix,
        seed=args.seed,
        run_name=args.run_name,
    )
    snapshot_config(config_path, run_dir)

    output_path = output_file_in_run_dir(run_dir, str(args.output) if args.output else "", default_name="batch.json")
    light_output_path = lightweight_output_path(output_path)
    report_path = run_dir / "report.json"

    if args.target_per_cell is not None:
        max_attempts_per_cell = None if int(args.max_attempts_per_cell) <= 0 else max(int(args.max_attempts_per_cell), 1)
        records, report = _generate_until_target(
            cells=cells,
            seed=args.seed,
            config_path=config_path,
            judge_settings=judge_settings,
            target_per_cell=max(int(args.target_per_cell), 1),
            max_attempts_per_cell=max_attempts_per_cell,
            output_path=output_path,
            light_output_path=light_output_path,
            report_path=report_path,
        )
    else:
        records = _generate_single_pass(
            cells=cells,
            seed=args.seed,
            config_path=config_path,
            judge_settings=judge_settings,
        )
        report = _build_report(
            cells=cells,
            accepted_records=records,
            attempted_count=len(records),
            rejected_count=0,
            rejection_reason_counter=Counter(),
            accepted_by_cell=Counter(str(record.get("sampling", {}).get("cell_id", "")) for record in records),
            attempts_by_cell=Counter(str(record.get("sampling", {}).get("cell_id", "")) for record in records),
            last_failure_reasons_by_cell={},
            target_per_cell=None,
            max_attempts_per_cell=None,
        )

    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    light_output_path.write_text(json.dumps(lightweight_records(records), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    unmet_cells = report.get("summary", {}).get("unmet_cells", [])
    if unmet_cells:
        print(
            "[batch] completed_with_unmet_cells count="
            f"{len(unmet_cells)} cells=" + ",".join(str(item) for item in unmet_cells),
            flush=True,
        )
    print(str(output_path))


if __name__ == "__main__":
    main()
