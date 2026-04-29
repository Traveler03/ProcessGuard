from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import DEFAULT_CONFIG_PATH, require_generation_llm
from .filters import build_final_record, validate_trajectory
from .orchestrator import generate_trajectory
from .planner import build_execution_plan
from .scenarios import load_scenario_by_name
from .schema import LabelCombo, parse_label_combo


ProgressCallback = Callable[[str], None]


def _emit_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(str(message))


def generate_sample(
    *,
    scenario_name: str,
    label_combo: str | dict[str, str] | LabelCombo,
    seed: int = 0,
    catalog_path: Path | None = None,
    config_path: Path | None = DEFAULT_CONFIG_PATH,
    risk_type_override: str | None = None,
    progress_callback: ProgressCallback | None = None,
    attach_debug_metadata: bool = False,
) -> dict[str, Any]:
    stage_timings_seconds: dict[str, float] = {}
    total_started_at = perf_counter()
    scenario = load_scenario_by_name(scenario_name, catalog_path=catalog_path)
    combo = parse_label_combo(label_combo)
    llm_settings = require_generation_llm(config_path)
    _emit_progress(
        progress_callback,
        f"stage_start stage=planning scenario={scenario_name} combo={combo.slug} risk_type={risk_type_override or 'auto'} seed={seed}",
    )
    stage_started_at = perf_counter()
    try:
        plan = build_execution_plan(
            scenario=scenario,
            label_combo=combo,
            llm_settings=llm_settings,
            seed=seed,
            config_path=config_path,
            risk_type_override=risk_type_override,
        )
    except Exception as exc:
        _emit_progress(
            progress_callback,
            f"stage_error stage=planning elapsed={perf_counter() - stage_started_at:.1f}s error={type(exc).__name__}:{exc}",
        )
        raise
    stage_timings_seconds["planning"] = perf_counter() - stage_started_at
    _emit_progress(
        progress_callback,
        (
            "stage_done "
            f"stage=planning elapsed={stage_timings_seconds['planning']:.1f}s "
            f"risky_tool={plan.risky_tool_name} branch_operator={plan.branch_operator} risk_type={plan.risk_setup.risk_type}"
        ),
    )

    _emit_progress(progress_callback, "stage_start stage=trajectory")
    stage_started_at = perf_counter()
    try:
        trajectory = generate_trajectory(plan, llm_settings=llm_settings)
    except Exception as exc:
        _emit_progress(
            progress_callback,
            f"stage_error stage=trajectory elapsed={perf_counter() - stage_started_at:.1f}s error={type(exc).__name__}:{exc}",
        )
        raise
    stage_timings_seconds["trajectory"] = perf_counter() - stage_started_at
    _emit_progress(
        progress_callback,
        (
            "stage_done "
            f"stage=trajectory elapsed={stage_timings_seconds['trajectory']:.1f}s "
            f"messages={len(trajectory.messages)} tool_calls={len(trajectory.tool_calls)} events={len(trajectory.events)}"
        ),
    )

    _emit_progress(progress_callback, "stage_start stage=validation")
    stage_started_at = perf_counter()
    try:
        validation = validate_trajectory(plan, trajectory)
    except Exception as exc:
        _emit_progress(
            progress_callback,
            f"stage_error stage=validation elapsed={perf_counter() - stage_started_at:.1f}s error={type(exc).__name__}:{exc}",
        )
        raise
    stage_timings_seconds["validation"] = perf_counter() - stage_started_at
    _emit_progress(
        progress_callback,
        (
            "stage_done "
            f"stage=validation elapsed={stage_timings_seconds['validation']:.1f}s "
            f"accepted={bool(validation.accepted)} reasons={len(validation.reasons)}"
        ),
    )

    record = build_final_record(plan=plan, trajectory=trajectory, validation=validation)
    if attach_debug_metadata:
        record["generation_debug"] = {
            "stage_timings_seconds": stage_timings_seconds,
            "elapsed_seconds": perf_counter() - total_started_at,
        }
    return record
