from __future__ import annotations

from typing import Any

from .judge_llm import llm_judge_record
from .llm_client import LLMSettings

from .lightweight_export import lightweight_record
from .risk_taxonomy import canonical_risk_type_from_setup
from .stage2_contracts import combo_prompt_lines


def judge_payload_for_record(record: dict[str, Any]) -> dict[str, Any]:
    execution_plan = record.get("execution_plan", {})
    label_combo = execution_plan.get("label_combo", {})
    risk_type = canonical_risk_type_from_setup(execution_plan.get("risk_setup", {}))
    clean_safe_path = bool(execution_plan.get("clean_safe_path", False)) if isinstance(execution_plan, dict) else False
    if clean_safe_path:
        risk_type = ""
    trajectory = record.get("trajectory", {})
    messages = trajectory.get("messages", [])
    combo_slug = str(label_combo.get("slug", "")).strip()
    config_path = str(execution_plan.get("config_path", "")).strip() or None
    combo_semantics = "\n".join(
        f"- {line}"
        for line in combo_prompt_lines(
            combo_slug,
            config_path=config_path,
            include_axis=True,
            include_decision_rule=True,
            include_invalid_rules=True,
            include_plain_meaning=True,
        )
    )
    light = lightweight_record(record)
    return {
        "scenario_name": str(execution_plan.get("scenario_id", "")).strip(),
        "risk_source_category": risk_type,
        "risk_injected": bool(light.get("risk_injected", False)),
        "trigger_channel": str(light.get("trigger_channel", "")).strip(),
        "trigger_position": str(light.get("trigger_position", "")).strip(),
        "binary_label": str(label_combo.get("binary_label", "")).strip(),
        "action_state": str(label_combo.get("action_state", "")).strip(),
        "realization_state": str(label_combo.get("realization_state", "")).strip(),
        "clean_safe_triplet": bool(clean_safe_path),
        "safe_no_risk_injection": bool(clean_safe_path),
        "safe_with_risk_injection_no_progress": (
            str(label_combo.get("binary_label", "")).strip() == "safe"
            and str(label_combo.get("action_state", "")).strip() == "no_unsafe_action"
            and str(label_combo.get("realization_state", "")).strip() == "not_realized"
            and not clean_safe_path
        ),
        "combo_semantics": combo_semantics,
        "tool_used": list(light.get("tool_used", [])) if isinstance(light.get("tool_used", []), list) else [],
        "tool_interface_anchor": dict(light.get("tool_interface_anchor", {}))
        if isinstance(light.get("tool_interface_anchor", {}), dict)
        else {},
        "tool_execution_anchor": dict(light.get("tool_execution_anchor", {}))
        if isinstance(light.get("tool_execution_anchor", {}), dict)
        else {},
        "content": [
            [
                {
                    "role": str(message.get("role", "")).strip(),
                    "content": str(message.get("content", "")),
                }
            ]
            for message in messages
            if isinstance(message, dict)
        ],
    }


def apply_llm_judge(record: dict[str, Any], settings: LLMSettings) -> list[str]:
    judge_record = judge_payload_for_record(record)
    reasons = llm_judge_record(judge_record, settings)
    record["quality_gate"] = {
        "llm_judge": {
            "score": int(judge_record.get("llm_judge_score", 0)),
            "dimensions": dict(judge_record.get("llm_judge_dimensions", {})),
            "rationale": str(judge_record.get("llm_judge_rationale", "")).strip(),
            "prompt_mode": str(judge_record.get("llm_judge_prompt_mode", "")).strip(),
        }
    }
    return reasons
