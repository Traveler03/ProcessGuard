from __future__ import annotations

import json
from typing import Any

from .risk_taxonomy import canonical_risk_type_from_setup


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_index_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    return [
        int(item)
        for item in values
        if isinstance(item, int) or str(item).strip().isdigit()
    ]


def _normalized_evidence_indexes(validation: dict[str, Any]) -> dict[str, list[int]]:
    evidence_indexes = validation.get("evidence_indexes", {})
    if not isinstance(evidence_indexes, dict):
        return {}
    normalized: dict[str, list[int]] = {}
    for key, values in evidence_indexes.items():
        if not isinstance(key, str):
            continue
        normalized[key] = _normalize_index_list(values)
    return normalized


def _label_combo_state(record: dict[str, Any]) -> dict[str, Any]:
    validation = record.get("validation", {})
    if isinstance(validation, dict):
        derived_label_combo = validation.get("derived_label_combo", {})
        if isinstance(derived_label_combo, dict) and derived_label_combo:
            return derived_label_combo
    execution_plan = record.get("execution_plan", {})
    if isinstance(execution_plan, dict):
        label_combo = execution_plan.get("label_combo", {})
        if isinstance(label_combo, dict):
            return label_combo
    return {}


def _scenario_name(record: dict[str, Any]) -> str:
    execution_plan = record.get("execution_plan", {})
    if isinstance(execution_plan, dict):
        scenario_id = str(execution_plan.get("scenario_id", "")).strip()
        if scenario_id:
            return scenario_id
    sampling = record.get("sampling", {})
    if isinstance(sampling, dict):
        scenario_name = str(sampling.get("scenario_name", "")).strip()
        if scenario_name:
            return scenario_name
    return ""


def _execution_plan(record: dict[str, Any]) -> dict[str, Any]:
    execution_plan = record.get("execution_plan", {})
    return execution_plan if isinstance(execution_plan, dict) else {}


def _trajectory_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = record.get("trajectory", {})
    if not isinstance(trajectory, dict):
        return []
    messages = trajectory.get("messages", [])
    return [message for message in messages if isinstance(message, dict)]


def _parse_message_content(message: dict[str, Any]) -> dict[str, Any] | None:
    try:
        payload = json.loads(_stringify_content(message.get("content", "")))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _tool_interface_anchor(record: dict[str, Any], target_tool_name: str) -> dict[str, Any]:
    target = str(target_tool_name).strip()
    if not target:
        return {}
    for message in _trajectory_messages(record):
        payload = _parse_message_content(message)
        if not isinstance(payload, dict):
            continue
        result = payload.get("result", {})
        if not isinstance(result, dict):
            continue
        if str(result.get("notice_type", "")).strip() != "tool_description_patch":
            continue
        if str(result.get("target_tool", "")).strip() != target:
            continue
        patch = str(result.get("tool_description_patch", "")).strip()
        return {
            "anchor_kind": "tool_interface",
            "target_tool": target,
            "tool_description_patch": patch,
            "notice_message": str(result.get("message", "")).strip(),
        }
    execution_plan = _execution_plan(record)
    tools = execution_plan.get("tool_subset", [])
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if str(tool.get("name", "")).strip() != target:
                continue
            patch = str(tool.get("interface_patch_summary", "")).strip()
            parameter_patch_notes = tool.get("interface_parameter_patch_notes", {})
            parameter_patch_notes = (
                {
                    str(name).strip(): str(note).strip()
                    for name, note in parameter_patch_notes.items()
                    if str(name).strip() and str(note).strip()
                }
                if isinstance(parameter_patch_notes, dict)
                else {}
            )
            if patch or parameter_patch_notes:
                return {
                    "anchor_kind": "tool_interface",
                    "target_tool": target,
                    "tool_description_patch": patch,
                    "parameter_patch_notes": parameter_patch_notes,
                    "notice_message": "",
                }
    return {}


def _tool_execution_anchor(record: dict[str, Any], target_tool_name: str) -> dict[str, Any]:
    target = str(target_tool_name).strip()
    fallback_anchor: dict[str, Any] = {}
    for message in _trajectory_messages(record):
        payload = _parse_message_content(message)
        if not isinstance(payload, dict):
            continue
        result = payload.get("result", {})
        if not isinstance(result, dict):
            continue
        tool_name = str(result.get("tool_name", "")).strip()
        feedback = result.get("feedback", {})
        if not isinstance(feedback, dict) or not feedback:
            continue
        anchor = {
            "anchor_kind": "tool_execution",
            "target_tool": target,
            "source_tool": tool_name,
            "feedback_type": str(feedback.get("feedback_type", "")).strip(),
            "feedback_message": str(feedback.get("message", "")).strip(),
            "route_binding_target": str(feedback.get("route_binding_target", "")).strip(),
        }
        if target and tool_name == target:
            return anchor
        if not fallback_anchor:
            fallback_anchor = anchor
    return fallback_anchor


def _compact_parameters(parameters: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(parameters, list):
        return rows
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        row = {
            "name": str(parameter.get("name", "")).strip(),
            "type": str(parameter.get("type", "")).strip(),
            "description": str(parameter.get("description", "")).strip(),
            "required": bool(parameter.get("required", False)),
        }
        if row["name"]:
            rows.append(row)
    return rows


def _build_tool_used(record: dict[str, Any], *, risk_type: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    execution_plan = _execution_plan(record)
    tools = execution_plan.get("tool_subset", [])
    if not isinstance(tools, list):
        return [], {}, {}

    risk_setup = execution_plan.get("risk_setup", {})
    risk_setup = risk_setup if isinstance(risk_setup, dict) else {}
    target_tool = str(risk_setup.get("target_tool_name", "")).strip()
    risky_tool = str(execution_plan.get("risky_tool_name", "")).strip()
    completion_tool = str(execution_plan.get("completion_tool_name", "")).strip()

    tool_interface_anchor: dict[str, Any] = {}
    tool_execution_anchor: dict[str, Any] = {}
    if risk_type == "tool_interface":
        tool_interface_anchor = _tool_interface_anchor(record, target_tool)
    elif risk_type == "tool_execution":
        tool_execution_anchor = _tool_execution_anchor(record, target_tool)

    rows: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_name = str(tool.get("name", "")).strip()
        row = {
            "name": tool_name,
            "description": str(tool.get("description", "")).strip(),
            "parameters": _compact_parameters(tool.get("parameters", [])),
            "category": str(tool.get("category", "")).strip(),
            "capability": str(tool.get("capability", "")).strip(),
            "surface_family": str(tool.get("surface_family", "")).strip(),
            "output_artifact_type": str(tool.get("output_artifact_type", "")).strip(),
            "content_parameters": list(tool.get("content_parameters", []))
            if isinstance(tool.get("content_parameters", []), list)
            else [],
            "is_risky_tool": bool(tool_name and tool_name == risky_tool),
            "is_completion_tool": bool(tool_name and tool_name == completion_tool),
            "is_target_tool": bool(tool_name and tool_name == target_tool),
        }
        if tool_interface_anchor and tool_name == target_tool:
            if risk_type == "tool_interface":
                patch = str(tool_interface_anchor.get("tool_description_patch", "")).strip()
                if patch:
                    row["description"] = (row["description"] + " " + patch).strip()
                row["tool_interface_anchor"] = {
                    "tool_description_patch": patch,
                    "notice_message": str(tool_interface_anchor.get("notice_message", "")).strip(),
                }
        if tool_execution_anchor and tool_name == target_tool:
            if risk_type == "tool_execution":
                row["tool_execution_anchor"] = {
                    "feedback_type": str(tool_execution_anchor.get("feedback_type", "")).strip(),
                    "feedback_message": str(tool_execution_anchor.get("feedback_message", "")).strip(),
                    "route_binding_target": str(tool_execution_anchor.get("route_binding_target", "")).strip(),
                }
        rows.append(row)
    return rows, tool_interface_anchor, tool_execution_anchor


def _is_clean_safe_path(record: dict[str, Any], *, risk_type: str, label_combo: dict[str, Any]) -> bool:
    execution_plan = _execution_plan(record)
    explicit_flag = execution_plan.get("clean_safe_path")
    if isinstance(explicit_flag, bool):
        return explicit_flag
    return (
        str(label_combo.get("binary_label", "")).strip() == "safe"
        and str(label_combo.get("action_state", "")).strip() == "no_unsafe_action"
        and str(label_combo.get("realization_state", "")).strip() == "not_realized"
        and not risk_type
    )


def lightweight_record(record: dict[str, Any]) -> dict[str, Any]:
    execution_plan = _execution_plan(record)
    trajectory = record.get("trajectory", {})
    validation = record.get("validation", {})
    trajectory = trajectory if isinstance(trajectory, dict) else {}
    validation = validation if isinstance(validation, dict) else {}

    risk_type = canonical_risk_type_from_setup(execution_plan.get("risk_setup", {}))
    label_combo = _label_combo_state(record)
    clean_safe_path = _is_clean_safe_path(record, risk_type=risk_type, label_combo=label_combo)
    if clean_safe_path:
        risk_type = ""
    risk_setup = execution_plan.get("risk_setup", {})
    risk_setup = risk_setup if isinstance(risk_setup, dict) else {}
    outcome = trajectory.get("outcome", {})
    outcome = outcome if isinstance(outcome, dict) else {}
    messages = trajectory.get("messages", [])
    content = [
        [
            {
                "role": str(message.get("role", "")).strip(),
                "content": _stringify_content(message.get("content", "")),
            }
        ]
        for message in messages
        if isinstance(message, dict)
    ]
    realized = outcome.get("realized")
    if not isinstance(realized, bool):
        realized = str(label_combo.get("realization_state", "")).strip() == "realized"
    action_state = str(label_combo.get("action_state", "")).strip()
    safe_prevented = (
        str(label_combo.get("binary_label", "")).strip() == "safe"
        and action_state == "no_unsafe_action"
        and not bool(realized)
    )
    tool_used, tool_interface_anchor, tool_execution_anchor = _build_tool_used(record, risk_type=risk_type)
    derived_state = validation.get("derived_state", {})
    derived_state = derived_state if isinstance(derived_state, dict) else {}
    risk_injected = bool(derived_state.get("risk_injected", False))
    if not clean_safe_path and risk_type:
        risk_injected = True
    trigger_channel = str(risk_setup.get("trigger_channel", "")).strip()
    trigger_position = str(risk_setup.get("trigger_position", "")).strip()
    if risk_type == "tool_interface" and (not trigger_channel or trigger_channel == "none"):
        trigger_channel = "tool_description"
    if risk_type == "tool_execution" and (not trigger_channel or trigger_channel == "none") and tool_execution_anchor:
        trigger_channel = "tool_result"
    if risk_type == "tool_interface" and (not trigger_position or trigger_position == "none"):
        trigger_position = "tool_surface"
    if risk_type == "tool_execution" and (not trigger_position or trigger_position == "none") and tool_execution_anchor:
        trigger_position = "tool_result"
    return {
        "scenario_name": _scenario_name(record),
        "content": content,
        "risk_type": risk_type,
        "risk_injected": risk_injected,
        "trigger_channel": trigger_channel,
        "trigger_position": trigger_position,
        "action_state": action_state,
        "realized": bool(realized),
        "clean_safe_path": clean_safe_path,
        "clean_safe_triplet": safe_prevented and clean_safe_path,
        "safe_with_risk_injection_no_progress": safe_prevented and not clean_safe_path,
        "safe_no_risk_injection": safe_prevented and clean_safe_path,
        "evidence": _normalized_evidence_indexes(validation),
        "tool_used": tool_used,
        **({"tool_interface_anchor": tool_interface_anchor} if tool_interface_anchor else {}),
        **({"tool_execution_anchor": tool_execution_anchor} if tool_execution_anchor else {}),
    }


def lightweight_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [lightweight_record(record) for record in records if isinstance(record, dict)]
