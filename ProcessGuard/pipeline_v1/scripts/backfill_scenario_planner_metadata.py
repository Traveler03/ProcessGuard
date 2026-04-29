from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import yaml


SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "config" / "scenarios_generated"

SEND_KEYWORDS = (
    "send",
    "email",
    "mail",
    "notify",
    "notification",
    "message",
    "post",
    "publish",
    "dispatch",
    "share",
    "alert",
    "reminder",
)
UPLOAD_KEYWORDS = (
    "upload",
    "attach",
    "attachment",
    "file",
    "document",
    "pdf",
)
READ_DISCLOSE_KEYWORDS = (
    "read",
    "parse",
    "extract",
    "identify",
    "analyze",
    "analyse",
    "summarize",
    "summary",
    "review",
    "scan",
    "check",
    "search",
    "fetch",
    "retrieve",
)
CONTENT_HINTS = (
    "body",
    "message",
    "content",
    "text",
    "summary",
    "details",
    "notes",
    "report",
    "analysis",
    "comment",
    "description",
    "prompt",
    "query",
    "supplemental",
)
NON_CONTENT_NAMES = {
    "recipient",
    "recipients",
    "to",
    "cc",
    "bcc",
    "subject",
    "ticker",
    "exchange",
    "priority",
    "due_date",
    "meeting_id",
    "task_id",
    "start_date",
    "end_date",
}


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def _tool_text(tool: dict[str, Any]) -> str:
    return " ".join(
        _normalize(value)
        for value in (
            tool.get("name", ""),
            tool.get("display_name", ""),
            tool.get("description", ""),
        )
        if str(value).strip()
    )


def _parameter_names(tool: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for param in tool.get("parameters", []) if isinstance(tool.get("parameters"), list) else []:
        if not isinstance(param, dict):
            continue
        name = str(param.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _content_parameter_names(tool: dict[str, Any]) -> list[str]:
    names = _parameter_names(tool)
    hinted: list[str] = []
    fallback: list[str] = []
    for name in names:
        lowered = name.lower()
        if lowered in NON_CONTENT_NAMES:
            continue
        if any(marker in lowered for marker in CONTENT_HINTS):
            hinted.append(name)
        else:
            fallback.append(name)
    picked = hinted or fallback[:1]
    deduped: list[str] = []
    seen: set[str] = set()
    for name in picked:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _infer_capability(tool: dict[str, Any]) -> str:
    text = _tool_text(tool)
    names = [name.lower() for name in _parameter_names(tool)]
    if any(keyword in text for keyword in SEND_KEYWORDS):
        return "send_or_upload"
    if any(keyword in text for keyword in UPLOAD_KEYWORDS):
        return "send_or_upload"
    if any(name in {"recipient", "recipients", "to", "email", "emails"} for name in names):
        return "send_or_upload"
    if any(keyword in text for keyword in READ_DISCLOSE_KEYWORDS):
        return "read_then_disclose"
    return "content_output"


def _infer_surface_family(tool: dict[str, Any], capability: str) -> str:
    text = _tool_text(tool)
    if capability == "send_or_upload":
        if any(keyword in text for keyword in UPLOAD_KEYWORDS):
            return "upload"
        return "send"
    if capability == "read_then_disclose":
        return "read"
    return "content_output"


def _infer_artifact_type(tool: dict[str, Any]) -> str:
    raw_name = str(tool.get("name", "")).strip()
    if raw_name:
        return f"{raw_name}_record"
    return "generated_record"


def _ensure_planner_semantics(tool: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    planner_semantics = tool.get("planner_semantics")
    if not isinstance(planner_semantics, dict):
        planner_semantics = {}
        tool["planner_semantics"] = planner_semantics
        changed = True

    capability = str(planner_semantics.get("capability", "")).strip()
    if not capability:
        capability = _infer_capability(tool)
        planner_semantics["capability"] = capability
        changed = True

    surface_family = str(planner_semantics.get("surface_family", "")).strip()
    if not surface_family:
        planner_semantics["surface_family"] = _infer_surface_family(tool, capability)
        changed = True

    artifact_type = str(planner_semantics.get("output_artifact_type", "")).strip()
    if not artifact_type:
        planner_semantics["output_artifact_type"] = _infer_artifact_type(tool)
        changed = True

    content_parameters = planner_semantics.get("content_parameters")
    if not isinstance(content_parameters, list) or not [str(item).strip() for item in content_parameters if str(item).strip()]:
        inferred = _content_parameter_names(tool)
        if inferred:
            planner_semantics["content_parameters"] = inferred
            changed = True

    return tool, changed


def _pick_workflow_constraints(tools: list[dict[str, Any]]) -> dict[str, Any]:
    names = [str(tool.get("name", "")).strip() for tool in tools if str(tool.get("name", "")).strip()]
    if not names:
        return {
            "start_tools": [],
            "completion_tools": [],
            "branchable_tools": [],
            "optional_mid_tools": [],
            "max_steps": 2,
        }

    def _capability(name: str) -> str:
        for tool in tools:
            if str(tool.get("name", "")).strip() == name:
                planner_semantics = tool.get("planner_semantics", {}) if isinstance(tool.get("planner_semantics"), dict) else {}
                return str(planner_semantics.get("capability", "")).strip()
        return ""

    completion_candidates = [name for name in names if _capability(name) == "send_or_upload"]
    if not completion_candidates:
        completion_candidates = [name for name in names if _capability(name) in {"content_output", "read_then_disclose"}]
    if not completion_candidates:
        completion_candidates = [names[0]]
    completion = completion_candidates[0]

    start_candidates = [name for name in names if name != completion]
    start_tools = [start_candidates[0]] if start_candidates else [completion]

    optional_mid_tools: list[str] = []
    for name in names:
        if name in start_tools or name == completion:
            continue
        optional_mid_tools.append(name)
        if len(optional_mid_tools) >= 2:
            break

    max_steps = 3 if optional_mid_tools else 2
    return {
        "start_tools": start_tools,
        "completion_tools": [completion],
        "branchable_tools": [completion],
        "optional_mid_tools": optional_mid_tools,
        "max_steps": max_steps,
    }


def _ensure_workflow_constraints(payload: dict[str, Any]) -> bool:
    changed = False
    planner_hints = payload.get("planner_hints")
    if not isinstance(planner_hints, dict):
        planner_hints = {}
        payload["planner_hints"] = planner_hints
        changed = True

    context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
    tools = context.get("available_tools", []) if isinstance(context.get("available_tools"), list) else []
    workflow_constraints = planner_hints.get("workflow_constraints")
    if not isinstance(workflow_constraints, dict):
        planner_hints["workflow_constraints"] = _pick_workflow_constraints(tools)
        return True

    defaults = _pick_workflow_constraints(tools)
    for key in ("start_tools", "completion_tools", "branchable_tools", "optional_mid_tools"):
        raw = workflow_constraints.get(key)
        normalized = [str(item).strip() for item in raw] if isinstance(raw, list) else []
        normalized = [item for item in normalized if item]
        if not normalized:
            workflow_constraints[key] = defaults[key]
            changed = True
            continue
        workflow_constraints[key] = normalized

    max_steps = int(workflow_constraints.get("max_steps", 0) or 0)
    if max_steps < 2:
        workflow_constraints["max_steps"] = defaults["max_steps"]
        changed = True

    completion_tools = set(workflow_constraints["completion_tools"])
    branchable_tools = set(workflow_constraints["branchable_tools"])
    if not (completion_tools & branchable_tools):
        workflow_constraints["branchable_tools"] = list(dict.fromkeys(workflow_constraints["branchable_tools"] + workflow_constraints["completion_tools"]))
        changed = True

    return changed


def _clean_identifier(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _ordered_tools_for_asset_inference(payload: dict[str, Any]) -> list[dict[str, Any]]:
    context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
    tools = context.get("available_tools", []) if isinstance(context.get("available_tools"), list) else []
    tool_rows = [tool for tool in tools if isinstance(tool, dict)]
    by_name = {
        str(tool.get("name", "")).strip(): tool
        for tool in tool_rows
        if str(tool.get("name", "")).strip()
    }

    planner_hints = payload.get("planner_hints", {}) if isinstance(payload.get("planner_hints"), dict) else {}
    workflow_constraints = (
        planner_hints.get("workflow_constraints", {})
        if isinstance(planner_hints.get("workflow_constraints"), dict)
        else {}
    )

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("start_tools", "optional_mid_tools", "completion_tools", "branchable_tools"):
        names = workflow_constraints.get(key, [])
        if not isinstance(names, list):
            continue
        for raw_name in names:
            name = str(raw_name).strip()
            if not name or name in seen:
                continue
            tool = by_name.get(name)
            if tool is not None:
                ordered.append(tool)
                seen.add(name)

    for tool in tool_rows:
        name = str(tool.get("name", "")).strip()
        if name and name in seen:
            continue
        ordered.append(tool)
    return ordered


def _infer_risk_asset_type(payload: dict[str, Any]) -> str:
    ordered_tools = _ordered_tools_for_asset_inference(payload)
    primary_artifact = ""
    fallback_artifact = ""

    for tool in ordered_tools:
        planner_semantics = tool.get("planner_semantics", {}) if isinstance(tool.get("planner_semantics"), dict) else {}
        capability = str(planner_semantics.get("capability", "")).strip()
        artifact = _clean_identifier(str(planner_semantics.get("output_artifact_type", "")).strip())
        if not artifact:
            continue
        if not fallback_artifact:
            fallback_artifact = artifact
        if capability != "send_or_upload":
            primary_artifact = artifact
            break

    artifact = primary_artifact or fallback_artifact
    if artifact:
        if artifact.endswith("_record"):
            return f"{artifact[:-7]}_material"
        if artifact.endswith(("_material", "_note", "_summary", "_correspondence", "_minutes")):
            return artifact
        return f"{artifact}_material"

    scenario_name = _clean_identifier(str(payload.get("name", "")).strip())
    if scenario_name:
        return f"{scenario_name}_material"
    return "workflow_material"


def _ensure_asset_constraints(payload: dict[str, Any]) -> bool:
    changed = False
    planner_hints = payload.get("planner_hints")
    if not isinstance(planner_hints, dict):
        planner_hints = {}
        payload["planner_hints"] = planner_hints
        changed = True

    asset_constraints = planner_hints.get("asset_constraints")
    if not isinstance(asset_constraints, dict):
        asset_constraints = {}
        planner_hints["asset_constraints"] = asset_constraints
        changed = True

    risk_asset_type = _clean_identifier(str(asset_constraints.get("risk_asset_type", "")).strip())
    if not risk_asset_type:
        asset_constraints["risk_asset_type"] = _infer_risk_asset_type(payload)
        changed = True
    elif risk_asset_type != str(asset_constraints.get("risk_asset_type", "")).strip():
        asset_constraints["risk_asset_type"] = risk_asset_type
        changed = True

    return changed


def backfill_scenario_file(path: Path) -> tuple[bool, str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"yaml_load_error:{exc}"
    if not isinstance(payload, dict):
        return False, "not_mapping"

    changed = False
    context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
    tools = context.get("available_tools", []) if isinstance(context.get("available_tools"), list) else []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        tool_after, tool_changed = _ensure_planner_semantics(tool)
        tools[index] = tool_after
        changed = changed or tool_changed
    if isinstance(context, dict):
        context["available_tools"] = tools
        payload["context"] = context

    changed = _ensure_workflow_constraints(payload) or changed
    changed = _ensure_asset_constraints(payload) or changed

    if changed:
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    return changed, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill planner_semantics, workflow_constraints, and asset_constraints for scenario YAML files."
    )
    parser.add_argument("--scenarios-dir", type=Path, default=SCENARIOS_DIR)
    args = parser.parse_args()

    scenarios_dir = args.scenarios_dir
    files = sorted(scenarios_dir.glob("*.yaml"))
    changed = 0
    failed: list[tuple[str, str]] = []
    for path in files:
        file_changed, status = backfill_scenario_file(path)
        if status != "ok":
            failed.append((path.name, status))
            continue
        if file_changed:
            changed += 1
    print(f"processed={len(files)} changed={changed} failed={len(failed)}")
    for name, error in failed[:20]:
        print(f"FAIL {name}: {error}")


if __name__ == "__main__":
    main()
