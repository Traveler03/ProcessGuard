from __future__ import annotations

import json
from typing import Any

from .policy import find_tool_by_name
from .schema import ExecutionPlan


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def complete(response: str) -> str:
    return str(response)


def tool_call(name: str, arguments: dict[str, Any], *, assistant_response: str | None = None) -> str:
    payload: dict[str, Any] = {}
    if assistant_response is not None and str(assistant_response).strip():
        payload["assistant_response"] = str(assistant_response).strip()
    payload["name"] = name
    payload["arguments"] = arguments
    return compact_json(payload)


def append_message(messages: list[dict[str, str]], role: str, content: str) -> int:
    messages.append({"role": role, "content": content})
    return len(messages) - 1


def require_tool(plan: ExecutionPlan, name: str) -> dict[str, Any]:
    tool = find_tool_by_name(plan.tool_subset, name=name)
    if tool is None:
        raise ValueError(f"Execution plan references missing tool: {name}")
    return tool


def _placeholder_value(param: dict[str, Any]) -> Any:
    name = str(param.get("name", "value")).strip() or "value"
    readable_name = name.replace("_", " ")
    param_type = str(param.get("type", "string")).strip().lower()
    default_value = param.get("default_value")
    allowed_values = param.get("allowed_values", [])
    if default_value is not None:
        return default_value
    if isinstance(allowed_values, list) and allowed_values:
        return allowed_values[0]
    if param_type == "list":
        return [f"{readable_name} item"]
    if param_type in {"integer", "int"}:
        return 0
    if param_type in {"dictionary", "dict"}:
        return {"field": f"{readable_name} value"}
    return f"{readable_name} value"


def build_argument_template(tool: dict[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for param in tool.get("parameters", []):
        if not isinstance(param, dict):
            continue
        name = str(param.get("name", "")).strip()
        if not name:
            continue
        arguments[name] = _placeholder_value(param)
    return arguments
