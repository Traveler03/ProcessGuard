from __future__ import annotations

from .schema import LabelCombo


DEFAULT_RISKY_CAPABILITY_PRIORITY: tuple[str, ...] = (
    "send_or_upload",
    "read_then_disclose",
    "content_output",
)


def risky_capability_priority_for_combo(label_combo: LabelCombo) -> tuple[str, ...]:
    if label_combo.action_state == "unsafe_action_executed":
        return ("send_or_upload", "content_output", "read_then_disclose")
    if label_combo.action_state == "unsafe_action_attempted" and label_combo.realization_state == "realized":
        return ("content_output", "read_then_disclose", "send_or_upload")
    return ("send_or_upload", "read_then_disclose", "content_output")


def choose_risky_tool(tools: list[dict[str, object]], *, label_combo: LabelCombo) -> dict[str, object] | None:
    candidates = [
        tool
        for tool in tools
        if str(tool.get("capability", "")).strip() in DEFAULT_RISKY_CAPABILITY_PRIORITY
    ]
    if not candidates:
        return None
    send_like = [
        tool
        for tool in candidates
        if str(tool.get("capability", "")).strip() == "send_or_upload"
        and str(tool.get("surface_family", "")).strip() in {"send", "upload"}
    ]
    if send_like:
        return send_like[0]
    for capability in risky_capability_priority_for_combo(label_combo):
        for tool in candidates:
            if str(tool.get("capability", "")).strip() == capability:
                return tool
    return candidates[0]


def choose_benign_tool(
    tools: list[dict[str, object]],
    *,
    risky_tool_name: str,
) -> dict[str, object] | None:
    normalized_risky = str(risky_tool_name).strip()
    for tool in tools:
        if str(tool.get("name", "")).strip() != normalized_risky:
            return tool
    return tools[0] if tools else None


def find_tool_by_name(
    tools: list[dict[str, object]],
    *,
    name: str,
) -> dict[str, object] | None:
    normalized = str(name).strip()
    for tool in tools:
        if str(tool.get("name", "")).strip() == normalized:
            return tool
    return None
