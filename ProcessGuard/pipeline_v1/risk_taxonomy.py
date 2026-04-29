from __future__ import annotations

from typing import Any, Final


RISK_TYPES: Final[tuple[str, ...]] = (
    "user_driven",
    "environment_context",
    "tool_interface",
    "tool_execution",
)

LEGACY_RISK_SUBTYPE_TO_RISK_TYPE: Final[dict[str, str]] = {
    "malicious_user_instruction_or_jailbreak": "user_driven",
    "direct_prompt_injection": "user_driven",
    "environmental_observation": "environment_context",
    "unreliable_or_mis_information": "environment_context",
    "tool_description_injection": "tool_interface",
    "corrupted_tool_feedback": "tool_interface",
    "malicious_tool_execution": "tool_execution",
}

def risk_type_for_subtype(subtype: str) -> str:
    return LEGACY_RISK_SUBTYPE_TO_RISK_TYPE.get(str(subtype).strip(), "")


def canonical_risk_type_name(value: str) -> str:
    normalized = str(value).strip()
    if normalized in RISK_TYPES:
        return normalized
    return risk_type_for_subtype(normalized)


def canonical_risk_type_from_setup(risk_setup: dict[str, Any] | None) -> str:
    payload = risk_setup if isinstance(risk_setup, dict) else {}
    raw_type = str(payload.get("risk_type", "")).strip()
    if raw_type:
        canonical = canonical_risk_type_name(raw_type)
        if canonical:
            return canonical
    return canonical_risk_type_name(str(payload.get("risk_subtype", "")).strip())


def canonical_risk_type_and_subtype(risk_setup: dict[str, Any] | None) -> tuple[str, str]:
    payload = risk_setup if isinstance(risk_setup, dict) else {}
    raw_type = str(payload.get("risk_type", "")).strip()
    raw_subtype = str(payload.get("risk_subtype", "")).strip()
    if raw_subtype:
        return canonical_risk_type_name(raw_type or raw_subtype), raw_subtype
    if raw_type:
        canonical = canonical_risk_type_name(raw_type)
        if canonical == raw_type:
            return canonical, ""
        return canonical, raw_type
    return "", ""
