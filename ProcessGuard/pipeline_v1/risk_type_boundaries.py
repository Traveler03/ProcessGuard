from __future__ import annotations

from dataclasses import dataclass

from .risk_taxonomy import canonical_risk_type_name
from .schema import ExecutionPlan


@dataclass(frozen=True)
class RiskTypeBoundarySpec:
    trigger_channel: str
    carrier_form: str
    attachment_kind: str
    anchor_object: str
    external_target_visibility_mode: str = "not_required"
    env_observation_type: str = ""
    inline_feedback_type: str = ""
    support_turn_types: tuple[str, ...] = ()
    max_support_turn_count: int | None = None
    direct_risky_tool_call: bool = False


RISK_TYPE_BOUNDARY_SPECS: dict[str, RiskTypeBoundarySpec] = {
    "user_driven": RiskTypeBoundarySpec(
        trigger_channel="user",
        carrier_form="user_follow_up",
        attachment_kind="follow_up_user",
        anchor_object="prior_artifact",
        external_target_visibility_mode="trigger_text",
        support_turn_types=("recipient_or_format_followup",),
    ),
    "environment_context": RiskTypeBoundarySpec(
        trigger_channel="environment",
        carrier_form="workflow_observation",
        attachment_kind="env_notice",
        anchor_object="environment",
        external_target_visibility_mode="metadata",
        env_observation_type="workflow_note",
        support_turn_types=("queue_or_status_followup",),
        direct_risky_tool_call=True,
    ),
    "tool_interface": RiskTypeBoundarySpec(
        trigger_channel="tool_description",
        carrier_form="capability_note",
        attachment_kind="tool_notice",
        anchor_object="target_tool",
        external_target_visibility_mode="metadata",
        support_turn_types=("benign_clarification",),
        direct_risky_tool_call=True,
    ),
    "tool_execution": RiskTypeBoundarySpec(
        trigger_channel="tool_result",
        carrier_form="result_feedback",
        attachment_kind="inline_result_feedback",
        anchor_object="latest_tool_result",
        external_target_visibility_mode="metadata",
        inline_feedback_type="tool_execution_feedback",
        max_support_turn_count=0,
        direct_risky_tool_call=True,
    ),
}


def boundary_spec_for_risk_type(risk_type: str) -> RiskTypeBoundarySpec:
    normalized = canonical_risk_type_name(risk_type)
    try:
        return RISK_TYPE_BOUNDARY_SPECS[normalized]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported risk type for boundary spec: {risk_type}") from exc


def expected_env_observation_type(risk_type: str) -> str:
    return boundary_spec_for_risk_type(risk_type).env_observation_type


def expected_inline_feedback_type(risk_type: str) -> str:
    return boundary_spec_for_risk_type(risk_type).inline_feedback_type


def external_target_visibility_mode_for_plan(plan: ExecutionPlan) -> str:
    if str(plan.branch_operator).strip() != "external_delivery":
        return "not_required"
    return boundary_spec_for_risk_type(plan.risk_setup.risk_type).external_target_visibility_mode


def plan_risk_type_boundary_reasons(plan: ExecutionPlan) -> list[str]:
    del plan
    return []


def use_direct_risky_tool_call(plan: ExecutionPlan) -> bool:
    return bool(
        plan.label_combo.binary_label == "unsafe"
        and str(plan.rollout_contract.shape).strip() == "direct_risky"
    )
