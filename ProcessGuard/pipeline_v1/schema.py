from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Final

from .risk_taxonomy import RISK_TYPES, canonical_risk_type_name


ACTION_STATES: Final[tuple[str, ...]] = (
    "no_unsafe_action",
    "unsafe_action_attempted",
    "unsafe_action_executed",
)
REALIZATION_STATES: Final[tuple[str, ...]] = (
    "not_realized",
    "realized",
)
BINARY_LABELS: Final[tuple[str, ...]] = ("safe", "unsafe")
BRANCH_OPERATORS: Final[tuple[str, ...]] = ("external_delivery", "content_exposure")
DRIFT_LAYERS: Final[tuple[str, ...]] = (
    "tool_selection",
    "recipient_target",
    "payload_content",
    "routing_path",
    "output_inclusion",
)
ACTION_TRACE_STATES: Final[tuple[str, ...]] = ("not_started", "attempted", "executed")
EXPOSURE_TRACE_STATES: Final[tuple[str, ...]] = (
    "not_realized",
    "partial",
    "finalized_visible",
    "quarantined",
)
ROLLOUT_SHAPES: Final[tuple[str, ...]] = (
    "safe_completion",
    "direct_risky",
    "friction_then_risky",
)

LABEL_COMBO_SPECS: Final[dict[str, dict[str, str]]] = {
    "safe_prevented": {
        "binary_label": "safe",
        "action_state": "no_unsafe_action",
        "realization_state": "not_realized",
    },
    "unsafe_attempted_not_realized": {
        "binary_label": "unsafe",
        "action_state": "unsafe_action_attempted",
        "realization_state": "not_realized",
    },
    "unsafe_attempted_realized": {
        "binary_label": "unsafe",
        "action_state": "unsafe_action_attempted",
        "realization_state": "realized",
    },
    "unsafe_executed_not_realized": {
        "binary_label": "unsafe",
        "action_state": "unsafe_action_executed",
        "realization_state": "not_realized",
    },
    "unsafe_executed_realized": {
        "binary_label": "unsafe",
        "action_state": "unsafe_action_executed",
        "realization_state": "realized",
    },
}

REQUIRED_EVIDENCE_ROLES_BY_COMBO: Final[dict[str, list[str]]] = {
    "safe_prevented": ["trigger", "guard_outcome", "non_realization_outcome"],
    "unsafe_attempted_not_realized": ["trigger", "risky_action", "non_realization_outcome"],
    "unsafe_attempted_realized": ["trigger", "risky_action", "partial_outcome"],
    "unsafe_executed_not_realized": ["trigger", "risky_action", "execution_outcome", "non_realization_outcome"],
    "unsafe_executed_realized": ["trigger", "risky_action", "execution_outcome", "harm_outcome"],
}


@dataclass(frozen=True)
class LabelCombo:
    slug: str
    binary_label: str
    action_state: str
    realization_state: str

    def __post_init__(self) -> None:
        if self.binary_label not in BINARY_LABELS:
            raise ValueError(f"Unsupported binary_label: {self.binary_label}")
        if self.action_state not in ACTION_STATES:
            raise ValueError(f"Unsupported action_state: {self.action_state}")
        if self.realization_state not in REALIZATION_STATES:
            raise ValueError(f"Unsupported realization_state: {self.realization_state}")


@dataclass
class RiskSetup:
    risk_type: str
    trigger_channel: str
    trigger_position: str
    risk_asset: str
    harm_target: str
    target_kind: str
    payload_brief: str
    target_tool_name: str = ""
    risk_payload_facts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        normalized_risk_type = canonical_risk_type_name(self.risk_type)
        if not normalized_risk_type:
            self.risk_type = ""
            return
        if normalized_risk_type not in RISK_TYPES:
            raise ValueError(f"Unsupported risk type: {self.risk_type}")
        self.risk_type = normalized_risk_type


@dataclass
class DriftSpec:
    branch_operator: str
    drift_layers: list[str]
    branch_tool_name: str
    benign_surface: str
    risky_surface: str

    def __post_init__(self) -> None:
        if self.branch_operator not in BRANCH_OPERATORS:
            raise ValueError(f"Unsupported branch operator: {self.branch_operator}")
        invalid = [layer for layer in self.drift_layers if layer not in DRIFT_LAYERS]
        if invalid:
            raise ValueError(f"Unsupported drift layers: {invalid}")


@dataclass
class TriggerContract:
    carrier_form: str
    attachment_kind: str
    motive_family: str
    max_information_load: int = 1
    required_signals: list[str] = field(default_factory=list)
    forbidden_signal_combos: list[str] = field(default_factory=list)


@dataclass
class ContextScaffold:
    support_turn_count: int = 0
    support_turn_types: list[str] = field(default_factory=list)
    anchor_object: str = ""


@dataclass
class ReasoningContract:
    allowed_affordance_sources: list[str] = field(default_factory=list)
    forbidden_inference_patterns: list[str] = field(default_factory=list)
    must_reference_trigger: bool = False


@dataclass
class RolloutContract:
    shape: str = "safe_completion"

    def __post_init__(self) -> None:
        if self.shape not in ROLLOUT_SHAPES:
            raise ValueError(f"Unsupported rollout shape: {self.shape}")

    @property
    def uses_agent_friction(self) -> bool:
        return self.shape == "friction_then_risky"


@dataclass
class SurfaceProfile:
    trigger_register: str = ""
    trigger_wrapper: str = ""
    trigger_lexical_family: str = ""
    safe_response_register: str = ""
    risky_response_register: str = ""
    friction_response_register: str = ""
    support_response_register: str = ""
    benign_result_register: str = ""
    safe_outcome_register: str = ""
    unsafe_outcome_register: str = ""
    artifact_noun_family: str = ""
    handoff_verb_family: str = ""


@dataclass
class ExecutionBlueprintStep:
    step_id: str
    phase_key: str
    step_tag: str
    actor: str
    objective: str
    anchor: str = ""
    render_mode: str = ""
    depends_on: list[str] = field(default_factory=list)
    knowledge_sources: list[str] = field(default_factory=list)
    newly_visible: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_avoid: list[str] = field(default_factory=list)


@dataclass
class TraceEvent:
    event_id: str
    event_type: str
    actor: str
    message_indexes: list[int] = field(default_factory=list)
    tool_name: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DerivedState:
    action_trace_state: str
    exposure_trace_state: str
    risk_injected: bool
    observed: bool

    def __post_init__(self) -> None:
        if self.action_trace_state not in ACTION_TRACE_STATES:
            raise ValueError(f"Unsupported action trace state: {self.action_trace_state}")
        if self.exposure_trace_state not in EXPOSURE_TRACE_STATES:
            raise ValueError(f"Unsupported exposure trace state: {self.exposure_trace_state}")


@dataclass
class ExecutionPlan:
    scenario_id: str
    label_combo: LabelCombo
    benign_task: str
    risk_setup: RiskSetup
    drift_spec: DriftSpec
    tool_subset: list[dict[str, Any]]
    workflow_steps: list[str]
    branch_step_index: int
    branch_operator: str
    benign_tool_name: str
    completion_tool_name: str
    risky_tool_name: str
    plan_sketch: list[str]
    required_evidence_roles: list[str]
    trigger_contract: TriggerContract
    context_scaffold: ContextScaffold
    reasoning_contract: ReasoningContract
    rollout_contract: RolloutContract
    clean_safe_path: bool = False
    surface_profile: SurfaceProfile = field(default_factory=SurfaceProfile)
    execution_blueprint: list[ExecutionBlueprintStep] = field(default_factory=list)
    config_path: str = ""
    planning_mode: str = "deterministic"
    seed: int = 0


@dataclass
class StepTag:
    step_id: str
    tag: str
    message_indexes: list[int]
    evidence_roles: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    query: str
    messages: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    events: list[TraceEvent]
    outcome: dict[str, Any]
    step_tags: list[StepTag]
    status: dict[str, Any]
    generation_mode: str = "rule"


@dataclass
class ValidationResult:
    accepted: bool
    reasons: list[str]
    derived_label_combo: LabelCombo
    derived_state: DerivedState
    evidence_indexes: dict[str, list[int]]


def parse_label_combo(raw: str | dict[str, str] | LabelCombo) -> LabelCombo:
    if isinstance(raw, LabelCombo):
        return raw
    if isinstance(raw, str):
        spec = LABEL_COMBO_SPECS.get(raw.strip())
        if spec is None:
            raise ValueError(f"Unknown label combo: {raw}")
        return LabelCombo(slug=raw.strip(), **spec)
    if isinstance(raw, dict):
        slug = str(raw.get("slug", "")).strip()
        if slug and slug in LABEL_COMBO_SPECS:
            return LabelCombo(slug=slug, **LABEL_COMBO_SPECS[slug])
        binary_label = str(raw.get("binary_label", "")).strip()
        action_state = str(raw.get("action_state", "")).strip()
        realization_state = str(raw.get("realization_state", "")).strip()
        for name, spec in LABEL_COMBO_SPECS.items():
            if (
                spec["binary_label"] == binary_label
                and spec["action_state"] == action_state
                and spec["realization_state"] == realization_state
            ):
                return LabelCombo(slug=name, **spec)
    raise ValueError(f"Unsupported label combo payload: {raw}")


def required_evidence_roles_for_combo(
    combo: LabelCombo,
    *,
    risk_type: str = "",
    has_trigger: bool = True,
) -> list[str]:
    roles = list(REQUIRED_EVIDENCE_ROLES_BY_COMBO[combo.slug])
    if not has_trigger:
        roles = [role for role in roles if role != "trigger"]
    return roles


def plan_sketch_for_combo(
    combo: LabelCombo,
    *,
    pre_branch_steps: int = 1,
    support_turn_steps: int = 0,
    rollout_shape: str = "direct_risky",
    has_trigger: bool = True,
) -> list[str]:
    prefix = (
        ["query"]
        + (["benign_tool_step"] * max(0, int(pre_branch_steps)))
        + (["support_turn"] * max(0, int(support_turn_steps)))
    )
    if has_trigger:
        prefix += ["risk_trigger"]
    if combo.slug == "safe_prevented":
        return prefix + ["safe_response", "completion_tool_step", "safe_outcome"]
    if rollout_shape == "friction_then_risky":
        return prefix + ["agent_friction", "risky_action", "outcome_step"]
    return prefix + ["risky_action", "outcome_step"]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
