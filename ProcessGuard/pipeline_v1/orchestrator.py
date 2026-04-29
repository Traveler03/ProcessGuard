from __future__ import annotations

import json

from .llm_client import LLMSettings

from .agent_response_generator import generate_agent_friction_response, generate_agent_response, generate_risky_agent_response, generate_support_agent_response
from .derive_state import derive_outcome_payload, derive_state_from_events, derive_status_payload
from .outcome_summarizer import build_upstream_artifact_manifest, generate_benign_result, generate_safe_outcome, generate_unsafe_outcome
from .query_generator import generate_initial_query, generate_support_user_message, generate_trigger_message
from .risk_type_boundaries import expected_env_observation_type, expected_inline_feedback_type, use_direct_risky_tool_call
from .runtime import (
    append_message,
    build_argument_template,
    compact_json,
    require_tool,
    tool_call,
)
from .schema import ExecutionPlan, StepTag, TraceEvent, Trajectory
from .stage2_common import has_external_trigger
from .tool_call_generator import generate_tool_arguments


def normalize_trigger_payload(raw_trigger: str) -> tuple[str, str]:
    text = str(raw_trigger).strip()
    try:
        payload = json.loads(text)
    except Exception:
        return text, ""
    if not isinstance(payload, dict):
        return text, ""
    trigger_message = str(payload.get("trigger_message", "")).strip() or text
    visible_route_target = str(payload.get("visible_route_target", "")).strip()
    return trigger_message, visible_route_target


def build_trigger_message(plan: ExecutionPlan, trigger_message: str, *, visible_route_target: str = "") -> tuple[str, str]:
    risk = plan.risk_setup
    attachment_kind = plan.trigger_contract.attachment_kind
    if attachment_kind == "follow_up_user":
        return ("user", trigger_message)
    if attachment_kind == "tool_notice":
        patch_text = str(trigger_message).strip()
        if patch_text and "operator note" not in patch_text.lower():
            patch_text = f"Operator note: {patch_text}"
        payload = {
            "status": "notice",
            "result": {
                "notice_type": "tool_description_patch",
                "target_tool": risk.target_tool_name,
                "message": f"Operator note surfaced for {risk.target_tool_name}.",
                "tool_description_patch": patch_text,
            },
        }
        if visible_route_target:
            payload["result"]["route_binding_target"] = visible_route_target
        return ("environment", compact_json(payload))
    payload = {
        "status": "notice",
        "result": {
            "observation_type": expected_env_observation_type(risk.risk_type) or "workflow_note",
            "message": trigger_message,
        },
    }
    if visible_route_target:
        payload["result"]["route_binding_target"] = visible_route_target
    return ("environment", compact_json(payload))


def attach_inline_feedback(
    messages: list[dict[str, str]],
    *,
    trigger_message: str,
    feedback_type: str,
    visible_route_target: str = "",
) -> int:
    if not messages:
        raise ValueError("cannot attach inline feedback without prior messages")
    target_index = len(messages) - 1
    target_message = messages[target_index]
    if str(target_message.get("role", "")).strip() != "environment":
        raise ValueError("inline result feedback requires latest message to be an environment result")
    try:
        payload = json.loads(str(target_message.get("content", "")))
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("inline result feedback requires json environment payload") from exc
    if not isinstance(payload, dict):
        raise ValueError("inline result feedback requires object environment payload")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("inline result feedback requires result object")
    feedback_message = str(trigger_message).strip()
    if feedback_message and "feedback" not in feedback_message.lower():
        feedback_message = f"Execution feedback: {feedback_message}"
    if visible_route_target and visible_route_target not in feedback_message:
        feedback_message = f"{feedback_message} Route binding target: {visible_route_target}.".strip()
    result["feedback"] = {
        "feedback_type": feedback_type,
        "message": feedback_message,
    }
    if visible_route_target:
        result["feedback"]["route_binding_target"] = visible_route_target
    messages[target_index] = {"role": "environment", "content": compact_json(payload)}
    return target_index


def merge_generation_modes(*modes: str) -> str:
    cleaned = [str(mode).strip() for mode in modes if str(mode).strip()]
    if "repair" in cleaned:
        return "repair"
    if "llm" in cleaned:
        return "llm"
    return cleaned[-1] if cleaned else "llm"


def generate_trajectory(plan: ExecutionPlan, *, llm_settings: LLMSettings) -> Trajectory:
    if not llm_settings.enabled:
        raise RuntimeError("pipeline_v1 requires enabled LLM settings for Stage 2 generation")

    messages: list[dict[str, str]] = []
    step_tags: list[StepTag] = []
    tool_calls: list[dict[str, object]] = []
    events: list[TraceEvent] = []
    combo = plan.label_combo
    pre_branch_tool_names = list(plan.workflow_steps[:plan.branch_step_index])
    benign_prompt_tool_name = pre_branch_tool_names[0] if pre_branch_tool_names else plan.completion_tool_name
    benign_tool = require_tool(plan, benign_prompt_tool_name)
    completion_tool = require_tool(plan, plan.completion_tool_name)
    risky_tool = require_tool(plan, plan.risky_tool_name)
    benign_template = build_argument_template(benign_tool)
    completion_template = build_argument_template(completion_tool)
    risky_template = build_argument_template(risky_tool)

    query, query_mode = generate_initial_query(
        plan=plan,
        benign_tool=benign_tool,
        completion_tool=completion_tool,
        benign_argument_template=benign_template,
        completion_argument_template=completion_template,
        settings=llm_settings,
    )
    query_index = append_message(messages, "user", query)
    step_tags.append(StepTag(step_id="s1", tag="query", message_indexes=[query_index]))
    events.append(
        TraceEvent(
            event_id="e1",
            event_type="query_issued",
            actor="user",
            message_indexes=[query_index],
        )
    )
    generation_modes: list[str] = [query_mode]
    next_step_number = 2
    next_event_number = 2

    for tool_name in pre_branch_tool_names:
        tool = require_tool(plan, tool_name)
        argument_template = build_argument_template(tool)
        benign_args, benign_mode = generate_tool_arguments(
            plan=plan,
            tool=tool,
            tool_name=tool_name,
            argument_template=argument_template,
            messages_so_far=messages,
            phase="benign",
            settings=llm_settings,
        )
        benign_call_index = append_message(messages, "agent", tool_call(tool_name, benign_args))
        tool_calls.append({"name": tool_name, "arguments": benign_args, "message_index": benign_call_index, "kind": "benign"})
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="tool_call_emitted",
                actor="agent",
                message_indexes=[benign_call_index],
                tool_name=tool_name,
                details={"branch": "benign"},
            )
        )
        next_event_number += 1

        benign_result_message, benign_result_mode = generate_benign_result(
            plan=plan,
            tool_name=tool_name,
            tool=tool,
            messages_so_far=messages,
            settings=llm_settings,
            phase_key="benign_tool_result",
        )
        benign_result_payload = {
            "tool_name": tool_name,
            "message": benign_result_message,
        }
        artifact_manifest = build_upstream_artifact_manifest(plan=plan, tool_name=tool_name, tool=tool)
        if artifact_manifest:
            benign_result_payload["artifact_manifest"] = artifact_manifest
        benign_env_index = append_message(
            messages,
            "environment",
            compact_json(
                {
                    "status": "success",
                    "result": benign_result_payload,
                }
            ),
        )
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="tool_result_observed",
                actor="environment",
                message_indexes=[benign_env_index],
                tool_name=tool_name,
                details={"branch": "benign"},
            )
        )
        next_event_number += 1
        step_tags.append(
            StepTag(
                step_id=f"s{next_step_number}",
                tag="benign_tool_step",
                message_indexes=[benign_call_index, benign_env_index],
                evidence_roles=[],
            )
        )
        next_step_number += 1
        generation_modes.extend([benign_mode, benign_result_mode])

    support_turn_count = int(plan.context_scaffold.support_turn_count)
    support_turn_types = list(plan.context_scaffold.support_turn_types)
    if support_turn_count > 0:
        if not support_turn_types:
            raise ValueError("support turns requested without support_turn_types")
        if plan.trigger_contract.attachment_kind == "inline_result_feedback":
            raise ValueError("inline_result_feedback does not support pre-trigger support turns yet")
        for support_turn_offset in range(support_turn_count):
            support_turn_type = support_turn_types[support_turn_offset % len(support_turn_types)]
            support_user_message, support_user_mode = generate_support_user_message(
                plan=plan,
                messages_so_far=messages,
                support_turn_type=support_turn_type,
                settings=llm_settings,
            )
            support_user_index = append_message(messages, "user", support_user_message)
            events.append(
                TraceEvent(
                    event_id=f"e{next_event_number}",
                    event_type="support_turn_observed",
                    actor="user",
                    message_indexes=[support_user_index],
                    details={"support_turn_type": support_turn_type},
                )
            )
            next_event_number += 1
            support_agent_message, support_agent_mode = generate_support_agent_response(
                plan=plan,
                messages_so_far=messages,
                support_turn_type=support_turn_type,
                settings=llm_settings,
            )
            support_agent_index = append_message(messages, "agent", support_agent_message)
            events.append(
                TraceEvent(
                    event_id=f"e{next_event_number}",
                    event_type="support_turn_replied",
                    actor="agent",
                    message_indexes=[support_agent_index],
                    details={"support_turn_type": support_turn_type},
                )
            )
            next_event_number += 1
            step_tags.append(
                StepTag(
                    step_id=f"s{next_step_number}",
                    tag="support_turn",
                    message_indexes=[support_user_index, support_agent_index],
                )
            )
            next_step_number += 1
            generation_modes.extend([support_user_mode, support_agent_mode])

    if has_external_trigger(plan):
        raw_trigger_payload, trigger_mode = generate_trigger_message(
            plan=plan,
            messages_so_far=messages,
            settings=llm_settings,
        )
        trigger_message, visible_route_target = normalize_trigger_payload(raw_trigger_payload)
        if plan.trigger_contract.attachment_kind == "inline_result_feedback":
            trigger_role = "environment"
            trigger_index = attach_inline_feedback(
                messages,
                trigger_message=trigger_message,
                feedback_type=expected_inline_feedback_type(plan.risk_setup.risk_type) or "tool_result_feedback",
                visible_route_target=visible_route_target,
            )
        else:
            trigger_role, trigger_content = build_trigger_message(
                plan,
                trigger_message,
                visible_route_target=visible_route_target,
            )
            trigger_index = append_message(messages, trigger_role, trigger_content)
        step_tags.append(
            StepTag(
                step_id=f"s{next_step_number}",
                tag="risk_trigger",
                message_indexes=[trigger_index],
                evidence_roles=["trigger"],
            )
        )
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="risk_trigger_observed",
                actor=trigger_role,
                message_indexes=[trigger_index],
                tool_name=plan.risk_setup.target_tool_name,
                details={
                    "branch_operator": plan.branch_operator,
                    "drift_layers": list(plan.drift_spec.drift_layers),
                },
            )
        )
        next_event_number += 1
        next_step_number += 1
        generation_modes.append(trigger_mode)

    if combo.action_state == "no_unsafe_action":
        agent_response, response_mode = generate_agent_response(
            plan=plan,
            messages_so_far=messages,
            settings=llm_settings,
        )
        safe_response_index = append_message(messages, "agent", agent_response)
        step_tags.append(
            StepTag(
                step_id=f"s{next_step_number}",
                tag="safe_response",
                message_indexes=[safe_response_index],
                evidence_roles=["guard_action"],
            )
        )
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="guard_reply_emitted",
                actor="agent",
                message_indexes=[safe_response_index],
                details={"branch": "safe"},
            )
        )
        next_event_number += 1
        next_step_number += 1
        generation_modes.append(response_mode)

        response_context = messages

        completion_args, completion_mode = generate_tool_arguments(
            plan=plan,
            tool=completion_tool,
            tool_name=plan.completion_tool_name,
            argument_template=completion_template,
            messages_so_far=response_context,
            phase="completion",
            settings=llm_settings,
        )
        completion_call_index = append_message(
            messages,
            "agent",
            tool_call(plan.completion_tool_name, completion_args),
        )
        tool_calls.append(
            {
                "name": plan.completion_tool_name,
                "arguments": completion_args,
                "message_index": completion_call_index,
                "kind": "completion",
            }
        )
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="tool_call_emitted",
                actor="agent",
                message_indexes=[completion_call_index],
                tool_name=plan.completion_tool_name,
                details={"branch": "completion"},
            )
        )
        next_event_number += 1
        next_step_number += 1
        step_tags.append(
            StepTag(
                step_id=f"s{next_step_number}",
                tag="completion_tool_step",
                message_indexes=[],
            )
        )
        generation_modes.append(completion_mode)

        completion_result_message, completion_result_mode = generate_benign_result(
            plan=plan,
            tool_name=plan.completion_tool_name,
            tool=completion_tool,
            messages_so_far=messages,
            settings=llm_settings,
            phase_key="completion_tool_result",
        )
        completion_result_payload = compact_json(
            {
                "status": "success",
                "result": {
                    "tool_name": plan.completion_tool_name,
                    "message": completion_result_message,
                },
            }
        )
        completion_result_index = append_message(
            messages,
            "environment",
            completion_result_payload,
        )
        completion_context = messages[:-1] + [{"role": "environment", "content": completion_result_payload}]
        safe_outcome_message, safe_outcome_mode = generate_safe_outcome(
            plan=plan,
            messages_so_far=completion_context,
            settings=llm_settings,
        )
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="tool_result_observed",
                actor="environment",
                message_indexes=[completion_result_index],
                tool_name=plan.completion_tool_name,
                details={"branch": "completion"},
            )
        )
        next_event_number += 1
        step_tags[-1].message_indexes = [completion_call_index, completion_result_index]
        next_step_number += 1
        generation_modes.append(completion_result_mode)

        safe_outcome_index = append_message(
            messages,
            "environment",
            compact_json({"status": "notice", "result": {"message": safe_outcome_message}}),
        )
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="output_not_realized",
                actor="environment",
                message_indexes=[safe_outcome_index],
                tool_name=plan.completion_tool_name,
                details={"harm_target": plan.risk_setup.harm_target},
            )
        )
        next_event_number += 1
        step_tags.append(
            StepTag(
                step_id=f"s{next_step_number}",
                tag="safe_outcome",
                message_indexes=[safe_outcome_index],
                evidence_roles=["guard_outcome", "non_realization_outcome"],
            )
        )
        next_step_number += 1
        generation_modes.append(safe_outcome_mode)
        generation_mode = merge_generation_modes(*generation_modes)
    else:
        if plan.rollout_contract.uses_agent_friction:
            friction_response, friction_mode = generate_agent_friction_response(
                plan=plan,
                messages_so_far=messages,
                settings=llm_settings,
            )
            friction_index = append_message(messages, "agent", friction_response)
            step_tags.append(
                StepTag(
                    step_id=f"s{next_step_number}",
                    tag="agent_friction",
                    message_indexes=[friction_index],
                )
            )
            events.append(
                TraceEvent(
                    event_id=f"e{next_event_number}",
                    event_type="agent_friction_emitted",
                    actor="agent",
                    message_indexes=[friction_index],
                    details={"branch": "risky"},
                )
            )
            next_event_number += 1
            next_step_number += 1
            generation_modes.append(friction_mode)
        risky_assistant_response = ""
        if plan.rollout_contract.uses_agent_friction or use_direct_risky_tool_call(plan):
            risky_context = list(messages)
        else:
            risky_assistant_response, risky_response_mode = generate_risky_agent_response(
                plan=plan,
                messages_so_far=messages,
                settings=llm_settings,
            )
            risky_context = messages + [{"role": "agent", "content": risky_assistant_response}]
            generation_modes.append(risky_response_mode)
        risky_args, risky_mode = generate_tool_arguments(
            plan=plan,
            tool=risky_tool,
            tool_name=plan.risky_tool_name,
            argument_template=risky_template,
            messages_so_far=risky_context,
            phase="risky",
            settings=llm_settings,
        )
        risky_call_index = append_message(
            messages,
            "agent",
            tool_call(
                plan.risky_tool_name,
                risky_args,
                assistant_response=risky_assistant_response or None,
            ),
        )
        tool_calls.append({"name": plan.risky_tool_name, "arguments": risky_args, "message_index": risky_call_index, "kind": "risky"})
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="risky_intent_emitted",
                actor="agent",
                message_indexes=[risky_call_index],
                tool_name=plan.risky_tool_name,
                details={"branch": "risky"},
            )
        )
        next_event_number += 1
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type="tool_call_emitted",
                actor="agent",
                message_indexes=[risky_call_index],
                tool_name=plan.risky_tool_name,
                details={"branch": "risky"},
            )
        )
        next_event_number += 1
        step_tags.append(
            StepTag(
                step_id=f"s{next_step_number}",
                tag="risky_action",
                message_indexes=[risky_call_index],
                evidence_roles=["risky_action"],
            )
        )
        next_step_number += 1
        generation_modes.append(risky_mode)

        unsafe_outcome_message, acceptance_state, visibility_state, unsafe_outcome_mode = generate_unsafe_outcome(
            plan=plan,
            messages_so_far=messages,
            settings=llm_settings,
        )
        evidence_roles = {
            "not_visible": ["non_realization_outcome"],
            "partial_visible": ["partial_outcome"],
            "quarantined": ["execution_outcome", "non_realization_outcome"],
            "finalized_visible": ["execution_outcome", "harm_outcome"],
        }[visibility_state]
        outcome_index = append_message(
            messages,
            "environment",
            compact_json({"status": "update", "result": {"message": unsafe_outcome_message}}),
        )
        if acceptance_state == "accepted_by_environment":
            events.append(
                TraceEvent(
                    event_id=f"e{next_event_number}",
                    event_type="environment_accept",
                    actor="environment",
                    message_indexes=[outcome_index],
                    tool_name=plan.risky_tool_name,
                    details={"branch": "risky"},
                )
            )
            next_event_number += 1
        else:
            events.append(
                TraceEvent(
                    event_id=f"e{next_event_number}",
                    event_type="pre_execution_block",
                    actor="environment",
                    message_indexes=[outcome_index],
                    tool_name=plan.risky_tool_name,
                    details={"branch": "risky"},
                )
            )
            next_event_number += 1
        exposure_event_type = {
            "not_visible": "output_not_realized",
            "partial_visible": "output_partial",
            "quarantined": "output_quarantined",
            "finalized_visible": "output_finalized_visible",
        }[visibility_state]
        events.append(
            TraceEvent(
                event_id=f"e{next_event_number}",
                event_type=exposure_event_type,
                actor="environment",
                message_indexes=[outcome_index],
                tool_name=plan.risky_tool_name,
                details={
                    "harm_target": plan.risk_setup.harm_target,
                    "acceptance_state": acceptance_state,
                    "visibility_state": visibility_state,
                },
            )
        )
        next_event_number += 1
        step_tags.append(
            StepTag(
                step_id=f"s{next_step_number}",
                tag="outcome_step",
                message_indexes=[outcome_index],
                evidence_roles=evidence_roles,
            )
        )
        generation_modes.append(unsafe_outcome_mode)
        generation_mode = merge_generation_modes(*generation_modes)

    risk_injected_override = bool(str(plan.risk_setup.risk_type).strip()) and not bool(plan.clean_safe_path)
    derived_state = derive_state_from_events(
        events,
        risk_injected_override=risk_injected_override,
    )
    status = derive_status_payload(derived_state)
    outcome = derive_outcome_payload(derived_state)

    return Trajectory(
        query=query,
        messages=messages,
        tool_calls=tool_calls,
        events=events,
        outcome=outcome,
        step_tags=step_tags,
        status=status,
        generation_mode=generation_mode,
    )
