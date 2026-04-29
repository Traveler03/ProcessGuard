from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .llm_client import LLMSettings, llm_available, load_llm_settings

from .risk_type_boundaries import boundary_spec_for_risk_type
from .scenarios import PIPELINE_ROOT
from .schema import LABEL_COMBO_SPECS
CONFIG_ROOT = PIPELINE_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "generation.yaml"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    return []


def _unique_strings(*groups: Any) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in _string_list(group):
            if item not in merged:
                merged.append(item)
    return merged


def _first_non_empty_string(*values: Any) -> str:
    for value in values:
        normalized = str(value).strip()
        if normalized:
            return normalized
    return ""


@lru_cache(maxsize=8)
def load_generation_config(config_path: str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Generation config must be a mapping: {path}")
    return payload


def load_pipeline_semantics(config_path: Path | None = None) -> dict[str, Any]:
    config = load_generation_config(str(config_path) if config_path else None)
    semantics = config.get("semantics", {})
    if not isinstance(semantics, dict):
        raise ValueError("generation config semantics must be a mapping")

    branch_rules = semantics.get("branch_operator_rules", [])
    if not isinstance(branch_rules, list) or not branch_rules:
        raise ValueError("generation config semantics.branch_operator_rules must be a non-empty list")
    normalized_branch_rules: list[dict[str, Any]] = []
    for index, rule in enumerate(branch_rules):
        if not isinstance(rule, dict):
            raise ValueError(f"semantics.branch_operator_rules[{index}] must be a mapping")
        branch_operator = str(rule.get("branch_operator", "")).strip()
        if not branch_operator:
            raise ValueError(f"semantics.branch_operator_rules[{index}].branch_operator is required")
        capability_any_of = [
            str(item).strip()
            for item in rule.get("capability_any_of", [])
            if str(item).strip()
        ] if isinstance(rule.get("capability_any_of"), list) else []
        surface_family_any_of = [
            str(item).strip()
            for item in rule.get("surface_family_any_of", [])
            if str(item).strip()
        ] if isinstance(rule.get("surface_family_any_of"), list) else []
        require_content_parameters = bool(rule.get("require_content_parameters", False))
        if not capability_any_of and not surface_family_any_of:
            raise ValueError(
                f"semantics.branch_operator_rules[{index}] must define capability_any_of or surface_family_any_of"
            )
        normalized_branch_rules.append(
            {
                "branch_operator": branch_operator,
                "capability_any_of": capability_any_of,
                "surface_family_any_of": surface_family_any_of,
                "require_content_parameters": require_content_parameters,
            }
        )

    compatibility = semantics.get("risk_type_compatibility", {})
    if not isinstance(compatibility, dict):
        raise ValueError("generation config semantics.risk_type_compatibility must be a mapping")

    raw_disallowed = compatibility.get("disallowed_by_branch_operator", {})
    if not isinstance(raw_disallowed, dict):
        raise ValueError("semantics.risk_type_compatibility.disallowed_by_branch_operator must be a mapping")
    disallowed_by_branch_operator: dict[str, list[str]] = {}
    for branch_operator, values in raw_disallowed.items():
        if not isinstance(values, list):
            raise ValueError(
                "semantics.risk_type_compatibility.disallowed_by_branch_operator values must be lists"
            )
        disallowed_by_branch_operator[str(branch_operator).strip()] = [
            str(item).strip() for item in values if str(item).strip()
        ]

    raw_allowed_capabilities = compatibility.get("allowed_capabilities_by_risk_type", {})
    if not isinstance(raw_allowed_capabilities, dict):
        raise ValueError("semantics.risk_type_compatibility.allowed_capabilities_by_risk_type must be a mapping")
    allowed_capabilities_by_risk_type: dict[str, list[str]] = {}
    for risk_type, values in raw_allowed_capabilities.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                "semantics.risk_type_compatibility.allowed_capabilities_by_risk_type values must be non-empty lists"
            )
        allowed_capabilities_by_risk_type[str(risk_type).strip()] = [
            str(item).strip() for item in values if str(item).strip()
        ]

    raw_combo_semantics = semantics.get("combo_semantics", {})
    if raw_combo_semantics and not isinstance(raw_combo_semantics, dict):
        raise ValueError("generation config semantics.combo_semantics must be a mapping")
    raw_axis_definitions = raw_combo_semantics.get("axis_definitions", {}) if isinstance(raw_combo_semantics, dict) else {}
    if raw_axis_definitions and not isinstance(raw_axis_definitions, dict):
        raise ValueError("generation config semantics.combo_semantics.axis_definitions must be a mapping")
    axis_definitions = {
        str(key).strip(): str(value).strip()
        for key, value in raw_axis_definitions.items()
        if str(key).strip() and str(value).strip()
    }
    decision_rule = str(raw_combo_semantics.get("decision_rule", "")).strip() if isinstance(raw_combo_semantics, dict) else ""
    raw_invalid_rules = raw_combo_semantics.get("invalid_rules", []) if isinstance(raw_combo_semantics, dict) else []
    if raw_invalid_rules and not isinstance(raw_invalid_rules, list):
        raise ValueError("generation config semantics.combo_semantics.invalid_rules must be a list")
    invalid_rules = _string_list(raw_invalid_rules)
    raw_combo_definitions = raw_combo_semantics.get("combos", {}) if isinstance(raw_combo_semantics, dict) else {}
    if raw_combo_definitions and not isinstance(raw_combo_definitions, dict):
        raise ValueError("generation config semantics.combo_semantics.combos must be a mapping")
    combo_semantics: dict[str, Any] = {
        "axis_definitions": axis_definitions,
        "decision_rule": decision_rule,
        "invalid_rules": invalid_rules,
        "combos": {},
    }
    if raw_combo_definitions:
        missing = sorted(set(LABEL_COMBO_SPECS) - {str(key).strip() for key in raw_combo_definitions})
        if missing:
            raise ValueError(f"generation config semantics.combo_semantics.combos missing canonical combos: {missing}")
        extra = sorted(
            str(key).strip()
            for key in raw_combo_definitions
            if str(key).strip() and str(key).strip() not in LABEL_COMBO_SPECS
        )
        if extra:
            raise ValueError(f"generation config semantics.combo_semantics.combos contain unknown combos: {extra}")
        normalized_combos: dict[str, dict[str, Any]] = {}
        for combo_slug, combo_profile in raw_combo_definitions.items():
            normalized_slug = str(combo_slug).strip()
            if not isinstance(combo_profile, dict):
                raise ValueError(f"generation config semantics.combo_semantics.combos.{normalized_slug} must be a mapping")
            definition = str(combo_profile.get("definition", "")).strip()
            if not definition:
                raise ValueError(f"generation config semantics.combo_semantics.combos.{normalized_slug}.definition must be non-empty")
            normalized_combos[normalized_slug] = {
                "definition": definition,
                "must_satisfy": _string_list(combo_profile.get("must_satisfy", [])),
                "key_constraint": str(combo_profile.get("key_constraint", "")).strip(),
                "plain_meaning": str(combo_profile.get("plain_meaning", "")).strip(),
            }
        combo_semantics["combos"] = normalized_combos

    stage2_contracts = semantics.get("stage2_contracts", {})
    if not isinstance(stage2_contracts, dict):
        raise ValueError("generation config semantics.stage2_contracts must be a mapping")
    defaults = stage2_contracts.get("defaults", {})
    if defaults and not isinstance(defaults, dict):
        raise ValueError("generation config semantics.stage2_contracts.defaults must be a mapping")
    raw_motive_prompt_hints = stage2_contracts.get("motive_prompt_hints", {})
    if raw_motive_prompt_hints and not isinstance(raw_motive_prompt_hints, dict):
        raise ValueError("generation config semantics.stage2_contracts.motive_prompt_hints must be a mapping")
    motive_prompt_hints = {
        str(key).strip(): str(value).strip()
        for key, value in raw_motive_prompt_hints.items()
        if str(key).strip() and str(value).strip()
    }
    trigger_max_information_load = int(defaults.get("trigger_max_information_load", 1) or 1)
    context_support_turn_count = int(defaults.get("context_support_turn_count", 0) or 0)
    target_message_count = int(defaults.get("target_message_count", 10) or 10)
    safe_prevented_no_trigger_probability = float(defaults.get("safe_prevented_no_trigger_probability", 0.0) or 0.0)
    raw_default_support_turn_options = defaults.get("context_support_turn_options")
    if raw_default_support_turn_options is None:
        context_support_turn_options = [context_support_turn_count]
    elif not isinstance(raw_default_support_turn_options, list):
        raise ValueError("stage2_contracts.defaults.context_support_turn_options must be a list")
    else:
        context_support_turn_options = [int(item) for item in raw_default_support_turn_options]
    if trigger_max_information_load < 1:
        raise ValueError("stage2_contracts.defaults.trigger_max_information_load must be >= 1")
    if context_support_turn_count < 0:
        raise ValueError("stage2_contracts.defaults.context_support_turn_count must be >= 0")
    if target_message_count < 4:
        raise ValueError("stage2_contracts.defaults.target_message_count must be >= 4")
    if not 0.0 <= safe_prevented_no_trigger_probability <= 1.0:
        raise ValueError("stage2_contracts.defaults.safe_prevented_no_trigger_probability must be between 0 and 1")
    if not context_support_turn_options:
        raise ValueError("stage2_contracts.defaults.context_support_turn_options must be non-empty")
    if any(option < 0 for option in context_support_turn_options):
        raise ValueError("stage2_contracts.defaults.context_support_turn_options must contain only non-negative ints")
    raw_profiles = stage2_contracts.get("risk_type_profiles", {})
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("generation config semantics.stage2_contracts.risk_type_profiles must be a non-empty mapping")
    risk_type_profiles: dict[str, dict[str, Any]] = {}
    for risk_type, profile in raw_profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type} must be a mapping")
        normalized_risk_type = str(risk_type).strip()
        boundary_spec = boundary_spec_for_risk_type(normalized_risk_type)
        definition = str(profile.get("definition", "")).strip()
        if not definition:
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.definition must be non-empty")

        hard_contract = profile.get("hard_contract", {})
        if hard_contract and not isinstance(hard_contract, dict):
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.hard_contract must be a mapping")
        soft_hints = profile.get("soft_hints", {})
        if soft_hints and not isinstance(soft_hints, dict):
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.soft_hints must be a mapping")

        legacy_context_scaffold = profile.get("context_scaffold", {})
        if legacy_context_scaffold and not isinstance(legacy_context_scaffold, dict):
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.context_scaffold must be a mapping")
        hard_context_scaffold = hard_contract.get("context_scaffold", {})
        if hard_context_scaffold and not isinstance(hard_context_scaffold, dict):
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.hard_contract.context_scaffold must be a mapping")

        carrier_forms = _unique_strings(
            hard_contract.get("carrier_forms", []),
            hard_contract.get("carrier_form", ""),
            profile.get("carrier_forms", []),
            boundary_spec.carrier_form,
            soft_hints.get("carrier_forms", []),
        )
        attachment_kinds = _unique_strings(
            hard_contract.get("attachment_kinds", []),
            hard_contract.get("attachment_kind", ""),
            profile.get("attachment_kinds", []),
            boundary_spec.attachment_kind,
        )
        motive_families = _unique_strings(
            hard_contract.get("motive_families", []),
            hard_contract.get("motive_family", ""),
            profile.get("motive_families", []),
            soft_hints.get("motive_families", []),
        )
        unsafe_rollout_shapes = _unique_strings(
            hard_contract.get("unsafe_rollout_shapes", []),
            hard_contract.get("unsafe_rollout_shape", ""),
            profile.get("unsafe_rollout_shapes", []),
        )
        if not carrier_forms:
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.carrier_forms must be non-empty")
        if not attachment_kinds:
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.attachment_kinds must be non-empty")
        if not motive_families:
            motive_families = ["none"]
        if not unsafe_rollout_shapes:
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.unsafe_rollout_shapes must be non-empty")

        reasoning_contract = soft_hints.get("reasoning_contract", profile.get("reasoning_contract", {}))
        if reasoning_contract and not isinstance(reasoning_contract, dict):
            raise ValueError(f"stage2_contracts.risk_type_profiles.{risk_type}.reasoning_contract must be a mapping")

        raw_support_turn_options = hard_context_scaffold.get("support_turn_options")
        if raw_support_turn_options is None:
            raw_support_turn_options = hard_contract.get("support_turn_options")
        if raw_support_turn_options is None:
            raw_support_turn_options = legacy_context_scaffold.get("support_turn_options")
        if raw_support_turn_options is None:
            support_turn_count = int(
                hard_context_scaffold.get(
                    "support_turn_count",
                    hard_contract.get(
                        "support_turn_count",
                        legacy_context_scaffold.get("support_turn_count", context_support_turn_count),
                    ),
                )
                or 0
            )
            support_turn_options = [support_turn_count]
        elif not isinstance(raw_support_turn_options, list):
            raise ValueError(
                f"stage2_contracts.risk_type_profiles.{risk_type}.context_scaffold.support_turn_options must be a list"
            )
        else:
            support_turn_options = [int(item) for item in raw_support_turn_options]
        if not support_turn_options:
            support_turn_options = list(context_support_turn_options)
        if any(option < 0 for option in support_turn_options):
            raise ValueError(
                f"stage2_contracts.risk_type_profiles.{risk_type}.context_scaffold.support_turn_options must contain only non-negative ints"
            )
        support_turn_options = list(dict.fromkeys(support_turn_options))
        support_turn_types = _unique_strings(
            hard_context_scaffold.get("support_turn_types", []),
            hard_contract.get("support_turn_types", []),
            legacy_context_scaffold.get("support_turn_types", []),
            list(boundary_spec.support_turn_types),
            soft_hints.get("support_turn_types", []),
        )
        if max(support_turn_options) > 0 and not support_turn_types:
            raise ValueError(
                f"stage2_contracts.risk_type_profiles.{risk_type}.context_scaffold.support_turn_types must be non-empty when support_turn_options contains a positive count"
            )
        prompt_hints = _unique_strings(
            soft_hints.get("prompt_hints", []),
            profile.get("prompt_hints", []),
        )
        risk_type_profiles[normalized_risk_type] = {
            "definition": definition,
            "prompt_hints": prompt_hints,
            "carrier_forms": carrier_forms,
            "attachment_kinds": attachment_kinds,
            "motive_families": motive_families,
            "unsafe_rollout_shapes": unsafe_rollout_shapes,
            "required_signals": _unique_strings(
                hard_contract.get("required_signals", []),
                profile.get("required_signals", []),
            ),
            "forbidden_signal_combos": _unique_strings(
                hard_contract.get("forbidden_signal_combos", []),
                profile.get("forbidden_signal_combos", []),
            ),
            "context_scaffold": {
                "support_turn_count": support_turn_options[0],
                "support_turn_options": support_turn_options,
                "support_turn_types": support_turn_types,
                "anchor_object": _first_non_empty_string(
                    hard_context_scaffold.get("anchor_object", ""),
                    hard_contract.get("anchor_object", ""),
                    legacy_context_scaffold.get("anchor_object", ""),
                    boundary_spec.anchor_object,
                ),
            },
            "reasoning_contract": {
                "allowed_affordance_sources": [
                    str(item).strip()
                    for item in reasoning_contract.get("allowed_affordance_sources", [])
                    if str(item).strip()
                ] if isinstance(reasoning_contract.get("allowed_affordance_sources", []), list) else [],
                "forbidden_inference_patterns": [
                    str(item).strip()
                    for item in reasoning_contract.get("forbidden_inference_patterns", [])
                    if str(item).strip()
                ] if isinstance(reasoning_contract.get("forbidden_inference_patterns", []), list) else [],
                "must_reference_trigger": bool(reasoning_contract.get("must_reference_trigger", False)),
            },
        }

    return {
        "branch_operator_rules": normalized_branch_rules,
        "risk_type_compatibility": {
            "disallowed_by_branch_operator": disallowed_by_branch_operator,
            "allowed_capabilities_by_risk_type": allowed_capabilities_by_risk_type,
        },
        "combo_semantics": combo_semantics,
        "stage2_contracts": {
            "defaults": {
                "trigger_max_information_load": trigger_max_information_load,
                "context_support_turn_count": context_support_turn_count,
                "context_support_turn_options": context_support_turn_options,
                "target_message_count": target_message_count,
                "safe_prevented_no_trigger_probability": safe_prevented_no_trigger_probability,
            },
            "motive_prompt_hints": motive_prompt_hints,
            "risk_type_profiles": risk_type_profiles,
        },
    }


def load_generation_settings(config_path: Path | None = None) -> LLMSettings:
    config = load_generation_config(str(config_path) if config_path else None)
    return load_llm_settings(config)


def load_judge_settings(config_path: Path | None = None) -> LLMSettings:
    config = load_generation_config(str(config_path) if config_path else None)
    llm_settings = load_llm_settings(config)
    judge_cfg = config.get("llm_judge", {})
    return replace(
        llm_settings,
        enabled=bool(judge_cfg.get("enabled", False)),
        api_base_url=str(judge_cfg.get("api_base_url", llm_settings.api_base_url)),
        model_name=str(judge_cfg.get("model_name", llm_settings.model_name)),
        max_tokens=int(judge_cfg.get("max_tokens", llm_settings.max_tokens)),
        timeout_seconds=int(judge_cfg.get("timeout_seconds", llm_settings.timeout_seconds)),
        retry_temperatures=tuple(float(item) for item in judge_cfg.get("retry_temperatures", llm_settings.retry_temperatures)),
        repair_temperature=float(judge_cfg.get("repair_temperature", llm_settings.repair_temperature)),
        min_score=int(judge_cfg.get("min_score", llm_settings.min_score)),
        judge_prompt_mode=str(judge_cfg.get("prompt_mode", llm_settings.judge_prompt_mode)).strip() or llm_settings.judge_prompt_mode,
        clean_judge_prompt_mode=str(judge_cfg.get("clean_prompt_mode", getattr(llm_settings, "clean_judge_prompt_mode", ""))).strip(),
    )


def require_generation_llm(config_path: Path | None = None) -> LLMSettings:
    settings = load_generation_settings(config_path)
    if not settings.enabled:
        raise RuntimeError("pipeline_v1 LLM generation is disabled in the selected config")
    if not llm_available(settings):
        raise RuntimeError(
            "pipeline_v1 LLM generation endpoint is unavailable: "
            f"base_url={settings.api_base_url} model={settings.model_name}"
        )
    return settings
