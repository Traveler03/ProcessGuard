from __future__ import annotations

import json
from typing import Any, Callable

from .llm_client import LLMSettings, generate_json, post_chat_completion


PayloadValidator = Callable[[dict[str, Any]], list[str]]
RepairPromptBuilder = Callable[[dict[str, Any], list[str]], list[dict[str, str]]]
TextPayloadParser = Callable[[str], tuple[dict[str, Any], list[str]]]
TextRepairPromptBuilder = Callable[[str, list[str]], list[dict[str, str]]]
RepairDecision = Callable[[list[str]], bool]


def render_errors(errors: list[str]) -> str:
    """Render validator errors in a compact bullet list."""
    return "\n".join(f"- {error}" for error in errors)


def generate_with_repair(
    *,
    settings: LLMSettings,
    initial_messages: list[dict[str, str]],
    validate_payload: PayloadValidator,
    repair_messages_builder: RepairPromptBuilder,
    max_repair_rounds: int = 2,
) -> tuple[dict[str, Any], str]:
    """Generate structured JSON, validate it, and repair only the failed parts."""
    messages = initial_messages
    mode = "llm"
    last_errors: list[str] = []
    prev_error_signature = ""
    stagnant_rounds = 0
    for attempt in range(max_repair_rounds + 1):
        payload, _ = generate_json(
            settings=settings,
            messages=messages,
        )
        last_errors = validate_payload(payload)
        if not last_errors:
            return payload, mode
        signature = json.dumps(sorted(set(last_errors)), ensure_ascii=False)
        if signature == prev_error_signature:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
            prev_error_signature = signature
        if stagnant_rounds >= 1:
            break
        if attempt >= max_repair_rounds:
            break
        messages = repair_messages_builder(payload, last_errors)
        mode = "repair"
    raise RuntimeError(f"structured generation failed after repair: {json.dumps(last_errors, ensure_ascii=False)}")


def generate_text_with_repair(
    *,
    settings: LLMSettings,
    initial_messages: list[dict[str, str]],
    parse_payload: TextPayloadParser,
    repair_messages_builder: TextRepairPromptBuilder,
    max_repair_rounds: int = 2,
    should_attempt_repair: RepairDecision | None = None,
) -> tuple[dict[str, Any], str]:
    """Generate weakly structured text, parse it, and repair only the failed parts."""
    messages = initial_messages
    mode = "llm"
    last_errors: list[str] = []
    last_raw = ""
    prev_error_signature = ""
    stagnant_rounds = 0
    empty_response_rounds = 0
    for attempt in range(max_repair_rounds + 1):
        raw = ""
        for temperature in settings.retry_temperatures:
            try:
                raw = post_chat_completion(settings=settings, messages=messages, temperature=temperature)
                break
            except Exception as exc:
                last_errors = [str(exc)]
                raw = ""
        if not raw:
            empty_response_rounds += 1
            if empty_response_rounds >= 2:
                break
            if attempt >= max_repair_rounds:
                break
            candidate_errors = last_errors or ["empty_generation_response"]
            if should_attempt_repair is not None and not should_attempt_repair(candidate_errors):
                last_errors = candidate_errors
                break
            last_errors = candidate_errors
            messages = repair_messages_builder(last_raw, last_errors)
            mode = "repair"
            continue
        empty_response_rounds = 0
        last_raw = raw
        payload, last_errors = parse_payload(raw)
        if not last_errors:
            return payload, mode
        signature = json.dumps(sorted(set(last_errors)), ensure_ascii=False)
        if signature == prev_error_signature:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
            prev_error_signature = signature
        if stagnant_rounds >= 1:
            break
        if attempt >= max_repair_rounds:
            break
        if should_attempt_repair is not None and not should_attempt_repair(last_errors):
            break
        messages = repair_messages_builder(raw, last_errors)
        mode = "repair"
    raise RuntimeError(f"text generation failed after repair: {json.dumps(last_errors, ensure_ascii=False)}")
