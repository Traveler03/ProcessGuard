from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMSettings:
    """OpenAI-compatible LLM endpoint settings."""

    enabled: bool
    api_base_url: str
    api_key: str
    model_name: str
    max_tokens: int
    timeout_seconds: int
    retry_temperatures: tuple[float, ...]
    repair_temperature: float
    min_score: int
    judge_prompt_mode: str = "trajectory_quality"
    clean_judge_prompt_mode: str = ""


def load_llm_settings(config: dict[str, Any]) -> LLMSettings:
    """Parse LLM config into a strongly-typed object."""
    llm_cfg = config.get("llm", {})
    retry_temperatures = tuple(float(item) for item in llm_cfg.get("retry_temperatures", [0.2]))
    return LLMSettings(
        enabled=bool(llm_cfg.get("enabled", False)),
        api_base_url=str(llm_cfg.get("api_base_url", "http://127.0.0.1:8027/v1")),
        api_key=str(llm_cfg.get("api_key", os.getenv("OPENAI_API_KEY", ""))),
        model_name=str(llm_cfg.get("model_name", "qwen3.5-27b-local")),
        max_tokens=int(llm_cfg.get("max_tokens", 1024)),
        timeout_seconds=int(llm_cfg.get("timeout_seconds", 180)),
        retry_temperatures=retry_temperatures,
        repair_temperature=float(llm_cfg.get("repair_temperature", 0.1)),
        min_score=int(llm_cfg.get("min_score", 3)),
        judge_prompt_mode=str(llm_cfg.get("judge_prompt_mode", "trajectory_quality")).strip() or "trajectory_quality",
        clean_judge_prompt_mode=str(llm_cfg.get("clean_judge_prompt_mode", "")).strip(),
    )


def strip_wrappers(text: str) -> str:
    """Remove think blocks and JSON fences."""
    cleaned = THINK_BLOCK_RE.sub("", text).strip()
    fenced = JSON_FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    return cleaned.strip()


def extract_json_payload(text: str) -> dict[str, Any]:
    """Parse the first JSON object found in a model response."""
    cleaned = strip_wrappers(text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise json.JSONDecodeError("Could not parse a JSON object", cleaned, 0)


def _build_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _normalized_base_url(base_url: str) -> str:
    return str(base_url).strip().lower()


def _needs_browser_like_user_agent(base_url: str) -> bool:
    normalized = _normalized_base_url(base_url)
    return "fushengyunsuan.cn" in normalized


def _extract_chat_message_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str):
            return value
    return ""


def _should_force_stream(settings: LLMSettings) -> bool:
    base_url = _normalized_base_url(settings.api_base_url)
    # Keep the forced-stream workaround only for endpoints that are known to
    # return empty non-stream content. Fusheng works in non-stream mode and the
    # user explicitly wants to run it that way for judge traffic.
    return "airouter.service.itstudio.club" in base_url


def _post_chat_completion_once(
    *,
    settings: LLMSettings,
    messages: list[dict[str, str]],
    temperature: float,
    stream: bool,
) -> str:
    payload = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": settings.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        payload["stream"] = True
    chat_url = settings.api_base_url.rstrip("/") + "/chat/completions"
    headers = _build_headers(settings.api_key)
    if _needs_browser_like_user_agent(settings.api_base_url):
        headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    if stream:
        headers["Accept"] = "text/event-stream"
    req = request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=settings.timeout_seconds) as response:
        if not stream:
            body = json.loads(response.read().decode("utf-8"))
            return _extract_chat_message_text(body)

        parts: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
        return "".join(parts)


def post_chat_completion(
    *,
    settings: LLMSettings,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    """Call one OpenAI-compatible chat completion endpoint."""
    if _should_force_stream(settings):
        return _post_chat_completion_once(
            settings=settings,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
    text = _post_chat_completion_once(
        settings=settings,
        messages=messages,
        temperature=temperature,
        stream=False,
    )
    if text:
        return text
    return _post_chat_completion_once(
        settings=settings,
        messages=messages,
        temperature=temperature,
        stream=True,
    )


def generate_json(
    *,
    settings: LLMSettings,
    messages: list[dict[str, str]],
    validator: Any | None = None,
    repair_messages_builder: Any | None = None,
) -> tuple[dict[str, Any], str]:
    """Generate one JSON object with retries and optional repair."""
    errors: list[str] = []
    last_raw = ""
    for temperature in settings.retry_temperatures:
        try:
            raw = post_chat_completion(settings=settings, messages=messages, temperature=temperature)
            last_raw = raw
            payload = extract_json_payload(raw)
            if validator is not None:
                validator(payload)
            return payload, "llm"
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError, IndexError) as exc:
            errors.append(str(exc))

    if repair_messages_builder is None or not last_raw:
        raise RuntimeError("; ".join(errors) if errors else "LLM generation failed")

    repair_messages = repair_messages_builder(last_raw, errors)
    raw = post_chat_completion(
        settings=settings,
        messages=repair_messages,
        temperature=settings.repair_temperature,
    )
    payload = extract_json_payload(raw)
    if validator is not None:
        validator(payload)
    return payload, "repair"


def llm_available(settings: LLMSettings) -> bool:
    """Probe whether the configured LLM endpoint responds."""
    if not settings.enabled:
        return False
    messages = [{"role": "user", "content": 'Return {"ok": true} as JSON only.'}]
    try:
        raw = post_chat_completion(settings=settings, messages=messages, temperature=0.0)
        payload = extract_json_payload(raw)
    except Exception:
        return False
    return payload.get("ok") is True
