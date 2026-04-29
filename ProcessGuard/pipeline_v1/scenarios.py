from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PIPELINE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PIPELINE_ROOT / "config"
DEFAULT_CATALOG_PATH = CONFIG_ROOT / "scenario_catalog.yaml"


def load_yaml_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    context = payload.get("context", {}) if isinstance(payload.get("context"), dict) else {}
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return {
        "scenario_name": str(payload.get("name", "")).strip(),
        "display_name": str(payload.get("display_name", payload.get("name", ""))).strip(),
        "description": str(payload.get("description", "")).strip(),
        "category": str(payload.get("category", "")).strip(),
        "generation_weight": float(payload.get("generation_weight", 1.0)),
        "sample_actions": list(payload.get("sample_actions", [])),
        "tools": list(context.get("available_tools", [])),
        "environment": context.get("environment", {}),
        "predefined_variables": context.get("predefined_variables", {}),
        "system_info": context.get("system_info", {}),
        "security_policy": context.get("security_policy", {}),
        "planner_hints": payload.get("planner_hints", {}),
        "selection_num": int(metadata.get("selection_num", 3) or 3),
        "required_permissions": list(payload.get("required_permissions", [])),
        "tags": list(payload.get("tags", [])),
    }


def load_scenarios(catalog_path: Path | None = None) -> list[dict[str, Any]]:
    catalog_path = catalog_path or DEFAULT_CATALOG_PATH
    catalog = load_yaml_file(catalog_path) or {}
    scenarios_dir_raw = str(catalog.get("scenarios_dir", "scenarios_generated")).strip()
    scenarios_dir = (
        Path(scenarios_dir_raw)
        if Path(scenarios_dir_raw).is_absolute()
        else catalog_path.parent / scenarios_dir_raw
    )
    include = [str(item).strip() for item in (catalog.get("include") or []) if str(item).strip()]
    paths = [scenarios_dir / f"{name}.yaml" for name in include] if include else sorted(scenarios_dir.glob("*.yaml"))
    scenarios: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = load_yaml_file(path)
        if not isinstance(payload, dict):
            continue
        normalized = normalize_scenario(payload)
        if normalized["scenario_name"] and normalized["tools"]:
            scenarios.append(normalized)
    if not scenarios:
        raise FileNotFoundError(f"No scenarios loaded from {scenarios_dir}")
    return scenarios


def load_scenario_by_name(name: str, catalog_path: Path | None = None) -> dict[str, Any]:
    target = str(name).strip()
    for scenario in load_scenarios(catalog_path=catalog_path):
        if scenario["scenario_name"] == target:
            return scenario
    raise KeyError(f"Scenario not found: {target}")
