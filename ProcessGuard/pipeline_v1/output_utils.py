from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shutil

from .scenarios import PIPELINE_ROOT


OUTPUT_ROOT = PIPELINE_ROOT / "output"


def _slug(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized or "run"


def create_run_dir(
    *,
    kind: str,
    scenario: str | None = None,
    label_combo: str | None = None,
    seed: int | None = None,
    run_name: str | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, _slug(kind)]
    if scenario:
        parts.append(_slug(scenario))
    if label_combo:
        parts.append(_slug(label_combo))
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    if run_name:
        parts.append(_slug(run_name))
    run_dir = OUTPUT_ROOT / "__".join(parts)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def clear_output_root() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_ROOT.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return OUTPUT_ROOT


def output_file_in_run_dir(run_dir: Path, output_name: str, *, default_name: str) -> Path:
    name = Path(output_name).name if output_name else default_name
    return run_dir / name


def lightweight_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.light{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.light.json")


def snapshot_config(config_path: Path, run_dir: Path) -> Path:
    destination = run_dir / "config.snapshot.yaml"
    shutil.copyfile(config_path, destination)
    return destination
