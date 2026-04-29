from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROCESS_GUARD_ROOT = Path(__file__).resolve().parents[2]
if str(PROCESS_GUARD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_GUARD_ROOT))

from pipeline_v1.pipeline import generate_sample  # noqa: E402
from pipeline_v1.config import DEFAULT_CONFIG_PATH  # noqa: E402
from pipeline_v1.lightweight_export import lightweight_record  # noqa: E402
from pipeline_v1.output_utils import create_run_dir, lightweight_output_path, output_file_in_run_dir, snapshot_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one ProcessGuard pipeline_v1 sample.")
    parser.add_argument("--scenario", required=True, help="Scenario name from rebuild scenario catalog.")
    parser.add_argument("--label-combo", required=True, help="One of the five canonical label combo slugs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--config", type=Path, default=None, help="Optional pipeline_v1 generation config path.")
    parser.add_argument("--risk-type", type=str, default=None, help="Optional explicit risk-type category override.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional suffix for the auto-created run directory.")
    parser.add_argument("--output", type=Path, default=None, help="Optional file name inside the auto-created run directory.")
    args = parser.parse_args()

    config_path = args.config or DEFAULT_CONFIG_PATH
    record = generate_sample(
        scenario_name=args.scenario,
        label_combo=args.label_combo,
        seed=args.seed,
        config_path=config_path,
        risk_type_override=(args.risk_type.strip() if args.risk_type else None),
        progress_callback=lambda message: print(f"[sample] {message}", flush=True),
    )
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    run_dir = create_run_dir(
        kind="sample",
        scenario=args.scenario,
        label_combo=args.label_combo,
        seed=args.seed,
        run_name=args.run_name,
    )
    output_path = output_file_in_run_dir(run_dir, str(args.output) if args.output else "", default_name="sample.json")
    light_output_path = lightweight_output_path(output_path)
    output_path.write_text(payload, encoding="utf-8")
    light_output_path.write_text(json.dumps(lightweight_record(record), ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot_config(config_path, run_dir)
    print(str(output_path))


if __name__ == "__main__":
    main()
