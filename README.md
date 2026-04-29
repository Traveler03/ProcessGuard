# ProcessGuard

This repository currently hosts the first paper-facing bundle for the
`ProcessGuard-1010` benchmark and its associated experiment results.

## Included in this first version

### 1. ProcessGuard-1010 benchmark

Final benchmark files live under:

- `benchmark/processguard_eval/processguard1010_final_benchmark.jsonl`
- `benchmark/processguard_eval/processguard1010_final_benchmark_manifest.json`

These are the main data files to use for evaluation and analysis.

### 2. Data generation pipeline

The main benchmark generation code lives under:

- `ProcessGuard/pipeline_v1/`

This directory contains the planning, trajectory synthesis, filtering,
configuration, and scenario definitions used in the benchmark pipeline.

The current workflow summary document is:

- `ProcessGuard/pipeline_v1/docs/processguard_pipeline_v1_benchmark1010_workflow_20260429.md`

### 3. SFT construction metadata

SFT split manifests used in the broader ProcessGuard workflow live under:

- `benchmark/sft_train/qwen35_9b_processguard10k_sft_v2_20260420/manifest.json`
- `benchmark/sft_train/qwen35_9b_processguard10k_sft_v2_with_explanation_20260420/manifest.json`

### 4. Compared benchmarks

This repository also tracks benchmark usage notes for:

- `ASSE`
- `ATBench`

See:

- `benchmark/README.md`

### 5. Experiment results

Paper-facing experiment bundles live under:

- `ProcessGuard/experiments/mybench_0428/`
- `ProcessGuard/experiments/asse_safety_qwen35_extbin_20260428/`
- `ProcessGuard/experiments/atbench_qwen35_extbin_20260428/`

These directories contain compact result summaries intended for writing and
analysis, rather than full raw evaluation dumps.

## Scope note

This first upload is intentionally lightweight. It includes:

- the final ProcessGuard-1010 benchmark
- the main `pipeline_v1` generation code
- SFT manifest metadata
- compact experiment bundles

It does not attempt to mirror every historical intermediate artifact, run log,
or temporary benchmark variant.
