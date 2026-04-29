# Benchmarks Used

This directory contains benchmark assets used in the ProcessGuard project.

## ProcessGuard-1010

The main benchmark introduced and analyzed in this repository is:

- `processguard_eval/processguard1010_final_benchmark.jsonl`
- `processguard_eval/processguard1010_final_benchmark_manifest.json`

## Compared benchmarks

The current first-version repository also reports experiments on:

- `ASSE`
- `ATBench`

For this first upload, the repository focuses on:

- the final `ProcessGuard-1010` benchmark data
- compact experiment result bundles under `ProcessGuard/experiments/`

The compared benchmark result bundles are:

- `ProcessGuard/experiments/asse_safety_qwen35_extbin_20260428/`
- `ProcessGuard/experiments/atbench_qwen35_extbin_20260428/`

Local reference benchmark directories may also exist in this workspace, but the
paper-facing first upload does not mirror every external benchmark copy or all
raw evaluation artifacts.
