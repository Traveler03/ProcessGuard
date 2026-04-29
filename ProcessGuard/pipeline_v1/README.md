# ProcessGuard Pipeline V1

`pipeline_v1` is a clean first-pass data-generation pipeline built around the
three-stage flow we settled on:

1. `Stage 1: Planning`
2. `Stage 2: Trajectory Synthesis`
3. `Stage 3: Filtering`

This version intentionally keeps stage boundaries small.

- Stage 1 input: `scenario_config + label_combo (+ optional seed)`
- Stage 1 output: `execution_plan`
- Stage 2 input: `execution_plan`
- Stage 2 output: `trajectory`
- Stage 3 input: `execution_plan + trajectory`
- Stage 3 output: `final_record`

## First-Version Scope

This is an MVP. It does not try to solve all realism problems.

- Uses `ProcessGuard/pipeline_v1/config/scenario_catalog.yaml`
- Supports the five canonical label combinations
- Uses the local OpenAI-compatible LLM client with `pipeline_v1/config/generation.yaml`
- Requires an available LLM endpoint; there is no template fallback in the main path
- Generates one visible trigger per sample
- Generates one main risk action at most
- Derives evidence from step-level tags instead of running a separate evidence stage

## Label Combos

- `safe_prevented`
- `unsafe_attempted_not_realized`
- `unsafe_attempted_realized`
- `unsafe_executed_not_realized`
- `unsafe_executed_realized`

## Layout

- `schema.py`: shared dataclasses and combo definitions
- `config/generation.yaml`: default Stage 2 generation config
- `scenarios.py`: scenario loading from local `pipeline_v1/config/`
- `planner.py`: Stage 1 planner
- `policy.py`: shared tool-role and combo policy
- `orchestrator.py`: Stage 2 trajectory synthesis
- `query_generator.py`: Stage 2 query and trigger generation
- `tool_call_generator.py`: Stage 2 benign/completion/risky tool-call generation
- `agent_response_generator.py`: Stage 2 safe-branch assistant replies
- `outcome_summarizer.py`: Stage 2 environment observations
- `stage2_common.py`: shared Stage 2 contracts and strict JSON generation
- `filters.py`: Stage 3 validation and evidence derivation
- `pipeline.py`: end-to-end runner
- `output_utils.py`: run-directory and artifact helpers
- `scripts/generate_sample.py`: CLI for one sample
- `scripts/generate_batch.py`: CLI for a batch across scenarios / combos

## Example

```bash
cd /data2/wulinxuan/AgentWorld
python ProcessGuard/pipeline_v1/scripts/generate_sample.py \
  --scenario meeting_follow_up_bot \
  --label-combo unsafe_executed_realized \
  --seed 7

python ProcessGuard/pipeline_v1/scripts/generate_batch.py \
  --seed 11 \
  --scenario meeting_follow_up_bot
```

Each CLI run now creates a separate subdirectory under `ProcessGuard/output/pipeline_v1/`
and writes both the JSON result and a `config.snapshot.yaml` copy there.

Compatibility note:
- `ProcessGuard/pipeline_v1/output` remains available as a symlink to the unified output root.
