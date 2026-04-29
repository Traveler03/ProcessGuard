This directory contains a display-only synthetic stress summary.

It is not a real evaluation output.
It is derived from the 1010-sample ground-truth distribution and a synthetic
confusion-count construction chosen for presentation/demo use.

The summary is strictly self-consistent:

- `binary_accuracy` is derivable from `binary_confusion_matrix`
- `combo_accuracy` is derivable from `combo_confusion_matrix`
- `risk_type_accuracy` is derivable from `risk_type_confusion_matrix`
- `combo_and_risk_type_joint_accuracy` is derivable from `correct_overlap_counts`

Do not treat these files as measured benchmark results.

Reference real run:

- `/data2/wulinxuan/AgentWorld/ProcessGuard/output/rebuild/qwen35_extbin_local_1010_originalprompt_20260428/summary.json`
