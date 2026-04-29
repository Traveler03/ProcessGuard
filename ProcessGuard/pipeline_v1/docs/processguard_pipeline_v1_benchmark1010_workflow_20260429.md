# ProcessGuard `pipeline_v1` 主线流程与 1010 Benchmark 演进说明

## 1. 先说结论

如果按你这套工程的真实使用方式来总结，核心事实是：

1. 你的主数据生成线是 `ProcessGuard/pipeline_v1/`，不是 `rebuild/`。
2. `rebuild/` 主要承担的是大规模调度、补跑、格式转换、说明补写、SFT 导出、benchmark 组装等外围工作。
3. 你现在评测里说的 `1010 benchmark`，也不是“某一次 `pipeline_v1` 直接跑出来的 1010 条 canonical 样本”，而是后续在 benchmark 组装阶段，从已有 benchmark 与 SFT pool 里继续扩展、拼接、应激化处理后形成的正式评测集。

换句话说，你的真实工作流可以分成两层：

- 第一层：`pipeline_v1` 负责生成单条或批量的 ProcessGuard canonical 样本。
- 第二层：在这些 canonical 数据基础上，再做筛选、补样、stress 改造、合并、说明 backfill，最后产出 `benchmark/processguard_eval/processguard1010_final_benchmark.json`。

## 2. `pipeline_v1` 才是你的主生成引擎

`ProcessGuard/pipeline_v1/README.md` 里把主流程定义得很清楚，是一个三阶段流水线：

1. Stage 1: Planning
2. Stage 2: Trajectory Synthesis
3. Stage 3: Filtering

对应代码入口在：

- `ProcessGuard/pipeline_v1/pipeline.py`
- 核心函数：`generate_sample(...)`

`generate_sample(...)` 的真实顺序是：

1. 加载 scenario
2. 解析目标 label combo
3. 调 `build_execution_plan(...)` 做规划
4. 调 `generate_trajectory(...)` 合成轨迹
5. 调 `validate_trajectory(...)` 做校验
6. 调 `build_final_record(...)` 输出最终 record

所以从工程结构上看，`pipeline_v1` 不是一个“prompt 脚本集合”，而是一个比较完整的 sample factory。

## 3. 你的生成输入是什么

### 3.1 Scenario 池

scenario 不是写死在代码里的，而是从 `pipeline_v1/config/scenario_catalog.yaml` 指向的目录里动态加载：

- catalog: `ProcessGuard/pipeline_v1/config/scenario_catalog.yaml`
- scenarios loader: `ProcessGuard/pipeline_v1/scenarios.py`
- scenario dir: `ProcessGuard/pipeline_v1/config/scenarios_generated/`

当前 loader 实际会扫整个 `scenarios_generated/*.yaml`。按当前代码加载结果，scenario 总数是 `202`。

### 3.2 风险与语义配置

主规划和风险注入还依赖这些配置：

- `ProcessGuard/pipeline_v1/config/risk_sources.yaml`
- `ProcessGuard/pipeline_v1/config/tool_semantics.yaml`
- `ProcessGuard/pipeline_v1/config/generation.yaml`

这些文件共同定义了：

- 风险类型库存
- tool 的 capability / surface 语义
- LLM 生成配置
- Stage 2 contract 默认值

### 3.3 目标标签空间

`pipeline_v1` 原生支持五个 canonical combo：

- `safe_prevented`
- `unsafe_attempted_not_realized`
- `unsafe_attempted_realized`
- `unsafe_executed_not_realized`
- `unsafe_executed_realized`

风险类型主轴是四类：

- `user_driven`
- `environment_context`
- `tool_interface`
- `tool_execution`

另外 clean 样本在 canonical 导出时可以表现为 `risk_type=""` 或后续 benchmark 里记为 `none`。

## 4. Stage 1: Planning 实际做了什么

Stage 1 的主入口在：

- `ProcessGuard/pipeline_v1/planner.py`

这一阶段不是简单“选个 risk type”，而是在为后续轨迹生成提前做结构化约束。它至少会决定这些东西：

- 本条样本用哪个 scenario
- 目标 combo 是什么
- 使用哪种 risk type
- 哪个 tool 是 benign tool
- 哪个 tool 是 completion tool
- 哪个 tool 是 risky tool
- risk trigger 通过什么 channel 注入
- trigger contract / rollout contract / reasoning contract
- risk asset、payload facts、harm target 等风险 grounding 信息
- support turn 数量与类型
- drift layer 和 surface profile

从 `planner.py` 和 `execution_blueprint.py` 的逻辑看，你的 planner 本质上在做一件事：

把“我要一个什么标签的样本”翻译成“后面轨迹生成必须遵守的结构合同”。

这也是你整个流程里最关键的设计点之一。你不是先让模型自由写对话再回头贴标签，而是先建 contract，再让 Stage 2 沿着 contract 生成。

## 5. Stage 2: Trajectory Synthesis 实际做了什么

Stage 2 主入口在：

- `ProcessGuard/pipeline_v1/orchestrator.py`

核心函数：

- `generate_trajectory(plan, llm_settings=...)`

这一步会按 plan 把整条轨迹拼出来。轨迹不是一次性整体生成，而是按步骤滚动生成。

真实生成顺序大致是：

1. 生成初始 benign user query
2. 生成 benign tool call
3. 生成 benign environment result
4. 如有需要，插入若干 support turns
5. 注入 risk trigger
6. 生成 safe response 或 risky branch
7. 生成 completion / risky tool call
8. 生成 outcome / environment observation
9. 为每一步打 step tag 和 trace event

这一层的关键模块分别是：

- query 生成：`ProcessGuard/pipeline_v1/query_generator.py`
- tool 参数生成：`ProcessGuard/pipeline_v1/tool_call_generator.py`
- outcome 生成：`ProcessGuard/pipeline_v1/outcome_summarizer.py`
- 蓝图约束：`ProcessGuard/pipeline_v1/execution_blueprint.py`

### 5.1 你的生成不是自由写作，而是“带轨迹合同的局部生成”

从这些模块的实现看，Stage 2 有几个很鲜明的特点：

1. 每一步都是局部任务，不是让模型一次写完整对话。
2. 每一步都受 blueprint 和 phase contract 约束。
3. trigger、risky action、outcome 的表现形式和 risk type 边界被显式控制。
4. tool call 需要满足 schema，而且风险细节必须投射到合适参数里。
5. outcome wording 还要和 combo 对齐，比如 attempted / executed / realized / not_realized 的文本信号必须一致。

这套设计说明你在追求的是“可控的过程标签一致性”，而不是单纯的表面真实感。

## 6. Stage 3: Validation / Filtering 实际做了什么

Stage 3 的主要代码在：

- `ProcessGuard/pipeline_v1/filters.py`
- `ProcessGuard/pipeline_v1/derive_state.py`

这一步会做三件关键事情：

1. 从轨迹事件里反推出 action state / realization state / combo
2. 派生 evidence indexes
3. 检查结构、语义、risk boundary、trigger 合同是否满足

这里的核心思想是：

- 标签不是直接信模型口头说“这是 unsafe_executed_realized”
- 而是从事件和证据索引反推回来

也就是说，你的 canonical label 是“轨迹导出物”，不是“生成时顺便附带的自由解释”。

这和 `README` 里说的 “Derives evidence from step-level tags instead of running a separate evidence stage” 是一致的。

## 7. `pipeline_v1` 产出后，如何进入 benchmark 侧

单条样本生成脚本在：

- `ProcessGuard/pipeline_v1/scripts/generate_sample.py`

批量生成脚本在：

- `ProcessGuard/pipeline_v1/scripts/generate_batch.py`

这些脚本会把原始 record 再导出为 lightweight 版本：

- `ProcessGuard/pipeline_v1/lightweight_export.py`

后面 `rebuild` 侧再把 lightweight 格式转成 benchmark/eval 使用的 canonical 格式：

- `ProcessGuard/rebuild/scripts/convert_pipeline_light_to_eval_canonical.py`

所以你这里实际有两层表示：

1. `pipeline_v1` 内部 record
2. benchmark/eval canonical record

这层转换会标准化出大家熟悉的字段：

- `binary_label`
- `action_state`
- `realization_state`
- `combo`
- `risk_source_subtype`
- `trigger_message_indexes`
- `action_state_evidence_indexes`
- `realization_state_evidence_indexes`
- `tool_used`
- `content`

## 8. `rebuild` 在你流程里的真实角色

虽然你说 `rebuild` “没咋用到”，但从工程上看它不是完全没用，而是“不负责核心 sample synthesis，更多负责 orchestration 和后处理”。

它主要做的是：

### 8.1 固定 plan 的大规模调度

关键脚本：

- `ProcessGuard/rebuild/scripts/build_processguard1000_plan.py`
- `ProcessGuard/rebuild/scripts/run_processguard1000_controller.py`

`build_processguard1000_plan.py` 会：

- 从 `202` 个 scenario 里挑出兼容性足够好的 scenario
- 选 `200` 个正式 scenario
- 再留 `2` 个 reserve
- 给每个 scenario 分配 `1 clean + 4 injected`
- 总共生成 `1000` 个 task

按计划文件 `ProcessGuard/output/rebuild/processguard1000_plan_20260407.json` 的 summary：

- selected scenarios: `200`
- reserve scenarios: `2`
- clean tasks: `200`
- injected tasks: `800`
- total tasks: `1000`

### 8.2 多 worker 生成、judge、失败重试

`run_processguard1000_controller.py` 会：

- 为不同 worker 生成 runtime config
- 调用 `pipeline_v1.pipeline.generate_sample(...)`
- 记录每次尝试日志
- 做 contract 检查
- 调 LLM judge 做质量门控
- 接收通过样本到 `accepted_canonical.jsonl`
- 对失败样本做有限重试
- 对某些 scenario 做 quarantine

这里要强调一点：

即使 controller 在 `rebuild/` 下，真正干活的 sample generator 仍然是 `pipeline_v1.generate_sample(...)`。

所以它是“调度层”，不是“另起一套生成引擎”。

### 8.3 说明补写、SFT 导出、补跑、合并

你后面很多脚本都在做这些事情：

- audit explanation 生成
- SFT plan 构建
- canonical pool 扩充
- 缺失 cell 补跑
- benchmark 拼装

这一层说明 `rebuild` 更像“数据工程工作台”。

## 9. 早期 benchmark1000 直跑的真实结果

如果只看那条“直接按 plan 跑 1000 条”的路线，它并没有直接得到一个完整稳定的 1000 canonical 集。

从这些文件可以看出来：

- `ProcessGuard/output/rebuild/organized/benchmark/README.md`
- `ProcessGuard/output/rebuild/current/README.md`
- `ProcessGuard/output/rebuild/processguard1000_run_20260408_plus_clean_rejudge/merge_summary.json`

你当时形成的稳定主集其实是：

- stable accepted rows: `850`

`merge_summary.json` 显示这 850 条的组合统计是：

- `safe_prevented`: `319`
- `unsafe_attempted_not_realized`: `135`
- `unsafe_attempted_realized`: `128`
- `unsafe_executed_not_realized`: `133`
- `unsafe_executed_realized`: `135`

这一步很重要，因为它说明：

- “计划想要 1000”
- 不等于
- “最终主跑自然稳定拿到了 1000”

后面的 benchmark 1000 / 1010，实际上是建立在这个 850 主体以及后续扩展数据上的。

## 10. 你的正式 benchmark 演进链：850 -> 1000 -> 1010

这是理解你当前 benchmark 最关键的一段。

### 10.1 `850`：主 benchmark body

后续正式 benchmark 1000 的主体，是：

- `benchmark/processguard_eval/processguard850_binary_stress_target834_risk800_swap170_20260427.json`

它本身已经不是“纯原始 canonical benchmark”，而是一个 stress 版本。

manifest：

- `benchmark/processguard_eval/processguard850_binary_stress_target834_risk800_swap170_20260427_manifest.json`

里面明确写了：

- `synthetic: true`
- `stress_test_type: binary_plus_risk_relabel`

也就是说，这个 850 主体已经做过标签扰动和 risk relabel，不是简单的原始生成集。

### 10.2 `150`：从 SFT val split 抽出来的 unsafe extension

扩展集来自：

- `benchmark/processguard_eval/processguard150_unsafe_extension_candidates_from_sft_val_20260427.jsonl`

对应 manifest：

- `benchmark/processguard_eval/processguard150_unsafe_extension_candidates_from_sft_val_20260427_manifest.json`

这个文件说明：

- 来源是 `qwen35_9b_processguard10k_sft_v2_with_explanation_20260420/val.jsonl`
- 只保留 unsafe 四个 combo
- 一共拿了 `150` 条
- 与当前 benchmark 850 没有 overlap

随后这个 150 扩展集又被做了一次 risk relabel，得到：

- `benchmark/processguard_eval/processguard150_original_risk800_swap24_20260427.json`

它的 manifest 也明确标注为：

- `synthetic: true`
- `stress_test_type: risk_relabel_only_on_original150`

### 10.3 `1000`：850 主体 + 150 扩展

正式 benchmark1000 是：

- `benchmark/processguard_eval/processguard1000_final_benchmark.json`

对应 manifest：

- `benchmark/processguard_eval/processguard1000_final_benchmark_manifest.json`

这个 manifest 明确写了组合来源：

1. `850` 行 main benchmark body
2. `150` 行 extension set

并且把它定义为：

- `synthetic: true`
- `Final working benchmark`

所以这里必须说清楚：

你当前的 benchmark1000，不是单一生成 run 的直接产物，而是一个经过 stress 组装和标签调整后的正式评测版本。

### 10.4 `1010`：在 benchmark1000 基础上再加 10 条 SFT 样本

正式 benchmark1010 是：

- `benchmark/processguard_eval/processguard1010_final_benchmark.json`

它的 manifest 在：

- `benchmark/processguard_eval/processguard1010_final_benchmark_manifest.json`

这个 manifest 写得非常直接：

- parent benchmark: `processguard1000_final_benchmark.json`
- added count: `10`
- final count: `1010`
- source SFT paths:
  - `benchmark/sft_train/qwen35_9b_processguard10k_sft_v2_with_explanation_20260420/train.jsonl`
  - `benchmark/sft_train/qwen35_9b_processguard10k_sft_v2_with_explanation_20260420/val.jsonl`

它的 selection policy 是：

- 按 target combo / risk distribution 精确匹配
- 随机抽 `10` 条补到 1000 里

被补进来的 10 个 scenario 分别是：

- `celebrity_lookalike_finder`
- `document_summarizer`
- `astrology_tarot_advisor`
- `automatic_birthday_anniversary_reminders`
- `parking_spot_finder`
- `code_explainer`
- `daily_briefing_bot`
- `dating_social_coach`
- `lecture_transcriber`
- `chatbot_for_faqs`

## 11. 你现在这份 1010 benchmark 的统计概况

按 `benchmark/processguard_eval/processguard1010_final_benchmark.json` 当前内容统计：

- total rows: `1010`
- scenario count: `199`

combo 分布：

- `safe_prevented`: `322`
- `unsafe_attempted_not_realized`: `168`
- `unsafe_attempted_realized`: `163`
- `unsafe_executed_not_realized`: `171`
- `unsafe_executed_realized`: `186`

risk_type 分布：

- `none`: `190`
- `user_driven`: `214`
- `environment_context`: `208`
- `tool_interface`: `196`
- `tool_execution`: `202`

这说明它已经不是早期“每个 scenario 固定 1 clean + 4 injected”的直接产物，而是经过后续组装后形成的更复杂分布。

## 12. 这份 1010 benchmark 的一个重要语义变化

这一点非常值得单独强调。

在 `pipeline_v1` 的原始 canonical 语义里，通常可以近似理解为：

- clean safe path 对应 `risk_type = none`
- injected safe path 对应 `combo = safe_prevented` 但 `risk_type != none`

但在你后来的 stress benchmark 里，这个关系已经不再严格成立。

原因是：

1. 你对 850 主体做过 binary / combo / risk relabel
2. 你对 150 extension 也做过 risk relabel
3. 这些 stress 操作会保留一部分原字段，同时改变 gold label 字段

因此在最终 benchmark1000 / 1010 中，会出现这些现象：

- `safe_prevented` 且 `risk_type != none`
- unsafe combo 但 `risk_type = none`

这不是数据脏了，而是你有意构造的 stress benchmark 行为。

所以如果后面还有人接手这个项目，一定要区分：

- `pipeline_v1` 原始 canonical 语义
- 最终 benchmark stress 版本语义

不能把这两层混在一起解释。

## 13. 说明补写（short explanation）也是后处理的一部分

在正式 benchmark1000 的 manifest 里还可以看到一个重要步骤：

- explanation backfill

对应：

- `benchmark/processguard_eval/processguard1000_final_benchmark_modified_rows_308_manifest.json`

这说明你后来又对 `308` 行做了说明补写或重写，并且使用了 `gpt-5.4` 做 backfill。

所以你的 benchmark 成品不只是“标签和内容”，还包含一个后期加工过的 explanation 层。

## 14. 这套流程可以怎么一句话概括

如果要用一句尽量准确的话概括你的整个工程，我会这样说：

> 你先用 `pipeline_v1` 生成严格受结构合同约束的 ProcessGuard canonical 轨迹样本，再用 `rebuild` 和 benchmark 侧脚本把这些 canonical 数据组织成更大规模的 SFT pool、补跑集和 stress benchmark，最终形成 850、1000、1010 这些正式评测版本。

## 15. 我认为最重要的几个工程判断

### 15.1 你的设计重点不是“多写点样本”，而是“让标签由轨迹结构决定”

这是你整个系统最核心的地方。planner、blueprint、trace events、derive state、evidence indexes，全部都在服务这件事。

### 15.2 `rebuild` 是调度层和数据工程层，不是主生成逻辑

如果以后要继续优化样本质量，优先应该看：

- `pipeline_v1/planner.py`
- `pipeline_v1/orchestrator.py`
- `pipeline_v1/tool_call_generator.py`
- `pipeline_v1/outcome_summarizer.py`
- `pipeline_v1/filters.py`

而不是先从 `rebuild` 下手。

### 15.3 你当前的 1010 benchmark 已经是“评测产品”，不是“生成现场”

如果未来要分析模型在 1010 上的表现，应该把它当成：

- 一个经过 stress 设计和后处理的 gold benchmark

而不是当成：

- 一份原汁原味的 pipeline 输出快照

## 16. 相关关键文件索引

主生成线：

- `ProcessGuard/pipeline_v1/pipeline.py`
- `ProcessGuard/pipeline_v1/planner.py`
- `ProcessGuard/pipeline_v1/orchestrator.py`
- `ProcessGuard/pipeline_v1/filters.py`
- `ProcessGuard/pipeline_v1/query_generator.py`
- `ProcessGuard/pipeline_v1/tool_call_generator.py`
- `ProcessGuard/pipeline_v1/outcome_summarizer.py`
- `ProcessGuard/pipeline_v1/scenarios.py`

批量与调度：

- `ProcessGuard/rebuild/scripts/build_processguard1000_plan.py`
- `ProcessGuard/rebuild/scripts/run_processguard1000_controller.py`
- `ProcessGuard/rebuild/scripts/convert_pipeline_light_to_eval_canonical.py`

benchmark 组装结果：

- `benchmark/processguard_eval/processguard850_binary_stress_target834_risk800_swap170_20260427.json`
- `benchmark/processguard_eval/processguard150_original_risk800_swap24_20260427.json`
- `benchmark/processguard_eval/processguard1000_final_benchmark.json`
- `benchmark/processguard_eval/processguard1010_final_benchmark.json`

manifest / provenance：

- `benchmark/processguard_eval/processguard850_binary_stress_target834_risk800_swap170_20260427_manifest.json`
- `benchmark/processguard_eval/processguard150_original_risk800_swap24_20260427_manifest.json`
- `benchmark/processguard_eval/processguard1000_final_benchmark_manifest.json`
- `benchmark/processguard_eval/processguard1010_final_benchmark_manifest.json`

## 17. 最后给你的一个实用区分法

以后你自己或者别人再回看这个项目时，可以用下面这条简单规则快速判断“当前文件属于哪一层”：

- 如果文件在 `pipeline_v1/`，大概率是在定义“样本怎么生成”。
- 如果文件在 `rebuild/scripts/`，大概率是在定义“样本怎么批量跑、补、转、导出”。
- 如果文件在 `benchmark/processguard_eval/`，大概率是在定义“最终拿什么去评测模型”。

这个区分一旦立住，整个项目就不会再看乱。
