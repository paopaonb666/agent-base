# ADR-011：工具调用 few-shot 与评测集（两臂对照）

- 状态：已采纳（0.3.x）
- 日期：2026-11-01
- 相关：ADR-006（`LLMJudgeGrader`，完成率判定复用）、ADR-008（quality_report 的
  纯函数 + CLI 形态）、ADR-010（共享 embedder 的来源）

## 背景

核查点 4：工具调用的准确性既没有样本注入（只有 docstring 里的静态示例），也没有
度量——"few-shot 有没有用"在本项目无法回答。方法论要求：自建评测集、有/无样本
两臂对照、三指标（工具选择准确率/参数准确率/任务完成率）、动态选样避免不匹配
样本误导。

## 决策

1. **动态选样**（`core/tool_fewshot.py::ToolFewShot`）：种子库批量嵌入一次（懒初始化），
   每轮按当前查询余弦相似度取 top-n（默认 2）注入；**低于 `min_score`（默认 0.30）
   宁可不注入**——零样本优于错样本。
2. **可用性过滤**：渲染时传入本轮实际绑定集（ADR-010 路由之后），未绑定工具的
   示例滤掉——给模型看一个调不存在的工具的示例，等于教它幻觉。
3. **文本段注入**：样本渲染成 system prompt 追加段（临时拼装、不落库、不参与
   修剪去重，与模块提示词同规则）；**不伪造 user/assistant 消息历史**——假历史
   会进 checkpointer 与记忆形成管线，污染真实对话轨迹。
4. **种子库打包内**（`tools/fewshot_seed.jsonl`，15 条：current_time/calculator/
   json_query/web_search/python_repl 各 2-4 条，web_search 的正误查询对入库）：
   `evals/` 在仓库根不进 wheel，种子必须随包分发才能"默认可用"；
   `TOOL_FEWSHOT_PATH` 指向任意 JSONL 覆盖。坏行跳过并告警，不废整个注入。
5. **默认关闭**（`TOOL_FEWSHOT_ENABLED=false`）；未启用/无 embedder/空样本库
   三种降级都安静退出为"不注入"，图工厂零变化。
6. **评测脚本** `scripts/tool_eval.py`（quality_report 同形态：纯函数核心 + 薄 CLI）：
   - 评测集 `evals/tool_calls.jsonl`（15 条，全部只依赖默认启用的工具，开箱即跑；
     覆盖唯一工具/参数格式陷阱（json_query 路径、calculator 表达式语法）/多步链/
     无工具对照/仅完成率用例）；
   - **两臂**：baseline 关 few-shot，treatment 开；嵌入不可用时报告显式注明
     "treatment 与 baseline 等价，非有效对照"；
   - **三指标**（纯函数，可单测）：工具选择准确率（首调命中/集合覆盖两个口径）、
     参数准确率（exact/contains/regex 三种字段匹配，**子集语义**——只校验用例
     列出的字段，模型多传不惩罚）、任务完成率（contains 零成本 / llm 复用
     `LLMJudgeGrader`）；
   - 分母纪律：选择指标只计声明了期望工具的用例；参数指标只计"工具被调过"的
     字段对（没调用已被选择指标惩罚，不重复罚）；完成率只计写了 success_criteria
     的用例；
   - 轨迹从图执行返回状态提取（全部 AIMessage.tool_calls），不回读审计表——
     一次运行一次落袋，不依赖存储后端；
   - 输出 JSON（沿 ADR-008 稳定契约风格，便于留档对比）+ 人读 text（每指标
     baseline/treatment/delta 三列）。

## 被否方案

- **静态全量注入**（所有样本常驻 system prompt）：样本随工具增长线性吃 prompt
  预算，且不匹配样本必然误导——与核查点 4.iv 正面冲突。
- **伪造 few-shot 消息对**（AIMessage(tool_calls) + ToolMessage 注入历史）：污染
  checkpointer/记忆形成/历史回放三处真实轨迹，代价远超收益。
- **评测轨迹从审计表回读**：需要时间窗/线程过滤，多一层数据依赖；图返回状态
  本来就有全部 tool_calls——直取更简单且确定性。
- **参数严格全等匹配**：模型多传无害字段（如 max_results 默认值）不该判错；
  json_text 这种长文本只能 contains/regex——所以匹配模式是 per-case 的。
- **把 few-shot 注入 planner**：planner 的 EXECUTE_SYSTEM 是结构化任务提示词，
  样本混入会干扰计划格式；首期只在共享图工厂生效。

## 后果

正面：

- 核查点 4 闭环：样本可注入（动态、带阈值护栏）、效果可度量（两臂三指标）、
  缺陷可规避（相似度选样 + 可用性过滤 + 分母纪律）；
- 评测脚本零后端依赖、纯函数可单测，`python scripts/tool_eval.py` 一条命令出对照表。

代价（明示）：

- 注入使 system prompt 每轮多一段（top_n 条样本，约百来 token）；
- 有效对照需要 `MEMORY_EMBEDDING_API_KEY`（否则两臂等价，报告注明）；
- 评测跑批消耗真实 LLM 配额（15 条 × 2 臂 × 多轮工具往返），按需跑而非 CI 常驻。
