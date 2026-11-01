# ADR-008：运行质量分析底座（Loop 4，规则式起步）

- 状态：已采纳（0.3.x）
- 日期：2026-11-01
- 相关：ADR-005 审批门、ADR-006 验证循环、ADR-007 触发器、
  `docs/agent-dev/agent-dev-knowledge.md` §Loop 4 爬山循环、
  `docs/agent-dev/integration-review.md` §3.5

## 背景

循环工程方法论的第四层循环（爬山循环）：生产运行产生 Trace → 分析 Agent
扫描 Trace → 自动改进 harness 配置（prompt/工具/评分器）→ 内层循环更有效。
其检测规则形态：`success_rate < 0.8 → critical/file_issue`、
`avg_duration_ms > 30000 → high`、`tool_errors > 5 → medium`、
`verification_fail_rate > 0.3 → high`。

本项目的数据面**已经齐备**：
- `tool_call_records` 表：每次工具调用（args/结局/耗时/request_id/thread/module）；
- `cost_usage` 表：每次 LLM 调用（model/profile/tokens/成本/request_id）；
- `memory_ops` 表：记忆形成管线逐步审计；
- metrics（滑动窗口 + 累计计数，单进程口径）。

缺的是：把这些数据聚合成"哪些地方出了问题、建议改什么"的分析产物。

## 决策

1. **规则式分析起步，LLM 分析后置**。新增 `scripts/quality_report.py`：
   聚合三表 → 输出报告（JSON / 文本两种形态）：
   - 工具侧：成功率、错误 Top-K（错误文本关键词归因：timeout/permission/
     not found/rate limit…照外部方法论的模式提取）、慢工具 Top-K（P95 耗时）；
   - 成本侧：日趋势、模型分布、超预算预警状态；
   - 记忆侧：形成管线各步骤（extract/consolidate/profile/summary）的
     ok/error 分布；
   - 四条检测规则照搬外部方法论阈值（可经参数覆盖），命中输出
     `{severity, pattern, evidence}` 形式的 issues 与规则式 recommendations。
2. **分析对象是既有表，零 schema 变更**：sqlite/mysql 两后端均可读
   （复用各自存储层连接原语）；支持 `--since` 时间窗与 `--format json|text`。
3. **分析产物进人审流程，不自动改配置**：报告由人（或经 ADR-007 的
   webhook/定时任务触发后推送）消费；改进建议的落地路径是人工改 prompt /
   工具配置 / 阈值——**不做 ConfigOptimizer 式自动改写**。
4. **LLM 分析 Agent 与自动改进列 P2**，触发条件（写入路线图）：规则式报告
   运行一段时间、积累出"规则抓不到但人能看出来"的失败模式清单后，再评估
   LLM 分析的收益；届时分析 Agent 的输入就是本报告的 JSON 形态（契约已稳定）。

## 被否方案

- **直接上 LLM 分析 Agent**（照外部方法论 TraceAnalyzer 全量实现）：
  在规则式报告尚未跑通前，LLM 分析的输入契约与成本预算都没有依据；
  且当前失败模式的主体（超时、配置错误、工具异常）规则即可归因。
- **自动改写 harness 配置**（ConfigOptimizer 式）：prompt/检索参数的自动变更
  缺少回归验证手段（无 eval 集），违背"人审把关"原则；外部方法论自身也把
  `prompt_optimization` 列入 `human_review_required`。
- **新建通用 trace 表（span/tree 结构）**：三张既有表 + checkpointer 状态
  已覆盖当前分析维度；通用 trace 是 P2 项（若 LLM 分析立项再评估），
  现阶段引入只增运维面。
- **做进 `/metrics`**：metrics 是进程内滑动窗口口径（重启清零、1000 样本），
  不适合跨周期质量趋势；质量报告走持久化表，两者互补不合并。

## 后果

正面：

- Loop 4 的"分析"半环落地，且完全建立在既有数据面上，无迁移成本；
- 报告 JSON 形态成为后续 LLM 分析与改进建议的稳定输入契约；
- 与 ADR-007 组合后可实现"每天自动出质量报告"的运营闭环（人审环节保留）。

代价（明示）：

- 规则式归因对未知失败模式无能为力（关键词不命中即归入 other）——这正是
  P2 LLM 分析的立项依据；
- 报告粒度是"时间窗聚合"，不能回放单次运行（单次回放仍走
  `GET .../tool-calls` 端点与日志）；
- mysql 后端的聚合查询在生产大表上需要索引配合（`tool_call_records` 已有
  thread 维度索引，时间维度索引视数据量后续补充迁移）。
