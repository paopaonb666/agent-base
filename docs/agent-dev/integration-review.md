# 循环工程知识融入评审报告（agent-base）

- 日期：2026-11-01
- 评审对象：agent-base @ main（0.3.0）
- 知识输入：[agent-dev-knowledge.md](agent-dev-knowledge.md)（腾讯文档「大模型应用开发8期」提取整理）
- 结论先行：**10 个可融入主题中 5 个立项实施（M0-M5）、2 个写入路线图（P2）、3 个明确不采纳**；
  全部决策已固化为 ADR-004 ~ ADR-008。

---

## 一、评审范围与方法

逐主题把知识文档内容映射到代码库现状（三个探索代理完成的全库深查，关键结论均带
文件路径复核），每个主题给出：文档出处 → 项目现状（文件级）→ 融入方案 → 优先级 →
工作量 → 风险。融入方案必须满足三条约束：

1. **不破坏依赖倒置边界**（ADR-002/003：基座源码不 import 记忆包、业务以模块接入）；
2. **默认关闭、启用即承诺**（fail-fast），不改变既有部署的任何默认行为；
3. **依赖地板**：除 `httpx`（openai SDK 已传递引入）外不新增运行时依赖。

## 二、成熟度定位（对照循环工程 Level 1-4 模型）

| 等级 | 循环工程定义 | agent-base 现状 | 判定 |
| --- | --- | --- | --- |
| Level 1 能让 Agent 干活 | 模型在循环中调工具 | LangGraph 编排 + 共享工具池（超时/审计/异常归一） | ✅ 已达成 |
| Level 2 保证干对活 | 验证循环（评分器 + 反馈重试） | planner 的 check/replan 节点是局部先例；无通用评分契约 | ⚠️ 部分 → M3 |
| Level 3 让 Agent 自动干活 | 事件驱动（cron/webhook/消息触发） | 仅外部 cron 脚本 + `spawn_background`；无触发器设施 | ❌ → M4 |
| Level 4 让 Agent 越干越好 | Trace 分析 → 自动改进 harness | `tool_call_records`/`cost_usage`/`memory_ops` 数据面已齐，缺分析→改进闭环 | ❌ → M5（数据已备，底座好） |

**总体判断**：本项目在"生产治理"维度（成本熔断、事件契约、审计）领先课程内容，
在"自动化运营"维度（Loop 3/4）落后于课程方法论——而 Loop 3/4 所需的数据底座
本项目已经建好，融入成本低、收益高。

## 三、逐主题融入点分析

### 3.1 记忆检索 rerank 精排层 ——【P0，M1 实施】

- **文档出处**：课堂笔记 §记忆体系统（"检索 topk=30 → rerank → 10 份"）；anything-chat-rag 检索管线。
- **项目现状**：`agent_base_memory/retrieval.py` 的 `recall_memories`（:319-381）四分量线性融合
  （vector 0.40 / keyword 0.30 / recency 0.15 / salience 0.15，权重可配）后**直接按总分截断 top_k**，
  无精排层。召回候选为 `store.list_memories(limit=2000)` 全量拉取。
- **融入方案**：宽召回（fetch_k = top_k × multiplier）→ 精排 → 截回 top_k。
  - `Reranker` 协议仿 `embeddings.py:EmbeddingClient`（**返回 None 即降级** + 失败冷却），
    实现走 openai 兼容 `/rerank` 端点（硅基流动 bge-reranker 系列）；
  - 插入点：`MemoryService.search`（service.py:314-344）——工具/HTTP/注入三条路径的汇合处；
  - rerank 分写入 `ScoredMemory.components["rerank"]`（frozen dataclass 的 dict 分量，零 schema 破坏）；
  - 宽召回阶段 min_score 减半（精排前宁多勿漏，与 service.py:326-330 既有降级注释同一哲学）；
  - 整合阶段（pipeline.py consolidate 的 top_k=3 相似检索）**旁路 rerank** 保延迟；
  - 知识库路径 `knowledge.py:search_knowledge` 同款接入。
- **工作量**：1 个新模块（rerank.py）+ 4 处接线 + 配置 7 项 + 测试文件。
- **风险**：精排引入的延迟（fetch 阶段已有限流/降级兜底）；rerank 分与混合分不同分域
  （用"rerank 主键、混合分次键"规避加权融合的语义问题）。
- **决策记录**：ADR-004。

### 3.2 敏感工具审批门（Human-in-the-Loop）——【P0-P1，M2 实施】

- **文档出处**：循环工程 §人类监督与专家介入（敏感工具调用前 `interrupt` 确认；
  approve/reject/modify 三分支）。
- **项目现状**：
  - `ToolSpec.safety`（read/network/exec）**注释明说"为将来的审批门预留的元数据；本阶段只登记不强制"**；
  - 全仓无 interrupt/approval 实现；`_TimeoutTool`（core/tools.py）是所有工具执行的唯一收口
    （含 MCP、记忆包工具）；
  - checkpointer（sqlite/mysql）已具备 LangGraph interrupt 状态持久化条件；
  - SSE 事件契约是封闭 Pydantic 模型集（extensions/events.py）。
- **融入方案**：**名单制审批**——`TOOL_APPROVAL_ENABLED` + `TOOL_APPROVAL_TOOLS`（CSV 工具名，
  覆盖 MCP 前缀名）+ `TOOL_APPROVAL_SAFETY_LEVELS`（作用于 TOOLKIT 已注册条目的 safety 级）。
  图工厂在 `agent → tools` 边之间插 gate 节点：命中名单 → `langgraph.types.interrupt()`；
  SSE 新增 `ApprovalRequestEvent`；新增 `POST .../threads/{id}/resume` 端点续流。
  三分支语义照文档：approve 直接过、reject 注入拒绝 ToolMessage、modify 换参重执行。
- **工作量**：五个文件改动（config/graphs/events/sse/routes）+ gate 节点与 resume 语义测试。
- **风险**：改动面最大（SSE 封闭契约扩展、interrupt 在 astream 中的流语义）；用"默认关闭 +
  未启用时零行为变化"的回归测试兜底。
- **决策记录**：ADR-005。

### 3.3 验证循环（Loop 2）——【P1，M3 实施】

- **文档出处**：循环工程 §Loop 2（Grader/Rubric、确定性 vs 代理式评分器、VerificationLoop
  引擎、历史最佳回退）；完整参考实现见 reference-loop-engineering-full.md §Loop 2。
- **项目现状**：无通用评分契约。`modules/planner/` 的 check/replan（`nodes_check.py`）是
  最接近的局部实现，但只服务 plan-execute 拓扑；`examples/custom_memory_port.py` 证明了
  Port 协议 + 组合根注入是基座认可的二开形态。
- **融入方案**：`core/verification.py` 提供 `Grade / GradingResult / Grader 协议 / VerificationLoop`
  （max_retries、pass_threshold、require_all_pass、加权聚合、反馈拼接、历史最佳回退）+
  `LLMJudgeGrader`（JSON verdict 契约、解析失败即 FAIL）。**可选能力**：模块自行取用，
  不强制接线（planner 已有自己的 check，不重复建设）。
- **工作量**：1 个新模块 + 测试。
- **风险**：低（纯增量、零接线）；主要风险是"建了没人用"——通过 README 使用示例缓解。
- **决策记录**：ADR-006。

### 3.4 事件驱动触发器（Loop 3）——【P1，M4 实施】

- **文档出处**：循环工程 §Loop 3（CronTrigger/WebhookTrigger/SlackTrigger 事件引擎；
  LangSmith Deployment 的声明式触发器配置形态）；课堂笔记 §NL2SQL（点赞回写知识库等自动化诉求）。
- **项目现状**：无 cron/webhook 设施。有：lifespan 周期任务先例（`_flush_cost_ledger` 5s 循环）、
  fire-and-forget 通道（`background.spawn_background`）、外部 cron 脚本
  （`scripts/memory_nightly_capture.py`，无观测无退避）、HMAC 认证（`server/auth.py`）。
- **融入方案**：`extensions/scheduler.py` 进程内调度器——零新依赖（interval 与 daily@HH:MM
  两种调度表达，**不引 croniter/APScheduler**）；`SCHEDULER_JOBS_JSON` 声明任务；
  内置 `memory_nightly_capture` 任务类型（复用脚本逻辑，脚本改薄壳）；
  webhook 入口 `POST /v1/triggers/{job}`（HMAC 复用）；失败指数退避 + /health 暴露。
- **工作量**：1 个新模块 + 1 个路由 + lifespan 接线 + 测试。
- **风险**：多 worker 部署下调度器会重复触发——ADR 记录"单进程口径"，多 worker 留 P2。
- **决策记录**：ADR-007。

### 3.5 运行质量分析闭环（Loop 4）——【P1，M5 实施】

- **文档出处**：循环工程 §Loop 4（TraceStore 统计、TraceAnalyzer 五维分析、四条检测规则、
  ConfigOptimizer）；"前三个循环自动化工作，第四个自动化改进"。
- **项目现状**：数据面已齐——`tool_call_records`（args/结局/耗时/request_id/thread）、
  `cost_usage`（model/tokens/成本/request_id）、`memory_ops`（形成管线审计）、
  metrics 滑动窗口；**缺的是聚合分析与规则告警**。
- **融入方案**：`scripts/quality_report.py` 规则式分析（不引 LLM）——成功率、工具错误 Top-K
  （关键词归因）、慢工具、成本日趋势、四条检测规则（success_rate<0.8 / latency>30s /
  tool_errors>5 / verification_fail>0.3）命中输出 issues+recommendations。
  LLM 分析 Agent 与自动改配置**后置 P2**，且改进建议先落 Issue 人审（符合文档
  `human_review_required` 原则）。
- **工作量**：1 个脚本 + 测试；无 schema 变更。
- **风险**：低。
- **决策记录**：ADR-008。

### 3.6 中间件对齐（langchain AgentMiddleware / wrap_model_call）——【不采纳】

- **文档出处**：课堂笔记 §中间件（`AgentMiddleware`、`before_model`、`wrap_model_call`、
  PDF 解析中间件、动态提示词中间件）。
- **项目现状**：项目**不用** AgentMiddleware——自研钩子 `build_model_input`
  （core/context.py:76）在共享图工厂 `call_model` 内统一调用，实现了与"中间件"等价的
  三个能力：注入去重、token 预算修剪、工具对不拆散；且全库单点收口。
- **不采纳理由**：
  1. 课程中间件的主要用途（PDF 解析、动态提示词、检索注入）在本项目分别由
     附件管线（filestore + 上下文冻结段）、`compose_context`/`recall_context` 注入、
     上下文引擎覆盖；
  2. 引入 AgentMiddleware 会形成两套钩子体系并存，违背"本地 == CI"的简单性纪律；
  3. planner 的自定义拓扑证明：需要更细控制时，直接用 LangGraph 原生 API + 自研钩子
     组合即可。
- **保留接口**：若未来确需模型调用级拦截（如全局重试/日志），优先扩展
  `build_llm` 的 callbacks 收口（CostMeterHandler 已在此先例），而非引入中间件框架。

### 3.7 graphiti 时序知识图谱记忆 ——【P2 路线图，不本次实施】

- **文档出处**：课堂笔记 §记忆图谱（graphiti：对话/实体/关系存为带时间戳节点，
  能回答"偏好如何演变"）+ NL2SQL 六阶段引擎的 Phase 0（Memory 唤起）。
- **项目现状**：`MemoryPort` 协议依赖倒置 + `agent_base.memory` entry point，
  记忆实现可整体替换；现有画像（Memobase 式）+ 事件记忆（时间半衰期）已覆盖
  部分时序语义，但无实体关系图谱。
- **融入路径**（写入路线图，不实施）：新增 `agent-base-memory-graphiti` 发行包，
  经同一 entry point 提供第二种实现；或先以 MCP 工具形态接入共享工具池验证价值。
  依赖评估：falkordb/graphiti 依赖较重，需独立部署图数据库。
- **判定**：P2——等 M5 质量报告能回答"哪些查询因缺关系记忆而失败"后再立项。

### 3.8 Skills 技能体系 ——【部分采纳，随 M3/M4 隐式落地】

- **文档出处**：课堂笔记 §Skills（提示词指令 + 工具调用引导；skills+mcp 解耦客户端）。
- **项目现状**：模块即"技能包"——`AgentModule` 契约（name/description/build_graph/get_tools）
  + supervisor handoff description 就是课程 Skills 语义的基座形态；提示词经
  `build_single_agent_graph(system_prompt=...)` 注入。
- **判定**：契约已等价，不另建 Skills 子系统。差距仅在"跨客户端复用"（把同一套
  skills 暴露给 claude code/cursor 等外部客户端）——这经 MCP 出口即可达成，
  列入 P2 观察项，不立项。

### 3.9 动态提示词中间件 ——【已有等价物】

- **文档出处**：课堂笔记 NL2SQL 改造（"基于动态提示词中间件完成系统提示词的更新"）。
- **项目现状**：每轮注入路径本身就是"动态提示词"——`compose_context` 在 invoke 时
  组装冻结段 + 召回段（`agent_base_memory/context.py`），配合基座级
  `build_model_input` 去重修剪。等价能力已存在，无需新机制。

### 3.10 明确不采纳清单（范围红线）

| 课程内容 | 不采纳理由 |
| --- | --- |
| NL2SQL / SQL-of-Thought 全套 | 垂直业务系统，不属基座（ADR-002 范围红线）；六阶段查询引擎思路供业务模块参考 |
| FalkorDB / Neo4j 图数据库 | 同上；且基座存储跟随 CHECKPOINTER_BACKEND，无图存储诉求 |
| 多数据库 SqlRunner 抽象（10 种库） | 基座只有 sqlite/mysql 两种存储后端，无多库诉求 |
| swarm 多智能体框架 | 课程结论与项目一致：单 Agent + tools（子智能体即工具，supervisor 模板已内置 handoff） |
| CopilotKit / AG-UI / Sigma.js 前端 | 前端契约已有 SSE 事件契约（step/delta/plan/tool），等价解耦；文档前端选型供前端仓库参考 |
| Mem0/Letta 直接引入 | 项目已自研吸收三者架构（mem0 两阶段/Letta 块/Memobase 画像），引入即架构回退 |

## 四、实施路线图

| 里程碑 | 内容 | 优先级 | 状态 |
| --- | --- | --- | --- |
| M0 | 死配置 `MEMORY_CAPTURE_EVERY_TURNS` 修复（orchestrator 消费 human_count 门控） | 微 | 本次实施 |
| M1 | 记忆 rerank 精排层（ADR-004） | P0 | 本次实施 |
| M2 | 敏感工具审批门 HITL（ADR-005） | P0-P1 | 本次实施 |
| M3 | 验证循环质量契约（ADR-006） | P1 | 本次实施 |
| M4 | 进程内事件触发器（ADR-007） | P1 | 本次实施 |
| M5 | 运行质量分析报告（ADR-008） | P1 | 本次实施 |
| P2-1 | graphiti 作为 MemoryPort 第二实现（ADR 待立项） | P2 | 路线图 |
| P2-2 | Loop 4 LLM 分析 Agent + 改进建议落 Issue（人审门） | P2 | 路线图 |
| P2-3 | 多 worker 调度器选主 / webhook 出站事件 | P2 | 路线图 |
| P2-4 | Skills 跨客户端复用（MCP 出口） | P2 | 路线图 |

## 五、与既有文档的一致性

- 本报告与 README「范围红线」「生产关切」表无冲突：M1-M5 全部落在"长期记忆 /
  成本治理 / 事件可观测"三件设施的延长线上，不越 ADR-002 模块契约边界；
- ADR-004 ~ 008 编号接续 docs/adr/ 既有 001-003；
- 默认行为零变化：所有新能力默认关闭，CI 现有测试应全数通过。
