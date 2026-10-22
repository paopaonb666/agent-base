# ADR-001：以 LangGraph 作为基座运行时

- 状态：已采纳（Accepted）
- 日期：2026-09-20
- 决策者：agent-base 维护者
- 相关：ADR-002（模块契约）、`docs/deployment.md`、`pyproject.toml` 依赖地板

## 背景

agent-base 的定位是"编排之上的治理基座"：长期记忆、成本治理、事件可观测、
工具治理。它必须回答一个前置问题——**图编排本身自研还是委托给成熟运行时**。

候选方案与当时（2026 年 9 月）的评估结论：

| 方案 | 状态持久化 | 流式输出 | 多 Agent 编排 | 生态 / 工具 | 自托管与成本 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| **LangGraph ≥1.2.6** | checkpointer 一等公民（memory / SQLite / MySQL 后端齐全） | `astream` 原生 token 级流 | `create_supervisor` / subgraph 组合 | LangChain 工具、回调、追踪生态 | 纯 Python，零托管依赖 | **采纳** |
| 自研 asyncio 状态机 | 需自建快照、恢复、迁移 | 需自建背压与事件契约 | 需自建调度与失败恢复 | 无，全部自造 | 低依赖，但长期维护成本最高 | 否决：把有限精力耗在编排轮子上，偏离"治理"定位 |
| LangChain 经典 `AgentExecutor` | 无持久化图状态（仅消息历史） | 回调式，token 流需额外适配 | 弱 | 成熟 | 低 | 否决：无法满足"重启恢复对话 + 跨轮线程" |
| OpenAI Assistants / Responses API | 托管线程状态 | 托管流 | 托管 handoff | 绑定单一厂商 | 按 token 计费 + 数据出域 | 否决：锁定厂商，与"openai 兼容、可切换 provider"冲突 |
| Temporal / Prefect 等工作流引擎 | 强（事件溯源） | 需自建 | 强 | 重 | 需额外集群与运维 | 否决：本地单机自用场景过重，运维面超出收益 |
| CrewAI / AutoGen 等多 Agent 框架 | 弱到中 | 中 | 强 | 中 | 低到中 | 否决：偏"角色扮演式"多 Agent，缺少可替换的持久化与细粒度治理钩子 |

## 决策

采用 **LangGraph 作为唯一编排运行时**，基座不自研图执行器。具体约定：

1. 模块通过 `AgentModule.build_graph(ctx)` 返回**编译后的 LangGraph 图**
   （`core/contracts.py` 的 `Graph` 别名），基座只把它当作不透明对象回传给入口。
2. 对话状态一律经 `ctx.checkpointer` 编译进图；后端由
   `CHECKPOINTER_BACKEND` 选择（`memory` / `sqlite` / `mysql`）。
3. 多 Agent 编排复用 `langgraph-supervisor`，基座提供 `supervisor` 入口名，
   不另造调度协议。
4. 基座只在 LangGraph 之上补治理设施（记忆、成本、事件、工具池），
   不包装、不隐藏 LangGraph 的 API——需要自定义拓扑的模块可直接使用原生 API。

### 版本地板：`langgraph>=1.2.6` 是安全下限，不是功能偏好

地板由三个已披露漏洞决定，任何低于该版本的安装都不被接受：

| CVE | 影响面 | 严重度 | 1.2.6 中的修复 |
| --- | --- | --- | --- |
| CVE-2025-68664 | 反序列化注入 | CVSS 9.3 | 收紧反序列化白名单 |
| CVE-2025-67644 | SQLite checkpointer SQL 注入 | 高 | 参数化 checkpointer 查询 |
| CVE-2026-34070 | checkpointer 路径穿越 | 高 | 规范化并限制落盘路径 |

因此 `pyproject.toml` 写死 `langgraph>=1.2.6`，CI 的 `pip-audit --skip-editable`
作为常驻兜底；升级时必须同时跑通四道质量门，而不是只改版本号。

## 后果

**正面**

- 持久化、流式、恢复、多 Agent 组合由上游维护，基座专注治理能力。
- checkpointer 后端可替换（memory / sqlite / mysql），测试可用 `memory` 零依赖跑通。
- 生态工具（LangChain `BaseTool`、回调、追踪）直接复用，工具池与计量可以挂标准钩子。

**约束与代价**

- **版本地板是硬约束**：下游项目若锁死旧版 LangGraph，本基座不承诺兼容；
  安全修复优先于兼容性。
- **上游 API 演进风险**：`add_node` 等泛型签名在 1.2.x 内部有过调整，
  基座在边界处集中做类型收敛（见 `modules/planner/graph.py` 的 `cast(Any, ...)`），
  避免 `# type: ignore` 随上游版本抖动。
- **抽象泄漏**：模块作者仍需理解 LangGraph 的 state / reducer / checkpointer 语义；
  基座的 `build_single_agent_graph` 只覆盖"单 LLM 节点 + 可选工具循环"这一最常见形态。
- **不提供迁移工具**：从旧版本或其它框架迁移图定义不在基座职责内。

## 备选方案为何不满足要求

- 自研编排：需要自建检查点格式、恢复语义、迁移链（现有 alembic 0001 复用
  LangGraph 内置迁移即为例证），维护成本与安全面都不可接受。
- 托管线程 API：与"可自托管、可切 provider、数据不出域"的部署目标冲突。

## 参考

- `pyproject.toml`：`langgraph>=1.2.6`、`langgraph-checkpoint-sqlite`、
  `langgraph-checkpoint-mysql`、`langgraph-supervisor` 的依赖地板与注释。
- `src/agent_base/extensions/memory.py`：checkpointer 后端装配。
- `docs/deployment.md`：三种部署形态与单进程口径。
