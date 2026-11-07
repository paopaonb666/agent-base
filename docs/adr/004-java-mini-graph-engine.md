# ADR-004：Java 侧自研迷你图运行时

- 状态：已采纳（Accepted）
- 日期：2026-11-07
- 决策者：agent-base 维护者
- 相关：ADR-001（Python 侧以 LangGraph 作为运行时）、ADR-002（模块契约）、ADR-003（记忆包拆分）、`docs/java-differences.md`（差异登记簿）、`java/README.md`

## 背景

`feat/java-rewrite` 分支把基座重写为 Java 21 / Spring Boot 3。JVM 生态里
**没有 LangGraph 的等价物**：langchain4j 提供 LLM 客户端与工具绑定，但不
提供"带 checkpointer 的图执行、条件路由、流式三通道、递归预算"这套编排
运行时；community 移植要么停更，要么语义与本仓依赖的 LangGraph 子集对不上。

ADR-001 的评估结论（"自研状态机 → 否决：把有限精力耗在编排轮子上"）与其
决策 1–4（`build_graph` 返回编译图、checkpointer 编译进图、复用
langgraph-supervisor、基座只补治理设施）**全部是 Python 侧的决策**——它
的前提"存在一个成熟、被上游持续维护的运行时"在 JVM 侧不成立，因此不构成
对 Java 侧的约束。

同时，Java 重写第一期的实际用面很窄：本仓全部模块（chat / writer /
planner / supervisor handoff）只用到 LangGraph 语义的一个子集，不存在
interrupt、subgraph、动态拓扑这些重特性。

## 决策

Java 侧**自研迷你图运行时**（`S/core/graph/`：`GraphBuilder` →
`CompiledGraph` → checkpointer SPI），只保留并锁定以下 LangGraph 语义子集：

| 保留的语义 | 实现与锚点 |
| --- | --- |
| 节点 + 条件边 | `GraphBuilder`（node/edge/conditionalEdge），编译期拓扑校验 |
| 三通道流式 | 对应 LangGraph `astream(stream_mode=["messages","updates","custom"])`，SSE 事件契约逐字节对齐 Python 版 |
| checkpointer 三后端 | memory / sqlite / mysql（SPI：`Checkpointer`），快照格式自管（见下） |
| recursionLimit + H5 预算推导 | 步数超 `RunConfig.recursionLimit` 抛 `GraphRecursionError`；planner 最坏步数编译期推导 `Graphs.plannerWorstSteps` = `8 + 2 * maxSubtasks * (subtaskToolRounds + 1)`（默认 98），组合根启动期不足即 WARN（#229；`S/core/graphs/Graphs.java:47-55`、`S/core/AgentRuntime.java:267-278`） |
| 编译期拓扑校验 | `GraphBuilder.compile` 三类 `IllegalStateException`：入口未设置/未注册（:94-99）、边后继未注册（:100-104）、节点缺出边（:105-109）（`S/core/graph/GraphBuilder.java:87-113`）。**已知缺口（#202）**：条件边未知目标目前由运行期兜底、执行语义由 #285 基线锁定——编译期声明式目标集校验是后续演进项 |
| per-thread 互斥 | `CompiledGraph` per-threadId `ReentrantLock`，第二个 run `tryLock` 快速失败 `ThreadBusyError` → 路由层 409（#197） |
| 快照版本化与损坏隔离 | `toJson` 带 `v=1`；`fromJsonDetail` 分类 ok/corrupt/version，未知版本拒载保留原行；sqlite 损坏行进 `ab_thread_states_quarantine` 隔离表（#200/201/235） |
| 工具池编译期绑定 | `ToolPool` 是 `compile` 参数不是编译后可覆写字段（消除 post-compile volatile 覆写的竞态面，#238/131；`GraphBuilder.java:87-91`） |

执行语义由 `CompiledGraphTest` 11 用例锁定（#285：递归触顶/步事件序/取消/
put 失败/条件边未知节点/并发互踩/跨 invoke 恢复/编译期校验）。

### 明确不做（与 LangGraph 的刻意偏离）

- **superstep / msgpack 检查点表**：状态存自有 `ab_thread_states` JSON 快照
  （版本化 + 隔离，见上）；业务表与 Python 版共用，checkpointer 表两版各管
  各的（`docs/java-differences.md` 第 1 行）。
- **interrupt / resume 全语义**：不实现人机环中断；恢复=同一 threadId 再
  invoke（快照回放）。
- **子图 checkpoint namespace**：supervisor handoff 的子图快照用派生键
  `<supThreadId>@<module>` 隔离（#150，`SupervisorGraphs.java:217-221`），
  不做通用子图机制。

### 演进规则（何时换轮子）

满足任一条即应重开决策，而不是继续加补丁：

1. **语义超集需求**：任一模块需要 interrupt/resume、动态拓扑或子图组合
   才能表达时——先评估 langchain4j 生态的编排扩展，再评估引擎 2.0；
2. **校验债务到期**：条件边声明式目标集（#202）之外再出现需要编译期保证
   而运行期兜底不住的语义时；
3. **维护成本倒挂**：迷你运行时的月均修复成本接近引入成熟运行时的迁移
   成本（以 git 计数与缺陷率复核）。

升级时快照兼容是硬约束：`ab_thread_states` 的 `v=1` 版本头必须被新引擎
读取或一次性迁移，禁止静默丢会话。

## 与 ADR-001 的关系

ADR-001 的 LangGraph 决策**在 Python 侧继续完全有效**（运行时选型、依赖
地板、pip-audit 兜底均不变）；本 ADR 是 Java 侧的**例外登记**：两侧以
"对外行为一致、内部运行时各自"并存，全部已知差异收敛在
`docs/java-differences.md` 登记簿中维护。

## 参考

- `S/core/graph/GraphBuilder.java`、`S/core/graph/CompiledGraph.java`：
  迷你运行时本体与校验/互斥/快照锚点。
- `S/core/graphs/Graphs.java:47-55`：planner 最坏步数推导（#229）。
- `docs/design-review.md` H5：递归预算口径与公式。
- `docs/adr/001-langgraph-as-runtime.md`：Python 侧运行时决策（本 ADR 的
  例外对象）。
