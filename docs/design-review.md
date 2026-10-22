# 设计边界与质量门

本文记录 agent-base 在 0.3.0 时点**明确接受的设计边界**。它们不是待办清单，
而是有意为之的取舍：每条都写明当前口径、触发条件与升级路径。发布前请连同
`docs/adr/` 一起阅读。

## H2：记忆检索按用户最多 2000 条候选

**口径**：记忆召回先在进程内取该用户的候选集（上限 2000 条），再跑
BM25 + 向量 + recency/salience 的混合排序。候选之外的内容不参与打分。

**为什么这样定**

- 单机自用的记忆规模通常在数百到数千条，全量打分延迟可接受；
- 召回质量取决于候选集覆盖率，盲目截断会漏掉旧但高相关的事实，
  因此保留 2000 这个相对宽松的上限，而不是 200；
- 上索引（ANN / 全文索引）会引入外部组件与一致性维护，超出当前"单进程、
  零外部依赖可跑"的定位。

**代价**

- 每轮召回的 CPU 成本随候选数线性增长，2000 条是延迟与召回率的折中点；
- 单用户记忆超过 2000 条后，更早的内容只能靠关键词/实体命中进入候选，
  纯语义相似的长尾会被稀释。

**升级路径**：把候选检索下推到存储层（MySQL 全文 / 向量列，或独立索引），
基座只保留 `MemoryPort` 的 recall 契约；届时上限变成查询参数而非常量。

## H3：治理状态是单进程口径

**口径**：成本账本、预算熔断、nonce 防重放、`/metrics` 计数器、
BM25 索引缓存、SSE 连接都按进程持有，不跨 worker 共享。

| 能力 | 重启后 | 多 worker |
| --- | --- | --- |
| 成本账本（sqlite/mysql） | 从库恢复 | 各进程独立，采集侧汇总 |
| 成本账本（memory） | 清零（只少记，不会误熔断） | 各自清零 |
| 预算熔断 | 依账本重建 | 各自判定，可能短暂超支 |
| nonce 缓存 | 丢失 | 需网关兜底 |
| `/metrics` | 归零 | 按实例聚合 |
| 检索缓存 | 自动重建 | 内容指纹保证一致 |

**为什么这样定**：单进程让所有治理逻辑可以用普通内存结构表达，无需分布式锁、
租约或外部计数器；对自托管单机场景，这是复杂度最低且可解释的方案。
**代价**：`--workers > 1` 时预算和防重放不再全局成立。
**升级路径**：把账本、nonce、metrics 外置到共享存储/网关，见
`docs/deployment.md` 的口径表与「外置方向」。

## H4：CLI 与 HTTP 入口能力不等价

**口径**：CLI 与 HTTP 共用同一套模块、图、工具池与 checkpointer，但 CLI
**不注入**记忆上下文、**不运行**记忆形成管线，也**不支持**附件上传与
多用户作用域；它面向"单用户、一次一问"的本地使用。

**为什么这样定**：记忆写入涉及用户身份、捕获频率、异步总结与成本，
在无身份、无持久会话的 CLI 里语义不成立；强行复用会让 CLI 行为依赖
隐式全局状态。把差异做成显式的入口差异，比在共享层加分支更可维护。

**代价**：同一段对话在 CLI 与 HTTP 下回答可能不同（HTTP 有长期记忆）。
文档已在 README 与 `docs/deployment.md` 标注，调用方需按入口预期使用。

**升级路径**：若需要 CLI 记忆，应先定义本地用户身份与 SQLite 作用域，
再把记忆装配抽成入口可选的 builder，而不是在 CLI 里直接调用服务。

## H5：planner 递归预算与全局递归上限不完全对齐

**口径**：`PLANNER_MAX_SUBTASKS` / `PLANNER_SUBTASK_TOOL_ROUNDS` /
`PLANNER_MAX_REPLANS` 共同决定 planner 的最坏步数；在当前默认值下最坏约
**34 步**，高于 `AGENT_RECURSION_LIMIT=25`。

**为什么这样定**：planner 的每次"子任务 + 工具轮次 + 重规划"都是一次
可观测的决策，压缩全局递归上限会先杀掉正常的长任务；反过来把上限调到 34+
会让失控循环的代价变高。当前选择是"让预算由 planner 参数表达，全局上限
只做兜底"。

**后果**：极端输入下 planner 可能先触发 LangGraph 的递归上限错误，
而不是优雅地返回部分结果；错误信息可定位到 recursion limit。
**缓解**：调低 `PLANNER_MAX_REPLANS` 或子任务工具轮次；把
`AGENT_RECURSION_LIMIT` 调到 `8 + 2 * max_subtasks * (subtask_tool_rounds + 1)`
可覆盖最坏路径。**升级路径**：为 planner 引入显式步数账本，在进入下一
子任务前检查剩余预算。

## H6：memory 后端的成本账本在重启后清零

**口径**：`CHECKPOINTER_BACKEND=memory` 时，成本账本只存在于进程内，
重启即清零；sqlite/mysql 后端会把 `cost_usage` 落库并在启动时恢复。

**为什么这样定**：memory 后端的语义就是"不持久化、进程即世界"，
单独为成本账本引入持久化会破坏该后端"零落盘"的承诺（测试、临时会话、
CI 都依赖它）。**代价**：memory 后端下，重启后的预算判断会低估已花费额，
可能多放行一些请求；这是"只少记、不多记"的方向性选择——不会因为账本
丢失而误熔断正常请求。**升级路径**：需要严格预算时使用 sqlite/mysql 后端，
或在外部计量系统里做累计。

## 质量门

发布与合并的硬性门槛（与 CI 一致）：

| 门 | 命令 | 通过标准 |
| --- | --- | --- |
| 静态检查 | `ruff check src tests examples packages` | All checks passed |
| 格式 | `ruff format --check src tests examples packages` | 无待格式化文件 |
| 类型 | `mypy` | Success: no issues found（strict；`files` 含基座与记忆包） |
| 测试 | `pytest` | 全绿，双包口径分支覆盖率 ≥85%（`--cov=agent_base --cov=agent_base_memory`） |
| 依赖审计 | `pip-audit --skip-editable` | 无未处理漏洞 |
| 容器 | `docker compose config --quiet` | exit 0 |

CI 矩阵：Python 3.10–3.13（lint 全跑，typecheck/test 覆盖两端版本），
test 使用 `mysql:8.0` 服务容器；另有 ubuntu/windows 的 test-core、
`pip-audit`、`memory-package`（构建记忆包 wheel 并断言
`agent_base.memory` 只暴露 `service` / `routes` 两个入口）与 `docker`
构建作业。质量门与运行时行为无关——它们只约束代码、依赖与打包。

## 架构演进记录

- **P1（已完成，0.3.0）**：memory 子系统拆为独立发行包 `agent-base-memory`，
  基座只留 `MemoryPort` / `MemoryServicePort` 契约与 `agent_base.memory`
  entry point 装配点，模块可来自独立包（`AGENT_MODULE_PACKAGES`）。
  决策与代价见 `docs/adr/003-split-memory-package.md`，落地清单见
  `docs/memory-package-split.md`。
- **P2（未做）**：多 worker 外置（成本账本、nonce、metrics 聚合），
  先完成 `docs/deployment.md` 的外置设计，再按需实现。

P2 与 H2/H3 的升级路径是同一件事的不同阶段；它不阻塞开源发布。
