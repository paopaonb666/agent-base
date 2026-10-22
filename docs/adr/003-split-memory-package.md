# ADR-003：记忆子系统拆为独立发行包

- 状态：已采纳（0.3.0）
- 日期：2026-10-22
- 相关：ADR-002 模块契约、`docs/memory-package-split.md`、`docs/deployment.md`

## 背景

0.2.0 的 `agent-base` 同时装着三样东西：LangGraph 运行时与入口层、自研
记忆子系统（store / embeddings / retrieval / pipeline / 知识库）、以及四个
样板模块。契约层已经依赖倒置（模块只认 `MemoryPort`），但物理上仍是一个
wheel：不用长期记忆的部署也必须安装全部记忆依赖并接受它的升级节奏；
记忆子系统想独立修 bug、独立发版，就得跟着基座一起走。

P0 把项目做到"可开源发布"，P1 要回答的是：**能不能只装基座？记忆能不能
单独升级？业务模块能不能来自别的发行包？**

## 决策

1. **记忆子系统成为独立发行包 `agent-base-memory`**，导入名
   `agent_base_memory`，源码在 `packages/agent-base-memory/`，有独立
   版本（0.1.0）与依赖约束 `agent-base>=0.3,<0.4`。
2. **基座源码不 import 记忆包**。装配点是 `agent_base.memory` entry point
   组的两个入口：`service`（`plugin:build`，返回服务与工具）与
   `routes`（`api:routers`，挂载 `/v1/memory`）。契约仍在
   `agent_base.core.contracts`（`MemoryPort` / `MemoryServicePort`）。
3. **`MEMORY_ENABLED` 默认仍为 true：启用即承诺**。装了基座却没装记忆包时，
   启动立刻失败并给出安装命令；显式关闭则零发现、零导入，行为与没有该包
   完全一致。不静默降级——"以为有记忆、其实没有"比启动失败更难排查。
4. **会话索引与计划快照留在基座**（`core/storage/`）。`MEMORY_ENABLED=false`
   时 HTTP 层仍需要线程属主校验、会话列表与计划快照；它们不是记忆实现。
5. **迁移链（Alembic）按所有者拆开**：基座 `0004/0005/0007` 保留 revision
   编号但改为 no-op 空壳（旧库不会 "Can't locate revision"）；记忆包用独立链
   与 `version_table=alembic_version_memory`，首迁移是幂等
   `CREATE TABLE IF NOT EXISTS`，老库无需 `stamp`。
6. **业务模块可来自独立包**：`registry.load_modules(names, *, packages)`
   按 `AGENT_MODULE_PACKAGES`（默认 `agent_base.modules`）顺序查找，
   同名以靠前的包为准。

## 被否方案

- **两个 wheel 共享 `agent_base/` 命名空间**：安装/卸载/升级会互相覆盖文件，
  且 pip 无法表达"共同拥有一个目录"的依赖关系。
- **基座运行时 import 记忆实现、用 try/except 降级**：依赖方向反了（基座
  编译期就需要记忆包），也违背"启用即承诺"——缺包会变成静默的 503。
- **直接把 `0004/0005/0007` 迁移文件搬到新包**：老库
  `alembic_version=0009` 在 `upgrade head` 时会因 revision 缺失而失败。
- **拆成 microservice / 独立进程**：P2 的多 worker 外置是另一个问题；
  当前目标是包边界，不是部署边界。

## 后果

正面：

- 只装基座即可跑对话、上传、线程与计划（`/v1/memory` 为 404）；
- 记忆子系统可独立发版、独立测试与审计，接口由端口固定；
- 业务模块可发布为独立发行包，基座导入图不变。

代价（明示）：

- sqlite 后端打开两个连接、MySQL 两个小池（记忆池 + 基座池）；
- 记忆包装配点需要处理"快档模型是 `ResilientLLM`、不是 `BaseChatModel`"，
  因此新增 `ChatInvokerPort` 端口而不是复用 `BaseChatModel`；
- 迁移资产随仓库与容器镜像分发，wheel 只含 Python 代码（与基座一致）：
  从 wheel 安装的部署需要从仓库/镜像取 `alembic/` 目录才能跑记忆链；
- 0.3.0 是破坏性变更：`agent_base.memory.*` 导入路径移除，
  `AgentRuntime.thread_index` / `plan_store` 的类型改为 `CoreStore`。

## 迁移与回滚

见 `docs/memory-package-split.md` §5：新库与老库都执行同一条"基座链 +
记忆链"命令；应用回滚到 0.2.x 不需要改表结构，记忆链 `downgrade base` 会
删表，只在显式执行时发生。混合部署（0.2 基座 + 0.3 记忆包）不受支持，
依赖约束 `agent-base>=0.3,<0.4` 已经封死。

## 验证

四道质量门（ruff / format / mypy / pytest 覆盖率）覆盖两个包，CI 增加
memory-package 作业：构建 wheel、安装、断言 entry point 注册成功；
另有两类回归测试守住边界：基座源码的静态 import 扫描、以及"缺包时
`MEMORY_ENABLED=true` 启动失败 / `false` 正常装配"。
