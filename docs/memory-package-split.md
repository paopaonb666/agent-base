# Memory 子系统独立包拆分方案（P1）

> 状态：已实施（分支 `feat/split-memory-package`，基线 0.3.0）
> 相关：docs/adr/002-module-contract.md、docs/adr/003-split-memory-package.md、
> docs/deployment.md、CHANGELOG.md

## 1. 背景

0.2.0 把基座、记忆子系统与四个样板模块放在同一个发行包 `agent-base` 里：
契约层（`MemoryPort`/`AgentModule`）已经隔离，但物理上无法单独升级、单独
裁剪记忆子系统。P1 要把**记忆子系统拆成独立发行包**，基座只保留契约与
装配点；业务模块也应能来自独立发行包。

约束：

- 不改业务行为与 `MemoryPort` 契约语义（导入路径、迁移、发布流程可以变）；
- 四道质量门（ruff / format / mypy / pytest≥85% / pip-audit）保持全绿；
- 老数据库升级后数据不丢、不回滚失败；
- 不 push，提交用英文祈使句。

## 2. 目标布局

```
agent-base/                        发行包 agent-base（基座）
├── src/agent_base/
│   ├── core/contracts.py          MemoryPort + MemoryServicePort（唯一记忆契约）
│   ├── core/storage/              thread_index / plan_snapshots 基座存储
│   ├── core/memory_plugin.py      entry point 发现与装配
│   └── ...
├── alembic/versions/              基座迁移链（记忆表迁移留空壳）
└── packages/agent-base-memory/    发行包 agent-base-memory（记忆子系统）
    ├── src/agent_base_memory/     原 src/agent_base/memory 全量迁入并改名
    ├── alembic/                   独立迁移链（version_table=alembic_version_memory）
    └── pyproject.toml
```

导入名从 `agent_base.memory.*` 改为 `agent_base_memory.*`（机械改名，
包内相对语义不变）。不采用"两个 wheel 同时写 `agent_base/` 目录"的
包共享 hack：命名空间共享会让两个发行包的卸载/升级互相破坏文件。

## 3. 装配契约

基座**不得**在源码里 import `agent_base_memory`。装配点：

- `core/contracts.py`
  - `MemoryPort`：保持原样（`search` / `compose_context` / `capture_turn` /
    `aclose`），业务模块只依赖它；
  - `MemoryServicePort`：入口层需要的完整面（`recall_context` /
    `purge_thread` / `ingest_document` / `revoke_document` /
    `search_knowledge` / `list_file_chunks` / `chunk_counts_by_file` /
    `health_probe` / `capture_ready`）。
- `core/memory_plugin.py`：按 entry point 组 `agent_base.memory` 发现实现：
  - `service = agent_base_memory.plugin:build` → `async build(*, settings, llm,
    fast_llm) -> MemoryPlugin | None`（返回服务实例、工具列表、HTTP 路由）；
  - `routes = agent_base_memory.api:routers` → 记忆管理路由（`/v1/memory`）。
- `create_runtime`：`MEMORY_ENABLED=true` 时**必须**发现实现，否则启动即报
  错（"启用即承诺"，与 `[mcp]` extras 同一语义）；`false` 时零导入、零
  解析，行为与没有该包时完全一致。
- `MEMORY_ENABLED` 默认值保持 `true`（不改变默认部署行为）：完整安装
  （`pip install agent-base[memory]`、仓库 dev 安装、Docker 镜像）都带记忆包；
  只装基座又开启记忆时会得到明确的安装提示，而不是静默降级。

记忆工具（`memory_search` 等）由插件返回，基座只把它们并入工具池。
`set_memory_scope` 的 ContextVar 下沉到基座 `core/memory_scope.py`，
记忆包反向导入——作用域传递是入口层语义，不属于记忆实现。

## 4. 基座 core storage

拆分的真正难点不是代码目录，而是 `thread_index` / `plan_snapshots`：
HTTP 层在 `MEMORY_ENABLED=false` 时仍然要工作（线程属主校验、会话列表、
计划快照）。这两个数据面原本寄居在 `MemoryStore` 里。

新增 `src/agent_base/core/storage/`：

- `ThreadIndex` 模型、`ThreadIndexRepo` / `PlanRepo` / `CoreStore` 协议、
  `thread_index` / `plan_snapshots` 的 sqlite/MySQL DDL（单一来源仍是 DDL 模块）；
- 三后端实现（memory / sqlite / mysql）与 `build_core_store(settings)`，
  后端选择规则与原记忆存储一致（跟随 `CHECKPOINTER_BACKEND`）；
- `AgentRuntime.thread_index` / `plan_store` 类型改为 `CoreStore`，
  生命周期由运行时独立关闭；记忆服务只负责自己的表。

代价：sqlite 后端会打开两个连接、MySQL 两个小池（记忆池 5 + 基座池）。
这是"基座不依赖记忆实现"的必要成本，文档中明示。

## 5. 迁移拆分与兼容

### 5.1 迁移链

- 基座链保持 revision id `0001..0009` 不变（老库 `alembic_version=0009`
  不会出现 "Can't locate revision"）。其中 `0004` / `0005` / `0007`
  三条记忆表迁移改为 **no-op 空壳**，docstring 说明所有权已移交。
- 记忆包新增独立链：`alembic.ini` + `alembic/env.py` +
  `versions/0001_memory_tables.py`，`version_table=alembic_version_memory`。
  初始迁移把六张表（`memories` / `memory_blocks` / `session_summaries` /
  `doc_chunks`（含 `offsets_json`）/ `memory_ops` / `memory_versions`）
  与全部索引定义为 `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`
  的幂等 DDL。

### 5.2 三类库的升级动作

| 库状态 | 升级命令 | 结果 |
| --- | --- | --- |
| 全新库 | `alembic upgrade head`（基座）+ `alembic -c packages/agent-base-memory/alembic.ini upgrade head` | 基座建 checkpointer/工具审计/上传/线程索引/计划表，记忆链建六张记忆表 |
| 老库（0.2.x，已含记忆表，`alembic_version=0009`） | 同上 | 基座 no-op 空壳不动数据；记忆链的幂等 DDL 命中已存在的表，只写入 `alembic_version_memory=0001` |
| 只装基座（记忆关闭） | 仅 `alembic upgrade head` | 不创建任何记忆表 |

因此**不需要 `alembic stamp`**：幂等初始迁移让老库和新库走同一条命令。
这是选择 no-op 空壳 + 幂等首迁移、而不是"把迁移文件直接搬到新包"的原因：
直接搬走会让老库在 `upgrade head` 时报 revision 缺失。

### 5.3 回滚

- 应用回滚：`pip install agent-base==0.2.x` 并恢复旧部署即可；记忆表结构
  0.2.0 与 0.3.0 一致（只改了 DDL 的归属，没有列变更）。
- 数据库回滚：记忆链 `downgrade base` 会 `DROP` 六张记忆表（破坏性，
  只在显式执行时发生）；基座空壳迁移的 `downgrade` 同样 no-op，不会
  误删数据。
- 混合部署（旧基座 + 新记忆包）不支持：0.3.0 的记忆包依赖新的
  `MemoryServicePort`，版本约束写死 `agent-base>=0.3,<0.4`。

## 6. 业务模块来自独立包

`core/registry.py` 增加多前缀加载：`load_modules(names, *, prefixes)`，
按顺序在 `AGENT_MODULE_PACKAGES`（逗号分隔，默认 `agent_base.modules`）
列出的命名空间里解析模块名；同一个名字命中多个前缀属于配置错误（fail-fast），
不存在则报 `RegistryError` 并列出已搜索的前缀。业务方因此可以发布
`my-company-agents`，在其中暴露 `my_company_agents.chat` 等模块。

## 7. 破坏性变更（0.3.0）

- `agent_base.memory.*` 导入路径移除，改为 `agent_base_memory.*`；
- 记忆子系统改为可选发行包，`/v1/memory` 路由由插件注册；
- `AgentRuntime.thread_index` / `plan_store` 的类型改为 `CoreStore`
  （方法签名不变，鸭子类型调用方不受影响）；
- docker 镜像与 compose 默认安装记忆包，行为不变。

## 8. 验收清单

1. `pip install -e . -e packages/agent-base-memory` 后四道门全绿；
2. 只装基座（临时卸载记忆包）时：`MEMORY_ENABLED=false` 正常启动、
   `MEMORY_ENABLED=true` 启动即报可操作的错误；
3. 全新 sqlite 库：只跑基座链 → 无记忆表；跑记忆链 → 六表齐全；
4. 老库样本（0.2.0 建的库）：两条链 upgrade 均成功且行数不变；
5. `alembic -c packages/agent-base-memory/alembic.ini downgrade base` 后
   记忆表消失、基座表保留；
6. `docker compose config` 与镜像构建通过（构建受本机 Docker 状态限制）。
