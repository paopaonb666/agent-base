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
  与全部索引写成幂等 DDL：表用 `CREATE TABLE IF NOT EXISTS`，索引在建表语句
  内联 `KEY`（MySQL 8 没有 `CREATE INDEX IF NOT EXISTS`）。老库上 `doc_chunks`
  已存在但缺 `offsets_json` 时，迁移在 `inspect` 确认后补列；offline（`--sql`）
  模式跳过该探测。

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

`core/registry.py` 增加多包加载：`load_modules(names, *,
packages)`（旧的 `prefix` 参数保留），按顺序在 `AGENT_MODULE_PACKAGES`
（逗号分隔，默认 `agent_base.modules`）列出的命名空间里解析模块名。
同一个名字命中多个包时**靠前的包优先**（命中即返回，与 Python 的模块
解析直觉一致）；找不到则报 `RegistryError` 并列出已搜索的包。
包名必须是点分小写标识符，空列表在启动时即报错。业务方因此可以发布
`my-company-agents`，在其中暴露 `my_company_agents.chat` 等模块。

## 7. 破坏性变更（0.3.0）

- `agent_base.memory.*` 导入路径移除，改为 `agent_base_memory.*`；
- 记忆子系统改为可选发行包，`/v1/memory` 路由由插件注册；
- `AgentRuntime.thread_index` / `plan_store` 的类型改为 `CoreStore`
  （方法签名不变，鸭子类型调用方不受影响）；
- docker 镜像与 compose 默认安装记忆包，行为不变。

## 8. 验收结果（实测）

| # | 检查 | 命令 | 结果 |
| --- | --- | --- | --- |
| 1 | 静态检查 | `ruff check src tests examples packages` | All checks passed |
| 2 | 格式 | `ruff format --check src tests examples packages` | 156 files already formatted |
| 3 | 类型 | `mypy`（files=src + packages/agent-base-memory/src） | Success: no issues found in 104 source files |
| 4 | 测试与覆盖率 | `pytest`（`--cov=agent_base --cov=agent_base_memory`） | 548 passed / 1 skipped / 0 failed；分支覆盖率 87.46%（门槛 85%） |
| 5 | 依赖审计 | `pip-audit --skip-editable` | No known vulnerabilities found |
| 6 | 记忆包 wheel | `python -m build --wheel packages/agent-base-memory` | 构建成功，entry point 为 service + routes，含 LICENSE/NOTICE |
| 7 | 记忆链 offline | `python -m alembic -c packages/agent-base-memory/alembic.ini upgrade head --sql` | 输出六表幂等 DDL 并按 `alembic_version_memory=0001` 收尾 |
| 8 | compose 语法 | `docker compose config --quiet` | exit 0 |

只装基座的两种语义由 `tests/test_memory_plugin.py` 固定下来：

- `MEMORY_ENABLED=true` + 无可发现实现 → `create_runtime` 抛
  `MemoryPluginError`，文案含 `pip install 'agent-base[memory]'`；
- `MEMORY_ENABLED=false` + 无可发现实现 → 正常装配，`runtime.memory is None`；
- 同文件还有静态 import 扫描：`src/agent_base` 下任何 `import agent_base_memory`
  都会让它失败（依赖方向靠测试守住，而不是靠约定）。

未在本机验证：真实 MySQL 上的老库升级（需要 0.2.0 库样本 + 服务容器）与
`docker build`（本机 Docker daemon 未运行）。这两项由 CI 的 test / docker
作业覆盖，发布前应在有 Docker 的环境补跑一次。

## 9. 落地清单与已知限制

落地：

- 新包 `packages/agent-base-memory/`：`pyproject.toml`（entry points、
  `agent-base>=0.3,<0.4`、LICENSE/NOTICE）、`src/agent_base_memory/`
  （原 `agent_base.memory` 全量迁入改名）、独立 `alembic.ini` + `alembic/`、
  安装/迁移说明 README；
- 基座新增 `core/storage/`（thread_index / plan_snapshots 三后端）、
  `core/memory_plugin.py`（entry point 发现）、`core/memory_scope.py`、
  `ChatInvokerPort`；`bootstrap` 用插件装配记忆、`create_app` 按插件挂路由；
- `registry` 多包加载 + `AGENT_MODULE_PACKAGES`（配置校验、first-wins）；
- 迁移：基座三条记忆迁移改 no-op，记忆包独立链；
- 打包与部署：根 `pyproject` 的 `[memory]` extra、Dockerfile/compose 双链迁移、
  CI 的 packages 路径与 memory-package 作业、`.env.example` 说明。

已知限制（不影响本次拆分结论，记录在案）：

- **wheel 不含迁移资产**：`alembic/` 与 `alembic.ini` 随仓库与容器镜像分发
  （与基座一致）。从 wheel 安装的部署要跑记忆链，需要从仓库或镜像取这两个
  路径；后续若上 PyPI，再把迁移作为包数据并提供短命令。
- **基座链 offline 模式仍不可用**：`alembic/versions/0001` 在 `--sql` 的
  MockConnection 上调用 `sa.inspect` 会抛 `NoInspectionAvailable`。这是拆分
  前就存在的问题，与记忆包无关，本次未修。记忆链的 offline 模式已处理。
- **PyMySQL 上限**：钉 `pymysql>=1.1,<1.2.1`。PyMySQL 1.2.1 起删除了
  `escape_bytes_prefixed`（1.2.3 换成字符串占位符，PyMySQL#1274），而 aiomysql 0.3.2
  在导入处就要这个名字：1.2.1/1.2.2 直接 ImportError，1.2.3 在 BLOB 参数上抛
  `TypeError`。等 aiomysql 发修复版后放宽。
- `agent-base-memory` 尚未发布到 PyPI，`[memory]` extra 目前只对"从仓库/
  镜像安装"有意义；发布顺序必须先基座 0.3.0、后记忆包 0.1.0。
