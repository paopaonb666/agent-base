# Changelog

本文件格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [0.3.0] - 2026-10-22

记忆子系统拆为独立发行包。基座只保留契约与装配点；不改任何业务语义
（检索排序、形成管线、API 路径与 JSON 形状都不变），但导入路径与
依赖方式是破坏性变更。

### Added

- 独立包 `agent-base-memory`（源码 `packages/agent-base-memory/`，
  导入名 `agent_base_memory`，版本 0.1.0，依赖 `agent-base>=0.3,<0.4`）：
  记忆存储（sqlite / MySQL）、向量与 BM25 检索、形成管线、知识库工具，
  以及 `/v1/memory` 路由（`agent_base_memory.api:routers`）。
- 装配契约：`agent_base.memory` entry point 组，`service`
  （`plugin:build`，返回服务 + 工具）与 `routes`（`api:routers`）
  两个入口；基座由 `core/memory_plugin.py` 发现并装配，源码不 import 记忆包。
- 基座存储层 `agent_base.core.storage`：线程索引与计划快照的
  sqlite / MySQL / memory 三后端及 `CoreStore` 端口，随基座版本管理。
- 新端口：`MemoryServicePort`（服务装配类型）与 `ChatInvokerPort`
  （快档模型只要求 `ainvoke`，不再要求 `BaseChatModel`）。
- 模块包：`AGENT_MODULE_PACKAGES` 配置让模块来自独立发行包；
  registry 按前缀顺序查找，同名 first-wins。
- 可选依赖：`pip install "agent-base[memory]"`。
- CI：`memory-package` 作业构建记忆包 wheel 并断言两个 entry point；
  `tests/test_memory_plugin.py` 用静态 import 扫描与缺包启动用例守住边界。
- 文档：`docs/adr/003-split-memory-package.md` 与
  `docs/memory-package-split.md`（目标布局、迁移兼容、验收结果、已知限制）。

### Changed

- `MEMORY_ENABLED=true`（默认）现在是承诺：没装记忆包即启动失败，
  错误信息给出安装命令；显式设 `false` 时零发现、零导入，`/v1/memory` 返回 404。
- 版本 0.3.0；mypy 的 `files`、ruff 的路径与 pytest 覆盖率口径都覆盖两个包。
- 容器与 compose：启动命令依次跑基座链与记忆链；镜像携带
  `/app/alembic` 与 `/app/memory-alembic` 两套迁移资产。
- `AgentRuntime.thread_index` / `plan_store` 的类型改为 `CoreStore`。
- `.env.example`、README、部署/模块/设计文档同步双包布局。

### Removed

- `src/agent_base/memory/` 整棵子树与 `agent_base.memory.*` 导入路径
  （机械改名为 `agent_base_memory.*`）；`entrypoints/routes/memory.py` 迁入记忆包。
- 基座 `alembic/versions/0004/0005/0007` 的建表逻辑改为空壳（revision 链保留，
  老库仍可 `upgrade head`）。

### Fixed

- MySQL 集成（`test_memory_store.py::test_mysql_store_semantics`）：PyMySQL 1.2.1 起
  删除了 `escape_bytes_prefixed`（PyMySQL#1274），1.2.3 又把它换成字符串占位符，
  而 aiomysql 0.3.2 在模块导入处 import 该名字——1.2.1/1.2.2 报 ImportError，
  1.2.3 让任何 BLOB 参数（记忆向量）抛 `TypeError: 'str' object is not callable`。
  现钉 `pymysql>=1.1,<1.2.1`。
- `memory_versions.seq` 缺独立索引，MySQL 8 建表报
  `1075: there can be only one auto column and it must be defined as a key`；
  建表与迁移各补 `KEY idx_versions_seq (seq)`。

### 升级 0.2 → 0.3

```bash
pip install -e ".[memory]"        # 或 pip install -e packages/agent-base-memory
python -m alembic upgrade head                                  # 基座链（不变）
python -m alembic -c packages/agent-base-memory/alembic.ini upgrade head
```

记忆链首迁移幂等（`CREATE TABLE IF NOT EXISTS`），0.2.0 的老库无需 `stamp`，
表已在、版本写入 `alembic_version_memory`。容器内路径换成 `memory-alembic/alembic.ini`。

回滚：记忆链 `downgrade base` 会删除六张记忆表并保留基座表；或卸载记忆包并设
`MEMORY_ENABLED=false`。应用回滚到 0.2.x 不需要改表结构。
## [0.2.0] - 2026-10-22

首个可公开 fork/star 的 P0 版本。仅工程、文档与打包改动，运行行为不变。

### Added

- **许可证**：`LICENSE`（Apache-2.0 官方全文）与 `NOTICE`（版权、项目 URL 与声明）。
- **公开文档**：`docs/adr/001-langgraph-as-runtime.md`（运行时选型、候选对比、
  CVE 版本地板）、`docs/adr/002-module-contract.md`（显式清单 + fail-fast 契约）、
  `docs/module-guide.md`（五步接入 + 检查清单）、`docs/deployment.md`
  （三种部署形态、环境变量、健康/指标、单进程口径、迁移、安全、容器）、
  `docs/design-review.md`（已知边界 H2–H6 与质量门）。
- **社区文件**：`CONTRIBUTING.md`、`SECURITY.md` 与本变更记录。
- **容器**：多阶段 `Dockerfile`（python:3.12-slim、非 root uid 10001、
  `HEALTHCHECK /health`、复制 alembic 资产）、`docker-compose.yml`
  （app + MySQL 8.0、utf8mb4、健康检查、命名卷、启动前迁移）与 `.dockerignore`。
- **打包**：`[migrate]` extra（Alembic / SQLAlchemy / PyMySQL，部署迁移用）。
- **CI**：新增 docker 构建 job；pip 缓存；MySQL 服务容器升级到 8.0。

### Changed

- `pyproject.toml`：版本 0.2.0；补齐 `license`、`authors`、`keywords`、
  `classifiers`（含 Apache-2.0 与 `Typing :: Typed`）与 `[project.urls]`。
- `docs/` 正式入库：`.gitignore` 删除 `docs/` 忽略行与"docs 不得入库"策略注释；
  内部计划文档继续由 `docs/plans/` 规则排除。
- README：新增 CI / License / Python 徽章与 `docker compose` 快速开始；
  范围红线改为 ADR 链接；项目结构树补全容器、许可证与社区文件；
  设计决策节改为仓库内链接；新增「许可证」「贡献」两节；
  修复配置表在「附件格式」引用块处断裂（三行回到表格）；
  MySQL 版本说明 5.7 → 8.0。
- CI：`test` job 的 MySQL 服务容器从 5.7 升到 8.0（与
  `langgraph-checkpoint-mysql` 的 `>=8.0.19` 要求对齐），乱码注释改为英文，
  `setup-python` 启用 pip 缓存。

### Fixed

- `modules/planner/graph.py`：`graph.add_node(name, make(ctx))` 改为
  `graph.add_node(name, cast(Any, make(ctx)))` 并删除 `# type: ignore`，
  消除 mypy 在上游 langgraph 版本间的 `arg-type` / `unused-ignore` 抖动。

## [0.1.0] - 2026-09-20

初始骨架：LangGraph 运行时 + 显式模块注册 + 嵌套配置 + CLI/HTTP 入口，
记忆、成本治理与 SSE 事件契约的首个可用版本。

### Added

- 核心装配层（config / contracts / registry / llm / bootstrap / graphs）与
  `chat` / `writer` / `hello` 样板模块。
- 长期记忆（混合检索、形成管线、版本审计、文档知识库）与 `MemoryPort` 契约。
- 成本计量、日/月预算熔断、`/metrics` 与 `/health`。
- 共享工具池（超时、审计、重名快速失败）、MCP 工具来源。
- FastAPI + SSE 服务入口、supervisor 多 Agent 模板、planner 任务模式。
