# Changelog

本文件格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

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
