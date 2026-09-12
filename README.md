# agent-base

Agent 软件基座——所有 agent 模块扩展的统一起点。

**基座 = LangGraph 运行时（不自研）+ 薄封装层（模块注册 / 配置 / 装配）+ 扩展点预留。**

> **范围红线**：RAG、长期记忆、知识库、业务鉴权、业务存储等**不属于基座**——
> 它们未来以"模块"形式接入（ADR 见本地 `docs/adr/`，不入库）。基座只提供运行时、
> 注册规范、配置管理、可运行入口与扩展点。

## 当前状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 工程地基（脚手架 / 工具链 / CI 四道门 / ADR） | ✅ 已完成 |
| 1 | 核心装配层（config / contracts / registry / llm / bootstrap）+ chat 样板模块 + CLI 入口 | ✅ 已完成 |
| 2 | 可观测性（request_id 贯穿 / 结构化日志 / LangSmith·Langfuse 开关） | ✅ 已完成 |
| 3 | 工具扩展点（共享工具池 + 超时 + 异常归一）+ 对话状态（checkpointer） | ✅ 已完成 |
| 4 | FastAPI + SSE 服务入口 + supervisor 多 Agent 模板 | ✅ 已完成 |
| 5 | 收尾 + 文档闭环（五步接入实操 / coverage ≥85% / CircuitBreaker 归档） | ✅ 已完成 |

## 架构图

```
                         ┌──────────────────────────────────────────────┐
                         │                agent-base（基座）             │
                         │                                              │
  CLI / SSE 服务 ───────▶ │  entrypoints   core                extensions│
  (python -m / uvicorn)   │  ┌──────────┐  ┌────────────────┐  ┌───────┐ │
                          │  │ cli.py   │─▶│ bootstrap      │◀─│memory │ │ ◀─ 对话状态
                          │  │ server.py│  │ （装配中枢）    │  │collab │ │ ◀─ supervisor
                          │  └──────────┘  └──┬──┬──┬──┬────┘  │events │ │ ◀─ SSE 契约
                          │                   │  │  │  │       │metrics│ │ ◀─ 指标
                          │                   ▼  ▼  ▼  ▼       │observ.│ │ ◀─ request_id
                          │              config contracts     └───────┘ │
                          │              registry llm tools            │
                          │                                              │
                          │  modules/（业务方按五步接入，不改基座代码）   │
                          │  ┌──────┐ ┌──────┐ ┌──────┐                 │
                          │  │ chat │ │writer│ │hello │  … 未来模块      │
                          │  └──────┘ └──────┘ └──────┘                 │
                          └──────────────────────────────────────────────┘
                                        ▲ AgentModule 契约（ADR-002）
                                        │
                                  LangGraph 运行时（不自研，>=1.2.6）
```

数据流：入口（CLI / HTTP SSE）→ `create_runtime` 按 `AGENT_MODULES` 契约装配 →
模块图挂在运行时上执行；扩展点（状态 / 协作 / 事件 / 指标 / 可观测）由基座统一供给。

## 环境要求

- Python ≥ 3.10（开发推荐 3.12）
- 可选：Node.js + pnpm（仅当使用配套前端 agent-base-ui 时）

## 依赖清单

运行依赖（`pyproject.toml` [project.dependencies]）：

| 依赖 | 用途 | 版本地板及理由 |
| --- | --- | --- |
| `langgraph` | 运行时（delegated，不自研） | `>=1.2.6`：修复 CVE-2025-68664（反序列化注入）、CVE-2025-67644（SQLite checkpointer SQL 注入）、CVE-2026-34070（路径遍历），见 ADR-001 |
| `langchain-core` / `langchain-openai` | 消息图元 + OpenAI 兼容模型客户端 | `>=1.6`：langchain-openai 1.x 的下限 |
| `pydantic-settings` | 配置即校验（fail-fast） | `>=2.2`：NoDecode 支持逗号分隔的列表类字段（`AGENT_MODULES`） |
| `langgraph-checkpoint-sqlite` + `aiosqlite` | sqlite 对话状态后端 | `>=2.0`：AsyncSqliteSaver 服务 astream |
| `langgraph-supervisor` | supervisor 多 Agent 模板 | `>=0.0.31` |
| `fastapi` / `uvicorn` | HTTP + SSE 服务入口 | `>=0.115` / `>=0.30` |
| `pypdf` | 工具库 parsing：PDF 文本提取（M4 前置） | `>=5.0`：纯 Python 且轻量，PDF 是文档读取的主格式，进主依赖开箱即用 |
| `httpx` / `tzdata` | 工具库基础：Tavily 搜索引擎 HTTP 客户端 / Windows 上的时区数据库 | `>=0.27`（原 dev 依赖转正）/ `>=2024.1`（POSIX 自带） |
| `python-multipart` | 文件上传端点的 multipart/form 解析 | `>=0.0.9`：FastAPI 官方要求 |

可选 extras：

- `[search]`：`ddgs`——DuckDuckGo 搜索引擎。未安装且未配 Tavily key 时
  启用 `web_search` 会在启动时报错。
- `[doc]`：`python-docx`——DOCX 解析。未安装时 tools/parsing 的 docx
  格式不注册（`pip install -e ".[doc]"` 安装），其余格式不受影响。

开发依赖（`[dev]`）：pytest / pytest-asyncio / pytest-cov / ruff / mypy / pip-audit / httpx。
安全审计由 CI 常驻：`pip-audit --skip-editable`。
数据库迁移（`mysql` 后端）：Alembic + SQLAlchemy + PyMySQL（均为 dev 依赖）。
对话状态表由 [`alembic/`](alembic/) 版本化管理，初始迁移复用 langgraph 内置迁移。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows（Linux/macOS: source .venv/bin/activate）
pip install -e ".[dev,doc]"    # [doc] 提供 DOCX 解析；缺它时该格式优雅降级
cp .env.example .env           # 填入 LLM_API_KEY（阶段 1 起由 config.py 消费）
pytest                         # 覆盖率门（≥85%）依赖 MySQL 集成测试（见下）
```

> **测试与覆盖率门**：memory/mysql57 的集成测试在探测到可用的 MySQL 时
> 自动启用——连接参数经 `MYSQL_HOST / MYSQL_PORT / MYSQL_USER /
> MYSQL_PASSWORD` 环境变量传入（默认 `root` / 空密码连 `127.0.0.1:3306`）。
> CI 在 test job 里挂了一个 MySQL 5.7 服务容器。本地没有 MySQL 时这些
> 测试会被跳过，总覆盖率会跌破 85% 门槛（`pytest` 因此失败）——这是
> 预期行为，起一个 MySQL 或接受本地红灯即可。

### 配置说明（.env）

`.env.example` 是配置契约（由 `core/config.py` 消费），关键项：

| 变量 | 说明 |
| --- | --- |
| `LLM_API_KEY` | 必填。OpenAI 兼容模型的 API key（DeepSeek / 智谱 / 自定义均走同一客户端）。production 下缺失将拒绝启动 |
| `LLM_PROVIDER` | `deepseek` \| `zhipu` \| `openai-compatible`（仅做拼写校验，不改变客户端） |
| `LLM_BASE_URL` / `LLM_MODEL` | 模型服务地址与模型名，切换 provider 只需改这两项 |
| `AGENT_MODULES` | 逗号分隔的模块清单，顺序即装配顺序（默认 `chat,writer`） |
| `CORS_ORIGINS` | 允许跨域调用 SSE 端点的来源（默认 `http://localhost:3000`，即 agent-base-ui） |
| `CHECKPOINTER_BACKEND` | `sqlite`（默认，重启可恢复对话）\| `memory`（零依赖，不持久化）\| `mysql`（MySQL 持久化，配 `CHECKPOINTER_MYSQL_*`） |
| `TOOL_TIMEOUT_SECONDS` | 工具单次执行超时（默认 30s） |
| `TOOL_CALL_LOG_ENABLED` | 工具调用审计（M5）：每条调用含参数/结果/耗时落库，UI 渲染调用面板 |
| `DOC_PARSE_MAX_INPUT_BYTES` / `DOC_PARSE_MAX_OUTPUT_CHARS` | 上传文档解析限额（默认 10MB / 5 万字符） |

> **附件格式**：文档（pdf/docx/txt/md）提取文本注入上下文；图片
>（png/jpg/webp/gif）按 magic bytes 校验后整字节入库，以多模态
> content blocks 直达 vision 模型（需搭配视觉模型使用）。
| `TOOLKIT_ENABLED` | 装配进共享池的基座内置工具清单（默认 `current_time,calculator,json_query`）；`web_search` 需配 `SEARCH_*`，`python_repl` 为 exec 级默认关 |
| `LOG_JSON` | `true` 输出结构化 JSON 日志（生产建议开启） |

运行一个对话（阶段 1 起可用）：

```bash
python -m agent_base --message "你好"      # 单轮对话（需 .env 已配 LLM_API_KEY）
python -m agent_base --module chat         # 交互式多轮对话
python -m agent_base --module chat --thread-id <id>   # 恢复之前的会话（阶段 3）
python -m agent_base --module supervisor   # 多 Agent 会话（阶段 4，编排在 AGENT_MODULES 里注册的所有模块）
python -m agent_base --version
```

运行 SSE 服务（阶段 4 起可用）：

```bash
uvicorn agent_base.entrypoints.server:app --reload

curl -N -X POST http://localhost:8000/v1/agents/chat/invoke \
  -H "Content-Type: application/json" -d '{"message": "你好"}'
# SSE 事件：ping → step(running/completed) → delta... → done(thread_id)
# 其他端点：GET /health（分项健康，degraded 不崩溃）；GET /metrics（Prometheus 文本，路由模板 label）
```

## 数据库迁移（mysql 后端）

对话状态（checkpointer）表由 **Alembic** 版本化管理，初迁移复用 langgraph 内置迁移：

```bash
# 全新库：应用全部迁移（建出 checkpoint_* 表）
python -m alembic upgrade head

# 已由 langgraph setup() 建过表的库：只对齐版本、不重复执行 DDL
python -m alembic stamp head

# 连接参数优先级：AGENT_BASE_DB_* 环境变量 > .env 的 CHECKPOINTER_MYSQL_*
# 例：AGENT_BASE_DB_NAME=mydb AGENT_BASE_DB_PASSWORD=*** python -m alembic upgrade head
```

迁移目录：[`alembic/`](alembic/)（`alembic.ini`、`env.py`、`versions/`）。
说明：迁移通过读取 `checkpoint_migrations` 当前最大版本判断进度，与
`AIOMySQLSaver.setup()` 的幂等逻辑一致，二者共存且不冲突。

## 配合前端 agent-base-ui

配套 Web 界面在独立仓库 `agent-base-ui`（同级目录 / `e:\ai_study\agent-base-ui`）：

```bash
# 1. 启动本后端（默认 8000 端口）
uvicorn agent_base.entrypoints.server:app --reload

# 2. 另开终端启动前端（见 agent-base-ui/README.md 的详细说明）
cd ../agent-base-ui
pnpm install
pnpm dev        # http://localhost:3000
```

前端设置页填入 `http://localhost:8000` 与目标模块（chat / writer / supervisor）即可对话。
两者通过 SSE 契约解耦，前端不依赖后端具体实现。（完整体验：先配 `.env` 的 `LLM_API_KEY`。）

## 质量门（四道，本地与 CI 完全一致）

```bash
ruff check src tests && ruff format --check src tests   # 1. lint + format
mypy                                                     # 2. 类型检查（strict）
pytest                                                   # 3. 测试（内置 branch coverage ≥85% 门槛）
pip-audit --skip-editable                               # 4. 依赖安全审计
```

## 项目结构

```
agent-base/
├── pyproject.toml            # 依赖 + 工具链配置（ruff / mypy / pytest + coverage 门槛 ≥85%）
├── .env.example             # 配置契约（由 core/config.py 消费）
├── src/agent_base/
│   ├── core/                # config / contracts / registry / llm / bootstrap / tools（工具池）
│   ├── extensions/          # 扩展点：observability（request_id + 日志）· memory（checkpointer）
│   │                        #           · collab（supervisor）· events（SSE 契约）· metrics
│   ├── modules/chat/        # 样板模块（graph + module + tools，兼作接入模板）
│   ├── modules/writer/      # 第二样板模块（供 supervisor 编排演示）
│   ├── modules/hello/       # 五步手册的真实落地实例（tests/test_hello_module.py 验证）
│   └── entrypoints/
│       ├── cli.py           # CLI 入口（--module / --message / --thread-id / --version）
│       └── server.py        # FastAPI + SSE 服务入口（/invoke · /health · /metrics）
├── alembic/                 # 数据库迁移（alembic.ini + env.py + versions/）
├── tests/                   # config / registry / llm / chat / cli / smoke / observability
│                            # / memory / tools / events / server / collab / hello
│                            # / bootstrap / metrics
├── docs/                    # 本地文档（按用户策略不入库；克隆后不存在是预期行为）
│   ├── adr/                 # 架构决策记录
│   ├── future/              # 未来模块路线图存档（circuit-breaker → reliability 模块）
│   └── module-guide.md      # 五步接入手册 + 应用层规范（新模块接入不改基座）
└── .github/workflows/ci.yml # CI：lint → typecheck → test（含 coverage 门槛）→ audit
```

## 设计决策

以下文档保存在本地工作区 `docs/`（按用户策略不入库，克隆后不会存在）：

- ADR-001：LangGraph 作为基座运行时（`docs/adr/001-langgraph-as-runtime.md`）
- ADR-002：显式清单模块注册（`docs/adr/002-module-contract.md`）
- module-guide：五步接入新模块 + 应用层规范（`docs/module-guide.md`）
