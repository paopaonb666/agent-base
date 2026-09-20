# agent-base

**LangGraph 之上的生产治理基座。** 编排交给 LangGraph（不自研）；基座补齐
"从 Demo 到生产"缺的三件设施——**长期记忆、成本治理、事件可观测**。
业务以模块接入，五步上手，不改基座代码。

## 为什么不是又一个 LangGraph 封装

LangGraph 解决"怎么把 Agent 编排起来"，本仓库解决"编排起来之后怎么敢上生产"。
下表每一行都对应仓库内已实现、有测试的代码，无一是路线图：

| 生产关切 | 常见现状（框架自理） | agent-base 内建 | 详见 |
| --- | --- | --- | --- |
| 长期记忆 | 记忆选型即锁定（Mem0/Zep/Letta 三种架构互不兼容） | 混合检索 + 形成管线 + 版本审计；`MemoryPort` 协议依赖倒置，模块不绑实现 | [记忆系统（M6）](#记忆系统m6) |
| 成本治理 | usage 散落各处，超支靠月底账单发现 | 全部 LLM 调用计量（含流式），日/月预算熔断（429），`/metrics`·`/health` 暴露 | [成本治理（T4）](#成本治理t4) |
| 事件可观测 | 黑箱运行，出错无从追踪 | SSE 事件契约（step / delta / plan / tool），request_id 贯穿 | [事件契约（SSE）](#事件契约sse) |
| 模块接入 | 业务与框架互相渗透 | `AGENT_MODULES` 显式清单 + fail-fast 校验，五步接入手册 | 见下文「项目结构」 |
| 工具治理 | 裸函数直连模型 | 共享工具池：超时包装 + 全量审计 + 重名快速失败 | 见下文「配置说明」 |

工程纪律是上述主张的前提：四道质量门（lint / mypy strict / pytest 覆盖率 ≥85% /
pip-audit）本地与 CI 完全一致，依赖地板锁定（langgraph ≥1.2.6，修复三个 CVE）。

> **范围红线**：业务鉴权、业务存储等**不属于基座**——它们以"模块"形式接入
> （ADR 见本地 `docs/adr/`，不入库）。基座只提供运行时、注册规范、配置管理、
> 可运行入口与扩展点。

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
                          │  │ server/  │  │ （装配中枢）    │  │collab │ │ ◀─ supervisor
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
# SSE 事件：ping → step(running/completed) → delta... → done(thread_id)，完整契约见下文「事件契约（SSE）」
# 复杂任务加 mode："plan"（强制）/ "auto"（启发式），见「任务模式」节
# 其他端点：GET /health（分项健康，degraded 不崩溃）；GET /metrics（Prometheus 文本，路由模板 label）
```

## 事件契约（SSE）

SSE 是前后端之间唯一的实时通道，也是解耦面：前端（agent-base-ui）只依赖
事件格式，不依赖后端实现。契约定义在 `extensions/events.py`——一组封闭的
Pydantic 模型，线上格式经过校验、可版本化，而不是临时字符串：

```
event: delta
data: {"type":"delta","content":"he"}
```

一次 invoke 的典型时序：`ping`（心跳保活）→ `step`（节点 running/completed）
→ `delta`...（增量 token，穿插 `tool_call` 与 `plan`）→ `done(thread_id)`；
失败以 `error` 终止。全部事件类型：

| type | 何时发出 | 载荷要点 |
| --- | --- | --- |
| `ping` | 空闲心跳，防止中间环节断流 | 无 |
| `step` | 图节点开始 / 结束 | `name`、`status`（running/completed/error）、`detail` |
| `delta` | 流式回复的每段 token | `content` |
| `tool_call` | 每次工具执行前后各一条，按 `call_id` 配对成完整调用面板 | `name`、`phase`（start/end）、`args`、`result`、`status`（ok/timeout/error）、`duration_ms` |
| `plan` | planner 拆解 / 推进 / 重规划 / 完成 | `status`（created/progress/replanned/done）、`plan` **全量**任务清单（前端整表替换，无需增量合并） |
| `sources` | 契约预留，基座从不发出 | `sources[]`（title/url）——未来 RAG 模块发布引用而不必改编码器 |
| `done` | 终止性成功 | `thread_id`（恢复会话的句柄） |
| `error` | 终止性失败 | `message`（人类可读的可行动摘要） |

错误排查路径：`error` 事件只带摘要；完整链路已带 `request_id` 落服务端
结构化日志（贯穿入口 / 模块 / 节点），凭摘要定位到 request_id 即可在日志
还原全过程；工具调用明细另有 `TOOL_CALL_LOG_ENABLED` 全量落库。

## 记忆系统（M6）

`src/agent_base/memory/` 是一套自研的混合式记忆系统（调研 mem0 / Letta / Memobase /
Zep / LangMem 后的落地形态），存储跟随 `CHECKPOINTER_BACKEND`（memory/sqlite/mysql），
六张自管表（含 thread_index 会话索引）由 alembic 迁移（MySQL，0004–0008）与运行时自举（sqlite，DDL 单一来源 `memory/store/ddl.py`）幂等建表：

| 能力 | 说明 |
| --- | --- |
| 跨会话长期记忆 | mem0 式两阶段形成：后台 LLM 抽取候选事实 → 与既有记忆比对做 ADD/UPDATE/DELETE/NOOP 整合（去重/纠错/失效剔除），全量审计落 `memory_ops` |
| 常驻记忆块 | Letta 式 persona/human/自定义块，agent 用 `memory_update_block` 工具自我编辑，每轮注入上下文 |
| 用户画像 | Memobase 式结构化 JSON 画像，随对话分节合并演化，整体注入（不参与相似度召回） |
| 会话滚动摘要 | 消息数超阈值后后台合并更新，供短期压缩替换超预算旧历史 |
| 文档知识库 | 上传的 PDF/DOCX/TXT 自动切块 + 向量化入 `doc_chunks`（用户级知识资产，独立于会话存活），`knowledge_search` 工具检索 |
| 混合检索 | 向量余弦（默认硅基流动 BAAI/bge-m3，1024 维）+ BM25（中文二元组）+ 时间半衰期 + 显著度；embedding 不可用自动降级为关键词路径 |
| 上下文工程 | 每轮注入「画像 + 记忆块 + 摘要 + 相关记忆」注入块；基座级上下文引擎（`core/context.py` + `core/graphs.py`）对**所有模块**的模型输入做注入去重与 token 预算修剪（工具配对不拆散），checkpointer 全量历史保留（recall 语义） |
| Agent 工具 | `memory_search` / `memory_save` / `memory_update_block` / `knowledge_search` 进共享工具池（超时 + 审计自动生效） |

管理端点（均以 `X-User-Id` 头为作用域，缺省 `default`）：

```
POST   /v1/memory                 # 手工写入记忆（落库前尽力向量化）
GET    /v1/memory?q=&module=&kind=  # q 存在→混合检索；否则按更新时间浏览
PATCH  /v1/memory/{memory_id}     # 部分更新（改内容会重新向量化）
DELETE /v1/memory/{memory_id}
GET    /v1/memory/blocks          # 列出常驻记忆块（?module= 限定模块）
PUT    /v1/memory/blocks/{label}  # 手工写块（覆盖式，版本递增）
DELETE /v1/memory/blocks/{label}
GET    /v1/memory/profile         # 当前用户的结构化画像
GET    /v1/memory/audit           # 记忆操作审计（抽取/整合/画像/摘要/摄取…）
GET    /v1/memory/{id}/versions   # 内容版本史（create/update/delete/restore 快照 + 前后差分）
POST   /v1/memory/{id}/versions/{vid}/restore   # 恢复到指定版本（产生 restore 新版本）
```

每次内容变更（写入/更新/删除/恢复，含画像演化）都会写一条版本快照，
删除留墓碑（保留最后内容供审计）；检索的 BM25 索引按候选内容指纹做
进程内 LRU 缓存，内容一变键即变，无需失效钩子。

**文档预览与切片可视化（M7）**：三个只读端点支撑前端"点开附件看预览与
切片方式"——`GET …/files/{id}/preview`（元信息 + 提取正文，与注入给模型
的内容同源）、`GET …/files/{id}/chunks`（切片列表：序号/正文/原文区间
offsets/是否已向量化；段落打包块是多区间，旧数据 null 前端降级）、
`GET …/files/{id}/raw`（原始字节：图片按存储 mime 直出，文档 attachment
下载）。切片边界来自 `chunk_text` 返回的 ChunkSpan（各段在原文中的字符
区间），落库于 `doc_chunks.offsets_json`。

**独立知识库页面（M8）**：`GET /v1/knowledge/files`——用户级文件列表
（含每文件切片数，`X-User-Id` 作用域），支撑前端 `/knowledge` 路由的
"文档知识库 / 长期记忆 / 用户画像"三页签可视化（前端仓库交付）。

**文档知识库生命周期**：上传时按段落感知切块（空行对齐，表格/列表
不被从中间劈开；单段超长退回固定窗口），重复摄取自动幂等；撤销传错
的文件用 `DELETE /v1/agents/{module}/files/{file_id}`——文件原始行与
全部知识分块级联删除，属主校验基于上传时记录的 user_id；更换
embedding 模型后 `python scripts/memory_backfill_embeddings.py` 同时
回填记忆与知识库分块的向量。

**身份与安全（S1/P0-5）**：身份解析只有一条路径——可插拔的 `AuthBackend`
（默认 `HmacHeaderAuth`：`X-User-Id` 头 + 可选 `X-User-Sig` HMAC 校验），
`create_app(auth_backend=...)` 可替换为网关层的真实身份体系。生产环境
（`ENV=production`）且记忆开启时**必须**配置 `MEMORY_AUTH_SECRET`，所有
用户作用域请求需附带 `X-User-Sig = HMAC-SHA256(X-User-Id, secret)`（缺省
用户也不例外），否则 401；签名生成：
`python scripts/memory_user_sig.py --user-id alice --secret <密钥>`。
浏览器不持有密钥——公网多用户部署请走服务端代理或真实身份体系。
已知局限：HMAC 签名只覆盖 user_id、无时间戳/nonce，截获的请求头可重放。

**会话线程作用域（S1）**：invoke 时把线程属主写入 `thread_index` 表
（存储跟随 `CHECKPOINTER_BACKEND`，alembic 0008；`MEMORY_ENABLED=false`
时索引仍然在位）。线程列表只返回**当前用户**的线程；历史 / 删除 /
工具审计端点做属主校验（404 掩蔽）。升级前的历史线程（无索引行）
视为不可访问；附件引用他人 file_id 在 invoke 即被拒绝（400）。
记忆删除为软/硬两级：`PATCH /v1/memory/{id}` 置 `status=archived` 是软删除
（保留数据、退出召回），`DELETE` 是硬删除；更换 embedding 模型/维度后运行
`python scripts/memory_backfill_embeddings.py` 为历史记忆回填向量。

**容量契约（H2）**：当前检索实现是"全量拉取候选 + 内存混合打分"，单用户
记忆候选上限 2000 条（知识库分块同量级）——达到上限会记 WARNING，更早的
记忆不参与召回而非静默截断。更大规模需要分片或引入候选下推索引
（sqlite-vec / FTS5 / 服务端向量库），见 `docs/design-review.md` H2。

**CLI 与 server 的能力差异**：CLI（`python -m agent_base`）面向单机自用，
不注入记忆上下文、不跑形成管线、不支持附件与多用户作用域；这些能力
仅在 HTTP 入口提供。指标为单进程口径，多 worker 部署需在采集侧聚合。

关键配置（完整清单见 `.env.example` 的「记忆系统」节）：`MEMORY_ENABLED` 总开关；
`MEMORY_EMBEDDING_API_KEY`（openai 兼容 `/embeddings`，留空降级关键词检索）；
`MEMORY_CAPTURE_ENABLED` / `MEMORY_CAPTURE_EVERY_TURNS`（形成频率）；
`MEMORY_CONTEXT_MAX_CHARS` / `MEMORY_CONTEXT_MAX_TOKENS`（注入与模型输入预算）。
指标：`/metrics` 的 `memory_operations_total{op,outcome}`；健康：`/health` 的
`memory` 组件（未启用时缺席，不算降级）。

## 任务模式（M9）

`AGENT_MODULES=chat,planner` 启用 planner 模块后，invoke 请求体可带
`mode` 字段做难度分流：

| mode | 行为 |
| --- | --- |
| `chat`（默认） | 请求模块自己的图，行为与此前完全一致 |
| `plan` | 强制走 planner 图（plan → execute → check →(replan)* → synthesize）；planner 未启用时 **503，不降级** |
| `auto` | 启发式：消息 >200 字或带附件 → plan；planner 缺席时静默降级 chat |

**图选择与线程命名空间解耦**：`mode` 只决定调哪张图，线程永远是
`{请求模块}:{thread_id}`——同一会话内 chat 轮与 plan 轮历史连续，
thread 属主、记忆 `agent_id`、附件绑定都不受影响。

planner 的执行语义：LLM 把目标拆解为 ≤`PLANNER_MAX_SUBTASKS` 条子任务，
逐条以**内联 ReAct 循环**执行（每子任务工具轮数上限
`PLANNER_SUBTASK_TOOL_ROUNDS`）；子任务失败（轮数耗尽 / 空输出门）触发
重规划（预算 `PLANNER_MAX_REPLANS`，保留已完成结果）；预算耗尽后做
best-effort 综合，如实标注未完成项。

SSE 事件新增 `plan`（全量任务清单，前端整表替换）：

```json
{"type": "plan", "status": "created",
 "plan": [{"id": 1, "goal": "查天气", "status": "pending"}],
 "detail": "拆解出 1 个子任务"}
```

`status` 取值 `created / progress / replanned / done`。配套只读端点
`GET /v1/agents/{module}/threads/{thread_id}/plan` 返回
`{tasks, cursor, replans}`（planner 未启用 503；从未跑过 plan 的线程
返回空默认值）。

已知限制：

- **chat 轮穿插会重置 plan 状态**（chat 图的 checkpoint 只含 messages
  通道）；跨轮 plan 延续若成为需求，升级路径是独立 plan 表（M10 候选）；
- 前端渲染 PlanEvent 属 agent-base-ui 仓库，本仓库只提供契约与端点；
- planner 重度使用的最坏图步数约 34，超出默认
  `AGENT_RECURSION_LIMIT=25`——按需在 .env 调高。

## 成本治理（T4）

自用全盘文档问答场景的 token 成本地基：查询分诊 + 模型分档 + 夜间批 +
预算熔断。目标形态下（Tier 0 吃六成、快档吃三成、主力只管裁决与门面），
月账单可压到个位数元（2026 年国产模型价格口径）。

**查询分诊（三档）**：

| 档 | 路径 | LLM | 何时用 |
| --- | --- | --- | --- |
| Tier 0 | `GET /v1/knowledge/search?q=` | 零 | 查找类问题（"那份保单的等待期"）——直接返回命中切片，字段与 M7 切片视图一致 |
| Tier 1 | invoke + `"profile": "fast"` | 快档 | 有切片为据的轻综合 |
| Tier 2 | invoke（默认） | 主力 | 深挖/综合/裁决 |

`profile` 与 `mode` 正交（mode 决定图，profile 决定图的模型）；`LLM_FAST_*`
未配置时传 `profile: "fast"` 显式 400，不静默降级。快档视图与主力共享
checkpointer——线程历史跨档连续。

**模型分档**：`LLM_FAST_*`（免费/低价模型，如智谱 glm 系列 flash——长期
免费但并发极低）吃"简单、大量、重复"的调用；`MEMORY_PIPELINE_PROFILE=split`
（默认）让形成管线的抽取/画像/摘要走快档（`ResilientLLM` 信号量限流 +
重试耗尽回退主力），整合裁决永远走主力（破坏性决策步不降级）；
`all_fast` 为显式覆盖。

**夜间批**：`MEMORY_CAPTURE_ENABLED=false` + `MEMORY_SUMMARY_ENABLED=false`
关掉每轮管线的 LLM 消耗，用 cron/计划任务每天跑一次：

```bash
python scripts/memory_nightly_capture.py            # force 旁路门控，批量抽取+整合
python scripts/kb_ingest.py --dir D:/my/docs --module chat --user-id alice --watch
```

`kb_ingest.py` 把本地目录（递归，pdf/docx/txt/md）批量摄取进知识库：
走既有上传端点（解析/切块/向量化/属主全复用），sha256 状态文件幂等
（重复运行 skip），`--watch` 轮询增量。注意服务端摄取是上传后的后台
任务，检索可见性滞后一两秒。

**计量与预算**：`COST_ENABLED=true` 后，`build_llm` 给全部客户端挂
usage 采集 callback（含流式 `stream_usage`），按 `COST_PRICES_JSON`
价目表计价（元/百万 tokens，手工维护）；累计落 `cost_usage` 表（跟随
checkpointer 后端；sqlite 为独立 sqlite 账本），进程重启后从账本恢复
当日/当月累计。`COST_DAILY_CNY` / `COST_MONTHLY_CNY` 任一超限：
`COST_ACTION=block` 时 invoke 返回 429（只挡新的 LLM 消耗，检索/历史/
记忆管理端点不受影响），`warn` 只告警放行；达阈值 80% 时告警一次。
可观测：`/metrics` 的 `llm_tokens_total{model,profile,kind}` 与
`llm_cost_cny_total`；`/health` 的 `cost` 组件（`ok|warn|blocked`，
未启用时缺席不算降级）。

**已知局限**：计量与预算为单进程口径（多 worker 需采集侧聚合，同
/metrics）；价目手工维护；mysql 后端的成本账本暂回退内存账本（重启
清零，方向安全：只会少记不会误熔断）；CLI 不受益（CLI 本就不注入
记忆与多用户）。

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
ruff check src tests examples && ruff format --check src tests examples   # 1. lint + format
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
│   ├── core/                # config（嵌套配置节）/ contracts（含 MemoryPort）/ registry / llm
│   │                        #           · bootstrap / tools（工具池）/ context（上下文引擎）/ graphs
│   ├── memory/              # 记忆系统（M6）：store/（包：models/protocol/ddl/三后端）
│   │                        #           · embeddings / retrieval / pipeline / textkit
│   │                        #           · context（注入组装）/ tools / service
│   ├── extensions/          # 扩展点：observability（request_id + 日志）· memory（checkpointer）
│   │                        #           · collab（supervisor）· events（SSE 契约）· metrics
│   │                        #           · toollog（工具审计）· filestore（附件存储）· mysql57
│   ├── modules/chat/        # 样板模块（graph + module + tools，兼作接入模板）
│   ├── modules/writer/      # 第二样板模块（供 supervisor 编排演示）
│   ├── modules/hello/       # 五步手册的真实落地实例（tests/test_hello_module.py 验证）
│   │                        # 五步接入手册见 docs/module-guide.md（本地文档，不入库）
│   └── entrypoints/
│       ├── cli.py           # CLI 入口（--module / --message / --thread-id / --version）
│       └── server/          # FastAPI + SSE 服务包（H1 拆分）：app 装配 + auth/deps/
│                            #           serializers/sse/background + routes/{agents,memory,files,system}
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

定位与边界依据见 ADR-001/002；对外叙事（本 README 开头的差异化表）与其
保持一致——基座不做编排，做编排之上的治理。
