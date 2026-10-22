# 部署指南

本文覆盖 agent-base 的三种部署形态、环境变量最小集、健康与指标口径、
数据库迁移、身份安全，以及容器用法。**单进程**是当前所有形态的共同前提：
成本账本、nonce 防重放、指标聚合与检索缓存都按进程持有（见下文口径表）。

## 1. 三种部署形态

| 形态 | 入口 | 适用场景 | 状态存储 | 身份 |
| --- | --- | --- | --- | --- |
| CLI 单机 | `python -m agent_base` | 本机自用、脚本化、快速验证 | `memory` 或 `sqlite` | 无（单用户） |
| HTTP 单进程 | `uvicorn agent_base.entrypoints.server:app` | 配合 agent-base-ui / 自建前端，单机多用户 | `sqlite`（默认）或 `mysql` | `HmacHeaderAuth`（`X-User-Id` + `X-User-Sig`） |
| 容器（app + MySQL） | `docker compose up` | 可复现部署、迁移与升级 | `mysql` | 同上；生产必须配 `MEMORY_AUTH_SECRET` |

### CLI 单机

```bash
pip install -e ".[dev]"
cp .env.example .env            # 至少填 LLM_API_KEY
python -m agent_base --module chat --message "你好"
```

CLI 不注入记忆上下文、不跑记忆形成管线、不支持附件与多用户作用域——
这些能力只在 HTTP 入口提供（见 `docs/design-review.md` H4）。

### HTTP 单进程

```bash
uvicorn agent_base.entrypoints.server:app --host 0.0.0.0 --port 8000
```

默认 `CHECKPOINTER_BACKEND=sqlite`（`./agent_base_state.db`），开箱即用；
需要多实例共享状态时切 `mysql`。TLS 由前置反向代理终止，应用不内置证书管理。

### 容器（app + MySQL 8.0）

```bash
cp .env.example .env            # 填 LLM_API_KEY / MEMORY_AUTH_SECRET
docker compose up --build
```

`docker-compose.yml` 定义两个服务：`app`（本仓库镜像，非 root，8000 端口）
与 `mysql`（`mysql:8.0`，utf8mb4，健康检查，命名卷）。`app` 在启动前执行
`alembic upgrade head`，然后以 uvicorn 起服务。数据库凭据通过 `.env` 注入，
不烘进镜像。

## 2. 环境变量最小集

完整清单见 `.env.example`（配置契约）。生产部署至少需要：

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `ENV` | 是 | `development` 或 `production`；production 会启用快速失败校验 |
| `LLM_API_KEY` | 是 | production 缺失即拒绝启动；不允许弱默认值 |
| `LLM_BASE_URL` / `LLM_MODEL` | 是 | OpenAI 兼容端点与模型名（provider 只做拼写校验，不改变客户端） |
| `LLM_PROVIDER` | 否 | `deepseek` / `zhipu` / `openai-compatible`，默认 `deepseek` |
| `AGENT_MODULES` | 是 | 逗号分隔的模块清单，顺序即装配顺序（默认 `chat,writer`） |
| `CHECKPOINTER_BACKEND` | 是 | `sqlite`（默认）/ `memory` / `mysql` |
| `CHECKPOINTER_SQLITE_PATH` | sqlite 时 | 落盘路径；容器内建议放命名卷（如 `/app/data/agent_base_state.db`） |
| `CHECKPOINTER_MYSQL_*` | mysql 时 | host / port / user / password / database；要求 MySQL ≥ 8.0.19 |
| `MEMORY_ENABLED` | 否 | 默认 `true`；关闭时不装配记忆服务 |
| `MEMORY_AUTH_SECRET` | production 且记忆开启时 | 缺失即拒绝启动；用于 `X-User-Sig` HMAC |
| `CORS_ORIGINS` | 否 | 允许的前端来源，默认 `http://localhost:3000`；禁止 `*` |
| `COST_ENABLED` / `COST_DAILY_CNY` / `COST_MONTHLY_CNY` | 否 | 成本计量与预算熔断，默认关闭 |
| `LOG_JSON` | 否 | 生产建议 `true`，输出结构化 JSON 日志 |

> 记忆后端跟随 `CHECKPOINTER_BACKEND`：选 `mysql` 时记忆、工具审计、
> 上传文件、计划快照与成本账本都落同一个 MySQL；选 `sqlite` 时共用同一
> sqlite 文件；选 `memory` 时全部不持久化。

## 3. 健康与指标

### `GET /health`

返回 `{"status": "ok" | "degraded", "components": {...}}`，HTTP 状态码恒为 200：

| 组件 | 取值 | 含义 |
| --- | --- | --- |
| `checkpointer` | `ok` / `error` / `unconfigured` | 探一次 `aget_tuple` |
| `model` | `ok` / `error` / `unconfigured` / `misconfigured` | 默认只做配置检查；`LLM_HEALTH_PROBE_MODEL=true` 才真实探活（30s 缓存） |
| `memory` | `ok` / `degraded` / `error` | 仅 `MEMORY_ENABLED=true` 时出现；缺席不算降级 |
| `cost` | `ok` / `warn` / `blocked` | 仅 `COST_ENABLED=true` 时出现；warn/blocked 都算 degraded |

容器 `HEALTHCHECK` 只验证 HTTP 可应答（存活探针）。**就绪探针**应解析
响应体并判断 `status == "ok"`——数据库断连时服务仍应被摘流，而不是继续接请求。

### `GET /metrics`

Prometheus 文本格式，主要指标：

- `http_requests_total{method,route,status}` / `http_request_duration_seconds`（route 用模板，避免基数爆炸）
- `tool_calls_total{tool,outcome}`、`memory_operations_total{op,outcome}`
- `llm_tokens_total{model,profile,kind}`、`llm_cost_cny_total`

指标是**进程内聚合**：多 worker 时需在采集侧按 instance 聚合，不能假设单点数值。

## 4. 单进程口径表

| 能力 | 作用域 | 进程重启 | 多 worker / 多副本 |
| --- | --- | --- | --- |
| 成本计量与账本 | 每个进程一份 meter + 账本 | sqlite/mysql 从 `cost_usage` 恢复；`memory` 后端清零（只会少记，不会误熔断） | 各进程独立累计，预算门各自判定；需采集侧汇总 |
| 预算熔断 | 进程内聚合值 | 账本恢复后重建 | 不跨 worker 共享，可能短暂超支 |
| nonce 防重放 | 每个 `HmacHeaderAuth` 实例，LRU 4096 | 丢失 | 多 worker 需网关侧防重放或自定义 `AuthBackend` |
| `/metrics` | 进程内计数器 | 归零 | 采集侧按实例聚合 |
| 检索候选 | 每用户最多 2000 条候选 | — | 检索在进程内完成；更大规模需下推索引（H2） |
| BM25 索引缓存 | 进程内 LRU，按内容指纹失效 | 丢失（自动重建） | 各进程各自缓存，一致性由内容指纹保证 |
| SSE 连接 | 绑定到单个 worker | 断开 | 需要粘性会话或单 worker；当前不提供跨 worker 广播 |
| 工具超时 | 每进程线程池 | — | 各自计时，互不影响 |

结论：**同一个数据库可以挂多个进程，但"治理状态"不跨进程共享**。
在实现外置之前，生产建议 `--workers 1`，或把预算/nonce/metrics 收到网关层。

### 外置方向（P2，尚未实现）

1. **成本账本与预算**：把 meter 改为写入共享表（`cost_usage` 增加
   `instance_id` 与时间桶），预算门按 `SUM()` 在库内原子判定；或接 Redis
   计数器。接口层只替换账本实现，`CostMeter` 的调用点不变。
2. **nonce 防重放**：从进程内 LRU 改为 Redis `SET key NX EX <窗口>`，
   或在 API 网关做重放过滤；`AuthBackend` 契约保持不变，由部署方注入。
3. **metrics 聚合**：保持每个 worker 各自暴露 `/metrics`，由 Prometheus
   按 `instance` 抓取并用 `sum by (job)` 聚合；不要把聚合逻辑塞进应用。
4. **SSE**：多 worker 需要粘性会话或消息总线（Redis Pub/Sub）广播事件；
   在此之前保持单 worker。

外置的目标是让"治理状态"成为可水平扩展的共享设施，而不是把单进程结构
直接复制到每个 worker。分阶段落地，不改变现有 `MemoryPort` / `AuthBackend`
契约。

## 5. 数据库迁移（MySQL）

对话状态（checkpointer）、记忆、工具审计、上传文件、计划快照与成本账本
的表结构由 Alembic 版本化管理：

```bash
python -m alembic upgrade head    # 全新库：应用全部迁移
python -m alembic stamp head      # 已由 langgraph setup() 建过表：只对齐版本
```

连接参数优先级：`AGENT_BASE_DB_*` 环境变量 > `.env` 的 `CHECKPOINTER_MYSQL_*`。
例如：

```bash
AGENT_BASE_DB_HOST=mysql AGENT_BASE_DB_NAME=agent_base \
AGENT_BASE_DB_USER=agent AGENT_BASE_DB_PASSWORD=*** \
python -m alembic upgrade head
```

要求 MySQL ≥ 8.0.19（官方 saver）。5.7 会走基座内置兼容 saver，但**不在
支持矩阵内**，新部署不要使用。迁移是显式步骤：应用启动**不会**自动改表，
容器镜像在 compose 的启动命令里先跑 `alembic upgrade head`。

## 6. 安全

### 身份头

所有用户作用域端点（记忆 / 文件 / 会话线程）经统一依赖解析身份：

- `X-User-Id`：缺省 `default`，只允许 `[A-Za-z0-9._-]{1,64}`，非法值 400；
- `X-User-Sig`：配置 `MEMORY_AUTH_SECRET` 后必填，
  `HMAC-SHA256(user_id + "\n" + X-User-Timestamp + "\n" + X-User-Nonce, secret)`（hex）；
- `X-User-Timestamp`：unix 秒，须落在 `now ± MEMORY_AUTH_REPLAY_WINDOW_SECONDS`
  （默认 300s）内，否则 401；
- `X-User-Nonce`：可选；在同一窗口内重复出现即 401（重放拒绝），缓存按进程持有。

签名生成：

```bash
python scripts/memory_user_sig.py --user-id alice --secret <密钥>
```

production 且 `MEMORY_ENABLED=true` 时未配 `MEMORY_AUTH_SECRET` 会**拒绝启动**；
`CORS_ORIGINS="*"` 在任何环境都被拒绝。浏览器不持有密钥——公网多用户部署应在
网关层实现真实身份体系，并通过 `create_app(auth_backend=...)` 注入，
无需修改路由。

### 其它边界

- 上传文件按 magic bytes 校验图片、文档解析有输入/输出上限；
- 工具执行有超时与全量审计；`python_repl` 默认关闭（exec 级能力）；
- 日志中的 key/secret 不进明文；`/metrics` 不暴露身份；
- nonce 缓存是单进程口径，多 worker 必须由网关兜底（见口径表）。

## 7. 容器用法

### 构建与运行

```bash
docker build -t agent-base:latest .
docker run --rm -p 8000:8000 --env-file .env \
  -e CHECKPOINTER_SQLITE_PATH=/app/data/agent_base_state.db \
  -v agent_base_data:/app/data \
  agent-base:latest
```

镜像特征：`python:3.12-slim` 多阶段构建、非 root（uid/gid 10001）、
`EXPOSE 8000`、`HEALTHCHECK` 访问 `/health`、启动命令为
`uvicorn agent_base.entrypoints.server:app`。镜像内不含 `.env`、测试与缓存。

### Compose

```bash
cp .env.example .env
docker compose config --quiet     # 校验语法与变量插值
docker compose up --build
```

- `mysql` 服务：`mysql:8.0`、utf8mb4、健康检查、命名卷 `mysql_data`；
- `app` 服务：`depends_on` 等 MySQL 健康后启动，先迁移再起服务；
- `app_data` 卷保留 sqlite 回退路径与临时文件；
- 停止：`docker compose down`（加 `-v` 会删除数据卷，慎用）。

### 升级

1. 备份数据库/卷；
2. 拉新镜像并 `docker compose up -d --build`（启动命令先跑迁移）；
3. 验证 `/health` 的 `status == "ok"` 与 `/metrics` 可抓取；
4. 回滚 = 回退镜像 + 数据库备份（迁移不自动降级）。

## 8. 相关文档

- `docs/adr/001-langgraph-as-runtime.md`：运行时选型与 CVE 地板；
- `docs/adr/002-module-contract.md`：显式清单与 fail-fast；
- `docs/module-guide.md`：新增模块；
- `docs/design-review.md`：已知边界（H2–H6）与质量门。
