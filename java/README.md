# agent-base（Java 重构）

`agent-base` 的 Java 21 / Spring Boot 3 重写版，对应仓库根的 Python 实现
（`src/agent_base` + `packages/agent-base-memory`）。目标：**对外行为一模
一样**——同一套 `.env` 配置契约、同一套 HTTP API + SSE 事件契约、同一套
业务表结构、同样的 fail-fast 语义。分支 `feat/java-rewrite`。

## 模块结构

```
java/
├── pom.xml                # 父 POM（版本集中在 dependencyManagement）
├── mvn-settings.xml       # 项目级 Maven 设置（阿里云 https 镜像 + 正确的本地仓库）
├── agent-base/            # 基座（对应 src/agent_base）
│   ├── core/              # config（配置契约）/ contracts / registry / llm / bootstrap
│   │                      # · context（上下文引擎）/ graphs（共享图工厂）/ graph（迷你图运行时★）
│   │                      # · tools（工具池）/ storage（thread_index+plan_snapshots）
│   │                      # · memory（ServiceLoader SPI 装配点）
│   ├── extensions/        # observability / events（SSE 契约）/ metrics / toollog
│   │                      # / costmeter / filestore / collab（supervisor）
│   ├── toolkit/           # current_time / calculator / json_query / web_search(Tavily) / parsing
│   ├── modules/           # chat / writer / hello / planner（与 Python 侧一一对应）
│   └── entrypoints/       # cli（java -jar）+ server（Spring Boot：routes/auth/sse）
└── agent-base-memory/     # 记忆子系统（对应 packages/agent-base-memory，独立发行包）
    └── memory/
        ├── store/         # 六张记忆表模型 + MemoryStore SPI + InMemory/Sqlite/Mysql
        ├── textkit/       # chunk_text 段落感知切块 + ChunkSpan + 画像截断
        ├── retrieval/     # 混合检索：向量余弦+BM25(中文二元组)+时间半衰+显著度
        ├── embeddings/    # openai 兼容客户端（永不抛异常+60s 冷却+探活缓存）
        ├── context/       # 注入块组装（冻结段+召回段）
        └── pipeline 等    # 形成管线/画像/知识库/service/api/插件
```

依赖倒置与 Python 侧同构：记忆包依赖基座契约；基座经 **ServiceLoader**
（`META-INF/services/io.github.paopaonb666.agentbase.core.memory.MemoryPluginProvider`）
发现记忆包——对应 Python 的 entry point 机制与「启用即承诺」。

## 构建

```bash
# 需 JDK 21 与 Maven 3.9+（-s 指定项目级 settings，绕过全局失效镜像）
cd java
mvn -s mvn-settings.xml test      # 全部测试
mvn -s mvn-settings.xml install   # 构建 fat jar（-exec 分类器）并刷新本地仓库
```

产物（`classifier=exec` 是 Spring Boot fat jar；普通 jar 保留给 reactor 解析）：

- `agent-base/target/agent-base-0.3.0-exec.jar` —— 基座（不含记忆包；装它时 `/v1/memory` 是 404，对应 Python「只装基座」语义）
- `agent-base-memory/target/agent-base-memory-0.3.0-exec.jar` —— **基座 + 记忆包的完整部署形态**（对应 Python 镜像内置记忆包的形态，日常用这个）

容器部署：`java/Dockerfile` 多阶段构建同一产物（非 root、HEALTHCHECK `/health`，
入口为完整形态 jar）。可单独 `docker build -t agent-base-java ./java`，或用
仓库根 compose 的 `app-java` 服务：`docker compose --profile java up --build`
（共享 MySQL 与 `.env`；Java 侧无 alembic，表结构由 SchemaMigrations 启动自举）。

## 运行

读取与 Python 版相同的 `.env`（工作目录下），配置契约逐项一致：

```bash
# CLI（单轮 / 交互式 / 恢复会话）
java -jar agent-base/target/agent-base-0.3.0-exec.jar --message "你好"
java -jar agent-base/target/agent-base-0.3.0-exec.jar --module chat
java -jar agent-base/target/agent-base-0.3.0-exec.jar --module chat --thread-id <id>
java -jar agent-base/target/agent-base-0.3.0-exec.jar --version

# SSE 服务（默认 8000 端口，与 Python 版相同的端点表）
java -jar agent-base-memory/target/agent-base-memory-0.3.0-exec.jar --server
```

**`--server` 旗标的语境说明（#281，事实以 CliMain 现码为准）**：`--server`
是 **CliMain 的服务入口旗标**（对应 Python `uvicorn agent_base.entrypoints.server:app`）
——`java -jar agent-base-...-exec.jar --server` 即起 SSE 服务；memory exec
jar 的 CI 冒烟用同一旗标。旗标被 CliMain 识别（不会落「未知参数」退 2 桶）；
真实启动面由 CI 冒烟与容器/compose 承载。

配套前端 `agent-base-ui` 无需任何改动：SSE 事件契约（`ping/step/delta/
tool_call/plan/sources/done/error`）、端点路径与状态码语义与 Python 版逐
字节对齐（见仓库根 README 的「事件契约（SSE）」节）。Java 侧扩展（#91，
向后兼容加键）：HTTP 错误体与 `error` 事件在人类可读 `detail`/`message`
旁挂结构化 `code`（封闭枚举 `rate_limited/budget_blocked/run_gate_saturated/
thread_busy/unauthorized/not_found/validation/bad_request/module_unavailable/
storage_unavailable/llm_unavailable/recursion_limit/internal_error` 等）——
前端只认 code 不认文案；缺省场景省键（旧 wire 逐字节不变），Python 版无
此字段。

## toolSafeStreaming：快档绑工具轮走非流式（#157）

快档（`profile="fast"`）运行时恒以工具安全形态装配（`toolSafeStreaming=true`）：
**绑定工具的图执行轮退回非流式**（`LangChain4jRuntime.stream` 的判定
`toolSafeStreaming && bindTools → nonStreamFallback`），让服务端聚合出合法
tool_calls；不绑工具的调用（纯对话、记忆形成管线）不受影响，仍走流式。
主档恒原生流式（`toolSafeStreaming=false`，无配置开关）。

**触发面**（哪些路径会观察到工具轮无 delta）：

| 路径 | 行为 |
| --- | --- |
| `profile=fast` 的图执行请求（Tier 1 单 agent / planner，均 bind_tools） | 工具轮非流式；工具轮之间的纯文本轮仍流式 |
| 快档未配置时的回落形态（#128：回落主力配置但保持 toolSafe） | 同上——回落流量不重新暴露在流式 tool_calls 协议缺陷下 |
| 记忆形成管线的快档 LLM 调用（抽取/画像/摘要，不绑工具） | 不触发，正常调用 |
| 主档（main） | 永不触发，原生流式 |

**实证触发端点/模型**：SiliconFlow 的 GLM-4-9B——流式模式下把工具名拆进
`arguments` delta（`function.name` 为空、id 为 null），违反 OpenAI 流式
协议：langchain4j 聚合不出合法 tool_calls，图执行静默拿到空工具调用与空
回答（同一请求非流式则完全正常）。其余 provider 未观测到同形缺陷；此清单
只登记实证事实，不臆测未测端点。

**部署指引**：

- 选型自查：候选模型在「流式 + 绑工具」下若出现空工具调用 + 空回答、而
  非流式同请求正常，即为同形协议违例——它作为快档时本机制已自动兜住
  （工具轮无 delta、首 token 延迟略增，属预期代价）；作为主档使用则无
  保护，请更换模型。
- 已实证违例的模型不要配为主档带工具使用；快档选型可接受非流式工具轮
  语义（成本敏感场景的常规取舍）。
- 该缺陷是服务端聚合行为，网关/代理侧无法修复。

## 与 Python 版的已知差异

以下为核心差异摘要（均为**内部实现口径**，不影响 HTTP 行为与配置契约）。
完整登记簿——**有意差异 / Java 侧扩展 / 待修** 三分类、逐条 file 锚点与
存活状态——见 [`docs/java-differences.md`](../docs/java-differences.md)：

| 差异 | 说明 |
| --- | --- |
| 删线程的 `purge_memories` 参数 | **隐私语义（#172）**：`DELETE /v1/agents/{module}/threads/{thread_id}` 默认只清滚动摘要、知识库分块与注入快照——**删线程不删长期记忆**，memories 行独立于线程存活、出处字段 `source_thread_id` 原样保留（与 Python 版同语义同保留）。Java 单边扩展：可选 `?purge_memories=true|1` 才连跨会话记忆一起清（GDPR 式彻底删除）；Python 端点暂未跟进（行为分歧登记 #315），默认行为两侧一致、无 breaking |
| 对话状态存储 | Java 用自有 `ab_thread_states` JSON 快照表（checkpointer SPI 三后端：memory/sqlite/mysql），不读写 LangGraph 的 msgpack superstep 表。HTTP 行为等价；业务表 DDL 逐条一致，可与 Python 版共用同一 MySQL 库（checkpointer 表除外） |
| 运行时 | 自建迷你图运行时复刻本项目实际用到的 LangGraph 语义子集（节点/条件边/三通道流式/按线程状态快照/recursion limit）；supervisor 以 handoff 工具模式自实现（决策记录：`docs/adr/004-java-mini-graph-engine.md`） |
| 未迁移项 | python_repl（专用「未迁移」错误）/ DuckDuckGo 引擎（构建期报错）——启用即承诺契约保留，默认配置不受影响。MCP 已完整迁移（stdio + HTTP/SSE transport + 池准入 + 健康重连；websocket transport 显式拒绝） |
| 多进程口径 | 与 Python 版一致：metrics / 预算 / nonce 防重放缓存均为单进程口径，多实例部署需在采集/网关侧聚合 |

其余差异（日志毫秒分隔符、422 文案语言、`_raw` 降级载荷、score 舍入层级、
supervisor 派生线程键、web_search/knowledge_search 引用格式扩展等）全部
登记在差异登记簿，不在此重复。

## 质量门

Java 侧质量门为 `mvn test`（JUnit 5）：核心行为测试覆盖配置契约、HMAC
签名防重放（golden 向量与 Python `scripts/memory_user_sig.py` 交叉验证）、
SSE 事件帧逐字节、混合检索打分、token 预算修剪、预算熔断、文档切块、
线程命名空间、图执行（scripted 模型）、服务器路由契约；JaCoCo 行覆盖门
生效（`java/pom.xml`）。CI（`.github/workflows/ci.yml`）已接入：`java` job
跑 `mvn verify` + memory exec jar 产物冒烟（`/health` 200 + `/v1/memory`
非 404），`java-dependency-check` job 跑 OWASP dependency-check 依赖审计门
（CVSS≥9 fail，经 `-Powasp` profile 触发）。
