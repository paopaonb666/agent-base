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

配套前端 `agent-base-ui` 无需任何改动：SSE 事件契约（`ping/step/delta/
tool_call/plan/sources/done/error`）、端点路径与状态码语义与 Python 版逐
字节对齐（见仓库根 README 的「事件契约（SSE）」节）。

## 与 Python 版的已知差异

以下差异均为**内部实现口径**，不影响 HTTP 行为与配置契约：

| 差异 | 说明 |
| --- | --- |
| 对话状态存储 | Java 用自有 `ab_thread_states` JSON 快照表（checkpointer SPI 三后端：memory/sqlite/mysql），不读写 LangGraph 的 msgpack superstep 表。HTTP 行为等价：历史回放/恢复/删除/线程列表一致；业务表（tool_call_records / uploaded_files / thread_index / plan_snapshots / cost_usage / 六张记忆表）DDL 逐条一致，可与 Python 版共用同一 MySQL 库（checkpointer 表除外，两版各自管理各自的表） |
| 运行时 | 自建迷你图运行时复刻本项目实际用到的 LangGraph 语义子集（节点/条件边/三通道流式/按线程状态快照/recursion limit）；supervisor 以 handoff 工具模式自实现（对应 langgraph-supervisor 行为） |
| MCP | 第一期未迁移：`MCP_ENABLED=true` 启动即报错（启用即承诺的契约语义保留），默认 false 不受影响 |
| python_repl | 未迁移：`TOOLKIT_ENABLED` 含 `python_repl` 时启动即报专用「未迁移」错误（说明 exec 级隔离语义未承诺，并指引从 TOOLKIT_ENABLED 移除该条目）；不是泛化的「未知工具」报错。合法名：`current_time/calculator/json_query/web_search` |
| DuckDuckGo 引擎 | 未迁移：`SEARCH_ENGINE_PRIORITY` 含 `duckduckgo` 且被选中时构建期报错；Tavily 完整可用 |
| 日志 | SLF4J/Logback 实现，`LOG_JSON=true` 输出与 Python 版同字段的 JSON 行（ts/level/logger/request_id/message） |
| 多进程口径 | 与 Python 版一致：metrics / 预算 / nonce 防重放缓存均为单进程口径，多实例部署需在采集/网关侧聚合 |

## 质量门

Java 侧第一期质量门为 `mvn test`（JUnit 5）：核心行为测试覆盖配置契约、
HMAC 签名防重放（golden 向量与 Python `scripts/memory_user_sig.py` 交叉
验证）、SSE 事件帧逐字节、混合检索打分、token 预算修剪、预算熔断、文档
切块、线程命名空间、图执行（scripted 模型）、服务器路由契约。CI 集成随
Python 侧工作流后续补齐。
