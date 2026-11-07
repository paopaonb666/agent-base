# Java ↔ Python 差异登记簿

`feat/java-rewrite` 分支的 Java 实现（`java/`，见 `java/README.md`）与 Python
实现（`src/agent_base` + `packages/agent-base-memory`）的差异登记簿。前提：
**对外行为（HTTP API + SSE 事件契约 + `.env` 配置契约 + 业务表结构）逐项
一致**；这里登记的是内部实现口径差异、Java 侧扩展与已知待修项。

登记规则：

- 每条带 file 锚点（缩写 `S/` = `java/agent-base/src/main/java/io/github/paopaonb666/agentbase/`，
  `M/` = `java/agent-base-memory/src/main/java/io/github/paopaonb666/agentbase/memory/`，
  `Py/` = `src/agent_base/`）；
- 只登记**仍然存活**的差异——已修掉的不登记（#142 降级线程列表时间语义
  已在 Wave 1 修复对齐；#151/#158 文案已随 #307 一致性锁定网锁死）；
- 分类：**有意差异**（含理由，稳定不改）/ **Java 侧扩展**（超出 Python
  契约的新增面）/ **待修**（登记待后续排期）。

## 一、有意差异（含理由）

| 差异 | 说明与理由 | 锚点 |
| --- | --- | --- |
| 对话状态存储 | Java 用自有 `ab_thread_states` JSON 快照表（checkpointer SPI 三后端：memory/sqlite/mysql），不读写 LangGraph 的 msgpack superstep 表。HTTP 行为等价：历史回放/恢复/删除/线程列表一致；业务表（`tool_call_records` / `uploaded_files` / `thread_index` / `plan_snapshots` / `cost_usage` / 六张记忆表）DDL 逐条一致，可与 Python 版共用同一 MySQL 库（checkpointer 表除外，两版各自管理各自的表） | `S/core/graph/Checkpointer.java`（SPI）、`S/core/graph/{Sqlite,Mysql}Checkpointer.java`、`S/core/storage/` |
| 运行时 | 自建迷你图运行时复刻本项目实际用到的 LangGraph 语义子集（节点/条件边/三通道流式/按线程状态快照/recursion limit）；supervisor 以 handoff 工具模式自实现（对应 langgraph-supervisor 行为）。决策记录见 `docs/adr/004-java-mini-graph-engine.md` | `S/core/graph/GraphBuilder.java`、`docs/adr/004-java-mini-graph-engine.md` |
| #150 supervisor 派生线程键 | LangGraph 用子图 checkpoint namespace 隔离父子图状态；Java 无子图 namespace，supervisor handoff 的子图快照落在派生键 `<supThreadId>@<module>` 下——同一轮会话在 supervisor 快照与模块快照是两行，互不覆盖但也**互不通用**（按原 thread_id 直查只会命中 supervisor 行） | `S/extensions/collab/SupervisorGraphs.java:55`（类注释）、`:217-221`（派生点） |
| #140 score 舍入层级 | 检索打分 round4：Python 在**路由层**舍入（`round(item.score, 4)`），Java 在**服务层**舍入（`Math.round(score*10000)/10000.0`）。同为 4 位小数，线上输出等价；仅调用服务层内部 API 的嵌入方需注意拿到的已是舍入值 | `M/service/MemoryService.java:292`、`:394`、`M/api/MemoryApiController.java:148`；`Py/entrypoints/server/routes/files.py:295` |
| #149 日志毫秒分隔符 | 人类可读日志的毫秒：Java 用点号（`HH:mm:ss.SSS`——logback 的 `%d{}` 里逗号是"日期格式,时区"语法分隔符，不能用），Python `%(asctime)s` 用逗号（logging 默认 `%s,%03d`）。字段序/字段集一致；`LOG_JSON=true` 的 JSON 行两侧同字段，机器解析不受影响 | `S/extensions/Observability.java:133-135`；`Py/extensions/observability.py:63-67` |
| #153 422 校验文案语言 | invoke 请求体 422 文案：Java 在路由层手写中文（字段语义与 FastAPI/pydantic 对齐），Python 是 pydantic 机器生成的英文。状态码、字段与触发条件一致，仅文案语言不同——前端若对 422 文案做字符串匹配需按语言分支 | `S/entrypoints/server/routes/AgentsRoutes.java:150,156,159,164,167`；`Py/entrypoints/server/routes/agents.py:53-59`（pydantic 模型） |
| #127 `_raw` 降级载荷 | 工具参数 JSON 解析失败的审计/事件降级载荷：Java `_raw` = argsJson 原串；Python `_json_safe` 里 `_raw` = `str(kwargs)`（dict 的 repr）。都保证可 JSON 序列化、都只在畸形参数路径出现；载荷字符串形态不同 | `S/core/tools/ToolPool.java:308-310`（`jsonSafe`）；`Py/core/tools.py:76,78` |
| #1+10 重试预算实现 | LLM 弹性包的重试预算：Java 用滑窗**全局**预算（60s/20，`ResilientModelRuntime`）替代候选设计的 run 级计数器——防风暴语义等价，但未走 NodeContext/RunConfig 管道（改造量大，语义已覆盖） | `S/core/llm/ResilientModelRuntime.java`；progress 登记 `docs/plans/upgrade-progress.md` Wave 1 `1+10+73+75` 行 |
| #68 乒乓检测口径 | supervisor 乒乓（A↔B 互递）检测只记 `last_handoff` 不强制中断，靠 `AGENT_HANDOFF_MAX_PER_RUN`（默认 10）总数上限兜底——乒乓止于预算而非 recursionLimit 裸错误 | `S/extensions/collab/SupervisorGraphs.java`；progress 登记 Wave 4 `68` 行 |
| #63 判分模型档位 | `PLANNER_VERDICT=llm` 的 verdict 判分走**模块主档**（ModuleContext 无快档通道），不用快档——判分是破坏性决策步，走主力模型与本仓"破坏性决策步不降级"的成本原则一致 | `S/modules/planner/graph/CheckNodes.java`（verdict 判分）；progress 登记 Wave 4 `63` 行 |
| #249 utc 语义（breaking 双侧修） | 审计/账本时间戳从本地挂钟改为真 UTC：Java `ZonedDateTime.now(UTC)` → isoformat 带 `+00:00`，Python `datetime.now(timezone.utc)` 同步改。**两侧保持一致**；相对旧文档描述的本地时间语义是 breaking（部署侧按 UTC 解析时间戳） | progress 登记 Wave 4 `249` 行 |
| MCP | 第一期未迁移：`MCP_ENABLED=true` 启动即报错（启用即承诺的契约语义保留），默认 false 不受影响 | `S/core/config/Settings.java`（applyMcp）、`java/README.md` |
| python_repl | 未迁移：`TOOLKIT_ENABLED` 含 `python_repl` 时启动即报专用「未迁移」错误（说明 exec 级隔离语义未承诺，并指引从 TOOLKIT_ENABLED 移除该条目）；不是泛化的「未知工具」报错。合法名：`current_time/calculator/json_query/web_search` | `S/toolkit/Toolkits.java`、`S/core/config/Settings.java` |
| DuckDuckGo 引擎 | 未迁移：`SEARCH_ENGINE_PRIORITY` 含 `duckduckgo` 且被选中时构建期报错（指路 Python 的 ddgs extra）；Tavily 完整可用 | `S/toolkit/search/Engines.java`、`S/core/config/Settings.java:634-639` |
| 日志实现 | SLF4J/Logback 实现，`LOG_JSON=true` 输出与 Python 版同字段的 JSON 行（ts/level/logger/request_id/message）；人类可读行的毫秒分隔符差异见 #149 行 | `S/extensions/Observability.java` |
| 多进程口径 | 与 Python 版一致：metrics / 预算 / nonce 防重放缓存均为单进程口径，多实例部署需在采集/网关侧聚合（见 `docs/deployment.md` §4 口径表） | `S/extensions/Metrics.java`、`S/entrypoints/server/AuthBackends.java` |
| #310 覆盖率门起步值 | JaCoCo BUNDLE 行覆盖门起步钉 52%（实测基线 agent-base 54% / memory 58%）；路线图 80% 在 Wave-5 测试组落地前不可达，收紧=改 `java/pom.xml` 单行——流程项（非行为差异），随测试组落地分步收紧 | `java/pom.xml`（check-coverage execution）；progress 登记 Wave 0 `310` 行 |

## 二、Java 侧扩展

| 扩展 | 说明 | 锚点 |
| --- | --- | --- |
| web_search 上下文尾行引用约定 | web_search 注入上下文的末尾追加一行「回答引用以上来源时，请用 [n] 标注对应序号」——引导模型输出可解析的引用锚。Python 侧无此尾行约定；配合 `done` 前发出的 cited sources 事件（`Citations` 把回复里的 [n] 解析回来源子集） | `S/toolkit/search/SearchPipeline.java:98-114`（`formatWebContext`）、`S/entrypoints/server/Citations.java`；progress 登记 Wave 4 `254+338` 行 |
| knowledge_search [n] 输出格式 | knowledge_search 输出带 [n] 序号锚 + 文件/段位元数据，同序 emit Source 事件（url=null，offsets 留在 tool_calls 审计）——wire 契约不动（保 Python 对齐），文本格式是 Java 侧扩展 | `M/tools/MemoryTools.java`（knowledge_search 工具）、`S/entrypoints/server/Citations.java`；progress 登记 Wave 4 `254+338` 行 |
| 搜索管线低质黑名单扩展点 | URL/标题低质模式（子串匹配）默认空、刻意作为扩展点：chat-agent 里的「字典站/拼音站」名单是它自己的业务垃圾源，不属于基座；部署方按需覆写。Python 侧该名单随业务代码自带（基座无此挂点）。#99 的其余诉求已兑现：`SEARCH_DDGS_BACKEND` 幽灵键已 WARN（Settings.java:657-664）、死配置清理已随 #30+245+276 收口、java/README 章节已存在 | `S/toolkit/search/SearchPipeline.java:13-27`、`S/core/config/Settings.java:657-664` |

## 三、待修（登记待后续）

| 项 | 说明 | 锚点 |
| --- | --- | --- |
| #75 attempt/fallback 列未进账本 schema | LLM 弹性包的重试 attempt 与回落（fast→main fallback）信息当前只进日志，未进 `cost_usage` 账本列——留 #76 做 schema 演进 | `S/core/llm/ResilientModelRuntime.java`；progress 登记 Wave 1 `1+10+73+75` 行 |
| #157 toolSafeStreaming 逐 provider 文档 | 语义两侧一致：快档一旦绑工具（Tier 1 / planner fast），工具轮走非流式让服务端聚合出合法 tool_calls（SiliconFlow GLM-4-9B 流式拆工具名的协议违例）；不绑工具仍流式。缺的是逐 provider 的行为文档（哪些端点/模型触发）——文档债，不是代码差异 | `S/core/llm/LlmFactory.java:26-33`（javadoc 已写缘由） |
| #202 条件边未知目标的编译期校验 | 条件边返回未注册目标时当前由运行期兜底（#285 基线锁定执行语义）；编译期声明式「条件边目标集」校验是后续演进项 | `S/core/graph/GraphBuilder.java:87-113`（compile 现有三类校验）、`docs/adr/004-java-mini-graph-engine.md` |
