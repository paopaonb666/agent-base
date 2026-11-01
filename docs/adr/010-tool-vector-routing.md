# ADR-010：向量工具路由（语义 top-k 召回 + 基础/高级分层）

- 状态：已采纳（0.3.x）
- 日期：2026-11-01
- 相关：ADR-003（基座不 import 记忆包）、ADR-004（记忆包 rerank/embedding 客户端）、
  ADR-009（黑名单注册表——路由的剔除来源）、`core/graphs.py`（绑定点）

## 背景

核查点 2/5：`bind_tools` 永远拿到全量池（`graphs.py` 的 `ctx.llm.bind_tools(pool)`）。
工具规模一大（MCP 接入后动辄上百），选项数放大模型的选择偏差与幻觉——把几百个
工具的 schema 全部塞进 prompt，再指望模型在里面对，是逆着注意力机制做事。
同时运行时坏掉的工具一直躺在绑定清单里（ADR-009 只挡住了"执行"，没挡住"被选"）。

记忆包已有完整的 embedding 基础设施（OpenAI 兼容客户端、批量、冷却降级），
但只服务记忆/知识内容，没有"工具描述向量化"这条路径。

## 决策

1. **分层先行**（`ToolSpec.tier`，封闭枚举 basic/advanced，默认 basic）：
   - `basic` 永远绑定：零依赖高确定性（current_time/calculator/json_query），
     数量少、prompt 预算可忽略；
   - `advanced` 参与路由：web_search、python_repl，**MCP 工具一律 advanced**
     （"几百个工具"的路由场景正是 MCP）。
   tier 经 `build_tool_pool(tiers=...)` 落到池包装器属性；未列名者 basic。
2. **每轮动态绑定**：`build_single_agent_graph` 的 agent 节点改为——路由 active 时
   按当轮查询 `router.select()` 后 `ctx.llm.bind_tools(selected)`；inactive 时保持
   建图时全量绑定的快路径（零变化）。planner 在子任务粒度路由（子任务内各轮共享
   目标，无需逐轮）。
3. **选择规则**：`basic 全集 ∪ top-k advanced（默认 k=5）∪ 近轮已用工具`。
   近轮回捞（`TOOL_ROUTING_RECENT_ROUNDS=3`）是正确性护栏：任务中途工具从绑定集
   消失会让历史里的 ToolMessage 变成"幽灵调用"，下一轮即 unknown tool。
4. **黑名单剔除**：ADR-009 冷却中的工具不进绑定集——模型看不到就不会再选，
   这是"规避重复选择失效工具"的提示词侧闭环（执行侧短路是兜底）。
5. **embedder 经 entry point 发现**：记忆包在 `agent_base.memory` 组注册 `embedder`
   入口（`plugin.build_embedder`，与 `MEMORY_ENABLED` 无关）；core 经
   `importlib.metadata` 发现，**源码零静态 import**（守卫测试把守）。共享
   `MEMORY_EMBEDDING_*` 提供方——一进程一个嵌入源，路由不另配 key。
6. **降级链（路由是优化不是依赖）**：未启用 / 未配 key / 记忆包缺席 / 嵌入冷却 /
   advanced 本来就 ≤ top_k——全部回退全量池，行为回到今天。默认关闭
   （`TOOL_ROUTING_ENABLED=false`）。
7. cosine 相似度 core 自带十几行实现（不 import 记忆包的 retrieval）；工具向量
   懒初始化（首次 select 批量嵌入一次并缓存——池是静态的，向量无需失效）。

## 被否方案

- **RAG 全量检索（含 basic 也参与 top-k）**：basic 工具极少且高确定性，召回失败
  的代价（模型拿不到计算器）远大于省下的那点 prompt。
- **LLM 先选工具再绑定（两段式）**：多一跳 LLM 延迟与成本，且把选择偏差问题
  原样搬进第一跳。
- **core 直接 import 记忆包取 embedding 客户端（含懒导入）**：被依赖方向守卫测试
  拦截——ADR-003 的约束是源码级静态零 import，entry point 是既有解法。
- **为路由另设一套 TOOL_ROUTING_EMBEDDING_* 配置**：一进程两套 embedding 提供方
  只有配置成本没有收益；复用 MEMORY_EMBEDDING_* 并在文档说明。
- **路由结果持久化/缓存查询向量**：查询文本几乎不重复，缓存命中率可忽略；
  工具向量才值得缓存（已做）。
- **按 category（basic/network/code）做分层**：category 是观测元数据，语义与
  "是否参与路由"不对齐——新增专用字段 tier，一票职责。

## 后果

正面：

- 核查点 2（向量召回 top-k + 规避失效工具）与点 5（路由/分层/裁剪）闭环；
- prompt 中的工具 schema 规模从 |pool| 降到 |basic| + k + 近轮，随 MCP 扩展不增长；
- 降级链完整：没有 embedding 的部署行为零变化。

代价（明示）：

- agent 节点每轮多一次查询嵌入（一次 HTTP 往返，冷却时降级）；工具索引首次
  select 多一次批量嵌入；
- 绑定集逐轮变化，模型在多轮任务里看到的工具清单可能不同（近轮回捞把该风险
  压到最小）；
- 记忆包新增 `embedder` 入口点——契约面 +1（向后兼容：旧版记忆包无此入口，
  路由安静降级）。
