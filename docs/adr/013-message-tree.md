# ADR-013：会话消息树（任意节点分岔）

状态：已采纳（0.5.x）
日期：2026-11-04
关联：ADR-012（Redis 只做治理状态）、core/context.py（注入标记与历史不改写哲学）

## 背景

用户需要"树型会话记忆"：在长对话的**任意消息节点**分出岔路，前端可视化
整棵消息树，每个子节点消息都完整保留、可维护（重命名标签 / 星标 / 归档 /
软删恢复 / 导出）。

## 调研结论（2026-11）

| 路线 | 代表 | 教训 |
|---|---|---|
| 消息树 + 活跃指针 | ChatGPT 内部模型（solmaz.io）、HF chat-ui、LibreChat | 活跃分支指针必须持久化（chat-ui 刷新回第一分支 bug）；树方向要单一权威（Open WebUI #29299 childrenIds/parentId 不一致 bug） |
| Fork 复制独立会话 | Open WebUI / LibreChat fork | 实现简单但前缀存储冗余、祖先关系弱化 |
| Checkpointer 时间旅行 | LangGraph | 粒度是 superstep 非 消息；fork 后 checkpoint id 行为怪（#4987）；官方 thread copy 慢（社区 12 分钟案例） |
| Swipe/Checkpoint | SillyTavern | **摘要扩展跨分支状态错乱（#3945）——记忆不感知分支的典型事故** |
| 多会话共享记忆 | Letta conversations | 完全不隔离——分岔场景的反面 |

四条设计原则：① 活跃分支指针必须落库；② 树方向以 parent 链为单一权威；
③ 摘要/记忆必须显式设计分支语义；④ 历史 append-only（与本项目
"checkpointer 历史永不改写"哲学一致）。

## 决策

### 1. 消息树是一等数据模型，checkpointer 是"活跃路径的执行缓存"

新表 `message_nodes`（append-only + 软删除）+ `tree_pointers`（会话级活跃
叶子指针）。分岔 = 在任意节点下 append 子节点；**从此分岔 / 编辑重发 /
重新生成都是同一原语的特例**（编辑 = 在原消息前一节点下挂新 human；重新
生成 = 在既有 human 节点下重跑、产生 AI 兄弟）。

- 被否：会话树 + 前缀复制（分支=新 thread）——树可视化需要跨线程拼接，
  "每个子节点消息都保留"的语义要靠复制实现，维护操作（删单节点/编辑）
  语义扭曲。
- 被否：只做 LangGraph time-travel——粒度不是用户可理解的消息节点，无
  维护面。

### 2. 消息身份：节点存 LangChain 消息 id（`msg_id`）

物化重建消息时原样带回 id。状态比对与尾部回写全部按 id 精确匹配——
内容匹配在"重新生成出两条相同回复"时必然歧义。invoke 必须在落树前给
human 消息赋显式 id（id 缺席时 langgraph 入 state 才另发新 id，树与
state 永远对不齐）。

### 3. 物化（materialize_branch）

invoke 前把目标路径写进 checkpointer：先按 `msg_id` 序列与 state 比对，
一致则跳过（同分支续聊零额外写）；不一致用 `REMOVE_ALL_MESSAGES` 全量
替换。只产生新 checkpoint——历史快照仍不可变，审计不丢；历史注入消息
随之被清出 state（`build_model_input` 本来就会剥离它们，模型输入等价）。

### 4. 回写（append_turn_tail）幂等自愈

turn 结束后（含审批暂停）把 state 新增消息按 id 回写进树，锚定本轮
human 节点；中断→恢复的重叠自动跳过。客户端断开/流异常时不回写——树
少一轮尾巴，下轮物化把 state 重写回树的事实，自愈。

### 5. 记忆分支语义（SillyTavern #3945 的反面设计）

- **滚动摘要按节点 memoize**（`node_summaries` 表）："root→该节点的路径
  摘要"。分支共享祖先摘要（写一次），分岔后各自独立续写。冻结注入段
  的缓存键带命中的摘要节点 id——同分支持续命中（供应商前缀缓存地基），
  切分支自动重建。
- 长期记忆（memories）保持 user 级共享不变——那是跨会话学习的本意；
  出处仍记 `source_thread_id`。

### 6. 维护语义

- 软删子树可恢复；活跃路径被删时指针回退到最近存活祖先；根不可删
  （删根 = 删会话，走既有 DELETE /threads/{thread_id} 级联）。
- 跳转分支 = 只改指针（懒物化，下次 invoke 前重写 state）。
- 树写操作（落节点/维护/删除/跳转）在会话写锁内执行：Redis 分布式锁
  `lock:tree:{thread_id}`，缺席退化进程内锁（与 ADR-012 的"Redis 是
  治理状态的家"一致——树数据在 SQL，锁在 Redis）。

## 并发模型（对抗性审查后固化）

所有树写路径的锁覆盖清单——任何新增写路径必须进这个表：

| 写路径 | 临界区 | 说明 |
|---|---|---|
| invoke 落 human 节点 + 推指针 | `tree_write_lock` | 锁内同时完成迁移/父解析/落节点/物化准备 |
| 维护端点（patch/delete/restore/jump） | `tree_write_lock` | 读端点（tree/export）的迁移也在锁内——迁移是写 |
| 捕获钩子的尾部回写（invoke/resume 轮） | `tree_write_lock` | 后台任务也要拿锁：与落节点在 `next_node_seq` 上竞态会让 seq 重复、打乱树结构；锁超时自弃、下轮物化自愈 |
| 删除会话的树级联 | 索引行闸门 | `append_turn_tail` 先查 `thread_index`，行消亡即自弃——删除与钩子并发不产生孤儿行 |

进程内锁按事件循环隔离（`asyncio.Lock` 绑定创建循环，跨循环复用会抛
"bound to a different event loop"——同进程多循环的测试/嵌入场景会踩）；
生产 uvicorn 单循环，行为不变。

另两条写安全线：

- **invoke 属主校验**：invoke 是唯一带写语义的会话端点，此前未校验属主
  ——知道他人 thread_id 即可向其树落节点/推指针/物化覆盖。"存在即校验"
  （索引无行 = 新线程放行，有行则属主必须匹配，404 掩蔽）。
- **物化与执行缓存**：物化在锁外（state 是缓存不是事实源，写错会被下轮
  物化覆盖）；树表是唯一事实源。

## 性能边界（已知量级与演进方向）

- `list_nodes` 全量加载（含 content）：每 invoke 一次 + 钩子两次。千节点
  会话 ≈ MB 级读/轮，百节点级无感。演进：结构查询与内容查询分离
  （brief 不拉 content/attachments 列）。
- 树 API 默认全量节点 + 80 字预览：千节点 ≈ 100KB 响应。演进：分页 /
  懒展开子树。
- 前端树面板 O(n) 渲染无虚拟化：千节点 DOM 压力。演进：虚拟滚动或
  折叠加载。
- `seq` 分配用线程内 MAX+1（锁内无竞态）：删除产生的序号空洞无影响
  （只用于排序）。

## 后果

- 正面：前端获得 ChatGPT 级体验（分支切换器 `< 1/2 >` + 树面板）；
  每个子节点消息永久保留；摘要/记忆显式分支感知。
- 代价：checkpointer 的 state 会随切分支被重写（历史快照仍可追溯）；
  树表与 state 短暂可能不一致（回写失败时），靠物化自愈收敛。
- 存量数据：旧线程首次访问树接口/树模式 invoke 时从 checkpointer 幂等
  迁移为根链。
- 限制：`plan_snapshots` 仍是会话级（不随分支切换）——planner 分支
  语义留待后续 ADR。
