# ADR-012：Redis 承载治理状态外置（多 worker 共享层）

- 状态：已采纳（0.4.x）
- 日期：2026-11-02
- 相关：ADR-007 进程内调度器（否决过消息队列——本 ADR 回应其边界）、
  `docs/deployment.md` §4（单进程口径表与"外置方向 P2"清单，本 ADR 兑现其中
  1/2/4 三项）、ADR-004 记忆精排（memory 包依赖收敛的先例）、
  `extensions/redissvc.py`（服务封装）

## 背景

`docs/deployment.md` §4 的结论：**同一个数据库可以挂多个进程，但"治理状态"
不跨进程共享**。具体五处进程内状态在多 worker 部署下语义破损：

| 状态 | 位置 | 多 worker 下的破损 |
| --- | --- | --- |
| 成本/预算累计 | `extensions/costmeter.py` `CostMeter` | 各进程独立累计，预算门各自判定——总额被 N 倍低估，可能超支 |
| nonce 防重放 | `entrypoints/server/auth.py` 进程内 LRU | 重放打到另一个 worker 即漏过（docstring 自己承认） |
| 记忆捕获锁 | memory 包 `CaptureLockRegistry` | 同用户并发捕获跨进程互不可见，重复 ADD 竞态回归 |
| 注入快照缓存 | memory 包 `ContextSnapshotCache` | 各进程各自冻结，同线程注入段跨 worker 不一致 |
| 夜间捕获批 | `extensions/scheduler.py` | 多实例同时 force 跑全量批，重复消耗 LLM |

在实现外置之前，生产建议是 `--workers 1`——单进程教义是**能力上限**而不是
设计目标。SSE 多 worker 广播（P2 第 4 项）也在同一清单上。

## 决策

1. **定位：Redis 只做"治理状态外置 + Pub/Sub 镜像"，不是任务系统**。
   ADR-007 否决的是 Celery/消息队列（部署形态升级为 broker+worker，与基座
   定位冲突）；本 ADR 不引入任何 worker/消费组/重排队列——Redis 是**状态
   的共享地**，不是执行器。对话与记忆的持久化后端仍是 sqlite/mysql
   （`KNOWN_CHECKPOINTER_BACKENDS` 不变，`test_config.py` 对 `redis`
   后端的拒绝立场不变）。
2. **主依赖 + 默认开启 + 启用即承诺**：`redis>=5.0` 进主依赖
   （`pyproject.toml`，自带 asyncio 客户端与类型标注）；`REDIS_ENABLED`
   默认 `true`，连接/握手失败即启动失败（`RedisService.create` 抛
   `RedisServiceError`）——治理状态"假共享"（以为共享了实际各算各的）比
   不可用更危险，不静默降级。单机无 Redis 的部署显式设
   `REDIS_ENABLED=false`，全部能力回退进程内实现（0.3 行为）。核心测试
   套件经 `tests/conftest.py` 默认关闭，零 Redis 依赖。
3. **配置节与封装**（`core/config.py` `RedisSettings` + `extensions/redissvc.py`
   `RedisService`）：`REDIS_URL`（redis/rediss/unix）/`REDIS_PASSWORD`/
   超时/池上限/**键前缀**（`REDIS_KEY_PREFIX`，默认 `agent_base:`——多应用
   共享实例的命名空间隔离）。全部键与频道自动拼前缀；消费方只依赖
   `RedisService` 原语面（get/set_ex/set_nx/delete_prefix/incr_by_float/
   publish/pubsub_messages/acquire_lock/release_lock），不触碰 redis 客户端
   ——测试用手写 fake（`tests/fakes.py::FakeRedisService`），memory 包也不
   import redis。分布式锁：`SET NX PX` + 持有者 token + Lua compare-and-del
   释放（不误删他人的锁），TTL 兜底持有者崩溃。
4. **成本计数器**（`RedisCostCounters`）：flush 循环（5s）把 pending 行按
   日/月键 `INCRBYFLOAT`（行按自身 ts 归位，批次跨午夜不串日），每次刷新
   `EXPIRE`（40 天/400 天）；预算门 `CostGovernor.acheck()/astatus()` 在
   计数器在场时读共享累计，缺席委托进程内同步路径（旧 API 保留）。计数器
   累加失败进入有界积压下轮重试；读取失败回退进程内值——方向安全
   （只会少记，不会误熔断）。明细行仍走 SQL 账本，两个数据面互不替代。
5. **nonce 防重放**：`AuthBackend.authenticate` 协议改 **async**（0.4 唯一
   破坏性契约变更，自家 0.x 契约，机械迁移）；`HmacHeaderAuth` 在 Redis
   在场时以 `SET nonce:{user_id}:{nonce} NX EX 2×窗口` 判重（deployment.md
   §4 预言的形态），键含 user_id——签名本身绑定 user_id，按用户分键不影响
   防重放语义，还避免跨用户误伤。Redis 故障回退进程内 LRU。
6. **记忆域共享**（memory 包 0.2，entry point 契约扩展为
   `build(*, settings, llm, fast_llm, redis=None)`，`build_memory_plugin`
   经 `inspect.signature` 判断后透传——0.1 老插件不破坏）：
   - `CaptureLocks` 协议 + `RedisCaptureLocks`：抢锁等待超时产出
     `acquired=False`，编排层**跳过本轮**——捕获每轮读线程全量消息，跳过的
     一轮由下一轮补上；重复 ADD 竞态与死等二选一，选前者；
   - `SnapshotCache` 协议 + `RedisSnapshotCache`：键
     `snap:{thread_id}/{user_id}/{agent_id}`（线程在前，`purge_thread` 按前缀
     清理），TTL 限界；get/put 故障降级为未命中/放弃写回——缓存层绝不破坏
     注入路径；
   - 夜间批全局锁 `lock:nightly_capture`：不等待，抢不到即让位
     （`reason=locked`）。
7. **SSE 事件镜像**：`_produce` 每个契约事件进本地队列（主链路、唯一事实
   源）后 best-effort `PUBLISH sse:{thread_id}`；新增只读观察端点
   `GET /v1/agents/{module}/threads/{id}/events`（属主校验同线程历史端点），
   未启用 Redis 时 503。**Pub/Sub 是 at-most-once**：订阅建立前与 Redis 抖动
   期间的事件不补发——镜像缺席不影响正确性，主对话流仍是请求内闭环，
   不拿镜像赌主链路。
8. **可观测与部署**：`/health` 新增 `components["redis"]`（未启用即缺席，
   与 memory 组件同口径；ping 失败 → degraded）；docker-compose 增 `redis:7.4`
   服务（healthcheck + AOF + 命名卷），app 容器覆盖 `REDIS_URL`；CI 的
   `test` job 挂 redis 服务容器，integration 标记的真连测试仿 MySQL 模式
   （importorskip + 探活跳过 + 每次运行唯一键前缀）。

## 被否方案

- **Celery / 消息队列 / 任务系统**：ADR-007 已否决（部署形态升级与基座定位
  冲突）；本 ADR 的 Redis 用法不含任何 worker/队列语义。
- **Redis 作 checkpointer / 记忆存储后端**：对话与记忆数据有强一致与查询
  需求，SQL 后端不可替代；`KNOWN_CHECKPOINTER_BACKENDS` 维持三后端不变。
- **连不上静默降级（启用了却回退进程内）**：多 worker 部署会以为预算门/
  防重放是共享的而实际各算各的——"假共享"比不可用更危险。正确姿势是
  fail-fast，让运维看到。
- **预算门每次 LLM 调用都读 Redis**：`observe()` 是同步 callback（LangChain），
  强行改异步会侵入全部模型客户端；改为 flush 周期批量累加 + 门检查时读一次
  （每请求一次），延迟 ≤5s 的累计口径可接受（预算是粗粒度治理，不是实时计费）。
- **SSE 主链路走 Redis 队列（Stream/List）**：把主链路的正确性押在 Redis
  可用性与 at-least-once 语义上；本地队列 + 镜像双写让镜像纯增益、零回归。
- **Valkey / KeyDB**：协议兼容但生态（客户端验证、运维工具、文档）劣于
  Redis 官方线；不引入第二选择。

## 后果

正面：

- `--workers N` 从"不建议"变为"开箱可用"：预算熔断按全worker总额判定、
  nonce 重放跨 worker 被拒、同用户记忆捕获跨进程串行、注入快照跨进程一致、
  夜间批多实例只跑一份；
- 单进程路径完整保留（`REDIS_ENABLED=false` 行为与 0.3 逐字节一致），核心
  测试套件与 85% 覆盖率门零 Redis 依赖；
- 观察端点给了多实例部署下"从任一实例看正在发生的对话"的能力（前端重挂、
  运维旁路观察）；
- 键前缀隔离让共享实例/多应用共存成为一等公民。

代价（明示）：

- 新增一个必选运维组件：默认开启意味着裸机部署要么起 Redis 要么显式关闭
  （README/env 注释已明示；docker compose 已内置）；
- `AuthBackend.authenticate` 协议变 async——自定义后端实现需要机械迁移
  （`async def` + 调用方 `await`）；
- 预算累计的共享口径有 ≤ 一个 flush 周期（5s）的延迟；
- SSE 镜像是 at-most-once：Redis 抖动期间的事件对观察者丢失（主链路不受
  影响）；观察端点长连接数受 Redis 订阅连接池上限约束；
- 每个事件多一次 `PUBLISH` 往返（socket_timeout 兜底，慢 Redis 不拖垮主链路
  ——发布 best-effort，失败仅告警）。
