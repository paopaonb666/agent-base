# ADR-007：进程内事件触发器（Loop 3）

- 状态：已采纳（0.3.x）
- 日期：2026-11-01
- 相关：ADR-002 模块契约、`docs/agent-dev/agent-dev-knowledge.md` §Loop 3 事件驱动循环、
  `docs/agent-dev/integration-review.md` §3.4、`entrypoints/server/background.py`、
  `scripts/memory_nightly_capture.py`

## 背景

循环工程方法论把"事件驱动循环"列为价值复合的关键一层：Agent 不再是手动调用
的东西，而是持续运行在更大系统中的组件（cron 定时任务、webhook、消息通道触发）。

本项目现状：
- 有进程内周期任务先例（server lifespan 中成本账本 5s flush 循环）；
- 有 fire-and-forget 通道（`spawn_background`：强引用 + 异常落日志）；
- 有"进程外 cron + 脚本"形态（`scripts/memory_nightly_capture.py`），但无观测、
  无失败退避、无手动触发入口；
- 有 HMAC 请求认证（`server/auth.py`）。

缺的是：把"定时/事件触发"变成基座的一等设施。

## 决策

1. **新增 `extensions/scheduler.py` 进程内调度器**，零新依赖：
   - 调度表达只支持两种：`{"every_seconds": N}` 与 `{"daily_at": "HH:MM"}`——
     覆盖课程 CronTrigger 的主频场景（健康检查、夜间批、周报），**不引
     croniter/APScheduler**（`0 */6 * * *` 类表达式可用 daily_at + 参数近似，
     极端调度需求仍走外部 cron 调 webhook）；
   - 任务声明走 `SCHEDULER_JOBS_JSON`（数组：`name/type/schedule/params`），
     解析失败启动 fail-fast（启用即承诺）；
   - 每任务独立执行循环：到点 → `spawn_background` 执行 → 失败**指数退避**
     （基数可配，封顶 1h），连续失败计数进 `/health`；
   - 调度器随 server lifespan 启停（仿 `_flush_cost_ledger` 先例）。
2. **任务类型注册表**（进程内 dict，模块可扩展）：首期内置两个类型——
   - `memory_nightly_capture`：把 `scripts/memory_nightly_capture.py` 的
     `capture_since` 逻辑收编为可调用（脚本改为薄壳，行为不变）；
   - `noop`（演示与冒烟）。
   自定义任务类型经注册 API 注入（组合根装配，不搞动态 import 插件）。
3. **webhook 手动触发入口**：`POST /v1/triggers/{job_name}`——复用既有 HMAC
   认证与线程无关性（触发即 `spawn_background`，立即 202 返回；执行状态经
   `/health` 与日志可见）。外部 cron 可以此替代 ssh + python 直跑，获得统一
   观测。
4. **单进程口径（明示约束）**：多 worker/多副本部署下调度器会重复触发——
   与 metrics 的单进程口径一致；生产多副本场景用 `SCHEDULER_ENABLED=false`
   关闭并保留外部 cron + webhook 入口。选主（leader election）列 P2。

## 被否方案

- **APScheduler**：功能覆盖（cron 表达式/持久化 job store）超出需求，引入
  重依赖违背依赖地板哲学；其 AsyncIOScheduler 与 lifespan 的生命周期磨合
  还需自定义封装。
- **Celery / 消息队列**：部署形态从"单进程"升级为"broker + worker"，与基座
  当前单进程定位不符（README：多 worker 外置是 P2 的另一个问题）。
- **仅保留外部 cron + 脚本**：无退避、无观测、无手动触发，失败只能翻机器日志；
  且脚本内 `create_runtime` 全量装配在每次触发时冷启动，浪费明显。
- **事件订阅总线**（pub/sub + handlers）：灵活性最高但当前只有一个消费者
  （SSE），抽象成本大于收益；`asyncio.Queue` 事件模型已够用。

## 后果

正面：

- Loop 3 语义落地：定时批处理（记忆夜间捕获、M5 质量报告）与外部系统回调
  （webhook）统一进基座，可观测、可退避、可手动触发；
- 零新依赖，单进程部署模型不变；
- 外部 cron 获得更好的替代入口（`/v1/triggers/*`），脚本保留为兼容形态。

代价（明示）：

- 调度表达受限（无 cron 表达式、无时区/DST 精细语义——`daily_at` 取服务器
  本地时间）；
- 多副本重复触发风险由配置纪律承担（文档 + `/health` 暴露调度器状态）；
- 任务执行无持久化队列：进程重启丢失"到点未执行"的窗口任务（对夜间批类
  任务可接受；幂等性由任务自身保证，如 `capture_turn` 的去重锁）。
