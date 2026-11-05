package io.github.paopaonb666.agentbase.core.storage;

import java.util.List;
import java.util.Map;

/**
 * 基座自有存储：线程索引（thread_index）+ 计划快照（plan_snapshots），
 * 与记忆子系统无关（对应 Python 侧 {@code core/storage/protocol.py} 的
 * {@code ThreadIndexRepo} / {@code PlanRepo} / {@code CoreStore} 三个协议
 * ——Java 无结构化类型，合并为一个接口）。
 *
 * <p>设计依据（移植 protocol.py 注释）：线程元数据自管——invoke 时
 * upsert（属主/模块/标题/更新时间），线程端点的列表与属主校验改查该表，
 * 不再全表扫描 checkpointer，也不再向任意用户返回全库线程列表；计划快照
 * 让 plan 检查端点读表而非图状态——chat 轮穿插不再重置计划。这两张表在
 * {@code MEMORY_ENABLED=false} 时仍然要工作。
 *
 * <p>SQL 实现用"整行覆盖式 upsert"（sqlite {@code INSERT OR REPLACE} /
 * MySQL {@code REPLACE INTO}，同为删旧插新的原子语义）。
 */
public interface CoreStore extends AutoCloseable {

    // ── 会话索引（thread_index；S1 线程作用域的数据面） ─────────────

    /** upsert 一行会话索引（整行覆盖）。 */
    void upsertThreadIndex(ThreadIndexRecord entry);

    /** 读取会话索引；缺席返回 null。 */
    ThreadIndexRecord getThreadIndex(String threadId);

    /**
     * 列出某用户的会话索引，按 updated_at 降序（{@code module} 传 null
     * 表示不过滤模块；{@code limit} 下限为 1）。
     */
    List<ThreadIndexRecord> listThreadIndexes(String userId, String module, int limit);

    /** 删除会话索引行。 */
    void deleteThreadIndex(String threadId);

    // ── 计划快照（plan_snapshots；M10 跨轮延续的数据面） ────────────

    /** upsert 计划快照；{@code tasks} 是任务 dict 清单（实现方自行 JSON 化）。 */
    void savePlan(String threadId, String module, List<Map<String, Object>> tasks,
                  int cursor, int replans);

    /**
     * 读取快照：{@code {tasks: List&lt;Map&gt;, cursor: int, replans: int,
     * module: String, updated_at}}；缺席返回 null。损坏的 tasks_json 按
     * 空清单处理（宽容语义，对应 Python json.loads 失败 → {@code []}）。
     */
    Map<String, Object> getPlan(String threadId);

    /** 删除快照。 */
    void deletePlan(String threadId);

    /** 释放底层资源（连接/池；字典实现清空缓存）。 */
    @Override
    void close();
}
