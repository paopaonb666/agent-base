package io.github.paopaonb666.agentbase.core.graph;

import java.util.List;
import java.util.Map;

/**
 * 对话状态存储器 SPI（对应 LangGraph BaseCheckpointSaver 在本项目里的
 * 实际使用面：按线程存取图状态、删线程、列线程）。
 *
 * <p>实现方：InMemory（零依赖，不持久化）/ Sqlite（默认，重启可恢复）/
 * Mysql（MySQL 持久化）。实现必须保证线程安全。
 */
public interface Checkpointer extends AutoCloseable {

    /** upsert 线程状态快照。 */
    void put(String threadId, GraphState state);

    /** 读取线程状态；缺席返回 null。 */
    GraphState get(String threadId);

    /** 删除线程（会话删除端点用）。 */
    void deleteThread(String threadId);

    /**
     * 列出线程 id（thread_index 缺席时的降级扫描源；实现可返回全量，
     * 调用方负责截断）。
     */
    List<String> listThreadIds();

    /**
     * 线程最后一次落盘的时间（epoch 毫秒，对应 Python checkpoint.ts 的
     * 列表时间语义）；未知/读取失败返回 0——调用方按「时间未知」降级，
     * 不报错。
     */
    default long updatedAt(String threadId) {
        return 0L;
    }

    /**
     * 健康探针：探一次存储可读性（对应 /health 的 checkpointer 组件
     * 探 aget_tuple）。不可读返回 false（health degraded，不崩溃）。
     */
    boolean healthy();

    @Override
    void close();
}
