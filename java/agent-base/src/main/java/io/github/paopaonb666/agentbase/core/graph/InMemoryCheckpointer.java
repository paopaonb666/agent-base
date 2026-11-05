package io.github.paopaonb666.agentbase.core.graph;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 进程内字典实现的对话状态存储（对应 Python 侧 {@code InMemorySaver}）：
 * 零依赖，仅进程生命周期内有效，重启即丢；memory 后端与测试共用。
 *
 * <p>存取两侧都做浅拷贝（{@link GraphState#copy()}）：存入的快照不被
 * 后续节点修改污染，取出的副本也不影响库存——语义对应 langgraph 的
 * serialize/deserialize 边界。
 */
public final class InMemoryCheckpointer implements Checkpointer {

    private final ConcurrentHashMap<String, GraphState> states = new ConcurrentHashMap<>();

    @Override
    public void put(String threadId, GraphState state) {
        states.put(threadId, state.copy());
    }

    @Override
    public GraphState get(String threadId) {
        GraphState state = states.get(threadId);
        return state == null ? null : state.copy();
    }

    @Override
    public void deleteThread(String threadId) {
        states.remove(threadId);
    }

    @Override
    public List<String> listThreadIds() {
        return List.copyOf(states.keySet());
    }

    /** 存储即进程内存，必然可读（/health 的组件探针恒绿）。 */
    @Override
    public boolean healthy() {
        return true;
    }

    /** 无底层资源可释放（对应 Python close_checkpointer 对无连接实现的空操作）。 */
    @Override
    public void close() {
    }
}
