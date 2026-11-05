package io.github.paopaonb666.agentbase.core.storage;

import io.github.paopaonb666.agentbase.core.AgentModule;

import java.util.List;
import java.util.Map;

/**
 * 组合根适配器：把 {@link CoreStore} 适配给模块契约
 * {@link AgentModule.PlanStorePort}（依赖倒置——core 契约不 import 存储
 * 层的实现，存储后端替换时模块与契约都不动）。
 *
 * <p>载荷形态（{@code getPlan}）：{@code {tasks: List&lt;Map&gt;,
 * cursor: int, replans: int, module: String, updated_at}}；缺席返回 null。
 * {@code updated_at} 是 epoch 秒（对应 Python {@code time.time()}）。
 */
public final class PlanStoreAdapter implements AgentModule.PlanStorePort {

    private final CoreStore store;

    public PlanStoreAdapter(CoreStore store) {
        this.store = store;
    }

    @Override
    public void savePlan(String threadId, String module, List<Map<String, Object>> tasks,
                         int cursor, int replans) {
        store.savePlan(threadId, module, tasks, cursor, replans);
    }

    @Override
    public Map<String, Object> getPlan(String threadId) {
        return store.getPlan(threadId);
    }

    @Override
    public void deletePlan(String threadId) {
        store.deletePlan(threadId);
    }
}
