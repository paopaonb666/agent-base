package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.data.message.ChatMessage;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 图状态：messages 通道（append reducer）+ 其余通道（replace reducer）。
 *
 * <p>对应 Python 侧 {@code MessagesState}（messages）与 {@code PlanState}
 * （messages + tasks/cursor/replans）的统一载体：messages 由图运行时做
 * 追加合并，其余键由节点整表替换（语义与 LangGraph 的通道 reducer 一致）。
 */
public final class GraphState {

    public static final String MESSAGES = "messages";

    private List<ChatMessage> messages;
    private Map<String, Object> extra;
    /**
     * #199 快照行元数据：本步落盘时是否为「取消的半截轮」。只进快照
     * （{@code GraphStateJson.toJson} 据此写 {@code cancelled} 键），不是
     * 状态通道——新 round 加载快照时复位，正常 persistState 整行覆盖后消失。
     */
    private boolean cancelledRound;

    public GraphState() {
        this(List.of(), Map.of());
    }

    public GraphState(List<ChatMessage> messages, Map<String, Object> extra) {
        this.messages = messages == null ? new ArrayList<>() : new ArrayList<>(messages);
        this.extra = extra == null ? new LinkedHashMap<>() : new LinkedHashMap<>(extra);
    }

    public List<ChatMessage> messages() {
        return messages;
    }

    /** 历史只读视图（调用方不得修改返回列表）。 */
    public Map<String, Object> extra() {
        return extra;
    }

    /** 读取一个替换通道的值（不存在返回 null，对应 Python state.get 语义）。 */
    @SuppressWarnings("unchecked")
    public <T> T get(String channel) {
        return (T) extra.get(channel);
    }

    @SuppressWarnings("unchecked")
    public <T> T getOrDefault(String channel, T fallback) {
        Object v = extra.get(channel);
        return v == null ? fallback : (T) v;
    }

    /**
     * 应用节点返回的局部更新：messages 追加，其余键整表替换。
     * null 值视为"无变化"（对应 Python 节点返回 {} 或不含该键）。
     */
    public void apply(Map<String, Object> update) {
        if (update == null) {
            return;
        }
        Object messagesUpdate = update.get(MESSAGES);
        if (messagesUpdate instanceof List<?> list) {
            for (Object item : list) {
                if (item instanceof ChatMessage message) {
                    messages.add(message);
                }
            }
        } else if (messagesUpdate instanceof ChatMessage message) {
            messages.add(message);
        }
        for (Map.Entry<String, Object> e : update.entrySet()) {
            if (e.getKey().equals(MESSAGES) || e.getValue() == null) {
                continue;
            }
            extra.put(e.getKey(), e.getValue());
        }
    }

    /** 深拷贝（checkpointer 存快照前用，防止后续节点修改影响已存状态）。 */
    public GraphState copy() {
        GraphState copied = new GraphState(new ArrayList<>(messages), new LinkedHashMap<>(extra));
        copied.cancelledRound = cancelledRound;
        return copied;
    }

    /** #199 取消轮标记的读取（快照元数据，见字段 javadoc）。 */
    public boolean cancelledRound() {
        return cancelledRound;
    }

    /** #199 取消轮标记的写入（仅取消落盘路径与加载复位使用）。 */
    public void cancelledRound(boolean cancelledRound) {
        this.cancelledRound = cancelledRound;
    }

    /** 以本状态作为 invoke 输入载荷（messages 追加 + extra 合并）。 */
    public Map<String, Object> updatePayload() {
        Map<String, Object> payload = new LinkedHashMap<>(extra);
        payload.put(MESSAGES, messages);
        return payload;
    }
}
