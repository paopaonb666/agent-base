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
 *
 * <p><b>不可变契约（#203/A-8）</b>：
 * <ul>
 *   <li><b>messages append-only</b>——节点只经 {@code apply(Map)} 追加；
 *       图执行层是唯一允许原地修改的角色（stripDanglingToolCalls 的取消
 *       善后），业务节点绝不改写历史消息；</li>
 *   <li><b>extra 键整表替换、extra 值不可变</b>——节点对某个 extra 键的
 *       修改 = 在返回的 update 里放一个全新值（fresh List/Map，A-8 的
 *       replacement-in-lambda 约定），绝不原地改写共享容器内的元素；
 *       SQL 后端在 put 内即时 toJson（写线程串行化），InMemory 后端存
 *       {@code copy()}（容器新、元素引用共享）——两者的安全前提都是
 *       本约定；</li>
 *   <li><b>cancelled 标记生命周期</b>——根级快照行元数据（#199），写侧
 *       在取消善后时置位、读侧刻意不解析、新 round 加载自动复位、正常
 *       落盘整行覆盖消失。</li>
 * </ul>
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

    /**
     * 容器级浅拷贝（checkpointer 存快照前用）：messages/extra 换新容器，
     * 消息对象与 extra 值仍共享引用——防「后续节点追加消息 / 整表替换
     * extra 键」有效，不防「原地修改消息对象或 extra 值」。extra 值按
     * 不可变约定使用（写入方一律 fresh List/Map，Graphs/planner/supervisor
     * 各写入点同构；InMemoryCheckpointer 的 javadoc 同口径）。SQL 后端在
     * put 内即时 toJson，不受共享引用影响。
     */
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
