package io.github.paopaonb666.agentbase.core.graphs;

import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * 工具循环检测（roadmap 大件 5）：连续以相同 {@code (工具名, 参数)} 调用
 * 同一工具达到阈值后，第 {@code threshold + 1} 次同参请求被拦截——不执行，
 * 回喂一条结构化提示 ToolMessage 引导模型换参数/换工具/直接收尾。
 *
 * <p>判据单源（仿 {@code ContextEngine.isExternalDataTool} 模式）：
 * {@code Graphs} 的 tools 节点与 planner 的 {@code ExecuteNode} 共用本类，
 * 防两处接线漂移。设计裁决（条目 5）：
 *
 * <ul>
 *   <li><b>无新状态通道</b>——(name, canonicalArgs) 序列从
 *       {@code state.messages()} 尾部反向扫描派生（AiMessage(tool_calls) →
 *       ToolMessage 段的连续工具轮）；遇到 UserMessage（或纯文本 AiMessage）
 *       即止——跨轮的历史重复隔着用户消息，不算"连续"。零序列化改动、
 *       免 reset（对比 supervisor #68 的 HANDOFF_COUNT extra 通道需要每轮
 *       init 清零）；反向扫描窗口封顶 {@link #MAX_SCAN_ROUNDS} 防长历史 O(N)。</li>
 *   <li><b>key 归一</b>——{@code request.arguments()} 是 JSON 串，模型可能
 *       重排序键/变空白/换数字写法：Jackson parse → 递归归一（全深度键序
 *       TreeMap + 数字形态 BigDecimal，L93）→ 规范串再比较
 *       （{@code ToolPool.jsonSafe} 先例；比较规范串即可，不做 hash 摘要，
 *       避免碰撞讨论）。解析失败（非法 JSON）退回原串比较。</li>
 *   <li><b>拦截形态</b>——被拦截的请求不执行（在 callAll 之前分桶，省成本
 *       的前提），每个拦截请求回喂一条 {@link #loopHintText} ToolMessage：
 *       配对完整，路由 END 时模型下轮输入不以未配对 tool_calls 结尾
 *       （OpenAI 兼容 provider 普遍 400 拒收，{@code CompiledGraph.stripDanglingToolCalls}
 *       既有注释明示此坑）。先例：supervisor #68 handoff 预算耗尽。</li>
 * </ul>
 */
public final class ToolLoopDetector {

    private static final Logger LOG = LoggerFactory.getLogger(ToolLoopDetector.class);

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** 反向扫描窗口（轮数）封顶：防长历史的 O(N) 扫描；截断只影响更早轮的计数。 */
    public static final int MAX_SCAN_ROUNDS = 20;

    /** 默认阈值：连续 2 次重复后第 3 次同参请求被拦截（0 = 关闭）。 */
    public static final int DEFAULT_THRESHOLD = 2;

    private ToolLoopDetector() {}

    // ── key 归一 ─────────────────────────────────────────────────────

    /**
     * 请求的循环 key：{@code (工具名, 规范化参数)}。参数经 Jackson parse →
     * TreeMap → 规范 JSON 串（键序稳定、无空白差异）；解析失败退回原串
     * ——非法 JSON 的两次"相同"原串仍判相同，绝不因归一失败抛错打断图。
     * {@code arguments == null} 与 {@code "{}"} 同键（执行通道同款归一）。
     */
    public static String canonicalKey(String toolName, String argumentsJson) {
        String name = toolName == null ? "" : toolName;
        return name + " " + canonicalArgs(argumentsJson);
    }

    /**
     * 参数的规范化串：parse → 递归归一 → 序列化；失败退回原串（null 视作
     * "{}"）。归一是全深度的（L93/R4-b 收口）：旧实现的
     * {@code TypeReference<TreeMap>} 只排序顶层键，嵌套对象保持原序
     * （LinkedHashMap）、数字形态（1 vs 1.0）原样——两类同参请求可绕过
     * 检测；现改为 {@link #normalize 递归归一} 后不再可绕（只会更严，
     * 方向安全不误拦——值确实不同的请求归一后仍不同）。
     */
    private static String canonicalArgs(String argumentsJson) {
        String raw = argumentsJson == null ? "{}" : argumentsJson;
        try {
            Object parsed = MAPPER.readValue(raw, Object.class);
            return MAPPER.writeValueAsString(normalize(parsed));
        } catch (Exception e) {
            return raw;
        }
    }

    /**
     * 递归归一：Map → 键 String 化的 TreeMap（键序全深度稳定）；List →
     * 逐元素归一（元素序保留——数组位置有语义，不排序）；数字统一走
     * {@code BigDecimal.stripTrailingZeros}（1 / 1.0 / 1.00 归一同形）。
     * 其余标量（字符串/布尔/null）原样。
     */
    private static Object normalize(Object value) {
        if (value instanceof Map<?, ?> map) {
            TreeMap<String, Object> out = new TreeMap<>();
            for (Map.Entry<?, ?> e : map.entrySet()) {
                out.put(String.valueOf(e.getKey()), normalize(e.getValue()));
            }
            return out;
        }
        if (value instanceof List<?> list) {
            List<Object> out = new ArrayList<>(list.size());
            for (Object item : list) {
                out.add(normalize(item));
            }
            return out;
        }
        if (value instanceof Number n) {
            return new java.math.BigDecimal(n.toString()).stripTrailingZeros();
        }
        return value;
    }

    /** 一批请求的 key 列表（按请求序，与逐位比对配套）。 */
    public static List<String> requestKeys(List<ToolExecutionRequest> requests) {
        List<String> keys = new ArrayList<>(requests.size());
        for (ToolExecutionRequest request : requests) {
            keys.add(canonicalKey(request.name(), request.arguments()));
        }
        return keys;
    }

    // ── 序列派生（messages 尾部反向扫描） ────────────────────────────

    /**
     * 从消息尾部反向扫描派生连续工具轮的 key 序列（旧 → 新；每轮 = 该轮
     * AiMessage 的请求 key 列表，按请求序）。{@code skipPendingTail} 为真时
     * 先跳过尾部"未配对的 AiMessage(tool_calls)"——那是当前批请求本身
     * （tools 节点视角），不算历史。遇到 UserMessage / SystemMessage /
     * 纯文本 AiMessage 即止：连续工具轮被真实轮次边界断开。
     */
    private static List<List<String>> deriveRounds(List<ChatMessage> messages, boolean skipPendingTail) {
        List<List<String>> rounds = new ArrayList<>();
        if (messages == null || messages.isEmpty()) {
            return rounds;
        }
        int i = messages.size() - 1;
        if (skipPendingTail
                && messages.get(i) instanceof AiMessage pending
                && pending.hasToolExecutionRequests()) {
            i--;
        }
        while (i >= 0 && rounds.size() < MAX_SCAN_ROUNDS) {
            // 跳过本轮的 ToolMessage 结果段，找发起这批结果的 AiMessage(tool_calls)。
            boolean sawToolMessage = false;
            while (i >= 0 && messages.get(i) instanceof ToolExecutionResultMessage) {
                i--;
                sawToolMessage = true;
            }
            if (!sawToolMessage || i < 0
                    || !(messages.get(i) instanceof AiMessage ai)
                    || !ai.hasToolExecutionRequests()) {
                break; // 轮次边界（UserMessage / 纯文本 AiMessage 等）→ 连续性终止
            }
            rounds.add(0, requestKeys(ai.toolExecutionRequests()));
            i--;
        }
        return rounds;
    }

    // ── 判定 ─────────────────────────────────────────────────────────

    /**
     * 当前批每条请求的连续重复次数：与紧邻前序轮<b>逐位</b>同 key 的轮数
     * （0 = 无前序重复，1 = 上一轮同位同参……）。窗口封顶见
     * {@link #MAX_SCAN_ROUNDS}。messages 应为 tools 节点视角（尾部挂着
     * 当前批的发起 AiMessage）。
     */
    public static int[] consecutiveRepeatCounts(List<ChatMessage> messages,
                                                List<ToolExecutionRequest> requests) {
        int[] counts = new int[requests.size()];
        List<List<String>> rounds = deriveRounds(messages, true);
        if (rounds.isEmpty()) {
            return counts;
        }
        for (int p = 0; p < requests.size(); p++) {
            ToolExecutionRequest request = requests.get(p);
            counts[p] = consecutiveRepeats(canonicalKey(request.name(), request.arguments()), p, rounds);
        }
        return counts;
    }

    /** 单请求的连续重复数：从最近一轮往前逐位比对同 key 的连续轮数。 */
    private static int consecutiveRepeats(String key, int position, List<List<String>> rounds) {
        int repeats = 0;
        for (int r = rounds.size() - 1; r >= 0; r--) {
            List<String> round = rounds.get(r);
            if (position < round.size() && round.get(position).equals(key)) {
                repeats++;
            } else {
                break;
            }
        }
        return repeats;
    }

    /** 拦截判定单点：阈值 &gt; 0（0 = 关闭）且连续重复数达到阈值。 */
    public static boolean isIntercepted(int repeatCount, int threshold) {
        return threshold > 0 && repeatCount >= threshold;
    }

    /**
     * tools 轮结束后的路由判定（Graphs 条件边用）：本批请求<b>全部</b>被
     * 拦截 → 真（END；提示 ToolMessage 已配对，协议安全）。从消息尾部定位
     * 刚补齐的工具轮的发起 AiMessage，取其之前的历史重跑同一判定——与
     * tools 节点分桶是同一纯函数，天然一致。结构不符（无尾部工具轮）回
     * false：路由回 agent 是安全兜底。
     */
    public static boolean allInterceptedAfterRound(List<ChatMessage> messages, int threshold) {
        if (threshold <= 0 || messages == null || messages.isEmpty()) {
            return false;
        }
        int i = messages.size() - 1;
        while (i >= 0 && messages.get(i) instanceof ToolExecutionResultMessage) {
            i--;
        }
        if (i < 0
                || !(messages.get(i) instanceof AiMessage ai)
                || !ai.hasToolExecutionRequests()) {
            return false;
        }
        List<ToolExecutionRequest> requests = ai.toolExecutionRequests();
        int[] counts = consecutiveRepeatCounts(messages.subList(0, i), requests);
        for (int count : counts) {
            if (!isIntercepted(count, threshold)) {
                return false;
            }
        }
        return true;
    }

    // ── 文案与可观测（#68 handoffBudgetExhaustedText 同款克制风格） ──

    /**
     * 被拦截请求的收敛提示（作为 ToolMessage 回喂）。{@code repeats} 是
     * 拦截前已连续同参<b>执行</b>的次数（不含本次未执行的请求）。
     */
    public static String loopHintText(String toolName, int repeats) {
        return ("你已连续 %d 次以相同参数调用工具 %s：请换用不同参数、改用其他工具，"
                + "或基于已有结果直接给出最终回答。").formatted(repeats, toolName == null ? "" : toolName);
    }

    /** planner ExecuteNode 的中断摘要（task.summary 用）。 */
    public static String loopBreakSummary(String toolName) {
        return "工具循环中断（连续重复调用 " + (toolName == null ? "" : toolName) + "）";
    }

    /** 拦截发生时的可观测口径（条目 5 裁决：仅 LOG.warn，不扩 metrics 枚举）。 */
    public static void logIntercepted(String toolName, int repeatCount, int threshold) {
        LOG.warn("tool {}: 连续同参调用达阈值（重复 {} 次 >= 阈值 {}），本请求不执行并回喂收敛提示",
                toolName, repeatCount, threshold);
    }
}
