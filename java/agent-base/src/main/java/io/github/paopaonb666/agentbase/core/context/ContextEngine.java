package io.github.paopaonb666.agentbase.core.context;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 基座级上下文引擎（H3 / P1-4）。
 *
 * <p>这是与记忆无关的<b>通用上下文保护</b>：任何模块的图在把持久化历史
 * 发给模型之前都应当经过它，而不是只有 chat 模块独享。两个职责：
 *
 * <ul>
 *   <li><b>注入型系统消息识别</b>（{@link #injectedKind} / {@link #isInjectedSystem}）：
 *       server 层注入的记忆上下文与附件全文以约定的标记前缀开头；修剪逻辑
 *       靠前缀区分"轮轮累积的注入块"与真实历史。没有标记的 system 消息
 *       （如 writer 模板的临时拼装，不落库）不受去重影响。</li>
 *   <li><b>模型输入修剪</b>（{@link #buildModelInput}）：全量历史去"注入型
 *       系统消息"（只保留本轮最新一条）+ token 预算修剪（保尾部、工具调用
 *       序列不拆散）。checkpointer 里的全量历史刻意不动——它是 recall 语义
 *       的载体，修剪只影响"本轮给模型看什么"。</li>
 * </ul>
 *
 * <p>对应 Python 侧 {@code core/context.py}，逐算法移植。
 */
public final class ContextEngine {

    private ContextEngine() {}

    // 注入型系统消息的标记前缀。server 层注入时必须以它开头；修剪逻辑靠
    // 它区分"轮轮累积的注入块"与真实历史。
    public static final String MEMORY_CONTEXT_PREFIX = "【长期记忆上下文】";
    // 冻结注入段（成本治理 T2.1）：画像/记忆块/摘要线程内快照，逐轮逐字节
    // 相同。buildModelInput 把它提升到模型输入头部——供应商前缀缓存按
    // 最长公共前缀命中，稳定的头部让历史体的缓存跨轮持续增长。
    public static final String MEMORY_CONTEXT_FROZEN_PREFIX = "【长期记忆上下文·固定】";
    public static final String ATTACHMENT_CONTEXT_PREFIX = "以下是用户上传的附件内容";

    // token 计量（#339/旧 #2）：默认真实 tokenizer（jtokkit cl100k_base）
    // ——上下文预算/压缩触发的口径；字符系数估算（±30% 误差）降级保留
    // 在 EstimatingTokenCounter，测试与极端场景可注入。
    private static volatile io.github.paopaonb666.agentbase.core.text.TokenCounter tokenCounter =
            io.github.paopaonb666.agentbase.core.text.JTokkitTokenCounter.CL100K;

    /** 当前 token 计量器（预算路径共用）。 */
    public static io.github.paopaonb666.agentbase.core.text.TokenCounter tokenCounter() {
        return tokenCounter;
    }

    /** 注入计量器（测试/装配）；null 回落默认。 */
    public static void setTokenCounter(io.github.paopaonb666.agentbase.core.text.TokenCounter counter) {
        tokenCounter = counter == null
                ? io.github.paopaonb666.agentbase.core.text.JTokkitTokenCounter.CL100K
                : counter;
    }

    /** CJK 感知的粗略 token 估算（降级口径，公共 API 保留）。 */
    public static int estimateTokens(String text) {
        return io.github.paopaonb666.agentbase.core.text.EstimatingTokenCounter.INSTANCE
                .countTokens(text);
    }

    /** 多段 content（部分 provider 的多模态结构）拼接出文本段。 */
    public static String contentText(ChatMessage message) {
        if (message instanceof UserMessage user) {
            StringBuilder sb = new StringBuilder();
            for (dev.langchain4j.data.message.Content content : user.contents()) {
                if (content instanceof dev.langchain4j.data.message.TextContent text) {
                    sb.append(text.text());
                } else if (content instanceof dev.langchain4j.data.message.ImageContent) {
                    sb.append("[image]"); // 图片块没有文本段（占位，渲染语义）
                }
            }
            return sb.toString();
        }
        if (message instanceof SystemMessage system) {
            return system.text();
        }
        if (message instanceof AiMessage ai) {
            return ai.text() == null ? "" : ai.text();
        }
        if (message instanceof ToolExecutionResultMessage tool) {
            return tool.text();
        }
        return String.valueOf(message);
    }

    /**
     * 单条消息的 token 计量（预算口径，#130）：文本块走 tokenCounter；
     * 多模态 UserMessage 的图片块按 {@link io.github.paopaonb666.agentbase.core.text.ImageTokenPolicy}
     * 固定估价（"[image]" 占位是渲染语义，不用于计费——一张真实图片数千
     * token，按 2 token 计会让修剪触发点严重失真）。非多模态消息与旧口径
     * 逐字节等价（单文本块求和 = 全文计数）。
     */
    public static int countMessageTokens(ChatMessage message) {
        if (message instanceof UserMessage user) {
            int total = 0;
            for (dev.langchain4j.data.message.Content content : user.contents()) {
                if (content instanceof dev.langchain4j.data.message.TextContent text) {
                    total += tokenCounter.countTokens(text.text());
                } else if (content instanceof dev.langchain4j.data.message.ImageContent image) {
                    total += io.github.paopaonb666.agentbase.core.text.ImageTokenPolicy
                            .estimate(image.image().mimeType());
                }
            }
            return total;
        }
        return tokenCounter.countTokens(contentText(message));
    }

    /** 输出预留（#4）：输入预算要给模型留出的生成空间（OpenAI 兼容默认输出上限量级）。 */
    public static final int OUTPUT_RESERVE_TOKENS = 4096;
    /** 窗口极小时的输入预算下限（防 0/负预算让修剪逻辑退化为只保一条以下）。 */
    private static final int MIN_INPUT_BUDGET = 1024;

    /**
     * 图执行输入预算推导（#4 per-model 窗口）：运行时携带了窗口（
     * {@code llm.contextWindowTokens() > 0}）时取
     * {@code min(contextMaxTokens, 窗口 - 输出预留)}——上下文窗口是硬顶，
     * 超窗请求会被 provider 直接拒绝；未配置窗口（0）沿用
     * {@code memory.contextMaxTokens} 现状，行为逐字节不变。
     */
    public static int resolveInputBudget(io.github.paopaonb666.agentbase.core.config.Settings settings,
                                         io.github.paopaonb666.agentbase.core.llm.ModelRuntime llm) {
        int memoryBudget = settings.memory.contextMaxTokens;
        int window = llm == null ? 0 : llm.contextWindowTokens();
        if (window <= 0) {
            return memoryBudget;
        }
        return Math.min(memoryBudget, Math.max(MIN_INPUT_BUDGET, window - OUTPUT_RESERVE_TOKENS));
    }

    /** 返回消息的注入类型（匹配的标记前缀）；非注入消息返回 null。 */
    public static String injectedKind(ChatMessage message) {
        if (!(message instanceof SystemMessage system)) {
            return null;
        }
        String content = system.text();
        if (content.startsWith(MEMORY_CONTEXT_FROZEN_PREFIX)) {
            return MEMORY_CONTEXT_FROZEN_PREFIX;
        }
        if (content.startsWith(MEMORY_CONTEXT_PREFIX)) {
            return MEMORY_CONTEXT_PREFIX;
        }
        if (content.startsWith(ATTACHMENT_CONTEXT_PREFIX)) {
            return ATTACHMENT_CONTEXT_PREFIX;
        }
        return null;
    }

    /** 判断一条消息是否是"注入型"系统消息（记忆上下文 / 附件全文）。 */
    public static boolean isInjectedSystem(ChatMessage message) {
        return injectedKind(message) != null;
    }

    /**
     * 构造发给模型的输入：冻结提升 + 注入去重 + 预算修剪（不改原列表）。
     *
     * <ul>
     *   <li><b>冻结段提升</b>（T2.1）：冻结注入段（画像/记忆块/摘要快照）取
     *       最后一条（本轮的）提升到输入<b>头部</b>，历史副本全部剔除——
     *       头部逐轮逐字节稳定，供应商前缀缓存从第二轮起覆盖全部历史；</li>
     *   <li>其余注入型系统消息（本轮召回/附件全文）只保留<b>最后一条 human
     *       之前的连续段</b>（贴尾注入：它们逐轮变化，放在尾部只牺牲自己
     *       之后的缓存，不动历史），历史里的全部剔除——否则会随 checkpoint
     *       轮轮累积，把上下文撑爆；</li>
     *   <li>从尾部按 token 预算保留近期消息（冻结头部不计入预算——它受组装
     *       期预算约束）；修剪边界不得落在工具调用序列中间，保证
     *       AIMessage(tool_calls) 与其 ToolMessage 成对出现；</li>
     *   <li>至少保留最后一条消息（单条超预算的极端情况原样保留）。</li>
     * </ul>
     */
    public static List<ChatMessage> buildModelInput(List<ChatMessage> messages, int maxTokens) {
        if (messages == null || messages.isEmpty()) {
            return List.of();
        }

        // 0) 冻结段提升：最后一条冻结注入是"本轮的"，历史副本不进模型输入。
        int frozenIndex = -1;
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (MEMORY_CONTEXT_FROZEN_PREFIX.equals(injectedKind(messages.get(i)))) {
                frozenIndex = i;
                break;
            }
        }
        ChatMessage frozenMessage = frozenIndex >= 0 ? messages.get(frozenIndex) : null;

        // 1) 注入去重：只保留"本轮"的贴尾注入段——最后一条 human 消息之前
        //    的连续非冻结注入段（召回块 + 附件全文都可能在本轮注入，必须
        //    同时保留）。历史轮次的注入块一律剔除。
        int lastHuman = -1;
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof UserMessage) {
                lastHuman = i;
                break;
            }
        }
        Set<Integer> currentInjections = new HashSet<>();
        if (lastHuman >= 0) {
            int i = lastHuman - 1;
            while (i >= 0 && injectedKind(messages.get(i)) != null
                    && !MEMORY_CONTEXT_FROZEN_PREFIX.equals(injectedKind(messages.get(i)))) {
                currentInjections.add(i);
                i--;
            }
        }
        List<Integer> keptIndices = new ArrayList<>();
        for (int i = 0; i < messages.size(); i++) {
            if (injectedKind(messages.get(i)) == null || currentInjections.contains(i)) {
                keptIndices.add(i);
            }
        }

        // 2) token 预算：从尾部向前累积（每条消息 +4 固定开销）。
        //    #130：图片块按固定估价计入（真实 vision 成本数量级），文本块口径不变。
        int total = 0;
        int cut = 0; // kept_indices 里保留的起点
        for (int run = keptIndices.size() - 1; run >= 0; run--) {
            int index = keptIndices.get(run);
            total += countMessageTokens(messages.get(index)) + 4;
            if (total > maxTokens && run < keptIndices.size() - 1) {
                // 超预算且至少保住最后一条；从这个更早的消息开始丢弃。
                cut = run + 1;
                break;
            }
        }
        List<Integer> kept = keptIndices.subList(cut, keptIndices.size());

        // 3) 修剪边界安全：起点不得是 ToolMessage（会拆散工具配对）。
        List<Integer> trimmed = new ArrayList<>(kept);
        while (trimmed.size() > 1 && cut > 0 // 只在真的发生修剪时才牺牲边界消息
                && messages.get(trimmed.get(0)) instanceof ToolExecutionResultMessage) {
            trimmed.remove(0);
        }

        List<ChatMessage> body = new ArrayList<>();
        for (int index : trimmed) {
            body.add(messages.get(index));
        }
        // 冻结段置顶：与本轮贴尾注入互不干扰，二者合计构成完整注入面。
        if (frozenMessage != null) {
            body.add(0, frozenMessage);
        }
        return body;
    }
}
