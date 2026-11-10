package io.github.paopaonb666.agentbase.core.context;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.text.EstimatingTokenCounter;
import io.github.paopaonb666.agentbase.core.text.ImageTokenPolicy;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * ContextEngine 边界测试（roadmap #287）：全部模块公共路径的注入去重、
 * 冻结提升、预算修剪与工具配对保护。算法与 Python core/context.py 逐条
 * 对齐（一致性佐证域），锁定后任一侧漂移即红。
 *
 * <p>#339 起默认计量器是真实 tokenizer（jtokkit cl100k）；预算边界类
 * 测试显式注入字符系数估算器（确定性口径），结束恢复默认。
 */
class ContextEngineTest {

    @AfterEach
    void restoreDefaultCounter() {
        ContextEngine.setTokenCounter(null); // null → 默认（JTokkit cl100k）
    }

    /** 预算边界测试用确定性计量器（估算口径与注释里的手算一致）。 */
    private static void useEstimator() {
        ContextEngine.setTokenCounter(EstimatingTokenCounter.INSTANCE);
    }

    private static SystemMessage memoryInjection(String body) {
        return SystemMessage.from(ContextEngine.MEMORY_CONTEXT_PREFIX + body);
    }

    private static SystemMessage frozenInjection(String body) {
        return SystemMessage.from(ContextEngine.MEMORY_CONTEXT_FROZEN_PREFIX + body);
    }

    private static SystemMessage attachmentInjection(String body) {
        return SystemMessage.from(ContextEngine.ATTACHMENT_CONTEXT_PREFIX + body);
    }

    /** 20 个 ASCII 字符：estimate=5，buildModelInput 的固定开销 +4 → 9/条。 */
    private static UserMessage shortHuman(String marker) {
        return UserMessage.from(marker + "x".repeat(20 - marker.length()));
    }

    @Test
    void estimateTokensWeightsCjkHigherThanAscii() {
        assertEquals(0, ContextEngine.estimateTokens(""));
        assertEquals(0, ContextEngine.estimateTokens(null));
        // CJK 0.6/字：2 字 → 1.2 → 截断 1。
        assertEquals(1, ContextEngine.estimateTokens("你好"));
        // ASCII 0.25/字符：4 字符 → 1。
        assertEquals(1, ContextEngine.estimateTokens("abcd"));
        // 混合：1 CJK + 4 ASCII = 0.6 + 1.0 = 1.6 → 1。
        assertEquals(1, ContextEngine.estimateTokens("你abcd"));
    }

    @Test
    void contentTextExtractsTextAndImagePlaceholder() {
        // 纯 tool_calls 的 AiMessage text() 为 null → contentText 回落空串。
        ChatMessage callsOnly = AiMessage.from(List.of(
                dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("c0").name("echo").arguments("{}").build()));
        assertEquals("", ContextEngine.contentText(callsOnly));
        UserMessage multimodal = UserMessage.from(List.of(
                dev.langchain4j.data.message.TextContent.from("看图"),
                dev.langchain4j.data.message.ImageContent.from("aGk=", "image/png")));
        // 图片块以 [image] 占位（估算不被低估太多）。
        assertEquals("看图[image]", ContextEngine.contentText(multimodal));
        assertEquals("工具结果",
                ContextEngine.contentText(ToolExecutionResultMessage.from("c1", "echo", "工具结果")));
    }

    @Test
    void injectedKindRecognizesTheThreePrefixes() {
        // 三前缀互不吞并：frozen 前缀与普通记忆前缀仅一字之差。
        assertEquals(ContextEngine.MEMORY_CONTEXT_PREFIX,
                ContextEngine.injectedKind(memoryInjection("画像...")));
        assertEquals(ContextEngine.MEMORY_CONTEXT_FROZEN_PREFIX,
                ContextEngine.injectedKind(frozenInjection("画像快照...")));
        assertEquals(ContextEngine.ATTACHMENT_CONTEXT_PREFIX,
                ContextEngine.injectedKind(attachmentInjection("doc.txt...")));
        assertNull(ContextEngine.injectedKind(SystemMessage.from("普通 system")));
        assertNull(ContextEngine.injectedKind(UserMessage.from("不是 system")));
    }

    @Test
    void buildModelInputOnEmptyHistory() {
        assertEquals(List.of(), ContextEngine.buildModelInput(null, 1000));
        assertEquals(List.of(), ContextEngine.buildModelInput(List.of(), 1000));
    }

    @Test
    void currentTailInjectionsKeptHistoricalInjectionsDropped() {
        // 历史轮的注入块（记忆/附件）随 checkpoint 轮轮累积——全部剔除，
        // 只保留最后一条 human 之前的连续贴尾注入段。
        List<ChatMessage> history = List.of(
                memoryInjection("旧召回"),
                UserMessage.from("u1"),
                AiMessage.from("a1"),
                attachmentInjection("旧附件全文"),
                UserMessage.from("u2"),
                AiMessage.from("a2"),
                memoryInjection("新召回"),
                attachmentInjection("新附件"),
                UserMessage.from("u3"));

        List<ChatMessage> input = ContextEngine.buildModelInput(history, 10_000);
        // 历史 注入×2 被剔除；本轮贴尾注入（新召回+新附件）保留。
        assertEquals(7, input.size());
        assertTrue(input.contains(memoryInjection("新召回")));
        assertTrue(input.contains(attachmentInjection("新附件")));
        assertFalse(input.contains(memoryInjection("旧召回")));
        assertFalse(input.contains(attachmentInjection("旧附件全文")));
    }

    @Test
    void frozenInjectionPromotedToHeadAndHistoricalCopiesRemoved() {
        List<ChatMessage> history = List.of(
                frozenInjection("旧快照"),
                UserMessage.from("u1"),
                AiMessage.from("a1"),
                frozenInjection("本轮快照"),
                UserMessage.from("u2"));

        List<ChatMessage> input = ContextEngine.buildModelInput(history, 10_000);
        // 冻结段置顶（缓存前缀稳定），历史副本不进输入。
        assertEquals(4, input.size());
        assertEquals(frozenInjection("本轮快照"), input.get(0));
        assertTrue(input.contains(UserMessage.from("u1")));
        assertTrue(input.contains(UserMessage.from("u2")));
        assertFalse(input.contains(frozenInjection("旧快照")));
    }

    @Test
    void budgetBoundaryExactlyAtLimitKeepsAll() {
        useEstimator();
        // 3 条 × 9 token（估算 5 + 开销 4）：预算 27 恰好装下 → 全保留；
        // 预算 26 时头部消息被丢。
        List<ChatMessage> history = List.of(
                shortHuman("a"), shortHuman("b"), shortHuman("c"));
        assertEquals(3, ContextEngine.buildModelInput(history, 27).size());
        assertEquals(2, ContextEngine.buildModelInput(history, 26).size());
        // 只保最后一条的下界：预算小到只够一条。
        assertEquals(1, ContextEngine.buildModelInput(history, 9).size());
    }

    @Test
    void singleMessageOverBudgetKeptVerbatim() {
        useEstimator();
        // 单条超预算的极端情况：原样保留，绝不输出空输入。
        UserMessage huge = UserMessage.from("x".repeat(400)); // 100 tokens
        List<ChatMessage> input = ContextEngine.buildModelInput(List.of(huge), 10);
        assertEquals(1, input.size());
        assertEquals(huge, input.get(0));
    }

    /** #4 预算推导用的最小运行时桩（只携带窗口；ModelRuntime 非函数接口）。 */
    private static io.github.paopaonb666.agentbase.core.llm.ModelRuntime stubRuntime(int window) {
        return new io.github.paopaonb666.agentbase.core.llm.ModelRuntime() {
            @Override public dev.langchain4j.model.chat.ChatModel chat() { throw new UnsupportedOperationException(); }
            @Override public dev.langchain4j.model.chat.StreamingChatModel streaming() { return null; }
            @Override public dev.langchain4j.data.message.AiMessage stream(java.util.List<ChatMessage> messages,
                    java.util.List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                    java.util.function.Consumer<String> onDelta) { throw new UnsupportedOperationException(); }
            @Override public dev.langchain4j.data.message.AiMessage invoke(java.util.List<ChatMessage> messages) { throw new UnsupportedOperationException(); }
            @Override public int contextWindowTokens() { return window; }
        };
    }

    @Test
    void resolveInputBudgetDerivesFromPerModelWindow() {
        // #4：未配置窗口 → memory.contextMaxTokens 现状；配置后取
        // min(memoryBudget, 窗口-输出预留)，窗口极小时有下限兜底。
        var settings = new io.github.paopaonb666.agentbase.core.config.Settings();
        int memoryBudget = settings.memory.contextMaxTokens;
        assertEquals(memoryBudget, ContextEngine.resolveInputBudget(settings, null));
        assertEquals(memoryBudget, ContextEngine.resolveInputBudget(settings, stubRuntime(0)));
        assertEquals(Math.min(memoryBudget, 8192 - ContextEngine.OUTPUT_RESERVE_TOKENS),
                ContextEngine.resolveInputBudget(settings, stubRuntime(8192)));
        // 窗口 - 预留 < 1024 → 下限兜底后再与 memoryBudget 取小。
        assertEquals(1024, ContextEngine.resolveInputBudget(settings, stubRuntime(2048)));
        // memoryBudget 本身更小 → 以其为预算（组装面约束仍在）。
        settings.memory.contextMaxTokens = 2000;
        assertEquals(2000, ContextEngine.resolveInputBudget(settings, stubRuntime(8192)));
    }

    @Test
    void imageContentCountsFixedEstimateInBudget() {
        // #130：图片块按固定估价（1500）计入预算——"[image]" 占位只是渲染
        // 语义，按 2 token 计会让多模态轮次的修剪触发点严重失真。
        useEstimator();
        UserMessage withImage = UserMessage.from(List.of(
                dev.langchain4j.data.message.TextContent.from("x".repeat(20)),
                dev.langchain4j.data.message.ImageContent.from("aGk=", "image/png")));
        // 文本（估算器口径）+ 1500（图）；+4 开销在预算循环层，不在本计数。
        assertEquals(EstimatingTokenCounter.INSTANCE.countTokens("x".repeat(20))
                        + ImageTokenPolicy.TOKENS_PER_IMAGE,
                ContextEngine.countMessageTokens(withImage));
        // 预算 1500 装不下 → 只保最后一条（原样）；纯文本对照同预算可装下
        // 多条——图片显著提前触发修剪。
        List<ChatMessage> history = List.of(shortHuman("a"), withImage);
        assertEquals(1, ContextEngine.buildModelInput(history, 1500).size());
        assertEquals(withImage, ContextEngine.buildModelInput(history, 1500).get(0));
        // 纯文本口径回归：单文本块求和 = 全文计数（旧口径逐字节等价）。
        assertEquals(EstimatingTokenCounter.INSTANCE.countTokens("abcd"),
                ContextEngine.countMessageTokens(UserMessage.from("abcd")));
    }

    @Test
    void trimBoundaryNeverStartsWithToolResult() {
        useEstimator();
        // 修剪边界安全：起点不得是 ToolMessage（拆散工具配对）——遇到
        // ToolMessage 就把它也丢掉，从下一条开始。
        List<ChatMessage> history = List.of(
                UserMessage.from("x".repeat(100)),      // 25 tokens
                AiMessage.from("y".repeat(100)),        // 25 tokens
                ToolExecutionResultMessage.from("c1", "echo", "z".repeat(100)), // 25
                UserMessage.from("w".repeat(20)));      // 5 tokens

        // 从尾部累计：9,38,67>60 → cut=2（起点恰是 ToolMessage）→
        // 边界守卫把 ToolMessage 也丢掉 → 只剩最后一条。
        List<ChatMessage> input = ContextEngine.buildModelInput(history, 60);
        assertEquals(1, input.size());
        assertFalse(input.get(0) instanceof ToolExecutionResultMessage);
        assertEquals("w".repeat(20), ((UserMessage) input.get(0)).singleText());
    }

    @Test
    void toolCallPairSurvivesTrimming() {
        // 反向：ai(tool_calls) 与其 ToolMessage 必须成对出现（保尾修剪下
        // 配对整体保留——一旦保留 ai(tool_calls)，其后的 tool 必然在预算内）。
        ChatMessage withCall = AiMessage.from("", List.of(
                dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("c1").name("echo").arguments("{}").build()));
        List<ChatMessage> history = List.of(
                UserMessage.from("x".repeat(100)),
                withCall,
                ToolExecutionResultMessage.from("c1", "echo", "z".repeat(100)),
                UserMessage.from("w".repeat(20)));
        List<ChatMessage> input = ContextEngine.buildModelInput(history, 10_000);
        assertEquals(4, input.size());
        // 配对相邻且顺序正确。
        int callIdx = -1;
        for (int i = 0; i < input.size(); i++) {
            if (input.get(i) == withCall) {
                callIdx = i;
            }
        }
        assertTrue(input.get(callIdx + 1) instanceof ToolExecutionResultMessage);
    }

    @Test
    void originalListIsNotMutated() {
        List<ChatMessage> history = new java.util.ArrayList<>(List.of(
                memoryInjection("注入"), UserMessage.from("u1"), UserMessage.from("u2")));
        ContextEngine.buildModelInput(history, 10_000);
        assertEquals(3, history.size());
    }

    @Test
    void wrapUntrustedMarksExternalToolsOnly() {
        // #84①：外部数据工具（web_search/knowledge_search/MCP 前缀）的
        // 非空结果包头部标记；豁免名单与 null/空结果原样返回。
        String marked = ContextEngine.wrapUntrusted("web_search", "网页正文");
        assertTrue(marked.startsWith(ContextEngine.UNTRUSTED_TOOL_RESULT_PREFIX), marked);
        assertTrue(marked.endsWith("网页正文"));
        assertTrue(ContextEngine.wrapUntrusted("knowledge_search", "文档内容")
                .startsWith(ContextEngine.UNTRUSTED_TOOL_RESULT_PREFIX));
        assertTrue(ContextEngine.wrapUntrusted("mcp_fs_read", "x")
                .startsWith(ContextEngine.UNTRUSTED_TOOL_RESULT_PREFIX));
        // 豁免：用户自有数据与无外部内容工具。
        assertEquals("记忆内容", ContextEngine.wrapUntrusted("memory_search", "记忆内容"));
        assertEquals("42", ContextEngine.wrapUntrusted("calculator", "42"));
        assertEquals("y", ContextEngine.wrapUntrusted("current_time", "y"));
        // null/空结果不包（包裹后空串会产生孤儿标记行）。
        assertNull(ContextEngine.wrapUntrusted("web_search", null));
        assertEquals("", ContextEngine.wrapUntrusted("web_search", ""));
        // 判据单源：Graphs/ExecuteNode 共用同一函数。
        assertTrue(ContextEngine.isExternalDataTool("web_search"));
        assertFalse(ContextEngine.isExternalDataTool("memory_search"));
        assertFalse(ContextEngine.isExternalDataTool(null));
    }

    @Test
    void tokenFactorScalesCountsAndKeepsUnityIdentical() {
        // #340：factor=1.0 与旧口径逐字节等价；factor>1 每条消息计量放大
        // （含 +4 固定开销）→ 同预算下修剪更早。
        useEstimator();
        List<ChatMessage> messages = List.of(
                UserMessage.from("第一条足够长的消息内容甲乙丙丁"),
                UserMessage.from("第二条足够长的消息内容戊己庚辛"));
        List<ChatMessage> unity = ContextEngine.buildModelInput(messages, 40);
        List<ChatMessage> same = ContextEngine.buildModelInput(messages, 40, 1.0);
        assertEquals(unity, same, "factor=1.0 必须与旧口径逐字节一致");

        List<ChatMessage> conservative = ContextEngine.buildModelInput(messages, 40, 2.0);
        assertTrue(conservative.size() <= unity.size(),
                "factor=2.0 的保留条数不得多于 factor=1.0：" + conservative.size());
        assertFalse(conservative.isEmpty(), "至少保留最后一条的守卫不因系数失效");
    }

    // ── 历史压缩触发检测（#328 纯函数） ──────────────────────────────────

    @Test
    void compactionNeededFollowsBudgetRatioMatrix() {
        useEstimator();
        // 3 条 × (估算 5 + 开销 4) = 27 token（与 buildModelInput 同口径）。
        List<ChatMessage> history = List.of(
                shortHuman("a"), shortHuman("b"), shortHuman("c"));
        // 27 > 30×0.7=21 → 触发。
        assertTrue(ContextEngine.compactionNeeded(history, 30, 0.7));
        // 27 > 27×0.9=24.3 → 触发；27 > 27×1.0=27 不成立（严格大于）→ 不触发。
        assertTrue(ContextEngine.compactionNeeded(history, 27, 0.9));
        assertFalse(ContextEngine.compactionNeeded(history, 27, 1.0));
        // 低于阈值：27 <= 40×0.7=28 → 不触发。
        assertFalse(ContextEngine.compactionNeeded(history, 40, 0.7));
    }

    @Test
    void compactionNeededSkipsEmptyShortHistoryAndBadBudget() {
        useEstimator();
        List<ChatMessage> history = List.of(shortHuman("a"), shortHuman("b"));
        // 空/null 历史与坏预算一律 false（防御）。
        assertFalse(ContextEngine.compactionNeeded(List.of(), 100, 0.7));
        assertFalse(ContextEngine.compactionNeeded(null, 100, 0.7));
        assertFalse(ContextEngine.compactionNeeded(history, 0, 0.7));
        assertFalse(ContextEngine.compactionNeeded(history, -5, 0.7));
    }

    @Test
    void compactionNeededIgnoresToolPairingAndDoesNotMutateMessages() {
        // 纯函数不裁消息：工具配对、注入去重都是写回侧折叠边界的事，触发
        // 检测只做计量——含配对序列照常计数，且不修改入参列表。
        useEstimator();
        ChatMessage withCall = AiMessage.from("", List.of(
                dev.langchain4j.agent.tool.ToolExecutionRequest.builder()
                        .id("c1").name("echo").arguments("{}").build()));
        List<ChatMessage> history = new java.util.ArrayList<>(List.of(
                shortHuman("a"),                                  // 9
                withCall,                                         // 0+4 = 4
                ToolExecutionResultMessage.from("c1", "echo", "z".repeat(20)), // 5+4 = 9
                shortHuman("b")));                                // 9
        int expected = 9 + 4 + 9 + 9;
        // 31 > 40×0.7=28 → 触发；40×0.8=32 → 不触发。
        assertTrue(ContextEngine.compactionNeeded(history, 40, 0.7));
        assertFalse(ContextEngine.compactionNeeded(history, 40, 0.8));
        assertEquals(4, history.size(), "纯函数不得修改入参");
    }

    // ── 四域预算拆分（条目 15 / R2-F1 收口） ──────────────────────────────

    /** 带 tokenFactor 的窗口桩（budgetAfterSystem 的系数透明口径用）。 */
    private static io.github.paopaonb666.agentbase.core.llm.ModelRuntime stubRuntime(int window,
                                                                                     double factor) {
        return new io.github.paopaonb666.agentbase.core.llm.ModelRuntime() {
            @Override public dev.langchain4j.model.chat.ChatModel chat() { throw new UnsupportedOperationException(); }
            @Override public dev.langchain4j.model.chat.StreamingChatModel streaming() { return null; }
            @Override public dev.langchain4j.data.message.AiMessage stream(java.util.List<ChatMessage> messages,
                    java.util.List<dev.langchain4j.agent.tool.ToolSpecification> tools,
                    java.util.function.Consumer<String> onDelta) { throw new UnsupportedOperationException(); }
            @Override public dev.langchain4j.data.message.AiMessage invoke(java.util.List<ChatMessage> messages) { throw new UnsupportedOperationException(); }
            @Override public int contextWindowTokens() { return window; }
            @Override public double tokenFactor() { return factor; }
        };
    }

    @Test
    void budgetAfterSystemFollowsWindowMatrix() {
        useEstimator();
        var settings = new io.github.paopaonb666.agentbase.core.config.Settings();
        settings.memory.contextMaxTokens = 1000;
        // window=0：原样返回（逐字节兼容，R2-F1 的缝只在窗口面）；prompt 有无都一样。
        assertEquals(1000, ContextEngine.budgetAfterSystem(settings, stubRuntime(0), "系统提示"));
        assertEquals(ContextEngine.resolveInputBudget(settings, stubRuntime(0)),
                ContextEngine.budgetAfterSystem(settings, stubRuntime(0), "x".repeat(400)));
        // window>0：systemPrompt 计量后从预算扣除（40 ASCII 字符 = 10 token）。
        assertEquals(1000 - 10,
                ContextEngine.budgetAfterSystem(settings, stubRuntime(8192), "x".repeat(40)));
        // null prompt：无 prompt 可计量，原样返回（Graphs 的 null 提示形态）。
        assertEquals(1000, ContextEngine.budgetAfterSystem(settings, stubRuntime(8192), null));
        // floor 兜底：扣减吞掉全部预算（raw < 1）→ 回落 min(budget, 下限)=1024，
        // 不为负不超预算；正常扣减（raw >= 1）诚实生效。
        settings.memory.contextMaxTokens = 1030;
        assertEquals(1024,
                ContextEngine.budgetAfterSystem(settings, stubRuntime(8192), "x".repeat(4200)));
        // tokenFactor 透明：1.0 与默认口径同值；2.0 计量放大（含 prompt）。
        settings.memory.contextMaxTokens = 1000;
        assertEquals(ContextEngine.budgetAfterSystem(settings, stubRuntime(8192), "x".repeat(40)),
                ContextEngine.budgetAfterSystem(settings, stubRuntime(8192, 1.0), "x".repeat(40)));
        assertEquals(1000 - 20,
                ContextEngine.budgetAfterSystem(settings, stubRuntime(8192, 2.0), "x".repeat(40)));
    }

    @Test
    void budgetAfterSystemFloorsBelowPreBudgetWithoutRaisingIt() {
        // floor 不高于扣减前预算：historyCap 把预算压到 MIN_INPUT_BUDGET 以下
        // 时，SYSTEM 扣减诚实生效（490），floor 只防 0/负、不得把预算抬回去
        // （下限不破独立上限——R2-F1 的目标恰是小预算场景）。
        useEstimator();
        var settings = new io.github.paopaonb666.agentbase.core.config.Settings();
        settings.memory.contextMaxTokens = 1000;
        settings.memory.contextBudgetHistoryTokens = 500;
        // 500 - 10 = 490（诚实扣减）；floor 仅在 raw < 1 时回落 min(budget, 1024)。
        assertEquals(490,
                ContextEngine.budgetAfterSystem(settings, stubRuntime(8192), "x".repeat(40)));
    }

    @Test
    void budgetAfterSystemEnforcesOptionalSystemHardCap() {
        // contextBudgetSystemTokens > 0 = systemPrompt 独立硬顶：计得 tokens
        // 超顶即 SettingsError（装配期校验语义；=顶不超、window=0 不计量）。
        useEstimator();
        var settings = new io.github.paopaonb666.agentbase.core.config.Settings();
        settings.memory.contextMaxTokens = 1000;
        settings.memory.contextBudgetSystemTokens = 5;
        var error = assertThrows(io.github.paopaonb666.agentbase.core.config.Settings.SettingsError.class,
                () -> ContextEngine.budgetAfterSystem(settings, stubRuntime(8192), "x".repeat(40)));
        assertTrue(error.getMessage().contains("context_budget_system_tokens"), error.getMessage());
        // 恰好到顶（10 = 10）不算超；cap=0（默认）不做硬顶；window=0 不计量。
        settings.memory.contextBudgetSystemTokens = 10;
        assertEquals(990, ContextEngine.budgetAfterSystem(settings, stubRuntime(8192), "x".repeat(40)));
        settings.memory.contextBudgetSystemTokens = 1;
        assertEquals(1000, ContextEngine.budgetAfterSystem(settings, stubRuntime(0), "x".repeat(40)));
    }

    @Test
    void resolveInputBudgetAppliesHistoryCapTopping() {
        // 条目 15 HISTORY 域：contextBudgetHistoryTokens > 0 = 历史体独立上限
        // （追加 min 封顶，与窗口无关）；=0 不影响既有公式。
        var settings = new io.github.paopaonb666.agentbase.core.config.Settings();
        int memoryBudget = settings.memory.contextMaxTokens;
        assertEquals(memoryBudget, ContextEngine.resolveInputBudget(settings, stubRuntime(0)));

        settings.memory.contextBudgetHistoryTokens = 5000;
        // window=0：min(12000, 5000) = 5000（独立上限与窗口无关）。
        assertEquals(5000, ContextEngine.resolveInputBudget(settings, stubRuntime(0)));
        assertEquals(5000, ContextEngine.resolveInputBudget(settings, null));
        // window>0：窗口推导已把预算压到 4096 → 更小的 cap(3000) 再封顶；
        // 更大的 cap(5000) 不抬高预算。
        assertEquals(4096, ContextEngine.resolveInputBudget(settings, stubRuntime(8192)));
        settings.memory.contextBudgetHistoryTokens = 3000;
        assertEquals(3000, ContextEngine.resolveInputBudget(settings, stubRuntime(8192)));
        // cap 大于 memoryBudget：取小后仍是 memoryBudget（封顶不抬高）。
        settings.memory.contextBudgetHistoryTokens = 5000;
        settings.memory.contextMaxTokens = 2000;
        assertEquals(2000, ContextEngine.resolveInputBudget(settings, stubRuntime(0)));
    }

    @Test
    void warnIfToolResultOverCapWarnsOnlyWhenOver() {
        // 条目 15 TOOL_RESULTS 域观测面：cap>0 且超限 → WARN 一条（带工具名
        // 与估算 token 数）；未超 / cap=0 / null/空输出 → 静默零事件。
        useEstimator();
        Logger engineLogger = (Logger) LoggerFactory.getLogger(ContextEngine.class);
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        engineLogger.addAppender(appender);
        try {
            ContextEngine.warnIfToolResultOverCap("web_search", "x".repeat(400), 0);  // cap=0：零开销零日志
            ContextEngine.warnIfToolResultOverCap("web_search", "x".repeat(400), 200); // 100 token ≤ 200：静默
            ContextEngine.warnIfToolResultOverCap("web_search", null, 50);             // null：静默
            ContextEngine.warnIfToolResultOverCap("web_search", "", 50);               // 空：静默
            assertEquals(0, warnCount(appender));
            ContextEngine.warnIfToolResultOverCap("web_search", "x".repeat(400), 50); // 100 > 50：告警
            assertEquals(1, warnCount(appender));
            assertTrue(appender.list.stream()
                    .filter(e -> e.getLevel() == Level.WARN)
                    .findFirst().orElseThrow().getFormattedMessage().contains("web_search"));
        } finally {
            engineLogger.detachAppender(appender);
            appender.stop();
        }
    }

    /** ContextEngine 日志的 WARN 计数（MetricsRenderTest 同款模式）。 */
    private static long warnCount(ListAppender<ILoggingEvent> appender) {
        return appender.list.stream().filter(e -> e.getLevel() == Level.WARN).count();
    }
}
