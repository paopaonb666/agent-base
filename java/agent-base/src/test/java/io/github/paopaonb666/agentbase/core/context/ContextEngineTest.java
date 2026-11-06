package io.github.paopaonb666.agentbase.core.context;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * ContextEngine 边界测试（roadmap #287）：全部模块公共路径的注入去重、
 * 冻结提升、预算修剪与工具配对保护。算法与 Python core/context.py 逐条
 * 对齐（一致性佐证域），锁定后任一侧漂移即红。
 */
class ContextEngineTest {

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
        // 单条超预算的极端情况：原样保留，绝不输出空输入。
        UserMessage huge = UserMessage.from("x".repeat(400)); // 100 tokens
        List<ChatMessage> input = ContextEngine.buildModelInput(List.of(huge), 10);
        assertEquals(1, input.size());
        assertEquals(huge, input.get(0));
    }

    @Test
    void trimBoundaryNeverStartsWithToolResult() {
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
}
