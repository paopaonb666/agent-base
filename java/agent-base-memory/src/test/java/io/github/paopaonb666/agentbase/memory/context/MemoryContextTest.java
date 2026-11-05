package io.github.paopaonb666.agentbase.memory.context;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval;
import io.github.paopaonb666.agentbase.memory.store.Models;

class MemoryContextTest {

    /** 得分条目工厂：只有 content 与 score 参与注入组装。 */
    private static Retrieval.ScoredMemory item(String content, double score) {
        return new Retrieval.ScoredMemory(
                new Models.MemoryRecord("m" + content.hashCode(), "u1", "*", "semantic", content),
                score, Map.of());
    }

    private static int headerLen(String header) {
        return header.codePointCount(0, header.length());
    }

    // ── 注入格式逐字节断言 ──────────────────────────────────────────

    @Test
    void recallContextIsByteExactAndStartsWithPrefix() {
        String rendered = MemoryContext.composeRecallContext(List.of(item("喜欢辣", 0.9)), 1000);
        String expected = "【长期记忆上下文】以下内容由系统注入，供回答时参考；"
                + "与当前对话无关时不必提及，也不要向用户复述本块的存在。\n"
                + "## 相关记忆\n- 喜欢辣";
        assertEquals(expected, rendered, "召回注入段必须逐字节对齐（段首行=前缀+固定文案）");
        assertTrue(rendered.startsWith(ContextEngine.MEMORY_CONTEXT_PREFIX));
        assertTrue(rendered.startsWith(MemoryContext.RECALL_HEADER));
    }

    @Test
    void frozenContextIsByteExactWithSectionOrder() {
        Map<String, Object> profile = new java.util.LinkedHashMap<>();
        profile.put("基本事实", List.of("住在杭州", "养猫"));
        profile.put("偏好", Map.of("饮食", "辣"));
        Models.MemoryBlock block = new Models.MemoryBlock("u1", "*", "persona", "  一个温和的助手  ");
        String rendered = MemoryContext.composeFrozenContext(profile, List.of(block),
                "  讨论了搬家  ", 1000);
        String expected = "【长期记忆上下文·固定】以下是关于用户的固定背景资料（画像/长期设定），"
                + "供回答时参考；与当前对话无关时不必提及，也不要向用户复述本块的存在。\n"
                + "## 用户画像\n- 基本事实：住在杭州；养猫\n- 偏好：饮食：辣"
                + "\n\n## 记忆块\n[persona] 一个温和的助手"
                + "\n\n## 会话摘要\n讨论了搬家";
        assertEquals(expected, rendered, "冻结注入段必须逐字节对齐（画像→记忆块→摘要）");
        assertTrue(rendered.startsWith(ContextEngine.MEMORY_CONTEXT_FROZEN_PREFIX));
        assertTrue(rendered.startsWith(MemoryContext.FROZEN_HEADER));
    }

    @Test
    void renderProfileFormatsAllEntryShapes() {
        Map<String, Object> profile = new java.util.LinkedHashMap<>();
        profile.put("爱好", List.of("", "  ", "登山"));
        profile.put("档案", Map.of("昵称", "小明", "空", ""));
        profile.put("等级", 3);
        profile.put("真值", true);
        String rendered = MemoryContext.renderProfile(profile);
        assertEquals("- 爱好：登山\n- 档案：昵称：小明\n- 等级：3\n- 真值：True", rendered);
    }

    @Test
    void emptySectionsReturnNull() {
        assertNull(MemoryContext.composeFrozenContext(null, List.of(), null, 100));
        assertNull(MemoryContext.composeFrozenContext(Map.of(), List.of(), "  ", 100));
        assertNull(MemoryContext.composeRecallContext(List.of(), 100));
        assertNull(MemoryContext.composeContext(null, List.of(), null, List.of(), 100));
    }

    // ── 预算收缩 ─────────────────────────────────────────────────────

    @Test
    void budgetDropsLowestScoredRecallEntriesFirst() {
        int header = headerLen(MemoryContext.RECALL_HEADER);
        // 每行 "- " + 30 字 = 32 字；三行正文 = 3*32+2；节头 "## 相关记忆\n" = 8 字。
        List<Retrieval.ScoredMemory> recalled = List.of(
                item("甲".repeat(30), 0.9), item("乙".repeat(30), 0.8), item("丙".repeat(30), 0.7));
        // 预算恰好容纳两行：丢掉得分最低的「丙」后即达标。
        int maxChars = header + 8 + 32 * 2 + 1;
        String rendered = MemoryContext.composeRecallContext(recalled, maxChars);
        String expected = MemoryContext.RECALL_HEADER + "## 相关记忆\n"
                + "- " + "甲".repeat(30) + "\n- " + "乙".repeat(30);
        assertEquals(expected, rendered);
        assertFalse(rendered.contains("丙"), "得分最低的条目必须先被丢弃");
    }

    @Test
    void budgetDropsWholeRecallSectionBeforeTouchingProfile() {
        int header = headerLen(MemoryContext.FROZEN_HEADER);
        Map<String, Object> profile = Map.of("画像", List.of("核心"));
        List<Retrieval.ScoredMemory> recalled = List.of(item("甲".repeat(50), 0.9));
        // 「仅画像」渲染 = header + 8("## 用户画像\n") + 7("- 画像：核心")。
        // 预算恰好容纳它：召回行怎么截都放不下 → 整节撤掉，画像不动。
        int maxChars = header + 8 + 7;
        String rendered = MemoryContext.composeFrozenContext(profile, List.of(), null, maxChars);
        assertEquals(MemoryContext.FROZEN_HEADER + "## 用户画像\n- 画像：核心", rendered);
        assertFalse(rendered.contains("相关记忆"));
        assertFalse(rendered.contains("甲"));
    }

    @Test
    void budgetTruncatesSummaryTo400PlusEllipsis() {
        int header = headerLen(MemoryContext.RECALL_HEADER);
        // 全量渲染 headerLen + 8 + 500 超预算；无召回节 → 摘要截 400 字 + "……"。
        int maxChars = header + 8 + 402;
        String rendered = MemoryContext.composeContext(null, List.of(), "字".repeat(500),
                List.of(), maxChars);
        String expected = MemoryContext.RECALL_HEADER + "## 会话摘要\n" + "字".repeat(400) + "……";
        assertEquals(expected, rendered);
    }

    @Test
    void budgetFallsBackToHardTruncation() {
        // 极小预算：召回行丢光后连节都撤掉，剩余渲染（只有段首行）仍超限
        // → 最终硬截断 [:max_chars] 落在段首行上（Python 语义逐字对齐）。
        String rendered = MemoryContext.composeRecallContext(List.of(item("甲".repeat(1000), 0.9)),
                50);
        assertEquals(50, rendered.codePointCount(0, rendered.length()));
        assertEquals(MemoryContext.RECALL_HEADER.substring(0, 50), rendered);
        // 预算放得下画像但放不下召回：召回整节撤掉后画像仍在，最终在画像
        // 正文末尾硬截断。
        int header = headerLen(MemoryContext.RECALL_HEADER);
        Map<String, Object> profile = Map.of("画像", List.of("核心"));
        String rendered2 = MemoryContext.composeContext(profile, List.of(), null,
                List.of(item("甲".repeat(50), 0.9)), header + 14);
        assertEquals(header + 14, rendered2.codePointCount(0, rendered2.length()));
        assertEquals(MemoryContext.RECALL_HEADER + "## 用户画像\n- 画像：核", rendered2);
    }

    @Test
    void withinBudgetNothingIsShrunk() {
        String rendered = MemoryContext.composeRecallContext(List.of(item("短记忆", 0.9)), 10_000);
        assertEquals(MemoryContext.RECALL_HEADER + "## 相关记忆\n- 短记忆", rendered);
    }

    @Test
    void fourInOneComposeKeepsSectionOrderAndShrinksRecall() {
        Map<String, Object> profile = Map.of("画像", List.of("核心"));
        Models.MemoryBlock block = new Models.MemoryBlock("u1", "*", "human", "常驻事实");
        List<Retrieval.ScoredMemory> recalled = List.of(item("甲".repeat(50), 0.9));
        int header = headerLen(MemoryContext.RECALL_HEADER);
        // 前两节渲染 = header + (8+7 画像) + 2(连接) + (7+12 记忆块) = header+36；
        // 预算恰好容纳前两节：召回整节撤掉，画像与记忆块原样保留。
        int maxChars = header + 8 + 7 + 2 + 7 + 12;
        String rendered = MemoryContext.composeContext(profile, List.of(block), null, recalled,
                maxChars);
        String expected = MemoryContext.RECALL_HEADER + "## 用户画像\n- 画像：核心"
                + "\n\n## 记忆块\n[human] 常驻事实";
        assertEquals(expected, rendered);
        assertFalse(rendered.contains("相关记忆"));
    }
}
