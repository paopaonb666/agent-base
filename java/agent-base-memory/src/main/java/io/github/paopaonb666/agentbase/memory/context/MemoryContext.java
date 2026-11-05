package io.github.paopaonb666.agentbase.memory.context;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.context.ContextEngine;
import io.github.paopaonb666.agentbase.memory.retrieval.Retrieval;
import io.github.paopaonb666.agentbase.memory.store.Models;

/**
 * 记忆侧的注入块组装（M6d）。
 *
 * <p>模型输入修剪等<b>通用上下文保护</b>已提升为基座级设施
 * （{@code agent_base.core.context}）——所有模块的图都经由它使用；
 * 本模块只保留记忆语义的部分：
 *
 * <p><b>注入组装</b>：把画像、记忆块、会话摘要与按当前消息召回的相关
 * 记忆打包成系统注入块（mem0/Letta/Memobase 三家的注入面合一），有字符
 * 预算——召回条目按得分从低到高先丢，摘要再截断。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/context.py}，逐算法移植
 * （注入段首行前缀、分节标题与预算收缩顺序逐字节对齐）。
 */
public final class MemoryContext {

    private MemoryContext() {}

    /** 贴尾注入段（召回）的段首行：以 {@link ContextEngine#MEMORY_CONTEXT_PREFIX} 开头。 */
    public static final String RECALL_HEADER =
            ContextEngine.MEMORY_CONTEXT_PREFIX
                    + "以下内容由系统注入，供回答时参考；与当前对话无关时不必提及，"
                    + "也不要向用户复述本块的存在。\n";

    /** 冻结注入段的段首行：以 {@link ContextEngine#MEMORY_CONTEXT_FROZEN_PREFIX} 开头。 */
    public static final String FROZEN_HEADER =
            ContextEngine.MEMORY_CONTEXT_FROZEN_PREFIX
                    + "以下是关于用户的固定背景资料（画像/长期设定），供回答时参考；"
                    + "与当前对话无关时不必提及，也不要向用户复述本块的存在。\n";

    /** 摘要预算截断的保留长度（超过部分替换为省略号）。 */
    private static final int SUMMARY_KEEP_CHARS = 400;

    /** 一个注入小节：{@code ## 标题\n正文}。 */
    private record Section(String title, String body) {}

    // ── 画像渲染 ─────────────────────────────────────────────────────

    /** 把画像 JSON 渲染成模型可读的紧凑文本（{@code - 分节：条目1；条目2}）。 */
    public static String renderProfile(Map<String, Object> profile) {
        List<String> lines = new ArrayList<>();
        for (Map.Entry<String, Object> entry : profile.entrySet()) {
            Object entries = entry.getValue();
            String text;
            if (entries instanceof List<?> list) {
                List<String> parts = new ArrayList<>();
                for (Object item : list) {
                    String s = pyStrip(pyStr(item));
                    if (!s.isEmpty()) {
                        parts.add(s);
                    }
                }
                text = String.join("；", parts);
            } else if (entries instanceof Map<?, ?> map) {
                List<String> parts = new ArrayList<>();
                for (Map.Entry<?, ?> item : map.entrySet()) {
                    String value = pyStr(item.getValue());
                    if (pyStrip(value).isEmpty()) {
                        continue;
                    }
                    parts.add(pyStr(item.getKey()) + "：" + value);
                }
                text = String.join("；", parts);
            } else {
                text = pyStrip(pyStr(entries));
            }
            if (!text.isEmpty()) {
                lines.add("- " + entry.getKey() + "：" + text);
            }
        }
        return String.join("\n", lines);
    }

    // ── 冻结注入段 ───────────────────────────────────────────────────

    /**
     * 组装线程级<b>冻结</b>注入段（T2.1）：画像 + 记忆块 + 会话摘要。
     *
     * <p>以 {@code MEMORY_CONTEXT_FROZEN_PREFIX} 开头——{@code buildModelInput}
     * 据此把它提升到模型输入头部。内容按线程快照冻结：画像/记忆块/摘要的
     * 更新在<b>下个会话</b>生效。这是接受的行为权衡——供应商前缀缓存按最长
     * 公共前缀命中，逐轮变化的头部会让全部历史缓存失效（命中与未命中输入
     * 价差约一个数量级）。
     */
    public static String composeFrozenContext(Map<String, Object> profile,
                                              List<Models.MemoryBlock> blocks,
                                              String summary, int maxChars) {
        List<Section> sections = new ArrayList<>();

        if (profile != null && !profile.isEmpty()) {
            String rendered = renderProfile(profile);
            if (!rendered.isEmpty()) {
                sections.add(new Section("用户画像", rendered));
            }
        }

        List<String> blockLines = new ArrayList<>();
        if (blocks != null) {
            for (Models.MemoryBlock block : blocks) {
                if (!pyStrip(block.content()).isEmpty()) {
                    blockLines.add("[" + block.label() + "] " + pyStrip(block.content()));
                }
            }
        }
        if (!blockLines.isEmpty()) {
            sections.add(new Section("记忆块", String.join("\n", blockLines)));
        }

        if (summary != null && !pyStrip(summary).isEmpty()) {
            sections.add(new Section("会话摘要", pyStrip(summary)));
        }

        return pack(FROZEN_HEADER, sections, maxChars);
    }

    // ── 召回注入段 ───────────────────────────────────────────────────

    /**
     * 组装本轮<b>召回</b>注入段（T2.1）：查询相关的记忆，逐轮变化。
     *
     * <p>以 {@code MEMORY_CONTEXT_PREFIX} 开头（贴尾注入段）：
     * {@code buildModelInput} 把它保留在本轮 human 之前——逐轮变化的内容
     * 放在尾部，只牺牲自己之后的缓存，不动冻结头部与历史体。
     */
    public static String composeRecallContext(List<Retrieval.ScoredMemory> recalled, int maxChars) {
        List<Section> sections = new ArrayList<>();
        if (recalled != null && !recalled.isEmpty()) {
            List<String> lines = new ArrayList<>(recalled.size());
            for (Retrieval.ScoredMemory item : recalled) {
                lines.add("- " + item.record().content());
            }
            sections.add(new Section("相关记忆", String.join("\n", lines)));
        }
        return pack(RECALL_HEADER, sections, maxChars);
    }

    // ── 四合一兼容版 ─────────────────────────────────────────────────

    /**
     * 组装完整注入块（历史契约保留）：画像+记忆块+摘要+相关记忆合一。
     *
     * <p>成本治理（T2.1）后服务端默认改走「冻结段 + 召回段」分离注入
     * （{@link #composeFrozenContext} + {@link #composeRecallContext}）；
     * 本函数保留给需要单块注入的调用方与既有测试。
     */
    public static String composeContext(Map<String, Object> profile,
                                        List<Models.MemoryBlock> blocks, String summary,
                                        List<Retrieval.ScoredMemory> recalled, int maxChars) {
        List<Section> sections = new ArrayList<>();

        if (profile != null && !profile.isEmpty()) {
            String rendered = renderProfile(profile);
            if (!rendered.isEmpty()) {
                sections.add(new Section("用户画像", rendered));
            }
        }

        List<String> blockLines = new ArrayList<>();
        if (blocks != null) {
            for (Models.MemoryBlock block : blocks) {
                if (!pyStrip(block.content()).isEmpty()) {
                    blockLines.add("[" + block.label() + "] " + pyStrip(block.content()));
                }
            }
        }
        if (!blockLines.isEmpty()) {
            sections.add(new Section("记忆块", String.join("\n", blockLines)));
        }

        if (summary != null && !pyStrip(summary).isEmpty()) {
            sections.add(new Section("会话摘要", pyStrip(summary)));
        }

        if (recalled != null && !recalled.isEmpty()) {
            List<String> lines = new ArrayList<>(recalled.size());
            for (Retrieval.ScoredMemory item : recalled) {
                lines.add("- " + item.record().content());
            }
            sections.add(new Section("相关记忆", String.join("\n", lines)));
        }

        return pack(RECALL_HEADER, sections, maxChars);
    }

    // ── 预算收缩 ─────────────────────────────────────────────────────

    /**
     * 渲染注入段；预算超限按「召回逐条丢 → 摘要截断 → 硬截断」收缩。
     *
     * <p>没有任何 section 时返回 null（本轮不注入该段）。
     */
    static String pack(String header, List<Section> sections, int maxChars) {
        if (sections.isEmpty()) {
            return null;
        }

        String renderedAll = render(header, sections);
        if (pyLen(renderedAll) <= maxChars) {
            return renderedAll;
        }

        // 预算超限：先丢相关记忆（得分最低的先丢）。
        int recalledIdx = -1;
        for (int i = 0; i < sections.size(); i++) {
            if (sections.get(i).title().equals("相关记忆")) {
                recalledIdx = i;
                break;
            }
        }
        if (recalledIdx >= 0) {
            Section recalled = sections.get(recalledIdx);
            List<String> lines = new ArrayList<>(splitLines(recalled.body()));
            List<Section> head = sections.subList(0, recalledIdx);
            List<Section> tail = sections.subList(recalledIdx + 1, sections.size());
            List<Section> candidate;
            while (true) {
                // 列表按得分降序，pop() 丢的是得分最低的行；丢光则撤掉整节。
                candidate = new ArrayList<>(head);
                if (!lines.isEmpty()) {
                    candidate.add(new Section(recalled.title(), String.join("\n", lines)));
                }
                candidate.addAll(tail);
                if (pyLen(render(header, candidate)) <= maxChars || lines.isEmpty()) {
                    sections = candidate;
                    break;
                }
                lines.remove(lines.size() - 1);
            }
            renderedAll = render(header, sections);
            if (pyLen(renderedAll) <= maxChars) {
                return renderedAll;
            }
        }

        // 摘要截断。
        List<Section> truncated = new ArrayList<>(sections.size());
        for (Section section : sections) {
            if (section.title().equals("会话摘要") && pyLen(section.body()) > SUMMARY_KEEP_CHARS) {
                truncated.add(new Section(section.title(),
                        pySubstring(section.body(), 0, SUMMARY_KEEP_CHARS) + "……"));
            } else {
                truncated.add(section);
            }
        }
        sections = truncated;
        renderedAll = render(header, sections);
        if (pyLen(renderedAll) <= maxChars) {
            return renderedAll;
        }

        return pySubstring(renderedAll, 0, maxChars);
    }

    /** {@code header + "## 标题\n正文" 以 \n\n 相接}。 */
    private static String render(String header, List<Section> sections) {
        StringBuilder out = new StringBuilder(header);
        for (int i = 0; i < sections.size(); i++) {
            if (i > 0) {
                out.append("\n\n");
            }
            Section section = sections.get(i);
            out.append("## ").append(section.title()).append("\n").append(section.body());
        }
        return out.toString();
    }

    // ── Python 语义的底层工具 ────────────────────────────────────────

    /** Python {@code str(v)} 的常用等价：null→None、布尔→True/False。 */
    private static String pyStr(Object value) {
        if (value == null) {
            return "None";
        }
        if (value instanceof Boolean b) {
            return b ? "True" : "False";
        }
        return String.valueOf(value);
    }

    /** Python {@code str.isspace()} 近似超集（含 NBSP/figure space/narrow NBSP）。 */
    private static boolean isPySpace(int cp) {
        return Character.isWhitespace(cp) || cp == 0x00A0 || cp == 0x2007 || cp == 0x202F;
    }

    /** Python {@code str.strip()}（码点级）。 */
    private static String pyStrip(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int[] cps = s.codePoints().toArray();
        int from = 0;
        while (from < cps.length && isPySpace(cps[from])) {
            from++;
        }
        int to = cps.length;
        while (to > from && isPySpace(cps[to - 1])) {
            to--;
        }
        return new String(cps, from, to - from);
    }

    /** Python {@code len(s)}（码点数）。 */
    private static int pyLen(String s) {
        return s.codePointCount(0, s.length());
    }

    /** Python {@code s[start:end]}（码点级切片）。 */
    private static String pySubstring(String s, int start, int end) {
        int[] cps = s.codePoints().toArray();
        int lo = Math.max(0, start);
        int hi = Math.min(cps.length, end);
        if (lo >= hi) {
            return "";
        }
        return new String(cps, lo, hi - lo);
    }

    /**
     * Python {@code str.splitlines()}（含 CRLF/CR/竖向制表/行分隔符等边界；
     * 结尾边界不产生空尾行）。
     */
    static List<String> splitLines(String s) {
        List<String> lines = new ArrayList<>();
        if (s.isEmpty()) {
            return lines;
        }
        int n = s.length();
        int start = 0;
        int i = 0;
        boolean endedAtBoundary = false;
        while (i < n) {
            char c = s.charAt(i);
            if (c == '\r') {
                lines.add(s.substring(start, i));
                i += (i + 1 < n && s.charAt(i + 1) == '\n') ? 2 : 1;
                start = i;
                endedAtBoundary = true;
            } else if (c == '\n' || c == '\u000B' || c == '\u000C' || c == '\u001C'
                    || c == '\u001D' || c == '\u001E' || c == '\u0085' || c == '\u2028'
                    || c == '\u2029') {
                lines.add(s.substring(start, i));
                i++;
                start = i;
                endedAtBoundary = true;
            } else {
                i++;
                endedAtBoundary = false;
            }
        }
        String tail = s.substring(start);
        if (!(endedAtBoundary && tail.isEmpty())) {
            lines.add(tail);
        }
        return lines;
    }
}
