package io.github.paopaonb666.agentbase.memory.textkit;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * 记忆侧的纯文本处理工具（P0-2 拆分）：分块与画像截断。
 *
 * <p>这些是无状态的纯函数，与"记忆服务"对象没有依赖关系——拆出来供
 * 知识库摄取、切片可视化与画像保存复用，也便于独立单测。
 *
 * <p>对应 Python 侧 {@code agent_base_memory/textkit.py}，逐算法移植。
 * 所有字符坐标与切片长度均按 Unicode 码点计数（与 Python {@code len()}
 * 一致），内部先转码点数组再运算。
 */
public final class TextKit {

    private TextKit() {}

    private static final ObjectMapper JSON = new ObjectMapper();

    /**
     * 一个切片：内容文本 + 在原文中的字符区间（前端高亮边界的依据）。
     *
     * <p>{@code segments} 是 {@code ((start, end), …)} 列表（每项
     * {@code long[]{start, end}}），坐标相对<b>调用方传入的原文</b>
     * （含 strip 偏移补偿）。段落打包块的文本由多段拼成，单一
     * {@code (start, end)} 不成立，因此是区间列表；固定窗口切片是单区间。
     */
    public record ChunkSpan(String text, List<long[]> segments) {
        public ChunkSpan {
            segments = segments == null ? List.of() : List.copyOf(segments);
        }

        /** 对应 Python {@code ChunkSpan.char_len}。 */
        public int charLen() {
            return text.codePointCount(0, text.length());
        }
    }

    // ── 画像截断 ─────────────────────────────────────────────────────

    /**
     * 画像结构级硬截断：从最后一个分节倒序逐条丢弃条目直到序列化
     * 长度达标；仍超长（存在巨型单条）则截断字符串值。输出永远是合法
     * JSON——绝不让画像把注入预算缓慢撑爆。
     *
     * <p>与 Python 的一处刻意分歧：Python 版在"所有字符串都已截到
     * {@code max_chars//2} 仍超长"时会因重复赋同一值而死循环；这里
     * 检测到无法再推进时转入极端兜底 {@code {"truncated": true}}（
     * Python 注释声明的本意即"绝不抛错"的兜底，Java 版让它真正可达）。
     * 在 Python 能终止的所有输入上行为逐字节一致。
     */
    public static Map<String, Object> trimProfileToChars(Map<String, Object> profile, int maxChars) {
        Map<String, Object> trimmed = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : profile.entrySet()) {
            Object value = e.getValue();
            if (value instanceof List<?> list) {
                trimmed.put(e.getKey(), new ArrayList<>(list));
            } else if (value instanceof Map<?, ?> map) {
                trimmed.put(e.getKey(), new LinkedHashMap<>(map));
            } else {
                trimmed.put(e.getKey(), value);
            }
        }

        while (jsonLength(trimmed) > maxChars) {
            String lastKey = null;
            List<String> keys = new ArrayList<>(trimmed.keySet());
            for (int i = keys.size() - 1; i >= 0; i--) {
                String key = keys.get(i);
                if (trimmed.get(key) instanceof List<?> list && !list.isEmpty()) {
                    lastKey = key;
                    break;
                }
            }
            if (lastKey == null) {
                break;
            }
            @SuppressWarnings("unchecked")
            List<Object> list = (List<Object>) trimmed.get(lastKey);
            list.remove(list.size() - 1);
            if (list.isEmpty()) {
                trimmed.remove(lastKey);
            } else {
                trimmed.put(lastKey, list);
            }
        }
        while (jsonLength(trimmed) > maxChars) {
            String strKey = null;
            for (Map.Entry<String, Object> e : trimmed.entrySet()) {
                if (e.getValue() instanceof String s && !s.isEmpty()) {
                    strKey = e.getKey();
                    break;
                }
            }
            if (strKey == null) {
                // 只剩数字/空结构的极端兜底：几乎不可达，但绝不抛错。
                return truncatedFallback();
            }
            int keep = Math.max(1, Math.floorDiv(maxChars, 2));
            String current = (String) trimmed.get(strKey);
            if (codePointLength(current) <= keep) {
                // 截无可截仍超长（其它键占满预算）——转入极端兜底。
                return truncatedFallback();
            }
            trimmed.put(strKey, substringCodePoints(current, 0, keep));
        }
        return trimmed;
    }

    /** 极端兜底：{@code {"truncated": true}}（对应 Python 的返回值）。 */
    private static Map<String, Object> truncatedFallback() {
        Map<String, Object> fallback = new LinkedHashMap<>();
        fallback.put("truncated", Boolean.TRUE);
        return fallback;
    }

    /** 对应 Python {@code len(json.dumps(trimmed, ensure_ascii=False))}。 */
    private static int jsonLength(Map<String, Object> value) {
        try {
            String json = JSON.writeValueAsString(value);
            return json.codePointCount(0, json.length());
        } catch (Exception e) {
            // 序列化失败（不可编码对象等）：视为超长，继续收缩。绝不抛错。
            return Integer.MAX_VALUE;
        }
    }

    // ── 文本切块 ─────────────────────────────────────────────────────

    /**
     * 把文档文本切块（边界加固：段落感知优先，固定窗口兜底）。
     *
     * <p>策略：
     * <ul>
     * <li>优先按空行分段、贪心把<b>完整段落</b>打包进 ≤ chunk_chars 的块——
     *     切块边界落在段落上，表格/列表/代码块不再被从中间劈开；</li>
     * <li>单段超过 chunk_chars 时段内退回固定窗口滑动（带 overlap）；</li>
     * <li>全文没有段落分隔（或只有一段）时行为与旧版固定窗口完全一致
     *     （向后兼容：纯文本长文切块结果不变）。</li>
     * </ul>
     *
     * <p>返回 span：除块文本外还带原文坐标，供前端把切片边界高亮回原文
     * （切片可视化）；坐标相对传入的 text 原始串（strip 前的完整范围）。
     *
     * <p>保证：任何块不超过 chunk_chars；非空内容全部覆盖；空文本返回空列表。
     */
    public static List<ChunkSpan> chunkText(String text, int chunkChars, int overlap) {
        int[] all = text.codePoints().toArray();
        int leading = 0;
        while (leading < all.length && isPySpace(all[leading])) {
            leading++;
        }
        int end = all.length;
        while (end > leading && isPySpace(all[end - 1])) {
            end--;
        }
        int[] stripped = Arrays.copyOfRange(all, leading, end);
        if (stripped.length == 0) {
            return List.of();
        }
        if (stripped.length <= chunkChars) {
            return List.of(new ChunkSpan(fromCodePoints(stripped, 0, stripped.length),
                    List.of(new long[]{leading, leading + stripped.length})));
        }

        // 段落切分带坐标：等价于 re.finditer(r"\n\s*\n", stripped) 的分段。
        List<Para> paragraphs = splitParagraphs(stripped);
        if (paragraphs.size() <= 1) {
            return window(stripped, stripped.length, leading, chunkChars, overlap);
        }

        List<ChunkSpan> chunks = new ArrayList<>();
        List<Para> current = new ArrayList<>();
        int currentLen = 0; // 已装段落总长（含段间空行分隔符）
        for (Para para : paragraphs) {
            if (para.cps().length > chunkChars) {
                // 超长段落：先结算已打包内容，段内固定窗口切。
                if (!current.isEmpty()) {
                    chunks.add(joinSpans(current, leading));
                    current = new ArrayList<>();
                    currentLen = 0;
                }
                chunks.addAll(window(para.cps(), para.cps().length, leading + para.pos(),
                        chunkChars, overlap));
                continue;
            }
            int extra = para.cps().length + (current.isEmpty() ? 0 : 2);
            if (currentLen + extra <= chunkChars) {
                current.add(para);
                currentLen += extra;
            } else {
                chunks.add(joinSpans(current, leading));
                current = new ArrayList<>();
                current.add(para);
                currentLen = para.cps().length;
            }
        }
        if (!current.isEmpty()) {
            chunks.add(joinSpans(current, leading));
        }
        return List.copyOf(chunks);
    }

    /** 单个待打包段落：内容（码点）+ 在 stripped 文本中的起始码点偏移。 */
    private record Para(int[] cps, int pos) {}

    /**
     * 段落切分带坐标：等价于 Python 对 stripped 文本跑
     * {@code re.finditer(r"\n\s*\n", text)} 后逐段收集
     * {@code (segment.strip(), last + 段首空白宽)}。
     *
     * <p>正则 {@code \n\s*\n} 的贪心回溯语义（\s* 吃掉整个空白 run 后
     * 回退到最后一个 \n）用显式扫描复现：匹配终点是该空白 run 中
     * 最后一个 {@code \n} 之后。
     */
    private static List<Para> splitParagraphs(int[] text) {
        List<Para> paragraphs = new ArrayList<>();
        int last = 0;
        int i = 0;
        while (i < text.length) {
            if (text[i] != '\n') {
                i++;
                continue;
            }
            // 空白 run [i+1, runEnd)：\s 含 \n 与其它空白。
            int runEnd = i + 1;
            while (runEnd < text.length && isPySpace(text[runEnd])) {
                runEnd++;
            }
            // 回退：找 run 内最后一个 \n（不含起点 i 自身）。
            int matchEnd = -1;
            for (int j = runEnd - 1; j > i; j--) {
                if (text[j] == '\n') {
                    matchEnd = j + 1;
                    break;
                }
            }
            if (matchEnd >= 0) {
                appendSegment(paragraphs, text, last, i);
                last = matchEnd;
                i = matchEnd;
            } else {
                i++;
            }
        }
        appendSegment(paragraphs, text, last, text.length);
        return paragraphs;
    }

    /** 收集 stripped[last:segmentEnd) 为一个段落（等 Python 循环体）。 */
    private static void appendSegment(List<Para> paragraphs, int[] text, int last, int segmentEnd) {
        int len = segmentEnd - last;
        if (len <= 0) {
            return;
        }
        int from = last;
        while (from < segmentEnd && isPySpace(text[from])) {
            from++;
        }
        int to = segmentEnd;
        while (to > from && isPySpace(text[to - 1])) {
            to--;
        }
        if (to > from) {
            paragraphs.add(new Para(Arrays.copyOfRange(text, from, to), last + (from - last)));
        }
    }

    /**
     * 固定窗口滑动切片（step = max(1, chunk_chars - overlap)），
     * {@code base} 为该段在原文中的起始偏移（含 leading 补偿）。
     */
    private static List<ChunkSpan> window(int[] block, int blockLen, int base,
                                          int chunkChars, int overlap) {
        int step = Math.max(1, chunkChars - overlap);
        List<ChunkSpan> spans = new ArrayList<>();
        for (int start = 0; start < blockLen; start += step) {
            int end = Math.min(start + chunkChars, blockLen);
            spans.add(new ChunkSpan(fromCodePoints(block, start, end - start),
                    List.of(new long[]{base + start, base + end})));
        }
        return spans;
    }

    /**
     * 把打包进同一块的若干段落合成一个 span：文本以 {@code \n\n} 相接，
     * segments 保留各段在原文中的真实区间（高亮时按段渲染）。
     */
    private static ChunkSpan joinSpans(List<Para> paragraphs, int leading) {
        StringBuilder joined = new StringBuilder();
        List<long[]> segments = new ArrayList<>();
        for (int i = 0; i < paragraphs.size(); i++) {
            Para para = paragraphs.get(i);
            if (i > 0) {
                joined.append("\n\n");
            }
            joined.append(fromCodePoints(para.cps(), 0, para.cps().length));
            segments.add(new long[]{leading + para.pos(), leading + para.pos() + para.cps().length});
        }
        return new ChunkSpan(joined.toString(), segments);
    }

    // ── Python 语义的底层工具 ────────────────────────────────────────

    /**
     * Python {@code str.isspace()} 的近似超集判定（str.strip 无参语义）：
     * Java 的 {@link Character#isWhitespace} 不含 NBSP/figure space/narrow
     * NBSP，这里补齐三处差异。
     */
    private static boolean isPySpace(int cp) {
        return Character.isWhitespace(cp) || cp == 0x00A0 || cp == 0x2007 || cp == 0x202F;
    }

    /** 码点级子串（对应 Python {@code s[start:end]}）。 */
    static String substringCodePoints(String s, int start, int end) {
        int[] cps = s.codePoints().toArray();
        int lo = Math.max(0, start);
        int hi = Math.min(cps.length, end);
        if (lo >= hi) {
            return "";
        }
        return fromCodePoints(cps, lo, hi - lo);
    }

    /** 码点计数长度（对应 Python {@code len(s)}）。 */
    static int codePointLength(String s) {
        return s.codePointCount(0, s.length());
    }

    private static String fromCodePoints(int[] cps, int from, int count) {
        return new String(cps, from, count);
    }
}
