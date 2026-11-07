package io.github.paopaonb666.agentbase.core.text;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 检索查询分词（#244 从记忆包 Retrieval 提为基座公共件）：ASCII 词
 * （小写）+ 中文二元组。
 *
 * <p>中文无空格分词，二元组是无词典检索的实用折中——"用户偏好"产出
 * {@code 用户}/{@code 户偏}/{@code 偏好}，足以命中改写查询。搜索相关性
 * 评分（SearchPipeline）与记忆 BM25（agent-base-memory Retrieval）共用
 * 本实现，保证两侧对「什么算一个词项」的口径一致。
 */
public final class QueryTerms {

    private static final Pattern ASCII_WORD_RE = Pattern.compile("[a-zA-Z0-9_]+");
    private static final Pattern CJK_RUN_RE = Pattern.compile("[\\x{4E00}-\\x{9FFF}]+");

    private QueryTerms() {
    }

    /** ASCII 词（小写）+ CJK 连续段二元组；单字 CJK 段保留为单词项。 */
    public static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        Matcher ascii = ASCII_WORD_RE.matcher(text);
        while (ascii.find()) {
            tokens.add(ascii.group().toLowerCase(Locale.ROOT));
        }
        Matcher cjk = CJK_RUN_RE.matcher(text);
        while (cjk.find()) {
            int[] run = cjk.group().codePoints().toArray();
            if (run.length == 1) {
                tokens.add(new String(run, 0, 1));
                continue;
            }
            for (int i = 0; i + 2 <= run.length; i++) {
                tokens.add(new String(run, i, 2));
            }
        }
        return List.copyOf(tokens);
    }
}
