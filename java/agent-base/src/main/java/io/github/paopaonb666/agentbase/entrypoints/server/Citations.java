package io.github.paopaonb666.agentbase.entrypoints.server;

import io.github.paopaonb666.agentbase.extensions.events.Events;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 回复引用解析（#338，与 #254 的序号格式配套）：模型按约定在最终回复里
 * 用 {@code [n]} 标注来源序号，这里把它解析回 sources 事件载荷。
 *
 * <p>容错口径（裁决原文「不解析则不 emit」）：回复里没有 {@code [n]}
 * 锚就不发事件；序号超出本轮来源登记簿的也不发对应条目。解析只用
 * {@code [\d+]} 形态且前面不是 ASCII 标识符字符——Markdown 链接
 * {@code [文字](url)}、代码数组下标 {@code arr[1]} 天然不匹配。
 */
public final class Citations {

    /**
     * {@code [n]} 引用锚（1..3 位数字；三位已远超单轮来源数）。负向前瞻
     * 排除代码标识符里的数组下标（{@code arr[1]} 前面是字母数字）——中文
     * 与标点前缀的 {@code 是[1]} 仍算引用。
     */
    private static final Pattern CITATION = Pattern.compile("(?<![A-Za-z0-9_])\\[(\\d{1,3})]");

    private Citations() {
    }

    /** 从回复文本解析引用序号（升序去重；无锚返回空表）。 */
    public static List<Integer> parse(String text) {
        List<Integer> out = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return out;
        }
        Matcher matcher = CITATION.matcher(text);
        while (matcher.find()) {
            int n = Integer.parseInt(matcher.group(1));
            if (n > 0 && !out.contains(n)) {
                out.add(n);
            }
        }
        return out;
    }

    /**
     * 来源登记簿去重（title+url 键，保首次出现序）：同一轮多个搜索调用
     * 可能重复报同一来源。
     */
    public static List<Events.Source> dedup(List<Events.Source> sources) {
        if (sources == null || sources.isEmpty()) {
            return List.of();
        }
        Map<String, Events.Source> seen = new LinkedHashMap<>();
        for (Events.Source source : sources) {
            if (source == null) {
                continue;
            }
            seen.putIfAbsent(source.title() + "\u0000" + source.url(), source);
        }
        return List.copyOf(seen.values());
    }

    /**
     * 把回复里的 {@code [n]} 解析为登记簿条目。登记簿序 = 本轮各来源事件
     * 的收集序（工具按序发射、去重保首现）；越界序号静默跳过（容错），
     * 一个都解析不出来时返回空表（调用方据此不 emit）。
     */
    public static List<Events.Source> resolve(String replyText, List<Events.Source> registry) {
        List<Integer> numbers = parse(replyText);
        if (numbers.isEmpty() || registry == null || registry.isEmpty()) {
            return List.of();
        }
        List<Events.Source> out = new ArrayList<>();
        for (int n : numbers) {
            if (n <= registry.size()) {
                out.add(registry.get(n - 1));
            }
        }
        return List.copyOf(out);
    }
}
