package io.github.paopaonb666.agentbase.toolkit;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;

import io.github.paopaonb666.agentbase.core.tools.AgentTool;

/**
 * {@code json_query} 工具：查询 JSON 文本中的值（零依赖基础工具之一，
 * 对应 Python 侧 {@code tools/basic.py} 的 {@code json_query}）。
 *
 * <p>查询语法与 Python 版同一子集：{@code a.b[0][1].c} 形式的点/下标
 * 路径（键名不支持含点号、不支持负下标、不支持引号键与通配符）；
 * path 为空时返回结构概要（顶层键或元素数）。解析用 Jackson
 * {@link JsonNode}（对应 Python 的 {@code json.loads}），只读访问。
 *
 * <p>输出对齐 {@code json.dumps(value, ensure_ascii=False)}：对象/数组
 * 用默认分隔符 {@code ", "} / {@code ": "}（带空格的紧凑形式），非
 * ASCII 原样保留；超出 4000 字符截断并加"…（已截断）"标记。
 *
 * <p>已声明差异：JSON 解析失败的行内详情取自 Jackson（如
 * "Unrecognized token 'x'"），措辞与 Python json 模块（"Expecting
 * value" 一类）不同，行号语义一致。
 */
public final class JsonQueryTool implements AgentTool {

    public static final String NAME = "json_query";

    /** json_query 返回体的长度封顶：查询结果可能是巨大的子树，超长输出
     * 既浪费上下文也没有可操作性（对应 Python _JSON_RESULT_MAX_CHARS）。 */
    static final int RESULT_MAX_CHARS = 4000;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** 给模型看的说明书（在 Python 版 docstring 基础上补 #246 语法扩展
     * 与三段式「何时不用」——#24 起与 Python 不再逐字对齐，差异登记项）。 */
    private static final String DESCRIPTION = """
            查询 JSON 文本中的值。path 为空时返回结构概要（顶层键或元素数）；
            否则按 ``a.b[0].c`` 形式的点/下标路径提取子值。支持：负下标取尾
            元素（``items[-1]`` = 最后一个）；含点号/特殊字符的键用引号
            （``a["b.c"]`` 或 ``a["b.c"][0]``）。不支持通配符。

            何时不用：文本不是合法 JSON、或需要改写/过滤整个结构时不要用
            ——本工具只读取值，无变换能力。

            Args:
                json_text: 要查询的 JSON 文本（对象或数组）。
                path: 可选的取值路径，如 "data.items[0].name" 或
                    "items[-1]" 或 'a["b.c"]'；留空返回概要。
            """.stripTrailing();

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public String description() {
        return DESCRIPTION;
    }

    @Override
    public ToolSpecification specification() {
        return ToolSpecification.builder()
                .name(NAME)
                .description(DESCRIPTION)
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("json_text", "要查询的 JSON 文本（对象或数组）。")
                        .addStringProperty("path",
                                "可选的取值路径，如 \"data.items[0].name\"；留空返回概要。")
                        .required("json_text")
                        .build())
                .build();
    }

    @Override
    public String call(String argsJson, ToolContext ctx) {
        JsonNode args = parseArgs(argsJson);
        JsonNode textNode = args.get("json_text");
        if (textNode == null || textNode.isNull()) {
            throw new IllegalArgumentException("缺少必需参数 json_text：要查询的 JSON 文本");
        }
        String jsonText = textNode.asText();
        JsonNode pathNode = args.get("path");
        String path = pathNode == null || pathNode.isNull() ? "" : pathNode.asText();

        JsonNode document;
        try {
            document = MAPPER.readTree(jsonText);
        } catch (JsonProcessingException e) {
            int line = e.getLocation() != null ? e.getLocation().getLineNr() : 1;
            throw new IllegalArgumentException(
                    "JSON 解析失败（第 " + line + " 行）：" + e.getOriginalMessage());
        }

        if (path.strip().isEmpty()) {
            return summarize(document);
        }

        JsonNode value = resolve(document, path.strip(), path);
        String rendered = dumps(value);
        if (rendered.length() > RESULT_MAX_CHARS) {
            rendered = rendered.substring(0, RESULT_MAX_CHARS) + "…（已截断）";
        }
        return rendered;
    }

    // ── 结构概要（path 为空） ────────────────────────────────────────

    private static String summarize(JsonNode document) {
        if (document.isObject()) {
            List<String> keys = new ArrayList<>();
            int shown = 0;
            for (Iterator<String> it = document.fieldNames(); it.hasNext() && shown < 50; shown++) {
                keys.add(it.next());
            }
            return "JSON 对象，" + document.size() + " 个键：" + String.join(", ", keys);
        }
        if (document.isArray()) {
            return "JSON 数组，" + document.size() + " 个元素；用下标路径（如 '[0]'）访问";
        }
        return "JSON 标量：" + scalarRepr(document);
    }

    /** 标量概要用 Python repr 语义：字符串带引号，None/True/False 用 Python 名。 */
    private static String scalarRepr(JsonNode node) {
        if (node.isNull()) {
            return "None";
        }
        if (node.isBoolean()) {
            return node.booleanValue() ? "True" : "False";
        }
        if (node.isTextual()) {
            return PyText.repr(node.textValue());
        }
        if (node.isNumber() && node.isFloatingPointNumber()) {
            return PyText.floatRepr(node.doubleValue());
        }
        return node.asText();
    }

    // ── 路径解析（a.b[0][1]、负下标 [-1]、引号键 a["b.c"]；#246） ─────

    private static JsonNode resolve(JsonNode document, String path, String origPath) {
        JsonNode current = document;
        for (String segment : splitSegments(path, origPath)) {
            List<String> steps = parseSegment(segment, origPath);
            for (String step : steps) {
                if (step.startsWith("[")) {
                    current = applyBracket(current, step, origPath);
                } else {
                    current = applyKey(current, step, origPath);
                }
            }
        }
        return current;
    }

    /** 按非引号内的点切段（引号键里的点不分段）；空段报错。 */
    private static List<String> splitSegments(String path, String origPath) {
        List<String> segments = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuote = false;
        for (int i = 0; i < path.length(); i++) {
            char c = path.charAt(i);
            if (c == '"') {
                inQuote = !inQuote;
                current.append(c);
            } else if (c == '.' && !inQuote) {
                String seg = current.toString().strip();
                if (seg.isEmpty()) {
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 含空段（连续的点号？）");
                }
                segments.add(seg);
                current.setLength(0);
            } else {
                current.append(c);
            }
        }
        if (inQuote) {
            throw new IllegalArgumentException(
                    "路径 " + PyText.repr(origPath) + " 的引号键缺少右引号");
        }
        String seg = current.toString().strip();
        if (!seg.isEmpty()) {
            segments.add(seg);
        } else if (segments.isEmpty()) {
            throw new IllegalArgumentException(
                    "路径 " + PyText.repr(origPath) + " 含空段（连续的点号？）");
        }
        // 尾随点容忍（"a." 等价 "a"，与旧 split 行为一致）
        return segments;
    }

    /**
     * 段内解析：可选的前导键 + 若干 {@code [...]} 步。{@code [...]} 内是
     * 非负/负整数下标或引号键；产出 token：下标步保留 {@code [...]} 形态，
     * 引号键与裸键产出键名。
     */
    private static List<String> parseSegment(String segment, String origPath) {
        List<String> steps = new ArrayList<>();
        int i = 0;
        int n = segment.length();
        while (i < n) {
            char c = segment.charAt(i);
            if (c == '[') {
                int close = segment.indexOf(']', i);
                if (close < 0) {
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 的下标 " + PyText.repr(
                                    segment.substring(i)) + " 缺少右括号");
                }
                steps.add(segment.substring(i, close + 1));
                i = close + 1;
            } else if (c == '"') {
                int close = segment.indexOf('"', i + 1);
                if (close < 0) {
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 的引号键缺少右引号");
                }
                steps.add(segment.substring(i + 1, close));
                i = close + 1;
                if (i < n && segment.charAt(i) != '[') {
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 的引号键 " + PyText.repr(
                                    steps.get(steps.size() - 1)) + " 后应跟 [ 或段结束");
                }
            } else {
                int next = i;
                while (next < n && segment.charAt(next) != '[' && segment.charAt(next) != '"') {
                    next++;
                }
                String key = segment.substring(i, next).strip();
                if (!key.isEmpty()) {
                    steps.add(key);
                }
                i = next;
            }
        }
        if (steps.isEmpty()) {
            throw new IllegalArgumentException(
                    "路径 " + PyText.repr(origPath) + " 含空段（连续的点号？）");
        }
        return steps;
    }

    /** 应用一步键访问（裸键或引号键），错误信息与旧版同款。 */
    private static JsonNode applyKey(JsonNode current, String key, String origPath) {
        boolean isObject = current.isObject();
        if (!isObject || !current.has(key)) {
            if (isObject) {
                // 可用键最多提示 20 个，避免巨型对象的错误信息刷屏。
                List<String> keys = new ArrayList<>();
                int shown = 0;
                for (Iterator<String> it = current.fieldNames(); it.hasNext() && shown < 20; shown++) {
                    keys.add(it.next());
                }
                throw new IllegalArgumentException(
                        "路径 " + PyText.repr(origPath) + " 在键 " + PyText.repr(key)
                                + " 处失败：当前是 dict，可用键：" + String.join(", ", keys));
            }
            throw new IllegalArgumentException(
                    "路径 " + PyText.repr(origPath) + " 在键 " + PyText.repr(key)
                            + " 处失败：当前是 " + typeName(current));
        }
        return current.get(key);
    }

    /**
     * 应用一步 {@code [...]}：非负整数 → 数组下标；负整数 → Python 语义的
     * 倒数下标（[-1] = 尾元素）；引号键 → dict 访问（含点号键可达）。
     */
    private static JsonNode applyBracket(JsonNode current, String step, String origPath) {
        String inner = step.substring(1, step.length() - 1).strip();
        if (inner.startsWith("\"") && inner.endsWith("\"") && inner.length() >= 2) {
            return applyKey(current, inner.substring(1, inner.length() - 1), origPath);
        }
        if (!inner.matches("-?\\d+")) {
            throw new IllegalArgumentException(
                    "路径 " + PyText.repr(origPath) + " 的下标 " + PyText.repr(inner)
                            + " 不是整数或引号键");
        }
        if (!current.isArray()) {
            throw new IllegalArgumentException(
                    "路径 " + PyText.repr(origPath) + " 在 [" + inner + "] 处失败：当前是 "
                            + typeName(current) + "，不可下标");
        }
        int size = current.size();
        int index;
        try {
            index = Integer.parseInt(inner);
        } catch (NumberFormatException e) {
            // Python 的 int 是任意精度，超大下标会自然落入"越界"分支。
            throw new IllegalArgumentException(
                    "路径 " + PyText.repr(origPath) + " 的下标 [" + inner + "] 越界：列表长度 "
                            + size);
        }
        if (index < 0) {
            index += size; // Python 负下标语义：[-1] = 尾元素
        }
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "路径 " + PyText.repr(origPath) + " 的下标 [" + inner + "] 越界：列表长度 "
                            + size + "（负下标有效范围 [-" + size + "..-1]）");
        }
        return current.get(index);
    }

    /** 节点的 Python 类型名（错误信息里出现，与 type(x).__name__ 对齐）。 */
    private static String typeName(JsonNode node) {
        if (node.isObject()) {
            return "dict";
        }
        if (node.isArray()) {
            return "list";
        }
        if (node.isTextual()) {
            return "str";
        }
        if (node.isBoolean()) {
            return "bool";
        }
        if (node.isNull()) {
            return "NoneType";
        }
        if (node.isNumber()) {
            return node.isFloatingPointNumber() ? "float" : "int";
        }
        return "object";
    }

    // ── 渲染（json.dumps(value, ensure_ascii=False) 的等价物） ───────

    static String dumps(JsonNode node) {
        if (node.isNull()) {
            return "null";
        }
        if (node.isBoolean()) {
            return node.booleanValue() ? "true" : "false";
        }
        if (node.isNumber()) {
            if (node.isFloatingPointNumber()) {
                double d = node.doubleValue();
                // json.dumps 对 inf/nan 输出 "Infinity"/"NaN"（allow_nan 默认开）。
                if (Double.isNaN(d)) {
                    return "NaN";
                }
                if (d == Double.POSITIVE_INFINITY) {
                    return "Infinity";
                }
                if (d == Double.NEGATIVE_INFINITY) {
                    return "-Infinity";
                }
                return PyText.floatRepr(d);
            }
            return node.asText();
        }
        if (node.isTextual()) {
            return jsonString(node.textValue());
        }
        if (node.isArray()) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < node.size(); i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(dumps(node.get(i)));
            }
            return sb.append("]").toString();
        }
        if (node.isObject()) {
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Iterator<Map.Entry<String, JsonNode>> it = node.fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> e = it.next();
                if (!first) {
                    sb.append(", ");
                }
                first = false;
                sb.append(jsonString(e.getKey())).append(": ").append(dumps(e.getValue()));
            }
            return sb.append("}").toString();
        }
        return "null";
    }

    /** ensure_ascii=False 的字符串转义：仅引号、反斜杠与控制字符。 */
    private static String jsonString(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 2);
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.append('"').toString();
    }

    private static JsonNode parseArgs(String argsJson) {
        try {
            return MAPPER.readTree(argsJson);
        } catch (Exception e) {
            throw new IllegalArgumentException("工具参数不是合法 JSON：" + e.getMessage());
        }
    }
}
