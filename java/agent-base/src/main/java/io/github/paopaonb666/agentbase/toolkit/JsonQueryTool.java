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

    /** 给模型看的说明书（逐字复制 Python 版 docstring）。 */
    private static final String DESCRIPTION = """
            查询 JSON 文本中的值。path 为空时返回结构概要（顶层键或元素数）；
            否则按 ``a.b[0].c`` 形式的点/下标路径提取子值。键名不支持含点号。

            Args:
                json_text: 要查询的 JSON 文本（对象或数组）。
                path: 可选的取值路径，如 "data.items[0].name"；留空返回概要。
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

    // ── 路径解析（a.b[0][1] 形式；键名不支持含点号） ──────────────────

    private static JsonNode resolve(JsonNode document, String path, String origPath) {
        JsonNode current = document;
        for (String segment : path.split("\\.")) {
            String token = segment.strip();
            if (token.isEmpty()) {
                throw new IllegalArgumentException(
                        "路径 " + PyText.repr(origPath) + " 含空段（连续的点号？）");
            }
            String[] parts = token.split("\\[", -1);
            String key = parts[0];
            if (!key.isEmpty()) {
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
                current = current.get(key);
            }
            for (int p = 1; p < parts.length; p++) {
                String raw = parts[p];
                if (!raw.endsWith("]")) {
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 的下标 " + PyText.repr(raw) + " 缺少右括号");
                }
                String indexText = raw.substring(0, raw.length() - 1);
                if (!indexText.matches("\\d+")) {
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 的下标 " + PyText.repr(indexText)
                                    + " 不是非负整数");
                }
                if (!current.isArray()) {
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 在 [" + indexText + "] 处失败：当前是 "
                                    + typeName(current) + "，不可下标");
                }
                int index;
                try {
                    index = Integer.parseInt(indexText);
                } catch (NumberFormatException e) {
                    // Python 的 int 是任意精度，超大下标会自然落入"越界"分支。
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 的下标 [" + indexText + "] 越界：列表长度 "
                                    + current.size());
                }
                if (index >= current.size()) {
                    throw new IllegalArgumentException(
                            "路径 " + PyText.repr(origPath) + " 的下标 [" + index + "] 越界：列表长度 "
                                    + current.size());
                }
                current = current.get(index);
            }
        }
        return current;
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
