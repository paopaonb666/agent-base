package io.github.paopaonb666.agentbase.toolkit.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchemaElement;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;

import java.util.ArrayList;
import java.util.List;

/**
 * MCP inputSchema（JSON Schema）→ langchain4j {@link JsonSchemaElement} 的
 * 保守转换。MCP server 用的是标准 JSON Schema 的一个子集；这里支持常见
 * 形态（string/integer/number/boolean/enum/array/object 的直接嵌套），
 * 未识别形态一律降级为 string——模型侧看到的是「宽松但可用」的参数面，
 * 不会因 server 用了冷门 JSON Schema 关键字而丢工具。
 *
 * <p>嵌套深度封顶 {@value #MAX_DEPTH}（JSON Schema 允许任意递归，模型
 * 工具面不需要更深）。
 */
public final class McpSchemas {

    private static final int MAX_DEPTH = 4;

    private McpSchemas() {
    }

    /** 把 tools/list 的 inputSchema 转成对象参数面；null/非对象按空对象处理。 */
    public static JsonObjectSchema toObjectSchema(JsonNode inputSchema) {
        JsonSchemaElement element = convert(inputSchema, 0);
        return element instanceof JsonObjectSchema obj ? obj : JsonObjectSchema.builder().build();
    }

    private static JsonSchemaElement convert(JsonNode schema, int depth) {
        if (schema == null || !schema.isObject() || depth > MAX_DEPTH) {
            return JsonObjectSchema.builder().build();
        }
        String type = schema.path("type").asText("");
        String description = schema.path("description").asText(null);
        switch (type) {
            case "string" -> {
                JsonNode enumValues = schema.get("enum");
                if (enumValues != null && enumValues.isArray() && !enumValues.isEmpty()) {
                    List<String> values = new ArrayList<>();
                    enumValues.forEach(v -> values.add(v.asText()));
                    JsonEnumSchema.Builder builder = JsonEnumSchema.builder().enumValues(values);
                    if (description != null) {
                        builder.description(description);
                    }
                    return builder.build();
                }
                return withString(description);
            }
            case "integer" -> {
                JsonIntegerSchema.Builder builder = JsonIntegerSchema.builder();
                if (description != null) {
                    builder.description(description);
                }
                return builder.build();
            }
            case "number" -> {
                JsonNumberSchema.Builder builder = JsonNumberSchema.builder();
                if (description != null) {
                    builder.description(description);
                }
                return builder.build();
            }
            case "boolean" -> {
                JsonBooleanSchema.Builder builder = JsonBooleanSchema.builder();
                if (description != null) {
                    builder.description(description);
                }
                return builder.build();
            }
            case "array" -> {
                JsonArraySchema.Builder builder = JsonArraySchema.builder()
                        .items(convert(schema.get("items"), depth + 1));
                if (description != null) {
                    builder.description(description);
                }
                return builder.build();
            }
            case "object" -> {
                return convertObject(schema, depth);
            }
            default -> {
                // 无 type / union / $ref 等冷门形态：string 兜底
                return withString(description);
            }
        }
    }

    private static JsonObjectSchema convertObject(JsonNode schema, int depth) {
        JsonObjectSchema.Builder builder = JsonObjectSchema.builder();
        String description = schema.path("description").asText(null);
        if (description != null) {
            builder.description(description);
        }
        JsonNode properties = schema.get("properties");
        if (properties != null && properties.isObject()) {
            properties.properties().forEach(prop ->
                    builder.addProperty(prop.getKey(), convert(prop.getValue(), depth + 1)));
        }
        JsonNode required = schema.get("required");
        if (required != null && required.isArray()) {
            List<String> names = new ArrayList<>();
            required.forEach(n -> {
                if (n.isTextual()) {
                    names.add(n.asText());
                }
            });
            if (!names.isEmpty()) {
                builder.required(names);
            }
        }
        return builder.build();
    }

    private static JsonStringSchema withString(String description) {
        JsonStringSchema.Builder builder = JsonStringSchema.builder();
        if (description != null) {
            builder.description(description);
        }
        return builder.build();
    }
}
