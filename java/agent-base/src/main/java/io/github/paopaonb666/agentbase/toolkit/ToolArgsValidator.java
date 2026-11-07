package io.github.paopaonb666.agentbase.toolkit;

import com.fasterxml.jackson.databind.JsonNode;
import dev.langchain4j.model.chat.request.json.JsonAnyOfSchema;
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
import java.util.Map;

/**
 * 工具参数 JSON Schema 校验（#334）：模型给的 argsJson 在执行前对
 * ToolSpecification 的参数面做一次轻校验，失败短路（不进工具线程），
 * 回喂结构化错误让模型一轮自愈（#335 契约，见 {@link #contractText}）。
 *
 * <p>langchain4j 1.4 只有 schema 建模没有校验器，这里按保守口径自实现：
 * 校验 required/类型/enum/数组项/一层嵌套对象；<b>未声明 properties 的
 * 对象放行任意键</b>（零参工具不因模型多给的键而失败）、未知 schema
 * 形态（anyOf/reference）放行——宁可漏报不可误杀。
 */
public final class ToolArgsValidator {

    private ToolArgsValidator() {
    }

    /**
     * 校验 argsJson 对 schema 的合法性；返回错误列表（空 = 通过）。
     *
     * @param schema 工具声明的参数面（可为 null = 无参数面，放行）
     * @param args   模型给定的参数树（要求 JSON 对象）
     */
    public static List<String> validate(JsonSchemaElement schema, JsonNode args) {
        List<String> errors = new ArrayList<>();
        if (schema == null) {
            return errors;
        }
        if (args == null || args.isNull() || args.isMissingNode()) {
            args = io.github.paopaonb666.agentbase.toolkit.ToolArgsValidator.emptyObject();
        }
        if (!args.isObject()) {
            errors.add("参数必须是 JSON 对象");
            return errors;
        }
        if (schema instanceof JsonObjectSchema objectSchema) {
            validateObject(objectSchema, args, errors, 0);
        }
        return errors;
    }

    private static JsonNode emptyObject() {
        return com.fasterxml.jackson.databind.node.JsonNodeFactory.instance.objectNode();
    }

    private static void validateObject(JsonObjectSchema schema, JsonNode args,
                                       List<String> errors, int depth) {
        if (depth > 3) { // 深层嵌套不展开：保守放行
            return;
        }
        for (String required : schema.required()) {
            JsonNode value = args.get(required);
            if (value == null || value.isNull() || value.isMissingNode()) {
                errors.add("缺少必需参数 '" + required + "'");
            }
        }
        Map<String, JsonSchemaElement> properties = schema.properties();
        if (properties == null || properties.isEmpty()) {
            return; // 未声明参数面的对象：任意键放行
        }
        var fields = args.properties();
        for (Map.Entry<String, JsonNode> field : fields) {
            JsonSchemaElement propertySchema = properties.get(field.getKey());
            if (propertySchema == null) {
                continue; // 未声明的键放行（模型多给的上下文）
            }
            validateValue(propertySchema, field.getValue(), field.getKey(), errors, depth);
        }
    }

    private static void validateValue(JsonSchemaElement schema, JsonNode value, String path,
                                      List<String> errors, int depth) {
        if (value == null || value.isNull() || value.isMissingNode()) {
            return; // 缺失由 required 层报告
        }
        if (schema instanceof JsonStringSchema) {
            if (!value.isTextual()) {
                errors.add(error(path, "类型应为 string，实为 " + jsonTypeName(value)));
            }
        } else if (schema instanceof JsonIntegerSchema) {
            if (!value.isIntegralNumber()) {
                errors.add(error(path, "类型应为 integer，实为 " + jsonTypeName(value)));
            }
        } else if (schema instanceof JsonNumberSchema) {
            if (!value.isNumber()) {
                errors.add(error(path, "类型应为 number，实为 " + jsonTypeName(value)));
            }
        } else if (schema instanceof JsonBooleanSchema) {
            if (!value.isBoolean()) {
                errors.add(error(path, "类型应为 boolean，实为 " + jsonTypeName(value)));
            }
        } else if (schema instanceof JsonEnumSchema enumSchema) {
            if (!value.isTextual() || !enumSchema.enumValues().contains(value.asText())) {
                errors.add(error(path, "取值应为 " + enumSchema.enumValues()
                        + " 之一，实为 " + value));
            }
        } else if (schema instanceof JsonArraySchema arraySchema) {
            if (!value.isArray()) {
                errors.add(error(path, "类型应为 array，实为 " + jsonTypeName(value)));
                return;
            }
            JsonSchemaElement items = arraySchema.items();
            if (items != null) {
                int index = 0;
                for (JsonNode item : value) {
                    validateValue(items, item, path + "[" + index + "]", errors, depth + 1);
                    index++;
                }
            }
        } else if (schema instanceof JsonObjectSchema objectSchema) {
            if (!value.isObject()) {
                errors.add(error(path, "类型应为 object，实为 " + jsonTypeName(value)));
                return;
            }
            validateObject(objectSchema, value, errors, depth + 1);
        } else if (schema instanceof JsonAnyOfSchema) {
            // 联合形态：任一支通过即可（保守——没有支匹配才报）
            List<String> branchErrors = new ArrayList<>();
            for (JsonSchemaElement branch : orEmpty(objectSchemasOf((JsonAnyOfSchema) schema))) {
                List<String> attempt = new ArrayList<>();
                validateValue(branch, value, path, attempt, depth);
                if (attempt.isEmpty()) {
                    return;
                }
                branchErrors.addAll(attempt);
            }
            errors.addAll(branchErrors);
        }
        // 其他形态（reference 等）：放行
    }

    private static List<JsonSchemaElement> objectSchemasOf(JsonAnyOfSchema schema) {
        return schema.anyOf();
    }

    private static <T> List<T> orEmpty(List<T> list) {
        return list == null ? List.of() : list;
    }

    private static String error(String path, String reason) {
        return "参数 '" + path + "' 校验失败: " + reason;
    }

    private static String jsonTypeName(JsonNode value) {
        if (value.isTextual()) {
            return "string";
        }
        if (value.isIntegralNumber()) {
            return "integer";
        }
        if (value.isNumber()) {
            return "number";
        }
        if (value.isBoolean()) {
            return "boolean";
        }
        if (value.isArray()) {
            return "array";
        }
        if (value.isObject()) {
            return "object";
        }
        return "null";
    }

    /**
     * #335 回喂契约：校验错误合并为模型可读文本——
     * {@code <e1>；<e2>。请修正参数后重试}，每条错误形如
     * {@code 参数 'x' 校验失败: <原因>}。契约由 ToolPoolValidationTest 锁定。
     */
    public static String contractText(List<String> errors) {
        return String.join("；", errors) + "。请修正参数后重试";
    }
}
