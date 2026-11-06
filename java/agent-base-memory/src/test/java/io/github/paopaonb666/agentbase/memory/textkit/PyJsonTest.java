package io.github.paopaonb666.agentbase.memory.textkit;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * PyJson 与 Python {@code json.dumps(ensure_ascii=False)} 的逐字节对齐。
 * 期望值全部由 CPython 3.12 实际运行产生（画像截断阈值与 merge prompt
 * 的字节形态依赖于此）。
 */
class PyJsonTest {

    private static Map<String, Object> sampleProfile() {
        Map<String, Object> profile = new LinkedHashMap<>();
        profile.put("name", "小明");
        profile.put("偏好", List.of("喜欢猫", "住在杭州"));
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("city", "杭州");
        meta.put("n", 1.0);
        profile.put("meta", meta);
        return profile;
    }

    @Test
    void defaultSeparatorsMatchPython() {
        // python -c "print(json.dumps(p, ensure_ascii=False))"
        assertEquals(
                "{\"name\": \"小明\", \"偏好\": [\"喜欢猫\", \"住在杭州\"],"
                        + " \"meta\": {\"city\": \"杭州\", \"n\": 1.0}}",
                PyJson.dumps(sampleProfile()));
    }

    @Test
    void indentMatchesPython() {
        // python -c "print(json.dumps(p, ensure_ascii=False, indent=1))"：
        // '{' 后换行、每层 1 空格、项间 ",\n"、空容器保持紧凑。
        assertEquals(
                "{\n"
                        + " \"name\": \"小明\",\n"
                        + " \"偏好\": [\n"
                        + "  \"喜欢猫\",\n"
                        + "  \"住在杭州\"\n"
                        + " ],\n"
                        + " \"meta\": {\n"
                        + "  \"city\": \"杭州\",\n"
                        + "  \"n\": 1.0\n"
                        + " }\n"
                        + "}",
                PyJson.dumpsIndent(sampleProfile(), 1));
    }

    @Test
    void escapingMatchesPython() {
        Map<String, Object> value = new LinkedHashMap<>();
        value.put("a", "he said \"hi\"\ntab\tx");
        value.put("ctl", "\u0001");
        value.put("empty", List.of());
        value.put("obj", Map.of());
        // python: {"a": "he said \"hi\"\ntab\tx", "ctl": "\u0001", "empty": [], "obj": {}}
        assertEquals(
                "{\"a\": \"he said \\\"hi\\\"\\ntab\\tx\", \"ctl\": \"\\u0001\","
                        + " \"empty\": [], \"obj\": {}}",
                PyJson.dumps(value));
    }

    @Test
    void lengthMatchesPythonLen() {
        // 长度按码点计（len(str)），画像截断阈值依赖它。
        String json = PyJson.dumps(sampleProfile());
        assertEquals(71, json.codePointCount(0, json.length()));
    }
}
