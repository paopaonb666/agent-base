package io.github.paopaonb666.agentbase.modules.planner.graph;

import io.github.paopaonb666.agentbase.core.text.JsonBlocks;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * plan 输出解析测试（roadmap #258）：平衡括号抽取取代贪婪 DOTALL——
 * 「多 JSON 块 + 尾注」形态取首个数组而不是最后一段括号；字符串字面量
 * 内的括号不参与配对；解析失败仍降级单任务。
 */
class PlanNodeParseTest {

    @Test
    void multipleJsonBlocksTakeTheFirstArrayNotTheLast() {
        // 贪婪 \[.*\] 会从首个 [ 一直吃到尾注里的 [1,2]——拆解错乱。
        String raw = """
                前置说明。
                [{"goal": "任务A"}, {"goal": "任务B"}]
                以上是拆解结果。参考数据 [1, 2]。
                """;
        List<Map<String, Object>> tasks = PlanNode.parseTasks(raw, 5, "兜底");
        assertEquals(2, tasks.size());
        assertEquals("任务A", tasks.get(0).get("goal"));
        assertEquals("任务B", tasks.get(1).get("goal"));
    }

    @Test
    void bracketsInsideStringLiteralsDoNotBreakPairing() {
        // goal 文本里带 [方括号] 与 {花括号}：字面量内不参与配对。
        String raw = "[{\"goal\": \"处理 [异常] 与 {对象} 字面\"}]";
        List<Map<String, Object>> tasks = PlanNode.parseTasks(raw, 5, "兜底");
        assertEquals(1, tasks.size());
        assertEquals("处理 [异常] 与 {对象} 字面", tasks.get(0).get("goal"));
    }

    @Test
    void parseFailureStillDegradesToSingleTask() {
        List<Map<String, Object>> tasks = PlanNode.parseTasks("完全没有数组", 5, "兜底目标");
        assertEquals(1, tasks.size());
        assertEquals("兜底目标", tasks.get(0).get("goal"));
        // 空数组同降级。
        assertEquals(1, PlanNode.parseTasks("[]", 5, "兜底目标").size());
        // goals 全空同降级。
        assertEquals(1, PlanNode.parseTasks("[{\"goal\": \"  \"}]", 5, "兜底目标").size());
    }

    @Test
    void jsonBlocksScannerBasics() {
        assertNull(JsonBlocks.firstBalancedArray(null));
        assertNull(JsonBlocks.firstBalancedArray("不平衡 [1, 2"));
        assertNull(JsonBlocks.firstBalancedArray("无数组"));
        assertEquals("[1, 2]", JsonBlocks.firstBalancedArray("x [1, 2] y [3] z"));
        // 字符串内的 close 不提前闭合。
        assertEquals("[\"a]b\"]", JsonBlocks.firstBalancedArray("[\"a]b\"]"));
        // 嵌套平衡。
        assertEquals("[{\"a\": [1]}]", JsonBlocks.firstBalancedArray("[{\"a\": [1]}] 尾注"));
        assertTrue(JsonBlocks.firstBalancedObject("{\"k\": \"v]\"}").endsWith("}"));
    }
}
