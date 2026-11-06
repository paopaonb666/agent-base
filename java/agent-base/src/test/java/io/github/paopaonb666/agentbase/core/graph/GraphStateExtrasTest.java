package io.github.paopaonb666.agentbase.core.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import dev.langchain4j.data.message.UserMessage;

/**
 * extras 通道的合并语义（附件元数据持久化依赖的行为契约）：apply 是
 * 按键合并——输入里缺席的键保留历史值，出现的键整键替换（对应
 * LangGraph 的通道 reducer：invoke 输入只提供 messages 时其余通道
 * 保持原值）。
 */
class GraphStateExtrasTest {

    @Test
    void applyKeepsAbsentKeys() {
        GraphState state = new GraphState(List.of(), Map.of(
                "attachments", Map.of("0", List.of(Map.of("file_id", "abc")))));
        state.apply(Map.of("messages", List.of(UserMessage.from("hi"))));
        assertEquals(Map.of("0", List.of(Map.of("file_id", "abc"))), state.get("attachments"),
                "输入不含 attachments 时历史附件表必须保留");
    }

    @Test
    void applyReplacesProvidedKeysWholesale() {
        GraphState state = new GraphState(List.of(), Map.of(
                "attachments", Map.of("0", List.of(Map.of("file_id", "old")))));
        state.apply(Map.of("attachments", Map.of(
                "0", List.of(Map.of("file_id", "old")),
                "3", List.of(Map.of("file_id", "new")))));
        Map<?, ?> attachments = state.get("attachments");
        assertEquals(2, attachments.size(), "attachments 键整表替换（旧表 + 新表合并由调用方完成）");
        assertEquals(List.of(Map.of("file_id", "new")), attachments.get("3"));
    }
}
