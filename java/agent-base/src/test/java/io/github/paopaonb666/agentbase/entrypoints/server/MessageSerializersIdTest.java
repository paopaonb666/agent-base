package io.github.paopaonb666.agentbase.entrypoints.server;

import dev.langchain4j.data.message.AiMessage;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** #123：assistant 行的稳定 id 上屏——有 id 才加键，旧 wire 不变。 */
class MessageSerializersIdTest {

    @Test
    void assistantRowCarriesIdWhenPresent() {
        AiMessage audited = AiMessage.builder()
                .text("回答")
                .attributes(new LinkedHashMap<>(Map.of("id", "msg-abc123")))
                .build();
        Map<String, Object> row = MessageSerializers.serializeMessage(audited, null, false);
        assertEquals("assistant", row.get("role"));
        assertEquals("msg-abc123", row.get("id"));
    }

    @Test
    void assistantRowWithoutIdKeepsLegacyWire() {
        // 旧快照重建的消息无 attributes → 不加 id 键（wire 向后兼容）。
        Map<String, Object> row = MessageSerializers.serializeMessage(
                AiMessage.from("旧消息"), null, false);
        assertEquals("assistant", row.get("role"));
        assertFalse(row.containsKey("id"));
        assertTrue(row.containsKey("content"));
    }
}
