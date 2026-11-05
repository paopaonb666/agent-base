package io.github.paopaonb666.agentbase.core.graph;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.core.config.Settings;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * checkpointer 三后端的存储往返测试：工厂分派、put/get/delete/
 * listThreadIds/healthy 生命周期、含 tool_calls 的 AiMessage 与多段
 * UserMessage（文本 + 图片块）的序列化往返。sqlite 用临时目录文件；
 * 不依赖网络/MySQL。
 */
class CheckpointerTest {

    @TempDir
    Path tempDir;

    // ── 工厂 ─────────────────────────────────────────────────────────

    @Test
    void factoryBuildsMemoryBackend() {
        Settings settings = new Settings();
        settings.checkpointer.backend = "memory";
        assertInstanceOf(InMemoryCheckpointer.class, CheckpointerFactory.build(settings));
    }

    @Test
    void factoryBuildsSqliteBackend() {
        Settings settings = new Settings();
        settings.checkpointer.backend = "sqlite";
        settings.checkpointer.sqlitePath = tempDir.resolve("factory.db").toString();
        try (Checkpointer checkpointer = CheckpointerFactory.build(settings)) {
            assertInstanceOf(SqliteCheckpointer.class, checkpointer);
        }
    }

    @Test
    void factoryRejectsUnknownBackend() {
        Settings settings = new Settings();
        settings.checkpointer.backend = "redis";
        // 文案对齐 Python build_checkpointer 的语义。
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> CheckpointerFactory.build(settings));
        assertTrue(error.getMessage().contains("unknown CHECKPOINTER_BACKEND"));
    }

    @Test
    void factoryRejectsUnwritableSqlitePath() {
        Settings settings = new Settings();
        settings.checkpointer.backend = "sqlite";
        settings.checkpointer.sqlitePath =
                tempDir.resolve("missing-dir").resolve("state.db").toString();
        // 移植 Python 的启动期路径预检：坏路径给可读的配置错误。
        Settings.SettingsError error = assertThrows(Settings.SettingsError.class,
                () -> CheckpointerFactory.build(settings));
        assertTrue(error.getMessage().contains("is not writable"));
    }

    // ── 往返与生命周期 ───────────────────────────────────────────────

    @Test
    void memoryRoundtripAndLifecycle() {
        try (InMemoryCheckpointer checkpointer = new InMemoryCheckpointer()) {
            roundtripAndLifecycle(checkpointer);
        }
    }

    @Test
    void sqliteRoundtripAndLifecycle() {
        Path db = tempDir.resolve("state.db");
        try (SqliteCheckpointer checkpointer = new SqliteCheckpointer(db.toString())) {
            roundtripAndLifecycle(checkpointer);
        }
    }

    @Test
    void sqliteToleratesCorruptedRow() throws SQLException {
        Path db = tempDir.resolve("broken.db");
        try (SqliteCheckpointer checkpointer = new SqliteCheckpointer(db.toString())) {
            // 直接用第二条连接塞入损坏行（绕过 put 的序列化）。
            try (var conn = DriverManager.getConnection("jdbc:sqlite:" + db);
                 Statement st = conn.createStatement()) {
                st.execute("INSERT INTO ab_thread_states (thread_id, state_json, updated_at) "
                        + "VALUES ('broken', 'not-json{', '2026-01-01T00:00:00')");
            }
            // 损坏行按缺失处理：get 返回 null，健康探针不受影响。
            assertNull(checkpointer.get("broken"));
            assertTrue(checkpointer.healthy());
        }
    }

    // ── 共享断言 ─────────────────────────────────────────────────────

    private void roundtripAndLifecycle(Checkpointer checkpointer) {
        assertNull(checkpointer.get("chat:t1"));

        GraphState original = sampleState();
        checkpointer.put("chat:t1", original);

        GraphState restored = checkpointer.get("chat:t1");
        assertNotNull(restored);
        // 序列化层面逐字段对齐（含 tool_calls 与多段 content 的消息形态）。
        assertEquals(GraphStateJson.toJson(original), GraphStateJson.toJson(restored));

        // 重复 put 覆盖（upsert 语义）。
        checkpointer.put("chat:t1", new GraphState(
                List.of(UserMessage.from("覆盖后的内容")), Map.of()));
        assertEquals(GraphStateJson.toJson(new GraphState(
                        List.of(UserMessage.from("覆盖后的内容")), Map.of())),
                GraphStateJson.toJson(checkpointer.get("chat:t1")));

        assertNull(checkpointer.get("chat:absent"));

        checkpointer.put("chat:t2", new GraphState(
                List.of(UserMessage.from("第二条")), Map.of()));
        List<String> ids = checkpointer.listThreadIds();
        assertTrue(ids.contains("chat:t1"));
        assertTrue(ids.contains("chat:t2"));

        checkpointer.deleteThread("chat:t1");
        assertNull(checkpointer.get("chat:t1"));
        assertFalse(checkpointer.listThreadIds().contains("chat:t1"));

        assertTrue(checkpointer.healthy());
    }

    /** 覆盖全部消息形态与 extra 通道的样本状态。 */
    private static GraphState sampleState() {
        List<ChatMessage> messages = List.of(
                SystemMessage.from("你是助手"),
                // 多段 content（文本 + 图片块）：UserMessage 的多模态形态。
                UserMessage.from(List.of(
                        TextContent.from("看这张图"),
                        ImageContent.from("aGVsbG8=", "image/png"))),
                // AiMessage 带 tool_calls：工具往返的请求侧。
                AiMessage.from("我先查一下", List.of(ToolExecutionRequest.builder()
                        .id("call-1").name("calculator").arguments("{\"a\":1,\"b\":2}")
                        .build())),
                // 工具结果侧。
                ToolExecutionResultMessage.from("call-1", "calculator", "3"));
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("tasks", List.of(Map.of("id", 1, "goal", "查天气", "status", "pending")));
        extra.put("cursor", 2);
        extra.put("replans", 0);
        return new GraphState(messages, extra);
    }
}
