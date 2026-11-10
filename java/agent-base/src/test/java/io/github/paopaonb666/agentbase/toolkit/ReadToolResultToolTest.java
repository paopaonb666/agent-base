package io.github.paopaonb666.agentbase.toolkit;

import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.core.tools.ToolPool;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileInfo;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStores;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * read_tool_result 分页与属主测试（条目 6/23）：分页窗口（offset/limit
 * 边界、next_offset、已到底、码点切窗不劈代理对）、属主拒绝（user 不匹配
 * /thread 不匹配/不存在同文案掩蔽）、limit/offset 越界报错不钳制、
 * readOnly 恒 true。
 */
class ReadToolResultToolTest {

    /** 事件旁路的 NodeContext 替身（工具内只读 config）。 */
    private static final class BypassContext implements NodeContext {
        private final RunConfig config;
        BypassContext(RunConfig config) { this.config = config; }
        @Override public RunConfig config() { return config; }
        @Override public void emit(Object event) { }
        @Override public String callTool(String name, String argsJson) { return ""; }
        @Override public ToolPool tools() { return null; }
        @Override public boolean cancelled() { return false; }
    }

    private static AgentTool.ToolContext ctxOf(String threadId, String userId) {
        return new AgentTool.ToolContext(new RunConfig(threadId, userId, 25), new BypassContext(
                new RunConfig(threadId, userId, 25)));
    }

    /** 落一条 module="tool_result" 的外置行（与 spillIfNeeded 的落库形态一致）。 */
    private static UploadedFileStores.MemoryUploadedFileStore storeWith(
            String fileId, String threadId, String userId, String text) {
        var store = new UploadedFileStores.MemoryUploadedFileStore();
        store.save(new UploadedFileInfo(fileId, threadId, "tool_result", userId,
                "web_search", "txt", null, null, false, text.length(), "", text, new byte[0]));
        return store;
    }

    private static ReadToolResultTool tool(UploadedFileStores.MemoryUploadedFileStore store) {
        return new ReadToolResultTool(store);
    }

    @Test
    void paginatesWithNextOffsetUntilEnd() throws Exception {
        var store = storeWith("f1", "chat:t1", "u1", "a".repeat(5000));
        var tool = tool(store);
        // 默认参数（offset=0、limit=2000；file_id 必填）→ next_offset=2000。
        String page1 = tool.call("{\"file_id\": \"f1\"}", ctxOf("chat:t1", "u1"));
        assertEquals("共 5000 字符 / 本屏 offset=0 limit=2000 / next_offset=2000\n"
                + "a".repeat(2000), page1);
        // 中间页：offset=2000 → next_offset=4000。
        String page2 = tool.call("{\"file_id\": \"f1\", \"offset\": 2000}", ctxOf("chat:t1", "u1"));
        assertTrue(page2.startsWith("共 5000 字符 / 本屏 offset=2000 limit=2000 / next_offset=4000\n"), page2);
        // 尾页：offset=4000 limit=2000 → 只剩 1000 字符，已到底。
        String page3 = tool.call("{\"file_id\": \"f1\", \"offset\": 4000}", ctxOf("chat:t1", "u1"));
        assertEquals("共 5000 字符 / 本屏 offset=4000 limit=2000 / 已到底\n" + "a".repeat(1000), page3);
    }

    @Test
    void offsetBeyondLengthReturnsLegalEmptyScreen() throws Exception {
        var store = storeWith("f1", "chat:t1", "u1", "abc");
        var tool = tool(store);
        // offset 超出文本长度：合法空屏 + 已到底（游标走完的自然终点）。
        assertEquals("共 3 字符 / 本屏 offset=3 limit=2000 / 已到底",
                tool.call("{\"file_id\": \"f1\", \"offset\": 3}", ctxOf("chat:t1", "u1")));
        assertEquals("共 3 字符 / 本屏 offset=99 limit=10 / 已到底",
                tool.call("{\"file_id\": \"f1\", \"offset\": 99, \"limit\": 10}", ctxOf("chat:t1", "u1")));
    }

    @Test
    void limitBoundsReportErrorWithoutClamping() throws Exception {
        var store = storeWith("f1", "chat:t1", "u1", "a".repeat(100));
        var tool = tool(store);
        // 越界报错不钳制（既有文案先例："limit 必须在 1..8000 之间"）。
        assertThrows(IllegalArgumentException.class,
                () -> tool.call("{\"file_id\": \"f1\", \"limit\": 0}", ctxOf("chat:t1", "u1")));
        assertThrows(IllegalArgumentException.class,
                () -> tool.call("{\"file_id\": \"f1\", \"limit\": 8001}", ctxOf("chat:t1", "u1")));
        assertThrows(IllegalArgumentException.class,
                () -> tool.call("{\"file_id\": \"f1\", \"limit\": -1}", ctxOf("chat:t1", "u1")));
        // offset 负数报错不钳制；offset=null 缺参报错。
        assertThrows(IllegalArgumentException.class,
                () -> tool.call("{\"file_id\": \"f1\", \"offset\": -1}", ctxOf("chat:t1", "u1")));
        // 上限恰等与缺省：合法。
        tool.call("{\"file_id\": \"f1\", \"limit\": 8000}", ctxOf("chat:t1", "u1"));
        // file_id 缺失/空 → 报错（必填）。
        assertThrows(IllegalArgumentException.class,
                () -> tool.call("{\"offset\": 0}", ctxOf("chat:t1", "u1")));
    }

    @Test
    void ownershipMismatchAndMissingIdShareMaskedDenialText() throws Exception {
        var store = storeWith("f1", "chat:t1", "u1", "内容");
        var tool = tool(store);
        // user 不匹配 / thread 不匹配 / 不存在：统一掩蔽文案（404 掩蔽风格）。
        assertEquals(ReadToolResultTool.DENIED_TEXT,
                tool.call("{\"file_id\": \"f1\"}", ctxOf("chat:t1", "u2")));
        assertEquals(ReadToolResultTool.DENIED_TEXT,
                tool.call("{\"file_id\": \"f1\"}", ctxOf("chat:t9", "u1")));
        assertEquals(ReadToolResultTool.DENIED_TEXT,
                tool.call("{\"file_id\": \"nope\"}", ctxOf("chat:t1", "u1")));
    }

    @Test
    void slicesByCodePointsWithoutSplittingSurrogatePairs() throws Exception {
        String text = "a".repeat(3) + "😀" + "b".repeat(3);
        var store = storeWith("f1", "chat:t1", "u1", text);
        var tool = tool(store);
        // 码点口径：总长 7（😀 计 1），[0,4) 码点 = aaa😀（emoji 整码点入屏不劈半）。
        String page1 = tool.call("{\"file_id\": \"f1\", \"limit\": 4}", ctxOf("chat:t1", "u1"));
        assertEquals("共 7 字符 / 本屏 offset=0 limit=4 / next_offset=4\n" + "aaa😀", page1);
        String page2 = tool.call("{\"file_id\": \"f1\", \"offset\": 4}", ctxOf("chat:t1", "u1"));
        assertEquals("共 7 字符 / 本屏 offset=4 limit=2000 / 已到底\nbbb", page2);
    }

    @Test
    void readOnlyStaysTrueAndToolNameIsStable() {
        var tool = tool(storeWith("f1", "chat:t1", "u1", "x"));
        // 纯读工具：#336 只读并发语义（default true，显式锁定防误改）。
        assertTrue(tool.readOnly());
        assertEquals("read_tool_result", tool.name());
        assertTrue(tool.description().codePointCount(0, tool.description().length()) >= 30);
    }

    @Test
    void nonJsonObjectArgsAreRejected() {
        var tool = tool(storeWith("f1", "chat:t1", "u1", "x"));
        assertThrows(Exception.class, () -> tool.call("[]", ctxOf("chat:t1", "u1")));
        assertThrows(Exception.class, () -> tool.call("not-json", ctxOf("chat:t1", "u1")));
    }
}
