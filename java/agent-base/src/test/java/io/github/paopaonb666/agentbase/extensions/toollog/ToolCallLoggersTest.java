package io.github.paopaonb666.agentbase.extensions.toollog;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import io.github.paopaonb666.agentbase.core.config.Settings;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 工具调用审计三后端测试（roadmap #266 + #302 的测试网）：listForThread
 * 升序语义（现实现 DESC 查询 + 反转）、memory 环形溢出、sqlite 有界队列
 * 灌满丢弃计数（offer 不阻塞）、close 尽力排空后重开读回。sqlite 用例跑在
 * 临时库文件上（模式抄 FilestoreTest 的 sqliteStore）；sqlite 写线程异步
 * 落库，断言统一走 deadline 轮询。mysql 后端需要真实服务器，跳过不测
 * （#307 域的既有口径：mysql 侧行为对齐 Python 由集成环境覆盖）。
 */
class ToolCallLoggersTest {

    @TempDir
    Path tempDir;

    // ── 装配辅助 ─────────────────────────────────────────────────────

    private static Settings settings(String backend, String sqlitePath) {
        Settings s = new Settings();
        s.checkpointer.backend = backend;
        if (sqlitePath != null) {
            s.checkpointer.sqlitePath = sqlitePath;
        }
        return s;
    }

    private ToolCallLoggers.SqliteToolCallRecorder sqliteRecorder(String fileName) {
        return new ToolCallLoggers.SqliteToolCallRecorder(tempDir.resolve(fileName).toString());
    }

    /**
     * 一条测试记录（errorText 随 status 走，成功为空串）。
     *
     * <p>历史注：rowToDict 对空 error_text 会 NPE（{@code Map.entry} 不容
     * null，#298 在 asDict 修过的同病、读回路径漏改）——该 bug 已修
     * （rowToDict 换 LinkedHashMap），成功记录读回恢复可测。
     */
    private static ToolCallRecord record(String callId, String threadId, String status) {
        return new ToolCallRecord(callId, "web_search", status, "{\"q\":\"x\"}",
                "结果-" + callId, status.equals("error") ? "boom" : "", 42,
                threadId, "chat", "req-" + callId);
    }

    private static List<String> callIds(List<Map<String, Object>> rows) {
        return rows.stream().map(r -> (String) r.get("call_id")).toList();
    }

    /** sqlite 写线程异步落库：deadline 轮询到期望条数，超时则失败。 */
    private static void awaitRows(ToolCallRecorder recorder, String threadId, int expected)
            throws InterruptedException {
        long deadline = System.currentTimeMillis() + 15_000;
        int seen = 0;
        while ((seen = recorder.listForThread(threadId, Integer.MAX_VALUE).size()) < expected
                && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        assertEquals(expected, seen, "写线程应在超时前把记录落库");
    }

    // ── 工厂分派 ─────────────────────────────────────────────────────

    @Test
    void factoryDispatchFollowsCheckpointerBackend() {
        assertNull(ToolCallLoggers.build(settings("redis", null))); // 未知后端 → null
        Settings disabled = settings("memory", null);
        disabled.toolkit.callLogEnabled = false;
        assertNull(ToolCallLoggers.build(disabled)); // 总开关关闭 → null
        assertTrue(ToolCallLoggers.build(settings("memory", null))
                instanceof ToolCallLoggers.MemoryToolCallRecorder);
        // sqlite 分派：临时文件建库后显式 close（接口非 AutoCloseable，
        // 且 Windows 上不释放句柄会让 @TempDir 清理失败）。
        ToolCallRecorder sqlite = ToolCallLoggers.build(settings("sqlite",
                tempDir.resolve("factory.db").toString()));
        assertTrue(sqlite instanceof ToolCallLoggers.SqliteToolCallRecorder);
        sqlite.close();
    }

    // ── memory 后端：环形溢出与升序 ──────────────────────────────────

    @Test
    void memoryBackendKeepsInsertionOrderAndRingCap() {
        ToolCallLoggers.MemoryToolCallRecorder recorder = new ToolCallLoggers.MemoryToolCallRecorder();
        // 探针先入：环形容量是全局的（跨线程共享），探针最终被挤出——
        // 顺带验证环形不做线程隔离。
        recorder.record(record("other", "t-other", "ok"));
        // 灌入 1050 条：环形 1000 挤掉最老的 50 条，剩余仍时间升序。
        for (int i = 0; i < ToolCallRecord.MEMORY_MAX_RECORDS + 50; i++) {
            recorder.record(record("c-" + i, "t-ring", "ok"));
        }

        List<Map<String, Object>> rows = recorder.listForThread("t-ring", Integer.MAX_VALUE);
        assertEquals(ToolCallRecord.MEMORY_MAX_RECORDS, rows.size());
        assertEquals("c-50", rows.get(0).get("call_id"));
        assertEquals("c-" + (ToolCallRecord.MEMORY_MAX_RECORDS + 49),
                rows.get(rows.size() - 1).get("call_id"));
        // limit 语义：最近 N 条且仍升序。
        assertEquals(List.of("c-" + (ToolCallRecord.MEMORY_MAX_RECORDS + 48),
                        "c-" + (ToolCallRecord.MEMORY_MAX_RECORDS + 49)),
                callIds(recorder.listForThread("t-ring", 2)));
        // 探针已被环形挤出（全局容量）。
        assertTrue(recorder.listForThread("t-other", 10).isEmpty());
    }

    // ── sqlite 后端：升序 / limit / 线程隔离 ─────────────────────────

    @Test
    void sqliteBackendListsAscendingWithLimitAndThreadIsolation() throws InterruptedException {
        ToolCallLoggers.SqliteToolCallRecorder recorder = sqliteRecorder("order.db");
        try {
            recorder.record(record("a", "t1", "error"));
            recorder.record(record("b", "t1", "error"));
            recorder.record(record("c", "t1", "error"));
            recorder.record(record("d", "t2", "error"));

            awaitRows(recorder, "t1", 3);
            // DESC 查询 + 反转 = 时间升序（与 Python 侧一致）。
            assertEquals(List.of("a", "b", "c"), callIds(recorder.listForThread("t1", 10)));
            // limit 取最近 N 条且仍升序。
            assertEquals(List.of("b", "c"), callIds(recorder.listForThread("t1", 2)));
            // 线程隔离与字段往返（status/error/result 落库一致）。
            List<Map<String, Object>> t2 = recorder.listForThread("t2", 10);
            assertEquals(List.of("d"), callIds(t2));
            assertEquals("error", t2.get(0).get("status"));
            assertEquals("boom", t2.get(0).get("error"));
            assertEquals("结果-d", t2.get(0).get("result"));
        } finally {
            recorder.close();
        }
    }

    // ── sqlite 后端：成功记录读回（rowToDict 空 error 修复回归） ─────

    @Test
    void sqliteOkRecordReadsBackWithNullError() throws InterruptedException {
        // rowToDict 曾对空 error_text NPE（Map.entry 不容 null，#298 在
        // asDict 修过的同病、读回路径漏改）——成功调用读回必炸。本用例
        // 锁定修复：ok 记录的 error 字段为 null 且其余字段完整往返。
        ToolCallLoggers.SqliteToolCallRecorder recorder = sqliteRecorder("okrow.db");
        try {
            recorder.record(record("ok-1", "t-ok", "ok"));
            awaitRows(recorder, "t-ok", 1);
            List<Map<String, Object>> rows = recorder.listForThread("t-ok", 10);
            assertEquals(1, rows.size());
            assertEquals("ok", rows.get(0).get("status"));
            assertNull(rows.get(0).get("error"), "成功记录 error 应为 null");
            assertEquals("结果-ok-1", rows.get(0).get("result"));
            assertEquals("req-ok-1", rows.get(0).get("request_id"));
        } finally {
            recorder.close();
        }
    }

    // ── sqlite 后端：队列灌满丢弃计数（#266） ────────────────────────

    @Test
    void sqliteQueueOverflowsWithDropsAndNeverBlocks() throws InterruptedException {
        ToolCallLoggers.SqliteToolCallRecorder recorder = sqliteRecorder("flood.db");
        try {
            // 灌入 5×容量：offer 非阻塞，写线程消化速度远低于投递速度，
            // 队列必然打满——丢弃计数上升且调用线程不被阻塞。
            // （对账要读回所有行，error 记录让 result 字段可区分。）
            int offered = ToolCallLoggers.SqliteToolCallRecorder.QUEUE_CAPACITY * 5;
            for (int i = 0; i < offered; i++) {
                recorder.record(record("f-" + i, "t-flood", "error"));
            }
            assertTrue(recorder.droppedRecordCount() > 0, "队列满应丢弃并计数");
            // 对账：投递数 = 丢弃数 + 落库数（未丢弃的记录最终全部写完）。
            awaitRows(recorder, "t-flood",
                    offered - (int) recorder.droppedRecordCount());
        } finally {
            recorder.close();
        }
    }

    // ── sqlite 后端：close 尽力排空 + 重开读回 ───────────────────────

    @Test
    void closeFlushesQueuedTailAndReopenReadsBack() throws InterruptedException {
        Path db = tempDir.resolve("close.db");
        ToolCallLoggers.SqliteToolCallRecorder recorder =
                new ToolCallLoggers.SqliteToolCallRecorder(db.toString());
        try {
            // 紧接着 close：无论写线程消化到哪，close 的排空语义保证
            // 已 record() 的记录全部落库（不因关闭丢尾）。
            recorder.record(record("tail-1", "t-tail", "error"));
            recorder.record(record("tail-2", "t-tail", "error"));
        } finally {
            recorder.close();
        }
        // 重开同一文件：记录已持久化，读回完整且升序。
        ToolCallLoggers.SqliteToolCallRecorder reopened = sqliteRecorder("close.db");
        try {
            List<Map<String, Object>> rows = reopened.listForThread("t-tail", 10);
            assertEquals(List.of("tail-1", "tail-2"), callIds(rows));
            assertEquals("boom", rows.get(0).get("error"));
        } finally {
            reopened.close();
        }
    }
}
