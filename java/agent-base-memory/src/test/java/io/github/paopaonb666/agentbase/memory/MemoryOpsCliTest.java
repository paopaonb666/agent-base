package io.github.paopaonb666.agentbase.memory;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 运维入口（#278）：子命令分发与退出码契约（成功 0 / 用法 2 / 运行 1）、
 * user-sig 头四件、kb-ingest 的路径收集与 dry-run。
 */
class MemoryOpsCliTest {

    /** 捕获 stdout/stderr 的运行。 */
    private record Run(int code, String out, String err) {
    }

    private static Run run(String... args) {
        PrintStream originalOut = System.out;
        PrintStream originalErr = System.err;
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        System.setOut(new PrintStream(out, true, StandardCharsets.UTF_8));
        System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
        try {
            return new Run(MemoryOpsCli.run(args),
                    out.toString(StandardCharsets.UTF_8),
                    err.toString(StandardCharsets.UTF_8));
        } finally {
            System.setOut(originalOut);
            System.setErr(originalErr);
        }
    }

    @Test
    void noArgumentsPrintsUsageAndExits2() {
        Run result = run();
        assertEquals(2, result.code());
        assertTrue(result.err().contains("用法"));
    }

    @Test
    void unknownSubcommandExits2() {
        assertEquals(2, run("nope").code());
    }

    @Test
    void missingRequiredOptionExits2() {
        Run result = run("kb-ingest", "--user", "u1");
        assertEquals(2, result.code());
        assertTrue(result.err().contains("--path"));
    }

    @Test
    void userSigPrintsFourAuthHeaders() {
        Run result = run("user-sig", "--user", "u1", "--secret", "s3cret");
        assertEquals(0, result.code());
        // Windows 的 println 是 \r\n：归一后断言
        String out = result.out().replace("\r\n", "\n");
        assertTrue(out.startsWith("X-User-Id: u1\n"), out);
        assertTrue(out.contains("X-User-Timestamp: "));
        assertTrue(out.contains("X-User-Nonce: "));
        assertTrue(out.contains("X-User-Sig: "));
        // 与服务端同规：签名 = HMAC(secret, user\nts\nnonce)
        String[] lines = out.strip().split("\n");
        String ts = lines[1].substring("X-User-Timestamp: ".length());
        String nonce = lines[2].substring("X-User-Nonce: ".length());
        String sig = lines[3].substring("X-User-Sig: ".length());
        String expected = io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends
                .HmacHeaderAuth.hmacSha256Hex("s3cret", "u1\n" + ts + "\n" + nonce);
        assertEquals(expected, sig);
    }

    @Test
    void pathHashIdIsStableAndPrefixed(@TempDir Path dir) throws Exception {
        String a = MemoryOpsCli.pathHashId("docs/a.pdf");
        String b = MemoryOpsCli.pathHashId("docs/a.pdf");
        assertEquals(a, b, "同路径哈希稳定（幂等摄取）");
        assertTrue(a.startsWith("kb"));
        String c = MemoryOpsCli.pathHashId("docs/b.pdf");
        assertTrue(!a.equals(c));
    }

    @Test
    void kbIngestDryRunListsCandidatesWithoutTouchingMemory(@TempDir Path dir) throws Exception {
        Files.writeString(dir.resolve("a.txt"), "文档 A", StandardCharsets.UTF_8);
        Files.writeString(dir.resolve("b.md"), "文档 B", StandardCharsets.UTF_8);
        Files.writeString(dir.resolve("skip.docx.bin"), "不识别扩展名", StandardCharsets.UTF_8);
        Run result = run("kb-ingest", "--user", "u1", "--path", dir.toString(), "--dry-run");
        assertEquals(0, result.code());
        assertTrue(result.out().contains("a.txt"));
        assertTrue(result.out().contains("b.md"));
        assertTrue(!result.out().contains("skip.docx.bin"), "非白名单扩展名不收集");
        assertTrue(result.out().contains("共 2 个待摄取文档"));
    }

    @Test
    void backfillErrorStateExitsOneWithoutEndlessPolling() throws Exception {
        // review B-P2：MemoryService 的回填终态词表是 done|error——修复前
        // CLI 只认 done/failed/not_running，error 态永不命中 → \r 刷屏无限
        // 轮询、退出码协议失效。真判据：error 序列在有限次轮询内退出且退 1
        //（修复前会继续轮询——第 3 次轮询抛错使本测试必红且不挂死）。
        java.util.concurrent.atomic.AtomicInteger polls = new java.util.concurrent.atomic.AtomicInteger();
        java.util.Queue<Map<String, Object>> sequence = new java.util.ArrayDeque<>(List.of(
                Map.of("state", "running", "processed", 1),
                Map.of("state", "error", "error", "embedding 中途失败")));
        int code = MemoryOpsCli.backfillExitCode(() -> {
            polls.incrementAndGet();
            Map<String, Object> next = sequence.poll();
            if (next == null) {
                throw new IllegalStateException("error 态未被识别为终态（无限轮询）");
            }
            return next;
        });
        assertEquals(1, code, "error 终态退 1");
        assertEquals(2, polls.get(), "error 态应在第 2 次轮询退出");
    }

    @Test
    void backfillDoneStateExitsZero() throws Exception {
        assertEquals(0, MemoryOpsCli.backfillExitCode(
                () -> Map.of("state", "done", "processed", 3)));
    }
}
