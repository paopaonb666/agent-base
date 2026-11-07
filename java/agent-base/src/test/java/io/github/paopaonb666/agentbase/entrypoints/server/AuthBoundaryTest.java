package io.github.paopaonb666.agentbase.entrypoints.server;

import io.github.paopaonb666.agentbase.core.config.Settings;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * HMAC 身份鉴权边界矩阵（roadmap #293）：窗口边界（含恰好等于窗口的
 * 容忍语义）、缺时间戳/非整数时间戳、nonce 非法格式、错签名不占用 nonce
 * 记账、重放拒绝、指纹过期回收与 LRU 淘汰上限。时钟经包内重载注入——
 * 边界值（==窗口、过期瞬间）不再依赖真实时钟的宽容裕度。
 */
class AuthBoundaryTest {

    private static final String SECRET = "boundary-secret";

    private static Settings settings(int window) {
        Settings s = new Settings();
        s.memory.authSecret = SECRET;
        s.memory.authReplayWindowSeconds = window;
        return s;
    }

    /** 无 secret 部署：仅走 user_id 白名单，不要求签名。 */
    private static Settings settingsNoSecret() {
        return new Settings();
    }

    private static String sig(String message) {
        return AuthBackends.HmacHeaderAuth.hmacSha256Hex(SECRET, message);
    }

    /** HttpError 带状态码断言的便捷断言。 */
    private static int statusOf(Runnable action) {
        try {
            action.run();
            return 200;
        } catch (HttpError e) {
            return e.status();
        }
    }

    @Test
    void userIdDefaultsAndValidates() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth();
        Settings noSecret = settingsNoSecret();
        // 缺头 → default；空串同缺省语义。
        assertEquals("default", auth.authenticate(null, null, null, null, noSecret));
        assertEquals("default", auth.authenticate("", null, null, null, noSecret));
        // 非法形态（空格/超长/非白名单字符）以 400 拒绝，不静默替换。
        assertEquals(400, statusOf(() -> auth.authenticate("bad user", null, null, null, noSecret)));
        assertEquals(400, statusOf(() -> auth.authenticate("a".repeat(65), null, null, null, noSecret)));
        assertEquals(400, statusOf(() -> auth.authenticate("用户", null, null, null, noSecret)));
        // 白名单边界：64 字符合法，点/下划线/连字符放行。
        assertEquals("a".repeat(64), auth.authenticate("a".repeat(64), null, null, null, noSecret));
        assertEquals("A.b_9-c", auth.authenticate("A.b_9-c", null, null, null, noSecret));
        // 白名单校验先于签名校验：非法 user_id 即使配置了 secret 也是 400。
        assertEquals(400, statusOf(() -> auth.authenticate("bad user", "deadbeef", null, null, settings(0))));
    }

    @Test
    void windowBoundaryIsInclusive() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth();
        Settings s = settings(120);
        // 恰好等于窗口（±120）：|now-ts| > window 才拒——边界值容忍。
        assertEquals("alice", auth.authenticate("alice",
                sig("alice\n1000120\n"), "1000120", null, s, 1000000));
        assertEquals("alice", auth.authenticate("alice",
                sig("alice\n999880\n"), "999880", null, s, 1000000));
        // 越界一秒即拒（过去与未来同判——时钟漂移过大同样失效）。
        HttpError past = assertThrows(HttpError.class, () -> auth.authenticate("alice",
                sig("alice\n999879\n"), "999879", null, s, 1000000));
        assertEquals(401, past.status());
        HttpError future = assertThrows(HttpError.class, () -> auth.authenticate("alice",
                sig("alice\n1000121\n"), "1000121", null, s, 1000000));
        assertEquals(401, future.status());
    }

    @Test
    void timestampIsRequiredAndMustBeUnixSeconds() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth();
        Settings s = settings(120);
        // 防重放窗口启用后缺时间戳即 401（而不是退回旧 user_id 口径）。
        HttpError missing = assertThrows(HttpError.class,
                () -> auth.authenticate("alice", sig("alice\n"), null, null, s, 1000000));
        assertEquals(401, missing.status());
        assertEquals(401, statusOf(() -> auth.authenticate("alice",
                sig("alice\nnot-a-number\n"), "not-a-number", null, s, 1000000)));
        // 前后空白可解析，但 message 用原始头值——签名必须覆盖原文。
        assertEquals("alice", auth.authenticate("alice",
                sig("alice\n 1000000 \n"), " 1000000 ", null, s, 1000000));
        // 签名按去空白后的值算 → 与覆盖原文的期望签名不等 → 401。
        assertEquals(401, statusOf(() -> auth.authenticate("alice",
                sig("alice\n1000000\n"), " 1000000 ", null, s, 1000000)));
    }

    @Test
    void nonceFormatIsWhitelisted() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth();
        Settings s = settings(120);
        String good = "alice\n1000000\nn-1";
        // 空 nonce 合法（message 尾段为空）。
        assertEquals("alice", auth.authenticate("alice", sig("alice\n1000000\n"),
                "1000000", null, s, 1000000));
        // 非白名单字符 → 400（进记忆表与日志的值同一白名单）。
        assertEquals(400, statusOf(() -> auth.authenticate("alice",
                sig("alice\n1000000\nbad nonce"), "1000000", "bad nonce", s, 1000000)));
        assertEquals(400, statusOf(() -> auth.authenticate("alice",
                sig(good), "1000000", "n".repeat(65), s, 1000000)));
    }

    @Test
    void badSignatureDoesNotConsumeTheNonce() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth();
        Settings s = settings(120);
        // 错签名 + nonce：401，且 nonce 不进记账（通过签名校验才记账）。
        assertEquals(401, statusOf(() -> auth.authenticate("alice",
                "deadbeef", "1000000", "n1", s, 1000000)));
        // 随后正确签名带同一 nonce：接受——错签名没有烧掉它。
        assertEquals("alice", auth.authenticate("alice",
                sig("alice\n1000000\nn1"), "1000000", "n1", s, 1000000));
    }

    @Test
    void nonceReplayIsRejectedUntilTheFingerprintExpires() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth();
        Settings s = settings(100);
        // 首次：ts=1000000 @ now=1000000，指纹存活至 ts+2*window=1000200。
        assertEquals("alice", auth.authenticate("alice", sig("alice\n1000000\nn1"),
                "1000000", "n1", s, 1000000));
        // 窗口内重放（换新 ts 保窗口检查通过，nonce 不变）：指纹在册 → 401。
        HttpError replay = assertThrows(HttpError.class, () -> auth.authenticate(
                "alice", sig("alice\n1000050\nn1"), "1000050", "n1", s, 1000050));
        assertEquals(401, replay.status());
        // 同 ts 原样重放（签名相同）：窗口检查通过后落在指纹在册分支 → 401。
        assertEquals(401, statusOf(() -> auth.authenticate(
                "alice", sig("alice\n1000000\nn1"), "1000000", "n1", s, 1000050)));
        // 指纹过期后（now > 1000200）：同 nonce 换新 ts 视作新请求放行。
        assertEquals("alice", auth.authenticate("alice", sig("alice\n1000250\nn1"),
                "1000250", "n1", s, 1000250));
        // 过期回收后重新记账：立刻重放仍拒。
        assertEquals(401, statusOf(() -> auth.authenticate(
                "alice", sig("alice\n1000250\nn1"), "1000250", "n1", s, 1000250)));
    }

    @Test
    void replayCacheEvictsEldestBeyondLimit() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth(2);
        Settings s = settings(3600);
        // 记满 4 个 nonce（上限 2）→ n1（被 n3 挤出）、n2（被 n4 挤出）淘汰，
        // 在册只剩 n3/n4。
        for (String nonce : new String[]{"n1", "n2", "n3", "n4"}) {
            assertEquals("alice", auth.authenticate("alice",
                    sig("alice\n1000000\n" + nonce), "1000000", nonce, s, 1000000));
        }
        assertEquals(401, statusOf(() -> auth.authenticate("alice",
                sig("alice\n1000000\nn3"), "1000000", "n3", s, 1000000)));
        assertEquals(401, statusOf(() -> auth.authenticate("alice",
                sig("alice\n1000000\nn4"), "1000000", "n4", s, 1000000)));
        // n1 已被 LRU 淘汰：重放成为可能（缓存上限的固有取舍，文档已注明）。
        assertEquals("alice", auth.authenticate("alice",
                sig("alice\n1000000\nn1"), "1000000", "n1", s, 1000000));
        // n1 重登记挤掉当前最老者（n3）→ n3 转为可重放；n4 仍在册被拒。
        assertEquals("alice", auth.authenticate("alice",
                sig("alice\n1000000\nn3"), "1000000", "n3", s, 1000000));
        assertEquals(401, statusOf(() -> auth.authenticate("alice",
                sig("alice\n1000000\nn1"), "1000000", "n1", s, 1000000)));
    }

    @Test
    void windowlessModeSkipsTimestampAndNonceBookkeeping() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth();
        Settings s = settings(0);
        // 窗口=0：message 只覆盖 user_id，时间戳/nonce 头即使给错也不参与。
        assertEquals("alice", auth.authenticate("alice", sig("alice"), "garbage", "n1", s, 1000000));
        // 无防重放：同签名可重复（文档明示不推荐的旧方案语义）。
        assertEquals("alice", auth.authenticate("alice", sig("alice"), null, null, s, 1000000));
        assertTrue(true);
    }
}
