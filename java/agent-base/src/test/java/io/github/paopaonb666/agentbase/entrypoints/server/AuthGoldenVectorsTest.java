package io.github.paopaonb666.agentbase.entrypoints.server;

import io.github.paopaonb666.agentbase.core.config.Secret;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * HMAC 身份鉴权的跨语言 golden 向量：期望签名由 Python 侧
 * {@code scripts/memory_user_sig.py} 的同构算法（hmac.new(secret, message,
 * sha256).hexdigest()）生成——Java 实现必须接受 Python 生成的签名头。
 */
class AuthGoldenVectorsTest {

    private static final String SECRET = "test-secret-123";

    @Test
    void pythonGeneratedSignaturesVerify() {
        // 向量由 Python hmac.new(...).hexdigest() 生成（防重放窗口 0 / 空_nonce / 带_nonce 三形态）。
        assertEquals("f1125da676b19ea04e1090686570cdebc279a8d2201784ccd72a66d1321c3398",
                AuthBackends.HmacHeaderAuth.hmacSha256Hex(SECRET, "alice"));
        assertEquals("c6ff0c742f0c2def6db5bdca2dfc78cf957bd2be0b5951090dab7ad3f9821d1a",
                AuthBackends.HmacHeaderAuth.hmacSha256Hex(SECRET, "alice\n1700000000\n"));
        assertEquals("e535f405c95198bcdf8e26e59479c40e25386c4582a65362d7cf040701275b35",
                AuthBackends.HmacHeaderAuth.hmacSha256Hex(SECRET, "alice\n1700000000\nabc123"));
    }

    @Test
    void windowModeAcceptsPythonVectorAndRejectsReplay() {
        AuthBackends.HmacHeaderAuth auth = new AuthBackends.HmacHeaderAuth();
        io.github.paopaonb666.agentbase.core.config.Settings settings =
                new io.github.paopaonb666.agentbase.core.config.Settings();
        settings.memory.authSecret = Secret.of(SECRET);
        settings.memory.authReplayWindowSeconds = 0; // 旧方案：message = user_id
        String sig = AuthBackends.HmacHeaderAuth.hmacSha256Hex(SECRET, "alice");
        assertEquals("alice", auth.authenticate("alice", sig, null, null, settings));
        assertThrows(HttpError.class,
                () -> auth.authenticate("alice", "deadbeef", null, null, settings));
        // 窗口=0 的旧方案无 nonce 记账：合法签名可重复使用（无防重放）。
        assertEquals("alice", auth.authenticate("alice", sig, null, null, settings));
    }

    @Test
    void unauthenticatedDeploymentIsFlaggedForTheStartupBanner() {
        // #196：memory 启用 + secret 空 = 无鉴权部署（全部请求落 default
        // 用户，共享记忆）——组合根以横幅提示；配置了 secret 或 memory
        // 关闭则不是无鉴权部署。
        io.github.paopaonb666.agentbase.core.config.Settings bare =
                new io.github.paopaonb666.agentbase.core.config.Settings();
        assertTrue(AuthBackends.unauthenticatedDeployment(bare),
                "默认配置（memory.enabled=true、secret 空）即无鉴权部署");
        assertEquals(true, AuthBackends.unauthenticatedBanner().contains("default"));
        bare.memory.authSecret = Secret.of("s");
        assertFalse(AuthBackends.unauthenticatedDeployment(bare));
        io.github.paopaonb666.agentbase.core.config.Settings memoryOff =
                new io.github.paopaonb666.agentbase.core.config.Settings();
        memoryOff.memory.enabled = false;
        assertFalse(AuthBackends.unauthenticatedDeployment(memoryOff));
    }
}
