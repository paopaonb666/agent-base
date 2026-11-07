package io.github.paopaonb666.agentbase.core.config;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** #114：引号内的 " #" 保留（对齐 python-dotenv）；无引号值保留行内注释截断语义。 */
class SettingsDotEnvTest {

    @Test
    void quotedHashIsPreserved() {
        assertEquals("a #b", Settings.parseEnvValue("\"a #b\""));
        assertEquals("a #b", Settings.parseEnvValue("'a #b'"));
        // 引号内不以 # 开头的行内注释形态（先 unquote 后判的老 bug 会截成 "a"）。
        assertEquals("password with # inside", Settings.parseEnvValue("\"password with # inside\""));
    }

    @Test
    void unquotedInlineCommentStillTruncates() {
        assertEquals("a", Settings.parseEnvValue("a #b"));
        // # 前无空白不是注释（与 python-dotenv 一致）。
        assertEquals("a#b", Settings.parseEnvValue("a#b"));
    }

    @Test
    void doubleQuoteEscapePairsDoNotRegress() {
        assertEquals("line\nx\"y", Settings.parseEnvValue("\"line\\nx\\\"y\""));
        assertEquals("tab\there", Settings.parseEnvValue("\"tab\\there\""));
    }

    @Test
    void dotEnvFileRoundTripKeepsQuotedHash(@TempDir Path dir) throws Exception {
        Path env = dir.resolve(".env");
        Files.writeString(env, "export MEMORY_AUTH_SECRET=\"a #b\"\n");
        Settings s = Settings.load(Map.of(), env);
        assertEquals("a #b", s.memory.authSecret.value());
    }
}
