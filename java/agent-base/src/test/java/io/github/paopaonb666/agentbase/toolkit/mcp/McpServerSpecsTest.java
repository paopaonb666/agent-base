package io.github.paopaonb666.agentbase.toolkit.mcp;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** {@link McpServerSpecs} 的结构校验契约（对应 Python _parse_servers 文案）。 */
class McpServerSpecsTest {

    @Test
    void parsesStdioServerWithArgsAndEnv() {
        Map<String, McpServerSpec> specs = McpServerSpecs.parse(
                "{\"docs\": {\"transport\": \"stdio\", \"command\": \"uvx\","
                        + " \"args\": [\"mcp-server-fetch\"], \"env\": {\"K\": \"V\"},"
                        + " \"timeout_seconds\": 12.5}}");
        McpServerSpec spec = specs.get("docs");
        assertEquals("stdio", spec.transport());
        assertEquals("uvx", spec.command());
        assertEquals(java.util.List.of("mcp-server-fetch"), spec.args());
        assertEquals(Map.of("K", "V"), spec.env());
        assertEquals(12.5, spec.timeoutSeconds());
    }

    @Test
    void emptyJsonFailsWithExample() {
        McpException e = assertThrows(McpException.class, () -> McpServerSpecs.parse("  "));
        assertEquals(true, e.getMessage().contains("MCP_SERVERS_JSON 为空"));
    }

    @Test
    void invalidJsonFails() {
        assertThrows(McpException.class, () -> McpServerSpecs.parse("{not json"));
    }

    @Test
    void nonObjectOrEmptyObjectFails() {
        assertThrows(McpException.class, () -> McpServerSpecs.parse("[]"));
        assertThrows(McpException.class, () -> McpServerSpecs.parse("{}"));
    }

    @Test
    void badServerNameFails() {
        assertThrows(McpException.class, () -> McpServerSpecs.parse(
                "{\"Bad-Name\": {\"transport\": \"stdio\", \"command\": \"x\"}}"));
        assertThrows(McpException.class, () -> McpServerSpecs.parse(
                "{\"1abc\": {\"transport\": \"stdio\", \"command\": \"x\"}}"));
    }

    @Test
    void unknownTransportFails() {
        McpException e = assertThrows(McpException.class, () -> McpServerSpecs.parse(
                "{\"docs\": {\"transport\": \"carrier_pigeon\", \"url\": \"http://x\"}}"));
        assertEquals(true, e.getMessage().contains("carrier_pigeon"));
    }

    @Test
    void stdioRequiresCommand() {
        assertThrows(McpException.class, () -> McpServerSpecs.parse(
                "{\"docs\": {\"transport\": \"stdio\", \"args\": [\"x\"]}}"));
    }

    @Test
    void httpTransportsRequireUrl() {
        assertThrows(McpException.class, () -> McpServerSpecs.parse(
                "{\"docs\": {\"transport\": \"streamable_http\"}}"));
    }

    @Test
    void timeoutMustBePositiveNumber() {
        assertThrows(McpException.class, () -> McpServerSpecs.parse(
                "{\"docs\": {\"transport\": \"stdio\", \"command\": \"x\","
                        + " \"timeout_seconds\": -1}}"));
        assertThrows(McpException.class, () -> McpServerSpecs.parse(
                "{\"docs\": {\"transport\": \"stdio\", \"command\": \"x\","
                        + " \"timeout_seconds\": \"soon\"}}"));
    }

    @Test
    void optionalFieldsDefault() {
        Map<String, McpServerSpec> specs = McpServerSpecs.parse(
                "{\"docs\": {\"transport\": \"stdio\", \"command\": \"x\"}}");
        McpServerSpec spec = specs.get("docs");
        assertEquals(java.util.List.of(), spec.args());
        assertEquals(Map.of(), spec.env());
        assertNull(spec.timeoutSeconds());
        assertNull(spec.url());
    }
}
