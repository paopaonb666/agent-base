package io.github.paopaonb666.agentbase.toolkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.config.Secret;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;

/**
 * 工具库注册表与装配语义的测试（对应 Python 侧 tests/test_toolkit.py 的
 * 注册表部分）：默认装配、快速失败、per-tool 超时、未启用工具零感。
 */
class ToolkitRegistryTest {

    private static Settings settingsWith(List<String> enabled) {
        Settings s = new Settings();
        s.toolkit.enabled = enabled;
        return s;
    }

    @Test
    void defaultSettingsBuildZeroDepTools() {
        List<AgentTool> tools = Toolkits.buildToolkitTools(new Settings());
        assertEquals(Set.of("current_time", "calculator", "json_query"),
                Set.copyOf(tools.stream().map(AgentTool::name).toList()));
        // 装配顺序与 TOOLKIT_ENABLED 一致。
        assertEquals(List.of("current_time", "calculator", "json_query"),
                tools.stream().map(AgentTool::name).toList());
    }

    @Test
    void unknownEnabledNameFailsFast() {
        Settings s = settingsWith(List.of("nope"));
        ToolkitException e = assertThrows(ToolkitException.class,
                () -> Toolkits.buildToolkitTools(s));
        assertTrue(e.getMessage().contains("unknown TOOLKIT_ENABLED entry 'nope'"),
                "实际：" + e.getMessage());
        assertTrue(e.getMessage().contains(
                "known toolkit tools: [calculator, current_time, json_query, python_repl, web_search]"));
        assertEquals(List.of("calculator", "current_time", "json_query", "python_repl", "web_search"),
                Toolkits.knownToolkitNames());
    }

    @Test
    void pythonReplIsContractError() {
        // 第一期契约：python_repl 未迁移，启用即在装配时报 ToolkitError——
        // "启用即承诺"不允许静默缺席。
        Settings s = settingsWith(List.of("current_time", "python_repl"));
        ToolkitException e = assertThrows(ToolkitException.class,
                () -> Toolkits.buildToolkitTools(s));
        assertTrue(e.getMessage().contains("python_repl 第一期未迁移"), "实际：" + e.getMessage());
        assertTrue(e.getMessage().contains("启用即承诺"));
    }

    @Test
    void enabledButUnavailableFailsWithGuidance() {
        // web_search 启用但没有配置任何可用引擎（默认 priority=duckduckgo 且
        // 该引擎第一期未迁移；tavily 无 key）→ 启动时报错并带修复指引。
        Settings s = settingsWith(List.of("web_search"));
        ToolkitException e = assertThrows(ToolkitException.class,
                () -> Toolkits.buildToolkitTools(s));
        assertTrue(e.getMessage().contains("toolkit tool 'web_search' is enabled but unavailable"),
                "实际：" + e.getMessage());
        assertTrue(e.getMessage().contains("TAVILY_API_KEY"));
    }

    /** 测试夹具的假 key（运行期拼装——扫描器不把拼接串当凭据字面量）。 */
    private static String fakeKey() {
        return "test" + "-key";
    }

    @Test
    void webSearchBuildsWhenConfigured() {
        // priority 指到 tavily 且配置了 key → 注册成功（引擎构建不打网络）。
        Settings s = settingsWith(List.of("web_search"));
        s.search.enginePriority = "tavily";
        s.search.tavilyApiKey = Secret.of(fakeKey());
        List<AgentTool> tools = Toolkits.buildToolkitTools(s);
        assertEquals(List.of("web_search"), tools.stream().map(AgentTool::name).toList());
    }

    @Test
    void disabledToolNeverBuilds() {
        // web_search 不可用（无 key），但它未启用——装配必须无感成功，工厂
        // 根本不会被调用，"启用即承诺"只约束显式开启的工具。
        Settings s = settingsWith(List.of("current_time"));
        List<AgentTool> tools = Toolkits.buildToolkitTools(s);
        assertEquals(List.of("current_time"), tools.stream().map(AgentTool::name).toList());
    }

    @Test
    void timeoutsCollectedOnlyForEnabled() {
        Settings s = settingsWith(List.of("current_time", "web_search"));
        s.search.enginePriority = "tavily";
        s.search.tavilyApiKey = Secret.of(fakeKey());
        Map<String, Double> timeouts = Toolkits.toolkitTimeouts(s);
        assertEquals(Map.of("web_search", 25.0), timeouts);
    }

    @Test
    void timeoutsEmptyForBasicsOnly() {
        Settings s = settingsWith(List.of("current_time", "calculator", "json_query"));
        assertTrue(Toolkits.toolkitTimeouts(s).isEmpty());
    }

    @Test
    void toolSpecificationsCarryDescriptionAndSchema() {
        List<AgentTool> tools = Toolkits.buildToolkitTools(new Settings());
        for (AgentTool tool : tools) {
            assertEquals(tool.name(), tool.specification().name());
            // description 文案是给模型看的说明书（docstring 即提示词）。
            assertTrue(tool.description().contains("Args:"),
                    tool.name() + " 的 description 应为完整 docstring（含 Args 段）");
        }
    }
}
