package io.github.paopaonb666.agentbase.toolkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.modules.chat.ChatModule;
import io.github.paopaonb666.agentbase.modules.hello.HelloModule;
import io.github.paopaonb666.agentbase.modules.planner.PlannerModule;
import io.github.paopaonb666.agentbase.modules.writer.WriterModule;
import io.github.paopaonb666.agentbase.toolkit.search.WebSearchTool;

/**
 * 工具描述三段式契约（roadmap #24）：全部装配可见工具的 description
 * 非空且信息量达标（≥ 30 字符）；echo 样板工具不再出现在 ChatModule
 * 默认装配；web_search 的 description 与 Python 侧 docstring 逐字一致
 * （豁免项锁定——它沿用 chat-agent 验证过的 Query 生成规范，改文案即
 * 破坏一致性承诺）。MCP 工具透传 server 侧描述，不进默认装配，无断言。
 */
class ToolDescriptionContractTest {

    /** 基座全部默认装配可见的工具：工具库 + 各模块 getTools 贡献。 */
    private static List<AgentTool> allAssembledTools() {
        List<AgentTool> tools = new ArrayList<>(Toolkits.buildToolkitTools(new Settings()));
        // web_search 默认不在 TOOLKIT_ENABLED，但一旦启用就进池——描述
        // 契约对它同样生效（豁免的是"三段式升级"，不是"信息量下限"）。
        // 测试路径直构（manager/cache 不参与 description）。
        tools.add(new WebSearchTool(null, null, 30.0));
        tools.addAll(new ChatModule().getTools());
        tools.addAll(new WriterModule().getTools());
        tools.addAll(new PlannerModule().getTools());
        tools.addAll(new HelloModule().getTools());
        return tools;
    }

    @Test
    void assembledToolDescriptionsMeetInformationFloor() {
        for (AgentTool tool : allAssembledTools()) {
            String description = tool.description();
            assertNotNull(description, tool.name() + " description 为 null");
            assertFalse(description.isBlank(), tool.name() + " description 为空");
            assertTrue(description.codePointCount(0, description.length()) >= 30,
                    tool.name() + " description 信息量不足（≥30 字符）：" + description);
        }
    }

    @Test
    void chatModuleNoLongerShipsEchoByDefault() {
        // #24：echo 样板工具移出默认装配（EchoTool 类保留供测试/样例）。
        List<String> names = new ChatModule().getTools().stream()
                .map(AgentTool::name).toList();
        assertFalse(names.contains("echo"), "echo 不应再出现在 ChatModule 默认装配：" + names);
    }

    @Test
    void webSearchDescriptionStaysVerbatimWithPythonDocstring() throws IOException {
        Path pythonSource = locatePythonToolSource();
        String docstring = extractTopLevelToolDocstring(Files.readAllLines(pythonSource));
        String javaDescription = new WebSearchTool(null, null, 30.0).description();
        assertEquals(docstring, javaDescription,
                "web_search description 必须与 Python docstring 逐字一致"
                        + "（一致性锁定，改文案需双端同步）");
    }

    /**
     * 向上逐级查找仓库内的 Python 源（surefire 工作目录是模块目录
     * {@code java/agent-base}，源码在仓库根 {@code src/} 下）。
     */
    private static Path locatePythonToolSource() throws IOException {
        Path dir = Paths.get("").toAbsolutePath();
        for (int i = 0; i < 8 && dir != null; i++, dir = dir.getParent()) {
            Path candidate = dir.resolve("src").resolve("agent_base")
                    .resolve("tools").resolve("search").resolve("tool.py");
            if (Files.isRegularFile(candidate)) {
                return candidate;
            }
        }
        throw new IOException("找不到 src/agent_base/tools/search/tool.py"
                + "（web_search 描述一致性锁定的比对基线）");
    }

    /**
     * 从 Python 源里抽出 {@code build_web_search_tool} 装配的 BaseTool
     * docstring（{@code """搜索互联网…"""}），剥公共缩进后返回。锁定的
     * 是工具 docstring 本身，不是模块顶部的说明注释。
     */
    private static String extractTopLevelToolDocstring(List<String> lines) {
        String openContent = null;
        List<String> raw = new ArrayList<>();
        int commonIndent = Integer.MAX_VALUE;
        for (String line : lines) {
            String stripped = line.strip();
            if (openContent == null) {
                if (stripped.startsWith("\"\"\"搜索互联网获取最新信息")) {
                    openContent = stripped.substring(3);
                }
                continue;
            }
            if (stripped.equals("\"\"\"")) {
                break;
            }
            raw.add(line);
            if (!stripped.isEmpty()) {
                commonIndent = Math.min(commonIndent, line.length()
                        - line.stripLeading().length());
            }
        }
        if (openContent == null) {
            throw new IllegalStateException("tool.py 里没有找到 web_search docstring");
        }
        StringBuilder out = new StringBuilder(openContent);
        for (String line : raw) {
            out.append('\n');
            // 剥公共缩进（docstring 的 8 空格基准），空行原样归空。
            out.append(line.isBlank() ? ""
                    : line.substring(Math.min(commonIndent, line.length())));
        }
        // 对齐 Java 侧 text block 的 stripTrailing()。
        return out.toString().stripTrailing();
    }
}
