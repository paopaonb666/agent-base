package io.github.paopaonb666.agentbase.modules.chat;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.tools.AgentTool;
import io.github.paopaonb666.agentbase.modules.chat.graph.ChatGraphs;

import java.util.List;
import java.util.Map;

/**
 * {@code ChatModule} 的 AgentModule 实现——对话样板模块（AgentModule 的
 * 参考实现）。
 *
 * <p>这是每个新模块都要照抄的模板。它完全位于 core 之外——证明模块
 * 接入时无需触碰基座代码。基座之所以能发现它，仅仅是因为 {@code chat}
 * 被列在 AGENT_MODULES 里（registry 按命名约定实例化本类）。图构建
 * 委托给 {@link ChatGraphs}，工具委托给 {@link EchoTool}。
 */
public final class ChatModule implements AgentModule {

    private final EchoTool echo = new EchoTool();

    @Override
    public String name() {
        return "chat";
    }

    @Override
    public String description() {
        return "Minimal conversational agent: a single LLM node over the message "
                + "history. Reference implementation of the AgentModule contract.";
    }

    @Override
    public CompiledGraph buildGraph(AgentModule.ModuleContext ctx) {
        return ChatGraphs.buildChatGraph(ctx);
    }

    @Override
    public List<AgentTool> getTools() {
        return List.of(echo);
    }

    /**
     * chat 模块贡献的工具：证明"贡献 → 池 → ToolNode"往返完整的样板
     * 工具；随着模块增多，请替换为真正的模块工具。
     */
    public static final class EchoTool implements AgentTool {

        @Override
        public String name() {
            return "echo";
        }

        @Override
        public String description() {
            return "把给定的文本原样返回，并加上 'echo:' 前缀。"
                    + "用于证明共享工具池端到端可用的样板工具；随着模块增多，请替换为真正的模块工具。";
        }

        @Override
        public ToolSpecification specification() {
            return ToolSpecification.builder()
                    .name(name())
                    .description(description())
                    .parameters(JsonObjectSchema.builder()
                            .addStringProperty("text", "要回显的文本")
                            .required("text")
                            .build())
                    .build();
        }

        @Override
        public String call(String argsJson, ToolContext ctx) throws Exception {
            Map<String, Object> args = new com.fasterxml.jackson.databind.ObjectMapper()
                    .readValue(argsJson == null ? "{}" : argsJson,
                            new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {});
            Object text = args.get("text");
            return "echo: " + (text == null ? "" : text);
        }
    }
}
