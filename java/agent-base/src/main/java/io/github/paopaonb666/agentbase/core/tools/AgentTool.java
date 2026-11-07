package io.github.paopaonb666.agentbase.core.tools;

import dev.langchain4j.agent.tool.ToolSpecification;
import io.github.paopaonb666.agentbase.core.graph.NodeContext;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;

/**
 * 一个可被模型调用的工具（对应 Python 侧 langchain 的 BaseTool 在本项目
 * 里的实际使用面）。模块工具与工具库（toolkit）工具都实现本接口，经
 * {@link ToolPool} 收集、超时包装与审计。
 */
public interface AgentTool {

    /** 稳定工具名（进入 ToolSpecification，必须全局唯一）。 */
    String name();

    /** 给模型看的描述。 */
    String description();

    /** 参数 JSON Schema（langchain4j ToolSpecification，含 name/description）。 */
    ToolSpecification specification();

    /**
     * 是否只读（#336）：只读工具在批量执行（{@link ToolPool#callAll}）里
     * 允许并发，有副作用的保持串行。默认 true——基座内置与记忆工具都是
     * 读取/追加语义；有外部副作用面的工具（web_search 网络请求、MCP
     * server 工具行为不可知）必须显式声明 false。
     */
    default boolean readOnly() {
        return true;
    }

    /**
     * 执行工具；返回给模型的结果文本。抛出的任何异常都会被池的包装层
     * 归一成模型可读文本（坏掉的工具绝不打断对话）。
     *
     * @param argsJson 模型给定的参数 JSON 串
     * @param ctx      执行上下文（run config + 事件通道）
     */
    String call(String argsJson, ToolContext ctx) throws Exception;

    /** 工具执行上下文。 */
    record ToolContext(RunConfig runConfig, NodeContext nodeContext) {

        /** 向流推送自定义事件（不在图上下文时为无操作）。 */
        public void emit(Object event) {
            if (nodeContext != null) {
                nodeContext.emit(event);
            }
        }
    }
}
