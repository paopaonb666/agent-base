package io.github.paopaonb666.agentbase.modules.chat.graph;

import io.github.paopaonb666.agentbase.core.AgentModule;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graphs.Graphs;

/**
 * chat 模块的图构建。
 *
 * <p>单 LLM 节点 + 可选工具循环，经基座共享图工厂装配（core/graphs）：
 * 上下文工程（注入去重 + token 预算修剪）、工具失败归一、流式累积与
 * 取消传播都是工厂的公共行为，本模块只剩差异化的声明（模块名，无提示词）。
 *
 * <p>该图以 {@code name="chat"} 编译：supervisor 要求每个 sub-agent 图
 * 都携带其模块的名字。对应 Python 侧 {@code modules/chat/graph.py}。
 */
public final class ChatGraphs {

    private ChatGraphs() {}

    public static final String MODULE_NAME = "chat";

    /** 构建并编译 chat 图（LLM 节点 + 可选的工具循环）。 */
    public static CompiledGraph buildChatGraph(AgentModule.ModuleContext ctx) {
        return Graphs.buildSingleAgentGraph(ctx, MODULE_NAME, null, null);
    }
}
