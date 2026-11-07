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

    /**
     * 给模型看的描述——description 就是提示词的一部分，直接决定模型
     * 会不会用、用得对不对。#24 起按<b>三段式</b>规范撰写（不强制死板
     * 格式，求信息完整）：
     *
     * <ol>
     *   <li><b>何时用</b>——工具覆盖什么场景，给一两个触发词/信号；</li>
     *   <li><b>何时不用</b>——最容易误用的相邻场景（有更合适的替代工具、
     *       答案就在上下文里、前置条件缺失）；</li>
     *   <li><b>参数要点与误用示例</b>——各参数的坑位（格式、边界、默认值），
     *       复杂参数给"错误示例 / 正确示例"对照。</li>
     * </ol>
     *
     * <p>约束：简洁（目标是 Python 侧对应 docstring 长度的 2 倍以内）、
     * 与参数 schema 不重复罗列、不写实现细节。历史基线：description
     * 非空且 ≥ 30 字符（ToolDescriptionContractTest 锁定）。两个已声明
     * 的豁免：web_search 与 Python docstring 逐字一致（一致性锁定，
     * 不升级为三段式）；MCP 工具透传 server 侧描述（内容不可控）。
     */
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
