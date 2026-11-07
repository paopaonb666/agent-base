package io.github.paopaonb666.agentbase.toolkit.mcp;

/**
 * MCP 工具来源装配失败（配置非法 / server 连不上 / 协议握手失败）时抛出。
 * 对齐 Python 侧 {@code tools/mcp.py} 的 {@code McpError}：启用即承诺，
 * 失败带可执行的修复指引，绝不静默缺席。
 */
public class McpException extends RuntimeException {

    public McpException(String message) {
        super(message);
    }

    public McpException(String message, Throwable cause) {
        super(message, cause);
    }
}
