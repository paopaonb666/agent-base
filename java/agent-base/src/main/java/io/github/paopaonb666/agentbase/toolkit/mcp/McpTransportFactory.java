package io.github.paopaonb666.agentbase.toolkit.mcp;

import java.io.IOException;

/**
 * MCP 传输工厂（#29 重连最小面）：每次 {@code create()} 返回<b>未启动</b>
 * 的新通道——{@link McpClient} 在断开后惰性重建（server 崩溃自愈，无需
 * 重启进程）。unchecked 的 {@link McpException}（如未知 transport）允许
 * 直接抛出，由调用方归一。
 */
@FunctionalInterface
public interface McpTransportFactory {

    /** 创建一条未启动的传输通道（stdio=新子进程；HTTP=新会话）。 */
    McpTransport create() throws IOException;
}
