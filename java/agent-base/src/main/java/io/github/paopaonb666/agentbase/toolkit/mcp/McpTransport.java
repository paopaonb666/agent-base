package io.github.paopaonb666.agentbase.toolkit.mcp;

import java.io.IOException;

/**
 * MCP 传输层：一条行式 JSON-RPC 通道（写一行、读一行）。
 *
 * <p>stdio 形态是 {@link McpStdioTransport}（子进程 stdin/stdout）；
 * HTTP 系形态见 #332。行 = 以 {@code \n} 结尾的单条 JSON-RPC 消息。
 */
public interface McpTransport extends AutoCloseable {

    /** 建立连接（启动子进程 / 建立会话）。失败抛 IOException 或 McpException。 */
    void start() throws IOException;

    /** 写一条 JSON-RPC 消息（调用方保证单行、已含换行由实现负责补齐）。 */
    void sendLine(String line) throws IOException;

    /**
     * 读下一条 JSON-RPC 消息（阻塞）。通道关闭返回 null；IO 失败抛
     * IOException。
     */
    String readLine() throws IOException;

    @Override
    void close();
}
