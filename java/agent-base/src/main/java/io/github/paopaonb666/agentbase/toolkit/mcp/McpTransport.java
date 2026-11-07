package io.github.paopaonb666.agentbase.toolkit.mcp;

import java.io.IOException;

/**
 * MCP 传输层：一条行式 JSON-RPC 通道（写一行、读一行）。
 *
 * <p>stdio 形态是 {@link McpStdioTransport}（子进程 stdin/stdout）；
 * HTTP 系形态（streamable_http / sse，#332）把响应喂进内部队列模拟同一
 * 接口。行 = 以 {@code \n} 结尾的单条 JSON-RPC 消息。
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

    /**
     * 握手后由客户端回调：告知协商出的协议版本（HTTP 系用于后续请求的
     * {@code MCP-Protocol-Version} 头；stdio 无用，默认空实现）。
     */
    default void onProtocolVersion(String version) {
    }

    @Override
    void close();
}
