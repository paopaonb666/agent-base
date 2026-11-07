package io.github.paopaonb666.agentbase.toolkit.mcp;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * MCP stdio 传输（#331）：把 server 子进程的 stdin/stdout 接成行式
 * JSON-RPC 通道（MCP stdio 帧约定：每条消息一行）。
 *
 * <p>子进程生命周期归本对象所有：{@link #start()} 用 {@link ProcessBuilder}
 * 拉起，{@link #close()} 优雅关闭（关 stdin → destroy → 超时强杀）。
 * stderr 透传到应用进程（server 的诊断信息进应用日志）。
 */
public final class McpStdioTransport implements McpTransport {

    private static final Logger LOG = LoggerFactory.getLogger(McpStdioTransport.class);

    /** 优雅退出等待窗口；超时 destroyForcibly。 */
    private static final long DESTROY_GRACE_SECONDS = 5;

    private final String command;
    private final List<String> args;
    private final Map<String, String> env;
    private final String serverName;

    private Process process;
    // 读写两个方向不加同一把锁：读线程会长时间阻塞在 readLine 上（等
    // server 的下一行输出），若与 sendLine 共用 transport 监视器，握手
    // 请求永远发不出去（实测死锁）。读只有读线程一个调用方，用 volatile
    // 快照防 close 竞态；写经独立 writeLock 串行化。
    private volatile BufferedReader reader;
    private volatile BufferedWriter writer;
    private final Object writeLock = new Object();

    public McpStdioTransport(String serverName, String command, List<String> args,
                             Map<String, String> env) {
        this.serverName = serverName;
        this.command = command;
        this.args = args;
        this.env = env;
    }

    @Override
    public void start() throws IOException {
        ProcessBuilder pb = new ProcessBuilder(toCommand());
        pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        if (!env.isEmpty()) {
            pb.environment().putAll(env);
        }
        process = pb.start();
        reader = new BufferedReader(new InputStreamReader(process.getInputStream(),
                StandardCharsets.UTF_8));
        writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(),
                StandardCharsets.UTF_8));
    }

    private List<String> toCommand() {
        List<String> cmd = new java.util.ArrayList<>();
        cmd.add(command);
        if (args != null) {
            cmd.addAll(args); // args 追加在命令之后（对应 stdio 连接的 argv 语义）
        }
        return List.copyOf(cmd);
    }

    @Override
    public void sendLine(String line) throws IOException {
        BufferedWriter w = writer;
        Process p = process;
        if (w == null || p == null || !p.isAlive()) {
            throw new IOException("MCP server '" + serverName + "' 进程未运行");
        }
        synchronized (writeLock) {
            w.write(line);
            w.write("\n");
            w.flush();
        }
    }

    @Override
    public String readLine() throws IOException {
        BufferedReader r = reader; // 仅读线程调用
        return r == null ? null : r.readLine();
    }

    @Override
    public void close() {
        if (process == null) {
            return;
        }
        try {
            BufferedWriter w = writer;
            if (w != null) {
                w.close(); // 关 stdin：多数 stdio server 以此触发优雅退出
            }
        } catch (IOException e) {
            LOG.debug("MCP server {}: 关闭 stdin 失败", serverName, e);
        }
        if (process.isAlive()) {
            process.destroy();
            try {
                if (!process.waitFor(DESTROY_GRACE_SECONDS, TimeUnit.SECONDS)) {
                    process.destroyForcibly();
                    LOG.warn("MCP server {}: 优雅退出超时，已强杀", serverName);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
            }
        }
        process = null;
        reader = null;
        writer = null;
    }
}
