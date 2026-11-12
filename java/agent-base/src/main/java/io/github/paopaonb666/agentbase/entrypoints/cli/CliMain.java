package io.github.paopaonb666.agentbase.entrypoints.cli;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;

import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.UUID;

import io.github.paopaonb666.agentbase.Version;
import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.core.graph.CompiledGraph;
import io.github.paopaonb666.agentbase.core.graph.GraphState;
import io.github.paopaonb666.agentbase.core.graph.RunConfig;
import io.github.paopaonb666.agentbase.extensions.Observability;

/**
 * 交互式 CLI 入口。
 *
 * <p>用法：
 * <pre>
 * java -jar agent-base.jar --message "你好"           # 单轮
 * java -jar agent-base.jar --module chat              # 交互式循环
 * java -jar agent-base.jar --module supervisor        # 多 Agent 会话
 * java -jar agent-base.jar --thread-id &lt;id&gt;           # 恢复一次对话
 * java -jar agent-base.jar --version
 * </pre>
 *
 * <p>对话状态：每个会话都在一个 thread_id 下运行，checkpointer 按
 * thread 持久化历史。传入 --thread-id 可恢复之前的对话——使用 sqlite
 * 后端时，进程重启后依然存在。每一轮只发送<b>新</b>的消息；历史从
 * checkpointer 回放，CLI 从不重发。
 *
 * <p>可观测性：每个会话都在一个全新的 request_id 下运行；会话期间发出
 * 的所有日志都会带上它。对应 Python 侧 {@code entrypoints/cli.py}。
 */
public final class CliMain {

    private CliMain() {}

    public static void main(String[] args) {
        System.exit(runCli(args));
    }

    /** 解析参数并运行；返回进程退出码（对应 argparse + main）。 */
    public static int runCli(String[] argv) {
        String module;
        String message;
        String threadId;
        try {
            String[] parsed = parseArgs(argv);
            module = parsed[0];
            message = parsed[1];
            threadId = parsed[2];
        } catch (IllegalArgumentException e) {
            // 参数解析错误对应 argparse 的 exit 2（缺值/非法形态），不裸抛。
            System.err.println("error: " + e.getMessage());
            return 2;
        }
        for (int i = 0; i < argv.length; i++) {
            switch (argv[i]) {
                case "--server" -> {
                    // SSE 服务入口（对应 uvicorn agent_base.entrypoints.server:app）。
                    // 反射加载避免 CLI 编译期依赖 Spring 装配类。
                    try {
                        Class<?> app = Class.forName(
                                "io.github.paopaonb666.agentbase.entrypoints.server.AgentBaseApplication");
                        org.springframework.boot.SpringApplication.run(app, argv);
                        return 0;
                    } catch (ClassNotFoundException e) {
                        System.err.println("error: server 模块不可用（" + e + "）");
                        return 1;
                    }
                }
                case "--version" -> {
                    System.out.println("agent-base " + Version.VERSION);
                    return 0;
                }
                case "--help", "-h" -> {
                    System.out.println("""
                            usage: agent-base [--module MODULE] [--message MESSAGE] \
                            [--thread-id ID] [--version]""");
                    return 0;
                }
                // 值型参数（--module/--message/-m/--thread-id）：parseArgs 已取
                // 值，这里连同其值一起跳过。审查 C-P0 回归锁：#295 重构曾在
                // 本 switch 丢失这些 case——文档化主用法（--message 单轮）落
                // default「未知参数」退 2，测试盲区（只测缺值路径）未抓到。
                case "--module", "--message", "-m", "--thread-id" -> i++;
                default -> {
                    System.err.println("error: 未知参数 " + argv[i]);
                    return 2;
                }
            }
        }
        try {
            return run(module, message, threadId);
        } catch (io.github.paopaonb666.agentbase.core.config.Settings.SettingsError exc) {
            System.err.println("configuration error: " + exc.getMessage());
            return 2;
        } catch (AgentRuntime.UnknownModuleError exc) {
            // 只接住"模块不存在"；图内部真正的键缺失不该被吞成这条文案。
            System.err.println("error: " + exc.getMessage());
            return 2;
        } catch (Exception exc) {
            // 入口以一条消息失败，而不是堆栈转储（会话日志已经携带
            // request_id 可供调试）。
            System.err.println("error: " + exc.getMessage());
            return 1;
        }
    }

    /** 解析 --module/--message/--thread-id；返回 [module, message, threadId]。 */
    private static String[] parseArgs(String[] argv) {
        String module = "chat";
        String message = null;
        String threadId = null;
        for (int i = 0; i < argv.length; i++) {
            switch (argv[i]) {
                case "--module" -> module = requireValue(argv, ++i, "--module");
                case "--message", "-m" -> message = requireValue(argv, ++i, "--message");
                case "--thread-id" -> threadId = requireValue(argv, ++i, "--thread-id");
                default -> { /* 立即返回型参数（--server/--version/--help）在调用方处理 */ }
            }
        }
        return new String[]{module, message, threadId};
    }

    private static String requireValue(String[] argv, int index, String flag) {
        if (index >= argv.length) {
            throw new IllegalArgumentException(flag + " 需要一个值");
        }
        return argv[index];
    }

    private static int run(String module, String message, String threadIdArg) {
        Settings settings = Settings.load(java.util.Map.of());
        AgentRuntime runtime = AgentRuntime.create(settings);
        try {
            Observability.setupLogging(settings.observability.logJson);
            Observability.logTracingConfig();
            String threadId = threadIdArg != null ? threadIdArg
                    : UUID.randomUUID().toString().replace("-", "").substring(0, 12);
            final int[] exit = {0};
            Observability.withRequestId(null, () -> {
                org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(CliMain.class);
                logger.info("cli: start module={} thread_id={}", module, threadId);
                try {
                    if (message != null) {
                        oneShot(runtime, module, threadId, message);
                    } else {
                        interactive(runtime, module, threadId);
                    }
                } finally {
                    // 始终关闭 trace——即便 LLM 调用爆炸了——这样一次会话在
                    // 同一个 request_id 下就有一个确定的结束标记。
                    logger.info("cli: done request_id={}", Observability.requestId());
                }
            });
            return exit[0];
        } finally {
            try {
                runtime.close();
            } catch (Exception ignored) {
                // 任何退出路径都要尝试释放 checkpointer 连接。
            }
        }
    }

    /** 发送一轮用户消息；其余历史由 checkpointer 回放。 */
    static GraphState invoke(AgentRuntime runtime, String moduleName, String threadId, String text) {
        CompiledGraph graph = runtime.graph(moduleName);
        // thread id 按模块划分命名空间：不同的图共用一个 checkpointer。
        RunConfig config = new RunConfig(moduleName + ":" + threadId, "default",
                runtime.settings().agentRecursionLimit);
        return graph.invoke(new GraphState(List.of(new UserMessage(text)), java.util.Map.of()), config);
    }

    /** 面向人类的回复：最后一条 assistant 消息（如果有的话）。 */
    static String reply(List<ChatMessage> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof AiMessage ai && ai.text() != null && !ai.text().isEmpty()) {
                return ai.text();
            }
        }
        return "";
    }

    private static void oneShot(AgentRuntime runtime, String module, String threadId, String text) {
        List<ChatMessage> updated = invoke(runtime, module, threadId, text).messages();
        String reply = reply(updated);
        if (!reply.isEmpty()) {
            System.out.println(reply);
        }
        System.err.println("[thread_id: " + threadId + "]");
    }

    private static void interactive(AgentRuntime runtime, String module, String threadId) {
        System.out.println("agent-base [" + module + "] — type a message; Ctrl+C or 'quit' to exit");
        System.out.println("thread_id: " + threadId + " (pass --thread-id " + threadId + " to resume)");
        Scanner scanner = new Scanner(System.in, java.nio.charset.StandardCharsets.UTF_8);
        Set<String> quitWords = Set.of("quit", "exit");
        while (true) {
            System.out.print("you> ");
            if (!scanner.hasNextLine()) {
                System.out.println();
                return;
            }
            String text = scanner.nextLine().strip();
            if (text.isEmpty()) {
                continue;
            }
            if (quitWords.contains(text.toLowerCase(java.util.Locale.ROOT))) {
                return;
            }
            String reply = reply(invoke(runtime, module, threadId, text).messages());
            if (!reply.isEmpty()) {
                System.out.println("agent> " + reply);
            }
        }
    }
}
