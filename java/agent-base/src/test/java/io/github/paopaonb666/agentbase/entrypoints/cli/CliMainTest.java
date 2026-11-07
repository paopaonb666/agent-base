package io.github.paopaonb666.agentbase.entrypoints.cli;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import io.github.paopaonb666.agentbase.Version;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * CLI 入口测试（roadmap #295）：参数解析退出码矩阵（version/help/未知/
 * 缺值——argparse 口径 exit 2）与面向人类的回复提取（最后一条带文本的
 * AiMessage）。只测装配前的纯路径：run() 起整套运行时，不在单测范围。
 */
class CliMainTest {

    /** 捕获 stdout/stderr 运行 runCli，返回 [exit, stdout, stderr]。 */
    private static String[] runCaptured(String... argv) {
        PrintStream originalOut = System.out;
        PrintStream originalErr = System.err;
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteArrayOutputStream err = new ByteArrayOutputStream();
        System.setOut(new PrintStream(out, true, StandardCharsets.UTF_8));
        System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
        try {
            int exit = CliMain.runCli(argv);
            return new String[]{String.valueOf(exit),
                    out.toString(StandardCharsets.UTF_8), err.toString(StandardCharsets.UTF_8)};
        } finally {
            System.setOut(originalOut);
            System.setErr(originalErr);
        }
    }

    @Test
    void versionPrintsAndExitsZero() {
        String[] r = runCaptured("--version");
        assertEquals("0", r[0]);
        assertEquals("agent-base " + Version.VERSION, r[1].strip());
    }

    @Test
    void helpPrintsUsageAndExitsZero() {
        String[] r = runCaptured("--help");
        assertEquals("0", r[0]);
        assertTrue(r[1].contains("usage: agent-base"));
        // -h 同义。
        assertEquals("0", runCaptured("-h")[0]);
    }

    @Test
    void unknownFlagExitsTwoWithMessage() {
        String[] r = runCaptured("--nope");
        assertEquals("2", r[0]);
        assertTrue(r[2].contains("未知参数 --nope"));
    }

    @Test
    void missingValueExitsTwoLikeArgparse() {
        // 修复前：requireValue 的 IAE 裸抛（未捕获 → 堆栈转储 + exit 1）。
        String[] missingMessage = runCaptured("--message");
        assertEquals("2", missingMessage[0]);
        assertTrue(missingMessage[2].contains("--message 需要一个值"));
        String[] missingModule = runCaptured("--module");
        assertEquals("2", missingModule[0]);
        assertTrue(missingModule[2].contains("--module 需要一个值"));
    }

    @Test
    void replyPicksTheLastTextualAiMessage() {
        assertEquals("", CliMain.reply(List.of()));
        // 只有用户消息（无回复）→ 空串（oneShot 据此跳过打印）。
        assertEquals("", CliMain.reply(List.of(UserMessage.from("hi"))));
        // 最后一条带文本的 AiMessage 胜出；中间的工具结果消息不参与。
        assertEquals("final", CliMain.reply(List.of(
                UserMessage.from("hi"),
                AiMessage.from("draft"),
                ToolExecutionResultMessage.from("tool", "id", "result"),
                AiMessage.from("final"))));
        // 尾部 AiMessage 无文本（纯工具调用）→ 回退到前一条有文本的。
        assertEquals("with text", CliMain.reply(List.of(
                AiMessage.from("with text"),
                AiMessage.from(""))));
    }
}
