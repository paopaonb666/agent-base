package io.github.paopaonb666.agentbase.toolkit.parsing;

/**
 * 解析失败（格式不支持、文件损坏、超出限额、依赖缺失等）；对应 Python
 * 侧 {@code parsing/base.py} 的 {@code DocumentParseError(ValueError)}。
 *
 * <p>失败哲学（对 chat-agent 参考实现的显式反转，继承自 Python 侧注释）：
 * chat-agent 的 {@code _extract_text} 在解析失败时静默返回空串，上层把
 * "解析不了"当成了"文档是空的"。本层的契约是坏输入必须抛出带可读信息
 * 的异常——错误文本会经共享池的 ToolMessage 回给模型，让它能理解失败
 * 原因并自行修正重试。
 */
public class DocumentParseError extends IllegalArgumentException {

    public DocumentParseError(String message) {
        super(message);
    }
}
