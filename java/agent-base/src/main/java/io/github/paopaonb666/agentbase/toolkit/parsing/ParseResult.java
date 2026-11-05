package io.github.paopaonb666.agentbase.toolkit.parsing;

/**
 * 统一解析结果（对应 Python 侧 {@code parsing/base.py} 的
 * {@code ParsedDocument}）：所有解析器的统一返回——纯数据、无 IO、无
 * 网络与 LLM 依赖；未来的 {@code read_document} 工具和 RAG 类模块都消费
 * 这一个结构。
 *
 * <ul>
 *   <li>{@code text} —— 提取出的纯文本（可能被截断，见 {@code truncated}）</li>
 *   <li>{@code format} —— 归一化的格式名（pdf / docx / txt / markdown）</li>
 *   <li>{@code pages} —— 文档总页数（仅 PDF 设置；供上层提示"还有更多页"）</li>
 *   <li>{@code paragraphs} —— 段落数（docx 为正文段落数；文本格式为非空行数）</li>
 *   <li>{@code truncated} —— 输出超过 {@code maxOutputChars} 被截断时为 true</li>
 * </ul>
 */
public record ParseResult(String text, String format, Integer pages, Integer paragraphs,
                          boolean truncated) {

    /** 对应 Python 的 {@code dataclasses.replace(parsed, text=..., truncated=True)}。 */
    public ParseResult withText(String newText) {
        return new ParseResult(newText, format, pages, paragraphs, true);
    }
}
