package io.github.paopaonb666.agentbase.toolkit.parsing;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 文档解析层：多格式 → 统一的 {@link ParseResult}（对应 Python 侧
 * {@code parsing/__init__.py}）。支持 PDF（PDFBox）、DOCX（POI XWPF）、
 * TXT / Markdown（直接解码）。纯函数式：bytes 进、record 出，不做 IO、
 * 不碰网络与 LLM，便于单测与上层复用。
 *
 * <p>本层刻意不做工具封装：{@code read_document} 等工具属于 M4 的工具
 * 批次，在这层之上包装路径安全，超时则由共享池统一添加。限额（输入
 * 大小、输出长度）在本层强制执行，工具层无需重复关心。
 */
public final class DocumentParsers {

    /** 输入大小封顶默认值（字节）：解析在内存中进行，上限挡住超大文件
     * 把工具线程拖入 OOM 的路径。与 Settings.DocParseSettings 默认值一致。 */
    public static final int DEFAULT_MAX_INPUT_BYTES = 10 * 1024 * 1024;

    /** 输出文本封顶默认值（字符）：防止整本 PDF 的文本一次性灌进模型上下文。 */
    public static final int DEFAULT_MAX_OUTPUT_CHARS = 50_000;

    /** 扩展名 → 归一化格式名；大小写在查询前归一（对应 _SUFFIX_FORMATS）。 */
    private static final Map<String, String> SUFFIX_FORMATS = Map.of(
            "pdf", "pdf",
            "docx", "docx",
            "txt", "txt",
            "md", "markdown",
            "markdown", "markdown");

    private DocumentParsers() {
    }

    /** 当前环境可用的格式清单；Java 侧 POI 是硬依赖，docx 恒在。 */
    public static List<String> availableFormats() {
        List<String> formats = new ArrayList<>(List.of("docx", "markdown", "pdf", "txt"));
        formats.sort(String::compareTo);
        return formats;
    }

    /** 错误信息里展示的格式清单（Python list repr 形态：['docx', ...]）。 */
    private static String formatsRepr() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < availableFormats().size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append("'").append(availableFormats().get(i)).append("'");
        }
        return sb.append("]").toString();
    }

    /**
     * 把文档字节流解析为统一的 {@link ParseResult}。
     *
     * @param filename        用于按扩展名推断格式（{@code report.pdf} / {@code notes.md}）
     * @param format          显式格式名，优先于扩展名推断
     * @param pages           页码选择（如 {@code "1-3,5"}）；仅 PDF 支持，其他格式忽略
     * @param maxInputBytes   输入超限直接报错
     * @param maxOutputChars  输出超限截断并置 {@code truncated} 标志
     */
    public static ParseResult parseDocument(byte[] data, String filename, String format,
                                            String pages, int maxInputBytes, int maxOutputChars) {
        if (maxInputBytes <= 0 || maxOutputChars <= 0) {
            throw new DocumentParseError(
                    "解析限额必须为正数：max_input_bytes=" + maxInputBytes
                            + ", max_output_chars=" + maxOutputChars);
        }
        if (data.length > maxInputBytes) {
            throw new DocumentParseError(
                    "文档超出大小上限：" + data.length + " 字节 > 上限 " + maxInputBytes
                            + " 字节；请拆分或压缩后重试");
        }

        String fmt = format != null ? format : inferFormat(filename);
        ParseResult parsed;
        switch (fmt) {
            case "pdf" -> parsed = PdfParser.parse(data, pages);
            case "docx" -> parsed = DocxParser.parse(data);
            case "txt", "markdown" -> parsed = TextParser.parse(data, fmt);
            default -> throw new DocumentParseError(
                    "不支持的文档格式 '" + fmt + "'；当前可用：" + formatsRepr());
        }

        if (parsed.text().length() > maxOutputChars) {
            parsed = parsed.withText(parsed.text().substring(0, maxOutputChars));
        }
        return parsed;
    }

    /** 按扩展名推断格式；无法推断时给出"怎么才能推断"的可读错误。 */
    static String inferFormat(String filename) {
        if (filename == null || filename.isEmpty()) {
            throw new DocumentParseError(
                    "无法确定文档格式：请提供带扩展名的 filename，或显式指定 format");
        }
        // PurePath().suffix：最后一个点之后的部分（文件名以点开头时无后缀）。
        String name = filename;
        int slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
        if (slash >= 0) {
            name = name.substring(slash + 1);
        }
        int dot = name.lastIndexOf('.');
        String suffix = dot > 0 ? name.substring(dot + 1).toLowerCase() : "";
        String fmt = SUFFIX_FORMATS.get(suffix);
        if (fmt == null) {
            throw new DocumentParseError(
                    "无法识别的文档扩展名 '" + suffix + "'（filename='" + filename
                            + "'）；当前可用：" + formatsRepr());
        }
        return fmt;
    }
}
