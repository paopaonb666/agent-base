package io.github.paopaonb666.agentbase.toolkit.parsing;

import java.io.ByteArrayInputStream;
import java.util.List;

import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;

/**
 * DOCX 解析器（POI XWPF，对应 Python 侧 python-docx 的
 * {@code parsing/docx.py}）。
 *
 * <p>Python 侧 python-docx 是 {@code [doc]} extras 的可选依赖，未安装时
 * 该格式不注册并给出安装指引；Java 侧 POI 是随基座编译的硬依赖，
 * {@link #isAvailable()} 恒为 true——降级路径在 Java 里不存在，文档化
 * 差异即契约差异。
 *
 * <p>只提取正文段落（与 chat-agent 的已验证行为一致）；表格、页眉页脚
 * 等留到有真实需求时再扩展。空段落不进入输出（只为 LLM 省 token），
 * 段落数元信息仍统计全部段落。
 */
public final class DocxParser {

    private DocxParser() {
    }

    /** POI 恒可用（硬依赖）；保留该谓词以对齐 Python 注册表的判断面。 */
    public static boolean isAvailable() {
        return true;
    }

    /** 解析 DOCX 字节流，返回正文文本与总段落数。 */
    public static ParseResult parse(byte[] data) {
        List<XWPFParagraph> paragraphs;
        try (XWPFDocument document = new XWPFDocument(new ByteArrayInputStream(data))) {
            paragraphs = document.getParagraphs();
        } catch (Exception e) {
            throw new DocumentParseError(
                    "DOCX 解析失败：文件可能已损坏或不是有效的 .docx（"
                            + e.getClass().getSimpleName() + ": " + e.getMessage() + "）");
        }
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (XWPFParagraph p : paragraphs) {
            String text = p.getText();
            if (text != null && !text.isEmpty()) {
                if (!first) {
                    sb.append("\n");
                }
                first = false;
                sb.append(text);
            }
        }
        return new ParseResult(sb.toString(), "docx", null, paragraphs.size(), false);
    }
}
