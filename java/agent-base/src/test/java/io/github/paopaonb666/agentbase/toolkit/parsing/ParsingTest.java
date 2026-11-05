package io.github.paopaonb666.agentbase.toolkit.parsing;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.encryption.AccessPermission;
import org.apache.pdfbox.pdmodel.encryption.StandardProtectionPolicy;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.junit.jupiter.api.Test;

/**
 * 文档解析层的测试（对应 Python 侧 tests/test_parsing.py 的验收项）：
 * 各格式正常解析、{@code pages} 页码选择、输入/输出限额、损坏文件抛
 * 可读异常、加密 PDF 的专门文案。PDF/DOCX 夹具在测试内确定性构造，
 * 不依赖二进制 fixture。
 */
class ParsingTest {

    private static IllegalArgumentException expectError(Runnable action, String contains) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, action::run);
        assertTrue(e.getMessage().contains(contains),
                "错误信息应含 '" + contains + "'，实际：" + e.getMessage());
        return e;
    }

    // ── 夹具构造 ─────────────────────────────────────────────────────

    /** 用 PDFBox 构造多页 PDF：每页一行 Helvetica 文本。 */
    private static byte[] makePdf(List<String> pageTexts) throws IOException {
        try (PDDocument document = new PDDocument()) {
            for (String text : pageTexts) {
                PDPage page = new PDPage(PDRectangle.LETTER);
                document.addPage(page);
                try (PDPageContentStream cs = new PDPageContentStream(document, page)) {
                    cs.beginText();
                    cs.setFont(PDType1Font.HELVETICA, 12);
                    cs.newLineAtOffset(72, 720);
                    cs.showText(text);
                    cs.endText();
                }
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            document.save(out);
            return out.toByteArray();
        }
    }

    /** 构造带密码的 PDF（PDFBox 自身的加密能力）。 */
    private static byte[] makeEncryptedPdf(List<String> pageTexts) throws IOException {
        try (PDDocument document = PDDocument.load(makePdf(pageTexts))) {
            StandardProtectionPolicy policy =
                    new StandardProtectionPolicy("owner-secret", "secret", new AccessPermission());
            policy.setEncryptionKeyLength(128);
            document.protect(policy);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            document.save(out);
            return out.toByteArray();
        }
    }

    /** 构造内存中的 .docx（POI 自身的写入能力）。 */
    private static byte[] makeDocx(List<String> paragraphs) throws IOException {
        try (org.apache.poi.xwpf.usermodel.XWPFDocument document =
                     new org.apache.poi.xwpf.usermodel.XWPFDocument()) {
            for (String text : paragraphs) {
                document.createParagraph().createRun().setText(text);
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            document.write(out);
            return out.toByteArray();
        }
    }

    private static ParseResult parse(byte[] data, String filename) {
        return DocumentParsers.parseDocument(data, filename, null, null,
                DocumentParsers.DEFAULT_MAX_INPUT_BYTES, DocumentParsers.DEFAULT_MAX_OUTPUT_CHARS);
    }

    // ── PDF ──────────────────────────────────────────────────────────

    @Test
    void pdfParsesAllPagesByDefault() throws IOException {
        ParseResult result = parse(makePdf(List.of("Hello one", "Hello two")), "doc.pdf");
        assertEquals("pdf", result.format());
        assertEquals(2, result.pages());
        assertFalse(result.truncated());
        assertNull(result.paragraphs());
        // PDFBox 的逐页提取按页拼接；行级断言（容忍提取器行尾空白差异，
        // 对齐 pypdf 逐页 extract_text 的拼接语义）。
        List<String> lines = result.text().lines().map(String::strip)
                .filter(s -> !s.isEmpty()).toList();
        assertEquals(List.of("Hello one", "Hello two"), lines);
    }

    @Test
    void pdfPagesSinglePageSelection() throws IOException {
        ParseResult result = DocumentParsers.parseDocument(
                makePdf(List.of("one", "two", "three")), "doc.pdf", null, "2",
                DocumentParsers.DEFAULT_MAX_INPUT_BYTES, DocumentParsers.DEFAULT_MAX_OUTPUT_CHARS);
        assertTrue(result.text().contains("two"));
        assertFalse(result.text().contains("one "));
        // pages 元信息始终是文档总页数，供上层提示"还有更多页"。
        assertEquals(3, result.pages());
    }

    @Test
    void pdfPagesRangeAndMixed() throws IOException {
        ParseResult result = DocumentParsers.parseDocument(
                makePdf(List.of("p1", "p2", "p3", "p4", "p5")), "d.pdf", null, "1-3,5",
                DocumentParsers.DEFAULT_MAX_INPUT_BYTES, DocumentParsers.DEFAULT_MAX_OUTPUT_CHARS);
        String text = result.text();
        assertTrue(text.contains("p1") && text.contains("p2") && text.contains("p3")
                && text.contains("p5"));
        assertFalse(text.contains("p4"));
    }

    @Test
    void pdfPagesDedupAndSorted() throws IOException {
        // 重复与乱序的页码选择按阅读顺序去重输出，不会重复提取。
        ParseResult result = DocumentParsers.parseDocument(
                makePdf(List.of("p1", "p2", "p3")), "d.pdf", null, "3,1,1-2",
                DocumentParsers.DEFAULT_MAX_INPUT_BYTES, DocumentParsers.DEFAULT_MAX_OUTPUT_CHARS);
        assertEquals(3, result.text().split("p", -1).length - 1); // 恰好 3 个页标记
    }

    @Test
    void encryptedPdfGetsDedicatedMessage() throws IOException {
        byte[] encrypted = makeEncryptedPdf(List.of("secret content"));
        expectError(() -> parse(encrypted, "doc.pdf"), "PDF 已加密");
    }

    @Test
    void corruptedPdfThrowsReadableError() {
        // 非法输入必须抛出带可读信息的 DocumentParseError，绝不静默返回空串。
        expectError(() -> parse("this is not a pdf".getBytes(), "doc.pdf"), "PDF 解析失败");
        expectError(() -> parse(new byte[]{0x25, 0x50, 0x44, 0x46, 0x2D}, "doc.pdf"), "PDF 解析失败");
    }

    // ── DOCX ─────────────────────────────────────────────────────────

    @Test
    void docxExtractsParagraphsAndCountsAll() throws IOException {
        ParseResult result = parse(makeDocx(List.of("第一段", "", "第二段")), "report.docx");
        assertEquals("docx", result.format());
        assertEquals("第一段\n第二段", result.text());
        // 段落数统计全部段落（含空段），正文只拼接非空段。
        assertEquals(3, result.paragraphs());
        assertNull(result.pages());
    }

    @Test
    void corruptedDocxThrowsReadableError() {
        expectError(() -> parse("not a docx".getBytes(), "report.docx"), "DOCX 解析失败");
    }

    // ── TXT / Markdown ───────────────────────────────────────────────

    @Test
    void textAndMarkdownDecodeAndCountNonBlankLines() {
        String content = "标题行\n\n正文一行\n  \n最后一行\n";
        ParseResult txt = parse(content.getBytes(java.nio.charset.StandardCharsets.UTF_8), "notes.txt");
        assertEquals("txt", txt.format());
        assertEquals(3, txt.paragraphs()); // 非空行数

        ParseResult md = parse(content.getBytes(java.nio.charset.StandardCharsets.UTF_8), "notes.md");
        assertEquals("markdown", md.format());
    }

    @Test
    void textDecodesUtf8WithBom() {
        byte[] bom = new byte[]{(byte) 0xEF, (byte) 0xBB, (byte) 0xBF};
        byte[] body = "正文".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        byte[] data = new byte[bom.length + body.length];
        System.arraycopy(bom, 0, data, 0, bom.length);
        System.arraycopy(body, 0, data, bom.length, body.length);
        ParseResult result = parse(data, "notes.txt");
        assertEquals("正文", result.text());
    }

    @Test
    void textFallsBackToGb18030() {
        byte[] data = "中文内容".getBytes(Charset.forName("GB18030"));
        ParseResult result = parse(data, "legacy.txt");
        assertEquals("中文内容", result.text());
    }

    @Test
    void undecodableBytesThrowReadableError() {
        // 单个 0x80 既不是合法 UTF-8 也不是合法 GB18030。
        expectError(() -> parse(new byte[]{(byte) 0x80}, "junk.txt"), "文本解码失败");
    }

    // ── 限额与格式分派 ───────────────────────────────────────────────

    @Test
    void inputOverLimitThrows() {
        expectError(() -> DocumentParsers.parseDocument(new byte[100], "doc.pdf", null, null,
                50, DocumentParsers.DEFAULT_MAX_OUTPUT_CHARS), "文档超出大小上限");
    }

    @Test
    void outputTruncationSetsFlag() {
        ParseResult result = DocumentParsers.parseDocument(
                "一".repeat(100).getBytes(java.nio.charset.StandardCharsets.UTF_8), "notes.txt",
                null, null, DocumentParsers.DEFAULT_MAX_INPUT_BYTES, 10);
        assertTrue(result.truncated());
        assertEquals(10, result.text().length());
    }

    @Test
    void nonPositiveLimitsThrow() {
        expectError(() -> DocumentParsers.parseDocument(new byte[0], "doc.pdf", null, null,
                0, 10), "解析限额必须为正数");
        expectError(() -> DocumentParsers.parseDocument(new byte[0], "doc.pdf", null, null,
                10, -1), "解析限额必须为正数");
    }

    @Test
    void formatInferenceAndExplicitOverride() {
        // 显式 format 优先于扩展名推断；无法推断时给"怎么才能推断"的指引。
        ParseResult byFormat = DocumentParsers.parseDocument(
                "plain".getBytes(java.nio.charset.StandardCharsets.UTF_8), "weird.unknown",
                "txt", null, DocumentParsers.DEFAULT_MAX_INPUT_BYTES,
                DocumentParsers.DEFAULT_MAX_OUTPUT_CHARS);
        assertEquals("txt", byFormat.format());

        expectError(() -> parse("x".getBytes(), "file.bin"), "无法识别的文档扩展名");
        expectError(() -> DocumentParsers.parseDocument("x".getBytes(), null, null, null,
                DocumentParsers.DEFAULT_MAX_INPUT_BYTES, DocumentParsers.DEFAULT_MAX_OUTPUT_CHARS),
                "无法确定文档格式");
        expectError(() -> DocumentParsers.parseDocument("x".getBytes(), "a.rtf", "rtf", null,
                DocumentParsers.DEFAULT_MAX_INPUT_BYTES, DocumentParsers.DEFAULT_MAX_OUTPUT_CHARS),
                "不支持的文档格式");
        assertEquals(List.of("docx", "markdown", "pdf", "txt"), DocumentParsers.availableFormats());
    }

    @Test
    void pageSelectionErrorsAreReadable() {
        expectError(() -> PageSelection.parse("1,,3", 5), "存在空的页码项");
        expectError(() -> PageSelection.parse("3-1", 5), "区间终点");
        expectError(() -> PageSelection.parse("0", 5), "页码从 1 开始");
        expectError(() -> PageSelection.parse("1-6", 5), "页码超出范围");
        expectError(() -> PageSelection.parse("1-3-5", 5), "不是页码或页码区间");
        expectError(() -> PageSelection.parse("x", 5), "不是整数页码");
        assertEquals(List.of(1, 2, 3, 5), PageSelection.parse("1-3,5", 5));
    }
}
