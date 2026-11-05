package io.github.paopaonb666.agentbase.toolkit.parsing;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.encryption.InvalidPasswordException;
import org.apache.pdfbox.text.PDFTextStripper;

/**
 * PDF 解析器（PDFBox，对应 Python 侧 pypdf 的 {@code parsing/pdf.py}）。
 *
 * <p>逐页提取文本后按页拼接（沿用 chat-agent 已验证的做法）；{@code pages}
 * 参数支持 {@code "1-3,5"} 这类页码选择，让模型可以分批读取大文档而不是
 * 一次性吞下全文。
 *
 * <p>扫描版 PDF（纯图片页）的提取结果是空文本——这不是错误：返回空文本
 * 并保留页数元信息，由上层工具决定如何向模型解释（例如提示"可能是
 * 扫描件"）。加密 PDF（带密码）抛出专门的文案（对应 Python 的
 * {@code is_encrypted} 检查；PDFBox 对带用户密码的文件在 load 期抛
 * {@link InvalidPasswordException}，同样归入该文案）。
 */
public final class PdfParser {

    private PdfParser() {
    }

    /** 解析 PDF 字节流；{@code pages} 为可选页码选择（如 {@code "1-3,5"}，1-based）。 */
    public static ParseResult parse(byte[] data, String pages) {
        List<String> texts = new ArrayList<>();
        int total;
        try (PDDocument document = PDDocument.load(data)) {
            if (document.isEncrypted()) {
                throw new DocumentParseError("PDF 已加密，无法解析：请先提供解密后的文件");
            }
            total = document.getNumberOfPages();
            List<Integer> pageNumbers = new ArrayList<>();
            if (pages != null) {
                pageNumbers.addAll(PageSelection.parse(pages, total));
            } else {
                for (int p = 1; p <= total; p++) {
                    pageNumbers.add(p);
                }
            }
            PDFTextStripper stripper = new PDFTextStripper();
            stripper.setSortByPosition(true);
            for (int pageNumber : pageNumbers) {
                stripper.setStartPage(pageNumber);
                stripper.setEndPage(pageNumber);
                texts.add(stripper.getText(document));
            }
        } catch (DocumentParseError e) {
            throw e;
        } catch (InvalidPasswordException e) {
            throw new DocumentParseError("PDF 已加密，无法解析：请先提供解密后的文件");
        } catch (IOException | RuntimeException e) {
            throw new DocumentParseError(
                    "PDF 解析失败：文件可能已损坏或不是有效的 PDF（"
                            + e.getClass().getSimpleName() + ": " + e.getMessage() + "）");
        }
        return new ParseResult(String.join("\n", texts), "pdf", total, null, false);
    }
}
