package io.github.paopaonb666.agentbase.toolkit.parsing;

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.charset.Charset;

/**
 * TXT / Markdown 解析器：直接解码（UTF-8 优先，容错 GBK 系；对应 Python
 * 侧 {@code parsing/text.py}）。
 *
 * <p>两种格式共用同一实现——解析层只负责忠实地取出文本；按标题切分等
 * "理解"逻辑留给上层工具与未来的 RAG 模块。编码策略：UTF-8（带 BOM
 * 剥离）严格解码优先，失败时退回 GB18030（GBK/GB2312 的官方超集）；
 * 两者都失败说明输入不是文本或是不认识的编码，按契约抛出可读错误，
 * 绝不静默替换字符。
 */
public final class TextParser {

    private TextParser() {
    }

    /** 把字节流解码为文本：UTF-8 严格解码优先（剥 BOM），失败退回 GB18030。 */
    public static String decodeText(byte[] data) {
        // utf-8-sig：仅在字节序列以 BOM 开头时剥离。
        byte[] body = data;
        if (body.length >= 3 && (body[0] & 0xFF) == 0xEF && (body[1] & 0xFF) == 0xBB
                && (body[2] & 0xFF) == 0xBF) {
            byte[] stripped = new byte[body.length - 3];
            System.arraycopy(body, 3, stripped, 0, stripped.length);
            body = stripped;
        }
        try {
            return strictDecode(body, StandardCharsets.UTF_8);
        } catch (CharacterCodingException ignored) {
            // 落到 GB18030 回退。
        }
        try {
            return strictDecode(body, Charset.forName("GB18030"));
        } catch (CharacterCodingException e) {
            throw new DocumentParseError(
                    "文本解码失败：输入既不是有效的 UTF-8，也不是 GBK/GB18030 编码");
        }
    }

    private static String strictDecode(byte[] data, Charset charset) throws CharacterCodingException {
        CharsetDecoder decoder = charset.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        return decoder.decode(ByteBuffer.wrap(data)).toString();
    }

    /** 解析 TXT / Markdown 字节流；{@code fmt} 是归一化后的格式名。 */
    public static ParseResult parse(byte[] data, String fmt) {
        String text = decodeText(data);
        // paragraphs = 非空行数（对应 Python 的 splitlines + strip 过滤）。
        int paragraphs = 0;
        for (String line : text.split("\\R", -1)) {
            if (!line.strip().isEmpty()) {
                paragraphs++;
            }
        }
        return new ParseResult(text, fmt, null, paragraphs, false);
    }
}
