package io.github.paopaonb666.agentbase.extensions.filestore;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 一条上传附件：原始字节、解析文本与元信息（对应 Python 侧
 * {@code extensions/filestore.py} 的 {@code UploadedFileInfo} dataclass）。
 *
 * <p>字段与组件顺序即 {@code uploaded_files} 表的列序（alembic 0003 +
 * 0006 合并形态），便于行映射逐列对位。record 组件不可变——绑定线程
 * 用 {@link #withThreadId(String)} 生成替换副本（对应
 * {@code dataclasses.replace(info, thread_id=…)}）。
 *
 * <p>{@code createdAt} 是 epoch 秒（浮点，对齐 Python {@code time.time()}
 * 的默认值语义）；sqlite 后端落库时转为本地时间 "YYYY-MM-DD HH:MM:SS"
 * 字符串，与 Python 侧写法一致。
 *
 * @param fileId        文件主键（上传时生成的随机 hex）
 * @param threadId      绑定的对话线程；空串 = 尚未被消息引用（孤儿候选）
 * @param module        上传时所属模块（{@code /agents/{module}/files}）
 * @param userId        上传者的记忆身份（X-User-Id）；空串 = 无主遗留，
 *                      允许任意已验身份用户撤销（alembic 0006 语义）
 * @param filename      原始文件名
 * @param format        解析格式（pdf / docx / txt…）
 * @param pages         文档页数（未知为 null）
 * @param paragraphs    文档段落数（未知为 null）
 * @param truncated     提取文本是否被截断
 * @param textLen       提取文本长度（字符）
 * @param warning       提取警告（如扫描件无文本）；空串 = 无警告
 * @param extractedText 解析提取文本（注入对话上下文的内容）
 * @param content       原始字节（入库 BLOB/LONGBLOB，可重解析、可扩展下载）
 * @param createdAt     创建时间（epoch 秒）
 */
public record UploadedFileInfo(
        String fileId,
        String threadId,
        String module,
        String userId,
        String filename,
        String format,
        Integer pages,
        Integer paragraphs,
        boolean truncated,
        int textLen,
        String warning,
        String extractedText,
        byte[] content,
        double createdAt) {

    /** 归一化 null：Python dataclass 的默认值语义（缺省空串/空字节/当前时间）。 */
    public UploadedFileInfo {
        fileId = fileId == null ? "" : fileId;
        threadId = threadId == null ? "" : threadId;
        module = module == null ? "" : module;
        userId = userId == null ? "" : userId;
        filename = filename == null ? "" : filename;
        format = format == null ? "" : format;
        warning = warning == null ? "" : warning;
        extractedText = extractedText == null ? "" : extractedText;
        content = content == null ? new byte[0] : content;
    }

    /** 便利构造器：{@code createdAt} 缺省为当前时间（对应 dataclass 默认 {@code time.time()}）。 */
    public UploadedFileInfo(String fileId, String threadId, String module, String userId,
                            String filename, String format, Integer pages, Integer paragraphs,
                            boolean truncated, int textLen, String warning, String extractedText,
                            byte[] content) {
        this(fileId, threadId, module, userId, filename, format, pages, paragraphs,
                truncated, textLen, warning, extractedText, content,
                System.currentTimeMillis() / 1000.0);
    }

    /**
     * 对外元信息视图（HTTP 端点的 JSON 载荷；不含提取文本与原始字节）。
     * 键集与顺序对齐 Python {@code meta()}：file_id / filename / format /
     * pages / text_len / truncated / user_id，仅在有警告时附加 warning。
     */
    public Map<String, Object> meta() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("file_id", fileId);
        out.put("filename", filename);
        out.put("format", format);
        out.put("pages", pages);
        out.put("text_len", textLen);
        out.put("truncated", truncated);
        out.put("user_id", userId);
        if (warning != null && !warning.isEmpty()) {
            out.put("warning", warning);
        }
        return out;
    }

    /** 替换副本：仅回填绑定线程（对应 {@code dataclasses.replace(info, thread_id=…)}）。 */
    public UploadedFileInfo withThreadId(String newThreadId) {
        return new UploadedFileInfo(fileId, newThreadId, module, userId, filename, format,
                pages, paragraphs, truncated, textLen, warning, extractedText, content, createdAt);
    }

    /** 值语义相等：{@code content} 按字节比较（record 默认的数组引用比较不够）。 */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof UploadedFileInfo other)) {
            return false;
        }
        return truncated == other.truncated
                && textLen == other.textLen
                && Double.compare(createdAt, other.createdAt) == 0
                && Objects.equals(fileId, other.fileId)
                && Objects.equals(threadId, other.threadId)
                && Objects.equals(module, other.module)
                && Objects.equals(userId, other.userId)
                && Objects.equals(filename, other.filename)
                && Objects.equals(format, other.format)
                && Objects.equals(pages, other.pages)
                && Objects.equals(paragraphs, other.paragraphs)
                && Objects.equals(warning, other.warning)
                && Objects.equals(extractedText, other.extractedText)
                && Arrays.equals(content, other.content);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(fileId, threadId, module, userId, filename, format,
                pages, paragraphs, truncated, textLen, warning, extractedText, createdAt);
        return 31 * result + Arrays.hashCode(content);
    }
}
