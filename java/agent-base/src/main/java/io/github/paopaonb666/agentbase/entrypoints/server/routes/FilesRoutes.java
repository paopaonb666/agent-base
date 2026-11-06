package io.github.paopaonb666.agentbase.entrypoints.server.routes;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import io.github.paopaonb666.agentbase.core.AgentRuntime;
import io.github.paopaonb666.agentbase.core.config.Settings;
import io.github.paopaonb666.agentbase.entrypoints.server.AuthBackends;
import io.github.paopaonb666.agentbase.entrypoints.server.Background;
import io.github.paopaonb666.agentbase.entrypoints.server.HttpError;
import io.github.paopaonb666.agentbase.entrypoints.server.MessageSerializers;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileInfo;
import io.github.paopaonb666.agentbase.extensions.filestore.UploadedFileStore;
import io.github.paopaonb666.agentbase.toolkit.parsing.DocumentParsers;
import io.github.paopaonb666.agentbase.toolkit.parsing.ParseResult;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * 附件与知识库文件端点：上传 / 撤销 / 预览 / 切片 / 原始字节。
 *
 * <p>属主模型（S1）：上传时记录 user_id；读取/撤销端点做属主比对（404
 * 掩蔽）。user_id == "" 视为无主遗留（旧行），允许撤销——file_id 本身
 * 12 位随机不可猜。
 *
 * <p>对应 Python 侧 {@code routes/files.py}，逐端点对齐。
 */
@RestController
@RequestMapping("/v1")
public class FilesRoutes {

    /** 上传读取的分块大小：流式读取 + 边读边判大小。 */
    private static final int UPLOAD_CHUNK_BYTES = 1 << 20;

    private final AgentRuntime runtime;
    private final AuthBackends.AuthBackend auth;

    public FilesRoutes(AgentRuntime runtime, AuthBackends.AuthBackend auth) {
        this.runtime = runtime;
        this.auth = auth;
    }

    private String user(String userIdHeader, String sigHeader, String tsHeader, String nonceHeader) {
        return auth.authenticate(userIdHeader, sigHeader, tsHeader, nonceHeader, runtime.settings());
    }

    private boolean knownModule(String module) {
        return runtime.modules().containsKey(module) || AgentRuntime.SUPERVISOR_MODULE.equals(module);
    }

    private void requireModule(String module) {
        if (!knownModule(module)) {
            throw new HttpError(404, "unknown module '" + module + "'");
        }
    }

    /** 取附件并校验属主；404 掩蔽（filestore 缺席由调用方先判）。 */
    private UploadedFileInfo ownedFile(String fileId, String userId) {
        UploadedFileStore fileStore = runtime.fileStore();
        if (fileStore == null) {
            throw new HttpError(503, "附件存储未启用（存储后端不可用）");
        }
        List<UploadedFileInfo> infos = fileStore.getMany(List.of(fileId));
        UploadedFileInfo info = infos.isEmpty() ? null : infos.get(0);
        if (info == null || !(info.userId().isEmpty() || info.userId().equals(userId))) {
            // 404 而非 403：不向第三方泄露文件 id 的存在性。
            throw new HttpError(404, "文件不存在：" + fileId);
        }
        return info;
    }

    // ── POST /v1/agents/{module}/files：上传附件 ─────────────────────────

    @PostMapping("/agents/{module}/files")
    public Map<String, Object> uploadFile(@PathVariable("module") String module,
                                          @RequestParam("file") MultipartFile file,
                                          @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                          @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                          @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                          @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) throws IOException {
        // 鉴权先行（对应 Python get_current_user 依赖先于一切处理逻辑）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        requireModule(module);
        UploadedFileStore fileStore = runtime.fileStore();
        if (fileStore == null) {
            throw new HttpError(503, "附件存储未启用（存储后端不可用）");
        }
        // 大小上限（M5/#144）：声明 size（getSize）只是早退优化——容器
        // 解析前不可信，真正的大小约束在流式读取里边读边判（分块
        // UPLOAD_CHUNK_BYTES）：超限立即 413，堆内存上限钉在 maxBytes
        // 而不是 getBytes() 的整份复制。文案与 Python 413 一致。
        Settings settings = runtime.settings();
        int maxBytes = settings.docParse.maxInputBytes;
        if (file.getSize() > maxBytes) {
            throw new HttpError(413, "文件超出大小上限：> " + maxBytes + " 字节");
        }
        byte[] data;
        try (java.io.InputStream in = file.getInputStream()) {
            java.io.ByteArrayOutputStream buffer =
                    new java.io.ByteArrayOutputStream(Math.min(maxBytes, UPLOAD_CHUNK_BYTES));
            byte[] chunk = new byte[UPLOAD_CHUNK_BYTES];
            long total = 0;
            int read;
            while ((read = in.read(chunk)) != -1) {
                total += read;
                if (total > maxBytes) {
                    throw new HttpError(413, "文件超出大小上限：> " + maxBytes + " 字节");
                }
                buffer.write(chunk, 0, read);
            }
            data = buffer.toByteArray();
        }
        if (data.length == 0) {
            throw new HttpError(400, "上传文件为空");
        }
        String filename = file.getOriginalFilename() == null ? "" : file.getOriginalFilename();
        String suffix = filename.contains(".")
                ? filename.substring(filename.lastIndexOf('.') + 1).toLowerCase(java.util.Locale.ROOT)
                : "";
        if (MessageSerializers.IMAGE_EXTENSIONS.contains(suffix)) {
            // 图片：不走文本解析，按 magic bytes 校验后整字节入库。
            String mime = MessageSerializers.sniffImageMime(data);
            if (mime == null) {
                throw new HttpError(400, "文件内容不是有效的图片：扩展名 '" + suffix
                        + "' 与实际内容不符（magic 校验失败）");
            }
            UploadedFileInfo info = new UploadedFileInfo(UUID.randomUUID().toString()
                    .replace("-", "").substring(0, 12), "", module, userId, filename,
                    mime.split("/", 2)[1], null, null, false, 0, "", "", data);
            fileStore.save(info);
            Background.spawn(fileStore::purgeOrphans, "purge-orphans");
            return info.meta();
        }
        ParseResult doc;
        try {
            doc = DocumentParsers.parseDocument(data, filename, null, null,
                    settings.docParse.maxInputBytes, settings.docParse.maxOutputChars);
        } catch (io.github.paopaonb666.agentbase.toolkit.parsing.DocumentParseError exc) {
            throw new HttpError(400, String.valueOf(exc.getMessage()));
        }
        // 空提取文本（扫描件/纯图片 PDF）不是错误，但要显式告知前端。
        String warning = doc.text().strip().isEmpty()
                ? "未能从文件中提取到文本：可能是扫描件或纯图片 PDF" : "";
        UploadedFileInfo info = new UploadedFileInfo(UUID.randomUUID().toString()
                .replace("-", "").substring(0, 12), "", module, userId, filename, doc.format(),
                doc.pages(), doc.paragraphs(), doc.truncated(), doc.text().length(),
                warning, doc.text(), data);
        fileStore.save(info);
        var memory = runtime.memory();
        // 知识库摄取（M6e）：fire-and-forget，失败不影响上传。
        if (memory != null && !info.extractedText().strip().isEmpty()) {
            Background.spawn(() -> memory.ingestDocument(info.fileId(), userId, module,
                    info.extractedText(), ""), "memory-ingest");
        }
        // 机会式清理 24h 未绑定的孤儿附件（fire-and-forget）。
        Background.spawn(fileStore::purgeOrphans, "purge-orphans");
        return info.meta();
    }

    // ── DELETE /v1/agents/{module}/files/{file_id}：撤销 ─────────────────

    @DeleteMapping("/agents/{module}/files/{file_id}")
    public Map<String, Object> revokeFile(@PathVariable("module") String module,
                                          @PathVariable("file_id") String fileId,
                                          @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                          @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                          @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                          @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 作为 FastAPI 依赖在任何
        // 处理逻辑之前解析——未签名请求一律 401，不给存在性探测留口子）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        requireModule(module);
        ownedFile(fileId, userId); // 属主校验（不存在/非属主即 404）
        UploadedFileStore fileStore = runtime.fileStore();
        boolean deleted = fileStore.deleteFile(fileId);
        var memory = runtime.memory();
        int chunksRemoved = memory != null ? memory.revokeDocument(fileId, userId) : 0;
        Background.spawn(fileStore::purgeOrphans, "purge-orphans");
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("deleted", deleted);
        out.put("chunks_removed", chunksRemoved);
        return out;
    }

    // ── GET …/preview：文档预览（元信息 + 提取正文，与注入内容同源） ─────

    @GetMapping("/agents/{module}/files/{file_id}/preview")
    public Map<String, Object> previewFile(@PathVariable("module") String module,
                                           @PathVariable("file_id") String fileId,
                                           @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                           @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                           @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                           @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 作为 FastAPI 依赖在任何
        // 处理逻辑之前解析——未签名请求一律 401，不给存在性探测留口子）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        requireModule(module);
        UploadedFileInfo info = ownedFile(fileId, userId);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("file_id", info.fileId());
        out.put("filename", info.filename());
        out.put("format", info.format());
        out.put("pages", info.pages());
        out.put("text_len", info.textLen());
        out.put("truncated", info.truncated());
        if (info.warning() != null && !info.warning().isEmpty()) {
            out.put("warning", info.warning());
        }
        out.put("extracted_text", info.extractedText());
        return out;
    }

    // ── GET …/chunks：切片存储视图（M7 可视化） ──────────────────────────

    @GetMapping("/agents/{module}/files/{file_id}/chunks")
    public Map<String, Object> fileChunks(@PathVariable("module") String module,
                                          @PathVariable("file_id") String fileId,
                                          @RequestParam(value = "limit", defaultValue = "200") int limit,
                                          @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
                                          @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
                                          @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
                                          @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 作为 FastAPI 依赖在任何
        // 处理逻辑之前解析——未签名请求一律 401，不给存在性探测留口子）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        requireModule(module);
        ownedFile(fileId, userId);
        var memory = runtime.memory();
        List<Map<String, Object>> chunks = memory != null
                ? memory.listFileChunks(userId, fileId, Math.max(1, Math.min(limit, 2000)))
                : List.of();
        return Map.of("chunks", chunks);
    }

    // ── GET …/raw：原始字节（图片直出 / 文档 attachment 下载） ───────────

    @GetMapping("/agents/{module}/files/{file_id}/raw")
    public org.springframework.http.ResponseEntity<byte[]> rawFile(
            @PathVariable("module") String module,
            @PathVariable("file_id") String fileId,
            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 作为 FastAPI 依赖在任何
        // 处理逻辑之前解析——未签名请求一律 401，不给存在性探测留口子）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        requireModule(module);
        UploadedFileInfo info = ownedFile(fileId, userId);
        if (info.content() == null || info.content().length == 0) {
            throw new HttpError(404, "文件内容不可用");
        }
        org.springframework.http.HttpHeaders headers = new org.springframework.http.HttpHeaders();
        if (MessageSerializers.IMAGE_STORED_FORMATS.contains(info.format())) {
            headers.setContentType(org.springframework.http.MediaType.parseMediaType(
                    "image/" + info.format()));
        } else {
            headers.setContentType(org.springframework.http.MediaType.APPLICATION_OCTET_STREAM);
            // RFC 5987 编码（中文文件名）。
            String encoded = java.net.URLEncoder.encode(info.filename(),
                    java.nio.charset.StandardCharsets.UTF_8).replace("+", "%20");
            headers.set("Content-Disposition", "attachment; filename*=UTF-8''" + encoded);
        }
        return new org.springframework.http.ResponseEntity<>(info.content(), headers,
                org.springframework.http.HttpStatus.OK);
    }

    // ── GET /v1/knowledge/search：Tier 0 零 LLM 检索直出 ─────────────────

    @GetMapping("/knowledge/search")
    public Map<String, Object> searchKnowledge(
            @RequestParam(value = "q") String q,
            @RequestParam(value = "file_id", required = false) String fileId,
            @RequestParam(value = "limit", defaultValue = "5") int limit,
            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 依赖）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        var memory = runtime.memory();
        if (memory == null) {
            throw new HttpError(503, "记忆系统未启用（MEMORY_ENABLED=false）");
        }
        String query = q == null ? "" : q.strip();
        if (query.isEmpty()) {
            throw new HttpError(422, "q 不能为空");
        }
        List<Map<String, Object>> scored =
                new ArrayList<>(memory.searchKnowledge(userId, null, query,
                        Math.max(1, Math.min(limit, 20))));
        if (fileId != null) {
            scored.removeIf(item -> !fileId.equals(String.valueOf(item.get("file_id"))));
        }
        // 文件名补全（尽力而为）：分块已按 user_id 隔离，名字只做展示。
        Map<String, String> names = new LinkedHashMap<>();
        if (runtime.fileStore() != null && !scored.isEmpty()) {
            List<String> ids = scored.stream()
                    .map(item -> String.valueOf(item.get("file_id"))).distinct().sorted().toList();
            for (UploadedFileInfo info : runtime.fileStore().getMany(ids)) {
                names.put(info.fileId(), info.filename());
            }
        }
        List<Map<String, Object>> results = new ArrayList<>();
        for (Map<String, Object> item : scored) {
            Map<String, Object> view = new LinkedHashMap<>();
            view.put("score", item.get("score"));
            view.put("file_id", item.get("file_id"));
            view.put("file_name", names.getOrDefault(String.valueOf(item.get("file_id")), ""));
            view.put("chunk_id", item.get("chunk_id"));
            view.put("ordinal", item.get("ordinal"));
            view.put("text", item.get("text"));
            view.put("offsets", item.get("offsets"));
            view.put("has_embedding", item.get("has_embedding"));
            results.add(view);
        }
        return Map.of("results", results);
    }

    // ── GET /v1/knowledge/files：知识库文件列表（M8） ────────────────────

    @GetMapping("/knowledge/files")
    public Map<String, Object> knowledgeFiles(
            @RequestHeader(value = "X-User-Id", required = false) String userIdHeader,
            @RequestHeader(value = "X-User-Sig", required = false) String sigHeader,
            @RequestHeader(value = "X-User-Timestamp", required = false) String tsHeader,
            @RequestHeader(value = "X-User-Nonce", required = false) String nonceHeader) {
        // 鉴权先行（对应 Python get_current_user 依赖）。
        String userId = user(userIdHeader, sigHeader, tsHeader, nonceHeader);
        UploadedFileStore fileStore = runtime.fileStore();
        if (fileStore == null) {
            throw new HttpError(503, "附件存储未启用（存储后端不可用）");
        }
        List<UploadedFileInfo> files = fileStore.listForUser(userId);
        Map<String, Integer> counts = Map.of();
        var memory = runtime.memory();
        if (memory != null) {
            counts = memory.chunkCountsByFile(userId);
        }
        List<Map<String, Object>> out = new ArrayList<>();
        for (UploadedFileInfo info : files) {
            Map<String, Object> view = new LinkedHashMap<>(info.meta());
            view.put("chunks", counts.getOrDefault(info.fileId(), 0));
            view.put("created_at", info.createdAt());
            out.add(view);
        }
        return Map.of("files", out);
    }
}
